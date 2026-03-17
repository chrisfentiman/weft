//! Semantic routing with per-domain hook lifecycle.
//!
//! Provides `route_with_hooks`, which builds routing domains, fires PreRoute
//! and PostRoute hooks for each domain, calls `route_domains` once, and
//! returns the routing result with accumulated hook context.

use tracing::{debug, info, warn};
use weft_commands::CommandRegistry;
use weft_core::{CommandStub, HookRoutingDomain, RoutingTrigger, WeftError};
use weft_core::HookEvent;
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::{Capability, ProviderService};
use weft_memory::MemoryService;
use weft_router::{
    RoutingCandidate, RoutingDomainKind, RoutingInput, RoutingResult, SemanticRouter,
    build_model_candidates, route_domains, tool_necessity_candidates,
};

use super::GatewayEngine;
use super::util::routing_domain_to_hook_domain;

impl<H, R, M, P, C> GatewayEngine<H, R, M, P, C>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    /// Route all domains with per-domain PreRoute/PostRoute hooks.
    ///
    /// Returns `(RoutingResult, Option<String>)` where the `Option<String>` is the
    /// accumulated hook context injected into the system prompt. `RoutingResult` carries
    /// the selected commands, inject_tools flag, selected model, and activity events.
    ///
    /// Per-domain hooks fire for each routing domain independently:
    /// - PreRoute can modify candidates or routing_input for that domain.
    /// - PostRoute can override selected candidates or scores for that domain.
    ///
    /// Hard blocks (403) on model domain terminate the request.
    /// Block on commands domain = empty command set (proceed with no tools).
    /// Block on tool_necessity domain = conservative (inject tools).
    ///
    /// On router failure, `route_domains` falls back to: all commands (capped by
    /// max_commands), `inject_tools = true` (conservative), and the default model.
    pub(super) async fn route_with_hooks(
        &self,
        user_message: &str,
        all_commands: &[CommandStub],
    ) -> Result<(RoutingResult, Option<String>), WeftError> {
        let threshold = self.config.router.classifier.threshold;
        let max_commands = self.config.router.classifier.max_commands;
        let skip_tools_when_unnecessary = self.config.router.skip_tools_when_unnecessary;
        let default_model = self.providers.default_name().to_string();

        // Build the Commands domain candidates: each command as "{name}: {description}".
        let command_candidates: Vec<RoutingCandidate> = all_commands
            .iter()
            .map(|cmd| RoutingCandidate {
                id: cmd.name.clone(),
                examples: vec![format!("{}: {}", cmd.name, cmd.description)],
            })
            .collect();

        // Build all routing domains.
        let mut domains: Vec<(RoutingDomainKind, Vec<RoutingCandidate>)> =
            vec![(RoutingDomainKind::Commands, command_candidates)];

        // Model domain: only include if there are multiple models to route between.
        // Filter candidates to only those supporting the required capability (chat_completions).
        let required_capability = Capability::new(Capability::CHAT_COMPLETIONS);
        let total_models: usize = self
            .config
            .router
            .providers
            .iter()
            .map(|p| p.models.len())
            .sum();
        if total_models > 1 {
            let eligible_models = self.providers.models_with_capability(&required_capability);
            let model_candidates: Vec<RoutingCandidate> = build_model_candidates(&self.config)
                .into_iter()
                .filter(|c| eligible_models.contains(&c.id))
                .collect();
            if !model_candidates.is_empty() {
                domains.push((RoutingDomainKind::Model, model_candidates));
            }
        }

        // ToolNecessity domain: include if tool-skipping is enabled in config.
        if skip_tools_when_unnecessary {
            domains.push((
                RoutingDomainKind::ToolNecessity,
                tool_necessity_candidates(),
            ));
        }

        // Memory domain: include when memory stores are configured AND config has a memory section.
        // The config gate preserves the pre-existing invariant: memory domain routing only activates
        // when the operator has explicitly configured memory stores. Without this gate, tests that
        // build a memory service without a matching config section would spuriously activate memory
        // domain routing at request-start (in route_all_domains).
        if self.memory.as_ref().is_some_and(|m| m.is_configured())
            && !self.memory_candidates.is_empty()
            && self.config.memory.is_some()
        {
            let memory_domain_enabled = self
                .config
                .router
                .domains
                .memory
                .as_ref()
                .map(|d| d.enabled)
                .unwrap_or(true);
            if memory_domain_enabled {
                domains.push((RoutingDomainKind::Memory, self.memory_candidates.clone()));
            }
        }

        // ── Per-domain PreRoute hooks (before single router call) ────────────
        // Fire PreRoute for each domain. Collect modified candidates for the router call.
        // Accumulated context from routing hooks.
        let mut context_parts: Vec<String> = Vec::new();

        // Track overrides from PreRoute hooks: domain -> (modified_candidates, modified_routing_input).
        let mut pre_route_overrides: std::collections::HashMap<
            usize,
            (Vec<RoutingCandidate>, Option<String>),
        > = std::collections::HashMap::new();

        // Track whether commands or tool_necessity were blocked by PreRoute hooks.
        let mut commands_blocked = false;
        let mut tool_necessity_blocked = false;

        for (domain_idx, (domain_kind, candidates)) in domains.iter().enumerate() {
            let hook_domain = routing_domain_to_hook_domain(domain_kind);
            let candidate_values: Vec<serde_json::Value> = candidates
                .iter()
                .map(|c| serde_json::json!({"id": c.id, "description": c.examples.first().cloned().unwrap_or_default()}))
                .collect();

            let pre_payload = serde_json::json!({
                "domain": hook_domain,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": candidate_values,
                "routing_input": user_message,
            });

            match self
                .hooks
                .run_chain(
                    HookEvent::PreRoute,
                    pre_payload,
                    Some(hook_domain.as_matcher_target()),
                )
                .await
            {
                HookChainResult::Blocked { reason, hook_name } => {
                    info!(
                        hook = %hook_name,
                        domain = ?domain_kind,
                        reason = %reason,
                        "PreRoute hook blocked routing domain"
                    );
                    match domain_kind {
                        RoutingDomainKind::Model => {
                            // Hard block on model domain — terminate request.
                            return Err(WeftError::HookBlocked {
                                event: "PreRoute".to_string(),
                                reason,
                                hook_name,
                            });
                        }
                        RoutingDomainKind::Commands => {
                            commands_blocked = true;
                        }
                        RoutingDomainKind::ToolNecessity => {
                            tool_necessity_blocked = true;
                        }
                        RoutingDomainKind::Memory => {
                            // Memory PreRoute block at request_start = hard block per spec.
                            return Err(WeftError::HookBlocked {
                                event: "PreRoute".to_string(),
                                reason,
                                hook_name,
                            });
                        }
                    }
                }
                HookChainResult::Allowed { payload, context } => {
                    if let Some(ctx) = context {
                        context_parts.push(ctx);
                    }
                    // Extract any modifications to candidates or routing_input.
                    let modified_candidates = payload
                        .get("candidates")
                        .and_then(|v| {
                            serde_json::from_value::<Vec<serde_json::Value>>(v.clone()).ok()
                        })
                        .map(|cv| {
                            cv.into_iter()
                                .filter_map(|c| {
                                    let id = c.get("id")?.as_str()?.to_string();
                                    let desc = c
                                        .get("description")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    Some(RoutingCandidate {
                                        id,
                                        examples: vec![desc],
                                    })
                                })
                                .collect::<Vec<_>>()
                        });
                    let modified_routing_input = payload
                        .get("routing_input")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    if modified_candidates.is_some() || modified_routing_input.is_some() {
                        let eff_candidates =
                            modified_candidates.unwrap_or_else(|| candidates.clone());
                        pre_route_overrides
                            .insert(domain_idx, (eff_candidates, modified_routing_input));
                    }
                }
            }
        }

        // Apply PreRoute overrides to the domains array.
        for (idx, (new_candidates, _)) in &pre_route_overrides {
            if let Some((_, cands)) = domains.get_mut(*idx) {
                *cands = new_candidates.clone();
            }
        }

        // Use modified routing_input if any domain had it overridden.
        // For simplicity, if multiple domains have different routing_input overrides,
        // the last one wins (spec says it affects only that domain's scoring, but
        // we use a single route() call so we use the user_message for all domains).
        let effective_routing_input = pre_route_overrides
            .values()
            .filter_map(|(_, ri)| ri.as_deref())
            .last()
            .unwrap_or(user_message);

        // ── Pure routing call ────────────────────────────────────────────────
        // `route_domains` handles the router fallback internally. On failure it
        // returns conservative defaults (all commands, inject_tools=true, default model).
        let routing_input = RoutingInput {
            user_message: effective_routing_input,
            all_commands,
            threshold,
            max_commands,
            skip_tools_when_unnecessary,
            default_model: &default_model,
            domains: domains.clone(),
        };

        let mut result = route_domains(&*self.router, &routing_input)
            .await
            .map_err(|e| WeftError::Routing(e.to_string()))?;

        // If commands were blocked by PreRoute, clear the selected commands.
        if commands_blocked {
            result.selected_commands.clear();
        }

        // If tool_necessity was blocked by PreRoute, force conservative inject_tools=true.
        if tool_necessity_blocked {
            result.inject_tools = true;
        }

        // Log router fallback if score is 0.0 (signals fallback path in route_domains).
        if result.activity_events.first().map(|a| a.score) == Some(0.0) {
            warn!(
                "semantic router failed, using fallback: all commands, inject tools, default model"
            );
        }

        // ── Per-domain PostRoute hooks ────────────────────────────────────────
        // PostRoute hooks may override the selections made by route_domains.

        // PostRoute: Commands domain.
        {
            let commands_domain_candidates = domains
                .iter()
                .find(|(k, _)| *k == RoutingDomainKind::Commands)
                .map(|(_, c)| c.clone())
                .unwrap_or_default();

            // Build scored commands representation for the hook payload.
            // Use score 1.0 for selected commands as a conservative approximation when
            // we don't have individual scores from route_domains (they're in RoutingResult).
            let selected_cmd_ids: Vec<String> = result
                .selected_commands
                .iter()
                .map(|c| c.name.clone())
                .collect();

            let post_commands_payload = serde_json::json!({
                "domain": HookRoutingDomain::Commands,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": commands_domain_candidates.iter().map(|c| serde_json::json!({"id": c.id, "description": c.examples.first().cloned().unwrap_or_default()})).collect::<Vec<_>>(),
                "scores": selected_cmd_ids.iter().map(|id| serde_json::json!({"id": id, "score": 1.0})).collect::<Vec<_>>(),
                "selected": selected_cmd_ids,
            });

            match self
                .hooks
                .run_chain(
                    HookEvent::PostRoute,
                    post_commands_payload,
                    Some(HookRoutingDomain::Commands.as_matcher_target()),
                )
                .await
            {
                HookChainResult::Blocked { reason, hook_name } => {
                    // Block on commands domain = empty commands (not 403).
                    info!(
                        hook = %hook_name,
                        reason = %reason,
                        "PostRoute hook blocked commands domain — using empty command set"
                    );
                    result.selected_commands.clear();
                }
                HookChainResult::Allowed { payload, context } => {
                    if let Some(ctx) = context {
                        context_parts.push(ctx);
                    }
                    // PostRoute may override the selected command IDs.
                    if let Some(override_ids) = payload
                        .get("selected")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect::<Vec<_>>()
                        })
                    {
                        result.selected_commands = all_commands
                            .iter()
                            .filter(|cmd| override_ids.contains(&cmd.name))
                            .cloned()
                            .collect();
                    }
                }
            }
        }

        // PostRoute: ToolNecessity domain (only if it was included).
        let tool_necessity_domain_exists = domains
            .iter()
            .any(|(k, _)| *k == RoutingDomainKind::ToolNecessity);

        if !tool_necessity_blocked && tool_necessity_domain_exists {
            let selected_necessity = if result.inject_tools {
                "needs_tools"
            } else {
                "no_tools"
            };
            let tn_domain_candidates = domains
                .iter()
                .find(|(k, _)| *k == RoutingDomainKind::ToolNecessity)
                .map(|(_, c)| c.clone())
                .unwrap_or_default();

            let post_tn_payload = serde_json::json!({
                "domain": HookRoutingDomain::ToolNecessity,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": tn_domain_candidates.iter().map(|c| serde_json::json!({"id": c.id})).collect::<Vec<_>>(),
                "scores": [],
                "selected": [selected_necessity],
            });

            match self
                .hooks
                .run_chain(
                    HookEvent::PostRoute,
                    post_tn_payload,
                    Some(HookRoutingDomain::ToolNecessity.as_matcher_target()),
                )
                .await
            {
                HookChainResult::Blocked { hook_name, reason } => {
                    // Block on tool_necessity = conservative (inject tools).
                    info!(
                        hook = %hook_name,
                        reason = %reason,
                        "PostRoute hook blocked tool_necessity domain — conservative: inject tools"
                    );
                    result.inject_tools = true;
                }
                HookChainResult::Allowed { payload, context } => {
                    if let Some(ctx) = context {
                        context_parts.push(ctx);
                    }
                    // PostRoute can override tool_necessity selection.
                    let override_selected = payload
                        .get("selected")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                        .map(String::from);

                    match override_selected.as_deref() {
                        Some("needs_tools") => result.inject_tools = true,
                        Some("no_tools") => result.inject_tools = false,
                        _ => {}
                    }
                }
            }
        }

        // PostRoute: Model domain (only if it was included).
        let model_domain_exists = domains.iter().any(|(k, _)| *k == RoutingDomainKind::Model);

        if model_domain_exists {
            let model_domain_candidates = domains
                .iter()
                .find(|(k, _)| *k == RoutingDomainKind::Model)
                .map(|(_, c)| c.clone())
                .unwrap_or_default();

            let model_score = result
                .activity_events
                .first()
                .map(|a| a.score)
                .unwrap_or(1.0);

            let post_model_payload = serde_json::json!({
                "domain": HookRoutingDomain::Model,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": model_domain_candidates.iter().map(|c| serde_json::json!({"id": c.id})).collect::<Vec<_>>(),
                "scores": [serde_json::json!({"id": result.selected_model, "score": model_score})],
                "selected": [result.selected_model],
            });

            match self
                .hooks
                .run_chain(
                    HookEvent::PostRoute,
                    post_model_payload,
                    Some(HookRoutingDomain::Model.as_matcher_target()),
                )
                .await
            {
                HookChainResult::Blocked { reason, hook_name } => {
                    // Hard block on model domain.
                    return Err(WeftError::HookBlocked {
                        event: "PostRoute".to_string(),
                        reason,
                        hook_name,
                    });
                }
                HookChainResult::Allowed { payload, context } => {
                    if let Some(ctx) = context {
                        context_parts.push(ctx);
                    }
                    let override_model = payload
                        .get("selected")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                        .map(String::from);

                    if let Some(model) = override_model {
                        // Validate that the model exists in the provider registry.
                        if self.providers.model_id(&model).is_some() {
                            // Update activity events to reflect override.
                            for event in &mut result.activity_events {
                                event.model = model.clone();
                            }
                            result.selected_model = model;
                        } else {
                            warn!(
                                model = %model,
                                "PostRoute hook override model not found in registry — using default"
                            );
                            result.selected_model = default_model.clone();
                        }
                    }
                }
            }
        }

        debug!(
            selected_commands = result.selected_commands.len(),
            total_commands = all_commands.len(),
            inject_tools = result.inject_tools,
            model = %result.selected_model,
            "semantic router decision"
        );

        let accumulated_context = if context_parts.is_empty() {
            None
        } else {
            Some(context_parts.join("\n"))
        };

        Ok((result, accumulated_context))
    }
}
