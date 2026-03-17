//! Semantic routing with per-domain hook lifecycle.
//!
//! Provides `route_with_hooks`, which builds routing domains, fires PreRoute
//! and PostRoute hooks for each domain, calls `route_domains` once, and
//! returns the routing result with accumulated hook context.

use tracing::{debug, info, warn};
use weft_commands::CommandRegistry;
use weft_core::HookEvent;
use weft_core::{CommandStub, HookRoutingDomain, RoutingTrigger, WeftError};
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::{Capability, ProviderService};
use weft_memory::MemoryService;
use weft_router::{
    RoutingCandidate, RoutingDomainKind, RoutingInput, RoutingResult, SemanticRouter,
    build_model_candidates, route_domains, tool_necessity_candidates,
};

use crate::GatewayEngine;
use crate::util::routing_domain_to_hook_domain;

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
    pub(crate) async fn route_with_hooks(
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
        // Domains fire in order: Commands, ToolNecessity, Model.  This order
        // is part of the hook contract — do not reorder.

        let tool_necessity_domain_exists = domains
            .iter()
            .any(|(k, _)| *k == RoutingDomainKind::ToolNecessity);
        let model_domain_exists = domains.iter().any(|(k, _)| *k == RoutingDomainKind::Model);

        /// Descriptor for a single domain in the PostRoute loop.
        struct PostRouteDomain {
            kind: RoutingDomainKind,
            hook_domain: HookRoutingDomain,
            /// Was this domain blocked by a PreRoute hook?  Blocked domains are skipped.
            pre_route_blocked: bool,
            /// Does this domain appear in the `domains` vec?  Absent domains are skipped.
            exists: bool,
        }

        let post_route_domains = [
            PostRouteDomain {
                kind: RoutingDomainKind::Commands,
                hook_domain: HookRoutingDomain::Commands,
                pre_route_blocked: commands_blocked,
                exists: true, // Commands domain is always present.
            },
            PostRouteDomain {
                kind: RoutingDomainKind::ToolNecessity,
                hook_domain: HookRoutingDomain::ToolNecessity,
                pre_route_blocked: tool_necessity_blocked,
                exists: tool_necessity_domain_exists,
            },
            PostRouteDomain {
                kind: RoutingDomainKind::Model,
                hook_domain: HookRoutingDomain::Model,
                pre_route_blocked: false, // Model PreRoute block returns Err — never reaches here.
                exists: model_domain_exists,
            },
        ];

        for domain_desc in &post_route_domains {
            if !domain_desc.exists || domain_desc.pre_route_blocked {
                continue;
            }

            let payload = build_post_route_payload(&domains, &result, &domain_desc.kind);

            match self
                .hooks
                .run_chain(
                    HookEvent::PostRoute,
                    payload,
                    Some(domain_desc.hook_domain.as_matcher_target()),
                )
                .await
            {
                HookChainResult::Blocked { reason, hook_name } => {
                    match domain_desc.kind {
                        RoutingDomainKind::Commands => {
                            info!(
                                hook = %hook_name,
                                reason = %reason,
                                "PostRoute hook blocked commands domain — using empty command set"
                            );
                            result.selected_commands.clear();
                        }
                        RoutingDomainKind::ToolNecessity => {
                            info!(
                                hook = %hook_name,
                                reason = %reason,
                                "PostRoute hook blocked tool_necessity domain — conservative: inject tools"
                            );
                            result.inject_tools = true;
                        }
                        RoutingDomainKind::Model => {
                            return Err(WeftError::HookBlocked {
                                event: "PostRoute".to_string(),
                                reason,
                                hook_name,
                            });
                        }
                        RoutingDomainKind::Memory => {
                            unreachable!("Memory domain PostRoute is handled in memory.rs")
                        }
                    }
                }
                HookChainResult::Allowed { payload, context } => {
                    if let Some(ctx) = context {
                        context_parts.push(ctx);
                    }
                    self.apply_post_route_override(
                        &payload,
                        &domain_desc.kind,
                        &mut result,
                        all_commands,
                        &default_model,
                    );
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

    /// Apply the override from an `Allowed` PostRoute hook payload to `result`.
    ///
    /// Extracts the `selected` array from the payload and applies domain-specific
    /// override logic:
    /// - **Commands:** filters `all_commands` to those whose name appears in
    ///   `selected`, replaces `result.selected_commands`.
    /// - **ToolNecessity:** reads the first `selected` entry (`"needs_tools"` /
    ///   `"no_tools"`), sets `result.inject_tools`.
    /// - **Model:** reads the first `selected` entry, validates against the
    ///   provider registry, updates `result.selected_model` and activity events.
    ///   Falls back to `default_model` if the override name is unknown.
    fn apply_post_route_override(
        &self,
        payload: &serde_json::Value,
        kind: &RoutingDomainKind,
        result: &mut RoutingResult,
        all_commands: &[CommandStub],
        default_model: &str,
    ) {
        match kind {
            RoutingDomainKind::Commands => {
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
            RoutingDomainKind::ToolNecessity => {
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
            RoutingDomainKind::Model => {
                let override_model = payload
                    .get("selected")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
                    .map(String::from);

                if let Some(model) = override_model {
                    if self.providers.model_id(&model).is_some() {
                        for event in &mut result.activity_events {
                            event.model = model.clone();
                        }
                        result.selected_model = model;
                    } else {
                        warn!(
                            model = %model,
                            "PostRoute hook override model not found in registry — using default"
                        );
                        result.selected_model = default_model.to_string();
                    }
                }
            }
            RoutingDomainKind::Memory => {
                unreachable!("Memory domain PostRoute is handled in memory.rs")
            }
        }
    }
}

/// Build the `candidates`, `scores`, and `selected` payload for a PostRoute hook.
///
/// Each domain has a slightly different payload shape:
/// - **Commands:** candidates include `id` + `description`; scores use 1.0 for
///   each selected command; selected is an array of command-name strings.
/// - **ToolNecessity:** candidates include `id` only; scores is empty; selected
///   is `["needs_tools"]` or `["no_tools"]`.
/// - **Model:** candidates include `id` only; scores include the model score from
///   the first activity event; selected is `[model_name]`.
fn build_post_route_payload(
    domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    result: &RoutingResult,
    kind: &RoutingDomainKind,
) -> serde_json::Value {
    let hook_domain = crate::util::routing_domain_to_hook_domain(kind);
    let domain_candidates: Vec<RoutingCandidate> = domains
        .iter()
        .find(|(k, _)| k == kind)
        .map(|(_, c)| c.clone())
        .unwrap_or_default();

    match kind {
        RoutingDomainKind::Commands => {
            let selected_cmd_ids: Vec<String> = result
                .selected_commands
                .iter()
                .map(|c| c.name.clone())
                .collect();
            serde_json::json!({
                "domain": hook_domain,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": domain_candidates.iter().map(|c| serde_json::json!({
                    "id": c.id,
                    "description": c.examples.first().cloned().unwrap_or_default(),
                })).collect::<Vec<_>>(),
                "scores": selected_cmd_ids.iter().map(|id| serde_json::json!({"id": id, "score": 1.0})).collect::<Vec<_>>(),
                "selected": selected_cmd_ids,
            })
        }
        RoutingDomainKind::ToolNecessity => {
            let selected_necessity = if result.inject_tools {
                "needs_tools"
            } else {
                "no_tools"
            };
            serde_json::json!({
                "domain": hook_domain,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": domain_candidates.iter().map(|c| serde_json::json!({"id": c.id})).collect::<Vec<_>>(),
                "scores": [],
                "selected": [selected_necessity],
            })
        }
        RoutingDomainKind::Model => {
            let model_score = result
                .activity_events
                .first()
                .map(|a| a.score)
                .unwrap_or(1.0);
            serde_json::json!({
                "domain": hook_domain,
                "trigger": RoutingTrigger::RequestStart,
                "candidates": domain_candidates.iter().map(|c| serde_json::json!({"id": c.id})).collect::<Vec<_>>(),
                "scores": [serde_json::json!({"id": result.selected_model, "score": model_score})],
                "selected": [result.selected_model],
            })
        }
        RoutingDomainKind::Memory => {
            unreachable!("Memory domain PostRoute is handled in memory.rs")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use weft_core::{
        ClassifierConfig, ContentPart, DomainConfig, DomainsConfig, GatewayConfig, MemoryConfig,
        MemoryStoreConfig, ModelEntry, ProviderConfig, Role, RouterConfig, ServerConfig, Source,
        StoreCapability, WeftConfig, WeftMessage,
    };
    use weft_llm::{
        Capability, Provider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
    };
    use weft_memory::{DefaultMemoryService, MemoryStoreMux, StoreInfo};
    use weft_router::{
        MemoryStoreRef, RoutingCandidate, RoutingDecision, RoutingDomainKind,
        build_memory_candidates,
    };

    // ── Build a WeftConfig with memory stores ──────────────────────────────

    /// Build a WeftConfig with memory stores for routing tests.
    ///
    /// `stores`: slice of `(name, endpoint, can_read, can_write, examples)`.
    /// `memory_threshold`: optional per-domain memory threshold.
    fn config_with_memory(
        stores: &[(&str, &str, bool, bool, Vec<&str>)],
        memory_threshold: Option<f32>,
    ) -> Arc<WeftConfig> {
        let store_configs: Vec<MemoryStoreConfig> = stores
            .iter()
            .map(|(name, endpoint, can_read, can_write, examples)| {
                let mut capabilities = Vec::new();
                if *can_read {
                    capabilities.push(StoreCapability::Read);
                }
                if *can_write {
                    capabilities.push(StoreCapability::Write);
                }
                MemoryStoreConfig {
                    name: name.to_string(),
                    endpoint: endpoint.to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities,
                    examples: examples.iter().map(|s| s.to_string()).collect(),
                }
            })
            .collect();

        let memory_domain = memory_threshold.map(|t| DomainConfig {
            enabled: true,
            threshold: Some(t),
        });

        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.5, // default threshold for memory routing tests
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: weft_core::WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                        capabilities: vec!["chat_completions".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig {
                    model: None,
                    tool_necessity: None,
                    memory: memory_domain,
                },
            },
            tool_registry: None,
            memory: Some(MemoryConfig {
                stores: store_configs,
            }),
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    /// Build an engine with the given config, mux, and router.
    ///
    /// The config provides routing thresholds, domain settings, AND store metadata
    /// (examples) for building routing candidates. The mux handles actual ops.
    /// Internally constructs a `DefaultMemoryService` from the config's memory section.
    fn make_engine_with_config_and_mux<R, C>(
        config: Arc<WeftConfig>,
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
        mux: Option<Arc<MemoryStoreMux>>,
    ) -> crate::GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
    where
        R: weft_router::SemanticRouter + Send + Sync + 'static,
        C: weft_commands::CommandRegistry + Send + Sync + 'static,
    {
        let memory = mux.map(|m| service_from_config_and_mux(&config, m));
        crate::GatewayEngine::new(
            config,
            registry,
            Arc::new(router),
            Arc::new(commands),
            memory,
            Arc::new(weft_hooks::HookRegistry::empty()),
        )
    }

    /// Build a `DefaultMemoryService` from a config and a mux.
    ///
    /// Extracts `StoreInfo` (name, capabilities, examples) from the config's memory section
    /// so that routing candidates are populated correctly, then wraps the mux for operations.
    fn service_from_config_and_mux(
        config: &WeftConfig,
        mux: Arc<MemoryStoreMux>,
    ) -> Arc<DefaultMemoryService> {
        let store_infos: Vec<StoreInfo> = config
            .memory
            .as_ref()
            .map(|mem| {
                mem.stores
                    .iter()
                    .map(|s| {
                        let mut caps = Vec::new();
                        if s.can_read() {
                            caps.push("read".to_string());
                        }
                        if s.can_write() {
                            caps.push("write".to_string());
                        }
                        StoreInfo {
                            name: s.name.clone(),
                            capabilities: caps,
                            examples: s.examples.clone(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Arc::new(DefaultMemoryService::new(mux, store_infos))
    }

    // ── Test: model selection uses router decision ─────────────────────────

    #[tokio::test]
    async fn test_model_selection_uses_router_decision() {
        // Router selects "complex-model"; we verify the correct model_id is passed.
        let recording_provider = Arc::new(RecordingLlmProvider::new("response"));
        let default_provider = Arc::new(MockLlmProvider::single("default response"));

        let mut providers = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            default_provider as Arc<dyn Provider>,
        );
        providers.insert(
            "complex-model".to_string(),
            recording_provider.clone() as Arc<dyn Provider>,
        );

        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert("complex-model".to_string(), "claude-complex".to_string());

        let mut max_tokens_map = HashMap::new();
        max_tokens_map.insert("default-model".to_string(), 1024u32);
        max_tokens_map.insert("complex-model".to_string(), 4096u32);

        // Both models support chat_completions for this test.
        let mut caps_map: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps_map.insert(
            "default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        caps_map.insert(
            "complex-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens_map,
            caps_map,
            "default-model".to_string(),
        ));

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        engine
            .handle_request(make_user_request("Complex task"))
            .await
            .expect("should succeed");

        // Verify the correct model_id was passed to the recording provider
        assert_eq!(
            recording_provider.recorded_model(),
            Some("claude-complex".to_string()),
            "complex-model provider must receive the correct model_id"
        );
    }

    // ── Test: tool skipping when tools_needed = Some(false) ───────────────

    #[tokio::test]
    async fn test_tool_skipping_when_tools_not_needed() {
        // Provider records the system prompt to verify no command stubs
        struct SystemPromptCapture {
            captured: std::sync::Mutex<Option<String>>,
            response_text: &'static str,
        }

        #[async_trait::async_trait]
        impl Provider for SystemPromptCapture {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Extract the system prompt from messages[0] (Role::System convention).
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                    let system_text = messages
                        .first()
                        .filter(|m| m.role == Role::System)
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| match p {
                                    ContentPart::Text(t) => Some(t.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    *self.captured.lock().unwrap() = Some(system_text);
                }
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text(self.response_text.to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "system-prompt-capture"
            }
        }

        let capture = Arc::new(SystemPromptCapture {
            captured: std::sync::Mutex::new(None),
            response_text: "no tools needed",
        });

        let registry = single_model_registry(
            {
                struct WrappedCapture(Arc<SystemPromptCapture>);
                #[async_trait::async_trait]
                impl Provider for WrappedCapture {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "wrapped-capture"
                    }
                }
                WrappedCapture(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            MockRouter::with_tools_needed(false),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("What is the capital of France?"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // When tools are skipped, the system prompt must NOT contain command stubs
        assert!(
            !captured.contains("```toon"),
            "system prompt should not contain toon block when tools are skipped: {captured}"
        );
        assert!(
            !captured.contains("web_search"),
            "system prompt should not contain command stubs when tools are skipped"
        );
    }

    // ── Test: tool injection when tools_needed = Some(true) ───────────────

    #[tokio::test]
    async fn test_tool_injection_when_tools_needed() {
        struct SystemPromptCapture2 {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait::async_trait]
        impl Provider for SystemPromptCapture2 {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Extract the system prompt from messages[0] (Role::System convention).
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                    let system_text = messages
                        .first()
                        .filter(|m| m.role == Role::System)
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| match p {
                                    ContentPart::Text(t) => Some(t.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    *self.captured.lock().unwrap() = Some(system_text);
                }
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text("response".to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "system-prompt-capture2"
            }
        }

        let capture = Arc::new(SystemPromptCapture2 {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture2>);
                #[async_trait::async_trait]
                impl Provider for W {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "w2"
                    }
                }
                W(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            MockRouter::with_tools_needed(true),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("Search for something"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // Commands must be injected in dash-list format with "Use when" trigger conditions
        assert!(
            captured.contains("- /web_search \u{2014} Use when Search the web"),
            "system prompt must contain dash-list command stubs when tools are needed"
        );
        assert!(
            captured.contains("BLOCKING REQUIREMENT"),
            "system prompt must contain blocking requirement language when tools are needed"
        );
    }

    // ── Test: tool injection when tools_needed = None (conservative) ──────

    #[tokio::test]
    async fn test_tool_injection_when_tools_needed_is_none() {
        // tools_needed = None -> conservative default: inject commands
        let router = MockRouter {
            command_score: 1.0,
            model_decision: None,
            tools_needed: None, // undecided
            memory_score: Some(0.9),
        };

        struct SystemPromptCapture3 {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait::async_trait]
        impl Provider for SystemPromptCapture3 {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Extract the system prompt from messages[0] (Role::System convention).
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                    let system_text = messages
                        .first()
                        .filter(|m| m.role == Role::System)
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| match p {
                                    ContentPart::Text(t) => Some(t.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    *self.captured.lock().unwrap() = Some(system_text);
                }
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text("response".to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "system-prompt-capture3"
            }
        }

        let capture = Arc::new(SystemPromptCapture3 {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture3>);
                #[async_trait::async_trait]
                impl Provider for W {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "w3"
                    }
                }
                W(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            router,
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("Tell me about Rust"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // None -> conservative: inject commands
        assert!(
            captured.contains("web_search"),
            "tools_needed=None must default to injecting commands"
        );
    }

    // ── Test: model fallback on non-rate-limit error ───────────────────────

    #[tokio::test]
    async fn test_model_fallback_on_non_rate_limit_error() {
        // complex-model fails with RequestFailed; default-model succeeds.
        let registry = two_model_registry(
            MockLlmProvider::single("fallback response"),
            FailingLlmProvider,
        );

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Complex task"))
            .await
            .expect("fallback to default must succeed");

        assert_eq!(resp_text(&resp), "fallback response");
    }

    // ── Test: rate limit does NOT trigger fallback ─────────────────────────

    #[tokio::test]
    async fn test_rate_limit_does_not_trigger_fallback() {
        // complex-model is rate-limited; must not fall back.
        let registry = two_model_registry(
            MockLlmProvider::single("default would succeed"),
            RateLimitedLlmProvider,
        );

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Complex task"))
            .await;
        assert!(
            matches!(result, Err(weft_core::WeftError::RateLimited { .. })),
            "rate limit must propagate without fallback, got: {:?}",
            result
        );
    }

    // ── Test: default model failure propagates error ───────────────────────

    #[tokio::test]
    async fn test_default_model_failure_propagates_error() {
        // Only the default model exists and it fails — no fallback available.
        let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");

        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(weft_core::WeftError::Llm(_))),
            "default model failure must propagate as Llm error, got: {:?}",
            result
        );
    }

    // ── Test: selected model IS default, no retry on failure ──────────────

    #[tokio::test]
    async fn test_selected_model_is_default_no_retry() {
        // Router selects "default-model" which is also the default — fails once,
        // must NOT retry (would infinite loop).
        let registry = two_model_registry(FailingLlmProvider, MockLlmProvider::single("alt"));

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            // Router selects "default-model" (same as the registry's default)
            MockRouter::with_model("default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Simple task"))
            .await;
        assert!(
            matches!(result, Err(weft_core::WeftError::Llm(_))),
            "selected=default failure must propagate without retry, got: {:?}",
            result
        );
    }

    // ── Test: context assembly uses TOON fenced blocks ─────────────────────

    #[tokio::test]
    async fn test_context_assembly_uses_toon_fenced_blocks() {
        use crate::context::assemble_system_prompt;
        use weft_core::CommandStub;

        let stubs = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];
        let prompt = assemble_system_prompt(&stubs, "You are a helpful assistant.", None);

        // Command stubs use dash-list format with "Use when" trigger conditions,
        // not TOON fenced blocks — TOON is reserved for results injection only.
        assert!(
            prompt.contains("- /web_search \u{2014} Use when Search the web"),
            "must use dash-list format with Use when trigger condition"
        );
        assert!(
            prompt.contains("BLOCKING REQUIREMENT"),
            "must include blocking requirement language"
        );
        assert!(
            !prompt.contains("commands[1]{name, description}:"),
            "must not use TOON array syntax for command stubs"
        );
    }

    // ── Test: no fenced blocks when tool skipping is active ───────────────

    #[tokio::test]
    async fn test_no_fenced_blocks_when_tool_skipping() {
        use crate::context::assemble_system_prompt_no_tools;

        let prompt = assemble_system_prompt_no_tools("You are a helpful assistant.");
        assert!(
            !prompt.contains("```toon"),
            "must not contain toon blocks when tools are skipped"
        );
        assert!(
            !prompt.contains("commands["),
            "must not contain command stubs when tools are skipped"
        );
    }

    // ── build_memory_candidates (via weft_router) ─────────────────────────

    #[test]
    fn test_build_memory_candidates_empty_when_no_stores() {
        let result = build_memory_candidates(&[]);
        assert!(result.all.is_empty());
        assert!(result.read.is_empty());
        assert!(result.write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_read_write_store() {
        let stores = vec![MemoryStoreRef {
            name: "conv".to_string(),
            capabilities: vec!["read".to_string(), "write".to_string()],
            examples: vec!["recall conv".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert_eq!(result.all[0].id, "conv");
        assert_eq!(result.read.len(), 1);
        assert_eq!(result.read[0].id, "conv");
        assert_eq!(result.write.len(), 1);
        assert_eq!(result.write[0].id, "conv");
    }

    #[test]
    fn test_build_memory_candidates_read_only_store() {
        let stores = vec![MemoryStoreRef {
            name: "kb".to_string(),
            capabilities: vec!["read".to_string()],
            examples: vec!["knowledge base".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert_eq!(result.read.len(), 1);
        assert!(result.write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_write_only_store() {
        let stores = vec![MemoryStoreRef {
            name: "audit".to_string(),
            capabilities: vec!["write".to_string()],
            examples: vec!["audit log".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert!(result.read.is_empty());
        assert_eq!(result.write.len(), 1);
    }

    #[test]
    fn test_build_memory_candidates_multiple_stores() {
        let stores = vec![
            MemoryStoreRef {
                name: "conv".to_string(),
                capabilities: vec!["read".to_string(), "write".to_string()],
                examples: vec!["conversation".to_string()],
            },
            MemoryStoreRef {
                name: "kb".to_string(),
                capabilities: vec!["read".to_string()],
                examples: vec!["knowledge base".to_string()],
            },
            MemoryStoreRef {
                name: "audit".to_string(),
                capabilities: vec!["write".to_string()],
                examples: vec!["audit".to_string()],
            },
        ];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 3);
        assert_eq!(result.read.len(), 2); // conv + kb
        assert_eq!(result.write.len(), 2); // conv + audit
        assert!(result.read.iter().any(|c| c.id == "conv"));
        assert!(result.read.iter().any(|c| c.id == "kb"));
        assert!(result.write.iter().any(|c| c.id == "conv"));
        assert!(result.write.iter().any(|c| c.id == "audit"));
    }

    // ── route_all_domains includes Memory domain ──────────────────────────

    #[tokio::test]
    async fn test_route_all_domains_includes_memory_when_configured() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "dark mode pref",
        )]));
        let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

        let config = config_with_memory(
            &[(
                "conv",
                "http://localhost:50052",
                true,
                true,
                vec!["conversation recall"],
            )],
            None, // no memory-domain threshold override — uses classifier.threshold = 0.5
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "I found memory about dark mode.",
            ]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Tell me about preferences"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "I found memory about dark mode.");
    }

    // ── /recall per-invocation routing ─────────────────────────────────────

    #[tokio::test]
    async fn test_recall_routes_based_on_query_not_user_message() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "user prefers dark mode",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::query_fails("should not be called"));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["user preferences"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["code architecture"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "Found: user prefers dark mode.",
            ]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("help me with my code"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found: user prefers dark mode.");
    }

    #[tokio::test]
    async fn test_recall_below_threshold_fans_out_to_all_readable() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "conv memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m2",
            "kb memory",
        )]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            Some(0.5), // memory domain threshold
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something unrelated\"",
                "Found memories from both stores.",
            ]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.1);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("something"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found memories from both stores.");
    }

    #[tokio::test]
    async fn test_recall_router_unavailable_fans_out_to_all_readable() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "conv memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m2",
            "kb memory",
        )]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"anything\"",
                "Got results from both stores.",
            ]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_unavailable();

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("anything"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Got results from both stores.");
    }

    // ── /remember per-invocation routing ───────────────────────────────────

    #[tokio::test]
    async fn test_remember_routes_based_on_content_not_user_message() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["user preferences"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit logs"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Noted, I'll remember that.",
            ]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("how do I configure nginx?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Noted, I'll remember that.");
    }

    #[tokio::test]
    async fn test_remember_below_threshold_picks_highest_scoring_store() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["conversation prefs"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit events"],
                ),
            ],
            Some(0.5), // threshold: 0.5
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/remember content: \"something\"", "Memory stored."]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.3);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("remember something"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Memory stored.");
    }

    #[tokio::test]
    async fn test_remember_router_unavailable_writes_to_first_writable() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["conversations"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/remember content: \"important note\"", "Saved."]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_unavailable();

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("remember this"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Saved.");
    }

    // ── Memory domain threshold from config ───────────────────────────────

    #[tokio::test]
    async fn test_memory_domain_threshold_gate_applied() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1", "memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            Some(0.8), // memory domain threshold = 0.8 (higher than classifier 0.5)
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"something\"", "Found memories."]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.7);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("recall something"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found memories.");
    }

    // ── RoutingDecision::fallback() memory_stores population ──────────────

    #[test]
    fn test_fallback_decision_populates_memory_stores() {
        let domains = vec![
            (
                RoutingDomainKind::Commands,
                vec![RoutingCandidate {
                    id: "search".to_string(),
                    examples: vec!["search: Search".to_string()],
                }],
            ),
            (
                RoutingDomainKind::Memory,
                vec![
                    RoutingCandidate {
                        id: "conv".to_string(),
                        examples: vec!["conversations".to_string()],
                    },
                    RoutingCandidate {
                        id: "kb".to_string(),
                        examples: vec!["knowledge base".to_string()],
                    },
                ],
            ),
        ];

        let decision = RoutingDecision::fallback(&domains);

        assert_eq!(decision.commands.len(), 1);
        assert_eq!(decision.memory_stores.len(), 2);
        assert_eq!(decision.memory_stores[0].id, "conv");
        assert_eq!(decision.memory_stores[1].id, "kb");
        for m in &decision.memory_stores {
            assert_eq!(m.score, 1.0, "fallback memory scores should be 1.0");
        }
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
    }

    #[test]
    fn test_fallback_decision_memory_stores_empty_when_no_memory_domain() {
        let domains = vec![(
            RoutingDomainKind::Commands,
            vec![RoutingCandidate {
                id: "search".to_string(),
                examples: vec!["search: Search".to_string()],
            }],
        )];

        let decision = RoutingDecision::fallback(&domains);
        assert!(decision.memory_stores.is_empty());
    }

    // ── Memory domain disabled via config ──────────────────────────────────

    #[tokio::test]
    async fn test_recall_with_memory_domain_disabled_fans_out_to_all() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "memory found",
        )]));
        let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

        let disabled_domain = DomainConfig {
            enabled: false,
            threshold: None,
        };
        let config = Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.5,
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: weft_core::WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                        capabilities: vec!["chat_completions".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig {
                    model: None,
                    tool_necessity: None,
                    memory: Some(disabled_domain),
                },
            },
            tool_registry: None,
            memory: Some(MemoryConfig {
                stores: vec![MemoryStoreConfig {
                    name: "conv".to_string(),
                    endpoint: "http://localhost:50052".to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities: vec![StoreCapability::Read, StoreCapability::Write],
                    examples: vec!["conversation".to_string()],
                }],
            }),
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"test query\"", "Found something."]),
            "test-model",
            "claude-test",
        );

        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("recall test"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found something.");
    }
}
