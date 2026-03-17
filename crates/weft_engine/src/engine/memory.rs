//! Memory command orchestration for the gateway engine.
//!
//! Provides `handle_builtin_with_hooks` (the full hook lifecycle for `/recall`
//! and `/remember`), the underlying `exec_recall_with_stores` and
//! `exec_remember_with_stores` executors, and `memory_domain_threshold`.

use tracing::{info, warn};
use weft_commands::CommandRegistry;
use weft_core::{
    CommandAction, CommandInvocation, CommandResult, HookRoutingDomain, RoutingTrigger,
};
use weft_core::HookEvent;
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::ProviderService;
use weft_memory::MemoryService;
use weft_router::{RoutingCandidate, SemanticRouter};

use super::GatewayEngine;
use super::util::builtin_describe_text;

impl<H, R, M, P, C> GatewayEngine<H, R, M, P, C>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    /// Handle a built-in memory command (`/recall` or `/remember`) with full hook lifecycle.
    ///
    /// Hook firing order for memory commands:
    /// 1. PreToolUse — can block entirely (feedback block) or modify arguments.
    /// 2. PreRoute(memory, trigger=recall|remember) — can block (feedback) or modify candidates.
    /// 3. PostRoute(memory, trigger=recall|remember) — can override selected stores or scores.
    /// 4. Execute the command.
    /// 5. PostToolUse — can modify the result.
    pub(super) async fn handle_builtin_with_hooks(
        &self,
        invocation: &CommandInvocation,
        user_message: &str,
    ) -> CommandResult {
        // ── [HOOK: PreToolUse] ───────────────────────────────────────────────
        // Fires before any routing hooks, regardless of command type.
        let tool_payload = serde_json::json!({
            "command": invocation.name,
            "arguments": invocation.arguments,
            "action": match &invocation.action {
                CommandAction::Execute => "execute",
                CommandAction::Describe => "describe",
            },
        });

        let (effective_invocation, pre_tool_blocked) = match self
            .hooks
            .run_chain(HookEvent::PreToolUse, tool_payload, Some(&invocation.name))
            .await
        {
            HookChainResult::Blocked { reason, hook_name } => {
                info!(
                    hook = %hook_name,
                    command = %invocation.name,
                    reason = %reason,
                    "PreToolUse hook blocked built-in memory command — routing hooks skipped"
                );
                return CommandResult {
                    command_name: invocation.name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(reason),
                };
            }
            HookChainResult::Allowed { payload, .. } => {
                // Extract potentially modified arguments.
                let mut inv = invocation.clone();
                if let Some(args) = payload.get("arguments")
                    && let Ok(updated) = serde_json::from_value(args.clone())
                {
                    inv.arguments = updated;
                }
                (inv, false)
            }
        };

        // If describe action, return after PreToolUse + PostToolUse (no routing hooks for describe).
        // PostToolUse fires for both execute and describe actions per spec.
        if matches!(effective_invocation.action, CommandAction::Describe) {
            let mut cmd_result = CommandResult {
                command_name: effective_invocation.name.clone(),
                success: true,
                output: builtin_describe_text(&effective_invocation.name),
                error: None,
            };

            let post_tool_payload = serde_json::json!({
                "command": effective_invocation.name,
                "action": "describe",
                "success": cmd_result.success,
                "output": cmd_result.output,
                "error": cmd_result.error,
            });

            match self
                .hooks
                .run_chain(
                    HookEvent::PostToolUse,
                    post_tool_payload,
                    Some(&effective_invocation.name),
                )
                .await
            {
                HookChainResult::Allowed { payload, .. } => {
                    if let Some(output) = payload.get("output").and_then(|v| v.as_str()) {
                        cmd_result.output = output.to_string();
                    }
                    if let Some(success) = payload.get("success").and_then(|v| v.as_bool()) {
                        cmd_result.success = success;
                    }
                }
                HookChainResult::Blocked { hook_name, reason } => {
                    warn!(
                        hook = %hook_name,
                        reason = %reason,
                        "PostToolUse hook returned Block (non-blocking event) — ignoring"
                    );
                }
            }

            return cmd_result;
        }

        let _ = pre_tool_blocked; // Always false at this point (we returned early on block).

        // Determine routing trigger and extract the routing text.
        let (trigger, routing_text) = match effective_invocation.name.as_str() {
            "recall" => {
                let query = effective_invocation
                    .arguments
                    .get("query")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| user_message.to_string());
                (RoutingTrigger::RecallCommand, query)
            }
            "remember" => {
                let content = effective_invocation
                    .arguments
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                (RoutingTrigger::RememberCommand, content)
            }
            name => {
                return CommandResult {
                    command_name: name.to_string(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("{name}: unknown built-in command")),
                };
            }
        };

        // Determine base candidates for this memory command type.
        let base_candidates = match trigger {
            RoutingTrigger::RecallCommand => &self.read_memory_candidates,
            RoutingTrigger::RememberCommand => &self.write_memory_candidates,
            RoutingTrigger::RequestStart => &self.memory_candidates, // unreachable here
        };

        // ── [HOOK: PreRoute(memory, trigger)] ────────────────────────────────
        let pre_route_payload = serde_json::json!({
            "domain": HookRoutingDomain::Memory,
            "trigger": trigger,
            "candidates": base_candidates.iter().map(|c| serde_json::json!({"id": c.id, "description": c.examples.first().cloned().unwrap_or_default()})).collect::<Vec<_>>(),
            "routing_input": routing_text,
        });

        let (effective_candidates, effective_routing_text) = match self
            .hooks
            .run_chain(
                HookEvent::PreRoute,
                pre_route_payload,
                Some(HookRoutingDomain::Memory.as_matcher_target()),
            )
            .await
        {
            HookChainResult::Blocked { reason, hook_name } => {
                // Feedback block — return failed CommandResult (LLM sees this).
                info!(
                    hook = %hook_name,
                    command = %effective_invocation.name,
                    reason = %reason,
                    "PreRoute memory hook blocked — returning failed CommandResult"
                );
                return CommandResult {
                    command_name: effective_invocation.name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(reason),
                };
            }
            HookChainResult::Allowed { payload, .. } => {
                let modified_candidates = payload
                    .get("candidates")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
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
                    })
                    .unwrap_or_else(|| base_candidates.clone());

                let modified_routing_input = payload
                    .get("routing_input")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or(routing_text.clone());

                (modified_candidates, modified_routing_input)
            }
        };

        // Score the (possibly modified) candidates.
        let threshold = self.memory_domain_threshold();
        let scored = self
            .router
            .score_memory_candidates(&effective_routing_text, &effective_candidates)
            .await;

        let (scored_candidates, above_threshold_ids) = match scored {
            Ok(scored) => {
                let above: Vec<String> = scored
                    .iter()
                    .filter(|c| c.score >= threshold)
                    .map(|c| c.id.clone())
                    .collect();
                (scored, above)
            }
            Err(e) => {
                warn!(error = %e, "router unavailable for memory scoring — using fallback");
                (vec![], vec![])
            }
        };

        // ── [HOOK: PostRoute(memory, trigger)] ───────────────────────────────
        let post_route_payload = serde_json::json!({
            "domain": HookRoutingDomain::Memory,
            "trigger": trigger,
            "candidates": effective_candidates.iter().map(|c| serde_json::json!({"id": c.id})).collect::<Vec<_>>(),
            "scores": scored_candidates.iter().map(|c| serde_json::json!({"id": c.id, "score": c.score})).collect::<Vec<_>>(),
            "selected": above_threshold_ids,
        });

        let final_store_ids = match self
            .hooks
            .run_chain(
                HookEvent::PostRoute,
                post_route_payload,
                Some(HookRoutingDomain::Memory.as_matcher_target()),
            )
            .await
        {
            HookChainResult::Blocked { reason, hook_name } => {
                // Feedback block — return failed CommandResult.
                info!(
                    hook = %hook_name,
                    command = %effective_invocation.name,
                    reason = %reason,
                    "PostRoute memory hook blocked — returning failed CommandResult"
                );
                return CommandResult {
                    command_name: effective_invocation.name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(reason),
                };
            }
            HookChainResult::Allowed { payload, .. } => {
                let overridden_ids =
                    payload
                        .get("selected")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect::<Vec<_>>()
                        });
                overridden_ids.unwrap_or(above_threshold_ids)
            }
        };

        // Validate that selected store names exist in the memory service.
        // Drop unknown stores with a warning. Capability is enforced by the service itself.
        let validated_store_ids = if let Some(mem) = &self.memory {
            let all_stores: std::collections::HashSet<String> = {
                let cap_filter = match trigger {
                    RoutingTrigger::RecallCommand => "read",
                    RoutingTrigger::RememberCommand => "write",
                    RoutingTrigger::RequestStart => "",
                };
                mem.stores()
                    .into_iter()
                    .filter(|s| {
                        cap_filter.is_empty() || s.capabilities.iter().any(|c| c == cap_filter)
                    })
                    .map(|s| s.name)
                    .collect()
            };

            let validated: Vec<String> = final_store_ids
                .iter()
                .filter(|id| {
                    if all_stores.contains(*id) {
                        true
                    } else {
                        warn!(
                            store = %id,
                            "PostRoute hook selected unknown memory store — dropping"
                        );
                        false
                    }
                })
                .cloned()
                .collect();
            validated
        } else {
            final_store_ids
        };

        // ── Execute the memory command ────────────────────────────────────────
        let mut cmd_result = match effective_invocation.name.as_str() {
            "recall" => {
                self.exec_recall_with_stores(
                    &effective_invocation,
                    user_message,
                    validated_store_ids,
                )
                .await
            }
            "remember" => {
                self.exec_remember_with_stores(&effective_invocation, validated_store_ids)
                    .await
            }
            name => CommandResult {
                command_name: name.to_string(),
                success: false,
                output: String::new(),
                error: Some(format!("{name}: unknown built-in command")),
            },
        };

        // ── [HOOK: PostToolUse] ──────────────────────────────────────────────
        let post_tool_payload = serde_json::json!({
            "command": effective_invocation.name,
            "action": "execute",
            "success": cmd_result.success,
            "output": cmd_result.output,
            "error": cmd_result.error,
        });

        match self
            .hooks
            .run_chain(
                HookEvent::PostToolUse,
                post_tool_payload,
                Some(&effective_invocation.name),
            )
            .await
        {
            HookChainResult::Allowed { payload, .. } => {
                if let Some(output) = payload.get("output").and_then(|v| v.as_str()) {
                    cmd_result.output = output.to_string();
                }
                if let Some(success) = payload.get("success").and_then(|v| v.as_bool()) {
                    cmd_result.success = success;
                }
            }
            HookChainResult::Blocked { hook_name, reason } => {
                warn!(
                    hook = %hook_name,
                    reason = %reason,
                    "PostToolUse hook returned Block (non-blocking event) — ignoring"
                );
            }
        }

        cmd_result
    }

    /// Execute a `/recall` invocation against pre-selected stores.
    ///
    /// Delegates to the `MemoryService`. The engine is responsible only for
    /// extracting the query argument and passing pre-routed store IDs.
    pub(super) async fn exec_recall_with_stores(
        &self,
        invocation: &CommandInvocation,
        user_message: &str,
        store_ids: Vec<String>,
    ) -> CommandResult {
        let mem = match &self.memory {
            Some(m) => m,
            None => {
                return CommandResult {
                    command_name: "recall".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("no memory stores configured".to_string()),
                };
            }
        };

        let query = invocation
            .arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        mem.recall(&store_ids, query, user_message).await
    }

    /// Execute a `/remember` invocation against pre-selected stores.
    ///
    /// Delegates to the `MemoryService` after extracting the `content` argument.
    pub(super) async fn exec_remember_with_stores(
        &self,
        invocation: &CommandInvocation,
        store_ids: Vec<String>,
    ) -> CommandResult {
        let mem = match &self.memory {
            Some(m) => m,
            None => {
                return CommandResult {
                    command_name: "remember".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("no memory stores configured".to_string()),
                };
            }
        };

        let content = match invocation.arguments.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return CommandResult {
                    command_name: "remember".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("missing required argument: content".to_string()),
                };
            }
        };

        mem.remember(&store_ids, content).await
    }

    /// Get the memory domain threshold.
    ///
    /// Uses the Memory domain-specific threshold if configured; otherwise falls back
    /// to the classifier threshold.
    pub(super) fn memory_domain_threshold(&self) -> f32 {
        self.config
            .router
            .domains
            .memory
            .as_ref()
            .and_then(|d| d.threshold)
            .unwrap_or(self.config.router.classifier.threshold)
    }
}
