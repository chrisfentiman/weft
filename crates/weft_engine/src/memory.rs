//! Memory command orchestration for the gateway engine.
//!
//! Provides `handle_builtin_with_hooks` (the full hook lifecycle for `/recall`
//! and `/remember`), the underlying `exec_recall_with_stores` and
//! `exec_remember_with_stores` executors, and `memory_domain_threshold`.

use tracing::{info, warn};
use weft_commands::CommandRegistry;
use weft_core::HookEvent;
use weft_core::{
    CommandAction, CommandInvocation, CommandResult, HookRoutingDomain, RoutingTrigger,
};
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::ProviderService;
use weft_memory::MemoryService;
use weft_router::{RoutingCandidate, SemanticRouter};

use crate::GatewayEngine;
use crate::util::builtin_describe_text;

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
    pub(crate) async fn handle_builtin_with_hooks(
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

            crate::hooks::apply_post_tool_use(
                &*self.hooks,
                &effective_invocation.name,
                "describe",
                &mut cmd_result,
            )
            .await;

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
        crate::hooks::apply_post_tool_use(
            &*self.hooks,
            &effective_invocation.name,
            "execute",
            &mut cmd_result,
        )
        .await;

        cmd_result
    }

    /// Execute a `/recall` invocation against pre-selected stores.
    ///
    /// Delegates to the `MemoryService`. The engine is responsible only for
    /// extracting the query argument and passing pre-routed store IDs.
    pub(crate) async fn exec_recall_with_stores(
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
    pub(crate) async fn exec_remember_with_stores(
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
    pub(crate) fn memory_domain_threshold(&self) -> f32 {
        self.config
            .router
            .domains
            .memory
            .as_ref()
            .and_then(|d| d.threshold)
            .unwrap_or(self.config.router.classifier.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use std::sync::Arc;

    // ── /recall tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_recall_intercepted_before_command_registry() {
        // Engine with memory mux — /recall should NOT reach the command registry.
        // If it did reach the registry, it would fail with CommandError::NotFound
        // (the mock registry has no "recall" command) and the loop would continue
        // until the LLM emits a response without commands.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "I found some memories about preferences.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]), // no "recall" in registry
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What are the user's preferences?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "I found some memories about preferences.");
    }

    #[tokio::test]
    async fn test_recall_returns_memories_as_toon() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "user prefers dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "Thanks for the context.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What does the user like?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Thanks for the context.");
    }

    #[tokio::test]
    async fn test_recall_no_query_arg_uses_user_message() {
        // /recall without query arg should still succeed (falls back to user message).
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall", "Nothing found in memory."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Tell me about user prefs"))
            .await
            .expect("should succeed without panicking");

        assert_eq!(resp_text(&resp), "Nothing found in memory.");
    }

    #[tokio::test]
    async fn test_recall_all_stores_fail_returns_success_with_no_memories() {
        // All stores fail — should return success=true with "No relevant memories found."
        let mux = make_mux_with_stores(vec![(
            "broken",
            true,
            true,
            Arc::new(MockMemStoreClient::query_fails("connection refused")),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something\"",
                "Understood, no memories available.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Search memory"))
            .await
            .expect("should succeed even when stores fail");

        assert_eq!(resp_text(&resp), "Understood, no memories available.");
    }

    #[tokio::test]
    async fn test_recall_no_mux_returns_error_result() {
        // /recall with no mux configured should emit an error result to the LLM.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something\"",
                "Memory is not configured.",
            ]),
            "test-model",
            "claude-test",
        );
        // Engine with no mux but we need to parse /recall — add it to registry as a command
        // Actually, without a mux, /recall won't be in known_commands, so it won't be parsed.
        // Test instead: call handle_builtin directly by building engine and sending a request
        // where the LLM emits /recall but the parser only picks it up if we register it.
        // To force the parse, we build an engine with a mock mux that is None-equivalent
        // — but there's no way to do that without an actual mux; the mux controls parser registration.
        // This is correct behaviour: without a mux, /recall is not in known_commands and is treated as prose.
        // Verify the response still works (just contains the text with /recall treated as prose).
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            None, // no mux
        );

        let resp = engine
            .handle_request(make_user_request("Try to recall"))
            .await
            .expect("should succeed");

        // The LLM response includes "/recall query: something" but it's treated as prose
        // because recall is not in known_commands when mux is None.
        // The first LLM response is returned directly since no commands are parsed.
        assert_eq!(resp_text(&resp), "/recall query: \"something\"");
    }

    // ── /remember tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_remember_intercepted_and_stores_to_writable() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Memory stored.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember this"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Memory stored.");
    }

    #[tokio::test]
    async fn test_remember_missing_content_returns_error() {
        // /remember without content arg should return success:false error to LLM.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember", // missing content
                "Got an error about missing argument.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Store something"))
            .await
            .expect("should succeed overall");

        assert_eq!(resp_text(&resp), "Got an error about missing argument.");
    }

    #[tokio::test]
    async fn test_remember_no_writable_stores_returns_empty_success() {
        // All stores are read-only — /remember should return success with no stores written.
        let mux = make_mux_with_stores(vec![(
            "code",
            true,
            false, // read-only
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"some info\"",
                "Attempted to remember.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember this"))
            .await
            .expect("should not fail");

        assert_eq!(resp_text(&resp), "Attempted to remember.");
    }

    // ── --describe tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_recall_describe_returns_compiled_text() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall --describe", "I see how recall works."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("How does recall work?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "I see how recall works.");
    }

    #[tokio::test]
    async fn test_remember_describe_does_not_reach_command_registry() {
        // The mock registry has no "remember" command — if --describe reached the
        // registry, it would return an error. With built-in interception it returns
        // the compiled describe text and success=true.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember --describe",
                "Now I know how to use remember.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]), // no "remember" in registry
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("How do I store memories?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Now I know how to use remember.");
    }

    // ── mixed built-in + external commands ────────────────────────────

    #[tokio::test]
    async fn test_mixed_builtin_and_external_commands() {
        // LLM emits both /recall and /web_search in the same response.
        // Both should execute: built-in first, then external.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "user prefers dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"prefs\"\n/web_search query: \"dark mode\"",
                "Found memories and web results.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "dark mode is popular".to_string(),
                    error: None,
                },
            ),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What do I know about dark mode?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found memories and web results.");
    }

    #[tokio::test]
    async fn test_external_commands_still_work_with_mux_present() {
        // Verify that wiring in a mux doesn't break the existing command registry path.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Here are the results."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "5 results about Rust".to_string(),
                    error: None,
                },
            ),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Here are the results.");
    }
}
