//! Higher-level routing service: `route_domains` and associated types.
//!
//! This module provides the pure routing logic extracted from the engine's
//! `route_all_domains_with_hooks`. The engine calls this between PreRoute and
//! PostRoute hooks; hook firing itself stays in the engine.
//!
//! ## Design
//!
//! - `route_domains` is a free function generic over `R: SemanticRouter`.
//! - `RoutingInput` borrows from the engine's data for a single routing call.
//! - `RoutingResult` is owned; the engine consumes it and applies PostRoute overrides.
//! - `weft_router` does NOT depend on `weft_memory` or `weft_hooks` — pure domain crate.
//! - `MemoryStoreRef` provides store metadata without a `weft_memory` import.

use weft_core::{CommandStub, RoutingActivity};

use crate::{
    RouterError, RoutingCandidate, RoutingDomainKind, ScoredCandidate, filter_by_threshold,
    take_top,
};

// ── Public types ─────────────────────────────────────────────────────────────

/// Complete routing result for a single request.
///
/// Named fields replace the 5-tuple return from the former
/// `route_all_domains_with_hooks`.
///
/// Hook context is NOT stored here — the engine accumulates that separately
/// across PreRoute/PostRoute hook calls before and after `route_domains`.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Commands selected by semantic routing (filtered, sorted by score).
    pub selected_commands: Vec<CommandStub>,
    /// Whether to inject tool definitions into the system prompt.
    pub inject_tools: bool,
    /// The model routing name to use for LLM calls.
    pub selected_model: String,
    /// Activity events for diagnostics / `options.activity` responses.
    pub activity_events: Vec<RoutingActivity>,
}

/// The input context needed for a single routing call.
///
/// Passed by the engine after applying PreRoute hook modifications.
/// Pure data — no hook references, no provider references.
///
/// The engine constructs model candidates (filtered by capability) BEFORE
/// building this struct, keeping `weft_router` independent of `weft_providers`.
#[derive(Debug)]
pub struct RoutingInput<'a> {
    /// The user message text (possibly modified by PreRoute hooks).
    pub user_message: &'a str,
    /// All available commands from the registry.
    pub all_commands: &'a [CommandStub],
    /// Minimum router score for a command to be selected.
    pub threshold: f32,
    /// Maximum number of commands to include after filtering.
    pub max_commands: usize,
    /// When true and tools_needed=false, omit tool definitions from the prompt.
    pub skip_tools_when_unnecessary: bool,
    /// Default model name (from `ProviderService::default_name()`).
    pub default_model: &'a str,
    /// Pre-built domain candidates, potentially modified by PreRoute hooks.
    ///
    /// The engine constructs these before calling `route_domains`; PreRoute
    /// hooks may have modified the candidates for each domain.
    pub domains: Vec<(RoutingDomainKind, Vec<RoutingCandidate>)>,
}

/// Grouped memory routing candidates by capability.
///
/// Returned by `build_memory_candidates` to avoid a 3-tuple return.
#[derive(Debug, Clone, Default)]
pub struct MemoryCandidates {
    /// All memory store candidates regardless of capability.
    pub all: Vec<RoutingCandidate>,
    /// Candidates filtered to read-capable stores only (used by `/recall`).
    pub read: Vec<RoutingCandidate>,
    /// Candidates filtered to write-capable stores only (used by `/remember`).
    pub write: Vec<RoutingCandidate>,
}

/// Minimal store metadata needed to build routing candidates.
///
/// Defined here so `weft_router` does not depend on `weft_memory`.
/// The engine converts `weft_memory::StoreInfo` to this type before
/// calling `build_memory_candidates`.
#[derive(Debug, Clone)]
pub struct MemoryStoreRef {
    /// Unique store name.
    pub name: String,
    /// Capability labels, e.g. `["read", "write"]`.
    pub capabilities: Vec<String>,
    /// Example phrases used by the semantic router for scoring.
    pub examples: Vec<String>,
}

// ── Candidate builders ───────────────────────────────────────────────────────

/// Build memory store routing candidates grouped by capability.
///
/// Accepts `&[MemoryStoreRef]` rather than `StoreInfo` so `weft_router`
/// does not depend on `weft_memory`. The engine converts `StoreInfo` to
/// `MemoryStoreRef` at the call site.
pub fn build_memory_candidates(stores: &[MemoryStoreRef]) -> MemoryCandidates {
    let mut result = MemoryCandidates::default();

    for store in stores {
        let candidate = RoutingCandidate {
            id: store.name.clone(),
            examples: store.examples.clone(),
        };
        if store.capabilities.iter().any(|c| c == "read") {
            result.read.push(candidate.clone());
        }
        if store.capabilities.iter().any(|c| c == "write") {
            result.write.push(candidate.clone());
        }
        result.all.push(candidate);
    }

    result
}

/// Static ToolNecessity domain candidates.
///
/// These examples are tunable defaults hardcoded in one place to avoid
/// duplicating them at multiple call sites.
pub fn tool_necessity_candidates() -> Vec<RoutingCandidate> {
    vec![
        RoutingCandidate {
            id: "needs_tools".to_string(),
            examples: vec![
                "Search the web for the latest news about Rust".to_string(),
                "Look up the current stock price of Apple".to_string(),
                "Run this code and show me the output".to_string(),
                "Find documents about our Q3 strategy".to_string(),
                "Execute a database query for user counts".to_string(),
            ],
        },
        RoutingCandidate {
            id: "no_tools".to_string(),
            examples: vec![
                "What is the capital of France?".to_string(),
                "Explain how async/await works in Rust".to_string(),
                "Write a poem about the ocean".to_string(),
                "What do you think about functional programming?".to_string(),
                "Hello, how are you today?".to_string(),
            ],
        },
    ]
}

// ── Pure routing function ────────────────────────────────────────────────────

/// Execute pure routing logic: score domains, filter, select.
///
/// This is the logic extracted from `route_all_domains_with_hooks`,
/// minus all hook firing. The engine calls this between PreRoute and
/// PostRoute hooks.
///
/// On router failure, falls back to conservative defaults:
/// - `selected_commands`: all commands capped by `max_commands` (sorted by name)
/// - `inject_tools`: `true`
/// - `selected_model`: `input.default_model`
/// - `activity_events`: single `RoutingActivity` with score `0.0` (signals fallback)
///
/// The fallback does NOT bypass PostRoute hook processing in the engine.
///
/// Generic over `R: SemanticRouter` for zero-cost dispatch. Tests provide
/// concrete mock routers without needing `dyn`.
pub async fn route_domains<R: crate::SemanticRouter>(
    router: &R,
    input: &RoutingInput<'_>,
) -> Result<RoutingResult, RouterError> {
    let decision = router.route(input.user_message, &input.domains).await;

    match decision {
        Err(e) => {
            // Router failure: conservative fallback.
            let mut fallback: Vec<CommandStub> = input
                .all_commands
                .iter()
                .take(input.max_commands)
                .cloned()
                .collect();
            fallback.sort_by(|a, b| a.name.cmp(&b.name));

            let fallback_activity = RoutingActivity {
                model: input.default_model.to_string(),
                // Score 0.0 signals fallback to the engine/activity layer.
                score: 0.0,
                filters: vec![format!("router_failure: {e}")],
            };

            Ok(RoutingResult {
                selected_commands: fallback,
                inject_tools: true,
                selected_model: input.default_model.to_string(),
                activity_events: vec![fallback_activity],
            })
        }

        Ok(routing_decision) => {
            // ── Commands ─────────────────────────────────────────────────
            let filtered = filter_by_threshold(routing_decision.commands.clone(), input.threshold);
            let top: Vec<ScoredCandidate> = take_top(filtered, input.max_commands);

            // Build command list from scored IDs, preserving score order.
            let score_map: std::collections::HashMap<&str, f32> =
                top.iter().map(|r| (r.id.as_str(), r.score)).collect();
            let top_ids: std::collections::HashSet<&str> =
                top.iter().map(|r| r.id.as_str()).collect();

            let mut selected_commands: Vec<CommandStub> = input
                .all_commands
                .iter()
                .filter(|cmd| top_ids.contains(cmd.name.as_str()))
                .cloned()
                .collect();

            // Preserve descending-score ordering.
            selected_commands.sort_by(|a, b| {
                let sa = score_map.get(a.name.as_str()).copied().unwrap_or(0.0);
                let sb = score_map.get(b.name.as_str()).copied().unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });

            // ── Tool necessity ────────────────────────────────────────────
            let inject_tools = !matches!(
                routing_decision.tools_needed,
                Some(false) if input.skip_tools_when_unnecessary
            );

            // ── Model selection ───────────────────────────────────────────
            let selected_model = routing_decision
                .model
                .as_ref()
                .map(|m| m.id.clone())
                .unwrap_or_else(|| input.default_model.to_string());

            // ── Activity event ────────────────────────────────────────────
            let model_score = routing_decision
                .model
                .as_ref()
                .map(|m| m.score)
                .unwrap_or(1.0_f32);

            let routing_activity = RoutingActivity {
                model: selected_model.clone(),
                score: model_score,
                filters: vec![],
            };

            Ok(RoutingResult {
                selected_commands,
                inject_tools,
                selected_model,
                activity_events: vec![routing_activity],
            })
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
        SemanticRouter,
    };
    use async_trait::async_trait;
    use weft_core::CommandStub;

    // ── Mock router ───────────────────────────────────────────────────────

    struct MockRouter {
        command_score: f32,
        model_decision: Option<String>,
        tools_needed: Option<bool>,
        fail: bool,
    }

    impl MockRouter {
        fn with_score(score: f32) -> Self {
            Self {
                command_score: score,
                model_decision: None,
                tools_needed: None,
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                command_score: 0.0,
                model_decision: None,
                tools_needed: None,
                fail: true,
            }
        }

        fn with_model(model: &str) -> Self {
            Self {
                command_score: 1.0,
                model_decision: Some(model.to_string()),
                tools_needed: None,
                fail: false,
            }
        }

        fn with_tools_needed(needed: bool) -> Self {
            Self {
                command_score: 1.0,
                model_decision: None,
                tools_needed: Some(needed),
                fail: false,
            }
        }
    }

    #[async_trait]
    impl SemanticRouter for MockRouter {
        async fn route(
            &self,
            _user_message: &str,
            domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            if self.fail {
                return Err(RouterError::ModelNotLoaded);
            }

            let mut decision = RoutingDecision::empty();
            for (kind, candidates) in domains {
                match kind {
                    RoutingDomainKind::Commands => {
                        decision.commands = candidates
                            .iter()
                            .map(|c| ScoredCandidate {
                                id: c.id.clone(),
                                score: self.command_score,
                            })
                            .collect();
                    }
                    RoutingDomainKind::Model => {
                        if let Some(ref model_id) = self.model_decision {
                            decision.model =
                                candidates.iter().find(|c| c.id == *model_id).map(|c| {
                                    ScoredCandidate {
                                        id: c.id.clone(),
                                        score: 0.9,
                                    }
                                });
                        }
                    }
                    RoutingDomainKind::ToolNecessity => {
                        decision.tools_needed = self.tools_needed;
                    }
                    RoutingDomainKind::Memory => {}
                }
            }
            Ok(decision)
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            candidates: &[RoutingCandidate],
        ) -> Result<Vec<ScoredCandidate>, RouterError> {
            Ok(candidates
                .iter()
                .map(|c| ScoredCandidate {
                    id: c.id.clone(),
                    score: 0.9,
                })
                .collect())
        }
    }

    fn make_commands(names: &[&str]) -> Vec<CommandStub> {
        names
            .iter()
            .map(|n| CommandStub {
                name: n.to_string(),
                description: format!("{n}: does stuff"),
            })
            .collect()
    }

    fn make_command_candidates(names: &[&str]) -> Vec<RoutingCandidate> {
        names
            .iter()
            .map(|n| RoutingCandidate {
                id: n.to_string(),
                examples: vec![format!("{n}: does stuff")],
            })
            .collect()
    }

    // ── route_domains tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_route_domains_selects_commands_above_threshold() {
        let router = MockRouter::with_score(0.8);
        let commands = make_commands(&["web_search", "code_review"]);
        let input = RoutingInput {
            user_message: "search for rust news",
            all_commands: &commands,
            threshold: 0.5,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["web_search", "code_review"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert_eq!(result.selected_commands.len(), 2);
        assert!(result.inject_tools);
        assert_eq!(result.selected_model, "default");
        assert!(!result.activity_events.is_empty());
    }

    #[tokio::test]
    async fn test_route_domains_filters_below_threshold() {
        let router = MockRouter::with_score(0.2);
        let commands = make_commands(&["web_search"]);
        let input = RoutingInput {
            user_message: "hello",
            all_commands: &commands,
            threshold: 0.5,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["web_search"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert!(result.selected_commands.is_empty());
        assert!(result.inject_tools);
    }

    #[tokio::test]
    async fn test_route_domains_respects_max_commands() {
        let router = MockRouter::with_score(0.9);
        let commands = make_commands(&["a", "b", "c", "d", "e"]);
        let input = RoutingInput {
            user_message: "do stuff",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 2,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["a", "b", "c", "d", "e"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert_eq!(result.selected_commands.len(), 2);
    }

    #[tokio::test]
    async fn test_route_domains_selects_model() {
        let router = MockRouter::with_model("complex");
        let commands = make_commands(&[]);
        let model_candidates = vec![
            RoutingCandidate {
                id: "complex".to_string(),
                examples: vec!["complex reasoning".to_string()],
            },
            RoutingCandidate {
                id: "fast".to_string(),
                examples: vec!["quick answer".to_string()],
            },
        ];
        let input = RoutingInput {
            user_message: "solve this hard problem",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "fast",
            domains: vec![(RoutingDomainKind::Model, model_candidates)],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert_eq!(result.selected_model, "complex");
    }

    #[tokio::test]
    async fn test_route_domains_uses_default_model_when_no_model_domain() {
        let router = MockRouter::with_score(0.9);
        let commands = make_commands(&["web_search"]);
        let input = RoutingInput {
            user_message: "search",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default-model",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["web_search"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert_eq!(result.selected_model, "default-model");
    }

    #[tokio::test]
    async fn test_route_domains_tool_necessity_suppresses_tools() {
        let router = MockRouter::with_tools_needed(false);
        let commands = make_commands(&[]);
        let input = RoutingInput {
            user_message: "what is the capital of France?",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 10,
            skip_tools_when_unnecessary: true,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::ToolNecessity,
                tool_necessity_candidates(),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert!(!result.inject_tools);
    }

    #[tokio::test]
    async fn test_route_domains_tool_necessity_conservative_when_skip_disabled() {
        // When skip_tools_when_unnecessary=false, tools always injected even if tools_needed=false.
        let router = MockRouter::with_tools_needed(false);
        let commands = make_commands(&[]);
        let input = RoutingInput {
            user_message: "what is two plus two",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::ToolNecessity,
                tool_necessity_candidates(),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert!(result.inject_tools);
    }

    #[tokio::test]
    async fn test_route_domains_router_failure_fallback_all_commands() {
        let router = MockRouter::failing();
        let commands = make_commands(&["a", "b", "c", "d", "e"]);
        let input = RoutingInput {
            user_message: "hello",
            all_commands: &commands,
            threshold: 0.5,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default-model",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["a", "b", "c", "d", "e"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        // Fallback: all commands, conservative inject_tools, default model
        assert_eq!(result.selected_commands.len(), 5);
        assert!(result.inject_tools);
        assert_eq!(result.selected_model, "default-model");
        // Score 0.0 in activity signals fallback.
        assert_eq!(result.activity_events[0].score, 0.0);
    }

    #[tokio::test]
    async fn test_route_domains_router_failure_fallback_respects_max_commands() {
        let router = MockRouter::failing();
        let commands = make_commands(&["a", "b", "c", "d", "e"]);
        let input = RoutingInput {
            user_message: "hello",
            all_commands: &commands,
            threshold: 0.5,
            max_commands: 3,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["a", "b", "c", "d", "e"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert_eq!(result.selected_commands.len(), 3);
    }

    #[tokio::test]
    async fn test_route_domains_fallback_commands_sorted_alphabetically() {
        let router = MockRouter::failing();
        let commands = make_commands(&["zebra", "alpha", "middle"]);
        let input = RoutingInput {
            user_message: "hello",
            all_commands: &commands,
            threshold: 0.5,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![(
                RoutingDomainKind::Commands,
                make_command_candidates(&["zebra", "alpha", "middle"]),
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        let names: Vec<&str> = result
            .selected_commands
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(names, ["alpha", "middle", "zebra"]);
    }

    #[tokio::test]
    async fn test_route_domains_activity_event_has_model() {
        let router = MockRouter::with_model("complex");
        let commands = make_commands(&[]);
        let input = RoutingInput {
            user_message: "complex task",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "fast",
            domains: vec![(
                RoutingDomainKind::Model,
                vec![
                    RoutingCandidate {
                        id: "complex".to_string(),
                        examples: vec!["complex reasoning".to_string()],
                    },
                    RoutingCandidate {
                        id: "fast".to_string(),
                        examples: vec!["quick".to_string()],
                    },
                ],
            )],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert_eq!(result.activity_events.len(), 1);
        assert_eq!(result.activity_events[0].model, "complex");
        assert!(result.activity_events[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_route_domains_empty_commands_empty_domains() {
        let router = MockRouter::with_score(0.9);
        let commands: Vec<CommandStub> = vec![];
        let input = RoutingInput {
            user_message: "hello",
            all_commands: &commands,
            threshold: 0.3,
            max_commands: 10,
            skip_tools_when_unnecessary: false,
            default_model: "default",
            domains: vec![],
        };

        let result = route_domains(&router, &input).await.unwrap();
        assert!(result.selected_commands.is_empty());
        assert!(result.inject_tools);
    }

    // ── build_memory_candidates tests ─────────────────────────────────────

    #[test]
    fn test_build_memory_candidates_empty() {
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
        assert_eq!(result.write.len(), 1);
    }

    #[test]
    fn test_build_memory_candidates_read_only() {
        let stores = vec![MemoryStoreRef {
            name: "kb".to_string(),
            capabilities: vec!["read".to_string()],
            examples: vec!["knowledge".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert_eq!(result.read.len(), 1);
        assert!(result.write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_write_only() {
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
        assert_eq!(result.read.len(), 2);
        assert_eq!(result.write.len(), 2);
        assert!(result.read.iter().any(|c| c.id == "conv"));
        assert!(result.read.iter().any(|c| c.id == "kb"));
        assert!(result.write.iter().any(|c| c.id == "conv"));
        assert!(result.write.iter().any(|c| c.id == "audit"));
    }

    // ── tool_necessity_candidates tests ───────────────────────────────────

    #[test]
    fn test_tool_necessity_candidates_has_needs_and_no_tools() {
        let candidates = tool_necessity_candidates();
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|c| c.id == "needs_tools"));
        assert!(candidates.iter().any(|c| c.id == "no_tools"));
    }

    #[test]
    fn test_tool_necessity_candidates_have_examples() {
        for candidate in tool_necessity_candidates() {
            assert!(
                !candidate.examples.is_empty(),
                "candidate '{}' must have examples",
                candidate.id
            );
        }
    }
}
