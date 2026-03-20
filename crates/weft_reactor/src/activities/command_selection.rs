//! CommandSelection activity: semantic command selection for the pre-loop.
//!
//! Replaces the command-routing half of `RouteActivity`. Selects relevant commands
//! via semantic scoring and emits `CommandsSelected`. Also determines whether
//! `/recall` and `/remember` are available based on memory configuration.
//!
//! **Fail mode: OPEN.** If command selection fails for any reason (router error,
//! hook block), the activity pushes `CommandsSelected { selected: vec![], ... }`
//! and completes normally. The model can still generate without commands.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use weft_core::{CommandStub, HookEvent};
use weft_hooks::HookChainResult;
use weft_router::{RoutingCandidate, RoutingDomainKind, filter_by_threshold, take_top};

use crate::activity::{Activity, ActivityInput, SemanticSelection};
use crate::event::PipelineEvent;
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Selects the commands relevant to the current turn via semantic routing.
///
/// If memory stores are configured, includes `/recall` and `/remember` as available
/// commands (the specific stores to query are determined per-invocation by those
/// commands' own routing logic).
///
/// **Name:** `"command_selection"`
///
/// **Events pushed:**
/// - `ActivityStarted { name: "command_selection" }`
/// - `HookEvaluated` / `HookBlocked` — for PreRoute / PostRoute hooks
/// - `CommandsSelected { selected, candidates_scored }` — always (fail-open)
/// - `ActivityCompleted { name: "command_selection", duration_ms, idempotency_key: None }`
pub struct CommandSelectionActivity;

impl CommandSelectionActivity {
    /// Construct a new `CommandSelectionActivity`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CommandSelectionActivity {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticSelection for CommandSelectionActivity {
    fn selection_domain(&self) -> &'static str {
        "commands"
    }
}

/// Stub stubs for /recall and /remember memory commands.
const RECALL_NAME: &str = "/recall";
const RECALL_DESC: &str = "Recall information from memory stores";
const REMEMBER_NAME: &str = "/remember";
const REMEMBER_DESC: &str = "Store information in memory stores";

#[async_trait::async_trait]
impl Activity for CommandSelectionActivity {
    fn name(&self) -> &str {
        "command_selection"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        services: &Services,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let start = Instant::now();

        let _ = event_tx
            .send(PipelineEvent::ActivityStarted {
                name: self.name().to_string(),
            })
            .await;

        if cancel.is_cancelled() {
            // Fail-open: push empty selection and complete rather than ActivityFailed.
            let _ = event_tx
                .send(PipelineEvent::CommandsSelected {
                    selected: vec![],
                    candidates_scored: 0,
                })
                .await;
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::ActivityCompleted {
                    name: self.name().to_string(),
                    duration_ms,
                    idempotency_key: None,
                })
                .await;
            return;
        }

        // Extract last user message text for semantic scoring.
        let user_message = extract_user_message(&input);

        // Build the full candidate list.
        // Start with ValidateActivity's available_commands, then add memory commands if configured.
        let mut full_candidates: Vec<CommandStub> = input.available_commands.clone();

        // Include /recall and /remember if memory has at least one store.
        let has_memory = services
            .config
            .memory
            .as_ref()
            .is_some_and(|m| !m.stores.is_empty());

        if has_memory {
            // Add memory commands if not already present.
            if !full_candidates.iter().any(|c| c.name == RECALL_NAME) {
                full_candidates.push(CommandStub {
                    name: RECALL_NAME.to_string(),
                    description: RECALL_DESC.to_string(),
                });
            }
            if !full_candidates.iter().any(|c| c.name == REMEMBER_NAME) {
                full_candidates.push(CommandStub {
                    name: REMEMBER_NAME.to_string(),
                    description: REMEMBER_DESC.to_string(),
                });
            }
        }

        // If no candidates, push empty selection and complete (no scoring needed).
        if full_candidates.is_empty() {
            let _ = event_tx
                .send(PipelineEvent::CommandsSelected {
                    selected: vec![],
                    candidates_scored: 0,
                })
                .await;
            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::ActivityCompleted {
                    name: self.name().to_string(),
                    duration_ms,
                    idempotency_key: None,
                })
                .await;
            return;
        }

        // Build routing candidates: each CommandStub becomes a RoutingCandidate.
        let routing_candidates: Vec<RoutingCandidate> = full_candidates
            .iter()
            .map(|stub| RoutingCandidate {
                id: stub.name.clone(),
                examples: vec![format!("{}: {}", stub.name, stub.description)],
            })
            .collect();

        let candidates_scored = routing_candidates.len();

        // Fire PreRoute hook for "commands" domain.
        let pre_payload = serde_json::json!({
            "domain": "commands",
            "user_message": user_message,
        });
        let hook_start = Instant::now();
        let pre_result = services
            .hooks
            .run_chain(HookEvent::PreRoute, pre_payload, Some("commands"))
            .await;
        let hook_duration = hook_start.elapsed().as_millis() as u64;

        match pre_result {
            HookChainResult::Blocked { reason, hook_name } => {
                // Fail-open: hook blocked means no commands, not an error.
                let _ = event_tx
                    .send(PipelineEvent::HookBlocked {
                        hook_event: "pre_route".to_string(),
                        hook_name,
                        reason,
                    })
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::CommandsSelected {
                        selected: vec![],
                        candidates_scored: 0,
                    })
                    .await;
                let duration_ms = start.elapsed().as_millis() as u64;
                let _ = event_tx
                    .send(PipelineEvent::ActivityCompleted {
                        name: self.name().to_string(),
                        duration_ms,
                        idempotency_key: None,
                    })
                    .await;
                return;
            }
            HookChainResult::Allowed { .. } => {
                let _ = event_tx
                    .send(PipelineEvent::HookEvaluated {
                        hook_event: "pre_route".to_string(),
                        hook_name: "pre_route".to_string(),
                        decision: "allow".to_string(),
                        duration_ms: hook_duration,
                    })
                    .await;
            }
        }

        // Score candidates via the semantic router.
        let domains = [(RoutingDomainKind::Commands, routing_candidates.clone())];
        let selected = match services.router.route(user_message, &domains).await {
            Err(e) => {
                // Fail-open: router error → include all commands capped by max_commands,
                // sorted alphabetically (matching route_domains fallback behaviour).
                warn!(error = %e, "command_selection: router error, including all commands");
                let max_commands = services.config.router.classifier.max_commands;
                let mut fallback = full_candidates.clone();
                fallback.sort_by(|a, b| a.name.cmp(&b.name));
                fallback.truncate(max_commands);
                fallback
            }
            Ok(decision) => {
                // Apply threshold and max_commands filtering.
                let threshold = services.config.router.classifier.threshold;
                let max_commands = services.config.router.classifier.max_commands;

                let filtered = filter_by_threshold(decision.commands, threshold);
                let top = take_top(filtered, max_commands);

                // Map scored IDs back to CommandStub from full_candidates.
                // Preserve descending-score order from take_top.
                top.iter()
                    .filter_map(|sc| full_candidates.iter().find(|c| c.name == sc.id).cloned())
                    .collect()
            }
        };

        // Fire PostRoute hook for "commands" domain.
        let post_payload = serde_json::json!({
            "domain": "commands",
            "selected_count": selected.len(),
        });
        let hook_start = Instant::now();
        let post_result = services
            .hooks
            .run_chain(HookEvent::PostRoute, post_payload, Some("commands"))
            .await;
        let hook_duration = hook_start.elapsed().as_millis() as u64;

        let final_selected = match post_result {
            HookChainResult::Blocked { reason, hook_name } => {
                // Fail-open: post_route blocking clears commands.
                let _ = event_tx
                    .send(PipelineEvent::HookBlocked {
                        hook_event: "post_route".to_string(),
                        hook_name,
                        reason,
                    })
                    .await;
                vec![]
            }
            HookChainResult::Allowed { .. } => {
                let _ = event_tx
                    .send(PipelineEvent::HookEvaluated {
                        hook_event: "post_route".to_string(),
                        hook_name: "post_route".to_string(),
                        decision: "allow".to_string(),
                        duration_ms: hook_duration,
                    })
                    .await;
                selected
            }
        };

        debug!(
            count = final_selected.len(),
            candidates_scored, "command_selection: commands selected"
        );

        let _ = event_tx
            .send(PipelineEvent::CommandsSelected {
                selected: final_selected,
                candidates_scored,
            })
            .await;

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = event_tx
            .send(PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms,
                idempotency_key: None,
            })
            .await;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the text of the last user-role message. Returns `""` if none found.
fn extract_user_message(input: &ActivityInput) -> &str {
    input
        .messages
        .iter()
        .rev()
        .find(|m| m.role == weft_core::Role::User)
        .and_then(|m| {
            m.content.iter().find_map(|p| {
                if let weft_core::ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
        })
        .unwrap_or("")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        NullEventLog, collect_events, make_test_input, make_test_services,
        make_test_services_with_blocking_hook,
    };
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_core::HookEvent;

    async fn run_command_selection(
        input: ActivityInput,
        services: crate::services::Services,
    ) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = CommandSelectionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name and domain ──────────────────────────────────────────────────────

    #[test]
    fn command_selection_name() {
        assert_eq!(CommandSelectionActivity::new().name(), "command_selection");
    }

    #[test]
    fn command_selection_domain() {
        assert_eq!(
            CommandSelectionActivity::new().selection_domain(),
            "commands"
        );
    }

    // ── Happy path: emits CommandsSelected ───────────────────────────────────

    #[tokio::test]
    async fn command_selection_emits_commands_selected() {
        let mut input = make_test_input();
        input.available_commands = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];

        let events = run_command_selection(input, make_test_services()).await;

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityStarted { name } if name == "command_selection")
            ),
            "expected ActivityStarted"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::CommandsSelected { .. })),
            "expected CommandsSelected event"
        );

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "command_selection")
            ),
            "expected ActivityCompleted"
        );

        // Fail-open: no ActivityFailed.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "did not expect ActivityFailed"
        );
    }

    // ── Empty available_commands produces empty selection ─────────────────────

    #[tokio::test]
    async fn command_selection_empty_commands_produces_empty_selection() {
        let mut input = make_test_input();
        input.available_commands = vec![]; // no commands from ValidateActivity

        let services = make_test_services();
        // Verify no memory configured in test services.
        assert!(services.config.memory.is_none());

        let events = run_command_selection(input, services).await;

        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::CommandsSelected {
                selected,
                candidates_scored,
            } = e
            {
                Some((selected.clone(), *candidates_scored))
            } else {
                None
            }
        });

        let (selected, candidates_scored) = selected.expect("CommandsSelected must be present");
        assert!(selected.is_empty(), "expected empty selection");
        assert_eq!(candidates_scored, 0, "expected zero candidates scored");
    }

    // ── Memory configured: includes /recall and /remember ─────────────────────

    #[tokio::test]
    async fn command_selection_includes_memory_commands_when_memory_configured() {
        let mut input = make_test_input();
        input.available_commands = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];

        // Build services with memory configured.
        let services = make_test_services_with_memory();

        let events = run_command_selection(input, services).await;

        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::CommandsSelected { selected, .. } = e {
                Some(selected.clone())
            } else {
                None
            }
        });

        let selected = selected.expect("CommandsSelected must be present");

        // /recall and /remember should be in the selected set (since stub router scores
        // all candidates above threshold).
        let has_recall = selected.iter().any(|c| c.name == RECALL_NAME);
        let has_remember = selected.iter().any(|c| c.name == REMEMBER_NAME);
        assert!(
            has_recall,
            "/recall must be in selected commands when memory is configured"
        );
        assert!(
            has_remember,
            "/remember must be in selected commands when memory is configured"
        );
    }

    // ── Memory not configured: no memory commands ─────────────────────────────

    #[tokio::test]
    async fn command_selection_no_memory_commands_without_memory_config() {
        let mut input = make_test_input();
        input.available_commands = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];

        let services = make_test_services(); // memory is None

        let events = run_command_selection(input, services).await;

        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::CommandsSelected { selected, .. } = e {
                Some(selected.clone())
            } else {
                None
            }
        });

        let selected = selected.expect("CommandsSelected must be present");
        assert!(
            !selected.iter().any(|c| c.name == RECALL_NAME),
            "/recall must not be included when memory is not configured"
        );
        assert!(
            !selected.iter().any(|c| c.name == REMEMBER_NAME),
            "/remember must not be included when memory is not configured"
        );
    }

    // ── PreRoute hook blocking: fail-open ─────────────────────────────────────

    #[tokio::test]
    async fn command_selection_pre_route_hook_blocking_is_fail_open() {
        let mut input = make_test_input();
        input.available_commands = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];

        let services =
            make_test_services_with_blocking_hook(HookEvent::PreRoute, "blocked commands");

        let events = run_command_selection(input, services).await;

        // Must NOT have ActivityFailed (fail-open).
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "hook blocking must be fail-open (no ActivityFailed)"
        );

        // Must have HookBlocked.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::HookBlocked { .. })),
            "expected HookBlocked event"
        );

        // Must have CommandsSelected with empty selection.
        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::CommandsSelected { selected, .. } = e {
                Some(selected.clone())
            } else {
                None
            }
        });
        let selected = selected.expect("CommandsSelected must be present even when hook blocks");
        assert!(
            selected.is_empty(),
            "commands must be empty when pre_route hook blocks"
        );

        // ActivityCompleted must be present (not ActivityFailed).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "expected ActivityCompleted (fail-open)"
        );
    }

    // ── PostRoute hook blocking: fail-open ────────────────────────────────────

    #[tokio::test]
    async fn command_selection_post_route_hook_blocking_clears_selection() {
        let mut input = make_test_input();
        input.available_commands = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];

        let services =
            make_test_services_with_blocking_hook(HookEvent::PostRoute, "post blocked commands");

        let events = run_command_selection(input, services).await;

        // Must NOT have ActivityFailed.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "post_route hook blocking must be fail-open"
        );

        // Must have CommandsSelected with empty selection (cleared by block).
        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::CommandsSelected { selected, .. } = e {
                Some(selected.clone())
            } else {
                None
            }
        });
        let selected = selected.expect("CommandsSelected must be present");
        assert!(
            selected.is_empty(),
            "commands must be empty when post_route hook blocks"
        );

        // ActivityCompleted must be present.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "expected ActivityCompleted (fail-open)"
        );
    }

    // ── Cancellation is fail-open ─────────────────────────────────────────────

    #[tokio::test]
    async fn command_selection_cancellation_is_fail_open() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancel

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        CommandSelectionActivity::new()
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // No ActivityFailed.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "cancellation must be fail-open (no ActivityFailed)"
        );

        // ActivityCompleted present.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "expected ActivityCompleted on cancellation (fail-open)"
        );
    }

    // ── Candidates_scored reflects pre-filtering count ─────────────────────────

    #[tokio::test]
    async fn command_selection_candidates_scored_is_pre_filter_count() {
        let mut input = make_test_input();
        input.available_commands = vec![
            CommandStub {
                name: "cmd_a".to_string(),
                description: "Command A".to_string(),
            },
            CommandStub {
                name: "cmd_b".to_string(),
                description: "Command B".to_string(),
            },
            CommandStub {
                name: "cmd_c".to_string(),
                description: "Command C".to_string(),
            },
        ];

        let events = run_command_selection(input, make_test_services()).await;

        let candidates_scored = events.iter().find_map(|e| {
            if let PipelineEvent::CommandsSelected {
                candidates_scored, ..
            } = e
            {
                Some(*candidates_scored)
            } else {
                None
            }
        });

        // 3 commands were passed, no memory configured → 3 candidates scored.
        assert_eq!(
            candidates_scored.expect("CommandsSelected must be present"),
            3
        );
    }

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Build test Services with a minimal memory configuration containing one store.
    fn make_test_services_with_memory() -> crate::services::Services {
        use std::sync::Arc;
        use weft_core::WeftConfig;

        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are helpful."

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "stub-model"
  model = "stub-model-v1"
  examples = ["example query"]

[[memory.stores]]
name = "conversations"
endpoint = "http://localhost:50052"
capabilities = ["read", "write"]
examples = ["conversation history"]
"#;
        let config: WeftConfig = toml::from_str(toml).expect("test config with memory must parse");

        let provider: Arc<dyn weft_llm::Provider> = Arc::new(StubProviderForTest);
        let hooks: Arc<dyn weft_hooks::HookRunner + Send + Sync> =
            Arc::new(weft_hooks::NullHookRunner);

        let providers: Arc<dyn weft_llm::ProviderService + Send + Sync> =
            Arc::new(StubProviderServiceForTest::new(provider));
        let router: Arc<dyn weft_router::SemanticRouter + Send + Sync> =
            Arc::new(StubRouterForTest);
        let commands: Arc<dyn weft_commands::CommandRegistry + Send + Sync> =
            Arc::new(StubCommandRegistryForTest);

        crate::services::Services {
            config: Arc::new(config),
            providers,
            router,
            commands,
            memory: None, // memory service not needed — config presence is what matters
            hooks,
            reactor_handle: std::sync::OnceLock::new(),
            request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
        }
    }

    // ── Local stubs for memory test ───────────────────────────────────────────
    // (Can't reuse test_support directly since its stub types are private.)

    pub struct StubProviderForTest;

    #[async_trait::async_trait]
    impl weft_llm::Provider for StubProviderForTest {
        async fn execute(
            &self,
            _request: weft_llm::ProviderRequest,
        ) -> Result<weft_llm::ProviderResponse, weft_llm::ProviderError> {
            let message = weft_core::WeftMessage {
                role: weft_core::Role::Assistant,
                source: weft_core::Source::Provider,
                model: Some("stub-model".to_string()),
                content: vec![weft_core::ContentPart::Text("stub".to_string())],
                delta: false,
                message_index: 0,
            };
            Ok(weft_llm::ProviderResponse::ChatCompletion {
                message,
                usage: None,
            })
        }

        fn name(&self) -> &str {
            "stub-provider"
        }
    }

    struct StubProviderServiceForTest {
        provider: std::sync::Arc<dyn weft_llm::Provider>,
        capabilities:
            std::collections::HashMap<String, std::collections::HashSet<weft_llm::Capability>>,
        capability_index:
            std::collections::HashMap<weft_llm::Capability, std::collections::HashSet<String>>,
        empty_string_set: std::collections::HashSet<String>,
    }

    impl StubProviderServiceForTest {
        fn new(provider: std::sync::Arc<dyn weft_llm::Provider>) -> Self {
            use std::collections::{HashMap, HashSet};
            let default = "stub-model".to_string();
            let chat_cap = weft_llm::Capability::new(weft_llm::Capability::CHAT_COMPLETIONS);

            let mut capabilities = HashMap::new();
            let mut cap_set = HashSet::new();
            cap_set.insert(chat_cap.clone());
            capabilities.insert(default.clone(), cap_set);

            let mut capability_index: HashMap<weft_llm::Capability, HashSet<String>> =
                HashMap::new();
            let mut model_set = HashSet::new();
            model_set.insert(default.clone());
            capability_index.insert(chat_cap, model_set);

            Self {
                provider,
                capabilities,
                capability_index,
                empty_string_set: HashSet::new(),
            }
        }
    }

    impl weft_llm::ProviderService for StubProviderServiceForTest {
        fn get(&self, _name: &str) -> &std::sync::Arc<dyn weft_llm::Provider> {
            &self.provider
        }
        fn model_id(&self, name: &str) -> Option<&str> {
            if name == "stub-model" {
                Some("stub-model-v1")
            } else {
                None
            }
        }
        fn max_tokens_for(&self, name: &str) -> Option<u32> {
            if name == "stub-model" {
                Some(4096)
            } else {
                None
            }
        }
        fn default_provider(&self) -> &std::sync::Arc<dyn weft_llm::Provider> {
            &self.provider
        }
        fn default_name(&self) -> &str {
            "stub-model"
        }
        fn models_with_capability(
            &self,
            capability: &weft_llm::Capability,
        ) -> &std::collections::HashSet<String> {
            self.capability_index
                .get(capability)
                .unwrap_or(&self.empty_string_set)
        }
        fn model_has_capability(
            &self,
            model_name: &str,
            capability: &weft_llm::Capability,
        ) -> bool {
            self.capabilities
                .get(model_name)
                .map(|caps| caps.contains(capability))
                .unwrap_or(false)
        }
        fn model_capabilities(
            &self,
            model_name: &str,
        ) -> Option<&std::collections::HashSet<weft_llm::Capability>> {
            self.capabilities.get(model_name)
        }
    }

    struct StubRouterForTest;

    #[async_trait::async_trait]
    impl weft_router::SemanticRouter for StubRouterForTest {
        async fn route(
            &self,
            _user_message: &str,
            domains: &[(
                weft_router::RoutingDomainKind,
                Vec<weft_router::RoutingCandidate>,
            )],
        ) -> Result<weft_router::RoutingDecision, weft_router::RouterError> {
            let mut decision = weft_router::RoutingDecision::empty();
            for (kind, candidates) in domains {
                match kind {
                    weft_router::RoutingDomainKind::Model => {
                        if let Some(first) = candidates.first() {
                            decision.model = Some(weft_router::ScoredCandidate {
                                id: first.id.clone(),
                                score: 0.9,
                            });
                        }
                    }
                    weft_router::RoutingDomainKind::Commands => {
                        decision.commands = candidates
                            .iter()
                            .map(|c| weft_router::ScoredCandidate {
                                id: c.id.clone(),
                                score: 0.9,
                            })
                            .collect();
                    }
                    _ => {}
                }
            }
            Ok(decision)
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            candidates: &[weft_router::RoutingCandidate],
        ) -> Result<Vec<weft_router::ScoredCandidate>, weft_router::RouterError> {
            Ok(candidates
                .iter()
                .map(|c| weft_router::ScoredCandidate {
                    id: c.id.clone(),
                    score: 0.9,
                })
                .collect())
        }
    }

    struct StubCommandRegistryForTest;

    #[async_trait::async_trait]
    impl weft_commands::CommandRegistry for StubCommandRegistryForTest {
        async fn list_commands(
            &self,
        ) -> Result<Vec<weft_core::CommandStub>, weft_commands::CommandError> {
            Ok(vec![])
        }
        async fn describe_command(
            &self,
            name: &str,
        ) -> Result<weft_core::CommandDescription, weft_commands::CommandError> {
            Ok(weft_core::CommandDescription {
                name: name.to_string(),
                description: format!("{name}: stub"),
                usage: format!("/{name}"),
                parameters_schema: None,
            })
        }
        async fn execute_command(
            &self,
            invocation: &weft_core::CommandInvocation,
        ) -> Result<weft_core::CommandResult, weft_commands::CommandError> {
            Ok(weft_core::CommandResult {
                command_name: invocation.name.clone(),
                success: true,
                output: "stub".to_string(),
                error: None,
            })
        }
    }
}
