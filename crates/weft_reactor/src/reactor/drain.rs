//! drain_pre_post_loop: drain events after a synchronous activity completes.
//!
//! Used for pre-loop, post-loop, and hook activities that run synchronously
//! (not spawned). Drains all pending events from the channel and dispatches
//! state updates until the channel is empty.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::ReactorError;
use crate::event::{
    ActivityEvent, CommandEvent, ContextEvent, FailureDetail, HookOutcome, MessageInjectionSource,
    PipelineEvent, SelectionEvent, SignalEvent,
};
use crate::execution::ExecutionId;
use crate::signal::Signal;

use super::Reactor;
use super::types::ExecutionState;

/// Outcome from draining events after an activity completes.
///
/// Separates "what happened" from "what to do about it." The drain function
/// reports; the caller decides consequences based on activity criticality.
#[derive(Debug)]
pub(super) enum DrainOutcome {
    /// All events processed, execution continues.
    Continue,
    /// Activity failed. Caller decides fatal vs. degraded based on criticality.
    ActivityFailed {
        name: String,
        error: String,
        /// Whether the activity indicated the failure is retryable.
        /// Preserved for callers that implement retry logic.
        #[allow(dead_code)]
        retryable: bool,
        detail: FailureDetail,
    },
    /// Hook blocked the request. Always fatal.
    HookBlocked { hook_name: String, reason: String },
    /// Cancellation requested. Always fatal.
    Cancelled { reason: String },
}

impl Reactor {
    /// Drain the channel after a synchronous pre/post-loop activity completes.
    ///
    /// Returns `DrainOutcome` describing what the drain encountered.
    /// Returns `Err` only for infrastructure failures (channel closed, event log error).
    pub(super) async fn drain_pre_post_loop(
        &self,
        execution_id: &ExecutionId,
        event_rx: &mut mpsc::Receiver<PipelineEvent>,
        state: &mut ExecutionState,
        cancel: &CancellationToken,
    ) -> Result<DrainOutcome, ReactorError> {
        loop {
            match event_rx.try_recv() {
                Ok(event) => {
                    self.record_event(execution_id, &event, None).await?;
                    match &event {
                        PipelineEvent::Activity(ActivityEvent::Failed {
                            name,
                            error,
                            retryable,
                            detail,
                        }) => {
                            return Ok(DrainOutcome::ActivityFailed {
                                name: name.clone(),
                                error: error.clone(),
                                retryable: *retryable,
                                detail: detail.clone(),
                            });
                        }
                        PipelineEvent::Hook(HookOutcome::Blocked {
                            hook_name, reason, ..
                        }) => {
                            return Ok(DrainOutcome::HookBlocked {
                                hook_name: hook_name.clone(),
                                reason: reason.clone(),
                            });
                        }
                        PipelineEvent::Execution(
                            crate::event::ExecutionEvent::ValidationPassed,
                        ) => {
                            // Commands are now communicated via Command(Available) events.
                        }
                        PipelineEvent::Command(CommandEvent::Available { commands }) => {
                            // Populate available_commands from the validate activity's event.
                            // This makes available_commands reconstructable from the event log.
                            state.available_commands = commands.clone();
                        }
                        PipelineEvent::Context(ContextEvent::MessageInjected {
                            message,
                            source,
                        }) => {
                            // Dispatch on source to determine insertion position.
                            // SystemPromptAssembly: insert/replace at messages[0] so the
                            //   provider sees the gateway-assembled prompt as the canonical
                            //   system prompt.
                            // CommandFormatInjection: insert after the system prompt (position
                            //   1 if a system message exists, else 0) so command descriptions
                            //   appear before user messages.
                            // All other sources: append (existing behavior).
                            match source {
                                MessageInjectionSource::SystemPromptAssembly => {
                                    if !state.messages.is_empty()
                                        && state.messages[0].role == weft_core::Role::System
                                    {
                                        state.messages[0] = message.clone();
                                    } else {
                                        state.messages.insert(0, message.clone());
                                    }
                                }
                                MessageInjectionSource::CommandFormatInjection => {
                                    // Insert after the system prompt, before user messages.
                                    let insert_pos = if !state.messages.is_empty()
                                        && state.messages[0].role == weft_core::Role::System
                                    {
                                        1
                                    } else {
                                        0
                                    };
                                    state.messages.insert(insert_pos, message.clone());
                                }
                                _ => {
                                    // Existing behavior: append.
                                    state.messages.push(message.clone());
                                }
                            }
                        }
                        PipelineEvent::Context(ContextEvent::ResponseAssembled { response }) => {
                            state.response = Some(response.clone());
                        }
                        PipelineEvent::Signal(SignalEvent::Received(Signal::Cancel { reason })) => {
                            cancel.cancel();
                            return Ok(DrainOutcome::Cancelled {
                                reason: reason.clone(),
                            });
                        }
                        // ── New pre-loop activity events (Phase 1+) ──────────────
                        PipelineEvent::Selection(SelectionEvent::ModelSelected {
                            model_name,
                            score,
                            ..
                        }) => {
                            state.selected_model = Some(model_name.clone());
                            // Maintain backward compat: populate routing snapshot so
                            // GenerateActivity continues to receive routing_result unchanged.
                            state.routing = Some(crate::activity::RoutingSnapshot {
                                model_routing: weft_core::RoutingActivity {
                                    model: model_name.clone(),
                                    score: *score,
                                    filters: vec![],
                                },
                                tool_necessity: None,
                                tool_necessity_score: None,
                            });
                        }
                        PipelineEvent::Selection(SelectionEvent::CommandsSelected {
                            selected,
                            ..
                        }) => {
                            state.selected_commands = selected.clone();
                        }
                        PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                            model_id,
                            provider_name,
                            capabilities,
                            max_tokens,
                            ..
                        }) => {
                            state.selected_model_id = Some(model_id.clone());
                            state.selected_provider = Some(provider_name.clone());
                            state.model_capabilities = capabilities.clone();
                            state.model_max_tokens = Some(*max_tokens);
                        }
                        PipelineEvent::Context(ContextEvent::SystemPromptAssembled { .. }) => {
                            // System prompt insertion is handled by the MessageInjected arm
                            // when source == SystemPromptAssembly. Nothing to do here.
                        }
                        PipelineEvent::Context(ContextEvent::CommandsFormatted {
                            format, ..
                        }) => {
                            state.command_format = Some(format.clone());
                        }
                        PipelineEvent::Context(ContextEvent::SamplingUpdated {
                            max_tokens,
                            temperature,
                            top_p,
                        }) => {
                            state.sampling_max_tokens = Some(*max_tokens);
                            state.sampling_temperature = *temperature;
                            state.sampling_top_p = *top_p;
                        }
                        _ => {}
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    return Err(ReactorError::ChannelClosed);
                }
            }
        }
        Ok(DrainOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::sync::Arc;

    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    use crate::budget::Budget;
    use crate::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
    use crate::event::{
        ActivityEvent, CommandEvent, ContextEvent, HookOutcome, MessageInjectionSource,
        PipelineEvent, SelectionEvent, SignalEvent,
    };
    use crate::event_log::EventLog;
    use crate::execution::{Execution, ExecutionId, ExecutionStatus};
    use crate::registry::ActivityRegistry;
    use crate::signal::Signal;
    use crate::test_support::{TestEventLog, make_test_services};
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    use super::super::Reactor;
    use super::super::types::ExecutionState;

    // ── Minimal stub activities ────────────────────────────────────────────

    struct NoOpActivity(String);

    #[async_trait::async_trait]
    impl crate::activity::Activity for NoOpActivity {
        fn name(&self) -> &str {
            &self.0
        }

        async fn execute(
            &self,
            _: &ExecutionId,
            _: crate::activity::ActivityInput,
            _: &dyn weft_reactor_trait::ServiceLocator,
            _: &dyn crate::event_log::EventLog,
            _: mpsc::Sender<PipelineEvent>,
            _: CancellationToken,
        ) {
        }
    }

    // ── Test fixture ──────────────────────────────────────────────────────

    fn build_reactor(event_log: Arc<TestEventLog>) -> Reactor {
        let services = Arc::new(make_test_services());
        let mut registry = ActivityRegistry::new();
        registry
            .register(Arc::new(NoOpActivity("generate".to_string())))
            .unwrap();
        registry
            .register(Arc::new(NoOpActivity("assemble_response".to_string())))
            .unwrap();
        registry
            .register(Arc::new(NoOpActivity("execute_command".to_string())))
            .unwrap();
        let config = ReactorConfig {
            pipelines: vec![PipelineConfig {
                name: "default".to_string(),
                pre_loop: vec![],
                post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
                generate: ActivityRef::Name("generate".to_string()),
                execute_command: ActivityRef::Name("execute_command".to_string()),
                loop_hooks: LoopHooks::default(),
            }],
            budget: BudgetConfig {
                max_generation_calls: 5,
                max_iterations: 5,
                max_depth: 3,
                timeout_secs: 300,
                generation_timeout_secs: 60,
                command_timeout_secs: 10,
            },
        };
        Reactor::new(services, event_log, Arc::new(registry), &config)
            .expect("reactor build should succeed in test")
    }

    fn test_state() -> ExecutionState {
        let services = make_test_services();
        ExecutionState::new(
            Budget::new(5, 5, 3, chrono::Utc::now() + chrono::Duration::hours(1)),
            services.resolved_config,
        )
    }

    async fn setup_execution(event_log: &Arc<TestEventLog>, execution_id: &ExecutionId) {
        event_log
            .create_execution(&Execution {
                id: execution_id.clone(),
                tenant_id: crate::execution::TenantId("t1".to_string()),
                request_id: crate::execution::RequestId("r1".to_string()),
                parent_id: None,
                pipeline_name: "default".to_string(),
                status: ExecutionStatus::Running,
                created_at: chrono::Utc::now(),
                depth: 0,
            })
            .await
            .unwrap();
    }

    /// Helper: run drain with a single event on a fresh channel.
    ///
    /// Keeps the sender alive so the drain loop sees `TryRecvError::Empty`
    /// (not `Disconnected`) after consuming the single event. This allows
    /// tests for non-fatal events to verify state changes rather than getting
    /// a spurious `ChannelClosed` error.
    async fn drain_single(
        reactor: &Reactor,
        execution_id: &ExecutionId,
        state: &mut ExecutionState,
        cancel: &CancellationToken,
        event: PipelineEvent,
    ) -> Result<super::DrainOutcome, crate::error::ReactorError> {
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(event).await.unwrap();
        // Keep `tx` alive so the channel is empty (not disconnected) after
        // the drain loop processes the one event.
        let result = reactor
            .drain_pre_post_loop(execution_id, &mut rx, state, cancel)
            .await;
        drop(tx); // drop after drain completes
        result
    }

    // ── ActivityFailed → returns DrainOutcome::ActivityFailed ────────────

    #[tokio::test]
    async fn drain_activity_failed_returns_error() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Activity(ActivityEvent::Failed {
                name: "validate".to_string(),
                error: "bad input".to_string(),
                retryable: false,
                detail: crate::event::FailureDetail::default(),
            }),
        )
        .await
        .unwrap();

        assert!(
            matches!(
                result,
                super::DrainOutcome::ActivityFailed {
                    ref name,
                    ..
                } if name == "validate"
            ),
            "ActivityFailed should return DrainOutcome::ActivityFailed"
        );
    }

    // ── HookBlocked → returns DrainOutcome::HookBlocked ───────────────────

    #[tokio::test]
    async fn drain_hook_blocked_returns_error() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Hook(HookOutcome::Blocked {
                hook_event: "pre_generate".to_string(),
                hook_name: "tripwire".to_string(),
                reason: "jailbreak detected".to_string(),
            }),
        )
        .await
        .unwrap();

        assert!(
            matches!(
                result,
                super::DrainOutcome::HookBlocked {
                    ref hook_name,
                    ..
                } if hook_name == "tripwire"
            ),
            "HookBlocked should return DrainOutcome::HookBlocked"
        );
    }

    // ── Command(Available) → populates available_commands ─────────────────

    #[tokio::test]
    async fn drain_command_available_populates_state() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();
        assert!(state.available_commands.is_empty());

        let commands = vec![weft_core::CommandStub {
            name: "search".to_string(),
            description: "Search the web".to_string(),
        }];

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Command(CommandEvent::Available {
                commands: commands.clone(),
            }),
        )
        .await
        .unwrap();

        assert!(
            matches!(result, super::DrainOutcome::Continue),
            "Command(Available) should not stop execution"
        );
        assert_eq!(state.available_commands.len(), 1);
        assert_eq!(state.available_commands[0].name, "search");
    }

    // ── Context(ResponseAssembled) → sets state.response ──────────────────

    #[tokio::test]
    async fn drain_response_assembled_sets_state_response() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();
        assert!(state.response.is_none());

        let response = weft_core::WeftResponse {
            id: "exec-1".to_string(),
            model: "stub".to_string(),
            messages: vec![],
            usage: weft_core::WeftUsage::default(),
            timing: weft_core::WeftTiming::default(),
            degradations: vec![],
        };

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::ResponseAssembled {
                response: response.clone(),
            }),
        )
        .await
        .unwrap();

        assert!(state.response.is_some(), "state.response should be set");
        assert_eq!(state.response.unwrap().model, "stub");
    }

    // ── Signal(Cancel) → returns Some(Cancelled) and cancels token ────────

    #[tokio::test]
    async fn drain_cancel_signal_cancels_token_and_returns_error() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();
        assert!(!cancel.is_cancelled());

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Signal(SignalEvent::Received(Signal::Cancel {
                reason: "user requested".to_string(),
            })),
        )
        .await
        .unwrap();

        assert!(
            matches!(
                result,
                super::DrainOutcome::Cancelled { ref reason } if reason == "user requested"
            ),
            "Cancel signal should return DrainOutcome::Cancelled"
        );
        assert!(
            cancel.is_cancelled(),
            "cancellation token should be cancelled"
        );
    }

    // ── Selection(ModelSelected) → sets selected_model + routing snapshot ──

    #[tokio::test]
    async fn drain_model_selected_populates_state() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();
        assert!(state.selected_model.is_none());
        assert!(state.routing.is_none());

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name: "claude-3-opus".to_string(),
                score: 0.95,
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            state.selected_model.as_deref(),
            Some("claude-3-opus"),
            "selected_model should be set from ModelSelected event"
        );
        assert!(
            state.routing.is_some(),
            "routing snapshot should be populated for backward compat"
        );
        assert_eq!(state.routing.unwrap().model_routing.model, "claude-3-opus");
    }

    // ── Selection(CommandsSelected) → sets selected_commands ──────────────

    #[tokio::test]
    async fn drain_commands_selected_populates_state() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let selected = vec![
            weft_core::CommandStub {
                name: "search".to_string(),
                description: "web search".to_string(),
            },
            weft_core::CommandStub {
                name: "read_file".to_string(),
                description: "read a file".to_string(),
            },
        ];

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Selection(SelectionEvent::CommandsSelected {
                selected: selected.clone(),
            }),
        )
        .await
        .unwrap();

        assert_eq!(state.selected_commands.len(), 2);
        assert_eq!(state.selected_commands[0].name, "search");
        assert_eq!(state.selected_commands[1].name, "read_file");
    }

    // ── Selection(ProviderResolved) → sets provider fields ────────────────

    #[tokio::test]
    async fn drain_provider_resolved_populates_state() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                model_name: "claude-3-opus".to_string(),
                model_id: "claude-3-5-sonnet-20241022".to_string(),
                provider_name: "anthropic".to_string(),
                capabilities: vec!["vision".to_string(), "tools".to_string()],
                max_tokens: 8192,
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            state.selected_model_id.as_deref(),
            Some("claude-3-5-sonnet-20241022")
        );
        assert_eq!(state.selected_provider.as_deref(), Some("anthropic"));
        assert!(state.model_capabilities.contains(&"vision".to_string()));
        assert_eq!(state.model_max_tokens, Some(8192));
    }

    // ── Context(SamplingUpdated) → sets sampling fields ───────────────────

    #[tokio::test]
    async fn drain_sampling_updated_populates_state() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::SamplingUpdated {
                max_tokens: 4096,
                temperature: Some(0.7),
                top_p: Some(0.9),
            }),
        )
        .await
        .unwrap();

        assert_eq!(state.sampling_max_tokens, Some(4096));
        assert_eq!(state.sampling_temperature, Some(0.7));
        assert_eq!(state.sampling_top_p, Some(0.9));
    }

    // ── Context(CommandsFormatted) → sets command_format ──────────────────

    #[tokio::test]
    async fn drain_commands_formatted_populates_state() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();
        assert!(state.command_format.is_none());

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::CommandsFormatted {
                format: crate::event::CommandFormat::Structured,
                command_count: 2,
            }),
        )
        .await
        .unwrap();

        assert!(state.command_format.is_some());
        assert_eq!(
            state.command_format.unwrap(),
            crate::event::CommandFormat::Structured
        );
    }

    // ── MessageInjection source routing ───────────────────────────────────

    /// SystemPromptAssembly: replaces messages[0] if it's a System message.
    #[tokio::test]
    async fn drain_system_prompt_assembly_replaces_existing_system_message() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        // Seed with an existing system message.
        state.messages.push(WeftMessage {
            role: Role::System,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("old system".to_string())],
            delta: false,
            message_index: 0,
        });

        let new_system = WeftMessage {
            role: Role::System,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("new system prompt".to_string())],
            delta: false,
            message_index: 0,
        };

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::MessageInjected {
                message: new_system.clone(),
                source: MessageInjectionSource::SystemPromptAssembly,
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            state.messages.len(),
            1,
            "should still have exactly 1 message"
        );
        assert!(
            matches!(&state.messages[0].content[0], ContentPart::Text(t) if t == "new system prompt"),
            "system message should be replaced"
        );
    }

    /// SystemPromptAssembly: inserts at position 0 when no system message exists.
    #[tokio::test]
    async fn drain_system_prompt_assembly_inserts_when_no_system_message() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        // Seed with a user message only.
        state.messages.push(WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        });

        let system_msg = WeftMessage {
            role: Role::System,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("injected system".to_string())],
            delta: false,
            message_index: 0,
        };

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::MessageInjected {
                message: system_msg,
                source: MessageInjectionSource::SystemPromptAssembly,
            }),
        )
        .await
        .unwrap();

        assert_eq!(state.messages.len(), 2);
        assert_eq!(
            state.messages[0].role,
            Role::System,
            "system msg should be at [0]"
        );
    }

    /// CommandFormatInjection: inserts after system message at position 1.
    #[tokio::test]
    async fn drain_command_format_injection_inserts_after_system_message() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        // Seed: system + user messages.
        state.messages.push(WeftMessage {
            role: Role::System,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("system".to_string())],
            delta: false,
            message_index: 0,
        });
        state.messages.push(WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("user".to_string())],
            delta: false,
            message_index: 1,
        });

        let cmd_msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("command descriptions".to_string())],
            delta: false,
            message_index: 0,
        };

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::MessageInjected {
                message: cmd_msg,
                source: MessageInjectionSource::CommandFormatInjection,
            }),
        )
        .await
        .unwrap();

        assert_eq!(state.messages.len(), 3);
        // system at [0], command descriptions at [1], user at [2]
        assert_eq!(state.messages[0].role, Role::System);
        assert!(
            matches!(&state.messages[1].content[0], ContentPart::Text(t) if t == "command descriptions"),
            "command format message should be at index 1"
        );
        assert_eq!(state.messages[2].role, Role::User);
    }

    /// Other injection sources: append to messages.
    #[tokio::test]
    async fn drain_other_message_injection_appends() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        state.messages.push(WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("original".to_string())],
            delta: false,
            message_index: 0,
        });

        let appended = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: None,
            content: vec![ContentPart::Text("appended".to_string())],
            delta: false,
            message_index: 0,
        };

        drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Context(ContextEvent::MessageInjected {
                message: appended,
                // SignalInjection hits the `_ => append` branch
                source: MessageInjectionSource::SignalInjection,
            }),
        )
        .await
        .unwrap();

        assert_eq!(state.messages.len(), 2);
        assert!(
            matches!(&state.messages[1].content[0], ContentPart::Text(t) if t == "appended"),
            "appended message should be at end"
        );
    }

    // ── Channel disconnected → returns Err(ChannelClosed) ────────────────

    #[tokio::test]
    async fn drain_disconnected_channel_returns_channel_closed_error() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let (tx, mut rx) = mpsc::channel::<PipelineEvent>(1);
        // Send an event then drop the sender, then send another event to the closed channel
        // — but to trigger Disconnected we need a closed channel with an event.
        // The drain loop reads until Empty or Disconnected. To get Disconnected:
        // close the sender and ensure there's still an event waiting.
        tx.send(PipelineEvent::Activity(ActivityEvent::Started {
            name: "x".to_string(),
        }))
        .await
        .unwrap();
        drop(tx); // channel is now disconnected (sender dropped)

        // drain_pre_post_loop: reads the Started event (ok), then gets Disconnected.
        let result = reactor
            .drain_pre_post_loop(&execution_id, &mut rx, &mut state, &cancel)
            .await;

        assert!(
            matches!(result, Err(crate::error::ReactorError::ChannelClosed)),
            "disconnected channel should return ChannelClosed error; got: {result:?}"
        );
    }

    // ── Phase 2 DrainOutcome tests ────────────────────────────────────────

    /// Benign event returns DrainOutcome::Continue.
    #[tokio::test]
    async fn drain_benign_event_returns_continue() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Activity(ActivityEvent::Started {
                name: "model_selection".to_string(),
            }),
        )
        .await
        .unwrap();

        assert!(
            matches!(result, super::DrainOutcome::Continue),
            "Started event should return DrainOutcome::Continue"
        );
    }

    /// ActivityFailed carries name, error, and detail.
    #[tokio::test]
    async fn drain_activity_failed_carries_detail() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let detail = crate::event::FailureDetail {
            error_code: "classifier_unavailable".to_string(),
            detail: serde_json::json!({ "model_path": "m.onnx" }),
            cause: Some("io error".to_string()),
            attempted: Some("score candidates".to_string()),
            fallback: None,
        };

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Activity(ActivityEvent::Failed {
                name: "command_selection".to_string(),
                error: "selection failed".to_string(),
                retryable: false,
                detail: detail.clone(),
            }),
        )
        .await
        .unwrap();

        match result {
            super::DrainOutcome::ActivityFailed {
                name,
                error,
                retryable,
                detail: d,
            } => {
                assert_eq!(name, "command_selection");
                assert_eq!(error, "selection failed");
                assert!(!retryable);
                assert_eq!(d.error_code, "classifier_unavailable");
            }
            other => panic!("expected ActivityFailed, got something else: {other:?}"),
        }
    }

    /// HookBlocked carries hook_name and reason.
    #[tokio::test]
    async fn drain_hook_blocked_carries_hook_name_and_reason() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Hook(HookOutcome::Blocked {
                hook_event: "pre_generate".to_string(),
                hook_name: "content_filter".to_string(),
                reason: "unsafe content".to_string(),
            }),
        )
        .await
        .unwrap();

        match result {
            super::DrainOutcome::HookBlocked { hook_name, reason } => {
                assert_eq!(hook_name, "content_filter");
                assert_eq!(reason, "unsafe content");
            }
            other => panic!("expected HookBlocked, got something else: {other:?}"),
        }
    }

    /// Cancelled carries reason and cancels token.
    #[tokio::test]
    async fn drain_cancelled_carries_reason() {
        let event_log = Arc::new(TestEventLog::new());
        let reactor = build_reactor(Arc::clone(&event_log));
        let execution_id = ExecutionId::new();
        setup_execution(&event_log, &execution_id).await;
        let mut state = test_state();
        let cancel = CancellationToken::new();

        let result = drain_single(
            &reactor,
            &execution_id,
            &mut state,
            &cancel,
            PipelineEvent::Signal(SignalEvent::Received(Signal::Cancel {
                reason: "deadline exceeded".to_string(),
            })),
        )
        .await
        .unwrap();

        match result {
            super::DrainOutcome::Cancelled { reason } => {
                assert_eq!(reason, "deadline exceeded");
            }
            other => panic!("expected Cancelled, got something else: {other:?}"),
        }
        assert!(cancel.is_cancelled(), "token should be cancelled");
    }
}
