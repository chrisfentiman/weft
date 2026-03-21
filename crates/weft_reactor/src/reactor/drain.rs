//! drain_pre_post_loop: drain events after a synchronous activity completes.
//!
//! Used for pre-loop, post-loop, and hook activities that run synchronously
//! (not spawned). Drains all pending events from the channel and dispatches
//! state updates until the channel is empty.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::ReactorError;
use crate::event::{
    ActivityEvent, CommandEvent, ContextEvent, HookOutcome, MessageInjectionSource, PipelineEvent,
    SelectionEvent, SignalEvent,
};
use crate::execution::ExecutionId;
use crate::signal::Signal;

use super::types::ExecutionState;
use super::Reactor;

impl Reactor {
    /// Drain the channel after a synchronous pre/post-loop activity completes.
    ///
    /// Returns Some(err) if execution should terminate (ActivityFailed or HookBlocked),
    /// None if execution should continue.
    pub(super) async fn drain_pre_post_loop(
        &self,
        execution_id: &ExecutionId,
        event_rx: &mut mpsc::Receiver<PipelineEvent>,
        state: &mut ExecutionState,
        cancel: &CancellationToken,
    ) -> Result<Option<ReactorError>, ReactorError> {
        loop {
            match event_rx.try_recv() {
                Ok(event) => {
                    self.record_event(execution_id, &event, None).await?;
                    match &event {
                        PipelineEvent::Activity(ActivityEvent::Failed { name, error, .. }) => {
                            return Ok(Some(ReactorError::ActivityFailed(
                                crate::activity::ActivityError::Failed {
                                    name: name.clone(),
                                    reason: error.clone(),
                                },
                            )));
                        }
                        PipelineEvent::Hook(HookOutcome::Blocked {
                            hook_name,
                            reason,
                            ..
                        }) => {
                            return Ok(Some(ReactorError::HookBlocked {
                                hook_name: hook_name.clone(),
                                reason: reason.clone(),
                            }));
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
                        PipelineEvent::Signal(SignalEvent::Received(Signal::Cancel {
                            reason,
                        })) => {
                            cancel.cancel();
                            return Ok(Some(ReactorError::Cancelled {
                                reason: reason.clone(),
                            }));
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
        Ok(None)
    }
}
