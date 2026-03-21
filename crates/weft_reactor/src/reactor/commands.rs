//! Command execution: `execute_commands` and the command-level dispatch loop.
//!
//! Runs commands sequentially, handles timeouts, and fires pre_tool_use and
//! post_tool_use hooks around each command invocation.

use std::sync::Arc;
use std::time::Duration;

use tracing::warn;

use crate::error::ReactorError;
use crate::event::{
    ActivityEvent, CommandEvent, ContextEvent, MessageInjectionSource, PipelineEvent,
};

use super::Reactor;
use super::retry::should_retry;
use super::types::CommandContext;

impl Reactor {
    /// Execute all queued commands sequentially.
    ///
    /// Returns Some(err) if execution should terminate, None on success.
    pub(super) async fn execute_commands(
        &self,
        cmd_ctx: &mut CommandContext<'_>,
        commands: &mut Vec<weft_core::CommandInvocation>,
    ) -> Result<Option<ReactorError>, ReactorError> {
        for (cmd_index, invocation) in commands.drain(..).enumerate() {
            if cmd_ctx.cancel.is_cancelled() {
                break;
            }

            // Idempotency key for this command.
            let cmd_key = format!(
                "{}:{}:{}:{}",
                cmd_ctx.execution_id, invocation.name, cmd_ctx.state.iteration, cmd_index
            );

            // Check idempotency — skip if already completed.
            if self
                .check_idempotency(cmd_ctx.execution_id, &cmd_key)
                .await?
                .is_some()
            {
                tracing::debug!(key = %cmd_key, "command idempotency hit: skipping");
                cmd_ctx.state.commands_executed += 1;
                continue;
            }

            // Run pre_tool_use hooks.
            for hook in &cmd_ctx.pipeline.loop_hooks.pre_tool_use {
                if cmd_ctx.cancel.is_cancelled() {
                    break;
                }
                let input = self.build_input(
                    cmd_ctx.execution_id,
                    cmd_ctx.state,
                    cmd_ctx.request,
                    hook,
                    None,
                );
                hook.activity
                    .execute(
                        cmd_ctx.execution_id,
                        input,
                        self.services.as_ref(),
                        self.event_log.as_ref(),
                        cmd_ctx.event_tx.clone(),
                        cmd_ctx.cancel.clone(),
                    )
                    .await;
                let terminate = self
                    .drain_pre_post_loop(
                        cmd_ctx.execution_id,
                        cmd_ctx.event_rx,
                        cmd_ctx.state,
                        cmd_ctx.cancel,
                    )
                    .await?;
                if let Some(err) = terminate {
                    return Ok(Some(err));
                }
            }

            // Build command input with idempotency key.
            let mut cmd_input = self.build_input(
                cmd_ctx.execution_id,
                cmd_ctx.state,
                cmd_ctx.request,
                &cmd_ctx.pipeline.execute_command,
                Some(cmd_key),
            );
            // Inject the specific invocation into metadata so the activity knows which command to run.
            // ExecuteCommandActivity::extract_invocation reads metadata["invocation"], so wrap it.
            cmd_input.metadata = serde_json::json!({
                "invocation": serde_json::to_value(&invocation).unwrap_or_default()
            });

            let cmd_timeout = Duration::from_secs(
                cmd_ctx
                    .pipeline
                    .execute_command
                    .timeout_secs
                    .unwrap_or(cmd_ctx.default_timeout_secs),
            );

            // Execute command with timeout.
            let cmd_activity = Arc::clone(&cmd_ctx.pipeline.execute_command.activity);
            let cmd_event_tx = cmd_ctx.event_tx.clone();
            let cmd_exec_id = cmd_ctx.execution_id.clone();
            let cmd_services = Arc::clone(&self.services);
            let cmd_event_log: Arc<dyn crate::event_log::EventLog> = Arc::clone(&self.event_log);
            let cmd_cancel = cmd_ctx.cancel.clone();

            let cmd_handle = tokio::spawn(async move {
                cmd_activity
                    .execute(
                        &cmd_exec_id,
                        cmd_input,
                        cmd_services.as_ref(),
                        cmd_event_log.as_ref(),
                        cmd_event_tx,
                        cmd_cancel,
                    )
                    .await;
            });

            // Drain events until the command completes.
            let cmd_deadline = tokio::time::Instant::now() + cmd_timeout;
            let mut cmd_completed = false;
            let mut cmd_failed = false;
            let mut cmd_failed_error = String::new();
            let mut cmd_failed_retryable = false;
            let cmd_name = invocation.name.clone();

            'cmd: loop {
                tokio::select! {
                    biased;
                    _ = cmd_ctx.cancel.cancelled() => {
                        cmd_handle.abort();
                        break 'cmd;
                    }
                    event_opt = cmd_ctx.event_rx.recv() => {
                        match event_opt {
                            None => return Err(ReactorError::ChannelClosed),
                            Some(event) => {
                                self.record_event(cmd_ctx.execution_id, &event, None).await?;
                                match &event {
                                    PipelineEvent::Command(CommandEvent::Completed { name, result }) if name == &cmd_name => {
                                        // Inject command result into messages and record the injection
                                        // as a MessageInjected event so the message list is fully
                                        // reconstructable from the event log.
                                        let result_msg = weft_core::WeftMessage {
                                            role: weft_core::Role::User,
                                            source: weft_core::Source::Client,
                                            model: None,
                                            content: vec![weft_core::ContentPart::Text(result.output.clone())],
                                            delta: false,
                                            message_index: cmd_ctx.state.messages.len() as u32,
                                        };
                                        cmd_ctx.state.messages.push(result_msg.clone());
                                        self.record_event(
                                            cmd_ctx.execution_id,
                                            &PipelineEvent::Context(ContextEvent::MessageInjected {
                                                message: result_msg,
                                                source: MessageInjectionSource::CommandResult {
                                                    command_name: name.clone(),
                                                },
                                            }),
                                            None,
                                        ).await?;
                                        cmd_ctx.state.commands_executed += 1;
                                        cmd_completed = true;
                                    }
                                    PipelineEvent::Command(CommandEvent::Failed { name, error }) if name == &cmd_name => {
                                        // Inject error into messages and record the injection
                                        // as a MessageInjected event so the message list is fully
                                        // reconstructable from the event log.
                                        let err_msg = weft_core::WeftMessage {
                                            role: weft_core::Role::User,
                                            source: weft_core::Source::Client,
                                            model: None,
                                            content: vec![weft_core::ContentPart::Text(format!("Command failed: {error}"))],
                                            delta: false,
                                            message_index: cmd_ctx.state.messages.len() as u32,
                                        };
                                        cmd_ctx.state.messages.push(err_msg.clone());
                                        self.record_event(
                                            cmd_ctx.execution_id,
                                            &PipelineEvent::Context(ContextEvent::MessageInjected {
                                                message: err_msg,
                                                source: MessageInjectionSource::CommandError {
                                                    command_name: name.clone(),
                                                },
                                            }),
                                            None,
                                        ).await?;
                                        cmd_ctx.state.commands_executed += 1;
                                        cmd_completed = true;
                                    }
                                    PipelineEvent::Activity(ActivityEvent::Failed { name, error, retryable }) if name == cmd_ctx.pipeline.execute_command.activity.name() => {
                                        cmd_failed = true;
                                        cmd_failed_error = error.clone();
                                        cmd_failed_retryable = *retryable;
                                    }
                                    PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == cmd_ctx.pipeline.execute_command.activity.name() => {
                                        cmd_completed = true;
                                    }
                                    _ => {}
                                }
                                if cmd_completed || cmd_failed {
                                    break 'cmd;
                                }
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(cmd_deadline) => {
                        cmd_handle.abort();
                        cmd_failed = true;
                        cmd_failed_retryable = true;
                        cmd_failed_error = format!("command timed out after {} secs", cmd_timeout.as_secs());
                        break 'cmd;
                    }
                }
            }

            // Handle command failure with retry.
            if cmd_failed {
                let policy = cmd_ctx.pipeline.execute_command.retry_policy.as_ref();
                if cmd_failed_retryable
                    && should_retry(policy, 0, &cmd_ctx.state.budget, cmd_ctx.cancel)
                {
                    // Command retry is not yet implemented. The spec's "on ActivityFailed with
                    // retryable=true, apply retry policy" applies to generation activities.
                    // Command-level retry requires per-command attempt tracking and a retry loop
                    // equivalent to the generate dispatch loop, which is deferred to a future
                    // phase. For now, log a warning and propagate the failure. Commands that need
                    // retry should handle it internally (e.g., via idempotent re-submission).
                    warn!(
                        "command '{}' failed (retryable=true): {}; command retry not yet implemented",
                        cmd_name, cmd_failed_error
                    );
                }
                return Ok(Some(ReactorError::ActivityFailed(
                    crate::activity::ActivityError::Failed {
                        name: cmd_name,
                        reason: cmd_failed_error,
                    },
                )));
            }

            // Run post_tool_use hooks.
            for hook in &cmd_ctx.pipeline.loop_hooks.post_tool_use {
                if cmd_ctx.cancel.is_cancelled() {
                    break;
                }
                let input = self.build_input(
                    cmd_ctx.execution_id,
                    cmd_ctx.state,
                    cmd_ctx.request,
                    hook,
                    None,
                );
                hook.activity
                    .execute(
                        cmd_ctx.execution_id,
                        input,
                        self.services.as_ref(),
                        self.event_log.as_ref(),
                        cmd_ctx.event_tx.clone(),
                        cmd_ctx.cancel.clone(),
                    )
                    .await;
                let terminate = self
                    .drain_pre_post_loop(
                        cmd_ctx.execution_id,
                        cmd_ctx.event_rx,
                        cmd_ctx.state,
                        cmd_ctx.cancel,
                    )
                    .await?;
                if let Some(err) = terminate {
                    return Ok(Some(err));
                }
            }
        }
        Ok(None)
    }
}
