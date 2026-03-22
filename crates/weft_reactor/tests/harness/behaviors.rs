//! Per-variant execution logic for `TestActivity`.
//!
//! This module is `pub(super)` — it is an implementation detail of the harness.
//! All public types live in `mod.rs`.
#![allow(dead_code)]

use std::sync::atomic::Ordering;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_core::{
    CommandAction, CommandInvocation, ContentPart, Role, Source, WeftMessage,
};
use weft_reactor::activity::ActivityInput;
use weft_reactor::event::{
    ActivityEvent, CommandEvent, ContextEvent, ExecutionEvent, FailureDetail, GeneratedEvent,
    GenerationEvent, HookOutcome, PipelineEvent,
};
use weft_reactor::execution::ExecutionId;
use weft_reactor_trait::ServiceLocator;

use super::{Behavior, CallAction, PreStallBehavior};

/// Drive a single `TestActivity::execute` call.
///
/// Accepts the fully-destructured inputs so that `mod.rs` can remain thin.
pub(super) async fn execute_behavior(
    behavior: &Behavior,
    name: &str,
    execution_id: &ExecutionId,
    input: ActivityInput,
    services: &dyn ServiceLocator,
    event_tx: mpsc::Sender<PipelineEvent>,
    cancel: CancellationToken,
) {
    match behavior {
        Behavior::Generate {
            response_text,
            input_tokens,
            output_tokens,
        } => {
            run_generate(
                name,
                &input,
                response_text.as_deref(),
                *input_tokens,
                *output_tokens,
                &event_tx,
                cancel,
            )
            .await;
        }

        Behavior::Fail {
            error,
            retryable,
            detail,
        } => {
            run_fail(name, error, *retryable, detail, &event_tx).await;
        }

        Behavior::FailThenSucceed { fail_count } => {
            run_fail_then_succeed(name, &input, fail_count, &event_tx).await;
        }

        Behavior::EmitEvents { events } => {
            run_emit_events(name, &input, events, &event_tx).await;
        }

        Behavior::AssembleResponse => {
            run_assemble_response(name, execution_id, &input, &event_tx).await;
        }

        Behavior::ExecuteCommand => {
            run_execute_command(name, &input, &event_tx).await;
        }

        Behavior::WaitForCancel { pre_stall } => {
            run_wait_for_cancel(name, &input, pre_stall, &event_tx, cancel).await;
        }

        Behavior::EmitCancel { reason, post_sleep } => {
            run_emit_cancel(name, reason, *post_sleep, &event_tx).await;
        }

        Behavior::PerCall {
            call_count,
            actions,
            default_action,
        } => {
            run_per_call(name, &input, call_count, actions, default_action, &event_tx).await;
        }

        Behavior::RunHook {
            hook_event,
            hook_event_name,
        } => {
            run_hook(name, *hook_event, hook_event_name, services, &event_tx).await;
        }

        Behavior::HookBlockOnce {
            hook_event,
            hook_name,
            block_reason,
            call_count,
        } => {
            run_hook_block_once(
                name,
                &input,
                hook_event,
                hook_name,
                block_reason,
                call_count,
                &event_tx,
            )
            .await;
        }

        Behavior::CaptureAndEmit {
            capture_field,
            captured,
            events,
        } => {
            run_capture_and_emit(name, &input, capture_field, captured, events, &event_tx).await;
        }

        Behavior::NoOp => {
            run_noop(name, &event_tx).await;
        }
    }
}

// ── Per-variant helpers ────────────────────────────────────────────────────────

async fn run_generate(
    name: &str,
    input: &ActivityInput,
    response_text: Option<&str>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    event_tx: &mpsc::Sender<PipelineEvent>,
    cancel: CancellationToken,
) {
    if cancel.is_cancelled() {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Failed {
                name: name.to_string(),
                error: "cancelled".to_string(),
                retryable: false,
                detail: FailureDetail::default(),
            }))
            .await;
        return;
    }
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Generation(GenerationEvent::Started {
            model: "stub-model".to_string(),
            message_count: input.messages.len(),
        }))
        .await;
    if let Some(text) = response_text {
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                GeneratedEvent::Content {
                    part: ContentPart::Text(text.to_string()),
                },
            )))
            .await;
    }
    let _ = event_tx
        .send(PipelineEvent::Generation(GenerationEvent::Chunk(
            GeneratedEvent::Done,
        )))
        .await;
    let content = response_text
        .map(|t| vec![ContentPart::Text(t.to_string())])
        .unwrap_or_default();
    let response_message = WeftMessage {
        role: Role::Assistant,
        source: Source::Provider,
        model: Some("stub-model".to_string()),
        content,
        delta: false,
        message_index: 0,
    };
    let _ = event_tx
        .send(PipelineEvent::Generation(GenerationEvent::Completed {
            model: "stub-model".to_string(),
            response_message,
            generated_events: vec![GeneratedEvent::Done],
            input_tokens,
            output_tokens,
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: input.idempotency_key.clone(),
        }))
        .await;
}

async fn run_fail(
    name: &str,
    error: &str,
    retryable: bool,
    detail: &FailureDetail,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Failed {
            name: name.to_string(),
            error: error.to_string(),
            retryable,
            detail: detail.clone(),
        }))
        .await;
}

async fn run_fail_then_succeed(
    name: &str,
    input: &ActivityInput,
    fail_count: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;

    let old = fail_count.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
        if v > 0 { Some(v - 1) } else { Some(0) }
    });
    let remaining = old.unwrap_or(0);

    if remaining > 0 {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Failed {
                name: name.to_string(),
                error: "transient failure".to_string(),
                retryable: true,
                detail: FailureDetail::default(),
            }))
            .await;
    } else {
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                GeneratedEvent::Done,
            )))
            .await;
        let response_message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![],
            delta: false,
            message_index: 0,
        };
        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Completed {
                model: "stub-model".to_string(),
                response_message,
                generated_events: vec![GeneratedEvent::Done],
                input_tokens: None,
                output_tokens: None,
            }))
            .await;
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: name.to_string(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;
    }
}

async fn run_emit_events(
    name: &str,
    _input: &ActivityInput,
    events: &[PipelineEvent],
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    for event in events.iter().cloned() {
        let _ = event_tx.send(event).await;
    }
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: None,
        }))
        .await;
}

async fn run_assemble_response(
    name: &str,
    execution_id: &ExecutionId,
    input: &ActivityInput,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let response_message = WeftMessage {
        role: Role::Assistant,
        source: Source::Provider,
        model: Some("stub-model".to_string()),
        content: vec![ContentPart::Text(input.accumulated_text.clone())],
        delta: false,
        message_index: 0,
    };
    let response = weft_core::WeftResponse {
        id: execution_id.to_string(),
        model: "stub-model".to_string(),
        messages: vec![response_message],
        usage: weft_core::WeftUsage::default(),
        timing: weft_core::WeftTiming::default(),
        degradations: vec![],
    };
    let _ = event_tx
        .send(PipelineEvent::Context(ContextEvent::ResponseAssembled {
            response,
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: None,
        }))
        .await;
}

async fn run_execute_command(
    name: &str,
    input: &ActivityInput,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let cmd_name = input
        .metadata
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Command(CommandEvent::Completed {
            name: cmd_name.clone(),
            result: weft_core::CommandResult {
                command_name: cmd_name,
                success: true,
                output: "stub output".to_string(),
                error: None,
            },
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: input.idempotency_key.clone(),
        }))
        .await;
}

async fn run_wait_for_cancel(
    name: &str,
    input: &ActivityInput,
    pre_stall: &PreStallBehavior,
    event_tx: &mpsc::Sender<PipelineEvent>,
    cancel: CancellationToken,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    match pre_stall {
        PreStallBehavior::None => {}
        PreStallBehavior::Heartbeats { count, interval } => {
            for _ in 0..*count {
                tokio::time::sleep(*interval).await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Heartbeat {
                        activity_name: name.to_string(),
                    }))
                    .await;
            }
        }
        PreStallBehavior::OneChunk { text } => {
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Started {
                    model: "stub-model".to_string(),
                    message_count: input.messages.len(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                    GeneratedEvent::Content {
                        part: ContentPart::Text(text.clone()),
                    },
                )))
                .await;
        }
    }
    cancel.cancelled().await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Failed {
            name: name.to_string(),
            error: "cancelled".to_string(),
            retryable: false,
            detail: FailureDetail::default(),
        }))
        .await;
}

async fn run_emit_cancel(
    name: &str,
    reason: &str,
    post_sleep: Option<std::time::Duration>,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Signal(
            weft_reactor::event::SignalEvent::Received(weft_reactor::signal::Signal::Cancel {
                reason: reason.to_string(),
            }),
        ))
        .await;
    if let Some(dur) = post_sleep {
        tokio::time::sleep(dur).await;
    }
}

async fn run_per_call(
    name: &str,
    input: &ActivityInput,
    call_count: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    actions: &[CallAction],
    default_action: &CallAction,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let call_n = call_count.fetch_add(1, Ordering::SeqCst) as usize;
    let action = actions.get(call_n).unwrap_or(default_action);

    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Generation(GenerationEvent::Started {
            model: "stub-model".to_string(),
            message_count: input.messages.len(),
        }))
        .await;

    match action {
        CallAction::Done => {}
        CallAction::WithCommands { commands } => {
            for cmd in commands {
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::CommandInvocation(CommandInvocation {
                            name: cmd.clone(),
                            action: CommandAction::Execute,
                            arguments: serde_json::json!({}),
                        }),
                    )))
                    .await;
            }
        }
        CallAction::UnlessErrorSeen => {
            let has_error = input.messages.iter().any(|m| {
                m.content
                    .iter()
                    .any(|c| matches!(c, ContentPart::Text(t) if t.contains("failed")))
            });
            if !has_error {
                let cmd_name = actions
                    .iter()
                    .find_map(|a| {
                        if let CallAction::WithCommands { commands } = a {
                            commands.first().cloned()
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "slow_tool".to_string());
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::CommandInvocation(CommandInvocation {
                            name: cmd_name,
                            action: CommandAction::Execute,
                            arguments: serde_json::json!({}),
                        }),
                    )))
                    .await;
            }
        }
    }

    let _ = event_tx
        .send(PipelineEvent::Generation(GenerationEvent::Chunk(
            GeneratedEvent::Done,
        )))
        .await;
    let response_message = WeftMessage {
        role: Role::Assistant,
        source: Source::Provider,
        model: Some("stub-model".to_string()),
        content: vec![],
        delta: false,
        message_index: 0,
    };
    let _ = event_tx
        .send(PipelineEvent::Generation(GenerationEvent::Completed {
            model: "stub-model".to_string(),
            response_message,
            generated_events: vec![GeneratedEvent::Done],
            input_tokens: None,
            output_tokens: None,
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: input.idempotency_key.clone(),
        }))
        .await;
}

async fn run_hook(
    name: &str,
    hook_event: weft_core::HookEvent,
    hook_event_name: &str,
    services: &dyn ServiceLocator,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;

    let result = services
        .hooks()
        .run_chain(hook_event, serde_json::json!({}), None)
        .await;

    match result {
        weft_hooks::HookChainResult::Blocked { hook_name, reason } => {
            let _ = event_tx
                .send(PipelineEvent::Hook(HookOutcome::Blocked {
                    hook_event: hook_event_name.to_string(),
                    hook_name,
                    reason,
                }))
                .await;
        }
        weft_hooks::HookChainResult::Allowed { .. } => {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: name.to_string(),
                    idempotency_key: None,
                }))
                .await;
        }
    }
}

async fn run_hook_block_once(
    name: &str,
    input: &ActivityInput,
    hook_event: &str,
    hook_name: &str,
    block_reason: &str,
    call_count: &std::sync::Arc<std::sync::atomic::AtomicU32>,
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let call_n = call_count.fetch_add(1, Ordering::SeqCst);

    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;

    if call_n == 0 {
        let _ = event_tx
            .send(PipelineEvent::Hook(HookOutcome::Blocked {
                hook_event: hook_event.to_string(),
                hook_name: hook_name.to_string(),
                reason: block_reason.to_string(),
            }))
            .await;
    } else {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: name.to_string(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;
    }
}

async fn run_capture_and_emit(
    name: &str,
    input: &ActivityInput,
    capture_field: &str,
    captured: &std::sync::Arc<std::sync::Mutex<Option<String>>>,
    events: &[PipelineEvent],
    event_tx: &mpsc::Sender<PipelineEvent>,
) {
    let selected = input
        .metadata
        .get(capture_field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    *captured.lock().unwrap() = selected;

    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    for event in events.iter().cloned() {
        let _ = event_tx.send(event).await;
    }
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: None,
        }))
        .await;
}

async fn run_noop(name: &str, event_tx: &mpsc::Sender<PipelineEvent>) {
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Started {
            name: name.to_string(),
        }))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
        .await;
    let _ = event_tx
        .send(PipelineEvent::Activity(ActivityEvent::Completed {
            name: name.to_string(),
            idempotency_key: None,
        }))
        .await;
}
