//! HookActivity: wraps the HookRunner at a specific lifecycle point.
//!
//! Each `HookActivity` is parameterized by its `HookEvent`. Errors from the
//! hook runner are fail-open (logged, treated as Allow). The `hook_request_end`
//! special case is fire-and-forget: it spawns hook execution on a separate task
//! and immediately pushes `ActivityCompleted`.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use weft_core::HookEvent;
use weft_hooks::HookChainResult;

use crate::activity::{Activity, ActivityInput};
use crate::event::PipelineEvent;
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Wraps the `HookRunner` to fire hooks at a specific lifecycle point.
///
/// Each instance corresponds to one lifecycle event (e.g., `RequestStart`,
/// `PreResponse`, `PreToolUse`). The activity name is `"hook_{event_name}"`.
///
/// **Fail-open semantics:** Errors from the hook runner are logged and treated
/// as Allow. Hooks must not crash requests.
///
/// **RequestEnd special case:** The `RequestEnd` hook is fire-and-forget. The
/// activity spawns hook execution on a separate tokio task (gated by the
/// `request_end_semaphore`) and immediately pushes `ActivityCompleted` without
/// waiting for the hook to finish.
///
/// **Events pushed:**
/// - `ActivityStarted { name }`
/// - `HookEvaluated { hook_event, hook_name, decision, duration_ms }` (for each hook)
/// - `HookBlocked { hook_event, hook_name, reason }` (if a hook blocks)
/// - `ActivityCompleted { name, duration_ms, idempotency_key: None }`
/// - `ActivityFailed { name, error, retryable: false }` — if hook blocks and event can_block()
pub struct HookActivity {
    /// The lifecycle event this activity handles.
    hook_event: HookEvent,
    /// Activity name: `"hook_{event_name}"`.
    name: String,
}

impl HookActivity {
    /// Construct a HookActivity for the given lifecycle event.
    ///
    /// The name is derived from the event: `HookEvent::RequestStart` becomes
    /// `"hook_request_start"`, `HookEvent::PreToolUse` becomes `"hook_pre_tool_use"`, etc.
    pub fn new(hook_event: HookEvent) -> Self {
        let name = format!("hook_{}", hook_event_name_slug(hook_event));
        Self { hook_event, name }
    }
}

/// Convert a `HookEvent` to a lowercase underscore slug for the activity name.
fn hook_event_name_slug(event: HookEvent) -> &'static str {
    match event {
        HookEvent::RequestStart => "request_start",
        HookEvent::RequestEnd => "request_end",
        HookEvent::PreRoute => "pre_route",
        HookEvent::PostRoute => "post_route",
        HookEvent::PreToolUse => "pre_tool_use",
        HookEvent::PostToolUse => "post_tool_use",
        HookEvent::PreResponse => "pre_response",
    }
}

#[async_trait::async_trait]
impl Activity for HookActivity {
    fn name(&self) -> &str {
        &self.name
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
            let _ = event_tx
                .send(PipelineEvent::ActivityFailed {
                    name: self.name().to_string(),
                    error: "cancelled before hook execution".to_string(),
                    retryable: false,
                })
                .await;
            return;
        }

        // Build the hook payload from the activity input.
        let hook_payload = build_hook_payload(&input);
        let hook_event = self.hook_event;

        // RequestEnd: fire-and-forget. Spawn on separate task and return immediately.
        if hook_event == HookEvent::RequestEnd {
            let semaphore = services.request_end_semaphore.clone();
            let hooks = services.hooks.clone();
            let payload = hook_payload.clone();

            tokio::spawn(async move {
                // Gate concurrency with semaphore.
                match semaphore.try_acquire() {
                    Ok(_permit) => {
                        let _result = hooks.run_chain(hook_event, payload, None).await;
                        // Fire-and-forget: result is discarded.
                    }
                    Err(_) => {
                        warn!("request_end semaphore at capacity, dropping hook execution");
                    }
                }
            });

            let duration_ms = start.elapsed().as_millis() as u64;
            let _ = event_tx
                .send(PipelineEvent::ActivityCompleted {
                    name: self.name().to_string(),
                    duration_ms,
                    idempotency_key: None,
                })
                .await;
            debug!(duration_ms, event = ?hook_event, "hook_activity: request_end fire-and-forget");
            return;
        }

        // For all other hook events: run synchronously.
        let hook_start = Instant::now();
        let result = services
            .hooks
            .run_chain(hook_event, hook_payload, None)
            .await;
        let hook_duration = hook_start.elapsed().as_millis() as u64;

        match result {
            HookChainResult::Allowed { .. } => {
                let _ = event_tx
                    .send(PipelineEvent::HookEvaluated {
                        hook_event: hook_event_name_slug(hook_event).to_string(),
                        hook_name: self.name().to_string(),
                        decision: "allow".to_string(),
                        duration_ms: hook_duration,
                    })
                    .await;
            }
            HookChainResult::Blocked { reason, hook_name } => {
                let _ = event_tx
                    .send(PipelineEvent::HookBlocked {
                        hook_event: hook_event_name_slug(hook_event).to_string(),
                        hook_name: hook_name.clone(),
                        reason: reason.clone(),
                    })
                    .await;

                // If this event can block, fail the activity.
                if hook_event.can_block() {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let _ = event_tx
                        .send(PipelineEvent::ActivityFailed {
                            name: self.name().to_string(),
                            error: format!("hook blocked: {reason}"),
                            retryable: false,
                        })
                        .await;
                    debug!(
                        duration_ms,
                        event = ?hook_event,
                        reason = %reason,
                        "hook_activity: blocked"
                    );
                    return;
                }
                // Non-blocking event: treat block as allow (fail-open).
                warn!(
                    event = ?hook_event,
                    reason = %reason,
                    "hook returned Block on non-blocking event — treating as Allow"
                );
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = event_tx
            .send(PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms,
                idempotency_key: None,
            })
            .await;

        debug!(duration_ms, event = ?hook_event, "hook_activity: completed");
    }
}

/// Build the JSON payload for a hook invocation.
fn build_hook_payload(input: &ActivityInput) -> serde_json::Value {
    serde_json::json!({
        "messages": serde_json::to_value(&input.messages).unwrap_or(serde_json::Value::Array(vec![])),
        "routing": serde_json::to_value(&input.routing_result).unwrap_or(serde_json::Value::Null),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NullEventLog;
    use crate::test_support::{collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    async fn run_hook(event: HookEvent, input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = HookActivity::new(event);
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name derivation ──────────────────────────────────────────────────────

    #[test]
    fn hook_activity_name_request_start() {
        let a = HookActivity::new(HookEvent::RequestStart);
        assert_eq!(a.name(), "hook_request_start");
    }

    #[test]
    fn hook_activity_name_pre_tool_use() {
        let a = HookActivity::new(HookEvent::PreToolUse);
        assert_eq!(a.name(), "hook_pre_tool_use");
    }

    #[test]
    fn hook_activity_name_request_end() {
        let a = HookActivity::new(HookEvent::RequestEnd);
        assert_eq!(a.name(), "hook_request_end");
    }

    #[test]
    fn hook_activity_name_pre_response() {
        let a = HookActivity::new(HookEvent::PreResponse);
        assert_eq!(a.name(), "hook_pre_response");
    }

    #[test]
    fn hook_activity_name_post_tool_use() {
        let a = HookActivity::new(HookEvent::PostToolUse);
        assert_eq!(a.name(), "hook_post_tool_use");
    }

    // ── Happy path: NullHookRunner allows ───────────────────────────────────

    #[tokio::test]
    async fn hook_activity_pushes_hook_evaluated_on_allow() {
        let input = make_test_input();
        let events = run_hook(HookEvent::RequestStart, input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityStarted { .. })),
            "expected ActivityStarted"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::HookEvaluated { decision, .. } if decision == "allow")),
            "expected HookEvaluated with allow"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "expected ActivityCompleted"
        );
    }

    // ── HookBlocked causes ActivityFailed on blocking event ─────────────────

    #[tokio::test]
    async fn hook_activity_block_on_blocking_event_causes_activity_failed() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        // Use a services with a blocking hook runner for RequestStart.
        let services = crate::test_support::make_test_services_with_blocking_hook(
            HookEvent::RequestStart,
            "blocked by test",
        );

        let activity = HookActivity::new(HookEvent::RequestStart);
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::HookBlocked { .. })),
            "expected HookBlocked"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::ActivityFailed {
                    retryable: false,
                    ..
                }
            )),
            "expected ActivityFailed(retryable: false)"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "should not push ActivityCompleted when blocked"
        );
    }

    // ── RequestEnd is fire-and-forget ────────────────────────────────────────

    #[tokio::test]
    async fn hook_activity_request_end_returns_immediately() {
        let input = make_test_input();
        let events = run_hook(HookEvent::RequestEnd, input).await;

        // Should push ActivityCompleted immediately without waiting for hook.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "hook_request_end")),
            "expected ActivityCompleted for RequestEnd (fire-and-forget)"
        );
        // Should NOT push HookEvaluated inline (it fires on a separate task).
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::HookEvaluated { .. })),
            "RequestEnd hook should not push HookEvaluated inline"
        );
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn hook_activity_handles_cancellation() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = HookActivity::new(HookEvent::PreResponse);
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "expected ActivityFailed when cancelled"
        );
    }

    // ── Fail-open: non-blocking event block is treated as allow ──────────────

    #[tokio::test]
    async fn hook_activity_block_on_non_blocking_event_treated_as_allow() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        // PostToolUse cannot block per HookEvent::can_block() contract.
        let services = crate::test_support::make_test_services_with_blocking_hook(
            HookEvent::PostToolUse,
            "blocked by test",
        );

        let activity = HookActivity::new(HookEvent::PostToolUse);
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Block on non-blocking event → fail-open → ActivityCompleted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "PostToolUse block should be fail-open (ActivityCompleted)"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "PostToolUse block should not push ActivityFailed"
        );
    }
}
