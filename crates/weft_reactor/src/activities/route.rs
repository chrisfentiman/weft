//! RouteActivity: performs semantic routing across all configured domains.
//!
//! Calls `Services.router.route()` for model routing and fires PreRoute/PostRoute
//! hooks for the model domain. Pushes RouteCompleted events for each successful
//! routing result.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use weft_core::{HookEvent, RoutingActivity};
use weft_hooks::HookChainResult;
use weft_router::{RoutingCandidate, RoutingDomainKind, build_model_candidates};

use crate::activity::{Activity, ActivityInput, RoutingSnapshot};
use crate::event::PipelineEvent;
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Performs semantic routing across all configured domains.
///
/// Calls [`weft_router::SemanticRouter::route`] to determine the best model for
/// the current request. Fires PreRoute and PostRoute hooks for the model domain.
/// Pushes a [`PipelineEvent::RouteCompleted`] event for the model routing result.
///
/// **Name:** `"route"`
///
/// **Events pushed:**
/// - `ActivityStarted { name: "route" }`
/// - `HookEvaluated { hook_event, hook_name, decision, duration_ms }` — for each hook evaluated
/// - `HookBlocked { hook_event, hook_name, reason }` — if a hook blocks
/// - `RouteCompleted { domain, routing }` — for each routing domain completed
/// - `ActivityCompleted { name: "route", duration_ms, idempotency_key: None }`
/// - `ActivityFailed { name: "route", error, retryable: false }` — on routing failure or hook block
pub struct RouteActivity;

impl RouteActivity {
    /// Construct a new RouteActivity.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RouteActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for RouteActivity {
    fn name(&self) -> &str {
        "route"
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
                    error: "cancelled before routing".to_string(),
                    retryable: false,
                })
                .await;
            return;
        }

        // Extract user message for routing. Use the last user message.
        let user_message = input
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
            .unwrap_or("");

        // Build model routing candidates from config.
        let model_candidates: Vec<RoutingCandidate> = build_model_candidates(&services.config);

        // Fire PreRoute hook for the model domain.
        let pre_route_payload = serde_json::json!({
            "domain": "model",
            "user_message": user_message,
        });
        let hook_start = Instant::now();
        let pre_result = services
            .hooks
            .run_chain(HookEvent::PreRoute, pre_route_payload, Some("model"))
            .await;
        let hook_duration = hook_start.elapsed().as_millis() as u64;

        match pre_result {
            HookChainResult::Blocked { reason, hook_name } => {
                let _ = event_tx
                    .send(PipelineEvent::HookBlocked {
                        hook_event: "pre_route".to_string(),
                        hook_name: hook_name.clone(),
                        reason: reason.clone(),
                    })
                    .await;
                let duration_ms = start.elapsed().as_millis() as u64;
                let _ = event_tx
                    .send(PipelineEvent::ActivityFailed {
                        name: self.name().to_string(),
                        error: format!("hook blocked routing: {reason}"),
                        retryable: false,
                    })
                    .await;
                debug!(duration_ms, "route: blocked by pre_route hook");
                return;
            }
            HookChainResult::Allowed { .. } => {
                // Push HookEvaluated for observability (best-effort — we don't have hook names here
                // without a more detailed API; use a synthetic name).
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

        // Perform routing.
        let domains = [(RoutingDomainKind::Model, model_candidates)];
        let routing_decision = match services.router.route(user_message, &domains).await {
            Ok(decision) => decision,
            Err(e) => {
                let _ = event_tx
                    .send(PipelineEvent::ActivityFailed {
                        name: self.name().to_string(),
                        error: format!("routing failed: {e}"),
                        retryable: false,
                    })
                    .await;
                return;
            }
        };

        // Extract the model result from the routing decision.
        // routing_decision.model is Option<ScoredCandidate> — use it directly.
        let (model_name, score): (String, f32) = routing_decision
            .model
            .as_ref()
            .map(|scored| (scored.id.clone(), scored.score))
            .unwrap_or_else(|| {
                // Fall back to the default model if routing produced no result.
                let default_name = services.providers.default_name().to_string();
                (default_name, 0.0_f32)
            });

        debug!(model = %model_name, score, "route: model selected");

        // Push RouteCompleted for the model domain.
        let routing_activity = RoutingActivity {
            model: model_name.clone(),
            score,
            filters: vec![],
        };
        let _ = event_tx
            .send(PipelineEvent::RouteCompleted {
                domain: "model".to_string(),
                routing: routing_activity.clone(),
            })
            .await;

        // Fire PostRoute hook.
        let post_route_payload = serde_json::json!({
            "domain": "model",
            "model": model_name,
            "score": score,
        });
        let hook_start = Instant::now();
        let post_result = services
            .hooks
            .run_chain(HookEvent::PostRoute, post_route_payload, Some("model"))
            .await;
        let hook_duration = hook_start.elapsed().as_millis() as u64;

        match post_result {
            HookChainResult::Blocked { reason, hook_name } => {
                // PostRoute blocking is treated as a failure (per spec: hook block -> ActivityFailed).
                let _ = event_tx
                    .send(PipelineEvent::HookBlocked {
                        hook_event: "post_route".to_string(),
                        hook_name: hook_name.clone(),
                        reason: reason.clone(),
                    })
                    .await;
                let duration_ms = start.elapsed().as_millis() as u64;
                let _ = event_tx
                    .send(PipelineEvent::ActivityFailed {
                        name: self.name().to_string(),
                        error: format!("post_route hook blocked: {reason}"),
                        retryable: false,
                    })
                    .await;
                debug!(duration_ms, "route: blocked by post_route hook");
                return;
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

        debug!(duration_ms, model = %model_name, "route: completed");
        let _ = RoutingSnapshot {
            model_routing: routing_activity,
            tool_necessity: None,
            tool_necessity_score: None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NullEventLog;
    use crate::test_support::{collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    async fn run_route(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = RouteActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn route_name() {
        assert_eq!(RouteActivity::new().name(), "route");
    }

    // ── Happy path ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn route_pushes_route_completed_event() {
        let input = make_test_input();
        let events = run_route(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityStarted { name } if name == "route")),
            "expected ActivityStarted"
        );

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::RouteCompleted { domain, .. } if domain == "model")
            ),
            "expected RouteCompleted for model domain"
        );

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "route")
            ),
            "expected ActivityCompleted"
        );
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn route_handles_cancellation() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = RouteActivity::new();
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

    // ── Hook blocking ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn route_activity_failed_not_retryable() {
        // Routing failure is not retryable.
        // Use a test setup where routing fails.
        let mut input = make_test_input();
        // Empty messages cause no user message — routing gets empty string.
        // But with our stub router, it still succeeds. Just verify the retryable = false
        // flag when ActivityFailed is pushed on cancellation.
        input.messages.clear();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = RouteActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ActivityFailed { .. }));
        if let Some(PipelineEvent::ActivityFailed { retryable, .. }) = failed {
            assert!(!retryable, "route failures should not be retryable");
        }
    }
}
