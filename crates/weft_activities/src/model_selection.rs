//! ModelSelection activity: semantic model selection for the pre-loop.
//!
//! Replaces the model-routing half of `RouteActivity`. Selects the best model
//! via semantic scoring and emits `ModelSelected`. Supports three routing modes:
//! - `RoutingMode::Direct`: resolve filters to candidates, bypass scoring if exactly one.
//! - `RoutingMode::Auto` / `RoutingMode::Council`: full semantic scoring against all
//!   configured models (filtered by any routing instruction filters).
//!
//! **Fail mode: CLOSED.** If no model can be determined, pushes `ActivityFailed`.
//! Generation requires a model — proceeding without one is undefined behaviour.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use weft_core::routing::resolve_filters;
use weft_core::{HookEvent, ModelInfo, RoutingMode};
use weft_hooks_trait::HookChainResult;
use weft_reactor_trait::{
    Activity, ActivityEvent, ActivityInput, Criticality, EventLog, ExecutionId, FailureDetail,
    HookOutcome, PipelineEvent, SelectionEvent, SemanticSelection, ServiceLocator,
};
use weft_router_trait::{RoutingCandidate, RoutingDomainKind};

use super::selection_util::extract_user_message;

/// Selects the model to use for the current execution via semantic routing.
///
/// **Name:** `"model_selection"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "model_selection" })`
/// - `Hook(HookOutcome::Evaluated)` / `Hook(HookOutcome::Blocked)` — for PreRoute / PostRoute hooks
/// - `Selection(SelectionEvent::ModelSelected { model_name, score })` — on success (`all_scores` stored via enrichment)
/// - `Activity(ActivityEvent::Completed { name: "model_selection", idempotency_key: None })`
/// - `Activity(ActivityEvent::Failed { name: "model_selection", error, retryable: false })` — on failure
pub struct ModelSelectionActivity;

impl ModelSelectionActivity {
    /// Construct a new `ModelSelectionActivity`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ModelSelectionActivity {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticSelection for ModelSelectionActivity {
    fn selection_domain(&self) -> &'static str {
        "model"
    }
}

#[async_trait::async_trait]
impl Activity for ModelSelectionActivity {
    fn name(&self) -> &str {
        "model_selection"
    }

    fn criticality(&self) -> Criticality {
        Criticality::SemiCritical
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        services: &dyn ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;

        if cancel.is_cancelled() {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "cancelled before model selection".to_string(),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "cancelled".to_string(),
                        detail: serde_json::Value::Null,
                        cause: Some(
                            "cancellation token was set before activity started".to_string(),
                        ),
                        attempted: Some("select model for request".to_string()),
                        fallback: None,
                    },
                }))
                .await;
            return;
        }

        // Extract last user message text for semantic scoring.
        let user_message = extract_user_message(&input);

        // Convert pre-computed ModelCandidates from config snapshot to RoutingCandidates.
        // Done once per request (not per-score-call) to avoid repeated allocation.
        let all_candidates: Vec<RoutingCandidate> = input
            .config
            .model_candidates
            .iter()
            .map(|mc| RoutingCandidate {
                id: mc.name.clone(),
                examples: mc.examples.clone(),
            })
            .collect();

        // Resolve the routing instruction to the candidate set.
        let instruction = &input.request.routing;

        // Use pre-computed ModelInfo list from config snapshot for filter resolution.
        let model_infos: Vec<ModelInfo> = input.config.model_infos.clone();

        // Fire PreRoute hook for "model" domain BEFORE routing (per spec Section 5.1 step 4).
        let pre_payload = serde_json::json!({
            "domain": "model",
            "user_message": user_message,
        });
        let pre_result = services
            .hooks()
            .run_chain(HookEvent::PreRoute, pre_payload, Some("model"))
            .await;

        match pre_result {
            HookChainResult::Blocked { reason, hook_name } => {
                let _ = event_tx
                    .send(PipelineEvent::Hook(HookOutcome::Blocked {
                        hook_event: "pre_route".to_string(),
                        hook_name,
                        reason: reason.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: format!("pre_route hook blocked model selection: {reason}"),
                        retryable: false,
                        detail: FailureDetail {
                            error_code: "hook_blocked".to_string(),
                            detail: serde_json::json!({ "hook_event": "pre_route", "reason": reason }),
                            cause: Some(reason.clone()),
                            attempted: Some("run pre_route hook before model scoring".to_string()),
                            fallback: None,
                        },
                    }))
                    .await;
                debug!("model_selection: blocked by pre_route hook");
                return;
            }
            HookChainResult::Allowed { .. } => {
                let _ = event_tx
                    .send(PipelineEvent::Hook(HookOutcome::Evaluated {
                        hook_event: "pre_route".to_string(),
                        hook_name: "pre_route".to_string(),
                        decision: "allow".to_string(),
                    }))
                    .await;
            }
        }

        // Determine the working candidate set based on routing mode and filters.
        // _all_scores is no longer in the event payload (observability-only, moved to enrichment).
        let (selected_model, score, _all_scores) = match instruction.mode {
            RoutingMode::Direct => {
                // Apply filters to narrow candidates.
                let surviving_names = if instruction.filters.is_empty() {
                    // No filters in Direct mode — treat as semantic scoring of all.
                    all_candidates.iter().map(|c| c.id.clone()).collect()
                } else {
                    resolve_filters(&instruction.filters, &model_infos)
                };

                if surviving_names.is_empty() {
                    // Zero survivors: fail closed.
                    let filter_str = instruction.filters.join("/");
                    let models_scored = all_candidates.len();
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Failed {
                            name: self.name().to_string(),
                            error: format!("no models match routing instruction '{filter_str}'"),
                            retryable: false,
                            detail: FailureDetail {
                                error_code: "no_matching_model".to_string(),
                                detail: serde_json::json!({
                                    "models_scored": models_scored,
                                    "best_score": 0.0_f32,
                                    "threshold": 1.0_f32,
                                    "filters": instruction.filters,
                                }),
                                cause: Some(format!("no models match filters: {filter_str}")),
                                attempted: Some(format!(
                                    "filter {} model candidates by routing instruction",
                                    models_scored
                                )),
                                fallback: Some("Fall back to default model".to_string()),
                            },
                        }))
                        .await;
                    return;
                }

                if surviving_names.len() == 1 {
                    // Exactly one survivor: use directly, score 1.0. No scoring needed.
                    let model_name = surviving_names
                        .into_iter()
                        .next()
                        .expect("surviving_names.len() == 1 checked above");
                    (model_name, 1.0_f32, vec![])
                } else {
                    // Multiple survivors: score them semantically.
                    let surviving_candidates: Vec<RoutingCandidate> = all_candidates
                        .into_iter()
                        .filter(|c| surviving_names.contains(&c.id))
                        .collect();

                    match semantic_score(
                        services,
                        user_message,
                        surviving_candidates,
                        &event_tx,
                        self.name(),
                    )
                    .await
                    {
                        Some(result) => result,
                        None => return, // error already pushed
                    }
                }
            }

            RoutingMode::Auto | RoutingMode::Council => {
                // Apply any filters from the routing instruction, then score semantically.
                let candidates = if instruction.filters.is_empty() {
                    all_candidates
                } else {
                    let surviving_names = resolve_filters(&instruction.filters, &model_infos);
                    all_candidates
                        .into_iter()
                        .filter(|c| surviving_names.contains(&c.id))
                        .collect()
                };

                if candidates.is_empty() {
                    // No candidates after filtering: fall back to default.
                    warn!("model_selection: no candidates after filter, using default");
                    let default_name = services.providers().default_name().to_string();
                    (default_name, 0.0_f32, vec![])
                } else {
                    match semantic_score(services, user_message, candidates, &event_tx, self.name())
                        .await
                    {
                        Some(result) => result,
                        None => return, // error already pushed
                    }
                }
            }
        };

        // Apply model threshold: if the selected model's score is below the domain threshold,
        // fall back to the default model (per spec Section 5.1 step 7).
        let model_threshold = input.config.model_domain_threshold;

        let (final_model, final_score) = if let Some(threshold) = model_threshold {
            if score < threshold && score > 0.0 {
                // Score below threshold and not a direct/fallback: use default.
                warn!(
                    score,
                    threshold, "model_selection: score below threshold, using default model"
                );
                (services.providers().default_name().to_string(), 0.0_f32)
            } else {
                (selected_model, score)
            }
        } else {
            (selected_model, score)
        };

        // Push ModelSelected event. all_scores is observability-only; the event log
        // enrichment mechanism preserves it as _obs_all_scores (Phase 2 slimming).
        let _ = event_tx
            .send(PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name: final_model.clone(),
                score: final_score,
            }))
            .await;

        debug!(model = %final_model, score = final_score, "model_selection: model selected");

        // Fire PostRoute hook for "model" domain.
        let post_payload = serde_json::json!({
            "domain": "model",
            "model": final_model,
            "score": final_score,
        });
        let post_result = services
            .hooks()
            .run_chain(HookEvent::PostRoute, post_payload, Some("model"))
            .await;

        match post_result {
            HookChainResult::Blocked { reason, hook_name } => {
                let _ = event_tx
                    .send(PipelineEvent::Hook(HookOutcome::Blocked {
                        hook_event: "post_route".to_string(),
                        hook_name,
                        reason: reason.clone(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: format!("post_route hook blocked model selection: {reason}"),
                        retryable: false,
                        detail: FailureDetail {
                            error_code: "hook_blocked".to_string(),
                            detail: serde_json::json!({ "hook_event": "post_route", "reason": reason }),
                            cause: Some(reason.clone()),
                            attempted: Some("run post_route hook after model scoring".to_string()),
                            fallback: None,
                        },
                    }))
                    .await;
                debug!("model_selection: blocked by post_route hook");
                return;
            }
            HookChainResult::Allowed { .. } => {
                let _ = event_tx
                    .send(PipelineEvent::Hook(HookOutcome::Evaluated {
                        hook_event: "post_route".to_string(),
                        hook_name: "post_route".to_string(),
                        decision: "allow".to_string(),
                    }))
                    .await;
            }
        }

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Score `candidates` semantically and return `(selected_model, score, all_scores)`.
///
/// Falls back to the default provider model on router error. Returns `None` only
/// if a fatal error occurs that already pushed `ActivityFailed` — callers must
/// return immediately in that case.
///
/// Note: this helper does NOT fire hooks. Hook firing happens in `execute` around
/// the call to this helper.
async fn semantic_score(
    services: &dyn ServiceLocator,
    user_message: &str,
    candidates: Vec<RoutingCandidate>,
    event_tx: &mpsc::Sender<PipelineEvent>,
    activity_name: &str,
) -> Option<(String, f32, Vec<(String, f32)>)> {
    let domains = [(RoutingDomainKind::Model, candidates.clone())];
    match services.router().route(user_message, &domains).await {
        Err(e) => {
            // Router failure: fall back to default with score 0.0.
            warn!(error = %e, "model_selection: router error, falling back to default");
            let default_name = services.providers().default_name().to_string();
            // Verify default exists in providers; if not, fail closed.
            if services.providers().model_id(&default_name).is_none() {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: activity_name.to_string(),
                        error: format!(
                            "router failed and default model '{default_name}' is not registered"
                        ),
                        retryable: false,
                        detail: FailureDetail {
                            error_code: "classifier_unavailable".to_string(),
                            detail: serde_json::json!({
                                "model_path": "unknown",
                                "error": e.to_string(),
                                "default_model": default_name,
                            }),
                            cause: Some(e.to_string()),
                            attempted: Some(format!(
                                "score {} candidates semantically",
                                candidates.len()
                            )),
                            fallback: None,
                        },
                    }))
                    .await;
                return None;
            }
            Some((default_name, 0.0_f32, vec![]))
        }
        Ok(decision) => {
            // decision.model carries the selected model (Option<ScoredCandidate>).
            // decision.commands will be empty since we only passed RoutingDomainKind::Model.
            let model_result = decision.model;
            let selected_model = model_result
                .as_ref()
                .map(|m| m.id.clone())
                .unwrap_or_else(|| {
                    warn!("model_selection: router returned no model, using default");
                    services.providers().default_name().to_string()
                });
            let selected_score = model_result.as_ref().map(|m| m.score).unwrap_or(0.0_f32);

            // Verify the selected model is in the provider registry.
            // If not, fall back to default with a warning.
            let (final_model, final_score) =
                if services.providers().model_id(&selected_model).is_none() {
                    warn!(
                        model = %selected_model,
                        "model_selection: router selected unknown model, using default"
                    );
                    let default_name = services.providers().default_name().to_string();
                    (default_name, 0.0_f32)
                } else {
                    (selected_model, selected_score)
                };

            // Build all_scores: include scored candidate for winner, 0.0 for others.
            // Provides full observability even when the router only returns the winner.
            let all_scores = build_all_scores(&candidates, &final_model, final_score);

            Some((final_model, final_score, all_scores))
        }
    }
}

/// Build the `all_scores` vec: the winning model gets `winner_score`; all other
/// candidates get 0.0.
///
/// **Non-winner scores are 0.0 due to a router limitation:** the semantic router
/// returns only the top-scored candidate per domain, not a ranked list of all
/// candidates. We assign 0.0 to non-winners rather than omitting them so that
/// observability tools can see the full candidate set alongside the winner's score.
/// Consumers should treat 0.0 as "not selected / score unknown", not "zero relevance".
fn build_all_scores(
    candidates: &[RoutingCandidate],
    winner_id: &str,
    winner_score: f32,
) -> Vec<(String, f32)> {
    candidates
        .iter()
        .map(|c| {
            let score = if c.id == winner_id { winner_score } else { 0.0 };
            (c.id.clone(), score)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        NullEventLog, collect_events, make_test_input, make_test_services,
        make_test_services_with_blocking_hook,
    };
    use pretty_assertions::assert_eq;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_core::HookEvent;

    async fn run_model_selection(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ModelSelectionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn model_selection_name() {
        assert_eq!(ModelSelectionActivity::new().name(), "model_selection");
    }

    #[test]
    fn model_selection_domain() {
        assert_eq!(ModelSelectionActivity::new().selection_domain(), "model");
    }

    // ── Criticality ──────────────────────────────────────────────────────

    #[test]
    fn model_selection_criticality_is_semi_critical() {
        use weft_reactor_trait::Criticality;
        assert_eq!(
            ModelSelectionActivity::new().criticality(),
            Criticality::SemiCritical
        );
    }

    // ── Happy path: emits ModelSelected ──────────────────────────────────────

    #[tokio::test]
    async fn model_selection_emits_model_selected() {
        let input = make_test_input();
        let events = run_model_selection(input).await;

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "model_selection")
            ),
            "expected Activity(Started)"
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ModelSelected { .. })
            )),
            "expected Selection(ModelSelected) event"
        );

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "model_selection")
            ),
            "expected Activity(Completed)"
        );

        // No failure events.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "did not expect Activity(Failed)"
        );
    }

    // ── ModelSelected carries model name and score ────────────────────────────
    // (all_scores was observability-only and moved to enrichment in Phase 2)

    #[tokio::test]
    async fn model_selected_has_model_name_and_score() {
        let input = make_test_input();
        let events = run_model_selection(input).await;

        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ModelSelected { model_name, score }) = e
            {
                Some((model_name.clone(), *score))
            } else {
                None
            }
        });

        let (model_name, score) = selected.expect("Selection(ModelSelected) must be present");
        assert!(!model_name.is_empty(), "model_name must not be empty");
        assert!(score >= 0.0, "score must be non-negative");
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn model_selection_handles_cancellation() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancel

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ModelSelectionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) on cancellation"
        );
        // Must NOT have ModelSelected when cancelled.
        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ModelSelected { .. })
            )),
            "must not emit Selection(ModelSelected) when cancelled"
        );
    }

    // ── PreRoute hook blocking ────────────────────────────────────────────────

    #[tokio::test]
    async fn model_selection_pre_route_hook_blocking_pushes_hook_blocked() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services =
            make_test_services_with_blocking_hook(HookEvent::PreRoute, "blocked by policy");
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ModelSelectionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let hook_blocked = events.iter().find(|e| {
            matches!(e, PipelineEvent::Hook(HookOutcome::Blocked { hook_event, .. }) if hook_event == "pre_route")
        });
        assert!(
            hook_blocked.is_some(),
            "expected Hook(Blocked) for pre_route"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) after pre_route hook block"
        );

        // ModelSelected must NOT be present.
        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ModelSelected { .. })
            )),
            "Selection(ModelSelected) must not be pushed when pre_route hook blocks"
        );
    }

    // ── PostRoute hook blocking ───────────────────────────────────────────────

    #[tokio::test]
    async fn model_selection_post_route_hook_blocking_pushes_hook_blocked() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services_with_blocking_hook(HookEvent::PostRoute, "post blocked");
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ModelSelectionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let hook_blocked = events.iter().find(|e| {
            matches!(e, PipelineEvent::Hook(HookOutcome::Blocked { hook_event, .. }) if hook_event == "post_route")
        });
        assert!(
            hook_blocked.is_some(),
            "expected Hook(Blocked) for post_route"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) after post_route hook block"
        );
    }

    // ── Direct routing mode ───────────────────────────────────────────────────

    #[tokio::test]
    async fn model_selection_direct_routing_single_survivor_score_one() {
        // Direct mode with a filter that matches exactly one model: score 1.0, bypass scoring.
        let mut input = make_test_input();
        // The test config has one model named "stub-model". Filter to it directly.
        input.request.routing = weft_core::ModelRoutingInstruction::parse("stub-model");
        assert_eq!(input.request.routing.mode, weft_core::RoutingMode::Direct);

        let events = run_model_selection(input).await;

        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ModelSelected { model_name, score }) = e
            {
                Some((model_name.clone(), *score))
            } else {
                None
            }
        });

        let (_model_name, score) = selected.expect("Selection(ModelSelected) must be present");
        assert!(
            (score - 1.0_f32).abs() < 1e-5,
            "direct single-model routing should have score 1.0"
        );
        // all_scores is now observability-only (moved to enrichment in Phase 2).
    }

    #[tokio::test]
    async fn model_selection_direct_routing_zero_survivors_fails() {
        // Direct mode with a filter that matches no models: ActivityFailed.
        let mut input = make_test_input();
        input.request.routing =
            weft_core::ModelRoutingInstruction::parse("nonexistent-provider-xyz");
        assert_eq!(input.request.routing.mode, weft_core::RoutingMode::Direct);

        let events = run_model_selection(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) when no models match filters"
        );

        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ModelSelected { .. })
            )),
            "Selection(ModelSelected) must not be pushed when zero survivors"
        );
    }

    // ── Auto routing mode ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn model_selection_auto_mode_emits_model_selected() {
        let mut input = make_test_input();
        input.request.routing = weft_core::ModelRoutingInstruction::parse("auto");
        assert_eq!(input.request.routing.mode, weft_core::RoutingMode::Auto);

        let events = run_model_selection(input).await;

        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ModelSelected { .. })
            )),
            "expected Selection(ModelSelected) in auto mode"
        );
    }

    // ── Fallback on ActivityFailed is not retryable ───────────────────────────

    #[tokio::test]
    async fn model_selection_activity_failed_not_retryable() {
        // Direct routing with zero survivors to trigger ActivityFailed.
        let mut input = make_test_input();
        input.request.routing =
            weft_core::ModelRoutingInstruction::parse("no-such-provider-or-model");

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        ModelSelectionActivity::new()
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })));
        if let Some(PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. })) = failed {
            assert!(!retryable, "model selection failures must not be retryable");
        } else {
            panic!("expected Activity(Failed)");
        }
    }

    // ── Router error falls back to default model ──────────────────────────────

    #[tokio::test]
    async fn model_selection_router_error_falls_back_to_default_model() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        // Use a router that always errors — model_selection should fall back to
        // the default model ("stub-model") rather than pushing ActivityFailed.
        let services = crate::test_support::make_test_services_with_failing_router();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        ModelSelectionActivity::new()
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // ModelSelected must be present (fallback, not failure).
        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name,
                score,
                ..
            }) = e
            {
                Some((model_name.clone(), *score))
            } else {
                None
            }
        });
        let (model_name, score) =
            selected.expect("Selection(ModelSelected) must be present on router error");
        assert_eq!(
            model_name,
            services.providers().default_name(),
            "router error must fall back to default model"
        );
        assert!(
            (score - 0.0_f32).abs() < 1e-5,
            "fallback model score must be 0.0"
        );

        // No ActivityFailed (router error is a soft fallback).
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "router error must not push Activity(Failed) (fallback mode)"
        );
    }

    // ── Threshold filtering falls back to default when score is below threshold ─

    #[tokio::test]
    async fn model_selection_score_below_threshold_uses_default_model() {
        // The stub router always scores at 0.9. We set the model domain threshold
        // above 0.9 (e.g. 0.95) so the selected model's score fails the threshold
        // and the activity should fall back to the default model.
        //
        // model_selection reads the threshold from input.config.model_domain_threshold,
        // so we construct an input with a ResolvedConfig derived from a config that has
        // threshold = 0.95.
        let mut input = make_test_input();

        // Override input.config with a ResolvedConfig that has model_domain_threshold = 0.95.
        let threshold_config: weft_core::WeftConfig = toml::from_str(
            r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are helpful."

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[router.domains.model]
threshold = 0.95

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "stub-model"
  model = "stub-model-v1"
  examples = ["example query"]
"#,
        )
        .expect("threshold test config must parse");
        input.config = Arc::new(weft_core::ResolvedConfig::from_operator(&threshold_config));

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        ModelSelectionActivity::new()
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // ModelSelected must be present.
        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name,
                score,
                ..
            }) = e
            {
                Some((model_name.clone(), *score))
            } else {
                None
            }
        });
        let (model_name, score) = selected.expect("Selection(ModelSelected) must be present");
        assert_eq!(
            model_name,
            services.providers().default_name(),
            "score below threshold must fall back to default model"
        );
        assert!(
            (score - 0.0_f32).abs() < 1e-5,
            "fallback score must be 0.0 when threshold not met"
        );
    }

    // ── Auto mode: empty candidates after filter falls back to default ─────────

    #[tokio::test]
    async fn model_selection_auto_mode_empty_candidates_after_filter_uses_default() {
        // Auto mode with a filter that matches no models: falls back to default
        // (unlike Direct mode which fails closed).
        let mut input = make_test_input();
        // Use Auto routing mode with a filter that won't match any configured model.
        // "auto/nonexistent-provider" parses to Auto mode with filter "nonexistent-provider".
        input.request.routing =
            weft_core::ModelRoutingInstruction::parse("auto/nonexistent-provider-xyz");
        assert_eq!(input.request.routing.mode, weft_core::RoutingMode::Auto);
        assert!(!input.request.routing.filters.is_empty());

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        ModelSelectionActivity::new()
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Fail-open: ModelSelected must be present (no ActivityFailed).
        let selected = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ModelSelected {
                model_name,
                score,
                ..
            }) = e
            {
                Some((model_name.clone(), *score))
            } else {
                None
            }
        });
        let (model_name, score) =
            selected.expect("Selection(ModelSelected) must be present (fail-open)");
        assert_eq!(
            model_name,
            services.providers().default_name(),
            "empty candidates after Auto filter must fall back to default model"
        );
        assert!(
            (score - 0.0_f32).abs() < 1e-5,
            "fallback score must be 0.0 when no candidates after filter"
        );

        // No ActivityFailed (Auto mode is fail-open for empty candidates).
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "Auto mode empty candidates must not push Activity(Failed)"
        );
    }

    // ── FailureDetail enrichment ──────────────────────────────────────────────

    #[tokio::test]
    async fn model_selection_direct_zero_survivors_failure_detail_has_no_matching_model() {
        use pretty_assertions::assert_ne;

        let mut input = make_test_input();
        input.request.routing =
            weft_core::ModelRoutingInstruction::parse("nonexistent-provider-xyz");
        assert_eq!(input.request.routing.mode, weft_core::RoutingMode::Direct);

        let events = run_model_selection(input).await;

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed) for zero survivors");

        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { detail, .. }) => {
                assert_ne!(
                    detail.error_code, "unknown",
                    "zero survivors failure must have non-unknown error_code"
                );
                assert_eq!(detail.error_code, "no_matching_model");
                assert!(
                    detail.cause.is_some(),
                    "cause must be populated for no_matching_model failure"
                );
                assert!(
                    detail.attempted.is_some(),
                    "attempted must be populated for no_matching_model failure"
                );
                // detail JSON must include models_scored
                assert!(
                    detail.detail.get("models_scored").is_some(),
                    "detail must include models_scored"
                );
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }
}
