//! Test doubles for the semantic router trait.
//!
//! Gated behind `feature = "test-support"`. Available to downstream crates
//! via `weft_router = { ..., features = ["test-support"] }` in `[dev-dependencies]`.
//!
//! **Available stubs:**
//! - [`StubRouter`] — selects the first model candidate and scores all command candidates at 0.9
//! - [`ErrorRouter`] — always returns `RouterError::InferenceFailed`

use async_trait::async_trait;

use crate::{
    RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
    SemanticRouter,
};

// ── StubRouter ────────────────────────────────────────────────────────────────

/// A stub router that selects the first model candidate and scores all command candidates at 0.9.
///
/// - `Model` domain: selects the first candidate or falls back to `"stub-model"` if none given.
/// - `Commands` domain: scores all candidates at 0.9.
/// - `ToolNecessity` domain: sets `tools_needed = Some(true)`.
/// - `Memory` domain: no-op.
///
/// `score_memory_candidates` returns all candidates scored at 0.9.
pub struct StubRouter;

#[async_trait]
impl SemanticRouter for StubRouter {
    async fn route(
        &self,
        _user_message: &str,
        domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, RouterError> {
        let mut decision = RoutingDecision::empty();

        for (kind, candidates) in domains {
            match kind {
                RoutingDomainKind::Model => {
                    // Select the first candidate, or "stub-model" if none provided.
                    if let Some(first) = candidates.first() {
                        decision.model = Some(ScoredCandidate {
                            id: first.id.clone(),
                            score: 0.9,
                        });
                    } else {
                        decision.model = Some(ScoredCandidate {
                            id: "stub-model".to_string(),
                            score: 0.9,
                        });
                    }
                }
                RoutingDomainKind::Commands => {
                    // Score all candidates at 0.9.
                    decision.commands = candidates
                        .iter()
                        .map(|c| ScoredCandidate {
                            id: c.id.clone(),
                            score: 0.9,
                        })
                        .collect();
                }
                RoutingDomainKind::ToolNecessity => {
                    decision.tools_needed = Some(true);
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

// ── ErrorRouter ───────────────────────────────────────────────────────────────

/// A stub router that always returns `RouterError::InferenceFailed`.
///
/// Used in tests that verify the fallback path when the router fails:
/// - `ModelSelectionActivity` falls back to the default model on router error.
/// - `CommandSelectionActivity` falls back to all commands capped by `max_commands`.
pub struct ErrorRouter {
    /// The reason string returned in the error.
    pub reason: String,
}

impl ErrorRouter {
    /// Construct an error router that fails with the given reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl SemanticRouter for ErrorRouter {
    async fn route(
        &self,
        _user_message: &str,
        _domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, RouterError> {
        Err(RouterError::InferenceFailed(self.reason.clone()))
    }

    async fn score_memory_candidates(
        &self,
        _text: &str,
        _candidates: &[RoutingCandidate],
    ) -> Result<Vec<ScoredCandidate>, RouterError> {
        Err(RouterError::InferenceFailed(self.reason.clone()))
    }
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates(ids: &[&str]) -> Vec<RoutingCandidate> {
        ids.iter()
            .map(|id| RoutingCandidate {
                id: id.to_string(),
                examples: vec![],
            })
            .collect()
    }

    #[tokio::test]
    async fn stub_router_selects_first_model_candidate() {
        let router = StubRouter;
        let candidates = make_candidates(&["fast", "smart"]);
        let decision = router
            .route("hello", &[(RoutingDomainKind::Model, candidates)])
            .await
            .unwrap();
        let model = decision.model.expect("model should be set");
        assert_eq!(model.id, "fast");
        assert!((model.score - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn stub_router_falls_back_to_stub_model_when_no_candidates() {
        let router = StubRouter;
        let decision = router
            .route("hello", &[(RoutingDomainKind::Model, vec![])])
            .await
            .unwrap();
        let model = decision.model.expect("model should be set");
        assert_eq!(model.id, "stub-model");
    }

    #[tokio::test]
    async fn stub_router_scores_all_command_candidates() {
        let router = StubRouter;
        let candidates = make_candidates(&["search", "calc", "weather"]);
        let decision = router
            .route("hello", &[(RoutingDomainKind::Commands, candidates)])
            .await
            .unwrap();
        assert_eq!(decision.commands.len(), 3);
        assert!(
            decision
                .commands
                .iter()
                .all(|c| (c.score - 0.9).abs() < 1e-6)
        );
    }

    #[tokio::test]
    async fn stub_router_produces_routing_decision_with_model_and_commands() {
        let router = StubRouter;
        let model_candidates = make_candidates(&["model-a"]);
        let cmd_candidates = make_candidates(&["cmd-x", "cmd-y"]);
        let decision = router
            .route(
                "query",
                &[
                    (RoutingDomainKind::Model, model_candidates),
                    (RoutingDomainKind::Commands, cmd_candidates),
                ],
            )
            .await
            .unwrap();
        assert!(decision.model.is_some());
        assert_eq!(decision.commands.len(), 2);
    }

    #[tokio::test]
    async fn stub_router_score_memory_candidates_returns_all_scored() {
        let router = StubRouter;
        let candidates = make_candidates(&["mem-a", "mem-b"]);
        let scored = router
            .score_memory_candidates("some text", &candidates)
            .await
            .unwrap();
        assert_eq!(scored.len(), 2);
        assert!(scored.iter().all(|s| (s.score - 0.9).abs() < 1e-6));
    }

    #[tokio::test]
    async fn error_router_returns_inference_failed() {
        let router = ErrorRouter::new("test router failure");
        let result = router
            .route("hello", &[(RoutingDomainKind::Model, vec![])])
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RouterError::InferenceFailed(_)
        ));
    }

    #[tokio::test]
    async fn error_router_score_memory_candidates_returns_error() {
        let router = ErrorRouter::new("test failure");
        let result = router.score_memory_candidates("text", &[]).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RouterError::InferenceFailed(_)
        ));
    }
}
