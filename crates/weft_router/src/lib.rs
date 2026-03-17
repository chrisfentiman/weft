//! `weft_router` — Semantic router trait, ModernBERT implementation, and pure routing service.
//!
//! Contains:
//! - `SemanticRouter` trait for making all routing decisions for a request
//! - `RouterError` error type
//! - `RoutingDomainKind`, `RoutingCandidate`, `RoutingDecision`, `ScoredCandidate` domain types
//! - `ModernBertRouter`: ModernBERT bi-encoder via ONNX Runtime
//! - Score filtering helpers: `filter_by_threshold` and `take_top`
//! - `routing_service`: Pure routing logic (`route_domains`, candidate builders, `RoutingResult`)

pub mod bert;
pub mod domain;
pub mod routing_service;
pub(crate) mod tokenizer;

pub use bert::ModernBertRouter;
pub use domain::{RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate};
pub use routing_service::{
    MemoryCandidates, MemoryStoreRef, RoutingInput, RoutingResult, build_memory_candidates,
    build_model_candidates, route_domains, tool_necessity_candidates,
};

use async_trait::async_trait;

/// Semantic router that makes all routing decisions for a request.
///
/// Send + Sync + 'static: shared via Arc, used from async handlers.
#[async_trait]
pub trait SemanticRouter: Send + Sync + 'static {
    /// Make all routing decisions for a user message across all configured domains.
    ///
    /// `user_message`: The latest user message text.
    /// `domains`: Slice of (domain kind, candidates) pairs. Only domains present are scored.
    ///
    /// Returns a RoutingDecision with scored results per domain.
    async fn route(
        &self,
        user_message: &str,
        domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, RouterError>;

    /// Score memory store candidates against the given text.
    ///
    /// Used for per-invocation routing by both `/recall` (with the query argument)
    /// and `/remember` (with the content argument). Each invocation routes
    /// independently based on its own argument content, not the user's original message.
    ///
    /// Returns scored candidates (unsorted, unfiltered — caller applies threshold and
    /// capability filtering). Returns `Err(RouterError::ModelNotLoaded)` if the
    /// embedding model is unavailable (fallback mode).
    async fn score_memory_candidates(
        &self,
        text: &str,
        candidates: &[RoutingCandidate],
    ) -> Result<Vec<ScoredCandidate>, RouterError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("model inference failed: {0}")]
    InferenceFailed(String),
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
    #[error("model not loaded")]
    ModelNotLoaded,
}

/// Filter scored candidates to those scoring at or above `threshold`.
///
/// Does not sort — ordering is preserved from the input.
pub fn filter_by_threshold(results: Vec<ScoredCandidate>, threshold: f32) -> Vec<ScoredCandidate> {
    results
        .into_iter()
        .filter(|r| r.score >= threshold)
        .collect()
}

/// Take the top `n` candidates by score (highest first).
///
/// If `results` has fewer than `n` elements, all are returned.
pub fn take_top(mut results: Vec<ScoredCandidate>, n: usize) -> Vec<ScoredCandidate> {
    // Sort descending by score. NaN scores sort last (treated as 0).
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(n);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(pairs: &[(&str, f32)]) -> Vec<ScoredCandidate> {
        pairs
            .iter()
            .map(|(id, score)| ScoredCandidate {
                id: id.to_string(),
                score: *score,
            })
            .collect()
    }

    // ---- filter_by_threshold ----

    #[test]
    fn test_filter_threshold_basic() {
        let input = scored(&[("a", 0.8), ("b", 0.2), ("c", 0.5)]);
        let out = filter_by_threshold(input, 0.4);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.score >= 0.4));
    }

    #[test]
    fn test_filter_threshold_all_pass() {
        let input = scored(&[("a", 0.9), ("b", 0.8)]);
        let out = filter_by_threshold(input, 0.0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_filter_threshold_none_pass() {
        let input = scored(&[("a", 0.1), ("b", 0.2)]);
        let out = filter_by_threshold(input, 0.5);
        assert!(out.is_empty());
    }

    #[test]
    fn test_filter_threshold_exact_boundary() {
        let input = scored(&[("a", 0.3), ("b", 0.3)]);
        // threshold is inclusive
        let out = filter_by_threshold(input, 0.3);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_filter_threshold_empty_input() {
        let out = filter_by_threshold(vec![], 0.5);
        assert!(out.is_empty());
    }

    // ---- take_top ----

    #[test]
    fn test_take_top_basic() {
        let input = scored(&[("a", 0.3), ("b", 0.9), ("c", 0.6)]);
        let out = take_top(input, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "b"); // highest
        assert_eq!(out[1].id, "c"); // second
    }

    #[test]
    fn test_take_top_more_than_available() {
        let input = scored(&[("a", 0.5)]);
        let out = take_top(input, 10);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_take_top_zero() {
        let input = scored(&[("a", 0.9)]);
        let out = take_top(input, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_take_top_empty_input() {
        let out = take_top(vec![], 5);
        assert!(out.is_empty());
    }

    #[test]
    fn test_take_top_sorted_descending() {
        let input = scored(&[("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.7)]);
        let out = take_top(input, 4);
        let scores: Vec<f32> = out.iter().map(|r| r.score).collect();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] >= scores[i],
                "results should be sorted descending"
            );
        }
    }

    // ---- ScoredCandidate ----

    #[test]
    fn test_scored_candidate_fields() {
        let r = ScoredCandidate {
            id: "web_search".to_string(),
            score: 0.75,
        };
        assert_eq!(r.id, "web_search");
        assert!((r.score - 0.75).abs() < 1e-6);
    }

    // ---- RouterError ----

    #[test]
    fn test_router_error_display() {
        let e = RouterError::InferenceFailed("tensor shape mismatch".to_string());
        assert!(e.to_string().contains("tensor shape mismatch"));

        let e = RouterError::TokenizationFailed("unknown token".to_string());
        assert!(e.to_string().contains("unknown token"));

        let e = RouterError::ModelNotLoaded;
        assert!(e.to_string().contains("model not loaded"));
    }
}
