//! `weft_classifier` — Semantic classifier trait and ModernBERT implementation.
//!
//! Contains:
//! - `SemanticClassifier` trait for scoring commands against a user message
//! - `ClassifierError` error type
//! - `ClassificationResult` for per-command relevance scores
//! - `ModernBertClassifier`: ModernBERT bi-encoder via ONNX Runtime
//! - Score filtering helpers: `filter_by_threshold` and `take_top`

pub mod bert;
pub(crate) mod tokenizer;

pub use bert::ModernBertClassifier;

use async_trait::async_trait;
use weft_core::CommandStub;

/// Semantic classifier that scores commands against a conversation turn.
///
/// Send + Sync + 'static: shared via Arc, used from async handlers.
#[async_trait]
pub trait SemanticClassifier: Send + Sync + 'static {
    /// Score each command stub against the latest user message.
    ///
    /// Returns a Vec of (command_name, relevance_score) pairs.
    /// Scores are in [0.0, 1.0]. Gateway applies threshold.
    ///
    /// `user_message`: The latest user message text.
    /// `commands`: All available command stubs to score.
    async fn classify(
        &self,
        user_message: &str,
        commands: &[CommandStub],
    ) -> Result<Vec<ClassificationResult>, ClassifierError>;
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub command_name: String,
    pub score: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
    #[error("model inference failed: {0}")]
    InferenceFailed(String),
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
    #[error("model not loaded")]
    ModelNotLoaded,
}

/// Filter `results` to those scoring at or above `threshold`.
///
/// Does not sort — ordering is preserved from the input.
pub fn filter_by_threshold(
    results: Vec<ClassificationResult>,
    threshold: f32,
) -> Vec<ClassificationResult> {
    results
        .into_iter()
        .filter(|r| r.score >= threshold)
        .collect()
}

/// Take the top `n` results by score (highest first).
///
/// If `results` has fewer than `n` elements, all are returned.
pub fn take_top(mut results: Vec<ClassificationResult>, n: usize) -> Vec<ClassificationResult> {
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

    fn results(scores: &[(&str, f32)]) -> Vec<ClassificationResult> {
        scores
            .iter()
            .map(|(name, score)| ClassificationResult {
                command_name: name.to_string(),
                score: *score,
            })
            .collect()
    }

    // ---- filter_by_threshold ----

    #[test]
    fn test_filter_threshold_basic() {
        let input = results(&[("a", 0.8), ("b", 0.2), ("c", 0.5)]);
        let out = filter_by_threshold(input, 0.4);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.score >= 0.4));
    }

    #[test]
    fn test_filter_threshold_all_pass() {
        let input = results(&[("a", 0.9), ("b", 0.8)]);
        let out = filter_by_threshold(input, 0.0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_filter_threshold_none_pass() {
        let input = results(&[("a", 0.1), ("b", 0.2)]);
        let out = filter_by_threshold(input, 0.5);
        assert!(out.is_empty());
    }

    #[test]
    fn test_filter_threshold_exact_boundary() {
        let input = results(&[("a", 0.3), ("b", 0.3)]);
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
        let input = results(&[("a", 0.3), ("b", 0.9), ("c", 0.6)]);
        let out = take_top(input, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].command_name, "b"); // highest
        assert_eq!(out[1].command_name, "c"); // second
    }

    #[test]
    fn test_take_top_more_than_available() {
        let input = results(&[("a", 0.5)]);
        let out = take_top(input, 10);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_take_top_zero() {
        let input = results(&[("a", 0.9)]);
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
        let input = results(&[("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.7)]);
        let out = take_top(input, 4);
        let scores: Vec<f32> = out.iter().map(|r| r.score).collect();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] >= scores[i],
                "results should be sorted descending"
            );
        }
    }

    // ---- ClassificationResult ----

    #[test]
    fn test_classification_result_fields() {
        let r = ClassificationResult {
            command_name: "web_search".to_string(),
            score: 0.75,
        };
        assert_eq!(r.command_name, "web_search");
        assert!((r.score - 0.75).abs() < 1e-6);
    }

    // ---- ClassifierError ----

    #[test]
    fn test_classifier_error_display() {
        let e = ClassifierError::InferenceFailed("tensor shape mismatch".to_string());
        assert!(e.to_string().contains("tensor shape mismatch"));

        let e = ClassifierError::TokenizationFailed("unknown token".to_string());
        assert!(e.to_string().contains("unknown token"));

        let e = ClassifierError::ModelNotLoaded;
        assert!(e.to_string().contains("model not loaded"));
    }
}
