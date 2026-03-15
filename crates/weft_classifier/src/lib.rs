//! `weft_classifier` — Semantic classifier trait and ModernBERT implementation.
//!
//! Phase 1 stub. Full implementation in Phase 3.
//!
//! Contains:
//! - `SemanticClassifier` trait for scoring commands against a user message
//! - `ClassifierError` error type
//! - `ClassificationResult` for per-command relevance scores
//! - ModernBERT bi-encoder via ONNX Runtime (Phase 3)

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
