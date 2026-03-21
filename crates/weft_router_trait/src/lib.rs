//! `weft_router_trait` — Semantic router trait and domain types.
//!
//! Contains the `SemanticRouter` trait and associated types. The implementation
//! lives in `weft_router`, which depends on this crate.
//!
//! Consumers that need the trait boundary without the implementation (e.g.
//! `weft_reactor_trait`) depend on this crate directly.

use async_trait::async_trait;

// ── Domain types ───────────────────────────────────────────────────────────

/// A routing domain represents a class of routing decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoutingDomainKind {
    Commands,
    Model,
    ToolNecessity,
    Memory,
}

impl RoutingDomainKind {
    /// Stable string prefix for cache keys.
    pub fn cache_prefix(&self) -> &'static str {
        match self {
            Self::Commands => "cmd",
            Self::Model => "model",
            Self::ToolNecessity => "tool",
            Self::Memory => "mem",
        }
    }
}

/// A candidate within a routing domain.
#[derive(Debug, Clone)]
pub struct RoutingCandidate {
    pub id: String,
    pub examples: Vec<String>,
}

/// The complete routing decision for a single request.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub commands: Vec<ScoredCandidate>,
    pub model: Option<ScoredCandidate>,
    pub tools_needed: Option<bool>,
    pub memory_stores: Vec<ScoredCandidate>,
}

impl RoutingDecision {
    /// Construct a fallback decision: all commands scored 1.0, default model.
    pub fn fallback(domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)]) -> Self {
        let mut decision = Self::empty();
        for (kind, candidates) in domains {
            match kind {
                RoutingDomainKind::Commands => {
                    let mut scored: Vec<ScoredCandidate> = candidates
                        .iter()
                        .map(|c| ScoredCandidate {
                            id: c.id.clone(),
                            score: 1.0,
                        })
                        .collect();
                    scored.sort_by(|a, b| a.id.cmp(&b.id));
                    decision.commands = scored;
                }
                RoutingDomainKind::Memory => {
                    let mut scored: Vec<ScoredCandidate> = candidates
                        .iter()
                        .map(|c| ScoredCandidate {
                            id: c.id.clone(),
                            score: 1.0,
                        })
                        .collect();
                    scored.sort_by(|a, b| a.id.cmp(&b.id));
                    decision.memory_stores = scored;
                }
                RoutingDomainKind::Model | RoutingDomainKind::ToolNecessity => {}
            }
        }
        decision
    }

    pub fn empty() -> Self {
        Self {
            commands: Vec::new(),
            model: None,
            tools_needed: None,
            memory_stores: Vec::new(),
        }
    }
}

/// A candidate with its similarity score.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub id: String,
    pub score: f32,
}

// ── RouterError ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("model inference failed: {0}")]
    InferenceFailed(String),
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
    #[error("model not loaded")]
    ModelNotLoaded,
}

// ── SemanticRouter trait ───────────────────────────────────────────────────

/// Semantic router that makes all routing decisions for a request.
///
/// `Send + Sync + 'static`: shared via Arc, used from async handlers.
#[async_trait]
pub trait SemanticRouter: Send + Sync + 'static {
    async fn route(
        &self,
        user_message: &str,
        domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, RouterError>;

    async fn score_memory_candidates(
        &self,
        text: &str,
        candidates: &[RoutingCandidate],
    ) -> Result<Vec<ScoredCandidate>, RouterError>;
}
