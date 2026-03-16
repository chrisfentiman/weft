//! Routing domain types.
//!
//! Defines the vocabulary for multi-domain semantic routing: what kinds of
//! routing decisions exist, how candidates are described, and how decisions
//! are represented.

/// A routing domain represents a class of routing decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoutingDomainKind {
    /// Which commands are relevant for this turn.
    Commands,
    /// Which model should handle this request.
    Model,
    /// Whether tools are needed at all for this turn.
    ToolNecessity,
    /// Which memory stores to query (interface only -- no stores connected).
    Memory,
}

impl RoutingDomainKind {
    /// Stable string prefix for cache keys. Do NOT use Debug output for cache keys.
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
///
/// For domains using centroid routing (Model, ToolNecessity, Memory), the
/// `examples` field contains multiple example texts whose embeddings are
/// averaged into a centroid vector.
///
/// For the Commands domain, `examples` contains a single entry:
/// `"{name}: {description}"` (matching the existing behavior).
#[derive(Debug, Clone)]
pub struct RoutingCandidate {
    /// Unique identifier within the domain.
    /// For Commands: the command name.
    /// For Model: the model routing name (e.g., "complex", "fast").
    /// For ToolNecessity: "needs_tools" or "no_tools".
    /// For Memory: the memory store name (future).
    pub id: String,
    /// Example texts for centroid embedding.
    /// For Commands: single entry `"{name}: {description}"`.
    /// For Model: the examples array from config.
    /// For ToolNecessity: representative example queries.
    pub examples: Vec<String>,
}

/// The complete routing decision for a single request.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Which commands are relevant (same semantics as old ClassificationResult).
    pub commands: Vec<ScoredCandidate>,
    /// Which model to use. None means use the default.
    pub model: Option<ScoredCandidate>,
    /// Whether tools are needed. None means undecided (use default: always inject).
    pub tools_needed: Option<bool>,
    /// Which memory stores to query. Empty for now.
    pub memory_stores: Vec<ScoredCandidate>,
}

impl RoutingDecision {
    /// Construct a fallback decision: all commands scored 1.0, default model,
    /// tools_needed=None.
    ///
    /// Fallback is used when the router cannot make a decision (model not loaded,
    /// inference failure). It is intentionally conservative:
    /// - Commands: all candidates scored 1.0 (threshold and max_commands filtering
    ///   still applies in the engine -- with threshold 0.3, all pass)
    /// - Model: None (use the default model from ProviderRegistry)
    /// - ToolNecessity: None (undecided -> inject commands as normal, conservative)
    /// - Memory: empty (no stores connected)
    ///
    /// The fallback does NOT bypass post-routing filtering.
    pub fn fallback(domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)]) -> Self {
        let mut decision = Self::empty();
        for (kind, candidates) in domains {
            if let RoutingDomainKind::Commands = kind {
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
            // Model: None (use default), ToolNecessity: None, Memory: empty
        }
        decision
    }

    /// Construct an empty decision.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_domain_kind_cache_prefix() {
        assert_eq!(RoutingDomainKind::Commands.cache_prefix(), "cmd");
        assert_eq!(RoutingDomainKind::Model.cache_prefix(), "model");
        assert_eq!(RoutingDomainKind::ToolNecessity.cache_prefix(), "tool");
        assert_eq!(RoutingDomainKind::Memory.cache_prefix(), "mem");
    }

    #[test]
    fn test_cache_prefix_no_collision() {
        // Verify that "cmd:complex" and "model:complex" don't collide
        let cmd_key = format!("{}:complex", RoutingDomainKind::Commands.cache_prefix());
        let model_key = format!("{}:complex", RoutingDomainKind::Model.cache_prefix());
        assert_ne!(cmd_key, model_key);
        assert_eq!(cmd_key, "cmd:complex");
        assert_eq!(model_key, "model:complex");
    }

    #[test]
    fn test_fallback_commands_sorted_alphabetically() {
        let domains = vec![(
            RoutingDomainKind::Commands,
            vec![
                RoutingCandidate {
                    id: "zebra".to_string(),
                    examples: vec!["zebra: Does Z".to_string()],
                },
                RoutingCandidate {
                    id: "alpha".to_string(),
                    examples: vec!["alpha: Does A".to_string()],
                },
                RoutingCandidate {
                    id: "middle".to_string(),
                    examples: vec!["middle: Does M".to_string()],
                },
            ],
        )];

        let decision = RoutingDecision::fallback(&domains);
        assert_eq!(decision.commands.len(), 3);
        assert_eq!(decision.commands[0].id, "alpha");
        assert_eq!(decision.commands[1].id, "middle");
        assert_eq!(decision.commands[2].id, "zebra");
        for c in &decision.commands {
            assert_eq!(c.score, 1.0);
        }
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
        assert!(decision.memory_stores.is_empty());
    }

    #[test]
    fn test_fallback_non_commands_domains_ignored() {
        // Model, ToolNecessity, Memory domains in fallback produce no scored output
        let domains = vec![
            (
                RoutingDomainKind::Model,
                vec![RoutingCandidate {
                    id: "fast".to_string(),
                    examples: vec!["quick answer".to_string()],
                }],
            ),
            (
                RoutingDomainKind::ToolNecessity,
                vec![RoutingCandidate {
                    id: "needs_tools".to_string(),
                    examples: vec!["search the web".to_string()],
                }],
            ),
        ];

        let decision = RoutingDecision::fallback(&domains);
        assert!(decision.commands.is_empty());
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
    }

    #[test]
    fn test_empty_decision() {
        let decision = RoutingDecision::empty();
        assert!(decision.commands.is_empty());
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
        assert!(decision.memory_stores.is_empty());
    }

    #[test]
    fn test_scored_candidate_fields() {
        let c = ScoredCandidate {
            id: "web_search".to_string(),
            score: 0.85,
        };
        assert_eq!(c.id, "web_search");
        assert!((c.score - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_routing_domain_kind_eq() {
        assert_eq!(RoutingDomainKind::Commands, RoutingDomainKind::Commands);
        assert_ne!(RoutingDomainKind::Commands, RoutingDomainKind::Model);
    }
}
