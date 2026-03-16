//! ModernBERT bi-encoder via ONNX Runtime.
//!
//! Loads a ModernBERT ONNX model and tokenizer, pre-computes candidate embeddings
//! at construction time if provided, and scores user messages against those
//! embeddings via cosine similarity at query time. Unknown candidates are embedded
//! lazily during `route()` and cached for subsequent calls.
//!
//! The bi-encoder approach embeds query and candidates separately:
//! - Candidate embeddings: computed lazily on first encounter, then cached.
//! - Query embedding: computed per request (~20-50ms CPU).
//! - Similarity: dot product of L2-normalized vectors (cosine similarity).
//!
//! Fallback: if the ONNX model or tokenizer fails to load, the router
//! returns all commands unfiltered (score 1.0). This keeps the gateway
//! functional even when the model is misconfigured.
//!
//! **Phase 1 note:** This implementation wraps the existing classify logic to handle
//! the Commands domain, producing a RoutingDecision with only `commands` populated.
//! Full multi-domain routing (Model, ToolNecessity, Memory) with centroid embeddings
//! is implemented in Phase 3.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use ort::inputs;
use ort::session::Session;
use ort::value::Value;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::tokenizer::BertTokenizer;
use crate::{
    RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
    SemanticRouter,
};

/// Combined session and embedding cache, held inside a single `Mutex`.
///
/// Collocating the cache with the session avoids a second lock: cache writes
/// happen inside inference calls, which already hold the session mutex.
struct SessionState {
    session: Session,
    /// L2-normalized candidate embeddings keyed by cache key (`"{prefix}:{id}"`).
    embeddings: HashMap<String, Vec<f32>>,
}

/// ModernBERT bi-encoder router.
///
/// Wraps an ONNX session and a HuggingFace tokenizer. Builds L2-normalized
/// embeddings for routing candidates lazily during `route()` (or eagerly at
/// construction time if candidates are provided). At query time, embeds the
/// user message and computes cosine similarity against cached candidate embeddings.
///
/// Thread-safe: `Session` is protected by a `tokio::sync::Mutex`. The
/// tokenizer is `Send + Sync` and needs no locking.
pub struct ModernBertRouter {
    /// ONNX Runtime session plus embedding cache, or `None` in fallback mode.
    ///
    /// Both the session and the cache are behind a single Mutex because cache
    /// writes require inference, which requires holding the session lock anyway.
    state: Option<Arc<Mutex<SessionState>>>,
    /// Tokenizer, or `None` if it failed to load (fallback mode).
    tokenizer: Option<Arc<BertTokenizer>>,
}

impl ModernBertRouter {
    /// Build a router, loading the ONNX model and tokenizer from disk.
    ///
    /// If `pre_embed` is non-empty, candidate embeddings are pre-computed now as a
    /// warm-start optimisation. If empty (`&[]`), the cache starts empty and
    /// candidates are embedded lazily on the first `route()` call.
    ///
    /// If the model or tokenizer cannot be loaded, logs a warning and constructs
    /// the router in fallback mode (no filtering, returns all commands with score 1.0).
    ///
    /// The constructor is `async` because it must use `.lock().await` on the
    /// session mutex. `blocking_lock()` panics when called from within the tokio
    /// runtime, and this constructor is called from `#[tokio::main]`.
    ///
    /// `model_path`: Path to the ONNX model file.
    /// `tokenizer_path`: Path to the HuggingFace `tokenizer.json` file.
    /// `pre_embed`: Candidates to pre-embed at startup. Pass `&[]` to skip.
    pub async fn new(
        model_path: &str,
        tokenizer_path: &str,
        pre_embed: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Self {
        let tokenizer = match BertTokenizer::from_file(tokenizer_path) {
            Ok(t) => {
                debug!("tokenizer loaded from '{tokenizer_path}'");
                Some(Arc::new(t))
            }
            Err(e) => {
                warn!("router falling back to passthrough: {e}");
                None
            }
        };

        let session = if tokenizer.is_some() {
            match Session::builder().and_then(|mut b: ort::session::builder::SessionBuilder| {
                b.commit_from_file(model_path)
            }) {
                Ok(s) => {
                    debug!("ONNX model loaded from '{model_path}'");
                    Some(Arc::new(Mutex::new(SessionState {
                        session: s,
                        embeddings: HashMap::new(),
                    })))
                }
                Err(e) => {
                    warn!("router falling back to passthrough: ONNX load failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Pre-embed any candidates provided at construction time.
        // The session mutex is freshly created and uncontested here, but we use
        // `.lock().await` because the constructor is called from within the tokio
        // runtime. `blocking_lock()` would panic in that context.
        if let (Some(state), Some(tok)) = (&session, &tokenizer) {
            let mut guard = state.lock().await;
            let mut count = 0;
            for (domain_kind, candidates) in pre_embed {
                for candidate in candidates {
                    let cache_key = format!("{}:{}", domain_kind.cache_prefix(), candidate.id);
                    // For Phase 1: pre-embed using the first example as the
                    // representative text (single-example behavior).
                    // Phase 3 replaces this with full centroid computation.
                    if let Some(text) = candidate.examples.first() {
                        match embed_text(tok, &mut guard.session, text) {
                            Ok(mut v) => {
                                l2_normalize(&mut v);
                                guard.embeddings.insert(cache_key, v);
                                count += 1;
                            }
                            Err(e) => {
                                warn!("failed to pre-embed candidate '{}': {e}", candidate.id);
                            }
                        }
                    }
                }
            }
            if count > 0 {
                debug!("pre-computed embeddings for {count} candidates");
            }
        }

        Self {
            state: session,
            tokenizer,
        }
    }

    /// Returns `true` if the router is in fallback mode (no model loaded).
    pub fn is_fallback(&self) -> bool {
        self.state.is_none()
    }

    /// Returns the number of cached embeddings.
    ///
    /// In fallback mode this is always 0.
    #[cfg(test)]
    pub async fn cached_count(&self) -> usize {
        match &self.state {
            None => 0,
            Some(state) => state.lock().await.embeddings.len(),
        }
    }
}

#[async_trait]
impl SemanticRouter for ModernBertRouter {
    /// Route a user message against all provided domains.
    ///
    /// **Phase 1 implementation:** Handles the Commands domain using the existing
    /// per-command embedding logic. For Model, ToolNecessity, and Memory domains,
    /// falls back to conservative defaults (None/empty). Full multi-domain centroid
    /// routing is implemented in Phase 3.
    async fn route(
        &self,
        user_message: &str,
        domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
    ) -> Result<RoutingDecision, RouterError> {
        // Fallback: model not loaded — return fallback decision.
        let (state, tokenizer) = match (&self.state, &self.tokenizer) {
            (Some(s), Some(t)) => (s, t),
            _ => {
                debug!("router in fallback mode, returning fallback decision");
                return Ok(RoutingDecision::fallback(domains));
            }
        };

        let mut guard = state.lock().await;

        // Embed the user message once.
        let mut query_vec = embed_text(tokenizer, &mut guard.session, user_message)
            .map_err(|e| RouterError::InferenceFailed(format!("query embedding: {e}")))?;
        l2_normalize(&mut query_vec);

        let mut decision = RoutingDecision::empty();

        for (domain_kind, candidates) in domains {
            match domain_kind {
                RoutingDomainKind::Commands => {
                    // Lazily embed any commands not yet in cache.
                    for candidate in candidates {
                        let cache_key = format!("{}:{}", domain_kind.cache_prefix(), candidate.id);
                        if !guard.embeddings.contains_key(&cache_key) {
                            // For Commands: use the first example as the representative text.
                            if let Some(text) = candidate.examples.first() {
                                match embed_text(tokenizer, &mut guard.session, text) {
                                    Ok(mut v) => {
                                        l2_normalize(&mut v);
                                        guard.embeddings.insert(cache_key, v);
                                        debug!("lazily embedded command '{}'", candidate.id);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "failed to lazily embed command '{}': {e}",
                                            candidate.id
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Score all commands.
                    let scored: Vec<ScoredCandidate> = candidates
                        .iter()
                        .map(|c| {
                            let cache_key = format!("{}:{}", domain_kind.cache_prefix(), c.id);
                            let score = guard
                                .embeddings
                                .get(&cache_key)
                                .map(|emb| cosine_similarity(&query_vec, emb))
                                .unwrap_or(0.0);
                            ScoredCandidate {
                                id: c.id.clone(),
                                score,
                            }
                        })
                        .collect();

                    decision.commands = scored;
                }
                // Phase 1: Model, ToolNecessity, Memory are not yet implemented.
                // These domains produce no output here. Full implementation in Phase 3.
                RoutingDomainKind::Model
                | RoutingDomainKind::ToolNecessity
                | RoutingDomainKind::Memory => {}
            }
        }

        Ok(decision)
    }
}

/// Run a single BERT embedding pass.
///
/// Tokenizes `text`, runs it through the ONNX session, and extracts the
/// `[CLS]` token's hidden state (index 0 across the hidden dimension) as
/// the sentence embedding.
fn embed_text(
    tokenizer: &BertTokenizer,
    session: &mut Session,
    text: &str,
) -> Result<Vec<f32>, String> {
    let (input_ids_raw, attention_mask_raw) = tokenizer.encode(text).map_err(|e| e.to_string())?;

    let seq_len = input_ids_raw.len();

    // Build ndarray tensors with shape [1, seq_len].
    let ids_array = ndarray::Array2::from_shape_vec((1, seq_len), input_ids_raw)
        .map_err(|e| format!("input_ids shape error: {e}"))?;
    let mask_array = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask_raw)
        .map_err(|e| format!("attention_mask shape error: {e}"))?;

    // Wrap ndarray arrays into ort Values.
    // Value::from_array returns Value<TensorValueType<T>>; .into() erases to Value<DynValueTypeMarker>.
    let ids_value: Value = Value::from_array(ids_array)
        .map_err(|e| format!("input_ids value error: {e}"))?
        .into();
    let mask_value: Value = Value::from_array(mask_array)
        .map_err(|e| format!("attention_mask value error: {e}"))?
        .into();

    // Run inference.
    let outputs = session
        .run(inputs![
            "input_ids" => ids_value,
            "attention_mask" => mask_value,
        ])
        .map_err(|e| format!("ONNX inference failed: {e}"))?;

    // Extract the first output tensor (last_hidden_state or pooler_output).
    let tensor = if let Some(v) = outputs.get("last_hidden_state") {
        v.try_extract_tensor::<f32>()
            .map_err(|e| format!("failed to extract last_hidden_state: {e}"))?
    } else if let Some(v) = outputs.get("pooler_output") {
        v.try_extract_tensor::<f32>()
            .map_err(|e| format!("failed to extract pooler_output: {e}"))?
    } else {
        // Fall back to the first output by index.
        outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("failed to extract output[0]: {e}"))?
    };

    // try_extract_tensor returns (&ort::value::Shape, &[T]).
    let (shape, data) = tensor;

    // ort::value::Shape derefs to &[i64].
    let dims: &[i64] = shape;

    // Handle both output shapes:
    // - [1, hidden_size]: pooled output (take directly)
    // - [1, seq_len, hidden_size]: last_hidden_state (take row 0 = [CLS] token)
    let embedding: Vec<f32> = if dims.len() == 2 {
        // Pooled output: [1, hidden_size]
        data.to_vec()
    } else if dims.len() == 3 {
        // Last hidden state: [1, seq_len, hidden_size] — take [CLS] at position 0.
        let hidden_size = dims[2] as usize;
        data.iter().take(hidden_size).copied().collect()
    } else {
        return Err(format!(
            "unexpected output tensor rank: {} (expected 2 or 3)",
            dims.len()
        ));
    };

    Ok(embedding)
}

/// L2-normalize a vector in place.
///
/// If the norm is zero (zero vector), the vector is left unchanged to avoid
/// division by zero.
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Cosine similarity between two L2-normalized vectors.
///
/// Both vectors must be pre-normalized (unit length). Under that precondition,
/// cosine similarity equals the dot product.
///
/// Returns a value in [-1.0, 1.0].
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // If dimensions differ, return 0 (no similarity) rather than panic.
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_commands_domain(names: &[(&str, &str)]) -> (RoutingDomainKind, Vec<RoutingCandidate>) {
        (
            RoutingDomainKind::Commands,
            names
                .iter()
                .map(|(n, d)| RoutingCandidate {
                    id: n.to_string(),
                    examples: vec![format!("{n}: {d}")],
                })
                .collect(),
        )
    }

    // ---- L2 normalization ----

    #[test]
    fn test_l2_normalize_unit_length() {
        let mut v = vec![3.0_f32, 4.0_f32];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
    }

    #[test]
    fn test_l2_normalize_already_unit() {
        let mut v = vec![1.0_f32, 0.0_f32, 0.0_f32];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0_f32, 0.0_f32];
        l2_normalize(&mut v);
        // Zero vector should remain unchanged (no division by zero).
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_negative_components() {
        let mut v = vec![-3.0_f32, 4.0_f32];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    // ---- Cosine similarity ----

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f32, 0.0_f32, 0.0_f32];
        let score = cosine_similarity(&v, &v);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "identical vectors should score 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0_f32];
        let b = vec![0.0_f32, 1.0_f32];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6, "orthogonal vectors should score 0.0");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0_f32, 0.0_f32];
        let b = vec![-1.0_f32, 0.0_f32];
        let score = cosine_similarity(&a, &b);
        assert!(
            (score + 1.0).abs() < 1e-6,
            "opposite vectors should score -1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = vec![1.0_f32, 0.0_f32];
        let b = vec![1.0_f32, 0.0_f32, 0.0_f32];
        let score = cosine_similarity(&a, &b);
        assert_eq!(score, 0.0, "mismatched dimensions should return 0.0");
    }

    #[test]
    fn test_cosine_similarity_partial() {
        // Both normalized: [0.6, 0.8] · [1.0, 0.0] = 0.6
        let a = vec![0.6_f32, 0.8_f32];
        let b = vec![1.0_f32, 0.0_f32];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.6).abs() < 1e-6);
    }

    // ---- Fallback behavior ----

    #[tokio::test]
    async fn test_fallback_when_model_missing() {
        let (domain, candidates) = make_commands_domain(&[
            ("web_search", "Search the web"),
            ("recall", "Retrieve from memory"),
        ]);

        let router = ModernBertRouter::new(
            "/nonexistent/path/model.onnx",
            "/nonexistent/path/tokenizer.json",
            &[],
        )
        .await;

        assert!(
            router.is_fallback(),
            "should be in fallback mode when model is missing"
        );

        let domains = vec![(domain, candidates)];
        let decision = router
            .route("find something", &domains)
            .await
            .expect("route should succeed in fallback mode");

        assert_eq!(
            decision.commands.len(),
            2,
            "should return all commands in fallback"
        );
        for c in &decision.commands {
            assert_eq!(c.score, 1.0, "fallback scores should be 1.0");
        }
    }

    #[tokio::test]
    async fn test_fallback_returns_all_commands() {
        let (domain, candidates) = make_commands_domain(&[
            ("cmd_a", "Does A"),
            ("cmd_b", "Does B"),
            ("cmd_c", "Does C"),
        ]);

        let router = ModernBertRouter::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        let decision = router
            .route("some query", &[(domain, candidates)])
            .await
            .expect("fallback should not error");

        let ids: Vec<&str> = decision.commands.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"cmd_a"));
        assert!(ids.contains(&"cmd_b"));
        assert!(ids.contains(&"cmd_c"));
    }

    #[tokio::test]
    async fn test_fallback_empty_commands() {
        let router = ModernBertRouter::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        let decision = router
            .route("something", &[(RoutingDomainKind::Commands, vec![])])
            .await
            .expect("empty commands should not error");

        assert!(decision.commands.is_empty());
    }

    #[tokio::test]
    async fn test_fallback_scores_all_commands_1_0() {
        let router = ModernBertRouter::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        let candidates = vec![RoutingCandidate {
            id: "unknown_extra".to_string(),
            examples: vec!["unknown_extra: Not in cache".to_string()],
        }];

        let decision = router
            .route("something", &[(RoutingDomainKind::Commands, candidates)])
            .await
            .expect("should not error");

        assert_eq!(decision.commands.len(), 1);
        assert_eq!(decision.commands[0].score, 1.0);
    }

    #[tokio::test]
    async fn test_fallback_results_sorted_alphabetically() {
        let (domain, candidates) = make_commands_domain(&[
            ("zebra", "Last alphabetically"),
            ("alpha", "First alphabetically"),
            ("middle", "Middle alphabetically"),
        ]);

        let router = ModernBertRouter::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        let decision = router
            .route("anything", &[(domain, candidates)])
            .await
            .expect("fallback should not error");

        let ids: Vec<&str> = decision.commands.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["alpha", "middle", "zebra"],
            "fallback results must be sorted alphabetically"
        );
    }

    #[tokio::test]
    async fn test_cache_starts_empty() {
        let router = ModernBertRouter::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;
        assert_eq!(router.cached_count().await, 0);
    }
}
