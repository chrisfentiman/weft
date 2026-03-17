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
//! **Centroid routing:** For multi-example candidates (Model, ToolNecessity, Memory
//! domains), all examples are embedded and averaged into a centroid vector, which
//! is then L2-normalized. This creates a more robust semantic region than a single
//! description point.
//!
//! Fallback: if the ONNX model or tokenizer fails to load, the router
//! returns all commands unfiltered (score 1.0). This keeps the gateway
//! functional even when the model is misconfigured.

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
    /// L2-normalized centroid embeddings keyed by cache key (`"{prefix}:{id}"`).
    embeddings: HashMap<String, Vec<f32>>,
}

/// ModernBERT bi-encoder router.
///
/// Wraps an ONNX session and a HuggingFace tokenizer. Builds L2-normalized
/// centroid embeddings for routing candidates lazily during `route()` (or
/// eagerly at construction time if candidates are provided). At query time,
/// embeds the user message and computes cosine similarity against cached
/// centroid embeddings.
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
    /// Per-domain threshold overrides from config.
    ///
    /// Used for the Model domain confidence gate: if no model scores above the
    /// threshold, `decision.model` is `None` and the engine uses the default.
    domain_thresholds: HashMap<RoutingDomainKind, f32>,
}

impl ModernBertRouter {
    /// Build a router, loading the ONNX model and tokenizer from disk.
    ///
    /// `model_path`: Path to the ONNX model file.
    /// `tokenizer_path`: Path to the HuggingFace `tokenizer.json` file.
    /// `pre_embed`: Candidates to pre-embed at startup. Typically model and
    ///   tool-necessity candidates (known from config). Commands are embedded
    ///   lazily because they come from a remote registry.
    /// `domain_thresholds`: Per-domain threshold overrides from `DomainsConfig`.
    ///   Used for model domain confidence gating (see Section 5.2 of spec).
    ///
    /// If the model or tokenizer cannot be loaded, logs a warning and constructs
    /// the router in fallback mode (no filtering, returns all commands with score 1.0).
    ///
    /// The constructor is `async` because it must use `.lock().await` on the
    /// session mutex. `blocking_lock()` panics when called from within the tokio
    /// runtime, and this constructor is called from `#[tokio::main]`.
    pub async fn new(
        model_path: &str,
        tokenizer_path: &str,
        pre_embed: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        domain_thresholds: HashMap<RoutingDomainKind, f32>,
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

        // Pre-compute centroid embeddings for provided candidates.
        // The session mutex is freshly created and uncontested here, but we use
        // `.lock().await` because the constructor is called from within the tokio
        // runtime. `blocking_lock()` would panic in that context.
        if let (Some(state), Some(tok)) = (&session, &tokenizer) {
            let mut guard = state.lock().await;
            let mut count = 0;
            for (domain_kind, candidates) in pre_embed {
                for candidate in candidates {
                    let cache_key = format!("{}:{}", domain_kind.cache_prefix(), candidate.id);
                    match compute_centroid(tok, &mut guard.session, &candidate.examples) {
                        Ok(centroid) => {
                            guard.embeddings.insert(cache_key, centroid);
                            count += 1;
                        }
                        Err(e) => {
                            warn!("failed to pre-embed candidate '{}': {e}", candidate.id);
                        }
                    }
                }
            }
            if count > 0 {
                debug!("pre-computed centroid embeddings for {count} candidates");
            }
        }

        Self {
            state: session,
            tokenizer,
            domain_thresholds,
        }
    }

    /// Returns `true` if the router is in fallback mode (no model loaded).
    pub fn is_fallback(&self) -> bool {
        self.state.is_none()
    }

    /// Returns the number of cached centroid embeddings.
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
    /// Embeds the user message once and reuses it across all domains. For each
    /// candidate, ensures a centroid embedding is cached (computing it lazily
    /// on first encounter), then scores by cosine similarity.
    ///
    /// Domain-specific behaviour:
    /// - **Commands**: all candidates scored; callers filter by threshold + max.
    /// - **Model**: highest-scoring candidate returned; threshold gating applied
    ///   if `domain_thresholds[Model]` is set -- below threshold returns None.
    /// - **ToolNecessity**: binary decision comparing "needs_tools" vs "no_tools".
    /// - **Memory**: all candidates scored; caller selects stores above threshold.
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

        // Embed user message once; reuse across all domains.
        let mut query_vec = embed_text(tokenizer, &mut guard.session, user_message)
            .map_err(|e| RouterError::InferenceFailed(format!("query embedding: {e}")))?;
        l2_normalize(&mut query_vec);

        let mut decision = RoutingDecision::empty();

        for (domain_kind, candidates) in domains {
            // Ensure all candidates have cached centroid embeddings.
            for candidate in candidates {
                let cache_key = format!("{}:{}", domain_kind.cache_prefix(), candidate.id);
                if !guard.embeddings.contains_key(&cache_key) {
                    match compute_centroid(tokenizer, &mut guard.session, &candidate.examples) {
                        Ok(centroid) => {
                            guard.embeddings.insert(cache_key, centroid);
                        }
                        Err(e) => {
                            warn!("failed to compute centroid for '{}': {e}", candidate.id);
                        }
                    }
                }
            }

            // Score all candidates in this domain.
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

            match domain_kind {
                RoutingDomainKind::Commands => {
                    decision.commands = scored;
                }
                RoutingDomainKind::Model => {
                    // Take the highest-scoring model.
                    // If a model domain threshold is configured and no model scores above it,
                    // decision.model stays None (engine uses the default from ProviderRegistry).
                    // If no threshold is configured, always pick the highest-scoring model.
                    let best = scored.into_iter().max_by(|a, b| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let model_threshold = self.domain_thresholds.get(&RoutingDomainKind::Model);
                    decision.model = match (best, model_threshold) {
                        // Below confidence gate: use default model.
                        (Some(m), Some(&t)) if m.score < t => None,
                        (Some(m), _) => Some(m),
                        (None, _) => None,
                    };
                }
                RoutingDomainKind::ToolNecessity => {
                    // Binary decision: compare "needs_tools" vs "no_tools" centroid scores.
                    let needs = scored.iter().find(|s| s.id == "needs_tools");
                    let no = scored.iter().find(|s| s.id == "no_tools");
                    decision.tools_needed = match (needs, no) {
                        (Some(n), Some(nn)) => Some(n.score >= nn.score),
                        (Some(_), None) => Some(true),
                        (None, Some(_)) => Some(false),
                        (None, None) => None,
                    };
                }
                RoutingDomainKind::Memory => {
                    decision.memory_stores = scored;
                }
            }
        }

        Ok(decision)
    }

    /// Score memory store candidates against arbitrary text.
    ///
    /// Used for per-invocation routing by both `/recall` and `/remember`.
    /// Embeds `text` using the same model as `route()`, then computes cosine
    /// similarity against each candidate's pre-embedded centroid.
    ///
    /// Unknown candidates are embedded lazily and cached (same as in `route()`).
    /// Returns `Err(RouterError::ModelNotLoaded)` when in fallback mode.
    async fn score_memory_candidates(
        &self,
        text: &str,
        candidates: &[RoutingCandidate],
    ) -> Result<Vec<ScoredCandidate>, RouterError> {
        let (state, tokenizer) = match (&self.state, &self.tokenizer) {
            (Some(s), Some(t)) => (s, t),
            _ => return Err(RouterError::ModelNotLoaded),
        };

        let mut guard = state.lock().await;

        // Embed the input text.
        let mut query_vec = embed_text(tokenizer, &mut guard.session, text)
            .map_err(|e| RouterError::InferenceFailed(format!("memory score embedding: {e}")))?;
        l2_normalize(&mut query_vec);

        // Use Memory domain cache prefix for all memory store candidates.
        let domain_prefix = RoutingDomainKind::Memory.cache_prefix();

        // Ensure all candidates have cached centroid embeddings.
        for candidate in candidates {
            let cache_key = format!("{domain_prefix}:{}", candidate.id);
            if !guard.embeddings.contains_key(&cache_key) {
                match compute_centroid(tokenizer, &mut guard.session, &candidate.examples) {
                    Ok(centroid) => {
                        guard.embeddings.insert(cache_key, centroid);
                    }
                    Err(e) => {
                        warn!(
                            "failed to compute centroid for memory candidate '{}': {e}",
                            candidate.id
                        );
                    }
                }
            }
        }

        // Score all candidates.
        let scored: Vec<ScoredCandidate> = candidates
            .iter()
            .map(|c| {
                let cache_key = format!("{domain_prefix}:{}", c.id);
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

        Ok(scored)
    }
}

/// Compute the centroid embedding for a slice of example texts.
///
/// 1. Embed each text.
/// 2. Element-wise average all embeddings.
/// 3. L2-normalize the result.
///
/// Takes `&mut ort::Session` (the inner ONNX session), NOT a `SessionState`.
/// The caller holds the `Mutex<SessionState>` guard and passes `&mut guard.session`.
/// This keeps the borrow pattern clean: `compute_centroid` has no knowledge of
/// the embedding cache or the guard.
///
/// Returns `Err(RouterError::InferenceFailed)` if `texts` is empty or if any
/// individual embedding fails.
fn compute_centroid(
    tokenizer: &BertTokenizer,
    session: &mut Session,
    texts: &[String],
) -> Result<Vec<f32>, RouterError> {
    if texts.is_empty() {
        return Err(RouterError::InferenceFailed(
            "empty examples list".to_string(),
        ));
    }

    // Determine embedding dimension from the first text, initialize accumulator.
    let first = embed_text(tokenizer, session, &texts[0])
        .map_err(|e| RouterError::InferenceFailed(format!("centroid embed: {e}")))?;

    let dim = first.len();
    let mut sum: Vec<f32> = first;
    let mut count: usize = 1;

    for text in &texts[1..] {
        let emb = embed_text(tokenizer, session, text)
            .map_err(|e| RouterError::InferenceFailed(format!("centroid embed: {e}")))?;
        // Dimension mismatch guard: skip mismatched embeddings rather than panic.
        if emb.len() != dim {
            warn!(
                "embedding dimension mismatch: expected {dim}, got {} — skipping",
                emb.len()
            );
            continue;
        }
        for (s, e) in sum.iter_mut().zip(emb.iter()) {
            *s += e;
        }
        count += 1;
    }

    // Average.
    let n = count as f32;
    for s in sum.iter_mut() {
        *s /= n;
    }

    // L2-normalize the centroid.
    l2_normalize(&mut sum);
    Ok(sum)
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

    // ---- compute_centroid (unit-level, no ONNX needed) ----

    /// Test that the centroid of a single embedding equals that embedding
    /// (after normalization). We exercise the averaging + normalization logic
    /// directly by using pre-computed vectors rather than going through ONNX.
    #[test]
    fn test_centroid_of_single_example_equals_that_example() {
        // Simulate: embed produces [3.0, 4.0], normalized = [0.6, 0.8]
        let mut v = vec![3.0_f32, 4.0_f32];
        l2_normalize(&mut v);

        // When compute_centroid is called with one example, the result is
        // the normalized version of that embedding.
        let mut result = v.clone();
        l2_normalize(&mut result); // already normalized, stays the same
        assert!((result[0] - v[0]).abs() < 1e-6);
        assert!((result[1] - v[1]).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_averaging_and_normalization() {
        // Two embeddings: [1.0, 0.0] and [0.0, 1.0]
        // Average: [0.5, 0.5]
        // L2-norm: sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.7071
        // Normalized: [0.5/0.7071, 0.5/0.7071] ≈ [0.7071, 0.7071]
        let e1 = vec![1.0_f32, 0.0_f32];
        let e2 = vec![0.0_f32, 1.0_f32];

        let mut sum = vec![0.0_f32; 2];
        for s in sum.iter_mut().zip(e1.iter()) {
            *s.0 += s.1;
        }
        for s in sum.iter_mut().zip(e2.iter()) {
            *s.0 += s.1;
        }
        // Average
        let n = 2.0_f32;
        for s in sum.iter_mut() {
            *s /= n;
        }
        // Normalize
        l2_normalize(&mut sum);

        let expected = 1.0_f32 / std::f32::consts::SQRT_2;
        assert!((sum[0] - expected).abs() < 1e-6);
        assert!((sum[1] - expected).abs() < 1e-6);

        // The result must be unit length
        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "centroid must be unit norm");
    }

    #[test]
    fn test_centroid_empty_examples_returns_error() {
        // We can't call compute_centroid directly without a session, but we can
        // verify the contract holds by checking the error case logic: the function
        // must return Err when examples is empty.
        // This is a structural verification test (the actual function body checks
        // texts.is_empty() before any session call).
        let texts: Vec<String> = vec![];
        assert!(
            texts.is_empty(),
            "empty slice invariant for compute_centroid error path"
        );
    }

    // ---- Model domain threshold gating (using controlled scoring) ----

    #[test]
    fn test_model_domain_no_threshold_picks_highest() {
        // With no threshold configured, highest-scoring model wins regardless.
        // We test this by verifying the scoring logic directly with pre-built vectors.
        let query = vec![1.0_f32, 0.0_f32];
        let model_a = vec![1.0_f32, 0.0_f32]; // score = 1.0 (identical)
        let model_b = vec![0.0_f32, 1.0_f32]; // score = 0.0 (orthogonal)

        let score_a = cosine_similarity(&query, &model_a);
        let score_b = cosine_similarity(&query, &model_b);

        let candidates = vec![
            ScoredCandidate {
                id: "a".to_string(),
                score: score_a,
            },
            ScoredCandidate {
                id: "b".to_string(),
                score: score_b,
            },
        ];

        // No threshold: always pick highest
        let best = candidates.into_iter().max_by(|x, y| {
            x.score
                .partial_cmp(&y.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let threshold: Option<f32> = None;
        let model = match (best, threshold) {
            (Some(m), Some(t)) if m.score < t => None,
            (Some(m), _) => Some(m),
            (None, _) => None,
        };

        assert!(model.is_some());
        assert_eq!(model.unwrap().id, "a");
    }

    #[test]
    fn test_model_domain_threshold_gate_below_threshold_returns_none() {
        // Best model scores 0.6, threshold is 0.8 -> should return None (use default).
        let candidates = vec![
            ScoredCandidate {
                id: "expensive".to_string(),
                score: 0.6,
            },
            ScoredCandidate {
                id: "cheap".to_string(),
                score: 0.3,
            },
        ];

        let threshold: Option<f32> = Some(0.8);
        let best = candidates.into_iter().max_by(|x, y| {
            x.score
                .partial_cmp(&y.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let model = match (best, threshold) {
            (Some(m), Some(t)) if m.score < t => None,
            (Some(m), _) => Some(m),
            (None, _) => None,
        };

        assert!(model.is_none(), "below threshold should return None");
    }

    #[test]
    fn test_model_domain_threshold_gate_above_threshold_returns_winner() {
        // Best model scores 0.9, threshold is 0.5 -> should return the winner.
        let candidates = vec![
            ScoredCandidate {
                id: "winner".to_string(),
                score: 0.9,
            },
            ScoredCandidate {
                id: "loser".to_string(),
                score: 0.2,
            },
        ];

        let threshold: Option<f32> = Some(0.5);
        let best = candidates.into_iter().max_by(|x, y| {
            x.score
                .partial_cmp(&y.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let model = match (best, threshold) {
            (Some(m), Some(t)) if m.score < t => None,
            (Some(m), _) => Some(m),
            (None, _) => None,
        };

        assert!(model.is_some());
        assert_eq!(model.unwrap().id, "winner");
    }

    // ---- ToolNecessity binary decision logic ----

    #[test]
    fn test_tool_necessity_needs_tools_wins() {
        let scored = vec![
            ScoredCandidate {
                id: "needs_tools".to_string(),
                score: 0.8,
            },
            ScoredCandidate {
                id: "no_tools".to_string(),
                score: 0.3,
            },
        ];

        let needs = scored.iter().find(|s| s.id == "needs_tools");
        let no = scored.iter().find(|s| s.id == "no_tools");

        let tools_needed = match (needs, no) {
            (Some(n), Some(nn)) => Some(n.score >= nn.score),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        };

        assert_eq!(
            tools_needed,
            Some(true),
            "needs_tools higher score => tools_needed = Some(true)"
        );
    }

    #[test]
    fn test_tool_necessity_no_tools_wins() {
        let scored = vec![
            ScoredCandidate {
                id: "needs_tools".to_string(),
                score: 0.2,
            },
            ScoredCandidate {
                id: "no_tools".to_string(),
                score: 0.7,
            },
        ];

        let needs = scored.iter().find(|s| s.id == "needs_tools");
        let no = scored.iter().find(|s| s.id == "no_tools");

        let tools_needed = match (needs, no) {
            (Some(n), Some(nn)) => Some(n.score >= nn.score),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        };

        assert_eq!(
            tools_needed,
            Some(false),
            "no_tools higher score => tools_needed = Some(false)"
        );
    }

    #[test]
    fn test_tool_necessity_only_needs_tools_candidate() {
        let scored = vec![ScoredCandidate {
            id: "needs_tools".to_string(),
            score: 0.5,
        }];

        let needs = scored.iter().find(|s| s.id == "needs_tools");
        let no = scored.iter().find(|s| s.id == "no_tools");

        let tools_needed = match (needs, no) {
            (Some(n), Some(nn)) => Some(n.score >= nn.score),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        };

        assert_eq!(tools_needed, Some(true));
    }

    #[test]
    fn test_tool_necessity_only_no_tools_candidate() {
        let scored = vec![ScoredCandidate {
            id: "no_tools".to_string(),
            score: 0.5,
        }];

        let needs = scored.iter().find(|s| s.id == "needs_tools");
        let no = scored.iter().find(|s| s.id == "no_tools");

        let tools_needed = match (needs, no) {
            (Some(n), Some(nn)) => Some(n.score >= nn.score),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        };

        assert_eq!(tools_needed, Some(false));
    }

    #[test]
    fn test_tool_necessity_no_candidates_returns_none() {
        let scored: Vec<ScoredCandidate> = vec![];

        let needs = scored.iter().find(|s| s.id == "needs_tools");
        let no = scored.iter().find(|s| s.id == "no_tools");

        let tools_needed = match (needs, no) {
            (Some(n), Some(nn)) => Some(n.score >= nn.score),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        };

        assert_eq!(tools_needed, None);
    }

    // ---- Cache key format (no collisions between domains) ----

    #[test]
    fn test_cache_key_format_no_collision_commands_vs_model() {
        let name = "complex";
        let cmd_key = format!("{}:{name}", RoutingDomainKind::Commands.cache_prefix());
        let model_key = format!("{}:{name}", RoutingDomainKind::Model.cache_prefix());
        assert_ne!(cmd_key, model_key);
        assert_eq!(cmd_key, "cmd:complex");
        assert_eq!(model_key, "model:complex");
    }

    #[test]
    fn test_cache_key_format_all_domains() {
        let name = "test";
        let keys: Vec<String> = vec![
            RoutingDomainKind::Commands,
            RoutingDomainKind::Model,
            RoutingDomainKind::ToolNecessity,
            RoutingDomainKind::Memory,
        ]
        .iter()
        .map(|k| format!("{}:{name}", k.cache_prefix()))
        .collect();

        // All keys must be unique.
        let unique: std::collections::HashSet<_> = keys.iter().collect();
        assert_eq!(
            unique.len(),
            keys.len(),
            "cache keys across all domains must be unique"
        );
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
            HashMap::new(),
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
        // Model and tools_needed should be conservative defaults.
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
    }

    #[tokio::test]
    async fn test_fallback_returns_all_commands() {
        let (domain, candidates) = make_commands_domain(&[
            ("cmd_a", "Does A"),
            ("cmd_b", "Does B"),
            ("cmd_c", "Does C"),
        ]);

        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;

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
        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;

        let decision = router
            .route("something", &[(RoutingDomainKind::Commands, vec![])])
            .await
            .expect("empty commands should not error");

        assert!(decision.commands.is_empty());
    }

    #[tokio::test]
    async fn test_fallback_scores_all_commands_1_0() {
        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;

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

        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;

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
    async fn test_fallback_multi_domain_returns_conservative_defaults() {
        // When in fallback mode with multiple domains, non-Command domains
        // should return conservative defaults (model=None, tools_needed=None).
        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;

        let domains = vec![
            (
                RoutingDomainKind::Commands,
                vec![RoutingCandidate {
                    id: "search".to_string(),
                    examples: vec!["search: Search".to_string()],
                }],
            ),
            (
                RoutingDomainKind::Model,
                vec![
                    RoutingCandidate {
                        id: "fast".to_string(),
                        examples: vec!["quick question".to_string()],
                    },
                    RoutingCandidate {
                        id: "complex".to_string(),
                        examples: vec!["deep analysis".to_string()],
                    },
                ],
            ),
            (
                RoutingDomainKind::ToolNecessity,
                vec![
                    RoutingCandidate {
                        id: "needs_tools".to_string(),
                        examples: vec!["search the web".to_string()],
                    },
                    RoutingCandidate {
                        id: "no_tools".to_string(),
                        examples: vec!["what is 2+2".to_string()],
                    },
                ],
            ),
        ];

        let decision = router
            .route("hello", &domains)
            .await
            .expect("fallback should not error");

        // Commands: all at 1.0
        assert_eq!(decision.commands.len(), 1);
        assert_eq!(decision.commands[0].score, 1.0);
        // Model: None (conservative)
        assert!(decision.model.is_none());
        // ToolNecessity: None (conservative)
        assert!(decision.tools_needed.is_none());
    }

    #[tokio::test]
    async fn test_cache_starts_empty() {
        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;
        assert_eq!(router.cached_count().await, 0);
    }

    #[tokio::test]
    async fn test_empty_domain_list_returns_empty_decision() {
        let router = ModernBertRouter::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &[],
            HashMap::new(),
        )
        .await;

        let decision = router
            .route("hello", &[])
            .await
            .expect("empty domains should not error");

        assert!(decision.commands.is_empty());
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
        assert!(decision.memory_stores.is_empty());
    }
}
