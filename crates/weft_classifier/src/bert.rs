//! ModernBERT bi-encoder via ONNX Runtime.
//!
//! Loads a ModernBERT ONNX model and tokenizer, pre-computes command embeddings
//! at construction time if provided, and scores user messages against those
//! embeddings via cosine similarity at query time. Unknown commands are embedded
//! lazily during `classify()` and cached for subsequent calls.
//!
//! The bi-encoder approach embeds query and commands separately:
//! - Command embeddings: computed lazily on first encounter, then cached.
//! - Query embedding: computed per request (~20-50ms CPU).
//! - Similarity: dot product of L2-normalized vectors (cosine similarity).
//!
//! Fallback: if the ONNX model or tokenizer fails to load, the classifier
//! returns all commands unfiltered. This keeps the gateway functional even
//! when the model is misconfigured.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use ort::inputs;
use ort::session::Session;
use ort::value::Value;
use tokio::sync::Mutex;
use tracing::{debug, warn};
use weft_core::CommandStub;

use crate::tokenizer::BertTokenizer;
use crate::{ClassificationResult, ClassifierError, SemanticClassifier};

/// Combined session and embedding cache, held inside a single `Mutex`.
///
/// Collocating the cache with the session avoids a second lock: cache writes
/// happen inside inference calls, which already hold the session mutex.
struct SessionState {
    session: Session,
    /// L2-normalized command embeddings keyed by command name.
    embeddings: HashMap<String, Vec<f32>>,
}

/// ModernBERT bi-encoder classifier.
///
/// Wraps an ONNX session and a HuggingFace tokenizer. Builds L2-normalized
/// embeddings for command stubs lazily during `classify()` (or eagerly at
/// construction time if commands are provided). At query time, embeds the user
/// message and computes cosine similarity against cached command embeddings.
///
/// Thread-safe: `Session` is protected by a `tokio::sync::Mutex`. The
/// tokenizer is `Send + Sync` and needs no locking.
pub struct ModernBertClassifier {
    /// ONNX Runtime session plus embedding cache, or `None` in fallback mode.
    ///
    /// Both the session and the cache are behind a single Mutex because cache
    /// writes require inference, which requires holding the session lock anyway.
    state: Option<Arc<Mutex<SessionState>>>,
    /// Tokenizer, or `None` if it failed to load (fallback mode).
    tokenizer: Option<Arc<BertTokenizer>>,
}

impl ModernBertClassifier {
    /// Build a classifier, loading the ONNX model and tokenizer from disk.
    ///
    /// If `commands` is non-empty, their embeddings are pre-computed now as a
    /// warm-start optimisation. If empty (`&[]`), the cache starts empty and
    /// commands are embedded lazily on the first `classify()` call.
    ///
    /// If the model or tokenizer cannot be loaded, logs a warning and constructs
    /// the classifier in fallback mode (no filtering, returns all commands).
    ///
    /// The constructor is `async` because it must use `.lock().await` on the
    /// session mutex. `blocking_lock()` panics when called from within the tokio
    /// runtime, and this constructor is called from `#[tokio::main]`.
    ///
    /// `model_path`: Path to the ONNX model file.
    /// `tokenizer_path`: Path to the HuggingFace `tokenizer.json` file.
    /// `commands`: Command stubs to pre-embed. Pass `&[]` to skip pre-embedding.
    pub async fn new(model_path: &str, tokenizer_path: &str, commands: &[CommandStub]) -> Self {
        let tokenizer = match BertTokenizer::from_file(tokenizer_path) {
            Ok(t) => {
                debug!("tokenizer loaded from '{tokenizer_path}'");
                Some(Arc::new(t))
            }
            Err(e) => {
                warn!("classifier falling back to passthrough: {e}");
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
                    warn!("classifier falling back to passthrough: ONNX load failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Pre-embed any commands provided at construction time.
        // The session mutex is freshly created and uncontested here, but we use
        // `.lock().await` because the constructor is called from within the tokio
        // runtime. `blocking_lock()` would panic in that context.
        if let (Some(state), Some(tok)) = (&session, &tokenizer) {
            let mut guard = state.lock().await;
            for cmd in commands {
                let text = format!("{}: {}", cmd.name, cmd.description);
                match embed_text(tok, &mut guard.session, &text) {
                    Ok(mut v) => {
                        l2_normalize(&mut v);
                        guard.embeddings.insert(cmd.name.clone(), v);
                    }
                    Err(e) => {
                        warn!("failed to embed command '{}': {e}", cmd.name);
                    }
                }
            }
            let count = guard.embeddings.len();
            if count > 0 {
                debug!("pre-computed embeddings for {count} commands");
            }
        }

        Self {
            state: session,
            tokenizer,
        }
    }

    /// Returns `true` if the classifier is in fallback mode (no model loaded).
    pub fn is_fallback(&self) -> bool {
        self.state.is_none()
    }

    /// Returns the number of commands whose embeddings are currently cached.
    ///
    /// In fallback mode this is always 0. In normal mode it reflects the number
    /// of commands that have been successfully embedded (at construction or
    /// lazily during `classify()`).
    #[cfg(test)]
    pub async fn cached_command_count(&self) -> usize {
        match &self.state {
            None => 0,
            Some(state) => state.lock().await.embeddings.len(),
        }
    }
}

#[async_trait]
impl SemanticClassifier for ModernBertClassifier {
    async fn classify(
        &self,
        user_message: &str,
        commands: &[CommandStub],
    ) -> Result<Vec<ClassificationResult>, ClassifierError> {
        // Fallback: model not loaded — return all commands with score 1.0.
        let (state, tokenizer) = match (&self.state, &self.tokenizer) {
            (Some(s), Some(t)) => (s, t),
            _ => {
                debug!("classifier in fallback mode, returning all commands");
                // Spec Section 5.5: fallback returns all commands sorted alphabetically.
                let mut results: Vec<ClassificationResult> = commands
                    .iter()
                    .map(|c| ClassificationResult {
                        command_name: c.name.clone(),
                        score: 1.0,
                    })
                    .collect();
                results.sort_by(|a, b| a.command_name.cmp(&b.command_name));
                return Ok(results);
            }
        };

        // Acquire the session lock once for the entire classify call.
        // Both cache writes and inference happen inside this critical section,
        // which is correct per spec Section 5.3 and Section 10.3.
        let mut guard = state.lock().await;

        // Spec Section 5.3: embed any commands not yet in the cache (lazy embedding).
        for cmd in commands {
            if !guard.embeddings.contains_key(&cmd.name) {
                let text = format!("{}: {}", cmd.name, cmd.description);
                match embed_text(tokenizer, &mut guard.session, &text) {
                    Ok(mut v) => {
                        l2_normalize(&mut v);
                        guard.embeddings.insert(cmd.name.clone(), v);
                        debug!("lazily embedded command '{}'", cmd.name);
                    }
                    Err(e) => {
                        warn!("failed to lazily embed command '{}': {e}", cmd.name);
                        // Omit from cache so the next call retries embedding.
                    }
                }
            }
        }

        // Embed the user message.
        let mut query_vec =
            embed_text(tokenizer, &mut guard.session, user_message).map_err(|e| {
                ClassifierError::InferenceFailed(format!("query embedding failed: {e}"))
            })?;
        l2_normalize(&mut query_vec);

        // Score each command against the query using cached embeddings.
        let results = commands
            .iter()
            .map(|cmd| {
                let score = guard
                    .embeddings
                    .get(&cmd.name)
                    .map(|emb| cosine_similarity(&query_vec, emb))
                    .unwrap_or(0.0);
                ClassificationResult {
                    command_name: cmd.name.clone(),
                    score,
                }
            })
            .collect();

        Ok(results)
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
    use weft_core::CommandStub;

    fn make_commands(names: &[(&str, &str)]) -> Vec<CommandStub> {
        names
            .iter()
            .map(|(n, d)| CommandStub {
                name: n.to_string(),
                description: d.to_string(),
            })
            .collect()
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
        let commands = make_commands(&[
            ("web_search", "Search the web"),
            ("recall", "Retrieve from memory"),
        ]);

        let classifier = ModernBertClassifier::new(
            "/nonexistent/path/model.onnx",
            "/nonexistent/path/tokenizer.json",
            &commands,
        )
        .await;

        assert!(
            classifier.is_fallback(),
            "should be in fallback mode when model is missing"
        );

        let results = classifier
            .classify("find something", &commands)
            .await
            .expect("classify should succeed in fallback mode");

        assert_eq!(results.len(), 2, "should return all commands in fallback");
        for result in &results {
            assert_eq!(result.score, 1.0, "fallback scores should be 1.0");
        }
    }

    #[tokio::test]
    async fn test_fallback_returns_all_commands() {
        let commands = make_commands(&[
            ("cmd_a", "Does A"),
            ("cmd_b", "Does B"),
            ("cmd_c", "Does C"),
        ]);

        let classifier =
            ModernBertClassifier::new("/bad/model.onnx", "/bad/tokenizer.json", &commands).await;

        let results = classifier
            .classify("some query", &commands)
            .await
            .expect("fallback should not error");

        let names: Vec<&str> = results.iter().map(|r| r.command_name.as_str()).collect();
        assert!(names.contains(&"cmd_a"));
        assert!(names.contains(&"cmd_b"));
        assert!(names.contains(&"cmd_c"));
    }

    #[tokio::test]
    async fn test_fallback_empty_commands() {
        let classifier =
            ModernBertClassifier::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        let results = classifier
            .classify("something", &[])
            .await
            .expect("empty commands should not error");

        assert!(results.is_empty());
    }

    // ---- Fallback scores all commands 1.0 ----

    #[tokio::test]
    async fn test_fallback_scores_all_commands_1_0() {
        // Fallback mode: every command passed to classify() returns score 1.0,
        // regardless of what commands (if any) were provided to the constructor.
        let commands_at_startup = make_commands(&[("known", "A known command")]);
        let classifier = ModernBertClassifier::new(
            "/bad/model.onnx",
            "/bad/tokenizer.json",
            &commands_at_startup,
        )
        .await;

        let extra_commands = make_commands(&[("unknown_extra", "Not in cache")]);
        let results = classifier
            .classify("something", &extra_commands)
            .await
            .expect("should not error");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 1.0);
    }

    // ---- Lazy embedding: cache starts empty, populated on first classify() ----

    #[tokio::test]
    async fn test_lazy_embedding_cache_starts_empty() {
        // Constructor called with no commands: cache should be empty.
        // In fallback mode (no model), cached_command_count is always 0.
        let classifier =
            ModernBertClassifier::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        assert_eq!(
            classifier.cached_command_count().await,
            0,
            "cache should be empty when constructed with no commands"
        );
    }

    // ---- Cache reuse: second classify() returns consistent results ----

    #[tokio::test]
    async fn test_cache_reuse_consistent_results() {
        // In fallback mode: two classify() calls with the same commands return
        // the same set of command names (fallback is stable).
        let commands = make_commands(&[
            ("cmd_alpha", "Does alpha things"),
            ("cmd_beta", "Does beta things"),
        ]);
        let classifier =
            ModernBertClassifier::new("/bad/model.onnx", "/bad/tokenizer.json", &[]).await;

        let result_a = classifier
            .classify("query one", &commands)
            .await
            .expect("first classify should succeed");
        let result_b = classifier
            .classify("query two", &commands)
            .await
            .expect("second classify should succeed");

        let names_a: Vec<&str> = result_a.iter().map(|r| r.command_name.as_str()).collect();
        let names_b: Vec<&str> = result_b.iter().map(|r| r.command_name.as_str()).collect();
        assert_eq!(
            names_a, names_b,
            "classify() must return consistent results across calls"
        );
    }

    // ---- Embedding cache populated at construction ----

    #[tokio::test]
    async fn test_embedding_cache_populated_at_construction() {
        // Fallback mode: no embeddings are cached (model never ran).
        let commands = make_commands(&[
            ("cmd_alpha", "Does alpha things"),
            ("cmd_beta", "Does beta things"),
        ]);
        let classifier =
            ModernBertClassifier::new("/bad/model.onnx", "/bad/tokenizer.json", &commands).await;

        assert_eq!(
            classifier.cached_command_count().await,
            0,
            "fallback classifier should have no cached embeddings"
        );

        // Results are consistent across multiple calls.
        let result_a = classifier
            .classify("query one", &commands)
            .await
            .expect("first classify should succeed");
        let result_b = classifier
            .classify("query two", &commands)
            .await
            .expect("second classify should succeed");

        let names_a: Vec<&str> = result_a.iter().map(|r| r.command_name.as_str()).collect();
        let names_b: Vec<&str> = result_b.iter().map(|r| r.command_name.as_str()).collect();
        assert_eq!(names_a, names_b, "results must be consistent across calls");
    }

    // ---- Fallback alphabetical ordering ----

    #[tokio::test]
    async fn test_fallback_results_sorted_alphabetically() {
        // Spec Section 5.5: fallback returns commands sorted alphabetically by name.
        let commands = make_commands(&[
            ("zebra", "Last alphabetically"),
            ("alpha", "First alphabetically"),
            ("middle", "Middle alphabetically"),
        ]);
        let classifier =
            ModernBertClassifier::new("/bad/model.onnx", "/bad/tokenizer.json", &commands).await;

        let results = classifier
            .classify("anything", &commands)
            .await
            .expect("fallback should not error");

        let names: Vec<&str> = results.iter().map(|r| r.command_name.as_str()).collect();
        assert_eq!(
            names,
            vec!["alpha", "middle", "zebra"],
            "fallback results must be sorted alphabetically by command_name"
        );
    }
}
