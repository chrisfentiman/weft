//! Tokenizer wrapper for ModernBERT.
//!
//! Wraps the HuggingFace `tokenizers` crate, providing a clean interface
//! for encoding text into `input_ids` and `attention_mask` tensors.

use tokenizers::Tokenizer;

/// Wraps a HuggingFace tokenizer for BERT-style encoding.
///
/// Produces `input_ids` and `attention_mask` as owned `Vec<i64>` for
/// ONNX Runtime input construction.
pub(crate) struct BertTokenizer {
    inner: Tokenizer,
}

impl BertTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file at `path`.
    ///
    /// Returns an error string if the file is not found or is malformed.
    pub(crate) fn from_file(path: &str) -> Result<Self, String> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| format!("failed to load tokenizer from '{path}': {e}"))?;
        Ok(Self { inner })
    }

    /// Encode `text` and return `(input_ids, attention_mask)`.
    ///
    /// Both vectors have the same length (the sequence length after truncation).
    /// Values are `i64` because that is what ONNX Runtime expects for `INT64` tensors.
    pub(crate) fn encode(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), String> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| format!("tokenization failed: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        Ok((input_ids, attention_mask))
    }
}
