//! Anthropic Messages API wire format types.
//!
//! These serde types represent the Anthropic wire format as used by both
//! outbound (provider client) and inbound (compat endpoint) directions.
//! They are the single source of truth for Anthropic wire format serialization.

use serde::{Deserialize, Serialize};

/// Anthropic Messages API request body.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    /// The system prompt. Optional because:
    /// 1. Inbound requests may omit the `system` field entirely.
    /// 2. Outbound requests always set it (from messages[0] positional convention).
    ///
    /// Uses `#[serde(default)]` so absent JSON field deserializes to `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Inbound-only: clients may request streaming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// An Anthropic-format message.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

/// Anthropic Messages API response body.
///
/// The `id`, `kind`, `role`, `model`, and `stop_reason` fields are added in
/// Phase 2 because they are part of the Anthropic wire format — the real API
/// returns them, and `parse_outbound_response` should be able to deserialize them.
/// These fields are `Option` with `#[serde(default)]` so existing outbound
/// deserialization is unaffected.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
    #[serde(default)]
    pub stop_reason: Option<String>,
}

/// A content block in an Anthropic response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// Token usage in Anthropic format.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
