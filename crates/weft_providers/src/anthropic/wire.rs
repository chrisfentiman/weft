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

/// Content of an Anthropic message.
///
/// The Anthropic API accepts content as either a plain string or an array of
/// typed content blocks. Both forms carry the same semantic meaning; the array
/// form is used by newer clients and the string form by older/simpler ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    /// Plain-text string content (legacy/simple form).
    Text(String),
    /// Array of typed content blocks (structured form).
    Blocks(Vec<InboundContentBlock>),
}

/// A content block inside an inbound Anthropic message.
///
/// Only `type: "text"` blocks are supported for inbound requests.
/// Other block types (image, tool_use, tool_result) are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InboundContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: String,
}

/// An Anthropic-format message.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
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
