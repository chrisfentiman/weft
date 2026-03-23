//! OpenAI Chat Completions API wire format types.
//!
//! These serde types represent the OpenAI wire format as used by both
//! outbound (provider client) and inbound (compat endpoint) directions.
//! They are the single source of truth for OpenAI wire format serialization.

use serde::{Deserialize, Serialize};

/// OpenAI-format chat completion request body.
///
/// Used for both outbound requests (to the provider API) and inbound
/// requests (from the compat endpoint). Fields are optional where
/// either direction may omit them.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Inbound-only: clients may request streaming. The compat handler
    /// validates and rejects this; it is never set on outbound requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// An OpenAI-format message.
///
/// The `role` field uses `String` for both directions. Translation functions
/// handle the mapping to/from the `Role` domain enum. Using `String` here
/// means deserialization always succeeds and unknown role strings are handled
/// gracefully by the translation layer rather than serde.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-format chat completion response body.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIResponse {
    /// Response ID. Absent from some provider responses (e.g., Ollama).
    /// Generated with `chatcmpl-` prefix for inbound compat responses.
    #[serde(default)]
    pub id: Option<String>,
    /// Object type. Always "chat.completion" in outbound responses.
    #[serde(default)]
    pub object: Option<String>,
    /// Unix timestamp. Generated for inbound compat responses.
    #[serde(default)]
    pub created: Option<u64>,
    /// Model identifier. Echoed from request for inbound compat.
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIChoice {
    #[serde(default)]
    pub index: u32,
    pub message: OpenAIMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Inbound-only field: total tokens for the compat response envelope.
    /// Some providers include this; some don't. Default to 0 when absent.
    #[serde(default)]
    pub total_tokens: u32,
}
