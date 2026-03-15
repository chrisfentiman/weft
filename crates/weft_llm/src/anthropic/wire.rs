/// Private wire types for the Anthropic Messages API.
/// These are not exported from the crate.
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(super) struct AnthropicRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub(super) struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicResponse {
    pub content: Vec<ContentBlock>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
