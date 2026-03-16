//! Universal provider types: request/response enums, error type, capability newtype.
//!
//! These types form the contract between the gateway engine and provider implementations.
//! The `Provider` trait (defined in `lib.rs`) uses these types exclusively.

use weft_core::Message;

/// A capability that a model supports.
///
/// Capabilities are strings, not an enum, because the set is open -- users can
/// declare arbitrary capability strings for providers we don't know about.
/// The gateway does not hardcode the list.
///
/// Well-known capability strings are provided as constants for type safety
/// in compiled code. Config-declared capabilities are validated against nothing --
/// they're free-form strings that only need to match between model declarations
/// and request routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Capability(String);

impl Capability {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// Well-known capability constants.
// These are the strings used in config files and routing lookups.
impl Capability {
    pub const CHAT_COMPLETIONS: &'static str = "chat_completions";
    pub const EMBEDDINGS: &'static str = "embeddings";
    pub const IMAGE_GENERATIONS: &'static str = "image_generations";
    pub const IMAGE_EDITS: &'static str = "image_edits";
    pub const VIDEO_GENERATIONS: &'static str = "video_generations";
    pub const AUDIO_SPEECH: &'static str = "audio_speech";
    pub const AUDIO_TRANSCRIPTIONS: &'static str = "audio_transcriptions";
    pub const VISION: &'static str = "vision";
    pub const TOOL_CALLING: &'static str = "tool_calling";
    pub const STREAMING: &'static str = "streaming";
    pub const MODERATIONS: &'static str = "moderations";
    pub const RERANK: &'static str = "rerank";
    pub const SEARCH: &'static str = "search";
    pub const REALTIME: &'static str = "realtime";
    pub const STRUCTURED_OUTPUT: &'static str = "structured_output";
}

/// A request to a provider. Variants cover all capability types.
///
/// The engine constructs the appropriate variant based on the operation.
/// Providers match on the variant they support and return `ProviderError::Unsupported`
/// for variants they don't handle.
#[derive(Debug, Clone)]
pub enum ProviderRequest {
    /// Chat completion (text generation).
    ChatCompletion(ChatCompletionInput),
    // Future variants -- defined here so the enum shape is visible,
    // but not wired into the engine yet:
    // Embedding(EmbeddingInput),
    // ImageGeneration(ImageGenerationInput),
    // AudioSpeech(AudioSpeechInput),
    // AudioTranscription(AudioTranscriptionInput),
    // Moderation(ModerationInput),
    // Rerank(RerankInput),
}

/// Input for a chat completion request.
#[derive(Debug, Clone)]
pub struct ChatCompletionInput {
    /// The assembled system prompt.
    pub system_prompt: String,
    /// The conversation messages.
    pub messages: Vec<Message>,
    /// Model identifier to send to the provider API.
    pub model: String,
    /// Maximum tokens in response.
    pub max_tokens: u32,
    /// Sampling temperature. If None, use provider default.
    pub temperature: Option<f32>,
}

/// A response from a provider.
#[derive(Debug, Clone)]
pub enum ProviderResponse {
    /// Chat completion response.
    ChatCompletion(ChatCompletionOutput),
    // Future variants:
    // Embedding(EmbeddingOutput),
    // ImageGeneration(ImageGenerationOutput),
    // etc.
}

/// Output from a chat completion request.
#[derive(Debug, Clone)]
pub struct ChatCompletionOutput {
    /// The assistant's response text.
    pub text: String,
    /// Token usage, if the provider reports it.
    pub usage: Option<TokenUsage>,
}

/// Token usage reported by a provider.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Error type for provider operations.
///
/// Replaces `LlmError`. Variant names are more precise and include
/// `WireScriptError` for Rhai custom wire format failures.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// The provider does not support this capability / request type.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// Network or HTTP request failure.
    #[error("request failed: {0}")]
    RequestFailed(String),

    /// Provider returned a non-success HTTP status.
    #[error("provider error: status={status}, body={body}")]
    ProviderHttpError { status: u16, body: String },

    /// Response could not be deserialized.
    #[error("deserialization failed: {0}")]
    DeserializationError(String),

    /// Rate limited by the provider.
    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    /// Rhai script execution error (custom wire format only).
    #[error("wire script error: {script}: {message}")]
    WireScriptError { script: String, message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Capability tests ──────────────────────────────────────────────────

    #[test]
    fn test_capability_construction() {
        let cap = Capability::new("chat_completions");
        assert_eq!(cap.as_str(), "chat_completions");
    }

    #[test]
    fn test_capability_display() {
        let cap = Capability::new("embeddings");
        assert_eq!(cap.to_string(), "embeddings");
    }

    #[test]
    fn test_capability_equality() {
        let a = Capability::new("chat_completions");
        let b = Capability::new("chat_completions");
        let c = Capability::new("embeddings");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_capability_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Capability::new("chat_completions"));
        set.insert(Capability::new("embeddings"));
        // Inserting duplicate should not increase size
        set.insert(Capability::new("chat_completions"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_capability_constants_match_expected_strings() {
        assert_eq!(Capability::CHAT_COMPLETIONS, "chat_completions");
        assert_eq!(Capability::EMBEDDINGS, "embeddings");
        assert_eq!(Capability::IMAGE_GENERATIONS, "image_generations");
        assert_eq!(Capability::IMAGE_EDITS, "image_edits");
        assert_eq!(Capability::VIDEO_GENERATIONS, "video_generations");
        assert_eq!(Capability::AUDIO_SPEECH, "audio_speech");
        assert_eq!(Capability::AUDIO_TRANSCRIPTIONS, "audio_transcriptions");
        assert_eq!(Capability::VISION, "vision");
        assert_eq!(Capability::TOOL_CALLING, "tool_calling");
        assert_eq!(Capability::STREAMING, "streaming");
        assert_eq!(Capability::MODERATIONS, "moderations");
        assert_eq!(Capability::RERANK, "rerank");
        assert_eq!(Capability::SEARCH, "search");
        assert_eq!(Capability::REALTIME, "realtime");
        assert_eq!(Capability::STRUCTURED_OUTPUT, "structured_output");
    }

    #[test]
    fn test_capability_from_constant() {
        let cap = Capability::new(Capability::CHAT_COMPLETIONS);
        assert_eq!(cap.as_str(), Capability::CHAT_COMPLETIONS);
        assert_eq!(cap, Capability::new("chat_completions"));
    }

    // ── ProviderRequest / ProviderResponse construction tests ─────────────

    #[test]
    fn test_chat_completion_input_construction() {
        use weft_core::{Message, Role};
        let input = ChatCompletionInput {
            system_prompt: "You are helpful.".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: "Hello".to_string(),
            }],
            model: "gpt-4".to_string(),
            max_tokens: 1024,
            temperature: Some(0.7),
        };
        assert_eq!(input.system_prompt, "You are helpful.");
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.model, "gpt-4");
        assert_eq!(input.max_tokens, 1024);
        assert_eq!(input.temperature, Some(0.7));
    }

    #[test]
    fn test_provider_request_chat_completion_construction() {
        use weft_core::{Message, Role};
        let request = ProviderRequest::ChatCompletion(ChatCompletionInput {
            system_prompt: "sys".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_string(),
            }],
            model: "claude-test".to_string(),
            max_tokens: 2048,
            temperature: None,
        });
        assert!(matches!(request, ProviderRequest::ChatCompletion(_)));
    }

    #[test]
    fn test_chat_completion_output_construction() {
        let output = ChatCompletionOutput {
            text: "Hello!".to_string(),
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        };
        assert_eq!(output.text, "Hello!");
        let usage = output.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_provider_response_chat_completion_construction() {
        let response = ProviderResponse::ChatCompletion(ChatCompletionOutput {
            text: "response text".to_string(),
            usage: None,
        });
        assert!(matches!(response, ProviderResponse::ChatCompletion(_)));
        // Only ChatCompletion variant exists; when future variants land this becomes refutable.
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion(output) = response else {
            panic!("expected ChatCompletion response");
        };
        assert_eq!(output.text, "response text");
        assert!(output.usage.is_none());
    }

    #[test]
    fn test_token_usage_field_access() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
        };
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
    }

    // ── ProviderError display formatting tests ────────────────────────────

    #[test]
    fn test_provider_error_unsupported_display() {
        let err = ProviderError::Unsupported("embeddings not supported".to_string());
        assert_eq!(
            err.to_string(),
            "unsupported operation: embeddings not supported"
        );
    }

    #[test]
    fn test_provider_error_request_failed_display() {
        let err = ProviderError::RequestFailed("connection refused".to_string());
        assert_eq!(err.to_string(), "request failed: connection refused");
    }

    #[test]
    fn test_provider_error_provider_http_error_display() {
        let err = ProviderError::ProviderHttpError {
            status: 503,
            body: "service unavailable".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "provider error: status=503, body=service unavailable"
        );
    }

    #[test]
    fn test_provider_error_deserialization_display() {
        let err = ProviderError::DeserializationError("missing field 'choices'".to_string());
        assert_eq!(
            err.to_string(),
            "deserialization failed: missing field 'choices'"
        );
    }

    #[test]
    fn test_provider_error_rate_limited_display() {
        let err = ProviderError::RateLimited {
            retry_after_ms: 60_000,
        };
        assert_eq!(err.to_string(), "rate limited, retry after 60000ms");
    }

    #[test]
    fn test_provider_error_wire_script_error_display() {
        let err = ProviderError::WireScriptError {
            script: "providers/banana.rhai".to_string(),
            message: "compilation failed: unexpected token".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "wire script error: providers/banana.rhai: compilation failed: unexpected token"
        );
    }
}
