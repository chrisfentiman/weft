//! Universal provider types: request/response enums, error type, capability newtype.
//!
//! These types form the contract between the gateway engine and provider implementations.
//! The `Provider` trait (defined in `lib.rs`) uses these types exclusively.

use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};

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
    ChatCompletion {
        /// Conversation messages in Weft Wire format.
        ///
        /// The system prompt, if present, is `messages[0]` with `Role::System`.
        /// This is a positional convention set by the gateway during context assembly.
        /// Providers check `messages[0]`: if `Role::System`, handle per wire format
        /// (Anthropic extracts to top-level `system` field, OpenAI leaves in place,
        /// Rhai passes through). If `messages[0]` is not `Role::System`, there is
        /// no system prompt.
        ///
        /// Providers extract the content parts they support (text, image, etc.)
        /// and translate to their wire format.
        messages: Vec<WeftMessage>,
        /// Model identifier to send to the provider API.
        model: String,
        /// Sampling and behavior options. Providers extract what they support
        /// and ignore the rest.
        options: SamplingOptions,
    },
    // Future variants (same enum shape, will carry WeftMessage content):
    // Embedding { input: Vec<WeftMessage>, model: String },
    // ImageGeneration { prompt: WeftMessage, model: String, options: ... },
    // AudioTranscription { audio: WeftMessage, model: String },
}

/// A response from a provider.
#[derive(Debug, Clone)]
pub enum ProviderResponse {
    /// Chat completion response.
    ChatCompletion {
        /// The assistant's response as a WeftMessage.
        ///
        /// Contains at minimum a Text content part with the response text.
        /// Future multimodal providers may include Image, Audio, etc.
        /// Role is always Assistant, Source is always Provider.
        message: WeftMessage,
        /// Token usage, if the provider reports it.
        usage: Option<TokenUsage>,
    },
    // Future variants:
    // Embedding { vectors: Vec<Vec<f32>>, usage: Option<TokenUsage> },
    // ImageGeneration { message: WeftMessage },
}

/// Token usage reported by a provider.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Extract text content from WeftMessages for provider wire format conversion.
///
/// Filters out activity messages (source: Gateway, role: System) since those
/// are gateway-internal activity content (routing events, hook results), not
/// conversational content for the LLM. This is a behavioral change from the
/// pre-migration code where ALL Role::System messages were forwarded to providers.
///
/// Concatenates all Text content parts within each message, separated by newlines.
/// Messages with no text content after extraction are omitted.
///
/// Tool results (CommandResult) are serialized as text for providers that don't
/// natively support tool calling.
pub fn extract_text_messages(messages: &[WeftMessage]) -> Vec<(Role, String)> {
    messages
        .iter()
        // Activity messages (source: Gateway, role: System) are gateway-internal
        // telemetry -- routing events, hook results. Not conversational content.
        .filter(|m| !(m.role == Role::System && m.source == Source::Gateway))
        .map(|m| {
            let text: String = m
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(t) => Some(t.as_str()),
                    ContentPart::CommandResult(cr) => {
                        // Serialize tool results as text for providers that don't
                        // natively support structured tool output.
                        // The engine has already formatted this.
                        Some(cr.output.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (m.role, text)
        })
        .filter(|(_, text)| !text.is_empty())
        .collect()
}

/// A chunk from a streaming provider response.
///
/// Providers that support true streaming yield one or more `Delta` chunks as tokens
/// arrive from the API, followed by a final `Complete` chunk carrying the assembled
/// `ProviderResponse` and usage data.
///
/// Providers that do not support streaming return a single `Complete` chunk via the
/// default `execute_stream` implementation, which wraps `execute()`. The
/// `GenerateActivity` handles both shapes through the same code path.
#[derive(Debug, Clone)]
pub enum ProviderChunk {
    /// A content delta (partial text, partial tool call, etc.) from a streaming
    /// provider. Multiple `Delta` chunks are assembled into the final response.
    Delta(ContentDelta),
    /// The complete response, either as the sole chunk (non-streaming providers)
    /// or as the final chunk in a streaming sequence (after all deltas).
    Complete(ProviderResponse),
}

/// A content delta yielded by a streaming provider.
///
/// Carries the incremental text token and/or partial tool call data from one
/// chunk of a streaming API response. Either field may be absent if the chunk
/// carries only the other content type.
#[derive(Debug, Clone)]
pub struct ContentDelta {
    /// The incremental text token, if this chunk carries text content.
    pub text: Option<String>,
    /// Partial tool/command call data, if this chunk carries tool call content.
    pub tool_call_delta: Option<ToolCallDelta>,
}

/// A partial tool call delta from a streaming provider.
///
/// Real streaming providers accumulate these deltas to reconstruct the full
/// tool call name and arguments as tokens arrive. The `GenerateActivity` may
/// use these to emit `CommandInvocation` events incrementally in the future;
/// for now they are collected and parsed at stream completion via
/// `parse_response_to_events`.
#[derive(Debug, Clone)]
pub struct ToolCallDelta {
    /// Index into the tool call list (for parallel tool calls).
    pub index: usize,
    /// Partial tool/function name, accumulated across deltas.
    pub name_fragment: Option<String>,
    /// Partial arguments JSON fragment, accumulated across deltas.
    pub arguments_fragment: Option<String>,
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
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    // ── Test helpers ──────────────────────────────────────────────────────

    fn make_weft_message(role: Role, source: Source, text: &str) -> WeftMessage {
        WeftMessage {
            role,
            source,
            model: None,
            content: vec![ContentPart::Text(text.to_string())],
            delta: false,
            message_index: 0,
        }
    }

    fn make_weft_message_no_content(role: Role, source: Source) -> WeftMessage {
        WeftMessage {
            role,
            source,
            model: None,
            content: vec![],
            delta: false,
            message_index: 0,
        }
    }

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
    fn test_provider_request_chat_completion_construction() {
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "hi")],
            model: "claude-test".to_string(),
            options: SamplingOptions::default(),
        };
        assert!(matches!(request, ProviderRequest::ChatCompletion { .. }));
    }

    #[test]
    fn test_provider_request_no_system_prompt_field() {
        // The new enum variant has no system_prompt -- it's messages[0] if Role::System.
        let system_msg = make_weft_message(Role::System, Source::Gateway, "You are helpful.");
        let user_msg = make_weft_message(Role::User, Source::Client, "Hello");
        let request = ProviderRequest::ChatCompletion {
            messages: vec![system_msg, user_msg],
            model: "gpt-4".to_string(),
            options: SamplingOptions::default(),
        };
        // Destructure to verify field names
        let ProviderRequest::ChatCompletion {
            messages,
            model,
            options: _,
        } = request;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(model, "gpt-4");
    }

    #[test]
    fn test_provider_response_chat_completion_construction() {
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("gpt-4".to_string()),
            content: vec![ContentPart::Text("response text".to_string())],
            delta: false,
            message_index: 0,
        };
        let response = ProviderResponse::ChatCompletion {
            message,
            usage: None,
        };
        assert!(matches!(response, ProviderResponse::ChatCompletion { .. }));
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, usage } = response else {
            panic!("expected ChatCompletion response");
        };
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.source, Source::Provider);
        assert!(usage.is_none());
    }

    #[test]
    fn test_provider_response_carries_weft_message() {
        let response = ProviderResponse::ChatCompletion {
            message: WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("test-model".to_string()),
                content: vec![ContentPart::Text("Hello!".to_string())],
                delta: false,
                message_index: 0,
            },
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        };
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, usage } = response else {
            panic!("expected ChatCompletion");
        };
        // Extract text from content parts
        let text = message
            .content
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "Hello!");
        assert_eq!(message.model, Some("test-model".to_string()));
        let u = usage.expect("usage should be present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
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

    // ── extract_text_messages tests ───────────────────────────────────────

    #[test]
    fn test_extract_text_messages_basic() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hello")];
        let pairs = extract_text_messages(&messages);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, Role::User);
        assert_eq!(pairs[0].1, "Hello");
    }

    #[test]
    fn test_extract_text_messages_filters_gateway_system() {
        // System + Gateway messages (activity messages) must be filtered out.
        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "You are helpful."),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let pairs = extract_text_messages(&messages);
        // System+Gateway is filtered, only User remains
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, Role::User);
    }

    #[test]
    fn test_extract_text_messages_system_prompt_filtered_out() {
        // The system prompt at messages[0] is Role::System, Source::Gateway.
        // extract_text_messages filters it out -- providers handle it separately.
        let system_msg = make_weft_message(Role::System, Source::Gateway, "System prompt here.");
        let pairs = extract_text_messages(&[system_msg]);
        assert!(
            pairs.is_empty(),
            "system+gateway message should be filtered"
        );
    }

    #[test]
    fn test_extract_text_messages_keeps_client_system_messages() {
        // System messages from Source::Client (not gateway activity) are NOT filtered.
        let messages = vec![
            make_weft_message(Role::System, Source::Client, "Additional instructions"),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let pairs = extract_text_messages(&messages);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, Role::System);
        assert_eq!(pairs[0].1, "Additional instructions");
    }

    #[test]
    fn test_extract_text_messages_multiple_text_parts_concatenated() {
        // Multiple Text content parts in one message are joined with newlines.
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![
                ContentPart::Text("Part one".to_string()),
                ContentPart::Text("Part two".to_string()),
            ],
            delta: false,
            message_index: 0,
        };
        let pairs = extract_text_messages(&[msg]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].1, "Part one\nPart two");
    }

    #[test]
    fn test_extract_text_messages_command_result_extracted() {
        use weft_core::CommandResultContent;
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Tool,
            model: None,
            content: vec![ContentPart::CommandResult(CommandResultContent {
                command: "search".to_string(),
                success: true,
                output: "Found 3 results.".to_string(),
                error: None,
            })],
            delta: false,
            message_index: 0,
        };
        let pairs = extract_text_messages(&[msg]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].1, "Found 3 results.");
    }

    #[test]
    fn test_extract_text_messages_non_text_content_omitted() {
        use weft_core::{MediaContent, MediaSource};
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Image(MediaContent {
                source: MediaSource::Url("https://example.com/img.png".to_string()),
                media_type: Some("image/png".to_string()),
            })],
            delta: false,
            message_index: 0,
        };
        // Image-only message produces empty text, so filtered out
        let pairs = extract_text_messages(&[msg]);
        assert!(
            pairs.is_empty(),
            "image-only message should be filtered (no text)"
        );
    }

    #[test]
    fn test_extract_text_messages_empty_text_messages_omitted() {
        // Messages whose text content is empty after extraction are omitted.
        let msg = make_weft_message_no_content(Role::User, Source::Client);
        let pairs = extract_text_messages(&[msg]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_extract_text_messages_empty_input() {
        let pairs = extract_text_messages(&[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_extract_text_messages_preserves_roles() {
        let messages = vec![
            make_weft_message(Role::User, Source::Client, "user text"),
            make_weft_message(Role::Assistant, Source::Provider, "assistant text"),
        ];
        let pairs = extract_text_messages(&messages);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, Role::User);
        assert_eq!(pairs[1].0, Role::Assistant);
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
