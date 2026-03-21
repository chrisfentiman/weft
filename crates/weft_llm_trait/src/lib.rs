//! `weft_llm_trait` — LLM provider trait and associated types.
//!
//! This crate contains the trait contract for LLM providers: `Provider`,
//! `ProviderService`, and all request/response/error types. The implementations
//! live in `weft_llm`, which depends on this crate.
//!
//! Consumers that need the trait boundary without the implementation (e.g.
//! `weft_reactor_trait`) depend on this crate directly.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};

// ── SingleChunkStream (helper for default execute_stream) ──────────────────

/// A stream that yields exactly one item, then ends.
///
/// Used by the default `Provider::execute_stream` implementation to wrap a
/// complete non-streaming response without requiring the `futures` crate.
struct SingleChunkStream<T> {
    item: Option<T>,
}

impl<T: Unpin> Stream for SingleChunkStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<T>> {
        Poll::Ready(self.item.take())
    }
}

// ── Capability ─────────────────────────────────────────────────────────────

/// A capability that a model supports.
///
/// Capabilities are strings, not an enum, because the set is open — users can
/// declare arbitrary capability strings for providers we don't know about.
///
/// Well-known capability strings are provided as constants on `Capability` for
/// type safety in compiled code. Config-declared capabilities are free-form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Capability(String);

impl Capability {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    // Well-known capability constants.
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

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Request / Response types ───────────────────────────────────────────────

/// A request to a provider. Variants cover all capability types.
#[derive(Debug, Clone)]
pub enum ProviderRequest {
    /// Chat completion (text generation).
    ///
    /// The system prompt, if present, is `messages[0]` with `Role::System`.
    /// This is a positional convention set by the gateway during context assembly.
    ChatCompletion {
        messages: Vec<WeftMessage>,
        model: String,
        options: SamplingOptions,
    },
}

/// A response from a provider.
#[derive(Debug, Clone)]
pub enum ProviderResponse {
    /// Chat completion response.
    ChatCompletion {
        message: WeftMessage,
        usage: Option<TokenUsage>,
    },
}

/// Token usage reported by a provider.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// A chunk from a streaming provider response.
#[derive(Debug, Clone)]
pub enum ProviderChunk {
    /// A content delta from a streaming provider.
    Delta(ContentDelta),
    /// The complete response (last chunk or sole chunk for non-streaming providers).
    Complete(ProviderResponse),
}

/// A content delta yielded by a streaming provider.
#[derive(Debug, Clone)]
pub struct ContentDelta {
    pub text: Option<String>,
    pub tool_call_delta: Option<ToolCallDelta>,
}

/// A partial tool call delta from a streaming provider.
#[derive(Debug, Clone)]
pub struct ToolCallDelta {
    pub index: usize,
    pub name_fragment: Option<String>,
    pub arguments_fragment: Option<String>,
}

// ── ProviderError ──────────────────────────────────────────────────────────

/// Error type for provider operations.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("request failed: {0}")]
    RequestFailed(String),
    #[error("provider error: status={status}, body={body}")]
    ProviderHttpError { status: u16, body: String },
    #[error("deserialization failed: {0}")]
    DeserializationError(String),
    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("wire script error: {script}: {message}")]
    WireScriptError { script: String, message: String },
}

// ── Provider trait ─────────────────────────────────────────────────────────

/// A universal AI provider.
///
/// Providers handle requests they support and return `ProviderError::Unsupported`
/// for request types they cannot handle.
///
/// `Send + Sync + 'static`: shared across request handlers via `Arc`.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Execute a provider request and return the response.
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError>;

    /// Execute a request and return a stream of response chunks.
    ///
    /// Providers with true streaming override this method. Providers without
    /// streaming use the default implementation, which wraps `execute()` in a
    /// single-element `Complete` stream so `GenerateActivity` has a unified path.
    async fn execute_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        let response = self.execute(request).await?;
        // Default: wrap the single response as a one-element Complete stream.
        // Uses a manual stream implementation to avoid depending on `futures` crate.
        Ok(Box::pin(SingleChunkStream {
            item: Some(Ok(ProviderChunk::Complete(response))),
        }))
    }

    /// The name of this provider instance, for logging and diagnostics.
    fn name(&self) -> &str;
}

// ── ProviderService trait ──────────────────────────────────────────────────

/// Provider service trait. Abstracts the provider registry for the engine.
///
/// The engine uses this to look up providers, model identifiers, token limits,
/// and capabilities. `ProviderRegistry` in `weft_llm` is the production implementation.
///
/// `Send + Sync + 'static`: shared via `Arc` across async request handlers.
pub trait ProviderService: Send + Sync + 'static {
    fn get(&self, name: &str) -> &Arc<dyn Provider>;
    fn model_id(&self, name: &str) -> Option<&str>;
    fn max_tokens_for(&self, name: &str) -> Option<u32>;
    fn default_provider(&self) -> &Arc<dyn Provider>;
    fn default_name(&self) -> &str;
    fn models_with_capability(&self, capability: &Capability) -> &HashSet<String>;
    fn model_has_capability(&self, model_name: &str, capability: &Capability) -> bool;
    fn model_capabilities(&self, model_name: &str) -> Option<&HashSet<Capability>>;
}

// ── extract_text_messages ──────────────────────────────────────────────────

/// Extract text content from WeftMessages for provider wire format conversion.
///
/// Filters out gateway activity messages (source: Gateway, role: System) that
/// contain NO text content — these are gateway-internal telemetry that must not
/// reach the LLM.
pub fn extract_text_messages(messages: &[WeftMessage]) -> Vec<(Role, String)> {
    messages
        .iter()
        .filter(|m| {
            !(m.role == Role::System
                && m.source == Source::Gateway
                && !m.content.iter().any(|p| matches!(p, ContentPart::Text(_))))
        })
        .map(|m| {
            let text: String = m
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(t) => Some(t.as_str()),
                    ContentPart::CommandResult(cr) => Some(cr.output.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (m.role, text)
        })
        .filter(|(_, text)| !text.is_empty())
        .collect()
}
