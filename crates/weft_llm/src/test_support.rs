//! Test doubles for the LLM provider traits.
//!
//! Gated behind `feature = "test-support"`. Available to downstream crates
//! via `weft_llm = { ..., features = ["test-support"] }` in `[dev-dependencies]`.
//!
//! **Available stubs:**
//! - [`StubProvider`] — returns a fixed text response on every call
//! - [`SingleUseErrorProvider`] — returns a `ProviderError` on first call, then `Unsupported`
//! - [`SlowProvider`] — delays before responding (for heartbeat tests with `tokio::time::pause`)
//! - [`ChunkStreamProvider`] — yields delta chunks then a `Complete` (for streaming tests)
//! - [`MidStreamErrorProvider`] — yields one delta then errors (for mid-stream error tests)
//! - [`StubProviderService`] — a `ProviderService` backed by a single stub provider

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use weft_core::{ContentPart, Role, Source, WeftMessage};

use crate::{
    Capability, ContentDelta, Provider, ProviderChunk, ProviderError, ProviderRequest,
    ProviderResponse, ProviderService, TokenUsage,
};

// ── StubProvider ──────────────────────────────────────────────────────────────

/// A provider that returns a fixed text response on every call.
///
/// Implements both `execute` (non-streaming) and `execute_stream` (default delegation).
/// Useful for any test that needs the provider to succeed with known output.
pub struct StubProvider {
    /// The text content of the assistant response.
    pub response_text: String,
}

impl StubProvider {
    /// Construct a new stub with the given response text.
    pub fn new(response_text: impl Into<String>) -> Self {
        Self {
            response_text: response_text.into(),
        }
    }
}

#[async_trait]
impl Provider for StubProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(self.response_text.clone())],
            delta: false,
            message_index: 0,
        };
        Ok(ProviderResponse::ChatCompletion {
            message,
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        })
    }

    fn name(&self) -> &str {
        "stub-provider"
    }
}

// ── SingleUseErrorProvider ────────────────────────────────────────────────────

// ProviderError doesn't implement Clone, so we store the error in a Mutex<Option<ProviderError>>
// and take it once. Subsequent calls return Unsupported. In practice, each test creates a fresh
// Services, so the provider is called at most once.

/// A provider that returns a `ProviderError` on the first call, then `Unsupported`.
///
/// Used in tests that verify error paths in `GenerateActivity`. Create a fresh instance
/// per test — it consumes the error on first call.
pub struct SingleUseErrorProvider {
    /// The error to return on the first call. Taken once via `Mutex<Option<...>>`.
    pub error: std::sync::Mutex<Option<ProviderError>>,
    /// Fallback error message for calls after the first.
    pub fallback_msg: String,
}

impl SingleUseErrorProvider {
    /// Construct a provider that returns `error` on the first call.
    pub fn new(error: ProviderError) -> Self {
        let fallback_msg = error.to_string();
        Self {
            error: std::sync::Mutex::new(Some(error)),
            fallback_msg,
        }
    }
}

#[async_trait]
impl Provider for SingleUseErrorProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let err = self
            .error
            .lock()
            .expect("SingleUseErrorProvider mutex poisoned")
            .take();
        match err {
            Some(e) => Err(e),
            None => Err(ProviderError::Unsupported(self.fallback_msg.clone())),
        }
    }

    fn name(&self) -> &str {
        "error-provider"
    }
}

// ── SlowProvider ──────────────────────────────────────────────────────────────

/// A provider that sleeps for a configurable duration before returning a response.
///
/// Used in tests that need the provider call to take time (e.g., heartbeat tests).
/// Only useful with `tokio::time::pause()` + `tokio::time::advance()`.
pub struct SlowProvider {
    /// How long to delay before responding, in seconds.
    pub delay_secs: u64,
    /// The text content to return in the response.
    pub response_text: String,
}

impl SlowProvider {
    /// Construct a provider that delays for `delay_secs` before returning `response_text`.
    pub fn new(delay_secs: u64, response_text: impl Into<String>) -> Self {
        Self {
            delay_secs,
            response_text: response_text.into(),
        }
    }
}

#[async_trait]
impl Provider for SlowProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        tokio::time::sleep(tokio::time::Duration::from_secs(self.delay_secs)).await;
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("stub-model".to_string()),
            content: vec![ContentPart::Text(self.response_text.clone())],
            delta: false,
            message_index: 0,
        };
        Ok(ProviderResponse::ChatCompletion {
            message,
            usage: None,
        })
    }

    fn name(&self) -> &str {
        "slow-provider"
    }
}

// ── ChunkStreamProvider ───────────────────────────────────────────────────────

/// A provider that yields a fixed sequence of text delta chunks, then a `Complete`.
///
/// Used in tests that verify `GenerateActivity` pushes `Generated(Content)` events
/// as each chunk arrives rather than buffering the full response.
pub struct ChunkStreamProvider {
    /// Text chunks to yield as `ProviderChunk::Delta` before the `Complete`.
    pub chunks: Vec<String>,
    /// Whether to include a `Complete` chunk at the end.
    ///
    /// When `true`, the last chunk is `Complete` carrying the concatenated text
    /// and usage data. When `false`, the stream ends after all delta chunks
    /// (simulates a provider that yields only deltas).
    pub include_complete: bool,
}

impl ChunkStreamProvider {
    /// Construct a provider that streams `chunks` with an optional `Complete` at the end.
    pub fn new(chunks: Vec<String>, include_complete: bool) -> Self {
        Self {
            chunks,
            include_complete,
        }
    }
}

#[async_trait]
impl Provider for ChunkStreamProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        // Assemble a response from all chunks for the non-streaming path.
        let text: String = self.chunks.join("");
        let message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("chunk-stream-model".to_string()),
            content: vec![ContentPart::Text(text)],
            delta: false,
            message_index: 0,
        };
        Ok(ProviderResponse::ChatCompletion {
            message,
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: self.chunks.len() as u32 * 2,
            }),
        })
    }

    async fn execute_stream(
        &self,
        _request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        let chunks = self.chunks.clone();
        let include_complete = self.include_complete;

        // Collect all items into a Vec upfront, then convert to a stream.
        let mut items: Vec<Result<ProviderChunk, ProviderError>> = chunks
            .iter()
            .map(|text| {
                Ok(ProviderChunk::Delta(ContentDelta {
                    text: Some(text.clone()),
                    tool_call_delta: None,
                }))
            })
            .collect();

        if include_complete {
            let full_text: String = chunks.join("");
            let message = WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("chunk-stream-model".to_string()),
                content: vec![ContentPart::Text(full_text)],
                delta: false,
                message_index: 0,
            };
            let response = ProviderResponse::ChatCompletion {
                message,
                usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: chunks.len() as u32 * 2,
                }),
            };
            items.push(Ok(ProviderChunk::Complete(response)));
        }

        Ok(Box::pin(futures::stream::iter(items)))
    }

    fn name(&self) -> &str {
        "chunk-stream-provider"
    }
}

// ── MidStreamErrorProvider ────────────────────────────────────────────────────

/// A provider whose stream yields one delta then errors.
///
/// Used in tests that verify `GenerateActivity` pushes `GenerationFailed` when
/// a mid-stream error occurs after partial content was already emitted.
pub struct MidStreamErrorProvider {
    /// The text chunk to emit before the error.
    pub first_chunk: String,
    /// The error to return as the second item. Taken once via `Mutex<Option<...>>`.
    pub error: std::sync::Mutex<Option<ProviderError>>,
    /// Fallback error message for calls after the first.
    pub fallback_error_msg: String,
}

impl MidStreamErrorProvider {
    /// Construct a provider that emits `first_chunk` then errors with `error`.
    pub fn new(first_chunk: impl Into<String>, error: ProviderError) -> Self {
        let fallback_error_msg = error.to_string();
        Self {
            first_chunk: first_chunk.into(),
            error: std::sync::Mutex::new(Some(error)),
            fallback_error_msg,
        }
    }
}

#[async_trait]
impl Provider for MidStreamErrorProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        Err(ProviderError::Unsupported(
            "mid-stream-error-provider: use execute_stream".to_string(),
        ))
    }

    async fn execute_stream(
        &self,
        _request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ProviderChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        let first = self.first_chunk.clone();
        let err = self
            .error
            .lock()
            .expect("MidStreamErrorProvider mutex poisoned")
            .take()
            .unwrap_or_else(|| ProviderError::Unsupported(self.fallback_error_msg.clone()));

        let items: Vec<Result<ProviderChunk, ProviderError>> = vec![
            Ok(ProviderChunk::Delta(ContentDelta {
                text: Some(first),
                tool_call_delta: None,
            })),
            Err(err),
        ];
        Ok(Box::pin(futures::stream::iter(items)))
    }

    fn name(&self) -> &str {
        "mid-stream-error-provider"
    }
}

// ── StubProviderService ───────────────────────────────────────────────────────

// `ProviderService::models_with_capability` returns `&HashSet<String>`, requiring
// the set to live long enough. We store an empty set in the service for the fallback.

/// A `ProviderService` backed by a single stub provider.
///
/// Registers the stub under the routing name `"stub-model"` with `Capability::CHAT_COMPLETIONS`.
/// All `get()` calls return the same stub provider regardless of the requested name.
pub struct StubProviderService {
    provider: Arc<dyn Provider>,
    default: String,
    capabilities: HashMap<String, HashSet<Capability>>,
    capability_index: HashMap<Capability, HashSet<String>>,
    empty_string_set: HashSet<String>,
}

impl StubProviderService {
    /// Construct a service wrapping `provider` as the sole stub backend.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        let default = "stub-model".to_string();
        let chat_cap = Capability::new(Capability::CHAT_COMPLETIONS);

        let mut capabilities = HashMap::new();
        let mut cap_set = HashSet::new();
        cap_set.insert(chat_cap.clone());
        capabilities.insert(default.clone(), cap_set);

        let mut capability_index: HashMap<Capability, HashSet<String>> = HashMap::new();
        let mut model_set = HashSet::new();
        model_set.insert(default.clone());
        capability_index.insert(chat_cap, model_set);

        Self {
            provider,
            default,
            capabilities,
            capability_index,
            empty_string_set: HashSet::new(),
        }
    }
}

impl ProviderService for StubProviderService {
    fn get(&self, _name: &str) -> &Arc<dyn Provider> {
        // Return the single stub provider regardless of name.
        &self.provider
    }

    fn model_id(&self, name: &str) -> Option<&str> {
        if name == self.default {
            Some("stub-model-v1")
        } else {
            None
        }
    }

    fn max_tokens_for(&self, name: &str) -> Option<u32> {
        if name == self.default {
            Some(4096)
        } else {
            None
        }
    }

    fn default_provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    fn default_name(&self) -> &str {
        &self.default
    }

    fn models_with_capability(&self, capability: &Capability) -> &HashSet<String> {
        self.capability_index
            .get(capability)
            .unwrap_or(&self.empty_string_set)
    }

    fn model_has_capability(&self, model_name: &str, capability: &Capability) -> bool {
        self.capabilities
            .get(model_name)
            .map(|caps| caps.contains(capability))
            .unwrap_or(false)
    }

    fn model_capabilities(&self, model_name: &str) -> Option<&HashSet<Capability>> {
        self.capabilities.get(model_name)
    }
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use weft_core::SamplingOptions;

    fn chat_request() -> ProviderRequest {
        ProviderRequest::ChatCompletion {
            messages: vec![],
            model: "stub-model".to_string(),
            options: SamplingOptions::default(),
        }
    }

    #[tokio::test]
    async fn stub_provider_returns_response() {
        let provider = StubProvider::new("test response");
        let result = provider.execute(chat_request()).await;
        assert!(result.is_ok());
        if let Ok(ProviderResponse::ChatCompletion { message, .. }) = result {
            assert!(matches!(message.content[0], ContentPart::Text(ref t) if t == "test response"));
        } else {
            panic!("expected ChatCompletion response");
        }
    }

    #[tokio::test]
    async fn stub_provider_name_is_stub_provider() {
        let provider = StubProvider::new("anything");
        assert_eq!(provider.name(), "stub-provider");
    }

    #[tokio::test]
    async fn single_use_error_provider_returns_error_then_unsupported() {
        let provider = SingleUseErrorProvider::new(ProviderError::RateLimited {
            retry_after_ms: 1000,
        });
        let first = provider.execute(chat_request()).await;
        assert!(first.is_err());
        assert!(matches!(
            first.unwrap_err(),
            ProviderError::RateLimited { .. }
        ));

        // Second call returns Unsupported (the original error was taken).
        let second = provider.execute(chat_request()).await;
        assert!(second.is_err());
        assert!(matches!(second.unwrap_err(), ProviderError::Unsupported(_)));
    }

    #[tokio::test]
    async fn slow_provider_returns_response_after_delay() {
        tokio::time::pause();
        let provider = SlowProvider::new(5, "delayed response");
        let handle = tokio::spawn(async move { provider.execute(chat_request()).await });
        tokio::time::advance(tokio::time::Duration::from_secs(5)).await;
        let result = handle.await.expect("task panicked");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn chunk_stream_provider_streams_chunks() {
        let provider =
            ChunkStreamProvider::new(vec!["hello".to_string(), " world".to_string()], true);
        let stream = provider.execute_stream(chat_request()).await;
        assert!(stream.is_ok());

        let mut stream = stream.unwrap();
        use futures::StreamExt;
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        // 2 delta chunks + 1 complete chunk.
        assert_eq!(items.len(), 3);
        assert!(items[0].is_ok());
        assert!(matches!(
            items[0].as_ref().unwrap(),
            ProviderChunk::Delta(_)
        ));
        assert!(matches!(
            items[2].as_ref().unwrap(),
            ProviderChunk::Complete(_)
        ));
    }

    #[tokio::test]
    async fn chunk_stream_provider_without_complete_yields_only_deltas() {
        let provider = ChunkStreamProvider::new(vec!["a".to_string(), "b".to_string()], false);
        let stream = provider.execute_stream(chat_request()).await.unwrap();
        use futures::StreamExt;
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 2);
        assert!(
            items
                .iter()
                .all(|i| matches!(i, Ok(ProviderChunk::Delta(_))))
        );
    }

    #[tokio::test]
    async fn mid_stream_error_provider_yields_delta_then_error() {
        let provider =
            MidStreamErrorProvider::new("partial", ProviderError::Unsupported("test".to_string()));
        let stream = provider.execute_stream(chat_request()).await.unwrap();
        use futures::StreamExt;
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], Ok(ProviderChunk::Delta(_))));
        assert!(items[1].is_err());
    }

    #[test]
    fn stub_provider_service_default_name_is_stub_model() {
        let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("hi"));
        let svc = StubProviderService::new(provider);
        assert_eq!(svc.default_name(), "stub-model");
    }

    #[test]
    fn stub_provider_service_has_chat_completions_capability() {
        let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("hi"));
        let svc = StubProviderService::new(provider);
        let cap = Capability::new(Capability::CHAT_COMPLETIONS);
        assert!(svc.model_has_capability("stub-model", &cap));
        let models = svc.models_with_capability(&cap);
        assert!(models.contains("stub-model"));
    }

    #[test]
    fn stub_provider_service_unknown_model_returns_none() {
        let provider: Arc<dyn Provider> = Arc::new(StubProvider::new("hi"));
        let svc = StubProviderService::new(provider);
        assert!(svc.model_id("unknown").is_none());
        assert!(svc.max_tokens_for("unknown").is_none());
        assert!(svc.model_capabilities("unknown").is_none());
    }
}
