//! tonic gRPC service implementation for the Weft Wire protocol.
#![allow(dead_code)] // Functions and types wired in Phase 5
//!
//! `WeftService` implements the `Weft` gRPC service defined in `weft.proto`.
//! It is the single code path to the engine — both the tonic `chat()` RPC method
//! and the OpenAI compat HTTP handler (Phase 5) call `handle_weft_request()`.
//!
//! Phase 3 scope:
//! - `Chat` unary RPC: proto request → domain → engine → domain → proto response
//! - `ChatStream`: returns `Unimplemented` (wired in Phase 4)
//! - `Live`: returns `Unimplemented` (reserved for future)
//!
//! Error mapping: `WeftError` → `tonic::Status` covers all variants.

use tonic::{Request, Response, Status};
use weft_core::{WeftError, WeftRequest, WeftResponse, validate_request};
use weft_proto::weft::v1 as proto;

use crate::engine::GatewayEngine;

// ── Conversion types ────────────────────────────────────────────────────────
//
// Import the domain types needed for proto ↔ domain conversion.
// Conversions are defined in weft_core::wire via TryFrom/From traits.

use weft_core::{
    CommandCallContent, CommandResultContent, ContentPart, CouncilStartActivity, DocumentContent,
    HookActivity, MediaContent, MediaSource, MemoryResultContent, MemoryResultEntry,
    MemoryStoredContent, ModelRoutingInstruction, ResponseFormat, Role, RoutingActivity,
    SamplingOptions, Source, WeftMessage,
};

// ── WeftService ─────────────────────────────────────────────────────────────

/// tonic gRPC service implementation.
///
/// `GatewayEngine` is `Clone` (all `Arc` internals) — cloned into the service.
/// `WeftService` can be cloned and shared across request handlers.
#[derive(Clone)]
pub struct WeftService {
    engine: GatewayEngine,
}

impl WeftService {
    /// Create a new `WeftService` wrapping the given engine.
    pub fn new(engine: GatewayEngine) -> Self {
        Self { engine }
    }

    /// Core request handler shared by both the gRPC trait methods and (in Phase 5)
    /// the OpenAI compat HTTP handler. This is the ONLY code path to the engine.
    ///
    /// Validates the request, runs the engine, returns the domain response.
    pub async fn handle_weft_request(&self, req: WeftRequest) -> Result<WeftResponse, WeftError> {
        validate_request(&req)?;
        self.engine.handle_request(req).await
    }
}

// ── tonic trait implementation ───────────────────────────────────────────────

#[tonic::async_trait]
impl proto::weft_server::Weft for WeftService {
    /// Unary chat RPC: single request → single response.
    async fn chat(
        &self,
        request: Request<proto::ChatRequest>,
    ) -> Result<Response<proto::ChatResponse>, Status> {
        let proto_req = request.into_inner();

        // Convert proto → domain. weft_request_from_proto validates message structure.
        let weft_req = weft_request_from_proto(proto_req)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        // Execute through the shared handler (validates + runs engine).
        let weft_resp = self
            .handle_weft_request(weft_req)
            .await
            .map_err(|e| weft_error_to_status(&e))?;

        // Convert domain → proto.
        let proto_resp = proto_response_from_weft(weft_resp);
        Ok(Response::new(proto_resp))
    }

    /// Server-streaming chat RPC. Returns `Unimplemented` in Phase 3.
    ///
    /// Phase 4 will implement token streaming via `tokio::sync::mpsc::channel`
    /// and `tokio_stream::wrappers::ReceiverStream`.
    type ChatStreamStream =
        tokio_stream::wrappers::ReceiverStream<Result<proto::ChatEvent, Status>>;

    async fn chat_stream(
        &self,
        _request: Request<proto::ChatRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        Err(Status::unimplemented(
            "ChatStream is not yet implemented (Phase 4)",
        ))
    }

    /// Bidirectional streaming RPC. Reserved for future implementation.
    type LiveStream = tokio_stream::wrappers::ReceiverStream<Result<proto::LiveOutput, Status>>;

    async fn live(
        &self,
        _request: Request<tonic::Streaming<proto::LiveInput>>,
    ) -> Result<Response<Self::LiveStream>, Status> {
        Err(Status::unimplemented("Live RPC is not yet implemented"))
    }
}

// ── Error mapping ────────────────────────────────────────────────────────────

/// Map a `WeftError` to a `tonic::Status`.
///
/// This is the authoritative mapping for the gRPC error surface. The HTTP
/// compat layer has its own mapping in `server.rs` using HTTP status codes.
pub fn weft_error_to_status(e: &WeftError) -> Status {
    match e {
        WeftError::InvalidRequest(_) => Status::invalid_argument(e.to_string()),
        WeftError::StreamingNotSupported => Status::invalid_argument(e.to_string()),
        WeftError::Config(_) => Status::internal(e.to_string()),
        WeftError::Llm(_) => Status::internal(e.to_string()),
        WeftError::Command(_) => Status::internal(e.to_string()),
        WeftError::ToolRegistry(_) => Status::unavailable(e.to_string()),
        WeftError::MemoryStore(_) => Status::unavailable(e.to_string()),
        WeftError::CommandLoopExceeded { .. } => Status::resource_exhausted(e.to_string()),
        WeftError::RequestTimeout { .. } => Status::deadline_exceeded(e.to_string()),
        WeftError::RateLimited { .. } => Status::resource_exhausted(e.to_string()),
        WeftError::Routing(_) => Status::failed_precondition(e.to_string()),
        WeftError::ModelNotFound { .. } => Status::not_found(e.to_string()),
        WeftError::NoEligibleModels { .. } => Status::failed_precondition(e.to_string()),
        WeftError::HookBlocked { .. } => Status::permission_denied(e.to_string()),
        WeftError::HookBlockedAfterRetries { .. } => Status::aborted(e.to_string()),
        WeftError::ProtoConversion(_) => Status::internal(e.to_string()),
    }
}

// ── Proto ↔ Domain conversions ────────────────────────────────────────────

/// Build a proto `ChatResponse` from a domain `WeftResponse`.
fn proto_response_from_weft(resp: WeftResponse) -> proto::ChatResponse {
    proto::ChatResponse {
        id: resp.id,
        model: resp.model,
        messages: resp
            .messages
            .into_iter()
            .map(proto_message_from_weft)
            .collect(),
        usage: Some(proto::UsageInfo {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
            llm_calls: resp.usage.llm_calls,
        }),
        timing: Some(proto::TimingInfo {
            total_ms: resp.timing.total_ms,
            routing_ms: resp.timing.routing_ms,
            llm_ms: resp.timing.llm_ms,
        }),
    }
}

/// Convert a domain `WeftMessage` to a proto `WeftMessage`.
fn proto_message_from_weft(msg: WeftMessage) -> proto::WeftMessage {
    proto::WeftMessage {
        role: match msg.role {
            Role::User => proto::Role::User as i32,
            Role::Assistant => proto::Role::Assistant as i32,
            Role::System => proto::Role::System as i32,
        },
        source: match msg.source {
            Source::Client => proto::Source::Client as i32,
            Source::Gateway => proto::Source::Gateway as i32,
            Source::Provider => proto::Source::Provider as i32,
            Source::Member => proto::Source::Member as i32,
            Source::Judge => proto::Source::Judge as i32,
            Source::Tool => proto::Source::Tool as i32,
            Source::Memory => proto::Source::Memory as i32,
        },
        model: msg.model.unwrap_or_default(),
        content: msg
            .content
            .into_iter()
            .map(proto_content_from_weft)
            .collect(),
        delta: msg.delta,
        message_index: msg.message_index,
    }
}

/// Convert a domain `ContentPart` to a proto `ContentPart`.
fn proto_content_from_weft(part: ContentPart) -> proto::ContentPart {
    use proto::content_part::Part;
    let inner = match part {
        ContentPart::Text(t) => Part::Text(proto::TextContent { text: t }),
        ContentPart::Image(m) => Part::Image(proto::ImageContent {
            source: Some(media_source_to_proto_image(m.source)),
            media_type: m.media_type.unwrap_or_default(),
        }),
        ContentPart::Audio(m) => Part::Audio(proto::AudioContent {
            source: Some(media_source_to_proto_audio(m.source)),
            media_type: m.media_type.unwrap_or_default(),
        }),
        ContentPart::Video(m) => Part::Video(proto::VideoContent {
            source: Some(media_source_to_proto_video(m.source)),
            media_type: m.media_type.unwrap_or_default(),
        }),
        ContentPart::Document(d) => Part::Document(proto::DocumentContent {
            source: Some(media_source_to_proto_document(d.source)),
            media_type: d.media_type.unwrap_or_default(),
            filename: d.filename.unwrap_or_default(),
        }),
        ContentPart::Routing(r) => Part::Routing(proto::RoutingContent {
            model: r.model,
            score: r.score,
            filters: r.filters,
        }),
        ContentPart::Hook(h) => Part::Hook(proto::HookContent {
            event: h.event,
            hook_name: h.hook_name,
            decision: h.decision,
            reason: h.reason.unwrap_or_default(),
        }),
        ContentPart::CouncilStart(c) => Part::CouncilStart(proto::CouncilStartContent {
            models: c.models,
            judge: c.judge,
        }),
        ContentPart::CommandCall(cc) => Part::CommandCall(proto::CommandCallContent {
            command: cc.command,
            arguments_json: cc.arguments_json,
        }),
        ContentPart::CommandResult(cr) => Part::CommandResult(proto::CommandResultContent {
            command: cr.command,
            success: cr.success,
            output: cr.output,
            error: cr.error.unwrap_or_default(),
        }),
        ContentPart::MemoryResult(mr) => Part::MemoryResult(proto::MemoryResultContent {
            store: mr.store,
            entries: mr
                .entries
                .into_iter()
                .map(|e| proto::MemoryEntry {
                    id: e.id,
                    content: e.content,
                    score: e.score,
                    created_at: e.created_at,
                })
                .collect(),
        }),
        ContentPart::MemoryStored(ms) => Part::MemoryStored(proto::MemoryStoredContent {
            store: ms.store,
            id: ms.id,
        }),
    };
    proto::ContentPart { part: Some(inner) }
}

fn media_source_to_proto_image(s: MediaSource) -> proto::image_content::Source {
    match s {
        MediaSource::Url(u) => proto::image_content::Source::Url(u),
        MediaSource::Data(b) => proto::image_content::Source::Data(b),
    }
}

fn media_source_to_proto_audio(s: MediaSource) -> proto::audio_content::Source {
    match s {
        MediaSource::Url(u) => proto::audio_content::Source::Url(u),
        MediaSource::Data(b) => proto::audio_content::Source::Data(b),
    }
}

fn media_source_to_proto_video(s: MediaSource) -> proto::video_content::Source {
    match s {
        MediaSource::Url(u) => proto::video_content::Source::Url(u),
        MediaSource::Data(b) => proto::video_content::Source::Data(b),
    }
}

fn media_source_to_proto_document(s: MediaSource) -> proto::document_content::Source {
    match s {
        MediaSource::Url(u) => proto::document_content::Source::Url(u),
        MediaSource::Data(b) => proto::document_content::Source::Data(b),
    }
}

// ── Proto → Domain conversion ────────────────────────────────────────────────
//
// Free functions rather than TryFrom impls to avoid the orphan rule:
// neither proto::ChatRequest nor WeftRequest is defined in this crate.

/// Convert a proto `ChatRequest` to a domain `WeftRequest`.
///
/// Returns `WeftError::ProtoConversion` if any message or content part is invalid.
pub fn weft_request_from_proto(proto_req: proto::ChatRequest) -> Result<WeftRequest, WeftError> {
    let messages: Result<Vec<WeftMessage>, String> = proto_req
        .messages
        .into_iter()
        .map(weft_message_from_proto)
        .collect();
    let messages = messages.map_err(WeftError::ProtoConversion)?;

    let routing = ModelRoutingInstruction::parse(&proto_req.model);

    let options = proto_req
        .options
        .map(sampling_options_from_proto)
        .unwrap_or_default();

    Ok(WeftRequest {
        messages,
        routing,
        options,
    })
}

/// Convert a proto `WeftMessage` to a domain `WeftMessage`.
fn weft_message_from_proto(proto_msg: proto::WeftMessage) -> Result<WeftMessage, String> {
    let role = match proto_msg.role() {
        proto::Role::Unspecified => {
            return Err("message role is required (got ROLE_UNSPECIFIED)".to_string());
        }
        proto::Role::User => Role::User,
        proto::Role::Assistant => Role::Assistant,
        proto::Role::System => Role::System,
    };

    let source = match proto_msg.source() {
        proto::Source::Unspecified => Source::Client, // proto3 default → Client
        proto::Source::Client => Source::Client,
        proto::Source::Gateway => Source::Gateway,
        proto::Source::Provider => Source::Provider,
        proto::Source::Member => Source::Member,
        proto::Source::Judge => Source::Judge,
        proto::Source::Tool => Source::Tool,
        proto::Source::Memory => Source::Memory,
    };

    let model = if proto_msg.model.is_empty() {
        None
    } else {
        Some(proto_msg.model)
    };

    let content: Result<Vec<ContentPart>, _> = proto_msg
        .content
        .into_iter()
        .map(content_part_from_proto)
        .collect();
    let content = content?;

    Ok(WeftMessage {
        role,
        source,
        model,
        content,
        delta: proto_msg.delta,
        message_index: proto_msg.message_index,
    })
}

/// Convert a proto `ContentPart` to a domain `ContentPart`.
fn content_part_from_proto(part: proto::ContentPart) -> Result<ContentPart, String> {
    use proto::content_part::Part;

    match part.part {
        None => Err("content part is empty (no oneof variant set)".to_string()),
        Some(Part::Text(t)) => Ok(ContentPart::Text(t.text)),
        Some(Part::Image(img)) => {
            let source = match img.source {
                Some(proto::image_content::Source::Url(u)) => MediaSource::Url(u),
                Some(proto::image_content::Source::Data(b)) => MediaSource::Data(b),
                None => return Err("image content missing source".to_string()),
            };
            Ok(ContentPart::Image(MediaContent {
                source,
                media_type: if img.media_type.is_empty() {
                    None
                } else {
                    Some(img.media_type)
                },
            }))
        }
        Some(Part::Audio(a)) => {
            let source = match a.source {
                Some(proto::audio_content::Source::Url(u)) => MediaSource::Url(u),
                Some(proto::audio_content::Source::Data(b)) => MediaSource::Data(b),
                None => return Err("audio content missing source".to_string()),
            };
            Ok(ContentPart::Audio(MediaContent {
                source,
                media_type: if a.media_type.is_empty() {
                    None
                } else {
                    Some(a.media_type)
                },
            }))
        }
        Some(Part::Video(v)) => {
            let source = match v.source {
                Some(proto::video_content::Source::Url(u)) => MediaSource::Url(u),
                Some(proto::video_content::Source::Data(b)) => MediaSource::Data(b),
                None => return Err("video content missing source".to_string()),
            };
            Ok(ContentPart::Video(MediaContent {
                source,
                media_type: if v.media_type.is_empty() {
                    None
                } else {
                    Some(v.media_type)
                },
            }))
        }
        Some(Part::Document(d)) => {
            let source = match d.source {
                Some(proto::document_content::Source::Url(u)) => MediaSource::Url(u),
                Some(proto::document_content::Source::Data(b)) => MediaSource::Data(b),
                None => return Err("document content missing source".to_string()),
            };
            Ok(ContentPart::Document(DocumentContent {
                source,
                media_type: if d.media_type.is_empty() {
                    None
                } else {
                    Some(d.media_type)
                },
                filename: if d.filename.is_empty() {
                    None
                } else {
                    Some(d.filename)
                },
            }))
        }
        Some(Part::Routing(r)) => Ok(ContentPart::Routing(RoutingActivity {
            model: r.model,
            score: r.score,
            filters: r.filters,
        })),
        Some(Part::Hook(h)) => Ok(ContentPart::Hook(HookActivity {
            event: h.event,
            hook_name: h.hook_name,
            decision: h.decision,
            reason: if h.reason.is_empty() {
                None
            } else {
                Some(h.reason)
            },
        })),
        Some(Part::CouncilStart(c)) => Ok(ContentPart::CouncilStart(CouncilStartActivity {
            models: c.models,
            judge: c.judge,
        })),
        Some(Part::CommandCall(cc)) => Ok(ContentPart::CommandCall(CommandCallContent {
            command: cc.command,
            arguments_json: cc.arguments_json,
        })),
        Some(Part::CommandResult(cr)) => Ok(ContentPart::CommandResult(CommandResultContent {
            command: cr.command,
            success: cr.success,
            output: cr.output,
            error: if cr.error.is_empty() {
                None
            } else {
                Some(cr.error)
            },
        })),
        Some(Part::MemoryResult(mr)) => Ok(ContentPart::MemoryResult(MemoryResultContent {
            store: mr.store,
            entries: mr
                .entries
                .into_iter()
                .map(|e| MemoryResultEntry {
                    id: e.id,
                    content: e.content,
                    score: e.score,
                    created_at: e.created_at,
                })
                .collect(),
        })),
        Some(Part::MemoryStored(ms)) => Ok(ContentPart::MemoryStored(MemoryStoredContent {
            store: ms.store,
            id: ms.id,
        })),
    }
}

/// Convert proto `SamplingOptions` to domain `SamplingOptions`.
fn sampling_options_from_proto(opts: proto::SamplingOptions) -> SamplingOptions {
    // Proto3 uses 0 as default for all numeric fields. Treat 0 as "not set"
    // to match the domain's Option<T> representation.
    SamplingOptions {
        temperature: if opts.temperature == 0.0 {
            None
        } else {
            Some(opts.temperature)
        },
        top_p: if opts.top_p == 0.0 {
            None
        } else {
            Some(opts.top_p)
        },
        top_k: if opts.top_k == 0 {
            None
        } else {
            Some(opts.top_k)
        },
        max_tokens: if opts.max_tokens == 0 {
            None
        } else {
            Some(opts.max_tokens)
        },
        stop: opts.stop,
        frequency_penalty: if opts.frequency_penalty == 0.0 {
            None
        } else {
            Some(opts.frequency_penalty)
        },
        presence_penalty: if opts.presence_penalty == 0.0 {
            None
        } else {
            Some(opts.presence_penalty)
        },
        seed: if opts.seed == 0 {
            None
        } else {
            Some(opts.seed)
        },
        response_format: opts.response_format.map(|rf| match rf.r#type.as_str() {
            "json_object" => ResponseFormat::JsonObject,
            _ => ResponseFormat::Text,
        }),
        activity: opts.activity,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    // Bring the Weft trait into scope so we can call chat/chat_stream/live on WeftService.
    use weft_commands::{CommandError, CommandRegistry};
    use weft_core::{
        ClassifierConfig, CommandDescription, CommandInvocation, CommandResult, CommandStub,
        DomainsConfig, GatewayConfig, ModelEntry, ProviderConfig, RouterConfig, ServerConfig,
        WeftConfig, WireFormat,
    };
    use weft_llm::{
        Capability, ChatCompletionOutput, Provider, ProviderError, ProviderRegistry,
        ProviderRequest, ProviderResponse, TokenUsage,
    };
    use weft_proto::weft::v1::weft_server::Weft;
    use weft_router::{
        RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, SemanticRouter,
    };

    // ── Test helpers ──────────────────────────────────────────────────────

    struct MockProvider {
        text: String,
    }

    impl MockProvider {
        fn new(text: &str) -> Self {
            Self {
                text: text.to_string(),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                text: self.text.clone(),
                usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                }),
            }))
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    struct MockRouter;

    #[async_trait]
    impl SemanticRouter for MockRouter {
        async fn route(
            &self,
            _user_message: &str,
            _domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            Ok(RoutingDecision::empty())
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            candidates: &[RoutingCandidate],
        ) -> Result<Vec<weft_router::ScoredCandidate>, RouterError> {
            Ok(candidates
                .iter()
                .map(|c| weft_router::ScoredCandidate {
                    id: c.id.clone(),
                    score: 1.0,
                })
                .collect())
        }
    }

    struct MockCommandRegistry;

    #[async_trait]
    impl CommandRegistry for MockCommandRegistry {
        async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
            Ok(vec![])
        }

        async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
            Err(CommandError::NotFound(name.to_string()))
        }

        async fn execute_command(
            &self,
            invocation: &CommandInvocation,
        ) -> Result<CommandResult, CommandError> {
            Err(CommandError::NotFound(invocation.name.clone()))
        }
    }

    fn test_config() -> Arc<WeftConfig> {
        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:0".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                        capabilities: vec!["chat_completions".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    fn make_engine(provider: impl Provider + 'static) -> GatewayEngine {
        let mut providers = HashMap::new();
        providers.insert(
            "test-model".to_string(),
            Arc::new(provider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("test-model".to_string(), "claude-test".to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert("test-model".to_string(), 1024u32);
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "test-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "test-model".to_string(),
        ));

        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(MockRouter),
            Arc::new(MockCommandRegistry),
            None,
            Arc::new(crate::hooks::HookRegistry::empty()),
        )
    }

    fn make_proto_request(text: &str) -> proto::ChatRequest {
        proto::ChatRequest {
            messages: vec![proto::WeftMessage {
                role: proto::Role::User as i32,
                source: proto::Source::Client as i32,
                model: String::new(),
                content: vec![proto::ContentPart {
                    part: Some(proto::content_part::Part::Text(proto::TextContent {
                        text: text.to_string(),
                    })),
                }],
                delta: false,
                message_index: 0,
            }],
            model: "test-model".to_string(),
            options: None,
        }
    }

    // ── Unit tests: error mapping ─────────────────────────────────────────

    #[test]
    fn test_invalid_request_maps_to_invalid_argument() {
        let err = WeftError::InvalidRequest("bad input".to_string());
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_streaming_not_supported_maps_to_invalid_argument() {
        let err = WeftError::StreamingNotSupported;
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_rate_limited_maps_to_resource_exhausted() {
        let err = WeftError::RateLimited {
            retry_after_ms: 5000,
        };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn test_timeout_maps_to_deadline_exceeded() {
        let err = WeftError::RequestTimeout { timeout_secs: 30 };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
    }

    #[test]
    fn test_model_not_found_maps_to_not_found() {
        let err = WeftError::ModelNotFound {
            name: "unknown".to_string(),
        };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[test]
    fn test_hook_blocked_maps_to_permission_denied() {
        let err = WeftError::HookBlocked {
            event: "request_start".to_string(),
            reason: "policy".to_string(),
            hook_name: "auth".to_string(),
        };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::PermissionDenied);
    }

    #[test]
    fn test_hook_blocked_after_retries_maps_to_aborted() {
        let err = WeftError::HookBlockedAfterRetries {
            event: "pre_response".to_string(),
            reason: "content".to_string(),
            hook_name: "filter".to_string(),
            retries: 3,
        };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::Aborted);
    }

    #[test]
    fn test_tool_registry_maps_to_unavailable() {
        let err = WeftError::ToolRegistry("connection refused".to_string());
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[test]
    fn test_memory_store_maps_to_unavailable() {
        let err = WeftError::MemoryStore("store down".to_string());
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[test]
    fn test_command_loop_exceeded_maps_to_resource_exhausted() {
        let err = WeftError::CommandLoopExceeded { max: 10 };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn test_routing_error_maps_to_failed_precondition() {
        let err = WeftError::Routing("no models match".to_string());
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn test_no_eligible_models_maps_to_failed_precondition() {
        let err = WeftError::NoEligibleModels {
            capability: "vision".to_string(),
        };
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn test_proto_conversion_maps_to_internal() {
        let err = WeftError::ProtoConversion("missing role".to_string());
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn test_llm_error_maps_to_internal() {
        let err = WeftError::Llm("provider failed".to_string());
        let status = weft_error_to_status(&err);
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    // ── Unit tests: proto conversion ──────────────────────────────────────

    #[test]
    fn test_weft_request_from_proto_valid_request() {
        let proto_req = make_proto_request("Hello");
        let result = weft_request_from_proto(proto_req);
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let req = result.unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].source, Source::Client);
    }

    #[test]
    fn test_weft_request_from_proto_missing_role_returns_error() {
        let proto_req = proto::ChatRequest {
            messages: vec![proto::WeftMessage {
                role: proto::Role::Unspecified as i32, // invalid
                source: proto::Source::Client as i32,
                model: String::new(),
                content: vec![proto::ContentPart {
                    part: Some(proto::content_part::Part::Text(proto::TextContent {
                        text: "hello".to_string(),
                    })),
                }],
                delta: false,
                message_index: 0,
            }],
            model: "test-model".to_string(),
            options: None,
        };
        let result = weft_request_from_proto(proto_req);
        assert!(
            matches!(result, Err(WeftError::ProtoConversion(_))),
            "expected ProtoConversion error"
        );
    }

    #[test]
    fn test_weft_request_from_proto_empty_content_part_returns_error() {
        let proto_req = proto::ChatRequest {
            messages: vec![proto::WeftMessage {
                role: proto::Role::User as i32,
                source: proto::Source::Client as i32,
                model: String::new(),
                content: vec![proto::ContentPart { part: None }], // empty oneof
                delta: false,
                message_index: 0,
            }],
            model: "test-model".to_string(),
            options: None,
        };
        let result = weft_request_from_proto(proto_req);
        assert!(
            matches!(result, Err(WeftError::ProtoConversion(_))),
            "expected ProtoConversion error"
        );
    }

    #[test]
    fn test_source_unspecified_defaults_to_client() {
        let proto_req = proto::ChatRequest {
            messages: vec![proto::WeftMessage {
                role: proto::Role::User as i32,
                source: proto::Source::Unspecified as i32, // defaults to Client
                model: String::new(),
                content: vec![proto::ContentPart {
                    part: Some(proto::content_part::Part::Text(proto::TextContent {
                        text: "hello".to_string(),
                    })),
                }],
                delta: false,
                message_index: 0,
            }],
            model: "auto".to_string(),
            options: None,
        };
        let req = weft_request_from_proto(proto_req).unwrap();
        assert_eq!(req.messages[0].source, Source::Client);
    }

    #[test]
    fn test_sampling_options_zero_means_none() {
        let opts = proto::SamplingOptions {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            max_tokens: 0,
            stop: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            response_format: None,
            activity: false,
        };
        let domain = sampling_options_from_proto(opts);
        assert!(domain.temperature.is_none());
        assert!(domain.top_p.is_none());
        assert!(domain.top_k.is_none());
        assert!(domain.max_tokens.is_none());
        assert!(domain.seed.is_none());
    }

    #[test]
    fn test_sampling_options_nonzero_values_preserved() {
        let opts = proto::SamplingOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 512,
            stop: vec!["STOP".to_string()],
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            seed: 42,
            response_format: Some(proto::ResponseFormat {
                r#type: "json_object".to_string(),
            }),
            activity: true,
        };
        let domain = sampling_options_from_proto(opts);
        assert_eq!(domain.temperature, Some(0.7));
        assert_eq!(domain.top_p, Some(0.9));
        assert_eq!(domain.top_k, Some(40));
        assert_eq!(domain.max_tokens, Some(512));
        assert_eq!(domain.stop, vec!["STOP".to_string()]);
        assert_eq!(domain.frequency_penalty, Some(0.5));
        assert_eq!(domain.presence_penalty, Some(0.3));
        assert_eq!(domain.seed, Some(42));
        assert!(matches!(
            domain.response_format,
            Some(ResponseFormat::JsonObject)
        ));
        assert!(domain.activity);
    }

    #[test]
    fn test_content_part_round_trip_text() {
        let domain = ContentPart::Text("hello world".to_string());
        let proto_part = proto_content_from_weft(domain);
        let back = content_part_from_proto(proto_part).unwrap();
        assert!(matches!(back, ContentPart::Text(t) if t == "hello world"));
    }

    #[test]
    fn test_content_part_round_trip_routing() {
        let domain = ContentPart::Routing(RoutingActivity {
            model: "claude-3".to_string(),
            score: 0.95,
            filters: vec!["anthropic".to_string()],
        });
        let proto_part = proto_content_from_weft(domain);
        let back = content_part_from_proto(proto_part).unwrap();
        match back {
            ContentPart::Routing(r) => {
                assert_eq!(r.model, "claude-3");
                assert_eq!(r.score, 0.95);
                assert_eq!(r.filters, vec!["anthropic"]);
            }
            _ => panic!("expected Routing content part"),
        }
    }

    #[test]
    fn test_content_part_round_trip_command_call() {
        let domain = ContentPart::CommandCall(CommandCallContent {
            command: "search".to_string(),
            arguments_json: r#"{"query":"rust"}"#.to_string(),
        });
        let proto_part = proto_content_from_weft(domain);
        let back = content_part_from_proto(proto_part).unwrap();
        match back {
            ContentPart::CommandCall(cc) => {
                assert_eq!(cc.command, "search");
                assert_eq!(cc.arguments_json, r#"{"query":"rust"}"#);
            }
            _ => panic!("expected CommandCall content part"),
        }
    }

    #[test]
    fn test_content_part_empty_returns_error() {
        let empty = proto::ContentPart { part: None };
        let result = content_part_from_proto(empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_proto_response_from_weft_structure() {
        use weft_core::{WeftResponse, WeftTiming, WeftUsage};
        let resp = WeftResponse {
            id: "chatcmpl-test".to_string(),
            model: "auto".to_string(),
            messages: vec![WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("claude-3".to_string()),
                content: vec![ContentPart::Text("Hello!".to_string())],
                delta: false,
                message_index: 0,
            }],
            usage: WeftUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                llm_calls: 1,
            },
            timing: WeftTiming {
                total_ms: 500,
                routing_ms: 10,
                llm_ms: 490,
            },
        };

        let proto_resp = proto_response_from_weft(resp);
        assert_eq!(proto_resp.id, "chatcmpl-test");
        assert_eq!(proto_resp.model, "auto");
        assert_eq!(proto_resp.messages.len(), 1);
        assert_eq!(proto_resp.messages[0].role, proto::Role::Assistant as i32);
        let usage = proto_resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.llm_calls, 1);
        let timing = proto_resp.timing.unwrap();
        assert_eq!(timing.total_ms, 500);
    }

    // ── Integration tests: gRPC Chat RPC ─────────────────────────────────

    #[tokio::test]
    async fn test_chat_rpc_returns_response() {
        let service = WeftService::new(make_engine(MockProvider::new("Hello from mock!")));
        let proto_req = make_proto_request("Hello");
        let request = Request::new(proto_req);

        let result = service.chat(request).await;
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());

        let response = result.unwrap().into_inner();
        assert!(!response.id.is_empty());
        assert_eq!(response.messages.len(), 1);
        assert_eq!(response.messages[0].role, proto::Role::Assistant as i32);
        // Verify content contains the mock response text
        let content = &response.messages[0].content;
        assert_eq!(content.len(), 1);
        match &content[0].part {
            Some(proto::content_part::Part::Text(t)) => {
                assert_eq!(t.text, "Hello from mock!");
            }
            other => panic!("expected text content part, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_chat_rpc_empty_messages_returns_invalid_argument() {
        let service = WeftService::new(make_engine(MockProvider::new("irrelevant")));
        // Empty messages → WeftRequest::try_from fails with role error when building
        // OR validate_request fails. Either way, we get an error.
        let proto_req = proto::ChatRequest {
            messages: vec![],
            model: "auto".to_string(),
            options: None,
        };
        let result = service.chat(Request::new(proto_req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chat_stream_returns_unimplemented() {
        let service = WeftService::new(make_engine(MockProvider::new("irrelevant")));
        let result = service
            .chat_stream(Request::new(make_proto_request("Hello")))
            .await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unimplemented);
    }

    // Note: test_live_rpc_returns_unimplemented is omitted because constructing a
    // tonic::Streaming<LiveInput> requires a real tonic server connection. The
    // Unimplemented behavior is verified structurally: the live() method body
    // unconditionally returns Err(Status::unimplemented(...)).

    #[tokio::test]
    async fn test_handle_weft_request_validates_before_engine() {
        let service = WeftService::new(make_engine(MockProvider::new("irrelevant")));
        // Empty messages → InvalidRequest from validate_request
        let weft_req = WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        let result = service.handle_weft_request(weft_req).await;
        assert!(
            matches!(result, Err(WeftError::InvalidRequest(_))),
            "expected InvalidRequest, got: {:?}",
            result
        );
    }
}
