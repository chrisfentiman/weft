//! tonic gRPC service implementation for the Weft Wire protocol.
//!
//! `WeftService` implements the `Weft` gRPC service defined in `weft.proto`.
//! It is the single code path to the reactor — both the tonic `chat()` RPC method
//! and the OpenAI compat HTTP handler call `handle_weft_request()`.
//!
//! `WeftService` is constructed once in `main.rs` and shared between the gRPC
//! server and the OpenAI compat HTTP handler via `Arc<WeftService>`.
//!
//! - `Chat` unary RPC: single request → single response
//! - `ChatStream` server-streaming RPC: buffer-then-forward, activity events, delta flag
//! - `Live`: returns `Unimplemented` (reserved for future)
//!
//! Error mapping: `WeftError` → `tonic::Status` covers all variants.

use std::sync::Arc;

use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use weft_core::{WeftConfig, WeftError, WeftRequest, WeftResponse, validate_request};
use weft_proto::weft::v1 as proto;
use weft_reactor::{ActivityError, ReactorError};

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

use weft_reactor::{ExecutionContext, Reactor, RequestId, TenantId};

// ── WeftService ─────────────────────────────────────────────────────────────

/// tonic gRPC service implementation.
///
/// Holds `Arc<Reactor>` and `Arc<WeftConfig>`. Non-generic.
/// `WeftService` can be cloned and shared across request handlers.
pub struct WeftService {
    reactor: Arc<Reactor>,
    config: Arc<WeftConfig>,
}

impl Clone for WeftService {
    fn clone(&self) -> Self {
        Self {
            reactor: Arc::clone(&self.reactor),
            config: Arc::clone(&self.config),
        }
    }
}

impl WeftService {
    /// Create a new `WeftService` wrapping the given reactor.
    pub fn new(reactor: Arc<Reactor>, config: Arc<WeftConfig>) -> Self {
        Self { reactor, config }
    }

    /// Core request handler shared by both the gRPC trait methods and the OpenAI
    /// compat HTTP handler. This is the ONLY code path to the reactor.
    ///
    /// Validates the request, runs the reactor, returns the domain response.
    pub async fn handle_weft_request(&self, req: WeftRequest) -> Result<WeftResponse, WeftError> {
        validate_request(&req)?;

        let tenant_id = TenantId("default".to_string());
        let request_id = RequestId(uuid::Uuid::new_v4().to_string());

        let (result, _signal_tx) = self
            .reactor
            .execute(
                ExecutionContext {
                    request: req,
                    tenant_id,
                    request_id,
                    parent_id: None,
                    parent_budget: None,
                    client_tx: None,
                },
                None,
            )
            .await
            .map_err(reactor_error_to_weft_error)?;

        Ok(result.response)
    }

    /// Expose engine configuration for health checks and server diagnostics.
    pub fn engine_config(&self) -> &WeftConfig {
        &self.config
    }
}

// ── tonic trait implementation ───────────────────────────────────────────────

#[tonic::async_trait]
impl proto::weft_server::Weft for WeftService {
    /// Unary chat RPC: single request → single response.
    ///
    /// Extracts W3C TraceContext from gRPC metadata. When a valid `traceparent`
    /// key is present, the current tracing span (from tower-http's TraceLayer) is
    /// re-parented to the incoming trace, enabling distributed tracing across services.
    async fn chat(
        &self,
        request: Request<proto::ChatRequest>,
    ) -> Result<Response<proto::ChatResponse>, Status> {
        // Extract the parent OTel context from incoming gRPC metadata and set it on
        // the current tracing span. `set_parent` does not use thread-local storage and
        // is safe across `await` points in Send futures.
        //
        // tower-http's TraceLayer has already created the current tracing span — by
        // setting its OTel parent here, child spans (including the reactor's `request`
        // span) will be correctly linked into the incoming distributed trace.
        let parent_ctx = crate::telemetry::propagation::extract_from_metadata(request.metadata());
        {
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            tracing::Span::current().set_parent(parent_ctx);
        }

        let proto_req = request.into_inner();

        // Convert proto → domain. weft_request_from_proto validates message structure.
        let weft_req = weft_request_from_proto(proto_req)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        // Execute through the shared handler (validates + runs reactor).
        let weft_resp = self
            .handle_weft_request(weft_req)
            .await
            .map_err(|e| weft_error_to_status(&e))?;

        // Convert domain → proto.
        let proto_resp = proto_response_from_weft(weft_resp);
        Ok(Response::new(proto_resp))
    }

    /// Server-streaming chat RPC.
    ///
    /// V1 implementation: buffer-then-forward. The reactor processes the request fully
    /// (no true token streaming), then emits events in order:
    ///   1. Activity messages (`source: gateway`, `delta: false`) — if `options.activity = true`
    ///   2. Assistant text message (`delta: true`, `message_index` per `(model, source)` stream)
    ///   3. `RequestMetadata` — always last
    ///
    /// On reactor error, emits a `ChatError` event and the stream terminates.
    type ChatStreamStream = ReceiverStream<Result<proto::ChatEvent, Status>>;

    async fn chat_stream(
        &self,
        request: Request<proto::ChatRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        // Extract the parent OTel context from incoming gRPC metadata and set it on
        // the current tracing span. Safe across await points (no thread-local guard).
        let parent_ctx = crate::telemetry::propagation::extract_from_metadata(request.metadata());
        {
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            tracing::Span::current().set_parent(parent_ctx);
        }

        let proto_req = request.into_inner();

        // Convert proto → domain. Validate before spawning the task so the caller gets
        // a gRPC error immediately on bad input rather than an error event in the stream.
        let weft_req = weft_request_from_proto(proto_req)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        validate_request(&weft_req).map_err(|e| weft_error_to_status(&e))?;

        // Bounded channel (32) prevents the spawned task from outrunning the consumer.
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let reactor = Arc::clone(&self.reactor);

        tokio::spawn(async move {
            // Construct tenant/request IDs inside the spawned task.
            // Same defaults as handle_weft_request().
            let tenant_id = TenantId("default".to_string());
            let request_id = RequestId(uuid::Uuid::new_v4().to_string());

            // Execute through the Reactor. Map ReactorError to WeftError
            // so the existing error_code() and ChatError logic works unchanged.
            let result = reactor
                .execute(
                    ExecutionContext {
                        request: weft_req,
                        tenant_id,
                        request_id,
                        parent_id: None,
                        parent_budget: None,
                        client_tx: None,
                    },
                    None,
                )
                .await;

            match result {
                Ok((execution_result, _signal_tx)) => {
                    let resp = execution_result.response;
                    // Track message_index per (model, source) stream key.
                    // Each unique (model, source) pair forms an independent logical stream.
                    let mut stream_counters: std::collections::HashMap<
                        (String, proto::Source),
                        u32,
                    > = std::collections::HashMap::new();

                    // 1. Emit activity messages (complete, delta=false).
                    //    These are source=Gateway messages; the reactor only includes them in
                    //    `resp.messages` when `options.activity = true` was set.
                    for msg in &resp.messages {
                        if msg.source == Source::Gateway {
                            // Activity messages are complete (delta=false) and do not
                            // participate in the delta stream counter.
                            let proto_msg = proto_message_from_weft(msg.clone());
                            let event = proto::ChatEvent {
                                event: Some(proto::chat_event::Event::Message(proto_msg)),
                            };
                            if tx.send(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }

                    // 2. Emit non-gateway messages as delta events.
                    //    For V1, the full text is sent as a single delta chunk (delta=true).
                    //    message_index counts from 0 within each (model, source) pair.
                    for msg in &resp.messages {
                        if msg.source != Source::Gateway {
                            let source_proto = match msg.source {
                                Source::Client => proto::Source::Client,
                                Source::Gateway => proto::Source::Gateway,
                                Source::Provider => proto::Source::Provider,
                                Source::Member => proto::Source::Member,
                                Source::Judge => proto::Source::Judge,
                                Source::Tool => proto::Source::Tool,
                                Source::Memory => proto::Source::Memory,
                            };
                            let stream_key = (msg.model.clone().unwrap_or_default(), source_proto);
                            let idx = stream_counters.entry(stream_key).or_insert(0);
                            let current_idx = *idx;
                            *idx += 1;

                            let delta_msg = weft_core::WeftMessage {
                                role: msg.role,
                                source: msg.source,
                                model: msg.model.clone(),
                                content: msg.content.clone(),
                                delta: true,
                                message_index: current_idx,
                            };
                            let event = proto::ChatEvent {
                                event: Some(proto::chat_event::Event::Message(
                                    proto_message_from_weft(delta_msg),
                                )),
                            };
                            if tx.send(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }

                    // 3. Emit metadata — always last.
                    let metadata_event = proto::ChatEvent {
                        event: Some(proto::chat_event::Event::Metadata(proto::RequestMetadata {
                            id: resp.id,
                            model: resp.model,
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
                        })),
                    };
                    // Ignore send error: the consumer may have dropped.
                    let _ = tx.send(Ok(metadata_event)).await;
                }
                Err(reactor_err) => {
                    // On reactor error, map to WeftError and emit a ChatError event.
                    let weft_err = reactor_error_to_weft_error(reactor_err);
                    let error_event = proto::ChatEvent {
                        event: Some(proto::chat_event::Event::Error(proto::ChatError {
                            code: error_code(&weft_err),
                            message: weft_err.to_string(),
                        })),
                    };
                    let _ = tx.send(Ok(error_event)).await;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
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

/// Map a `ReactorError` to a `WeftError` at the WeftService boundary.
///
/// This is the authoritative mapping. Lives in the binary crate where both
/// `ReactorError` (from `weft_reactor`) and `WeftError` (from `weft_core`)
/// are available. Avoids adding `weft_reactor` as a dep of `weft_core`.
pub fn reactor_error_to_weft_error(err: ReactorError) -> WeftError {
    match err {
        ReactorError::ActivityFailed(activity_err) => map_activity_error(activity_err),
        ReactorError::BudgetExhausted(ref reason) => match reason.as_str() {
            "iterations" => WeftError::CommandLoopExceeded { max: 0 },
            "deadline" => WeftError::RequestTimeout { timeout_secs: 0 },
            "generation_calls" => WeftError::CommandLoopExceeded { max: 0 },
            // "depth" and any other exhaustion reason
            _ => WeftError::Llm(format!("budget exhausted: {reason}")),
        },
        ReactorError::HookBlocked { hook_name, reason } => WeftError::HookBlocked {
            event: "reactor".to_string(),
            reason,
            hook_name,
        },
        ReactorError::Cancelled { .. } => WeftError::RequestTimeout { timeout_secs: 0 },
        ReactorError::PipelineNotFound(name) => {
            WeftError::Config(format!("pipeline not found: {name}"))
        }
        ReactorError::Config(msg) => WeftError::Config(msg),
        ReactorError::ActivityNotFound(name) => {
            WeftError::Config(format!("activity not found: {name}"))
        }
        ReactorError::EventLog(err) => WeftError::Llm(format!("event log error: {err}")),
        ReactorError::Registry(err) => WeftError::Config(format!("registry error: {err}")),
        ReactorError::Serialization(msg) => WeftError::Llm(format!("serialization error: {msg}")),
        ReactorError::ChannelClosed => WeftError::Llm("internal channel error".to_string()),
    }
}

/// Map an `ActivityError` to a `WeftError`.
///
/// Handles all 5 `ActivityError` variants explicitly. String-based pattern matching
/// is applied only to `ActivityError::Failed` to determine the failure domain.
fn map_activity_error(err: ActivityError) -> WeftError {
    match err {
        ActivityError::Failed { reason, .. } => map_failed_reason(&reason),
        ActivityError::Cancelled { .. } => WeftError::RequestTimeout { timeout_secs: 0 },
        ActivityError::Timeout { .. } => WeftError::RequestTimeout { timeout_secs: 0 },
        ActivityError::InvalidInput { reason, .. } => WeftError::InvalidRequest(reason),
        ActivityError::EventLog { name, source } => {
            WeftError::Llm(format!("event log error in activity '{name}': {source}"))
        }
    }
}

/// Map an `ActivityError::Failed` reason string to a `WeftError` variant.
///
/// Uses substring matching on the reason string to identify the failure domain.
/// Order matters: more specific patterns are checked before generic ones.
fn map_failed_reason(reason: &str) -> WeftError {
    if reason.contains("rate limited") {
        // Parse retry_after_ms from reason string if present; fall back to 1000ms.
        let retry_after_ms = extract_retry_after_ms(reason).unwrap_or(1000);
        WeftError::RateLimited { retry_after_ms }
    } else if reason.contains("no eligible models") {
        let capability = extract_capability_from_reason(reason);
        WeftError::NoEligibleModels { capability }
    } else if reason.contains("model not found") || reason.contains("not found in provider") {
        let name = extract_model_name_from_reason(reason);
        WeftError::ModelNotFound { name }
    } else if reason.contains("hook blocked after") || reason.contains("retries exhausted") {
        let hook_name = extract_hook_name_from_reason(reason);
        WeftError::HookBlockedAfterRetries {
            event: "reactor".to_string(),
            reason: reason.to_string(),
            hook_name,
            retries: 0,
        }
    } else if reason.contains("invalid request") {
        WeftError::InvalidRequest(reason.to_string())
    } else if reason.contains("provider") {
        WeftError::Llm(reason.to_string())
    } else if reason.contains("command") {
        WeftError::Command(reason.to_string())
    } else if reason.contains("tool registry") {
        WeftError::ToolRegistry(reason.to_string())
    } else if reason.contains("memory") {
        WeftError::MemoryStore(reason.to_string())
    } else if reason.contains("routing") {
        WeftError::Routing(reason.to_string())
    } else {
        // Catch-all: treat as an LLM/provider error.
        WeftError::Llm(reason.to_string())
    }
}

/// Extract retry-after milliseconds from a reason string.
///
/// Handles two formats:
/// - `"retry_after_ms: NNN"` — explicit millisecond value
/// - `"retry after NNNms"` — provider error display format
///
/// Returns `None` if no parseable value is found.
fn extract_retry_after_ms(reason: &str) -> Option<u64> {
    // Try "retry_after_ms: NNN" pattern first.
    if let Some(pos) = reason.find("retry_after_ms:") {
        let after = &reason[pos + "retry_after_ms:".len()..];
        let trimmed = after.trim_start();
        let end = trimmed
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(trimmed.len());
        return trimmed[..end].parse().ok();
    }
    // Try "retry after NNNms" pattern (ProviderError::RateLimited display format).
    if let Some(pos) = reason.find("retry after ") {
        let after = &reason[pos + "retry after ".len()..];
        let end = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());
        return after[..end].parse().ok();
    }
    None
}

/// Extract a capability string from a reason containing "no eligible models".
///
/// Looks for "capability: X" or "for capability X" in the reason string.
/// Returns "unknown" if parsing fails.
fn extract_capability_from_reason(reason: &str) -> String {
    // Look for patterns like "no eligible models for capability: vision"
    // or "no eligible models for 'vision'"
    for prefix in &["capability: ", "capability '", "for capability "] {
        if let Some(pos) = reason.find(prefix) {
            let after = &reason[pos + prefix.len()..];
            let end = after
                .find(['\'', '"', ',', '.', ' '])
                .unwrap_or(after.len());
            let cap = after[..end].trim_matches(['\'', '"']);
            if !cap.is_empty() {
                return cap.to_string();
            }
        }
    }
    "unknown".to_string()
}

/// Extract a model name from a reason containing "model not found" or "not found in provider".
///
/// Returns the full reason string if no model name can be parsed.
fn extract_model_name_from_reason(reason: &str) -> String {
    // Look for patterns like "model not found: 'claude-3'" or "model 'x' not found in provider"
    for prefix in &["model not found: ", "model '", "model \""] {
        if let Some(pos) = reason.find(prefix) {
            let after = &reason[pos + prefix.len()..];
            let end = after.find(['\'', '"', ',', '.']).unwrap_or(after.len());
            let name = after[..end].trim_matches(['\'', '"']);
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    reason.to_string()
}

/// Extract a hook name from a reason containing "hook blocked after" or "retries exhausted".
///
/// Returns an empty string if no hook name can be parsed.
fn extract_hook_name_from_reason(reason: &str) -> String {
    // Look for patterns like "hook 'my-hook' blocked after N retries"
    for prefix in &["hook '", "hook \"", "hook_name: "] {
        if let Some(pos) = reason.find(prefix) {
            let after = &reason[pos + prefix.len()..];
            let end = after.find(['\'', '"', ' ']).unwrap_or(after.len());
            let name = after[..end].trim_matches(['\'', '"']);
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    String::new()
}

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

/// Map a `WeftError` to a machine-readable error code string.
///
/// Used in `ChatError` events emitted by the streaming RPC to give clients
/// a stable code they can switch on without parsing the human-readable message.
pub fn error_code(e: &WeftError) -> String {
    match e {
        WeftError::InvalidRequest(_) => "invalid_request".to_string(),
        WeftError::StreamingNotSupported => "invalid_request".to_string(),
        WeftError::Config(_) => "configuration_error".to_string(),
        WeftError::Llm(_) => "provider_error".to_string(),
        WeftError::Command(_) => "command_error".to_string(),
        WeftError::ToolRegistry(_) => "service_unavailable".to_string(),
        WeftError::MemoryStore(_) => "service_unavailable".to_string(),
        WeftError::CommandLoopExceeded { .. } => "command_loop_exceeded".to_string(),
        WeftError::RequestTimeout { .. } => "timeout".to_string(),
        WeftError::RateLimited { .. } => "rate_limited".to_string(),
        WeftError::Routing(_) => "routing_error".to_string(),
        WeftError::ModelNotFound { .. } => "model_not_found".to_string(),
        WeftError::NoEligibleModels { .. } => "routing_error".to_string(),
        WeftError::HookBlocked { .. } => "hook_blocked".to_string(),
        WeftError::HookBlockedAfterRetries { .. } => "hook_blocked".to_string(),
        WeftError::ProtoConversion(_) => "internal_error".to_string(),
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
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use weft_core::{ContentPart, Role, Source};
    use weft_reactor::{ActivityError, EventLogError, ReactorError};

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

    // ── ReactorError → WeftError mapping (parametrized) ────────────────────────
    //
    // Each case maps one ReactorError variant to the expected WeftError variant.
    // Add one `#[case]` line when a new ReactorError variant is introduced.

    #[rstest]
    #[case::budget_iterations(
        ReactorError::BudgetExhausted("iterations".to_string()),
        "command_loop_exceeded"
    )]
    #[case::budget_deadline(
        ReactorError::BudgetExhausted("deadline".to_string()),
        "request_timeout"
    )]
    #[case::budget_generation_calls(
        ReactorError::BudgetExhausted("generation_calls".to_string()),
        "command_loop_exceeded"
    )]
    #[case::budget_depth(
        ReactorError::BudgetExhausted("depth".to_string()),
        "llm"
    )]
    #[case::cancelled(
        ReactorError::Cancelled { reason: "signal".to_string() },
        "request_timeout"
    )]
    #[case::pipeline_not_found(
        ReactorError::PipelineNotFound("missing-pipeline".to_string()),
        "config"
    )]
    #[case::activity_not_found(
        ReactorError::ActivityNotFound("missing-activity".to_string()),
        "config"
    )]
    #[case::channel_closed(ReactorError::ChannelClosed, "llm")]
    #[case::activity_failed_provider(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "generate".to_string(),
            reason: "provider returned an error".to_string(),
        }),
        "llm"
    )]
    #[case::activity_failed_routing(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "route".to_string(),
            reason: "routing failed: no models match".to_string(),
        }),
        "routing"
    )]
    #[case::activity_failed_command(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "execute_command".to_string(),
            reason: "command execution failed".to_string(),
        }),
        "command"
    )]
    #[case::activity_failed_tool_registry(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "execute_command".to_string(),
            reason: "tool registry unavailable".to_string(),
        }),
        "tool_registry"
    )]
    #[case::activity_failed_memory(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "recall".to_string(),
            reason: "memory store connection failed".to_string(),
        }),
        "memory_store"
    )]
    #[case::activity_failed_invalid_request(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "validate".to_string(),
            reason: "invalid request: messages array empty".to_string(),
        }),
        "invalid_request"
    )]
    #[case::activity_failed_catch_all(
        ReactorError::ActivityFailed(ActivityError::Failed {
            name: "unknown".to_string(),
            reason: "some completely unknown error".to_string(),
        }),
        "llm"
    )]
    #[case::activity_cancelled(
        ReactorError::ActivityFailed(ActivityError::Cancelled {
            name: "generate".to_string(),
        }),
        "request_timeout"
    )]
    #[case::activity_timeout(
        ReactorError::ActivityFailed(ActivityError::Timeout {
            name: "generate".to_string(),
        }),
        "request_timeout"
    )]
    fn reactor_error_maps_to_weft_error_variant(
        #[case] err: ReactorError,
        #[case] expected_variant: &str,
    ) {
        let weft_err = reactor_error_to_weft_error(err);
        let actual_variant = match &weft_err {
            WeftError::CommandLoopExceeded { .. } => "command_loop_exceeded",
            WeftError::RequestTimeout { .. } => "request_timeout",
            WeftError::Llm(_) => "llm",
            WeftError::Config(_) => "config",
            WeftError::Routing(_) => "routing",
            WeftError::Command(_) => "command",
            WeftError::ToolRegistry(_) => "tool_registry",
            WeftError::MemoryStore(_) => "memory_store",
            WeftError::InvalidRequest(_) => "invalid_request",
            WeftError::RateLimited { .. } => "rate_limited",
            WeftError::HookBlocked { .. } => "hook_blocked",
            WeftError::HookBlockedAfterRetries { .. } => "hook_blocked_after_retries",
            WeftError::ModelNotFound { .. } => "model_not_found",
            WeftError::NoEligibleModels { .. } => "no_eligible_models",
            WeftError::StreamingNotSupported => "streaming_not_supported",
            WeftError::ProtoConversion(_) => "proto_conversion",
        };
        assert_eq!(actual_variant, expected_variant);
    }

    // These ReactorError cases have additional field assertions beyond variant matching.

    #[test]
    fn test_hook_blocked_preserves_fields() {
        let err = ReactorError::HookBlocked {
            hook_name: "auth-hook".to_string(),
            reason: "policy violation".to_string(),
        };
        let weft_err = reactor_error_to_weft_error(err);
        assert!(matches!(weft_err, WeftError::HookBlocked { .. }));
        if let WeftError::HookBlocked {
            event,
            hook_name,
            reason,
        } = weft_err
        {
            assert_eq!(event, "reactor");
            assert_eq!(hook_name, "auth-hook");
            assert_eq!(reason, "policy violation");
        }
    }

    #[test]
    fn test_pipeline_not_found_message_contains_name() {
        let err = ReactorError::PipelineNotFound("missing-pipeline".to_string());
        let weft_err = reactor_error_to_weft_error(err);
        if let WeftError::Config(msg) = weft_err {
            assert!(msg.contains("pipeline not found"));
        } else {
            panic!("expected Config error");
        }
    }

    #[test]
    fn test_activity_failed_rate_limited_parses_retry_after_ms() {
        let err = ReactorError::ActivityFailed(ActivityError::Failed {
            name: "generate".to_string(),
            reason: "rate limited: retry_after_ms: 2000".to_string(),
        });
        let weft_err = reactor_error_to_weft_error(err);
        assert!(matches!(
            weft_err,
            WeftError::RateLimited {
                retry_after_ms: 2000
            }
        ));
    }

    #[test]
    fn test_activity_failed_rate_limited_defaults_to_1000ms() {
        let err = ReactorError::ActivityFailed(ActivityError::Failed {
            name: "generate".to_string(),
            reason: "rate limited by provider".to_string(),
        });
        let weft_err = reactor_error_to_weft_error(err);
        assert!(matches!(
            weft_err,
            WeftError::RateLimited {
                retry_after_ms: 1000
            }
        ));
    }

    #[test]
    fn test_activity_invalid_input_preserves_reason() {
        let err = ReactorError::ActivityFailed(ActivityError::InvalidInput {
            name: "validate".to_string(),
            reason: "empty messages".to_string(),
        });
        let weft_err = reactor_error_to_weft_error(err);
        assert!(matches!(weft_err, WeftError::InvalidRequest(ref r) if r == "empty messages"));
    }

    #[test]
    fn test_activity_event_log_includes_activity_name_in_message() {
        let err = ReactorError::ActivityFailed(ActivityError::EventLog {
            name: "generate".to_string(),
            source: EventLogError::Storage("write failed".to_string()),
        });
        let weft_err = reactor_error_to_weft_error(err);
        assert!(matches!(weft_err, WeftError::Llm(_)));
        if let WeftError::Llm(msg) = weft_err {
            assert!(msg.contains("event log error in activity 'generate'"));
        }
    }

    // ── WeftError → tonic::Status mapping (parametrized) ─────────────────────
    //
    // Each case maps one WeftError variant to the expected gRPC status code.
    // Add one `#[case]` line when a new WeftError variant is introduced.

    #[rstest]
    #[case::invalid_request(WeftError::InvalidRequest("bad".to_string()), tonic::Code::InvalidArgument)]
    #[case::streaming_not_supported(WeftError::StreamingNotSupported, tonic::Code::InvalidArgument)]
    #[case::config(WeftError::Config("bad".to_string()), tonic::Code::Internal)]
    #[case::llm(WeftError::Llm("fail".to_string()), tonic::Code::Internal)]
    #[case::command(WeftError::Command("fail".to_string()), tonic::Code::Internal)]
    #[case::tool_registry(WeftError::ToolRegistry("fail".to_string()), tonic::Code::Unavailable)]
    #[case::memory_store(WeftError::MemoryStore("fail".to_string()), tonic::Code::Unavailable)]
    #[case::command_loop_exceeded(
        WeftError::CommandLoopExceeded { max: 10 },
        tonic::Code::ResourceExhausted
    )]
    #[case::request_timeout(
        WeftError::RequestTimeout { timeout_secs: 30 },
        tonic::Code::DeadlineExceeded
    )]
    #[case::rate_limited(
        WeftError::RateLimited { retry_after_ms: 5000 },
        tonic::Code::ResourceExhausted
    )]
    #[case::routing(WeftError::Routing("fail".to_string()), tonic::Code::FailedPrecondition)]
    #[case::model_not_found(
        WeftError::ModelNotFound { name: "x".to_string() },
        tonic::Code::NotFound
    )]
    #[case::no_eligible_models(
        WeftError::NoEligibleModels { capability: "vision".to_string() },
        tonic::Code::FailedPrecondition
    )]
    #[case::hook_blocked(
        WeftError::HookBlocked {
            event: "e".to_string(),
            reason: "r".to_string(),
            hook_name: "h".to_string(),
        },
        tonic::Code::PermissionDenied
    )]
    #[case::hook_blocked_after_retries(
        WeftError::HookBlockedAfterRetries {
            event: "e".to_string(),
            reason: "r".to_string(),
            hook_name: "h".to_string(),
            retries: 3,
        },
        tonic::Code::Aborted
    )]
    #[case::proto_conversion(
        WeftError::ProtoConversion("fail".to_string()),
        tonic::Code::Internal
    )]
    fn weft_error_maps_to_grpc_status(
        #[case] error: WeftError,
        #[case] expected_code: tonic::Code,
    ) {
        let status = weft_error_to_status(&error);
        assert_eq!(status.code(), expected_code);
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
            other => panic!("expected Routing, got {other:?}"),
        }
    }
}
