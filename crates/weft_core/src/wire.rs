//! Weft Wire domain types: request, response, sampling options, and source validation.
//!
//! Also contains proto conversion traits (`TryFrom`/`From`) since `weft_core`
//! depends on `weft_proto` for the generated wire types.

use crate::error::WeftError;
use crate::message::{
    CommandCallContent, CommandResultContent, ContentPart, CouncilStartActivity, DocumentContent,
    HookActivity, MediaContent, MediaSource, MemoryResultContent, MemoryResultEntry,
    MemoryStoredContent, Role, RoutingActivity, Source, WeftMessage,
};
use crate::routing::ModelRoutingInstruction;

// ── Domain types ──────────────────────────────────────────────────────────────

/// A validated chat request, ready for the engine.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WeftRequest {
    /// Conversation messages.
    pub messages: Vec<WeftMessage>,
    /// Parsed model routing instruction.
    pub routing: ModelRoutingInstruction,
    /// Sampling options.
    pub options: SamplingOptions,
}

/// Sampling and behavior options.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SamplingOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub response_format: Option<ResponseFormat>,
    /// Include activity messages in the response.
    pub activity: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

/// The engine's response, before wire-format conversion.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WeftResponse {
    pub id: String,
    /// Echoed routing instruction string.
    pub model: String,
    /// Response messages (may include activity if requested).
    pub messages: Vec<WeftMessage>,
    pub usage: WeftUsage,
    pub timing: WeftTiming,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct WeftUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub llm_calls: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct WeftTiming {
    pub total_ms: u64,
    pub routing_ms: u64,
    pub llm_ms: u64,
}

// ── Source Validation ─────────────────────────────────────────────────────────

/// Validate that a message's source is consistent with its content parts.
///
/// Returns Ok(()) if all content parts are valid for the declared source.
/// Returns Err with a human-readable error if any content part violates source rules.
pub fn validate_message_source(msg: &WeftMessage) -> Result<(), SourceValidationError> {
    for (i, part) in msg.content.iter().enumerate() {
        if !is_valid_source_for_content(msg.source, part) {
            return Err(SourceValidationError {
                message_source: msg.source,
                content_index: i,
                content_type: part.type_name(),
                allowed_sources: allowed_sources_for_content(part),
            });
        }
    }
    Ok(())
}

/// Check if a source is valid for a given content part.
fn is_valid_source_for_content(source: Source, part: &ContentPart) -> bool {
    match part {
        // General content: any source
        ContentPart::Text(_)
        | ContentPart::Image(_)
        | ContentPart::Audio(_)
        | ContentPart::Video(_)
        | ContentPart::Document(_) => true,

        // Command calls: only LLM sources
        ContentPart::CommandCall(_) => {
            matches!(source, Source::Provider | Source::Member | Source::Judge)
        }

        // Activity: only gateway
        ContentPart::Routing(_) | ContentPart::Hook(_) | ContentPart::CouncilStart(_) => {
            source == Source::Gateway
        }

        // Tool results: only tool
        ContentPart::CommandResult(_) => source == Source::Tool,

        // Memory results: only memory
        ContentPart::MemoryResult(_) | ContentPart::MemoryStored(_) => source == Source::Memory,
    }
}

/// Returns the allowed sources for a given content part type, for error messages.
fn allowed_sources_for_content(part: &ContentPart) -> &'static [Source] {
    match part {
        ContentPart::Text(_)
        | ContentPart::Image(_)
        | ContentPart::Audio(_)
        | ContentPart::Video(_)
        | ContentPart::Document(_) => &[
            Source::Client,
            Source::Gateway,
            Source::Provider,
            Source::Member,
            Source::Judge,
            Source::Tool,
            Source::Memory,
        ],
        ContentPart::CommandCall(_) => &[Source::Provider, Source::Member, Source::Judge],
        ContentPart::Routing(_) | ContentPart::Hook(_) | ContentPart::CouncilStart(_) => {
            &[Source::Gateway]
        }
        ContentPart::CommandResult(_) => &[Source::Tool],
        ContentPart::MemoryResult(_) | ContentPart::MemoryStored(_) => &[Source::Memory],
    }
}

/// Source validation error carrying context for human-readable messages.
#[derive(Debug)]
pub struct SourceValidationError {
    pub message_source: Source,
    pub content_index: usize,
    pub content_type: &'static str,
    pub allowed_sources: &'static [Source],
}

/// Full request validation, called before the engine processes a request.
pub fn validate_request(req: &WeftRequest) -> Result<(), WeftError> {
    // 1. Must have at least one message
    if req.messages.is_empty() {
        return Err(WeftError::InvalidRequest(
            "messages must not be empty".into(),
        ));
    }

    // 2. Must have at least one user message
    if !req.messages.iter().any(|m| m.role == Role::User) {
        return Err(WeftError::InvalidRequest(
            "messages must contain at least one user message".into(),
        ));
    }

    // 3. Must have at least one content part per message
    for (i, msg) in req.messages.iter().enumerate() {
        if msg.content.is_empty() {
            return Err(WeftError::InvalidRequest(format!(
                "message[{}]: content must not be empty",
                i
            )));
        }
    }

    // 4. Validate source consistency for each message
    for (i, msg) in req.messages.iter().enumerate() {
        validate_message_source(msg).map_err(|e| {
            WeftError::InvalidRequest(format!(
                "message[{}]: source {:?} cannot contain {} content (allowed: {:?})",
                i, e.message_source, e.content_type, e.allowed_sources
            ))
        })?;
    }

    Ok(())
}

// ── Proto Conversion Traits ───────────────────────────────────────────────────

/// Error type for proto-to-domain conversion failures.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("message role is required")]
    MissingRole,
    #[error("content part is empty (no oneof variant set)")]
    EmptyContentPart,
    #[error("invalid content part at index {index}: {reason}")]
    InvalidContentPart { index: usize, reason: String },
}

use weft_proto::weft::v1 as proto;

impl TryFrom<proto::WeftMessage> for WeftMessage {
    type Error = ConversionError;

    fn try_from(p: proto::WeftMessage) -> Result<Self, Self::Error> {
        let role = match p.role() {
            proto::Role::Unspecified => return Err(ConversionError::MissingRole),
            proto::Role::User => Role::User,
            proto::Role::Assistant => Role::Assistant,
            proto::Role::System => Role::System,
        };

        let source = match p.source() {
            proto::Source::Unspecified => Source::Client, // default per spec
            proto::Source::Client => Source::Client,
            proto::Source::Gateway => Source::Gateway,
            proto::Source::Provider => Source::Provider,
            proto::Source::Member => Source::Member,
            proto::Source::Judge => Source::Judge,
            proto::Source::Tool => Source::Tool,
            proto::Source::Memory => Source::Memory,
        };

        let model = if p.model.is_empty() {
            None
        } else {
            Some(p.model)
        };

        let content: Result<Vec<ContentPart>, ConversionError> = p
            .content
            .into_iter()
            .enumerate()
            .map(|(idx, cp)| content_part_from_proto(cp, idx))
            .collect();

        Ok(WeftMessage {
            role,
            source,
            model,
            content: content?,
            delta: p.delta,
            message_index: p.message_index,
        })
    }
}

fn content_part_from_proto(
    cp: proto::ContentPart,
    idx: usize,
) -> Result<ContentPart, ConversionError> {
    use proto::content_part::Part;

    match cp.part {
        None => Err(ConversionError::EmptyContentPart),

        Some(Part::Text(t)) => Ok(ContentPart::Text(t.text)),

        Some(Part::Image(img)) => {
            let source = media_source_from_image(img.source, idx)?;
            Ok(ContentPart::Image(MediaContent {
                source,
                media_type: if img.media_type.is_empty() {
                    None
                } else {
                    Some(img.media_type)
                },
            }))
        }

        Some(Part::Audio(aud)) => {
            let source = media_source_from_audio(aud.source, idx)?;
            Ok(ContentPart::Audio(MediaContent {
                source,
                media_type: if aud.media_type.is_empty() {
                    None
                } else {
                    Some(aud.media_type)
                },
            }))
        }

        Some(Part::Video(vid)) => {
            let source = media_source_from_video(vid.source, idx)?;
            Ok(ContentPart::Video(MediaContent {
                source,
                media_type: if vid.media_type.is_empty() {
                    None
                } else {
                    Some(vid.media_type)
                },
            }))
        }

        Some(Part::Document(doc)) => {
            let source = media_source_from_document(doc.source, idx)?;
            Ok(ContentPart::Document(DocumentContent {
                source,
                media_type: if doc.media_type.is_empty() {
                    None
                } else {
                    Some(doc.media_type)
                },
                filename: if doc.filename.is_empty() {
                    None
                } else {
                    Some(doc.filename)
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

fn media_source_from_image(
    src: Option<proto::image_content::Source>,
    idx: usize,
) -> Result<MediaSource, ConversionError> {
    match src {
        None => Err(ConversionError::InvalidContentPart {
            index: idx,
            reason: "image source is required".to_string(),
        }),
        Some(proto::image_content::Source::Url(u)) => Ok(MediaSource::Url(u)),
        Some(proto::image_content::Source::Data(d)) => Ok(MediaSource::Data(d)),
    }
}

fn media_source_from_audio(
    src: Option<proto::audio_content::Source>,
    idx: usize,
) -> Result<MediaSource, ConversionError> {
    match src {
        None => Err(ConversionError::InvalidContentPart {
            index: idx,
            reason: "audio source is required".to_string(),
        }),
        Some(proto::audio_content::Source::Url(u)) => Ok(MediaSource::Url(u)),
        Some(proto::audio_content::Source::Data(d)) => Ok(MediaSource::Data(d)),
    }
}

fn media_source_from_video(
    src: Option<proto::video_content::Source>,
    idx: usize,
) -> Result<MediaSource, ConversionError> {
    match src {
        None => Err(ConversionError::InvalidContentPart {
            index: idx,
            reason: "video source is required".to_string(),
        }),
        Some(proto::video_content::Source::Url(u)) => Ok(MediaSource::Url(u)),
        Some(proto::video_content::Source::Data(d)) => Ok(MediaSource::Data(d)),
    }
}

fn media_source_from_document(
    src: Option<proto::document_content::Source>,
    idx: usize,
) -> Result<MediaSource, ConversionError> {
    match src {
        None => Err(ConversionError::InvalidContentPart {
            index: idx,
            reason: "document source is required".to_string(),
        }),
        Some(proto::document_content::Source::Url(u)) => Ok(MediaSource::Url(u)),
        Some(proto::document_content::Source::Data(d)) => Ok(MediaSource::Data(d)),
    }
}

// ── Domain to Proto ───────────────────────────────────────────────────────────

impl From<WeftMessage> for proto::WeftMessage {
    fn from(msg: WeftMessage) -> Self {
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
            content: msg.content.into_iter().map(content_part_to_proto).collect(),
            delta: msg.delta,
            message_index: msg.message_index,
        }
    }
}

fn content_part_to_proto(part: ContentPart) -> proto::ContentPart {
    use proto::content_part::Part;

    let p = match part {
        ContentPart::Text(t) => Part::Text(proto::TextContent { text: t }),

        ContentPart::Image(img) => Part::Image(proto::ImageContent {
            source: Some(match img.source {
                MediaSource::Url(u) => proto::image_content::Source::Url(u),
                MediaSource::Data(d) => proto::image_content::Source::Data(d),
            }),
            media_type: img.media_type.unwrap_or_default(),
        }),

        ContentPart::Audio(aud) => Part::Audio(proto::AudioContent {
            source: Some(match aud.source {
                MediaSource::Url(u) => proto::audio_content::Source::Url(u),
                MediaSource::Data(d) => proto::audio_content::Source::Data(d),
            }),
            media_type: aud.media_type.unwrap_or_default(),
        }),

        ContentPart::Video(vid) => Part::Video(proto::VideoContent {
            source: Some(match vid.source {
                MediaSource::Url(u) => proto::video_content::Source::Url(u),
                MediaSource::Data(d) => proto::video_content::Source::Data(d),
            }),
            media_type: vid.media_type.unwrap_or_default(),
        }),

        ContentPart::Document(doc) => Part::Document(proto::DocumentContent {
            source: Some(match doc.source {
                MediaSource::Url(u) => proto::document_content::Source::Url(u),
                MediaSource::Data(d) => proto::document_content::Source::Data(d),
            }),
            media_type: doc.media_type.unwrap_or_default(),
            filename: doc.filename.unwrap_or_default(),
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

    proto::ContentPart { part: Some(p) }
}

// ── Proto ChatRequest -> WeftRequest ──────────────────────────────────────────

impl TryFrom<proto::ChatRequest> for WeftRequest {
    type Error = ConversionError;

    fn try_from(req: proto::ChatRequest) -> Result<Self, Self::Error> {
        let messages: Result<Vec<WeftMessage>, ConversionError> = req
            .messages
            .into_iter()
            .map(WeftMessage::try_from)
            .collect();

        let routing = ModelRoutingInstruction::parse(&req.model);

        let options = req.options.map(SamplingOptions::from).unwrap_or_default();

        Ok(WeftRequest {
            messages: messages?,
            routing,
            options,
        })
    }
}

impl From<proto::SamplingOptions> for SamplingOptions {
    fn from(opts: proto::SamplingOptions) -> Self {
        let response_format = opts.response_format.as_ref().map(|rf| {
            if rf.r#type == "json_object" {
                ResponseFormat::JsonObject
            } else {
                ResponseFormat::Text
            }
        });

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
            response_format,
            activity: opts.activity,
        }
    }
}

// ── WeftResponse -> Proto ChatResponse ───────────────────────────────────────

impl From<WeftResponse> for proto::ChatResponse {
    fn from(resp: WeftResponse) -> Self {
        proto::ChatResponse {
            id: resp.id,
            model: resp.model,
            messages: resp
                .messages
                .into_iter()
                .map(proto::WeftMessage::from)
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
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{
        CommandCallContent, CommandResultContent, CouncilStartActivity, DocumentContent,
        HookActivity, MediaContent, MediaSource, MemoryResultContent, MemoryResultEntry,
        MemoryStoredContent, RoutingActivity,
    };

    // ── Helpers ────────────────────────────────────────────────────────────

    fn user_text_msg(text: &str) -> WeftMessage {
        WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text(text.to_string())],
            delta: false,
            message_index: 0,
        }
    }

    fn assistant_text_msg(text: &str) -> WeftMessage {
        WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("gpt-4".to_string()),
            content: vec![ContentPart::Text(text.to_string())],
            delta: false,
            message_index: 0,
        }
    }

    // ── Source validation: general content (any source) ────────────────────

    #[test]
    fn test_source_valid_text_client() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hi".to_string())],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_valid_text_any_source() {
        for source in [
            Source::Client,
            Source::Gateway,
            Source::Provider,
            Source::Member,
            Source::Judge,
            Source::Tool,
            Source::Memory,
        ] {
            let msg = WeftMessage {
                role: Role::User,
                source,
                model: None,
                content: vec![ContentPart::Text("x".to_string())],
                delta: false,
                message_index: 0,
            };
            assert!(
                validate_message_source(&msg).is_ok(),
                "Text should be valid for source {:?}",
                source
            );
        }
    }

    #[test]
    fn test_source_valid_image_any_source() {
        let image = ContentPart::Image(MediaContent {
            source: MediaSource::Url("https://example.com/img.png".to_string()),
            media_type: Some("image/png".to_string()),
        });
        for source in [Source::Client, Source::Provider, Source::Gateway] {
            let msg = WeftMessage {
                role: Role::User,
                source,
                model: None,
                content: vec![image.clone()],
                delta: false,
                message_index: 0,
            };
            assert!(
                validate_message_source(&msg).is_ok(),
                "Image should be valid for source {:?}",
                source
            );
        }
    }

    // ── Source validation: CommandCall (LLM sources only) ─────────────────

    #[test]
    fn test_source_valid_command_call_provider() {
        let msg = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("gpt-4".to_string()),
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "search".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_valid_command_call_member() {
        let msg = WeftMessage {
            role: Role::Assistant,
            source: Source::Member,
            model: Some("claude".to_string()),
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "read_file".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_valid_command_call_judge() {
        let msg = WeftMessage {
            role: Role::Assistant,
            source: Source::Judge,
            model: Some("opus".to_string()),
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "analyze".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_invalid_command_call_client() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "fake_command".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    #[test]
    fn test_source_invalid_command_call_gateway() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "nope".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    #[test]
    fn test_source_invalid_command_call_tool() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Tool,
            model: None,
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "nope".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    // ── Source validation: Activity content (gateway only) ─────────────────

    #[test]
    fn test_source_valid_routing_gateway() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::Routing(RoutingActivity {
                model: "auto".to_string(),
                score: 0.9,
                filters: vec![],
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_invalid_routing_client() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Routing(RoutingActivity {
                model: "auto".to_string(),
                score: 0.9,
                filters: vec![],
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    #[test]
    fn test_source_invalid_hook_provider() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Provider,
            model: None,
            content: vec![ContentPart::Hook(HookActivity {
                event: "pre_request".to_string(),
                hook_name: "rate-limiter".to_string(),
                decision: "allow".to_string(),
                reason: None,
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    #[test]
    fn test_source_valid_council_start_gateway() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::CouncilStart(CouncilStartActivity {
                models: vec!["gpt-4".to_string(), "claude".to_string()],
                judge: "opus".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    // ── Source validation: CommandResult (tool only) ───────────────────────

    #[test]
    fn test_source_valid_command_result_tool() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Tool,
            model: None,
            content: vec![ContentPart::CommandResult(CommandResultContent {
                command: "search".to_string(),
                success: true,
                output: "results".to_string(),
                error: None,
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_invalid_command_result_client() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::CommandResult(CommandResultContent {
                command: "search".to_string(),
                success: true,
                output: "results".to_string(),
                error: None,
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    // ── Source validation: Memory content (memory only) ───────────────────

    #[test]
    fn test_source_valid_memory_result_memory() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Memory,
            model: None,
            content: vec![ContentPart::MemoryResult(MemoryResultContent {
                store: "default".to_string(),
                entries: vec![],
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_valid_memory_stored_memory() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Memory,
            model: None,
            content: vec![ContentPart::MemoryStored(MemoryStoredContent {
                store: "default".to_string(),
                id: "mem-1".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_ok());
    }

    #[test]
    fn test_source_invalid_memory_result_gateway() {
        let msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::MemoryResult(MemoryResultContent {
                store: "default".to_string(),
                entries: vec![],
            })],
            delta: false,
            message_index: 0,
        };
        assert!(validate_message_source(&msg).is_err());
    }

    // ── Source validation error structure ─────────────────────────────────

    #[test]
    fn test_source_validation_error_fields() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "x".to_string(),
                arguments_json: "{}".to_string(),
            })],
            delta: false,
            message_index: 0,
        };
        let err = validate_message_source(&msg).unwrap_err();
        assert_eq!(err.message_source, Source::Client);
        assert_eq!(err.content_index, 0);
        assert_eq!(err.content_type, "CommandCall");
        assert!(err.allowed_sources.contains(&Source::Provider));
    }

    // ── Request validation ────────────────────────────────────────────────

    #[test]
    fn test_validate_request_empty_messages() {
        let req = WeftRequest {
            messages: vec![],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        let err = validate_request(&req).unwrap_err();
        assert!(matches!(err, WeftError::InvalidRequest(_)));
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_validate_request_no_user_message() {
        let req = WeftRequest {
            messages: vec![WeftMessage {
                role: Role::System,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("system prompt".to_string())],
                delta: false,
                message_index: 0,
            }],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        let err = validate_request(&req).unwrap_err();
        assert!(matches!(err, WeftError::InvalidRequest(_)));
        assert!(err.to_string().contains("at least one user message"));
    }

    #[test]
    fn test_validate_request_empty_content_in_message() {
        let req = WeftRequest {
            messages: vec![WeftMessage {
                role: Role::User,
                source: Source::Client,
                model: None,
                content: vec![],
                delta: false,
                message_index: 0,
            }],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        let err = validate_request(&req).unwrap_err();
        assert!(matches!(err, WeftError::InvalidRequest(_)));
        assert!(err.to_string().contains("content must not be empty"));
    }

    #[test]
    fn test_validate_request_source_violation() {
        let req = WeftRequest {
            messages: vec![WeftMessage {
                role: Role::User,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::CommandCall(CommandCallContent {
                    command: "x".to_string(),
                    arguments_json: "{}".to_string(),
                })],
                delta: false,
                message_index: 0,
            }],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        let err = validate_request(&req).unwrap_err();
        assert!(matches!(err, WeftError::InvalidRequest(_)));
    }

    #[test]
    fn test_validate_request_valid() {
        let req = WeftRequest {
            messages: vec![user_text_msg("hello")],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn test_validate_request_multi_message_valid() {
        let req = WeftRequest {
            messages: vec![
                WeftMessage {
                    role: Role::System,
                    source: Source::Client,
                    model: None,
                    content: vec![ContentPart::Text("You are helpful.".to_string())],
                    delta: false,
                    message_index: 0,
                },
                user_text_msg("hello"),
                assistant_text_msg("hi there"),
                user_text_msg("thanks"),
            ],
            routing: ModelRoutingInstruction::parse("auto"),
            options: SamplingOptions::default(),
        };
        assert!(validate_request(&req).is_ok());
    }

    // ── Proto round-trip: Text ────────────────────────────────────────────

    #[test]
    fn test_proto_roundtrip_text() {
        let original = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hello world".to_string())],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.clone().into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        assert_eq!(roundtripped.role, original.role);
        assert_eq!(roundtripped.source, original.source);
        assert_eq!(roundtripped.model, original.model);
        assert_eq!(roundtripped.delta, original.delta);
        assert_eq!(roundtripped.message_index, original.message_index);
        assert_eq!(roundtripped.content.len(), 1);
        if let ContentPart::Text(t) = &roundtripped.content[0] {
            assert_eq!(t, "hello world");
        } else {
            panic!("Expected Text content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_delta_and_index() {
        let original = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("gpt-4".to_string()),
            content: vec![ContentPart::Text("chunk".to_string())],
            delta: true,
            message_index: 7,
        };

        let proto_msg: proto::WeftMessage = original.clone().into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        assert_eq!(roundtripped.delta, true);
        assert_eq!(roundtripped.message_index, 7);
        assert_eq!(roundtripped.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_proto_roundtrip_image_url() {
        let original = WeftMessage {
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

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::Image(img) = &roundtripped.content[0] {
            assert_eq!(img.media_type, Some("image/png".to_string()));
            if let MediaSource::Url(url) = &img.source {
                assert_eq!(url, "https://example.com/img.png");
            } else {
                panic!("Expected URL source");
            }
        } else {
            panic!("Expected Image content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_image_data() {
        let data = vec![0u8, 1, 2, 3, 255];
        let original = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Image(MediaContent {
                source: MediaSource::Data(data.clone()),
                media_type: Some("image/jpeg".to_string()),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::Image(img) = &roundtripped.content[0] {
            if let MediaSource::Data(d) = &img.source {
                assert_eq!(d, &data);
            } else {
                panic!("Expected Data source");
            }
        } else {
            panic!("Expected Image content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_audio() {
        let original = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Audio(MediaContent {
                source: MediaSource::Url("https://example.com/audio.wav".to_string()),
                media_type: Some("audio/wav".to_string()),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        assert!(matches!(roundtripped.content[0], ContentPart::Audio(_)));
    }

    #[test]
    fn test_proto_roundtrip_video() {
        let original = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Video(MediaContent {
                source: MediaSource::Url("https://example.com/video.mp4".to_string()),
                media_type: Some("video/mp4".to_string()),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        assert!(matches!(roundtripped.content[0], ContentPart::Video(_)));
    }

    #[test]
    fn test_proto_roundtrip_document() {
        let original = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Document(DocumentContent {
                source: MediaSource::Url("https://example.com/doc.pdf".to_string()),
                media_type: Some("application/pdf".to_string()),
                filename: Some("doc.pdf".to_string()),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::Document(doc) = &roundtripped.content[0] {
            assert_eq!(doc.filename, Some("doc.pdf".to_string()));
            assert_eq!(doc.media_type, Some("application/pdf".to_string()));
        } else {
            panic!("Expected Document content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_routing_activity() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::Routing(RoutingActivity {
                model: "anthropic/opus".to_string(),
                score: 0.95,
                filters: vec!["anthropic".to_string()],
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::Routing(r) = &roundtripped.content[0] {
            assert_eq!(r.model, "anthropic/opus");
            assert!((r.score - 0.95).abs() < 1e-5);
            assert_eq!(r.filters, vec!["anthropic".to_string()]);
        } else {
            panic!("Expected Routing content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_hook_activity() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::Hook(HookActivity {
                event: "pre_request".to_string(),
                hook_name: "rate-limiter".to_string(),
                decision: "allow".to_string(),
                reason: None,
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::Hook(h) = &roundtripped.content[0] {
            assert_eq!(h.event, "pre_request");
            assert_eq!(h.decision, "allow");
            // Empty reason maps to None
            assert_eq!(h.reason, None);
        } else {
            panic!("Expected Hook content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_hook_with_reason() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::Hook(HookActivity {
                event: "pre_request".to_string(),
                hook_name: "content-filter".to_string(),
                decision: "block".to_string(),
                reason: Some("profanity detected".to_string()),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::Hook(h) = &roundtripped.content[0] {
            assert_eq!(h.reason, Some("profanity detected".to_string()));
        } else {
            panic!("Expected Hook content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_council_start() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::CouncilStart(CouncilStartActivity {
                models: vec!["gpt-4".to_string(), "claude".to_string()],
                judge: "opus".to_string(),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::CouncilStart(c) = &roundtripped.content[0] {
            assert_eq!(c.models.len(), 2);
            assert_eq!(c.judge, "opus");
        } else {
            panic!("Expected CouncilStart content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_command_call() {
        let original = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some("gpt-4".to_string()),
            content: vec![ContentPart::CommandCall(CommandCallContent {
                command: "search".to_string(),
                arguments_json: r#"{"query":"rust"}"#.to_string(),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::CommandCall(cc) = &roundtripped.content[0] {
            assert_eq!(cc.command, "search");
            assert_eq!(cc.arguments_json, r#"{"query":"rust"}"#);
        } else {
            panic!("Expected CommandCall content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_command_result() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Tool,
            model: None,
            content: vec![ContentPart::CommandResult(CommandResultContent {
                command: "search".to_string(),
                success: true,
                output: "results here".to_string(),
                error: None,
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::CommandResult(cr) = &roundtripped.content[0] {
            assert_eq!(cr.command, "search");
            assert!(cr.success);
            assert_eq!(cr.output, "results here");
            assert_eq!(cr.error, None);
        } else {
            panic!("Expected CommandResult content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_command_result_with_error() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Tool,
            model: None,
            content: vec![ContentPart::CommandResult(CommandResultContent {
                command: "search".to_string(),
                success: false,
                output: String::new(),
                error: Some("timeout".to_string()),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::CommandResult(cr) = &roundtripped.content[0] {
            assert!(!cr.success);
            assert_eq!(cr.error, Some("timeout".to_string()));
        } else {
            panic!("Expected CommandResult content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_memory_result() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Memory,
            model: None,
            content: vec![ContentPart::MemoryResult(MemoryResultContent {
                store: "default".to_string(),
                entries: vec![MemoryResultEntry {
                    id: "mem-1".to_string(),
                    content: "some memory".to_string(),
                    score: 0.9,
                    created_at: "2026-03-16T00:00:00Z".to_string(),
                }],
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::MemoryResult(mr) = &roundtripped.content[0] {
            assert_eq!(mr.store, "default");
            assert_eq!(mr.entries.len(), 1);
            assert_eq!(mr.entries[0].id, "mem-1");
            assert!((mr.entries[0].score - 0.9_f32).abs() < 1e-5);
        } else {
            panic!("Expected MemoryResult content part");
        }
    }

    #[test]
    fn test_proto_roundtrip_memory_stored() {
        let original = WeftMessage {
            role: Role::System,
            source: Source::Memory,
            model: None,
            content: vec![ContentPart::MemoryStored(MemoryStoredContent {
                store: "default".to_string(),
                id: "mem-42".to_string(),
            })],
            delta: false,
            message_index: 0,
        };

        let proto_msg: proto::WeftMessage = original.into();
        let roundtripped = WeftMessage::try_from(proto_msg).unwrap();

        if let ContentPart::MemoryStored(ms) = &roundtripped.content[0] {
            assert_eq!(ms.store, "default");
            assert_eq!(ms.id, "mem-42");
        } else {
            panic!("Expected MemoryStored content part");
        }
    }

    // ── Proto conversion errors ───────────────────────────────────────────

    #[test]
    fn test_proto_conversion_missing_role() {
        let proto_msg = proto::WeftMessage {
            role: proto::Role::Unspecified as i32, // unspecified => error
            source: proto::Source::Client as i32,
            model: String::new(),
            content: vec![],
            delta: false,
            message_index: 0,
        };
        let result = WeftMessage::try_from(proto_msg);
        assert!(matches!(result, Err(ConversionError::MissingRole)));
    }

    #[test]
    fn test_proto_conversion_empty_content_part() {
        let proto_msg = proto::WeftMessage {
            role: proto::Role::User as i32,
            source: proto::Source::Client as i32,
            model: String::new(),
            content: vec![proto::ContentPart { part: None }],
            delta: false,
            message_index: 0,
        };
        let result = WeftMessage::try_from(proto_msg);
        assert!(matches!(result, Err(ConversionError::EmptyContentPart)));
    }

    #[test]
    fn test_proto_source_unspecified_defaults_to_client() {
        let proto_msg = proto::WeftMessage {
            role: proto::Role::User as i32,
            source: proto::Source::Unspecified as i32, // should default to Client
            model: String::new(),
            content: vec![proto::ContentPart {
                part: Some(proto::content_part::Part::Text(proto::TextContent {
                    text: "hi".to_string(),
                })),
            }],
            delta: false,
            message_index: 0,
        };
        let msg = WeftMessage::try_from(proto_msg).unwrap();
        assert_eq!(msg.source, Source::Client);
    }

    // ── Proto ChatRequest -> WeftRequest conversion ───────────────────────

    #[test]
    fn test_chat_request_conversion_basic() {
        let proto_req = proto::ChatRequest {
            messages: vec![proto::WeftMessage {
                role: proto::Role::User as i32,
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
            model: "auto".to_string(),
            options: None,
        };

        let weft_req = WeftRequest::try_from(proto_req).unwrap();
        assert_eq!(weft_req.messages.len(), 1);
        assert_eq!(weft_req.routing.raw, "auto");
        assert_eq!(weft_req.routing.mode, crate::routing::RoutingMode::Auto);
    }

    #[test]
    fn test_chat_request_conversion_with_options() {
        let proto_req = proto::ChatRequest {
            messages: vec![proto::WeftMessage {
                role: proto::Role::User as i32,
                source: proto::Source::Client as i32,
                model: String::new(),
                content: vec![proto::ContentPart {
                    part: Some(proto::content_part::Part::Text(proto::TextContent {
                        text: "hi".to_string(),
                    })),
                }],
                delta: false,
                message_index: 0,
            }],
            model: "anthropic/opus".to_string(),
            options: Some(proto::SamplingOptions {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                max_tokens: 1024,
                stop: vec!["STOP".to_string()],
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                seed: 42,
                response_format: Some(proto::ResponseFormat {
                    r#type: "json_object".to_string(),
                }),
                activity: true,
            }),
        };

        let weft_req = WeftRequest::try_from(proto_req).unwrap();
        assert_eq!(weft_req.options.temperature, Some(0.7));
        assert_eq!(weft_req.options.top_p, Some(0.9));
        assert_eq!(weft_req.options.top_k, Some(40u32));
        assert_eq!(weft_req.options.max_tokens, Some(1024u32));
        assert_eq!(weft_req.options.stop, vec!["STOP".to_string()]);
        assert_eq!(weft_req.options.seed, Some(42i64));
        assert!(weft_req.options.activity);
        assert!(matches!(
            weft_req.options.response_format,
            Some(ResponseFormat::JsonObject)
        ));
    }

    // ── WeftResponse -> Proto ChatResponse ───────────────────────────────

    #[test]
    fn test_weft_response_to_proto() {
        let resp = WeftResponse {
            id: "req-1".to_string(),
            model: "auto".to_string(),
            messages: vec![WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some("gpt-4".to_string()),
                content: vec![ContentPart::Text("hello".to_string())],
                delta: false,
                message_index: 0,
            }],
            usage: WeftUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                llm_calls: 1,
            },
            timing: WeftTiming {
                total_ms: 500,
                routing_ms: 10,
                llm_ms: 490,
            },
        };

        let proto_resp: proto::ChatResponse = resp.into();
        assert_eq!(proto_resp.id, "req-1");
        assert_eq!(proto_resp.model, "auto");
        assert_eq!(proto_resp.messages.len(), 1);
        let usage = proto_resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
        assert_eq!(usage.llm_calls, 1);
        let timing = proto_resp.timing.unwrap();
        assert_eq!(timing.total_ms, 500);
        assert_eq!(timing.routing_ms, 10);
        assert_eq!(timing.llm_ms, 490);
    }
}
