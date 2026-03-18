//! Domain message types for the Weft Wire protocol.
//!
//! `WeftMessage` is the validated, typed representation used throughout the engine.
//! Proto-generated types are wire types only — conversion happens at the gRPC boundary.

/// A fully attributed message in the Weft protocol.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WeftMessage {
    pub role: Role,
    pub source: Source,
    /// Which model produced this. None for user/system messages.
    pub model: Option<String>,
    /// Content parts. At least one required.
    pub content: Vec<ContentPart>,
    /// True when this is an incremental text delta (streaming only).
    /// False for complete messages (request history, tool results, activity).
    pub delta: bool,
    /// Delta sequence counter within each (model, source) pair.
    /// Each unique (model, source) combination is an independent logical stream.
    /// Deltas within a stream are numbered 0, 1, 2, ... in order.
    pub message_index: u32,
}

/// Where a message originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Client,
    Gateway,
    Provider,
    Member,
    Judge,
    Tool,
    Memory,
}

/// Who produced this message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// A typed content part within a message.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ContentPart {
    // --- General content ---
    Text(String),
    Image(MediaContent),
    Audio(MediaContent),
    Video(MediaContent),
    Document(DocumentContent),

    // --- Activity content (source: gateway only) ---
    Routing(RoutingActivity),
    Hook(HookActivity),
    CouncilStart(CouncilStartActivity),

    // --- Tool content ---
    CommandCall(CommandCallContent),
    CommandResult(CommandResultContent),
    MemoryResult(MemoryResultContent),
    MemoryStored(MemoryStoredContent),
}

impl ContentPart {
    /// Returns the capability required by this content part, if any.
    /// Returns None for content types that don't imply a specific capability
    /// (text, activity, tool content).
    pub fn required_capability(&self) -> Option<&'static str> {
        match self {
            Self::Image(_) => Some("vision"),
            Self::Audio(_) => Some("audio_transcriptions"),
            Self::Video(_) => Some("video"),
            Self::Document(_) => Some("document_understanding"),
            _ => None,
        }
    }

    /// Returns the type name as a static string, used in validation error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Text(_) => "Text",
            Self::Image(_) => "Image",
            Self::Audio(_) => "Audio",
            Self::Video(_) => "Video",
            Self::Document(_) => "Document",
            Self::Routing(_) => "Routing",
            Self::Hook(_) => "Hook",
            Self::CouncilStart(_) => "CouncilStart",
            Self::CommandCall(_) => "CommandCall",
            Self::CommandResult(_) => "CommandResult",
            Self::MemoryResult(_) => "MemoryResult",
            Self::MemoryStored(_) => "MemoryStored",
        }
    }
}

/// Media content that can be a URL or inline data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MediaContent {
    pub source: MediaSource,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum MediaSource {
    Url(String),
    Data(Vec<u8>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DocumentContent {
    pub source: MediaSource,
    pub media_type: Option<String>,
    pub filename: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutingActivity {
    pub model: String,
    pub score: f32,
    pub filters: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HookActivity {
    pub event: String,
    pub hook_name: String,
    pub decision: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CouncilStartActivity {
    pub models: Vec<String>,
    pub judge: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandCallContent {
    pub command: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandResultContent {
    pub command: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryResultContent {
    pub store: String,
    pub entries: Vec<MemoryResultEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryResultEntry {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub created_at: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStoredContent {
    pub store: String,
    pub id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serde_lowercase() {
        let json = serde_json::to_string(&Role::User).unwrap();
        assert_eq!(json, "\"user\"");
        let json = serde_json::to_string(&Role::System).unwrap();
        assert_eq!(json, "\"system\"");
        let json = serde_json::to_string(&Role::Assistant).unwrap();
        assert_eq!(json, "\"assistant\"");
    }

    #[test]
    fn test_role_deserialization() {
        let r: Role = serde_json::from_str("\"user\"").unwrap();
        assert_eq!(r, Role::User);
        let r: Role = serde_json::from_str("\"system\"").unwrap();
        assert_eq!(r, Role::System);
        let r: Role = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(r, Role::Assistant);
    }

    #[test]
    fn test_source_serde_lowercase() {
        let json = serde_json::to_string(&Source::Client).unwrap();
        assert_eq!(json, "\"client\"");
        let json = serde_json::to_string(&Source::Gateway).unwrap();
        assert_eq!(json, "\"gateway\"");
        let json = serde_json::to_string(&Source::Provider).unwrap();
        assert_eq!(json, "\"provider\"");
        let json = serde_json::to_string(&Source::Memory).unwrap();
        assert_eq!(json, "\"memory\"");
    }

    #[test]
    fn test_source_deserialization() {
        let s: Source = serde_json::from_str("\"client\"").unwrap();
        assert_eq!(s, Source::Client);
        let s: Source = serde_json::from_str("\"tool\"").unwrap();
        assert_eq!(s, Source::Tool);
        let s: Source = serde_json::from_str("\"judge\"").unwrap();
        assert_eq!(s, Source::Judge);
    }

    #[test]
    fn test_required_capability_text_none() {
        let part = ContentPart::Text("hello".to_string());
        assert_eq!(part.required_capability(), None);
    }

    #[test]
    fn test_required_capability_image_vision() {
        let part = ContentPart::Image(MediaContent {
            source: MediaSource::Url("https://example.com/img.png".to_string()),
            media_type: Some("image/png".to_string()),
        });
        assert_eq!(part.required_capability(), Some("vision"));
    }

    #[test]
    fn test_required_capability_audio() {
        let part = ContentPart::Audio(MediaContent {
            source: MediaSource::Url("https://example.com/audio.wav".to_string()),
            media_type: None,
        });
        assert_eq!(part.required_capability(), Some("audio_transcriptions"));
    }

    #[test]
    fn test_required_capability_video() {
        let part = ContentPart::Video(MediaContent {
            source: MediaSource::Data(vec![]),
            media_type: None,
        });
        assert_eq!(part.required_capability(), Some("video"));
    }

    #[test]
    fn test_required_capability_document() {
        let part = ContentPart::Document(DocumentContent {
            source: MediaSource::Url("https://example.com/doc.pdf".to_string()),
            media_type: Some("application/pdf".to_string()),
            filename: Some("doc.pdf".to_string()),
        });
        assert_eq!(part.required_capability(), Some("document_understanding"));
    }

    #[test]
    fn test_required_capability_activity_none() {
        let part = ContentPart::Routing(RoutingActivity {
            model: "auto".to_string(),
            score: 0.9,
            filters: vec![],
        });
        assert_eq!(part.required_capability(), None);

        let part = ContentPart::Hook(HookActivity {
            event: "pre_request".to_string(),
            hook_name: "rate-limiter".to_string(),
            decision: "allow".to_string(),
            reason: None,
        });
        assert_eq!(part.required_capability(), None);
    }

    #[test]
    fn test_required_capability_tool_none() {
        let part = ContentPart::CommandCall(CommandCallContent {
            command: "search".to_string(),
            arguments_json: "{}".to_string(),
        });
        assert_eq!(part.required_capability(), None);

        let part = ContentPart::CommandResult(CommandResultContent {
            command: "search".to_string(),
            success: true,
            output: "results".to_string(),
            error: None,
        });
        assert_eq!(part.required_capability(), None);
    }

    #[test]
    fn test_type_name() {
        assert_eq!(ContentPart::Text("x".to_string()).type_name(), "Text");
        assert_eq!(
            ContentPart::Image(MediaContent {
                source: MediaSource::Url("u".to_string()),
                media_type: None
            })
            .type_name(),
            "Image"
        );
        assert_eq!(
            ContentPart::CommandCall(CommandCallContent {
                command: "x".to_string(),
                arguments_json: "{}".to_string()
            })
            .type_name(),
            "CommandCall"
        );
        assert_eq!(
            ContentPart::MemoryStored(MemoryStoredContent {
                store: "s".to_string(),
                id: "i".to_string()
            })
            .type_name(),
            "MemoryStored"
        );
    }

    #[test]
    fn test_weft_message_constructible() {
        let msg = WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        };
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.source, Source::Client);
        assert!(msg.model.is_none());
        assert_eq!(msg.content.len(), 1);
        assert!(!msg.delta);
        assert_eq!(msg.message_index, 0);
    }
}
