//! `weft_proto` — Proto definitions and generated types for the Weft Wire protocol.
//!
//! This crate is the source of truth for the wire format. It contains only proto
//! definitions and tonic/prost generated code. It has no internal dependencies.
//!
//! # Usage
//!
//! ```rust
//! use weft_proto::weft::v1::{ChatRequest, WeftMessage, Role};
//! ```

pub mod weft {
    pub mod v1 {
        tonic::include_proto!("weft.v1");
    }
}

#[cfg(test)]
mod tests {
    use super::weft::v1::*;

    /// Verify all top-level generated message types can be constructed.
    /// This is a compile-time check — if generated types change shape, these fail.
    #[test]
    fn construct_chat_request() {
        let _req = ChatRequest {
            messages: vec![],
            model: String::new(),
            options: None,
        };
    }

    #[test]
    fn construct_sampling_options() {
        let _opts = SamplingOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 1024,
            stop: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            response_format: None,
            activity: false,
        };
    }

    #[test]
    fn construct_response_format() {
        let _fmt = ResponseFormat {
            r#type: "text".to_string(),
        };
    }

    #[test]
    fn construct_weft_message() {
        let _msg = WeftMessage {
            role: Role::User as i32,
            source: Source::Client as i32,
            model: String::new(),
            content: vec![],
            delta: false,
            message_index: 0,
        };
    }

    #[test]
    fn construct_content_part_text() {
        let _part = ContentPart {
            part: Some(content_part::Part::Text(TextContent {
                text: "hello".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_image_url() {
        let _part = ContentPart {
            part: Some(content_part::Part::Image(ImageContent {
                source: Some(image_content::Source::Url(
                    "https://example.com/img.png".to_string(),
                )),
                media_type: "image/png".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_image_data() {
        let _part = ContentPart {
            part: Some(content_part::Part::Image(ImageContent {
                source: Some(image_content::Source::Data(vec![0u8, 1u8, 2u8].into())),
                media_type: "image/jpeg".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_audio() {
        let _part = ContentPart {
            part: Some(content_part::Part::Audio(AudioContent {
                source: Some(audio_content::Source::Url(
                    "https://example.com/audio.wav".to_string(),
                )),
                media_type: "audio/wav".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_video() {
        let _part = ContentPart {
            part: Some(content_part::Part::Video(VideoContent {
                source: Some(video_content::Source::Url(
                    "https://example.com/video.mp4".to_string(),
                )),
                media_type: "video/mp4".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_document() {
        let _part = ContentPart {
            part: Some(content_part::Part::Document(DocumentContent {
                source: Some(document_content::Source::Url(
                    "https://example.com/doc.pdf".to_string(),
                )),
                media_type: "application/pdf".to_string(),
                filename: "doc.pdf".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_routing() {
        let _part = ContentPart {
            part: Some(content_part::Part::Routing(RoutingContent {
                model: "anthropic/opus".to_string(),
                score: 0.95,
                filters: vec![],
            })),
        };
    }

    #[test]
    fn construct_content_part_hook() {
        let _part = ContentPart {
            part: Some(content_part::Part::Hook(HookContent {
                event: "pre_request".to_string(),
                hook_name: "rate_limiter".to_string(),
                decision: "allow".to_string(),
                reason: String::new(),
            })),
        };
    }

    #[test]
    fn construct_content_part_council_start() {
        let _part = ContentPart {
            part: Some(content_part::Part::CouncilStart(CouncilStartContent {
                models: vec!["gpt-4".to_string(), "claude-3".to_string()],
                judge: "claude-3-opus".to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_command_call() {
        let _part = ContentPart {
            part: Some(content_part::Part::CommandCall(CommandCallContent {
                command: "search".to_string(),
                arguments_json: r#"{"query":"rust"}"#.to_string(),
            })),
        };
    }

    #[test]
    fn construct_content_part_command_result() {
        let _part = ContentPart {
            part: Some(content_part::Part::CommandResult(CommandResultContent {
                command: "search".to_string(),
                success: true,
                output: "results here".to_string(),
                error: String::new(),
            })),
        };
    }

    #[test]
    fn construct_content_part_memory_result() {
        let _part = ContentPart {
            part: Some(content_part::Part::MemoryResult(MemoryResultContent {
                store: "default".to_string(),
                entries: vec![MemoryEntry {
                    id: "mem-1".to_string(),
                    content: "some memory".to_string(),
                    score: 0.9,
                    created_at: "2026-03-16T00:00:00Z".to_string(),
                }],
            })),
        };
    }

    #[test]
    fn construct_content_part_memory_stored() {
        let _part = ContentPart {
            part: Some(content_part::Part::MemoryStored(MemoryStoredContent {
                store: "default".to_string(),
                id: "mem-42".to_string(),
            })),
        };
    }

    #[test]
    fn construct_chat_response() {
        let _resp = ChatResponse {
            id: "req-1".to_string(),
            model: "auto".to_string(),
            messages: vec![],
            usage: None,
            timing: None,
        };
    }

    #[test]
    fn construct_usage_info() {
        let _usage = UsageInfo {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            llm_calls: 1,
        };
    }

    #[test]
    fn construct_timing_info() {
        let _timing = TimingInfo {
            total_ms: 500,
            routing_ms: 10,
            llm_ms: 490,
        };
    }

    #[test]
    fn construct_chat_event_message() {
        let _event = ChatEvent {
            event: Some(chat_event::Event::Message(WeftMessage {
                role: Role::Assistant as i32,
                source: Source::Provider as i32,
                model: "gpt-4".to_string(),
                content: vec![],
                delta: false,
                message_index: 0,
            })),
        };
    }

    #[test]
    fn construct_chat_event_metadata() {
        let _event = ChatEvent {
            event: Some(chat_event::Event::Metadata(RequestMetadata {
                id: "req-1".to_string(),
                model: "auto".to_string(),
                usage: None,
                timing: None,
            })),
        };
    }

    #[test]
    fn construct_chat_event_error() {
        let _event = ChatEvent {
            event: Some(chat_event::Event::Error(ChatError {
                code: "internal_error".to_string(),
                message: "something went wrong".to_string(),
            })),
        };
    }

    #[test]
    fn construct_live_input_setup() {
        let _input = LiveInput {
            input: Some(live_input::Input::Setup(ChatRequest {
                messages: vec![],
                model: "auto".to_string(),
                options: None,
            })),
        };
    }

    #[test]
    fn construct_live_input_message() {
        let _input = LiveInput {
            input: Some(live_input::Input::Message(WeftMessage {
                role: Role::User as i32,
                source: Source::Client as i32,
                model: String::new(),
                content: vec![],
                delta: false,
                message_index: 0,
            })),
        };
    }

    #[test]
    fn construct_live_input_control() {
        let _input = LiveInput {
            input: Some(live_input::Input::Control(LiveControl {
                action: "cancel".to_string(),
            })),
        };
    }

    #[test]
    fn construct_live_output_event() {
        let _output = LiveOutput {
            output: Some(live_output::Output::Event(ChatEvent { event: None })),
        };
    }

    #[test]
    fn construct_live_output_control() {
        let _output = LiveOutput {
            output: Some(live_output::Output::Control(LiveControl {
                action: "pong".to_string(),
            })),
        };
    }

    #[test]
    fn role_enum_values() {
        // Prost strips the enum name prefix per proto3 conventions.
        assert_eq!(Role::Unspecified as i32, 0);
        assert_eq!(Role::User as i32, 1);
        assert_eq!(Role::Assistant as i32, 2);
        assert_eq!(Role::System as i32, 3);
    }

    #[test]
    fn source_enum_values() {
        // Prost strips the enum name prefix per proto3 conventions.
        assert_eq!(Source::Unspecified as i32, 0);
        assert_eq!(Source::Client as i32, 1);
        assert_eq!(Source::Gateway as i32, 2);
        assert_eq!(Source::Provider as i32, 3);
        assert_eq!(Source::Member as i32, 4);
        assert_eq!(Source::Judge as i32, 5);
        assert_eq!(Source::Tool as i32, 6);
        assert_eq!(Source::Memory as i32, 7);
    }
}
