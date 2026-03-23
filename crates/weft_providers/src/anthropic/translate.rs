//! Anthropic wire format translation functions.
//!
//! Direction-tagged functions for converting between Weft domain types
//! and Anthropic wire format types. Outbound functions are used by the
//! provider client. Inbound functions will be added in Phase 3.

use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};

use super::wire::{AnthropicMessage, AnthropicRequest, AnthropicResponse};
use crate::{TokenUsage, provider::extract_text_messages};

// ── Outbound (Weft -> Anthropic) ─────────────────────────────────────────────

/// Build an Anthropic wire request from Weft domain types.
///
/// Extracts the system prompt from `messages[0]` to the top-level `system`
/// field (`Some(text)`). Additional system-role messages are concatenated.
/// If no system messages exist, `system` is `None`.
/// Anthropic does not allow system role in the messages array.
/// Defaults `max_tokens` to 4096 when unspecified.
pub fn build_outbound_request(
    messages: &[WeftMessage],
    model: &str,
    options: &SamplingOptions,
) -> AnthropicRequest {
    let mut system_parts: Vec<String> = Vec::new();
    let mut wire_messages = Vec::new();

    // Extract system prompt from messages[0] if present.
    // Anthropic requires the system prompt in a dedicated top-level `system` field,
    // NOT in the messages array.
    let conversation_start = if messages.first().map(|m| m.role) == Some(Role::System) {
        let system_text: String = messages[0]
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !system_text.is_empty() {
            system_parts.push(system_text);
        }
        1
    } else {
        0
    };

    // Process remaining messages. System-role messages that survive the gateway
    // activity filter are concatenated into the system field (Anthropic does not
    // allow system role in the messages array).
    for (role, text) in extract_text_messages(&messages[conversation_start..]) {
        match role {
            Role::System => system_parts.push(text),
            Role::User => wire_messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: text,
            }),
            Role::Assistant => wire_messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: text,
            }),
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    // Anthropic requires max_tokens; default to 4096 if not specified.
    let max_tokens = options.max_tokens.unwrap_or(4096);

    AnthropicRequest {
        model: model.to_string(),
        system,
        messages: wire_messages,
        max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        stream: None,
    }
}

/// Parse an Anthropic wire response into a `WeftMessage` and optional `TokenUsage`.
///
/// Extracts text from the first content block with `type: "text"`.
/// Maps `input_tokens` / `output_tokens` to `TokenUsage`.
pub fn parse_outbound_response(
    response: &AnthropicResponse,
    model: String,
) -> (weft_core::WeftMessage, Option<TokenUsage>) {
    let text = response
        .content
        .iter()
        .find(|b| b.kind == "text")
        .and_then(|b| b.text.clone())
        .unwrap_or_default();

    let usage = response.usage.as_ref().map(|u| TokenUsage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
    });

    let message = weft_core::WeftMessage {
        role: Role::Assistant,
        source: Source::Provider,
        model: Some(model),
        content: vec![ContentPart::Text(text)],
        delta: false,
        message_index: 0,
    };

    (message, usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};

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

    // ── build_outbound_request ───────────────────────────────────────────────

    #[test]
    fn test_build_outbound_system_prompt_extracted() {
        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "You are helpful."),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let req = build_outbound_request(&messages, "claude-3", &SamplingOptions::default());
        assert_eq!(req.system, Some("You are helpful.".to_string()));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "Hello");
    }

    #[test]
    fn test_build_outbound_no_system_yields_none() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hello")];
        let req = build_outbound_request(&messages, "claude-3", &SamplingOptions::default());
        assert!(req.system.is_none());
    }

    #[test]
    fn test_build_outbound_empty_system_yields_none() {
        // messages[0] is Role::System but has no text content → system is None
        let system_msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![],
            delta: false,
            message_index: 0,
        };
        let messages = vec![
            system_msg,
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let req = build_outbound_request(&messages, "claude-3", &SamplingOptions::default());
        assert!(req.system.is_none());
    }

    #[test]
    fn test_build_outbound_additional_system_messages_concatenated() {
        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "base system"),
            make_weft_message(Role::System, Source::Client, "You are a helpful assistant."),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let req = build_outbound_request(&messages, "claude-3", &SamplingOptions::default());
        assert_eq!(
            req.system,
            Some("base system\n\nYou are a helpful assistant.".to_string())
        );
        // Only user message in messages array
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn test_build_outbound_gateway_activity_filtered() {
        let activity_msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![],
            delta: false,
            message_index: 0,
        };
        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "sys prompt"),
            activity_msg,
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let req = build_outbound_request(&messages, "claude-3", &SamplingOptions::default());
        assert_eq!(req.system, Some("sys prompt".to_string()));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn test_build_outbound_max_tokens_defaults_to_4096() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hello")];
        let req = build_outbound_request(
            &messages,
            "claude-3",
            &SamplingOptions {
                max_tokens: None,
                ..Default::default()
            },
        );
        assert_eq!(req.max_tokens, 4096);
    }

    #[test]
    fn test_build_outbound_max_tokens_from_options() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hello")];
        let req = build_outbound_request(
            &messages,
            "claude-3",
            &SamplingOptions {
                max_tokens: Some(512),
                ..Default::default()
            },
        );
        assert_eq!(req.max_tokens, 512);
    }

    #[test]
    fn test_build_outbound_sampling_options_mapped() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hi")];
        let options = SamplingOptions {
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(50),
            max_tokens: Some(100),
            ..Default::default()
        };
        let req = build_outbound_request(&messages, "claude-3", &options);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.95));
        assert_eq!(req.top_k, Some(50));
        assert_eq!(req.max_tokens, 100);
        // stream is always None for outbound
        assert!(req.stream.is_none());
    }

    // ── parse_outbound_response ──────────────────────────────────────────────

    #[test]
    fn test_parse_outbound_response_extracts_text() {
        use super::super::wire::{AnthropicUsage, ContentBlock};
        let response = AnthropicResponse {
            id: None,
            kind: None,
            role: None,
            model: None,
            content: vec![ContentBlock {
                kind: "text".to_string(),
                text: Some("Hello!".to_string()),
            }],
            usage: Some(AnthropicUsage {
                input_tokens: 10,
                output_tokens: 8,
            }),
            stop_reason: None,
        };
        let (msg, usage) = parse_outbound_response(&response, "claude-3".to_string());
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.source, Source::Provider);
        assert_eq!(msg.model, Some("claude-3".to_string()));
        let text = match &msg.content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "Hello!");
        let u = usage.expect("usage must be present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 8);
    }

    #[test]
    fn test_parse_outbound_response_non_text_block_skipped() {
        use super::super::wire::ContentBlock;
        let response = AnthropicResponse {
            id: None,
            kind: None,
            role: None,
            model: None,
            content: vec![
                ContentBlock {
                    kind: "image".to_string(),
                    text: None,
                },
                ContentBlock {
                    kind: "text".to_string(),
                    text: Some("Found text".to_string()),
                },
            ],
            usage: None,
            stop_reason: None,
        };
        let (msg, _) = parse_outbound_response(&response, "claude-3".to_string());
        let text = match &msg.content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "Found text");
    }

    #[test]
    fn test_parse_outbound_response_empty_content() {
        let response = AnthropicResponse {
            id: None,
            kind: None,
            role: None,
            model: None,
            content: vec![],
            usage: None,
            stop_reason: None,
        };
        let (msg, usage) = parse_outbound_response(&response, "claude-3".to_string());
        let text = match &msg.content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "");
        assert!(usage.is_none());
    }
}
