//! Anthropic wire format translation functions.
//!
//! Direction-tagged functions for converting between Weft domain types
//! and Anthropic wire format types. Outbound functions are used by the
//! provider client. Inbound functions are used by the compat endpoint.

use weft_core::{ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftMessage};

use super::wire::{
    AnthropicContent, AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicUsage,
    ContentBlock,
};
use crate::{TokenUsage, provider::extract_text_messages, translate::TranslationError};

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
                content: AnthropicContent::Text(text),
            }),
            Role::Assistant => wire_messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: AnthropicContent::Text(text),
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

// ── Inbound (Anthropic -> Weft) ───────────────────────────────────────────────

/// Flatten an `AnthropicContent` value into a plain text string.
///
/// - `Text(s)` → returns `s` directly.
/// - `Blocks(blocks)` → concatenates the `text` field of all `type: "text"` blocks,
///   joined by newline. Non-text block types are silently ignored because this
///   compat endpoint handles text completions only.
fn flatten_inbound_content(content: AnthropicContent) -> String {
    match content {
        AnthropicContent::Text(s) => s,
        AnthropicContent::Blocks(blocks) => blocks
            .into_iter()
            .filter(|b| b.content_type == "text")
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Translate an Anthropic-format inbound request into a domain `WeftRequest`.
///
/// Maps message roles: "user" → `Source::Client`, "assistant" → `Source::Provider`.
/// Extracts the top-level `system` field as `messages[0]` with `Role::System` and
/// `Source::Gateway` when the field is `Some(non-empty string)`.
/// Maps `temperature`, `max_tokens`, `top_p`, `top_k` to `SamplingOptions`.
/// Parses `ModelRoutingInstruction` from the model string.
///
/// Returns `Err` if any message has an unrecognized role string.
pub fn parse_inbound_request(
    request: AnthropicRequest,
) -> Result<weft_core::WeftRequest, TranslationError> {
    let mut messages = Vec::new();

    // Prepend system field as messages[0] when non-empty.
    // Absent or empty system fields produce no system message.
    if let Some(system_text) = request.system
        && !system_text.is_empty()
    {
        messages.push(WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![ContentPart::Text(system_text)],
            delta: false,
            message_index: 0,
        });
    }

    for msg in request.messages {
        let (role, source) = match msg.role.as_str() {
            "user" => (Role::User, Source::Client),
            "assistant" => (Role::Assistant, Source::Provider),
            unknown => return Err(TranslationError::UnrecognizedRole(unknown.to_string())),
        };

        // Flatten both content forms into a single text string.
        // Block arrays concatenate all `type: "text"` blocks; other block types
        // (image, tool_use, tool_result) are intentionally ignored since this
        // compat endpoint only handles text completions.
        let text = flatten_inbound_content(msg.content);

        messages.push(WeftMessage {
            role,
            source,
            model: None,
            content: vec![ContentPart::Text(text)],
            delta: false,
            message_index: 0,
        });
    }

    let routing = ModelRoutingInstruction::parse(&request.model);

    // Anthropic format always provides max_tokens (required field).
    let options = SamplingOptions {
        temperature: request.temperature,
        max_tokens: Some(request.max_tokens),
        top_p: request.top_p,
        top_k: request.top_k,
        // Anthropic does not support these — leave at defaults.
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        stop: vec![],
        ..Default::default()
    };

    Ok(weft_core::WeftRequest {
        messages,
        routing,
        options,
    })
}

/// Translate a domain `WeftResponse` into an Anthropic-format response.
///
/// Extracts the last assistant/provider text message. Constructs the Anthropic
/// response envelope with content blocks, usage, stop_reason.
/// Sets `id` from `response.id` with `msg_` prefix if not already present.
/// Sets `kind: "message"`, `role: "assistant"`, `model: request_model`,
/// `stop_reason: "end_turn"`.
pub fn build_inbound_response(
    response: weft_core::WeftResponse,
    request_model: String,
) -> AnthropicResponse {
    // Extract the last assistant/provider text message.
    let assistant_text = response
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && m.source == Source::Provider)
        .and_then(|m| {
            m.content.iter().find_map(|part| match part {
                ContentPart::Text(text) => Some(text.clone()),
                _ => None,
            })
        })
        .unwrap_or_default();

    // Prefix id with "msg_" to match Anthropic wire format.
    let id = if response.id.starts_with("msg_") {
        response.id
    } else {
        format!("msg_{}", response.id)
    };

    AnthropicResponse {
        id: Some(id),
        kind: Some("message".to_string()),
        role: Some("assistant".to_string()),
        model: Some(request_model),
        content: vec![ContentBlock {
            kind: "text".to_string(),
            text: Some(assistant_text),
        }],
        usage: Some(AnthropicUsage {
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        }),
        stop_reason: Some("end_turn".to_string()),
    }
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
        assert_eq!(
            req.messages[0].content,
            super::super::wire::AnthropicContent::Text("Hello".to_string())
        );
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

    // ── parse_inbound_request ────────────────────────────────────────────────

    fn make_inbound_request(system: Option<&str>, messages: Vec<(&str, &str)>) -> AnthropicRequest {
        use super::super::wire::{AnthropicContent, AnthropicMessage};
        AnthropicRequest {
            model: "claude-3-opus-20240229".to_string(),
            system: system.map(|s| s.to_string()),
            messages: messages
                .into_iter()
                .map(|(role, content)| AnthropicMessage {
                    role: role.to_string(),
                    content: AnthropicContent::Text(content.to_string()),
                })
                .collect(),
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
        }
    }

    #[test]
    fn test_parse_inbound_system_field_prepended() {
        let req = make_inbound_request(Some("You are helpful."), vec![("user", "Hello")]);
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.messages.len(), 2);
        assert_eq!(weft_req.messages[0].role, Role::System);
        assert_eq!(weft_req.messages[0].source, Source::Gateway);
        let text = match &weft_req.messages[0].content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "You are helpful.");
    }

    #[test]
    fn test_parse_inbound_no_system_field() {
        let req = make_inbound_request(None, vec![("user", "Hello")]);
        let weft_req = parse_inbound_request(req).expect("should succeed");
        // No system message prepended
        assert_eq!(weft_req.messages.len(), 1);
        assert_eq!(weft_req.messages[0].role, Role::User);
    }

    #[test]
    fn test_parse_inbound_empty_system_field_skipped() {
        // Some("") → no system message prepended
        let req = make_inbound_request(Some(""), vec![("user", "Hello")]);
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.messages.len(), 1);
        assert_eq!(weft_req.messages[0].role, Role::User);
    }

    #[test]
    fn test_parse_inbound_user_assistant_roles() {
        let req = make_inbound_request(
            None,
            vec![
                ("user", "Hello"),
                ("assistant", "Hi there"),
                ("user", "How are you?"),
            ],
        );
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.messages[0].role, Role::User);
        assert_eq!(weft_req.messages[0].source, Source::Client);
        assert_eq!(weft_req.messages[1].role, Role::Assistant);
        assert_eq!(weft_req.messages[1].source, Source::Provider);
        assert_eq!(weft_req.messages[2].role, Role::User);
        assert_eq!(weft_req.messages[2].source, Source::Client);
    }

    #[test]
    fn test_parse_inbound_unrecognized_role_returns_error() {
        use super::super::wire::{AnthropicContent, AnthropicMessage};
        let mut req = make_inbound_request(None, vec![]);
        req.messages.push(AnthropicMessage {
            role: "system".to_string(),
            content: AnthropicContent::Text("should not be here".to_string()),
        });
        let result = parse_inbound_request(req);
        assert!(matches!(
            result,
            Err(TranslationError::UnrecognizedRole(r)) if r == "system"
        ));
    }

    #[test]
    fn test_parse_inbound_tool_role_returns_error() {
        use super::super::wire::{AnthropicContent, AnthropicMessage};
        let mut req = make_inbound_request(None, vec![]);
        req.messages.push(AnthropicMessage {
            role: "tool".to_string(),
            content: AnthropicContent::Text("tool result".to_string()),
        });
        let result = parse_inbound_request(req);
        assert!(matches!(
            result,
            Err(TranslationError::UnrecognizedRole(r)) if r == "tool"
        ));
    }

    #[test]
    fn test_parse_inbound_sampling_options_mapped() {
        use super::super::wire::{AnthropicContent, AnthropicMessage};
        let req = AnthropicRequest {
            model: "claude-3-opus-20240229".to_string(),
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Text("hi".to_string()),
            }],
            max_tokens: 512,
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
            stream: None,
        };
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.options.max_tokens, Some(512));
        assert_eq!(weft_req.options.temperature, Some(0.7));
        assert_eq!(weft_req.options.top_p, Some(0.95));
        assert_eq!(weft_req.options.top_k, Some(40));
        // Fields not supported by Anthropic are None/default
        assert!(weft_req.options.frequency_penalty.is_none());
        assert!(weft_req.options.presence_penalty.is_none());
        assert!(weft_req.options.seed.is_none());
        assert!(weft_req.options.stop.is_empty());
    }

    // ── build_inbound_response ───────────────────────────────────────────────

    fn make_weft_response(id: &str, text: &str) -> weft_core::WeftResponse {
        use weft_core::{WeftTiming, WeftUsage};
        weft_core::WeftResponse {
            id: id.to_string(),
            model: "test-model".to_string(),
            messages: vec![WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: None,
                content: vec![ContentPart::Text(text.to_string())],
                delta: false,
                message_index: 0,
            }],
            usage: WeftUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                llm_calls: 1,
            },
            timing: WeftTiming::default(),
            degradations: vec![],
        }
    }

    #[test]
    fn test_build_inbound_response_content_block() {
        let resp = make_weft_response("test-id", "Hello from Anthropic!");
        let anthropic_resp = build_inbound_response(resp, "claude-3-opus-20240229".to_string());
        assert_eq!(anthropic_resp.content.len(), 1);
        assert_eq!(anthropic_resp.content[0].kind, "text");
        assert_eq!(
            anthropic_resp.content[0].text,
            Some("Hello from Anthropic!".to_string())
        );
    }

    #[test]
    fn test_build_inbound_response_usage_mapping() {
        let resp = make_weft_response("id", "Hi");
        let anthropic_resp = build_inbound_response(resp, "claude-3".to_string());
        let usage = anthropic_resp.usage.expect("usage must be present");
        // prompt_tokens → input_tokens, completion_tokens → output_tokens
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_build_inbound_response_id_prefix() {
        // id without prefix gets "msg_" prepended
        let resp = make_weft_response("abc-123", "Hi");
        let anthropic_resp = build_inbound_response(resp, "claude-3".to_string());
        assert_eq!(anthropic_resp.id, Some("msg_abc-123".to_string()));
    }

    #[test]
    fn test_build_inbound_response_id_already_prefixed() {
        // id with "msg_" prefix is not double-prefixed
        let resp = make_weft_response("msg_already", "Hi");
        let anthropic_resp = build_inbound_response(resp, "claude-3".to_string());
        assert_eq!(anthropic_resp.id, Some("msg_already".to_string()));
    }

    #[test]
    fn test_build_inbound_response_envelope_fields() {
        let resp = make_weft_response("id", "Response text");
        let anthropic_resp = build_inbound_response(resp, "claude-3-haiku".to_string());
        assert_eq!(anthropic_resp.kind, Some("message".to_string()));
        assert_eq!(anthropic_resp.role, Some("assistant".to_string()));
        assert_eq!(anthropic_resp.model, Some("claude-3-haiku".to_string()));
        assert_eq!(anthropic_resp.stop_reason, Some("end_turn".to_string()));
    }

    #[test]
    fn test_build_inbound_response_empty_messages() {
        use weft_core::{WeftTiming, WeftUsage};
        let resp = weft_core::WeftResponse {
            id: "test".to_string(),
            model: "auto".to_string(),
            messages: vec![],
            usage: WeftUsage::default(),
            timing: WeftTiming::default(),
            degradations: vec![],
        };
        let anthropic_resp = build_inbound_response(resp, "claude-3".to_string());
        // Empty messages → empty content text
        assert_eq!(anthropic_resp.content[0].text, Some(String::new()));
    }

    // ── AnthropicContent serialization / deserialization ────────────────────

    /// String-form content round-trips through serde without change.
    #[test]
    fn test_content_string_form_round_trips() {
        use super::super::wire::AnthropicContent;
        let content = AnthropicContent::Text("hello world".to_string());
        let json = serde_json::to_string(&content).expect("serialize");
        assert_eq!(json, r#""hello world""#);
        let back: AnthropicContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, AnthropicContent::Text("hello world".to_string()));
    }

    /// Array-form content deserializes to `Blocks`.
    #[test]
    fn test_content_array_form_deserializes() {
        use super::super::wire::AnthropicContent;
        let json = r#"[{"type":"text","text":"block one"},{"type":"text","text":"block two"}]"#;
        let content: AnthropicContent = serde_json::from_str(json).expect("deserialize");
        assert!(matches!(content, AnthropicContent::Blocks(_)));
        if let AnthropicContent::Blocks(blocks) = content {
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].text, "block one");
            assert_eq!(blocks[1].text, "block two");
        }
    }

    // ── flatten_inbound_content ──────────────────────────────────────────────

    /// String content flattens to the string value unchanged.
    #[test]
    fn test_flatten_string_content() {
        use super::super::wire::AnthropicContent;
        let result = flatten_inbound_content(AnthropicContent::Text("hello".to_string()));
        assert_eq!(result, "hello");
    }

    /// Block array with a single text block flattens to that block's text.
    #[test]
    fn test_flatten_single_text_block() {
        use super::super::wire::{AnthropicContent, InboundContentBlock};
        let content = AnthropicContent::Blocks(vec![InboundContentBlock {
            content_type: "text".to_string(),
            text: "hello from block".to_string(),
        }]);
        let result = flatten_inbound_content(content);
        assert_eq!(result, "hello from block");
    }

    /// Block array with multiple text blocks joins them with newline.
    #[test]
    fn test_flatten_multiple_text_blocks_joined_with_newline() {
        use super::super::wire::{AnthropicContent, InboundContentBlock};
        let content = AnthropicContent::Blocks(vec![
            InboundContentBlock {
                content_type: "text".to_string(),
                text: "first".to_string(),
            },
            InboundContentBlock {
                content_type: "text".to_string(),
                text: "second".to_string(),
            },
        ]);
        let result = flatten_inbound_content(content);
        assert_eq!(result, "first\nsecond");
    }

    /// Non-text blocks (image, tool_use) are ignored during flattening.
    #[test]
    fn test_flatten_non_text_blocks_ignored() {
        use super::super::wire::{AnthropicContent, InboundContentBlock};
        let content = AnthropicContent::Blocks(vec![
            InboundContentBlock {
                content_type: "image".to_string(),
                text: String::new(),
            },
            InboundContentBlock {
                content_type: "text".to_string(),
                text: "actual text".to_string(),
            },
            InboundContentBlock {
                content_type: "tool_use".to_string(),
                text: "ignored".to_string(),
            },
        ]);
        let result = flatten_inbound_content(content);
        assert_eq!(result, "actual text");
    }

    /// An empty block array flattens to an empty string.
    #[test]
    fn test_flatten_empty_blocks_yields_empty_string() {
        use super::super::wire::AnthropicContent;
        let result = flatten_inbound_content(AnthropicContent::Blocks(vec![]));
        assert_eq!(result, "");
    }

    // ── parse_inbound_request with array content ─────────────────────────────

    /// An inbound request with array-format message content is parsed correctly.
    #[test]
    fn test_parse_inbound_array_content_flattened_to_text() {
        use super::super::wire::{AnthropicContent, AnthropicMessage, InboundContentBlock};
        let req = AnthropicRequest {
            model: "claude-3-opus-20240229".to_string(),
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Blocks(vec![
                    InboundContentBlock {
                        content_type: "text".to_string(),
                        text: "Hello, ".to_string(),
                    },
                    InboundContentBlock {
                        content_type: "text".to_string(),
                        text: "world!".to_string(),
                    },
                ]),
            }],
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
        };
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.messages.len(), 1);
        assert_eq!(weft_req.messages[0].role, Role::User);
        let text = match &weft_req.messages[0].content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "Hello, \nworld!");
    }

    /// JSON string form deserializes into `AnthropicMessage` with `Text` variant.
    #[test]
    fn test_inbound_message_deserializes_string_content() {
        use super::super::wire::AnthropicMessage;
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).expect("deserialize");
        assert_eq!(msg.role, "user");
        assert_eq!(
            msg.content,
            super::super::wire::AnthropicContent::Text("hello".to_string())
        );
    }

    /// JSON array form deserializes into `AnthropicMessage` with `Blocks` variant.
    #[test]
    fn test_inbound_message_deserializes_array_content() {
        use super::super::wire::AnthropicMessage;
        let json = r#"{"role":"user","content":[{"type":"text","text":"hi"}]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).expect("deserialize");
        assert_eq!(msg.role, "user");
        assert!(matches!(
            msg.content,
            super::super::wire::AnthropicContent::Blocks(_)
        ));
    }
}
