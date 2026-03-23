//! OpenAI wire format translation functions.
//!
//! Direction-tagged functions for converting between Weft domain types
//! and OpenAI wire format types. Used by both the outbound provider client
//! and the inbound compat endpoint.

use weft_core::{ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftMessage};

use super::wire::{OpenAIChoice, OpenAIMessage, OpenAIRequest, OpenAIResponse, OpenAIUsage};
use crate::{TokenUsage, provider::extract_text_messages, translate::TranslationError};

// ── Outbound (Weft -> OpenAI) ────────────────────────────────────────────────

/// Build an OpenAI wire request from Weft domain types.
///
/// Extracts the system prompt from `messages[0]` if present (positional
/// convention: `Role::System` at index 0 is the gateway system prompt).
/// Filters gateway activity messages (system-role, no text content).
/// Maps `SamplingOptions` to the OpenAI-supported subset (all fields except `top_k`).
pub fn build_outbound_request(
    messages: &[WeftMessage],
    model: &str,
    options: &SamplingOptions,
) -> OpenAIRequest {
    let mut wire_messages = Vec::new();

    // Extract system prompt from messages[0] if present.
    // For OpenAI, system messages are in the messages array with role "system".
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
            wire_messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system_text,
            });
        }
        1
    } else {
        0
    };

    // Extract text from remaining WeftMessages, skipping gateway activity messages.
    for (role, text) in extract_text_messages(&messages[conversation_start..]) {
        let role_str = match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        wire_messages.push(OpenAIMessage {
            role: role_str.to_string(),
            content: text,
        });
    }

    OpenAIRequest {
        model: model.to_string(),
        messages: wire_messages,
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        seed: options.seed,
        stop: if options.stop.is_empty() {
            None
        } else {
            Some(options.stop.clone())
        },
        stream: None,
    }
}

/// Parse an OpenAI wire response into a `WeftMessage` and optional `TokenUsage`.
///
/// Extracts text from `choices[0].message.content`. Constructs a `WeftMessage`
/// with `Role::Assistant`, `Source::Provider`, and the given model name.
pub fn parse_outbound_response(
    response: &OpenAIResponse,
    model: String,
) -> (WeftMessage, Option<TokenUsage>) {
    let text = response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    let usage = response.usage.as_ref().map(|u| TokenUsage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
    });

    let message = WeftMessage {
        role: Role::Assistant,
        source: Source::Provider,
        model: Some(model),
        content: vec![ContentPart::Text(text)],
        delta: false,
        message_index: 0,
    };

    (message, usage)
}

// ── Inbound (OpenAI -> Weft) ─────────────────────────────────────────────────

/// Translate an OpenAI-format inbound request into a domain `WeftRequest`.
///
/// Maps message roles: "user" → `Source::Client`, "assistant" → `Source::Provider`,
/// "system" → `Source::Gateway`. Parses `ModelRoutingInstruction` from the model
/// string. Maps all available `SamplingOptions` fields (temperature, max_tokens,
/// top_p, frequency_penalty, presence_penalty, seed, stop).
///
/// Returns `Err` if any message has an unrecognized role string.
pub fn parse_inbound_request(
    request: OpenAIRequest,
) -> Result<weft_core::WeftRequest, TranslationError> {
    let mut messages = Vec::with_capacity(request.messages.len());

    for msg in request.messages {
        let (role, source) = match msg.role.as_str() {
            "system" => (Role::System, Source::Gateway),
            "user" => (Role::User, Source::Client),
            "assistant" => (Role::Assistant, Source::Provider),
            unknown => return Err(TranslationError::UnrecognizedRole(unknown.to_string())),
        };

        messages.push(WeftMessage {
            role,
            source,
            model: None,
            content: vec![ContentPart::Text(msg.content)],
            delta: false,
            message_index: 0,
        });
    }

    let routing = ModelRoutingInstruction::parse(&request.model);

    let options = SamplingOptions {
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        seed: request.seed,
        stop: request.stop.unwrap_or_default(),
        ..Default::default()
    };

    Ok(weft_core::WeftRequest {
        messages,
        routing,
        options,
    })
}

/// Translate a domain `WeftResponse` into an OpenAI-format response.
///
/// Extracts the last assistant/provider text message. Generates the response
/// envelope (id with `chatcmpl-` prefix, `object: "chat.completion"`, timestamp).
/// Echoes the request model string so clients see the model they asked for.
pub fn build_inbound_response(
    response: weft_core::WeftResponse,
    request_model: String,
) -> OpenAIResponse {
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

    // Prefix id with "chatcmpl-" to match OpenAI wire format.
    let id = if response.id.starts_with("chatcmpl-") {
        response.id
    } else {
        format!("chatcmpl-{}", response.id)
    };

    OpenAIResponse {
        id: Some(id),
        object: Some("chat.completion".to_string()),
        created: Some(unix_timestamp()),
        model: Some(request_model),
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: assistant_text,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(OpenAIUsage {
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            total_tokens: response.usage.total_tokens,
        }),
    }
}

/// Return the current Unix timestamp in seconds.
fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage, WeftResponse};

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

    fn make_weft_response(id: &str, text: &str) -> WeftResponse {
        use weft_core::{WeftTiming, WeftUsage};
        WeftResponse {
            id: id.to_string(),
            model: "test-model".to_string(),
            messages: vec![make_weft_message(Role::Assistant, Source::Provider, text)],
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

    // ── build_outbound_request ───────────────────────────────────────────────

    #[test]
    fn test_build_outbound_system_prompt_extracted() {
        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "You are helpful."),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let req = build_outbound_request(&messages, "gpt-4", &SamplingOptions::default());
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[0].content, "You are helpful.");
        assert_eq!(req.messages[1].role, "user");
        assert_eq!(req.messages[1].content, "Hello");
    }

    #[test]
    fn test_build_outbound_no_system_prompt() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hello")];
        let req = build_outbound_request(&messages, "gpt-4", &SamplingOptions::default());
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn test_build_outbound_gateway_activity_filtered() {
        // Activity messages: Role::System, Source::Gateway, no text content.
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
        let req = build_outbound_request(&messages, "gpt-4", &SamplingOptions::default());
        // system + user only, activity filtered out
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "user");
    }

    #[test]
    fn test_build_outbound_sampling_options_mapped() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hi")];
        let options = SamplingOptions {
            max_tokens: Some(256),
            temperature: Some(0.5),
            top_p: Some(0.9),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.2),
            seed: Some(42),
            stop: vec!["STOP".to_string()],
            ..Default::default()
        };
        let req = build_outbound_request(&messages, "gpt-4", &options);
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.frequency_penalty, Some(0.1));
        assert_eq!(req.presence_penalty, Some(0.2));
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.stop, Some(vec!["STOP".to_string()]));
        // stream is always None for outbound
        assert!(req.stream.is_none());
    }

    #[test]
    fn test_build_outbound_empty_stop_omitted() {
        let messages = vec![make_weft_message(Role::User, Source::Client, "Hi")];
        let options = SamplingOptions {
            stop: vec![],
            ..Default::default()
        };
        let req = build_outbound_request(&messages, "gpt-4", &options);
        assert!(req.stop.is_none());
    }

    // ── parse_outbound_response ──────────────────────────────────────────────

    #[test]
    fn test_parse_outbound_response_extracts_text() {
        let response = OpenAIResponse {
            id: Some("chatcmpl-1".to_string()),
            object: Some("chat.completion".to_string()),
            created: Some(0),
            model: Some("gpt-4".to_string()),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };
        let (msg, usage) = parse_outbound_response(&response, "gpt-4".to_string());
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.source, Source::Provider);
        assert_eq!(msg.model, Some("gpt-4".to_string()));
        let text = match &msg.content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "Hello!");
        let u = usage.expect("usage must be present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
    }

    #[test]
    fn test_parse_outbound_response_empty_choices() {
        let response = OpenAIResponse {
            id: None,
            object: None,
            created: None,
            model: None,
            choices: vec![],
            usage: None,
        };
        let (msg, usage) = parse_outbound_response(&response, "gpt-4".to_string());
        let text = match &msg.content[0] {
            ContentPart::Text(t) => t.as_str(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "");
        assert!(usage.is_none());
    }

    // ── parse_inbound_request ────────────────────────────────────────────────

    #[test]
    fn test_parse_inbound_assigns_sources() {
        let req = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![
                OpenAIMessage {
                    role: "system".to_string(),
                    content: "sys".to_string(),
                },
                OpenAIMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
                OpenAIMessage {
                    role: "assistant".to_string(),
                    content: "hi".to_string(),
                },
            ],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            stream: None,
        };
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.messages[0].source, Source::Gateway);
        assert_eq!(weft_req.messages[0].role, Role::System);
        assert_eq!(weft_req.messages[1].source, Source::Client);
        assert_eq!(weft_req.messages[1].role, Role::User);
        assert_eq!(weft_req.messages[2].source, Source::Provider);
        assert_eq!(weft_req.messages[2].role, Role::Assistant);
    }

    #[test]
    fn test_parse_inbound_unrecognized_role_returns_error() {
        let req = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAIMessage {
                role: "tool".to_string(),
                content: "some tool result".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            stream: None,
        };
        let result = parse_inbound_request(req);
        assert!(matches!(
            result,
            Err(TranslationError::UnrecognizedRole(r)) if r == "tool"
        ));
    }

    #[test]
    fn test_parse_inbound_function_role_returns_error() {
        let req = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAIMessage {
                role: "function".to_string(),
                content: "".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            stream: None,
        };
        let result = parse_inbound_request(req);
        assert!(matches!(
            result,
            Err(TranslationError::UnrecognizedRole(r)) if r == "function"
        ));
    }

    #[test]
    fn test_parse_inbound_routing_parsed() {
        use weft_core::RoutingMode;
        let req = OpenAIRequest {
            model: "auto".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            stream: None,
        };
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.routing.mode, RoutingMode::Auto);
    }

    #[test]
    fn test_parse_inbound_sampling_options_mapped() {
        let req = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            max_tokens: Some(256),
            temperature: Some(0.5),
            top_p: Some(0.9),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.2),
            seed: Some(42),
            stop: Some(vec!["STOP".to_string()]),
            stream: None,
        };
        let weft_req = parse_inbound_request(req).expect("should succeed");
        assert_eq!(weft_req.options.max_tokens, Some(256));
        assert_eq!(weft_req.options.temperature, Some(0.5));
        assert_eq!(weft_req.options.top_p, Some(0.9));
        assert_eq!(weft_req.options.frequency_penalty, Some(0.1));
        assert_eq!(weft_req.options.presence_penalty, Some(0.2));
        assert_eq!(weft_req.options.seed, Some(42));
        assert_eq!(weft_req.options.stop, vec!["STOP".to_string()]);
    }

    // ── build_inbound_response ───────────────────────────────────────────────

    #[test]
    fn test_build_inbound_response_extracts_assistant_text() {
        let resp = make_weft_response("test-id", "Hello!");
        let openai_resp = build_inbound_response(resp, "gpt-4".to_string());
        assert_eq!(openai_resp.choices[0].message.content, "Hello!");
        assert_eq!(openai_resp.choices[0].message.role, "assistant");
        assert_eq!(openai_resp.object, Some("chat.completion".to_string()));
        let usage = openai_resp.usage.expect("usage must be present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_build_inbound_response_id_prefix() {
        // id without prefix gets "chatcmpl-" prepended
        let resp = make_weft_response("abc-123", "Hi");
        let openai_resp = build_inbound_response(resp, "gpt-4".to_string());
        assert_eq!(openai_resp.id, Some("chatcmpl-abc-123".to_string()));
    }

    #[test]
    fn test_build_inbound_response_id_already_prefixed() {
        // id with "chatcmpl-" prefix is not double-prefixed
        let resp = make_weft_response("chatcmpl-already", "Hi");
        let openai_resp = build_inbound_response(resp, "gpt-4".to_string());
        assert_eq!(openai_resp.id, Some("chatcmpl-already".to_string()));
    }

    #[test]
    fn test_build_inbound_response_empty_messages_empty_content() {
        use weft_core::{WeftTiming, WeftUsage};
        let resp = weft_core::WeftResponse {
            id: "test".to_string(),
            model: "auto".to_string(),
            messages: vec![],
            usage: WeftUsage::default(),
            timing: WeftTiming::default(),
            degradations: vec![],
        };
        let openai_resp = build_inbound_response(resp, "auto".to_string());
        assert_eq!(openai_resp.choices[0].message.content, "");
    }

    #[test]
    fn test_build_inbound_response_model_echoed() {
        let resp = make_weft_response("id", "Hi");
        let openai_resp = build_inbound_response(resp, "gpt-4-turbo".to_string());
        assert_eq!(openai_resp.model, Some("gpt-4-turbo".to_string()));
    }

    // ── Round-trip ──────────────────────────────────────────────────────────

    #[test]
    fn test_outbound_request_round_trips_through_serde() {
        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "You are helpful."),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];
        let options = SamplingOptions {
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        };
        let req = build_outbound_request(&messages, "gpt-4", &options);
        let json = serde_json::to_string(&req).expect("serialize must succeed");
        let deserialized: OpenAIRequest =
            serde_json::from_str(&json).expect("deserialize must succeed");
        assert_eq!(deserialized.model, "gpt-4");
        assert_eq!(deserialized.messages.len(), 2);
        assert_eq!(deserialized.max_tokens, Some(100));
    }
}
