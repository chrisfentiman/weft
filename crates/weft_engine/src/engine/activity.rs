//! Response assembly for the gateway engine.
//!
//! Provides `assemble_response`, which builds the final `WeftResponse` from
//! the LLM output, optional token usage, and optional activity events.

use weft_core::{ContentPart, Role, Source, WeftMessage, WeftResponse, WeftTiming, WeftUsage};
use weft_llm::TokenUsage;

use super::util::ActivityEvent;

/// Assemble the final `WeftResponse` from the LLM output and optional activity events.
///
/// When `include_activity` is `true`, gateway activity events (routing decisions, hook
/// events) are prepended to the message list as `source: Gateway` system messages.
/// The assistant response message always follows.
pub(crate) fn assemble_response(
    model_instruction: &str,
    llm_text: String,
    model_name: &str,
    usage: Option<TokenUsage>,
    activity_events: &[ActivityEvent],
    include_activity: bool,
) -> WeftResponse {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let (prompt_tokens, completion_tokens) = usage
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));

    let total_tokens = prompt_tokens + completion_tokens;

    let mut messages = Vec::new();

    // Prepend activity messages when requested.
    if include_activity {
        for event in activity_events {
            messages.push(event.to_message());
        }
    }

    messages.push(WeftMessage {
        role: Role::Assistant,
        source: Source::Provider,
        model: Some(model_name.to_string()),
        content: vec![ContentPart::Text(llm_text)],
        delta: false,
        message_index: 0,
    });

    WeftResponse {
        id,
        model: model_instruction.to_string(),
        messages,
        usage: WeftUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            llm_calls: 1,
        },
        timing: WeftTiming::default(),
    }
}
