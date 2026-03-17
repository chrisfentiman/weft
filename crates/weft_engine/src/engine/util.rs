//! Utility types and functions shared across engine modules.
//!
//! Contains the `ActivityEvent` enum, domain conversion helpers, built-in
//! command constants and describe text, and the `extract_latest_user_text`
//! helper used by the request loop.

use weft_core::{
    ContentPart, HookRoutingDomain, Role, Source, WeftMessage,
};
use weft_router::RoutingDomainKind;

/// Built-in command names intercepted by the engine before the command registry.
pub(crate) const BUILTIN_COMMANDS: &[&str] = &["recall", "remember"];

// ── Activity events ───────────────────────────────────────────────────────────

/// An activity event collected during request processing.
///
/// When `options.activity = true`, these are converted to `source: gateway`
/// system messages and prepended to the response message list.
/// `Hook` and `CouncilStart` variants are wired in future phases (hook firing
/// and council mode respectively) — defined here so the types are ready.
#[derive(Debug, Clone)]
pub(crate) enum ActivityEvent {
    Routing(weft_core::RoutingActivity),
    #[allow(dead_code)]
    Hook(weft_core::HookActivity),
    #[allow(dead_code)]
    CouncilStart(weft_core::CouncilStartActivity),
}

impl ActivityEvent {
    /// Convert this activity event to a `WeftMessage` with `source: Gateway`.
    pub(crate) fn to_message(&self) -> WeftMessage {
        let content = match self {
            ActivityEvent::Routing(r) => ContentPart::Routing(r.clone()),
            ActivityEvent::Hook(h) => ContentPart::Hook(h.clone()),
            ActivityEvent::CouncilStart(c) => ContentPart::CouncilStart(c.clone()),
        };
        WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![content],
            delta: false,
            message_index: 0,
        }
    }
}

/// Convert from router's `RoutingDomainKind` to hooks' `HookRoutingDomain`.
///
/// Lives at the integration boundary — `weft_core` cannot depend on
/// `weft_router`, so the conversion happens here where both types are visible.
/// Using a free function avoids the orphan rule (both types are from external crates).
pub(crate) fn routing_domain_to_hook_domain(kind: &RoutingDomainKind) -> HookRoutingDomain {
    match kind {
        RoutingDomainKind::Model => HookRoutingDomain::Model,
        RoutingDomainKind::Commands => HookRoutingDomain::Commands,
        RoutingDomainKind::ToolNecessity => HookRoutingDomain::ToolNecessity,
        RoutingDomainKind::Memory => HookRoutingDomain::Memory,
    }
}

/// Compiled-in describe text for `/recall`.
const RECALL_DESCRIBE: &str = "\
recall: Retrieve relevant memories based on a query

Usage: /recall query: \"what do we know about user preferences\"

The gateway searches configured memory stores for content matching your query
and returns the most relevant results. If you omit the query, the current
conversation context is used automatically.";

/// Compiled-in describe text for `/remember`.
const REMEMBER_DESCRIBE: &str = "\
remember: Store information in memory for future recall

Usage: /remember content: \"the user prefers dark mode and compact layouts\"

Stores the given content in the most relevant memory store(s), selected
automatically based on the content. Retrieve stored memories later with
/recall. Use this when the user shares preferences, decisions, or important
context worth preserving.";

/// Return compiled-in describe text for a built-in command.
pub(crate) fn builtin_describe_text(name: &str) -> String {
    match name {
        "recall" => RECALL_DESCRIBE.to_string(),
        "remember" => REMEMBER_DESCRIBE.to_string(),
        _ => format!("{name}: unknown built-in command"),
    }
}

/// Extract the text of the last user message from the conversation.
///
/// Concatenates all Text content parts from the last user message.
/// Returns None if there is no user message or no text content in the last user message.
pub(crate) fn extract_latest_user_text(messages: &[WeftMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| {
            m.content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|s| !s.is_empty())
}
