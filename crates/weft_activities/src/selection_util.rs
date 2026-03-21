//! Shared utilities for semantic selection activities.
//!
//! Used by [`super::model_selection`] and [`super::command_selection`].

use weft_reactor_trait::ActivityInput;

/// Extract the text of the last user-role message. Returns `""` if none found.
pub(super) fn extract_user_message(input: &ActivityInput) -> &str {
    input
        .messages
        .iter()
        .rev()
        .find(|m| m.role == weft_core::Role::User)
        .and_then(|m| {
            m.content.iter().find_map(|p| {
                if let weft_core::ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
        })
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_test_input;
    use pretty_assertions::assert_eq;
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    #[test]
    fn extract_user_message_returns_last_user_text() {
        let input = make_test_input();
        let msg = extract_user_message(&input);
        assert_eq!(msg, "hello");
    }

    #[test]
    fn extract_user_message_returns_empty_when_no_user_message() {
        let mut input = make_test_input();
        input.messages.clear();
        let msg = extract_user_message(&input);
        assert_eq!(msg, "");
    }

    #[test]
    fn extract_user_message_picks_last_user_message() {
        let mut input = make_test_input();
        input.messages.push(WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("second user message".to_string())],
            delta: false,
            message_index: 1,
        });
        let msg = extract_user_message(&input);
        assert_eq!(msg, "second user message");
    }

    #[test]
    fn extract_user_message_skips_non_text_parts() {
        let mut input = make_test_input();
        input.messages.clear();
        input.messages.push(WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            // No text parts — only image or tool-result parts.
            content: vec![],
            delta: false,
            message_index: 0,
        });
        let msg = extract_user_message(&input);
        assert_eq!(msg, "");
    }
}
