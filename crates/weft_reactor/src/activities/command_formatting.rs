//! CommandFormatting activity: format selected commands based on provider capabilities.
//!
//! Reads capabilities and selected command names from `input.metadata` (set by the reactor
//! from `ExecutionState`). Determines how commands should be presented to the provider:
//!
//! - `CommandFormat::NoCommands` — no commands were selected
//! - `CommandFormat::Structured` — provider supports `tool_calling`; commands are passed
//!   as native structured tool definitions (the provider adapter handles marshalling)
//! - `CommandFormat::PromptInjected` — provider does not support `tool_calling`; commands
//!   are formatted as a text block and injected into the conversation via `MessageInjected`
//!
//! **Fail mode: OPEN.** On any error (missing metadata, empty commands), the activity completes
//! with `CommandFormat::NoCommands`. The model can still generate text.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::activity::{Activity, ActivityInput};
use crate::event::{
    ActivityEvent, CommandFormat, ContextEvent, MessageInjectionSource, PipelineEvent,
};
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use weft_reactor_trait::ServiceLocator;

/// Formats selected commands for the target provider based on its capabilities.
///
/// **Name:** `"command_formatting"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "command_formatting" })`
/// - `Context(ContextEvent::MessageInjected { message, source: CommandFormatInjection })` — only for `PromptInjected` format
/// - `Context(ContextEvent::CommandsFormatted { format, command_count })`
/// - `Activity(ActivityEvent::Completed { name: "command_formatting", idempotency_key: None })`
pub struct CommandFormattingActivity;

impl CommandFormattingActivity {
    /// Construct a new `CommandFormattingActivity`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CommandFormattingActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for CommandFormattingActivity {
    fn name(&self) -> &str {
        "command_formatting"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &dyn ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;

        // Extract capabilities from metadata. Missing metadata defaults to PromptInjected
        // (conservative — assume no tool_calling support rather than breaking generation).
        let capabilities: Vec<String> = input
            .metadata
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Extract selected command names from metadata.
        let selected_names: Vec<String> = input
            .metadata
            .get("selected_commands")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Filter available_commands to only those that were selected.
        // If a selected name doesn't match any available command, skip it silently.
        let selected_commands: Vec<&weft_core::CommandStub> = input
            .available_commands
            .iter()
            .filter(|cmd| selected_names.contains(&cmd.name))
            .collect();

        if selected_commands.is_empty() {
            // No commands selected — nothing to format.
            let _ = event_tx
                .send(PipelineEvent::Context(ContextEvent::CommandsFormatted {
                    format: CommandFormat::NoCommands,
                    command_count: 0,
                }))
                .await;

            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Completed {
                    name: self.name().to_string(),
                    idempotency_key: None,
                }))
                .await;
            return;
        }

        let command_count = selected_commands.len();
        let has_tool_calling = capabilities.iter().any(|c| c == "tool_calling");

        if has_tool_calling {
            // Structured: provider handles tool definitions natively.
            let _ = event_tx
                .send(PipelineEvent::Context(ContextEvent::CommandsFormatted {
                    format: CommandFormat::Structured,
                    command_count,
                }))
                .await;
        } else {
            // PromptInjected: build a text block describing available commands and inject it.
            let command_text = build_command_text(&selected_commands);

            let injected_message = weft_core::WeftMessage {
                role: weft_core::Role::System,
                source: weft_core::Source::Gateway,
                model: None,
                content: vec![weft_core::ContentPart::Text(command_text)],
                delta: false,
                message_index: 0,
            };

            let _ = event_tx
                .send(PipelineEvent::Context(ContextEvent::MessageInjected {
                    message: injected_message,
                    source: MessageInjectionSource::CommandFormatInjection,
                }))
                .await;

            let _ = event_tx
                .send(PipelineEvent::Context(ContextEvent::CommandsFormatted {
                    format: CommandFormat::PromptInjected,
                    command_count,
                }))
                .await;
        }

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a human-readable text block describing the available commands.
///
/// Format:
/// ```text
/// You have access to the following commands:
///
/// /command_name: Command description.
/// /other_command: Other description.
/// ```
fn build_command_text(commands: &[&weft_core::CommandStub]) -> String {
    let mut lines = vec![
        "You have access to the following commands:".to_string(),
        String::new(),
    ];
    for cmd in commands {
        lines.push(format!("/{}: {}", cmd.name, cmd.description));
    }
    lines.join("\n")
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{NullEventLog, collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_core::CommandStub;

    /// Run the activity with the given input and return all emitted events.
    async fn run_formatting(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = CommandFormattingActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    /// Build an input with specified capabilities, selected command names, and available commands.
    fn make_input_with_commands(
        capabilities: &[&str],
        selected_names: &[&str],
        available: Vec<CommandStub>,
    ) -> ActivityInput {
        let mut input = make_test_input();
        input.available_commands = available;
        input.metadata = serde_json::json!({
            "capabilities": capabilities,
            "selected_commands": selected_names,
        });
        input
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn command_formatting_name() {
        assert_eq!(
            CommandFormattingActivity::new().name(),
            "command_formatting"
        );
    }

    // ── No commands: NoCommands format ───────────────────────────────────────

    #[tokio::test]
    async fn no_selected_commands_emits_no_commands_format() {
        let input = make_input_with_commands(&["chat_completions"], &[], vec![]);
        let events = run_formatting(input).await;

        let formatted = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::CommandsFormatted {
                format,
                command_count,
            }) = e
            {
                Some((format.clone(), *command_count))
            } else {
                None
            }
        });

        let (format, count) = formatted.expect("CommandsFormatted must be present");
        assert_eq!(
            format,
            CommandFormat::NoCommands,
            "no commands → NoCommands"
        );
        assert_eq!(count, 0, "command_count must be 0");
    }

    #[tokio::test]
    async fn selected_commands_not_in_available_produces_no_commands() {
        // Selected names reference commands that don't exist in available_commands.
        let input = make_input_with_commands(
            &["tool_calling"],
            &["nonexistent_cmd"],
            vec![], // no available commands
        );
        let events = run_formatting(input).await;

        let format = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::CommandsFormatted { format, .. }) = e {
                Some(format.clone())
            } else {
                None
            }
        });

        assert_eq!(
            format.expect("CommandsFormatted must be present"),
            CommandFormat::NoCommands,
            "selected names with no matching available commands → NoCommands"
        );
    }

    // ── Structured format when tool_calling capability present ───────────────

    #[tokio::test]
    async fn tool_calling_capability_produces_structured_format() {
        let cmds = vec![CommandStub {
            name: "search".to_string(),
            description: "Search the web".to_string(),
        }];
        let input =
            make_input_with_commands(&["chat_completions", "tool_calling"], &["search"], cmds);
        let events = run_formatting(input).await;

        let formatted = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::CommandsFormatted {
                format,
                command_count,
            }) = e
            {
                Some((format.clone(), *command_count))
            } else {
                None
            }
        });

        let (format, count) = formatted.expect("CommandsFormatted must be present");
        assert_eq!(
            format,
            CommandFormat::Structured,
            "tool_calling → Structured"
        );
        assert_eq!(count, 1, "command_count must match selected commands");
    }

    #[tokio::test]
    async fn structured_format_does_not_emit_message_injected() {
        let cmds = vec![CommandStub {
            name: "calc".to_string(),
            description: "Calculate math expressions".to_string(),
        }];
        let input = make_input_with_commands(&["tool_calling"], &["calc"], cmds);
        let events = run_formatting(input).await;

        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Context(ContextEvent::MessageInjected {
                    source: MessageInjectionSource::CommandFormatInjection,
                    ..
                })
            )),
            "Structured format must NOT emit MessageInjected"
        );
    }

    // ── PromptInjected format when tool_calling not present ──────────────────

    #[tokio::test]
    async fn no_tool_calling_produces_prompt_injected_format() {
        let cmds = vec![CommandStub {
            name: "search".to_string(),
            description: "Search the web".to_string(),
        }];
        let input = make_input_with_commands(&["chat_completions"], &["search"], cmds);
        let events = run_formatting(input).await;

        let format = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::CommandsFormatted { format, .. }) = e {
                Some(format.clone())
            } else {
                None
            }
        });

        assert_eq!(
            format.expect("CommandsFormatted must be present"),
            CommandFormat::PromptInjected,
            "no tool_calling → PromptInjected"
        );
    }

    // ── MessageInjected with CommandFormatInjection source for prompt-injected ──

    #[tokio::test]
    async fn prompt_injected_emits_message_injected_with_correct_source() {
        let cmds = vec![CommandStub {
            name: "search".to_string(),
            description: "Search the web".to_string(),
        }];
        let input = make_input_with_commands(&["chat_completions"], &["search"], cmds);
        let events = run_formatting(input).await;

        let found = events.iter().any(|e| {
            matches!(
                e,
                PipelineEvent::Context(ContextEvent::MessageInjected {
                    source: MessageInjectionSource::CommandFormatInjection,
                    ..
                })
            )
        });
        assert!(
            found,
            "PromptInjected must emit MessageInjected with CommandFormatInjection source"
        );
    }

    #[tokio::test]
    async fn prompt_injected_message_has_system_role_and_gateway_source() {
        let cmds = vec![CommandStub {
            name: "calc".to_string(),
            description: "Do math".to_string(),
        }];
        let input = make_input_with_commands(&[], &["calc"], cmds);
        let events = run_formatting(input).await;

        let msg = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::CommandFormatInjection,
            }) = e
            {
                Some(message.clone())
            } else {
                None
            }
        });

        let msg = msg.expect("MessageInjected must be present");
        assert_eq!(
            msg.role,
            weft_core::Role::System,
            "injected message must be Role::System"
        );
        assert_eq!(
            msg.source,
            weft_core::Source::Gateway,
            "injected message must be Source::Gateway"
        );
    }

    #[tokio::test]
    async fn prompt_injected_message_contains_command_names_and_descriptions() {
        let cmds = vec![
            CommandStub {
                name: "search".to_string(),
                description: "Search the web".to_string(),
            },
            CommandStub {
                name: "calculator".to_string(),
                description: "Evaluate math expressions".to_string(),
            },
        ];
        let input = make_input_with_commands(&[], &["search", "calculator"], cmds);
        let events = run_formatting(input).await;

        let text = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::CommandFormatInjection,
            }) = e
                && let Some(weft_core::ContentPart::Text(t)) = message.content.first()
            {
                return Some(t.clone());
            }
            None
        });

        let text = text.expect("MessageInjected must carry text content");
        assert!(text.contains("search"), "text must mention search command");
        assert!(
            text.contains("Search the web"),
            "text must include search description"
        );
        assert!(
            text.contains("calculator"),
            "text must mention calculator command"
        );
        assert!(
            text.contains("Evaluate math expressions"),
            "text must include calculator description"
        );
    }

    // ── Missing capabilities metadata defaults to PromptInjected ─────────────

    #[tokio::test]
    async fn missing_capabilities_defaults_to_prompt_injected() {
        let cmds = vec![CommandStub {
            name: "search".to_string(),
            description: "Search the web".to_string(),
        }];
        let mut input = make_test_input();
        input.available_commands = cmds;
        // Metadata has selected_commands but no capabilities key.
        input.metadata = serde_json::json!({ "selected_commands": ["search"] });

        let events = run_formatting(input).await;

        let format = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::CommandsFormatted { format, .. }) = e {
                Some(format.clone())
            } else {
                None
            }
        });

        assert_eq!(
            format.expect("CommandsFormatted must be present"),
            CommandFormat::PromptInjected,
            "missing capabilities → conservative PromptInjected"
        );
    }

    // ── Lifecycle events always present ─────────────────────────────────────

    #[tokio::test]
    async fn lifecycle_events_always_emitted() {
        let input = make_input_with_commands(&["chat_completions"], &[], vec![]);
        let events = run_formatting(input).await;

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "command_formatting")
            ),
            "expected ActivityStarted"
        );
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "command_formatting")
            ),
            "expected ActivityCompleted"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "did not expect ActivityFailed"
        );
    }

    // ── build_command_text helper ────────────────────────────────────────────

    #[test]
    fn build_command_text_formats_commands() {
        let cmds = [
            CommandStub {
                name: "search".to_string(),
                description: "Search the web".to_string(),
            },
            CommandStub {
                name: "calc".to_string(),
                description: "Math evaluator".to_string(),
            },
        ];
        let refs: Vec<&CommandStub> = cmds.iter().collect();
        let text = build_command_text(&refs);
        assert!(text.contains("/search:"), "must contain /search:");
        assert!(text.contains("/calc:"), "must contain /calc:");
        assert!(
            text.contains("Search the web"),
            "must contain search description"
        );
        assert!(
            text.contains("Math evaluator"),
            "must contain calc description"
        );
    }
}
