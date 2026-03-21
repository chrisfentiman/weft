//! SystemPromptAssembly activity: layer the system prompt from gateway config and caller request.
//!
//! Assembles the final system prompt by combining:
//! - Layer 1: gateway foundational prompt from `services.config().gateway.system_prompt`
//! - Layer 2: caller-provided system prompt, if `input.messages[0]` has `Role::System`
//!
//! Agent instructions (a future layer 2) are deferred until agent config exists in the schema.
//!
//! The assembled prompt is pushed as a `Context(MessageInjected)` with source `SystemPromptAssembly`.
//! The reactor inserts this message at `messages[0]`, replacing any existing system-role message.
//!
//! **Fail mode: OPEN.** If assembly encounters an error (should not happen with current inputs),
//! the activity completes with an empty prompt. The model can still generate without a system prompt.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use weft_reactor_trait::ServiceLocator;

use crate::activity::{Activity, ActivityInput};
use crate::event::{ActivityEvent, ContextEvent, MessageInjectionSource, PipelineEvent};
use crate::event_log::EventLog;
use crate::execution::ExecutionId;

/// Assembles the system prompt from gateway config and caller-provided layers.
///
/// **Name:** `"system_prompt_assembly"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "system_prompt_assembly" })`
/// - `Context(ContextEvent::MessageInjected { message, source: SystemPromptAssembly })` — the assembled system prompt
/// - `Context(ContextEvent::SystemPromptAssembled { message_count })`
/// - `Activity(ActivityEvent::Completed { name: "system_prompt_assembly", idempotency_key: None })`
pub struct SystemPromptAssemblyActivity;

impl SystemPromptAssemblyActivity {
    /// Construct a new `SystemPromptAssemblyActivity`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SystemPromptAssemblyActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for SystemPromptAssemblyActivity {
    fn name(&self) -> &str {
        "system_prompt_assembly"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        services: &dyn ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        _cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;

        // Layer 1: gateway foundational prompt.
        let gateway_prompt = services.config().gateway.system_prompt.clone();

        // Layer 2: caller-provided system prompt.
        // Present when messages[0] has Role::System and Source::Client.
        let caller_prompt = input
            .messages
            .first()
            .filter(|m| m.role == weft_core::Role::System && m.source == weft_core::Source::Client)
            .and_then(|m| {
                // Extract the text content from the caller's system message.
                m.content.iter().find_map(|part| {
                    if let weft_core::ContentPart::Text(text) = part {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            });

        // Assemble layers. Only non-empty layers contribute to the count and the text.
        let mut layers: Vec<String> = Vec::new();
        if !gateway_prompt.is_empty() {
            layers.push(gateway_prompt);
        }
        if let Some(caller) = caller_prompt
            && !caller.is_empty()
        {
            layers.push(caller);
        }

        let assembled_prompt = layers.join("\n\n");

        // Build the WeftMessage to inject at messages[0].
        let injected_message = weft_core::WeftMessage {
            role: weft_core::Role::System,
            source: weft_core::Source::Gateway,
            model: None,
            content: vec![weft_core::ContentPart::Text(assembled_prompt)],
            delta: false,
            message_index: 0,
        };

        // MessageInjected carries the assembled prompt. The reactor inserts it at messages[0].
        let _ = event_tx
            .send(PipelineEvent::Context(ContextEvent::MessageInjected {
                message: injected_message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }))
            .await;

        // The message_count reflects the state AFTER the system prompt is inserted.
        // The reactor's drain will either replace messages[0] (if it was already System)
        // or insert at position 0. Either way the count is input.messages.len() if the
        // first message was already System, or input.messages.len() + 1 if not.
        // We report the count as it will be after the injection.
        let message_count = if input
            .messages
            .first()
            .map(|m| m.role == weft_core::Role::System)
            .unwrap_or(false)
        {
            // First message is System: it will be replaced, so count stays the same.
            input.messages.len()
        } else {
            // No existing System message: insertion adds one.
            input.messages.len() + 1
        };

        let _ = event_tx
            .send(PipelineEvent::Context(
                ContextEvent::SystemPromptAssembled { message_count },
            ))
            .await;

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: None,
            }))
            .await;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{NullEventLog, collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    /// Run the activity with the given input and return all emitted events.
    async fn run_assembly(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = SystemPromptAssemblyActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn system_prompt_assembly_name() {
        assert_eq!(
            SystemPromptAssemblyActivity::new().name(),
            "system_prompt_assembly"
        );
    }

    // ── Happy path: gateway prompt only ─────────────────────────────────────

    #[tokio::test]
    async fn gateway_only_emits_message_injected_and_assembled() {
        // make_test_input has one User message, no System message.
        let input = make_test_input();
        let events = run_assembly(input).await;

        // Activity(Started) must be first.
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "system_prompt_assembly")
            ),
            "expected Activity(Started)"
        );

        // A Context(MessageInjected) event must be present with SystemPromptAssembly source.
        let injected = events.iter().find(|e| {
            matches!(
                e,
                PipelineEvent::Context(ContextEvent::MessageInjected {
                    source: MessageInjectionSource::SystemPromptAssembly,
                    ..
                })
            )
        });
        assert!(
            injected.is_some(),
            "expected Context(MessageInjected) with SystemPromptAssembly source"
        );

        // Context(SystemPromptAssembled) must follow.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Context(ContextEvent::SystemPromptAssembled { .. })
            )),
            "expected Context(SystemPromptAssembled)"
        );

        // Activity(Completed) must be last.
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "system_prompt_assembly")
            ),
            "expected Activity(Completed)"
        );

        // No Activity(Failed).
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "did not expect Activity(Failed)"
        );
    }

    // ── Gateway prompt content is assembled into the injected message ────────

    #[tokio::test]
    async fn gateway_prompt_in_injected_message_content() {
        let input = make_test_input();
        // make_test_services uses config with gateway.system_prompt = "You are helpful."
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = SystemPromptAssemblyActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;
        let events = collect_events(&mut rx);

        let injected_text = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }) = e
                && let Some(ContentPart::Text(text)) = message.content.first()
            {
                return Some(text.clone());
            }
            None
        });

        let text = injected_text.expect("Context(MessageInjected) must carry text content");
        // The test config uses "You are helpful." as the gateway system prompt.
        assert!(
            text.contains("You are helpful"),
            "expected gateway prompt in assembled text, got: {text}"
        );
    }

    // ── Gateway + caller system prompt combined ──────────────────────────────

    #[tokio::test]
    async fn gateway_and_caller_system_prompt_combined() {
        let mut input = make_test_input();
        // Prepend a caller system message.
        input.messages.insert(
            0,
            WeftMessage {
                role: Role::System,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("Always speak in rhymes.".to_string())],
                delta: false,
                message_index: 0,
            },
        );

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = SystemPromptAssemblyActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;
        let events = collect_events(&mut rx);

        let injected_text = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }) = e
                && let Some(ContentPart::Text(text)) = message.content.first()
            {
                return Some(text.clone());
            }
            None
        });

        let text = injected_text.expect("Context(MessageInjected) must be present");
        assert!(
            text.contains("You are helpful"),
            "expected gateway prompt in assembled text"
        );
        assert!(
            text.contains("Always speak in rhymes"),
            "expected caller prompt in assembled text"
        );

        // SystemPromptAssembled event must be present.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Context(ContextEvent::SystemPromptAssembled { .. })
            )),
            "expected Context(SystemPromptAssembled)"
        );
    }

    // ── Empty gateway prompt ─────────────────────────────────────────────────

    #[tokio::test]
    async fn empty_gateway_prompt_assembles_empty_prompt() {
        use std::sync::Arc;

        let input = make_test_input();

        // Build services with an empty gateway system prompt.
        let config: weft_core::WeftConfig = toml::from_str(
            r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = ""

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "stub-model"
  model = "stub-model-v1"
  examples = ["example query"]
"#,
        )
        .expect("test config must parse");

        let services = crate::services::Services {
            config: Arc::new(config),
            ..make_test_services()
        };

        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = SystemPromptAssemblyActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;
        let events = collect_events(&mut rx);

        // SystemPromptAssembled event must be present.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Context(ContextEvent::SystemPromptAssembled { .. })
            )),
            "expected Context(SystemPromptAssembled)"
        );

        // Injected message content must be empty when gateway prompt is empty.
        let injected_text = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }) = e
                && let Some(ContentPart::Text(text)) = message.content.first()
            {
                return Some(text.clone());
            }
            None
        });
        let text = injected_text.expect("Context(MessageInjected) must be present");
        assert_eq!(text, "", "empty gateway prompt → injected content is empty");
    }

    // ── MessageInjected source is SystemPromptAssembly ────────────────────────

    #[tokio::test]
    async fn message_injected_source_is_system_prompt_assembly() {
        let input = make_test_input();
        let events = run_assembly(input).await;

        let found = events.iter().any(|e| {
            matches!(
                e,
                PipelineEvent::Context(ContextEvent::MessageInjected {
                    source: MessageInjectionSource::SystemPromptAssembly,
                    ..
                })
            )
        });
        assert!(found, "source must be SystemPromptAssembly");
    }

    // ── Injected message has Role::System and Source::Gateway ────────────────

    #[tokio::test]
    async fn injected_message_role_system_source_gateway() {
        let input = make_test_input();
        let events = run_assembly(input).await;

        let msg = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }) = e
            {
                Some(message.clone())
            } else {
                None
            }
        });

        let msg = msg.expect("Context(MessageInjected) must be present");
        assert_eq!(
            msg.role,
            Role::System,
            "injected message must have Role::System"
        );
        assert_eq!(
            msg.source,
            Source::Gateway,
            "injected message must have Source::Gateway"
        );
        assert!(!msg.delta, "injected message must not be a delta");
        assert_eq!(
            msg.message_index, 0,
            "injected message must have message_index 0"
        );
    }

    // ── message_count after injection ────────────────────────────────────────

    #[tokio::test]
    async fn message_count_increments_when_no_existing_system_message() {
        // make_test_input has 1 User message. After injection, count should be 2.
        let input = make_test_input();
        assert_eq!(input.messages.len(), 1, "precondition: 1 message");
        assert_eq!(
            input.messages[0].role,
            Role::User,
            "precondition: first message is User"
        );

        let events = run_assembly(input).await;

        let count = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SystemPromptAssembled {
                message_count,
                ..
            }) = e
            {
                Some(*message_count)
            } else {
                None
            }
        });

        assert_eq!(
            count.expect("Context(SystemPromptAssembled) must be present"),
            2,
            "inserting a system message at position 0 should make message_count = 2"
        );
    }

    #[tokio::test]
    async fn message_count_unchanged_when_existing_system_message_replaced() {
        let mut input = make_test_input();
        // Insert a System message at position 0 (replacing nothing — it's the first message).
        input.messages.insert(
            0,
            WeftMessage {
                role: Role::System,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("caller system".to_string())],
                delta: false,
                message_index: 0,
            },
        );
        // Now we have 2 messages: [System, User].
        assert_eq!(input.messages.len(), 2, "precondition: 2 messages");
        assert_eq!(
            input.messages[0].role,
            Role::System,
            "precondition: first message is System"
        );

        let events = run_assembly(input).await;

        let count = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SystemPromptAssembled {
                message_count,
                ..
            }) = e
            {
                Some(*message_count)
            } else {
                None
            }
        });

        // The system message at [0] will be replaced (not inserted), so count stays at 2.
        assert_eq!(
            count.expect("Context(SystemPromptAssembled) must be present"),
            2,
            "replacing existing System message should keep message_count = 2"
        );
    }

    // ── Caller system prompt not included when Source is not Client ──────────

    #[tokio::test]
    async fn non_client_system_message_not_treated_as_caller_prompt() {
        let mut input = make_test_input();
        // Insert a System message from Source::Gateway (not Client) — should not be treated as layer 2.
        input.messages.insert(
            0,
            WeftMessage {
                role: Role::System,
                source: Source::Gateway,
                model: None,
                content: vec![ContentPart::Text("gateway injected".to_string())],
                delta: false,
                message_index: 0,
            },
        );

        let events = run_assembly(input).await;

        // Only the gateway config prompt contributes. The Source::Gateway message at [0]
        // is NOT treated as a caller layer — injected text must not include "gateway injected".
        let injected_text = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::MessageInjected {
                message,
                source: MessageInjectionSource::SystemPromptAssembly,
            }) = e
                && let Some(ContentPart::Text(text)) = message.content.first()
            {
                return Some(text.clone());
            }
            None
        });
        let text = injected_text.expect("Context(MessageInjected) must be present");
        assert!(
            !text.contains("gateway injected"),
            "Source::Gateway system message must not be included as a caller layer"
        );
        assert!(
            text.contains("You are helpful"),
            "gateway config prompt must be included"
        );
    }
}
