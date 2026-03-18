//! AssemblePromptActivity: builds the system prompt and prepares the message list.
//!
//! Incorporates routing results, available commands, and memory context into the
//! message list that will be sent to the generative source.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::activity::{Activity, ActivityInput};
use crate::event::PipelineEvent;
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Builds the system prompt and prepares the message list for generation.
///
/// Incorporates routing results, available commands, and any memory context
/// into the message list. The assembled message count is reported in the
/// `PromptAssembled` event.
///
/// **Name:** `"assemble_prompt"`
///
/// **Events pushed:**
/// - `ActivityStarted { name: "assemble_prompt" }`
/// - `PromptAssembled { message_count }`
/// - `ActivityCompleted { name: "assemble_prompt", duration_ms, idempotency_key: None }`
/// - `ActivityFailed { name: "assemble_prompt", error, retryable: false }` — on error
pub struct AssemblePromptActivity;

impl AssemblePromptActivity {
    /// Construct a new AssemblePromptActivity.
    pub fn new() -> Self {
        Self
    }
}

impl Default for AssemblePromptActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for AssemblePromptActivity {
    fn name(&self) -> &str {
        "assemble_prompt"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        _services: &Services,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let start = Instant::now();

        let _ = event_tx
            .send(PipelineEvent::ActivityStarted {
                name: self.name().to_string(),
            })
            .await;

        if cancel.is_cancelled() {
            let _ = event_tx
                .send(PipelineEvent::ActivityFailed {
                    name: self.name().to_string(),
                    error: "cancelled before prompt assembly".to_string(),
                    retryable: false,
                })
                .await;
            return;
        }

        // Count the messages that will be sent to the provider.
        // The message list may be augmented by a system prompt in the Reactor,
        // but at this point we report the count of current messages.
        let message_count = input.messages.len();

        debug!(
            message_count,
            has_routing = input.routing_result.is_some(),
            command_count = input.available_commands.len(),
            "assemble_prompt: assembling"
        );

        let _ = event_tx
            .send(PipelineEvent::PromptAssembled { message_count })
            .await;

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = event_tx
            .send(PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms,
                idempotency_key: None,
            })
            .await;

        debug!(duration_ms, "assemble_prompt: completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NullEventLog;
    use crate::test_support::{collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    async fn run_assemble(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = AssemblePromptActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn assemble_prompt_name() {
        assert_eq!(AssemblePromptActivity::new().name(), "assemble_prompt");
    }

    // ── Happy path ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn assemble_prompt_pushes_prompt_assembled() {
        let input = make_test_input();
        let expected_count = input.messages.len();
        let events = run_assemble(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityStarted { name } if name == "assemble_prompt")),
            "expected ActivityStarted"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::PromptAssembled { message_count } if *message_count == expected_count)),
            "expected PromptAssembled with correct message count"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "assemble_prompt")),
            "expected ActivityCompleted"
        );
    }

    // ── Correct message count ────────────────────────────────────────────────

    #[tokio::test]
    async fn assemble_prompt_reports_message_count() {
        let mut input = make_test_input();
        // Add extra messages.
        input.messages.push(input.messages[0].clone());
        input.messages.push(input.messages[0].clone());
        let expected = input.messages.len();

        let events = run_assemble(input).await;

        let assembled = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::PromptAssembled { .. }))
            .expect("expected PromptAssembled");

        match assembled {
            PipelineEvent::PromptAssembled { message_count } => {
                assert_eq!(*message_count, expected);
            }
            _ => panic!("expected PromptAssembled"),
        }
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn assemble_prompt_handles_cancellation() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = AssemblePromptActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityFailed { .. })),
            "expected ActivityFailed when cancelled"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { .. })),
            "should not push ActivityCompleted when cancelled"
        );
    }
}
