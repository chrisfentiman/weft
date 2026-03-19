//! AssembleResponseActivity: constructs the final WeftResponse.
//!
//! Builds a WeftResponse from the accumulated execution state available in
//! `ActivityInput.accumulated_text`, routing information, and token usage
//! derived from the event log.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use weft_core::{ContentPart, Role, Source, WeftMessage, WeftResponse, WeftTiming};

use crate::activity::{Activity, ActivityInput};
use crate::event::PipelineEvent;
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Constructs the final WeftResponse from accumulated execution state.
///
/// Reads `accumulated_text` and routing info from the activity input,
/// builds a `WeftResponse`, and pushes it as a `ResponseAssembled` event.
///
/// **Name:** `"assemble_response"`
///
/// **Events pushed:**
/// - `ActivityStarted { name: "assemble_response" }`
/// - `ResponseAssembled { response }` — with the constructed WeftResponse
/// - `ActivityCompleted { name: "assemble_response", duration_ms, idempotency_key: None }`
/// - `ActivityFailed { name: "assemble_response", error, retryable: false }` — on error
pub struct AssembleResponseActivity;

impl AssembleResponseActivity {
    /// Construct a new AssembleResponseActivity.
    pub fn new() -> Self {
        Self
    }
}

impl Default for AssembleResponseActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for AssembleResponseActivity {
    fn name(&self) -> &str {
        "assemble_response"
    }

    async fn execute(
        &self,
        execution_id: &ExecutionId,
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
                    error: "cancelled before response assembly".to_string(),
                    retryable: false,
                })
                .await;
            return;
        }

        // Determine model from routing result.
        let model = input
            .routing_result
            .as_ref()
            .map(|r| r.model_routing.model.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Build the response message from accumulated text.
        let response_message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some(model.clone()),
            content: vec![ContentPart::Text(input.accumulated_text.clone())],
            delta: false,
            message_index: 0,
        };

        // Build the WeftResponse.
        let response = WeftResponse {
            id: execution_id.to_string(),
            model: model.clone(),
            messages: vec![response_message],
            usage: input.accumulated_usage.clone(),
            timing: WeftTiming::default(),
        };

        debug!(
            model = %model,
            text_len = input.accumulated_text.len(),
            "assemble_response: built response"
        );

        let _ = event_tx
            .send(PipelineEvent::ResponseAssembled { response })
            .await;

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = event_tx
            .send(PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms,
                idempotency_key: None,
            })
            .await;

        debug!(duration_ms, "assemble_response: completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NullEventLog;
    use crate::test_support::{collect_events, make_test_input, make_test_services};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    async fn run_assemble_response(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = AssembleResponseActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn assemble_response_name() {
        assert_eq!(AssembleResponseActivity::new().name(), "assemble_response");
    }

    // ── Happy path ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn assemble_response_pushes_response_assembled() {
        let mut input = make_test_input();
        input.accumulated_text = "The answer is 42.".to_string();

        let events = run_assemble_response(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityStarted { name } if name == "assemble_response")),
            "expected ActivityStarted"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ResponseAssembled { .. })),
            "expected ResponseAssembled"
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "assemble_response")),
            "expected ActivityCompleted"
        );
    }

    // ── Response contains accumulated text ───────────────────────────────

    #[tokio::test]
    async fn assemble_response_includes_accumulated_text() {
        let mut input = make_test_input();
        input.accumulated_text = "This is the final response.".to_string();

        let events = run_assemble_response(input).await;

        let assembled = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ResponseAssembled { .. }))
            .expect("expected ResponseAssembled");

        match assembled {
            PipelineEvent::ResponseAssembled { response } => {
                // The response messages should contain the accumulated text.
                let has_text = response.messages.iter().any(|m| {
                    m.content.iter().any(|p| {
                        matches!(p, ContentPart::Text(t) if t.contains("This is the final response."))
                    })
                });
                assert!(has_text, "response should contain accumulated text");
            }
            _ => panic!("expected ResponseAssembled"),
        }
    }

    // ── WeftResponse includes execution ID ───────────────────────────────

    #[tokio::test]
    async fn assemble_response_id_matches_execution_id() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();
        let exec_id_str = exec_id.to_string();

        let activity = AssembleResponseActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        let assembled = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ResponseAssembled { .. }))
            .expect("expected ResponseAssembled");

        match assembled {
            PipelineEvent::ResponseAssembled { response } => {
                assert_eq!(response.id, exec_id_str);
            }
            _ => panic!("expected ResponseAssembled"),
        }
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn assemble_response_handles_cancellation() {
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = AssembleResponseActivity::new();
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
