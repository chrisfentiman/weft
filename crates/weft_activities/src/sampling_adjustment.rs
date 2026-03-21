//! SamplingAdjustment activity: clamp and propagate sampling parameters.
//!
//! Reads the model's `max_tokens` from `input.metadata` (set by the reactor from
//! `ExecutionState.model_max_tokens`). Reads `max_tokens`, `temperature`, and `top_p`
//! from the original request's `options`.
//!
//! Clamping rule:
//! `final_max_tokens = min(request_max_tokens.unwrap_or(model_max), model_max)`
//!
//! Temperature and top_p pass through unchanged. No model-level constraints exist
//! for these in the current config schema.
//!
//! **Fail mode: OPEN.** If metadata is missing, defaults to 4096 for max_tokens.
//! The model can still generate with default sampling parameters.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use weft_reactor_trait::{
    Activity, ActivityEvent, ActivityInput, ContextEvent, EventLog, ExecutionId, PipelineEvent,
    ServiceLocator,
};

/// The default model max_tokens when the metadata value is missing.
const DEFAULT_MODEL_MAX_TOKENS: u32 = 4096;

/// Adjusts sampling parameters based on model constraints.
///
/// **Name:** `"sampling_adjustment"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "sampling_adjustment" })`
/// - `Context(ContextEvent::SamplingUpdated { max_tokens, temperature, top_p })`
/// - `Activity(ActivityEvent::Completed { name: "sampling_adjustment", idempotency_key: None })`
pub struct SamplingAdjustmentActivity;

impl SamplingAdjustmentActivity {
    /// Construct a new `SamplingAdjustmentActivity`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SamplingAdjustmentActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for SamplingAdjustmentActivity {
    fn name(&self) -> &str {
        "sampling_adjustment"
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

        // Read model max_tokens from metadata. Default to 4096 when absent.
        let model_max_tokens: u32 = input
            .metadata
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(DEFAULT_MODEL_MAX_TOKENS);

        // Read request sampling options.
        let request_max_tokens = input.request.options.max_tokens;
        let temperature = input.request.options.temperature;
        let top_p = input.request.options.top_p;

        // Clamp max_tokens: use the smaller of what was requested and what the model supports.
        // If the request didn't specify max_tokens, use the model's limit.
        let final_max_tokens = match request_max_tokens {
            Some(requested) => requested.min(model_max_tokens),
            None => model_max_tokens,
        };

        let _ = event_tx
            .send(PipelineEvent::Context(ContextEvent::SamplingUpdated {
                max_tokens: final_max_tokens,
                temperature,
                top_p,
            }))
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
    use weft_core::SamplingOptions;

    /// Run the activity with the given input and return all emitted events.
    async fn run_sampling(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = SamplingAdjustmentActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    /// Build an input with specific model max_tokens and request options.
    fn make_input_with_sampling(
        model_max_tokens: Option<u64>,
        request_max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> ActivityInput {
        let mut input = make_test_input();
        input.metadata = match model_max_tokens {
            Some(t) => serde_json::json!({ "max_tokens": t }),
            None => serde_json::Value::Null,
        };
        input.request.options = SamplingOptions {
            max_tokens: request_max_tokens,
            temperature,
            top_p,
            ..SamplingOptions::default()
        };
        input
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn sampling_adjustment_name() {
        assert_eq!(
            SamplingAdjustmentActivity::new().name(),
            "sampling_adjustment"
        );
    }

    // ── Clamp to model limit ─────────────────────────────────────────────────

    #[tokio::test]
    async fn request_max_tokens_clamped_to_model_limit() {
        // Request asks for more than model supports.
        let input = make_input_with_sampling(Some(4096), Some(8192), None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            4096,
            "request max_tokens (8192) must be clamped to model limit (4096)"
        );
    }

    #[tokio::test]
    async fn request_max_tokens_below_model_limit_passes_through() {
        // Request asks for less than model supports — no clamping needed.
        let input = make_input_with_sampling(Some(4096), Some(1024), None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            1024,
            "request max_tokens (1024) is below model limit (4096), should pass through"
        );
    }

    // ── No request max_tokens defaults to model limit ────────────────────────

    #[tokio::test]
    async fn no_request_max_tokens_defaults_to_model_limit() {
        let input = make_input_with_sampling(Some(8192), None, None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            8192,
            "no request max_tokens → use model limit (8192)"
        );
    }

    // ── Missing metadata defaults to 4096 ────────────────────────────────────

    #[tokio::test]
    async fn missing_metadata_defaults_to_4096() {
        let input = make_input_with_sampling(None, None, None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            4096,
            "missing metadata max_tokens must default to 4096"
        );
    }

    #[tokio::test]
    async fn missing_metadata_with_request_max_tokens_clamps_against_4096() {
        // Metadata missing but request specifies max_tokens. Default model limit is 4096.
        let input = make_input_with_sampling(None, Some(2048), None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            2048,
            "request max_tokens (2048) is below default limit (4096), should pass through"
        );
    }

    #[tokio::test]
    async fn missing_metadata_with_large_request_max_tokens_clamped_to_4096() {
        // Request specifies more than the default 4096.
        let input = make_input_with_sampling(None, Some(16384), None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            4096,
            "request max_tokens (16384) must be clamped to default limit (4096)"
        );
    }

    // ── Temperature passed through unchanged ─────────────────────────────────

    #[tokio::test]
    async fn temperature_passed_through_unchanged() {
        let input = make_input_with_sampling(Some(4096), None, Some(0.7), None);
        let events = run_sampling(input).await;

        let temp = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { temperature, .. }) = e {
                Some(*temperature)
            } else {
                None
            }
        });

        let temp = temp.expect("SamplingUpdated must be present");
        assert!(temp.is_some(), "temperature must be Some when specified");
        assert!(
            (temp.unwrap() - 0.7_f32).abs() < 1e-5,
            "temperature must pass through unchanged"
        );
    }

    #[tokio::test]
    async fn no_temperature_passes_as_none() {
        let input = make_input_with_sampling(Some(4096), None, None, None);
        let events = run_sampling(input).await;

        let temp = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { temperature, .. }) = e {
                Some(*temperature)
            } else {
                None
            }
        });

        let temp = temp.expect("SamplingUpdated must be present");
        assert!(temp.is_none(), "no temperature specified → must be None");
    }

    // ── Top_p passed through unchanged ───────────────────────────────────────

    #[tokio::test]
    async fn top_p_passed_through_unchanged() {
        let input = make_input_with_sampling(Some(4096), None, None, Some(0.95));
        let events = run_sampling(input).await;

        let tp = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { top_p, .. }) = e {
                Some(*top_p)
            } else {
                None
            }
        });

        let tp = tp.expect("SamplingUpdated must be present");
        assert!(tp.is_some(), "top_p must be Some when specified");
        assert!(
            (tp.unwrap() - 0.95_f32).abs() < 1e-5,
            "top_p must pass through unchanged"
        );
    }

    #[tokio::test]
    async fn no_top_p_passes_as_none() {
        let input = make_input_with_sampling(Some(4096), None, None, None);
        let events = run_sampling(input).await;

        let tp = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { top_p, .. }) = e {
                Some(*top_p)
            } else {
                None
            }
        });

        let tp = tp.expect("SamplingUpdated must be present");
        assert!(tp.is_none(), "no top_p specified → must be None");
    }

    // ── Event records current values, not originals ──────────────────────────

    #[tokio::test]
    async fn sampling_updated_records_clamped_value_not_original() {
        // Request asked for 16384, model max is 4096. Event must record 4096.
        let input = make_input_with_sampling(Some(4096), Some(16384), Some(0.5), Some(0.8));
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated {
                max_tokens,
                temperature,
                top_p,
            }) = e
            {
                Some((*max_tokens, *temperature, *top_p))
            } else {
                None
            }
        });

        let (max_tokens, temperature, top_p) = updated.expect("SamplingUpdated must be present");
        // The event carries the final (clamped) value, not the original request value.
        assert_eq!(max_tokens, 4096, "event must carry clamped max_tokens");
        assert!(temperature.is_some(), "temperature must be Some");
        assert!((temperature.unwrap() - 0.5_f32).abs() < 1e-5);
        assert!(top_p.is_some(), "top_p must be Some");
        assert!((top_p.unwrap() - 0.8_f32).abs() < 1e-5);
    }

    // ── Lifecycle events always present ─────────────────────────────────────

    #[tokio::test]
    async fn lifecycle_events_always_emitted() {
        let input = make_input_with_sampling(Some(4096), None, None, None);
        let events = run_sampling(input).await;

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "sampling_adjustment")
            ),
            "expected ActivityStarted"
        );
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "sampling_adjustment")
            ),
            "expected ActivityCompleted"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "did not expect ActivityFailed (fail-open)"
        );
    }

    // ── Exact boundary: request == model limit ────────────────────────────────

    #[tokio::test]
    async fn request_max_tokens_equal_to_model_limit_passes_through() {
        let input = make_input_with_sampling(Some(4096), Some(4096), None, None);
        let events = run_sampling(input).await;

        let updated = events.iter().find_map(|e| {
            if let PipelineEvent::Context(ContextEvent::SamplingUpdated { max_tokens, .. }) = e {
                Some(*max_tokens)
            } else {
                None
            }
        });

        assert_eq!(
            updated.expect("SamplingUpdated must be present"),
            4096,
            "request == model limit → value passes through unchanged"
        );
    }
}
