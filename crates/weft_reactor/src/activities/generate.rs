//! GenerateActivity: calls the generative source and streams tokens.
//!
//! Calls the LLM provider via `Services.providers`, buffers the response, and
//! pushes `Generated(Content { .. })` events onto the channel one at a time to
//! simulate streaming. Handles cancellation, heartbeat emission, and retryable
//! error classification.

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
#[cfg(test)]
use weft_core::Role;
use weft_core::{ContentPart, WeftMessage};
use weft_llm::{ProviderError, ProviderRequest, ProviderResponse};

use crate::activity::{Activity, ActivityInput};
use crate::event::{GeneratedEvent, PipelineEvent};
use crate::event_log::EventLog;
use crate::execution::ExecutionId;
use crate::services::Services;

/// Calls the generative source (LLM provider) and streams the response.
///
/// Fetches the response from the provider, parses it into [`GeneratedEvent`]
/// values, and pushes them onto the event channel one at a time. This simulates
/// streaming even when the underlying provider returns a complete response at once.
///
/// **Name:** `"generate"`
///
/// **Heartbeat support:** If `input.metadata["heartbeat_interval_secs"]` is set
/// to a positive integer, a background task is spawned that pushes
/// `Heartbeat { activity_name: "generate" }` events at the configured interval.
/// The background task is cancelled when generation completes.
///
/// **Retryable error classification:**
/// - `ProviderError::RateLimited` (429) → `retryable: true`
/// - `ProviderError::RequestFailed` (network/503) → `retryable: true`
/// - `ProviderError::ProviderHttpError { status: 429 | 503 }` → `retryable: true`
/// - `ProviderError::ProviderHttpError { status: 401 | 400 }` → `retryable: false`
/// - Cancellation → `retryable: false`
/// - All other errors → `retryable: false`
///
/// **Events pushed:**
/// - `ActivityStarted { name: "generate" }`
/// - `GenerationStarted { model, message_count }`
/// - `Heartbeat { activity_name: "generate" }` (if heartbeat configured)
/// - `Generated(Content { part })` — one per content chunk
/// - `Generated(CommandInvocation(..))` — one per parsed command call
/// - `Generated(Reasoning { content })` — for thinking tokens (if any)
/// - `Generated(Done)` — when generation is complete
/// - `GenerationCompleted { model, response_message, generated_events, input_tokens, output_tokens }`
/// - `ActivityCompleted { name: "generate", duration_ms, idempotency_key }`
/// - `GenerationFailed { model, error }` + `ActivityFailed { retryable }` — on error
pub struct GenerateActivity;

impl GenerateActivity {
    /// Construct a new GenerateActivity.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GenerateActivity {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify a provider error as retryable.
///
/// Retryable: rate limits (429), service unavailable (503), network failures.
/// Non-retryable: auth errors (401), bad request (400), content policy, unsupported.
fn is_retryable(err: &ProviderError) -> bool {
    match err {
        // Rate limited — always retry.
        ProviderError::RateLimited { .. } => true,
        // Network failures — transient, retry.
        ProviderError::RequestFailed(_) => true,
        // HTTP status-based classification.
        ProviderError::ProviderHttpError { status, .. } => {
            matches!(status, 429 | 503 | 500 | 502 | 504)
        }
        // Auth, bad request, unsupported, script errors — non-retryable.
        ProviderError::DeserializationError(_)
        | ProviderError::Unsupported(_)
        | ProviderError::WireScriptError { .. } => false,
    }
}

#[async_trait::async_trait]
impl Activity for GenerateActivity {
    fn name(&self) -> &str {
        "generate"
    }

    async fn execute(
        &self,
        _execution_id: &ExecutionId,
        input: ActivityInput,
        services: &Services,
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

        // Check cancellation before starting work.
        if cancel.is_cancelled() {
            let model = extract_model_name(&input, services);
            let _ = event_tx
                .send(PipelineEvent::GenerationFailed {
                    model: model.clone(),
                    error: "cancelled".to_string(),
                })
                .await;
            let _ = event_tx
                .send(PipelineEvent::ActivityFailed {
                    name: self.name().to_string(),
                    error: "cancelled before generation".to_string(),
                    retryable: false,
                })
                .await;
            return;
        }

        // Determine which model to use.
        let model = extract_model_name(&input, services);
        let model_id = services
            .providers
            .model_id(&model)
            .unwrap_or(&model)
            .to_string();

        // Parse heartbeat interval from metadata.
        let heartbeat_interval_secs: Option<u64> = input
            .metadata
            .get("heartbeat_interval_secs")
            .and_then(|v| v.as_u64());

        let message_count = input.messages.len();

        let _ = event_tx
            .send(PipelineEvent::GenerationStarted {
                model: model.clone(),
                message_count,
            })
            .await;

        debug!(model = %model, message_count, "generate: starting");

        // Spawn heartbeat task if configured.
        let heartbeat_cancel = cancel.child_token();
        let heartbeat_handle = if let Some(interval_secs) = heartbeat_interval_secs {
            let hb_tx = event_tx.clone();
            let hb_cancel = heartbeat_cancel.clone();
            let activity_name = self.name().to_string();
            Some(tokio::spawn(async move {
                let interval = tokio::time::Duration::from_secs(interval_secs);
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep(interval) => {
                            let _ = hb_tx.send(PipelineEvent::Heartbeat {
                                activity_name: activity_name.clone(),
                            }).await;
                        }
                        _ = hb_cancel.cancelled() => {
                            break;
                        }
                    }
                }
            }))
        } else {
            None
        };

        // Build the provider request.
        let provider_request = ProviderRequest::ChatCompletion {
            messages: input.messages.clone(),
            model: model_id.clone(),
            options: input.request.options.clone(),
        };

        // Get the provider and execute the request.
        let provider = services.providers.get(&model);

        // Use tokio::select! to support cancellation during the provider call.
        let provider_result = tokio::select! {
            result = provider.execute(provider_request) => result,
            _ = cancel.cancelled() => {
                // Cancellation during provider call.
                heartbeat_cancel.cancel();
                if let Some(handle) = heartbeat_handle {
                    let _ = handle.await;
                }
                let _ = event_tx
                    .send(PipelineEvent::GenerationFailed {
                        model: model.clone(),
                        error: "cancelled during generation".to_string(),
                    })
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::ActivityFailed {
                        name: self.name().to_string(),
                        error: "cancelled during generation".to_string(),
                        retryable: false,
                    })
                    .await;
                return;
            }
        };

        // Cancel heartbeat regardless of outcome.
        heartbeat_cancel.cancel();
        if let Some(handle) = heartbeat_handle {
            let _ = handle.await;
        }

        // Handle provider result.
        let (response_message, input_tokens, output_tokens) = match provider_result {
            Ok(ProviderResponse::ChatCompletion { message, usage }) => {
                let input_tokens = usage.as_ref().map(|u| u.prompt_tokens);
                let output_tokens = usage.as_ref().map(|u| u.completion_tokens);
                (message, input_tokens, output_tokens)
            }
            Err(e) => {
                let retryable = is_retryable(&e);
                warn!(model = %model, error = %e, retryable, "generate: provider error");
                let _ = event_tx
                    .send(PipelineEvent::GenerationFailed {
                        model: model.clone(),
                        error: e.to_string(),
                    })
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::ActivityFailed {
                        name: self.name().to_string(),
                        error: e.to_string(),
                        retryable,
                    })
                    .await;
                return;
            }
        };

        // Parse the response into GeneratedEvent values.
        // Buffer the complete response and push events one at a time.
        let generated_events = parse_response_to_events(&response_message);

        // Push each Generated event onto the channel.
        for event in &generated_events {
            let _ = event_tx.send(PipelineEvent::Generated(event.clone())).await;

            // Check cancellation between events.
            if cancel.is_cancelled() {
                let _ = event_tx
                    .send(PipelineEvent::GenerationFailed {
                        model: model.clone(),
                        error: "cancelled during streaming".to_string(),
                    })
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::ActivityFailed {
                        name: self.name().to_string(),
                        error: "cancelled during streaming".to_string(),
                        retryable: false,
                    })
                    .await;
                return;
            }
        }

        // Ensure Done is the last event if not already present.
        if !generated_events
            .iter()
            .any(|e| matches!(e, GeneratedEvent::Done))
        {
            let _ = event_tx
                .send(PipelineEvent::Generated(GeneratedEvent::Done))
                .await;
        }

        let _ = event_tx
            .send(PipelineEvent::GenerationCompleted {
                model: model.clone(),
                response_message,
                generated_events: generated_events.clone(),
                input_tokens,
                output_tokens,
            })
            .await;

        let duration_ms = start.elapsed().as_millis() as u64;
        let _ = event_tx
            .send(PipelineEvent::ActivityCompleted {
                name: self.name().to_string(),
                duration_ms,
                idempotency_key: input.idempotency_key.clone(),
            })
            .await;

        debug!(duration_ms, model = %model, "generate: completed");
    }
}

/// Extract the model routing name from the input.
///
/// Checks `input.generation_config["model"]`, then `input.routing_result.model_routing.model`,
/// then falls back to the provider's default.
fn extract_model_name(input: &ActivityInput, services: &Services) -> String {
    // Check generation_config override first.
    if let Some(ref config) = input.generation_config
        && let Some(model) = config.get("model").and_then(|v| v.as_str())
    {
        return model.to_string();
    }
    // Fall back to routing result.
    if let Some(ref routing) = input.routing_result {
        return routing.model_routing.model.clone();
    }
    // Final fallback: provider default.
    services.providers.default_name().to_string()
}

/// Parse a WeftMessage response into a sequence of GeneratedEvent values.
///
/// Content parts become `Content { part }` events. Slash-command patterns in
/// text produce `CommandInvocation` events. A `Done` event is always appended.
fn parse_response_to_events(message: &WeftMessage) -> Vec<GeneratedEvent> {
    let mut events = Vec::new();
    let mut full_text = String::new();

    for part in &message.content {
        match part {
            ContentPart::Text(text) => {
                full_text.push_str(text);
                // Push each text part as a Content event (simulates per-chunk streaming).
                events.push(GeneratedEvent::Content {
                    part: ContentPart::Text(text.clone()),
                });
            }
            ContentPart::CommandCall(call) => {
                // CommandCall content part maps to a CommandInvocation event.
                // CommandCallContent has fields: .command (name) and .arguments_json (JSON string).
                let arguments: serde_json::Value =
                    serde_json::from_str(&call.arguments_json).unwrap_or(serde_json::Value::Null);
                let invocation = weft_core::CommandInvocation {
                    name: call.command.clone(),
                    action: weft_core::CommandAction::Execute,
                    arguments,
                };
                events.push(GeneratedEvent::CommandInvocation(invocation));
            }
            other => {
                // Other content types (Image, etc.) are pushed as-is.
                events.push(GeneratedEvent::Content {
                    part: other.clone(),
                });
            }
        }
    }

    // Parse any slash commands from the accumulated text.
    // This handles text-based command invocations like "/command args".
    // parse_response requires the set of known command names to match against.
    // Since we don't have access to the command registry here, we use an empty set.
    // Slash command parsing from plain text is a best-effort feature in Phase 3;
    // the Reactor (Phase 4) will provide the known command set via ActivityInput.
    if !full_text.is_empty() {
        let known_commands = std::collections::HashSet::new();
        let parsed = weft_commands::parse_response(&full_text, &known_commands);
        for cmd in parsed.invocations {
            // Avoid duplicating CommandInvocations that were already added from CommandCall parts.
            let already_added = events.iter().any(
                |e| matches!(e, GeneratedEvent::CommandInvocation(inv) if inv.name == cmd.name),
            );
            if !already_added {
                events.push(GeneratedEvent::CommandInvocation(cmd));
            }
        }
    }

    // Always append Done.
    events.push(GeneratedEvent::Done);
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NullEventLog;
    use crate::test_support::{
        collect_events, make_test_input, make_test_services_with_response,
        make_test_services_with_slow_provider,
    };
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn generate_name() {
        assert_eq!(GenerateActivity::new().name(), "generate");
    }

    // ── Happy path: pushes GenerationStarted, Content events, Done, GenerationCompleted ──

    #[tokio::test]
    async fn generate_pushes_generation_started_and_content_events() {
        let response_text = "Hello, world!";
        let services = make_test_services_with_response(response_text);
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push ActivityStarted.
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityStarted { name } if name == "generate")
            ),
            "expected ActivityStarted"
        );

        // Must push GenerationStarted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::GenerationStarted { .. })),
            "expected GenerationStarted"
        );

        // Must push at least one Generated(Content) event.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Generated(GeneratedEvent::Content { .. }))),
            "expected Generated(Content) events"
        );

        // Must push Generated(Done).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Generated(GeneratedEvent::Done))),
            "expected Generated(Done)"
        );

        // Must push GenerationCompleted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::GenerationCompleted { .. })),
            "expected GenerationCompleted"
        );

        // Must push ActivityCompleted.
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "generate")
            ),
            "expected ActivityCompleted"
        );
    }

    // ── GenerationCompleted includes response_message ─────────────────────

    #[tokio::test]
    async fn generate_completed_includes_response_message() {
        let response_text = "The answer is 42.";
        let services = make_test_services_with_response(response_text);
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let completed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::GenerationCompleted { .. }))
            .expect("expected GenerationCompleted");

        match completed {
            PipelineEvent::GenerationCompleted {
                response_message, ..
            } => {
                // The response_message should be an Assistant message.
                assert_eq!(response_message.role, Role::Assistant);
                // Content should contain the response text.
                let has_text = response_message
                    .content
                    .iter()
                    .any(|p| matches!(p, ContentPart::Text(t) if t.contains(response_text)));
                assert!(
                    has_text,
                    "response_message should contain '{response_text}'"
                );
            }
            _ => panic!("expected GenerationCompleted"),
        }
    }

    // ── Idempotency key in ActivityCompleted ─────────────────────────────

    #[tokio::test]
    async fn generate_includes_idempotency_key_in_completed() {
        let services = make_test_services_with_response("response");
        let mut input = make_test_input();
        input.idempotency_key = Some("exec123:generate:0".to_string());

        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let completed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ActivityCompleted { name, .. } if name == "generate"))
            .expect("expected ActivityCompleted");

        match completed {
            PipelineEvent::ActivityCompleted {
                idempotency_key, ..
            } => {
                assert_eq!(
                    *idempotency_key,
                    Some("exec123:generate:0".to_string()),
                    "idempotency_key should be carried through to ActivityCompleted"
                );
            }
            _ => panic!("expected ActivityCompleted"),
        }
    }

    // ── Cancellation: pushes GenerationFailed + ActivityFailed (retryable: false) ──

    #[tokio::test]
    async fn generate_cancellation_pushes_generation_failed_not_retryable() {
        let services = make_test_services_with_response("response");
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        cancel.cancel(); // Pre-cancelled

        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push GenerationFailed.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::GenerationFailed { .. })),
            "expected GenerationFailed on cancellation"
        );

        // Must push ActivityFailed with retryable: false.
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ActivityFailed { .. }))
            .expect("expected ActivityFailed");
        match failed {
            PipelineEvent::ActivityFailed { retryable, .. } => {
                assert!(!retryable, "cancellation should not be retryable");
            }
            _ => panic!("expected ActivityFailed"),
        }
    }

    // ── Provider rate limit (429): retryable: true ────────────────────────

    #[tokio::test]
    async fn generate_rate_limit_error_is_retryable() {
        let services =
            crate::test_support::make_test_services_with_error(ProviderError::RateLimited {
                retry_after_ms: 1000,
            });
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ActivityFailed { .. }))
            .expect("expected ActivityFailed");
        match failed {
            PipelineEvent::ActivityFailed { retryable, .. } => {
                assert!(*retryable, "rate limit should be retryable");
            }
            _ => panic!("expected ActivityFailed"),
        }
    }

    // ── Auth error (401): retryable: false ────────────────────────────────

    #[tokio::test]
    async fn generate_auth_error_is_not_retryable() {
        let services =
            crate::test_support::make_test_services_with_error(ProviderError::ProviderHttpError {
                status: 401,
                body: "unauthorized".to_string(),
            });
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = crate::execution::ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::ActivityFailed { .. }))
            .expect("expected ActivityFailed");
        match failed {
            PipelineEvent::ActivityFailed { retryable, .. } => {
                assert!(!retryable, "auth error should not be retryable");
            }
            _ => panic!("expected ActivityFailed"),
        }
    }

    // ── Heartbeat emission ────────────────────────────────────────────────

    #[tokio::test]
    async fn generate_emits_heartbeat_events_when_configured() {
        // Use a slow provider (2 second delay) so the heartbeat fires before
        // the provider call completes. With tokio::time::pause(), we control
        // the clock: advance 1.5s to trigger the heartbeat, then another 1s
        // to complete the provider's 2-second sleep.
        let services = make_test_services_with_slow_provider(2, "response");
        let mut input = make_test_input();
        // Set heartbeat_interval_secs = 1 in metadata.
        input.metadata = serde_json::json!({ "heartbeat_interval_secs": 1 });

        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let exec_id = crate::execution::ExecutionId::new();

        // Pause the Tokio clock so we control time advancement.
        tokio::time::pause();

        let activity = GenerateActivity::new();
        // Spawn the activity so we can advance time while it runs.
        let exec_id_clone = exec_id.clone();
        let input_clone = input.clone();
        let handle = tokio::spawn(async move {
            activity
                .execute(
                    &exec_id_clone,
                    input_clone,
                    &services,
                    &NullEventLog,
                    tx,
                    cancel,
                )
                .await;
        });

        // Yield so the spawned task starts executing and reaches the select! loop.
        tokio::task::yield_now().await;

        // Advance 1.5 seconds: crosses the 1s heartbeat interval, fires Heartbeat.
        tokio::time::advance(tokio::time::Duration::from_millis(1500)).await;
        // Yield again to let the heartbeat task run.
        tokio::task::yield_now().await;

        // Advance another second to complete the provider's 2-second sleep.
        tokio::time::advance(tokio::time::Duration::from_millis(1000)).await;
        tokio::task::yield_now().await;

        handle.await.expect("activity task panicked");

        let events = collect_events(&mut rx);

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Heartbeat { activity_name } if activity_name == "generate")),
            "expected at least one Heartbeat event"
        );
    }

    // ── is_retryable classification ──────────────────────────────────────

    #[test]
    fn retryable_rate_limited() {
        assert!(is_retryable(&ProviderError::RateLimited {
            retry_after_ms: 1000
        }));
    }

    #[test]
    fn retryable_request_failed() {
        assert!(is_retryable(&ProviderError::RequestFailed(
            "connection reset".to_string()
        )));
    }

    #[test]
    fn retryable_503() {
        assert!(is_retryable(&ProviderError::ProviderHttpError {
            status: 503,
            body: "service unavailable".to_string(),
        }));
    }

    #[test]
    fn not_retryable_401() {
        assert!(!is_retryable(&ProviderError::ProviderHttpError {
            status: 401,
            body: "unauthorized".to_string(),
        }));
    }

    #[test]
    fn not_retryable_400() {
        assert!(!is_retryable(&ProviderError::ProviderHttpError {
            status: 400,
            body: "bad request".to_string(),
        }));
    }

    #[test]
    fn not_retryable_unsupported() {
        assert!(!is_retryable(&ProviderError::Unsupported(
            "embeddings not supported".to_string()
        )));
    }
}
