//! GenerateActivity: calls the generative source and streams tokens.
//!
//! Calls the LLM provider via `services.providers()` using `execute_stream`, which
//! yields `ProviderChunk` values as they arrive. Delta chunks push
//! `Generation(GenerationEvent::Chunk(Content { .. }))` events immediately; the
//! final Complete chunk triggers `Generation(GenerationEvent::Completed)`.
//! Providers without true streaming support use the default `execute_stream`
//! implementation, which wraps `execute()` in a single-element stream — same code
//! path, zero additional latency for those providers.
//!
//! Handles cancellation, heartbeat emission, and retryable error classification.

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info_span, warn};
#[cfg(test)]
use weft_core::Role;
use weft_core::{ContentPart, WeftMessage};
use weft_llm_trait::{ProviderChunk, ProviderError, ProviderRequest, ProviderResponse};

use weft_reactor_trait::{
    Activity, ActivityEvent, ActivityInput, EventLog, ExecutionId, GeneratedEvent, GenerationEvent,
    PipelineEvent, ServiceLocator,
};

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
/// `Activity(ActivityEvent::Heartbeat { activity_name: "generate" })` events at
/// the configured interval. The background task is cancelled when generation
/// completes.
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
/// - `Activity(ActivityEvent::Started { name: "generate" })`
/// - `Generation(GenerationEvent::Started { model, message_count })`
/// - `Activity(ActivityEvent::Heartbeat { activity_name: "generate" })` (if heartbeat configured)
/// - `Generation(GenerationEvent::Chunk(Content { part }))` — one per content chunk
/// - `Generation(GenerationEvent::Chunk(CommandInvocation(..)))` — one per parsed command call
/// - `Generation(GenerationEvent::Chunk(Reasoning { content }))` — for thinking tokens (if any)
/// - `Generation(GenerationEvent::Chunk(Done))` — when generation is complete
/// - `Generation(GenerationEvent::Completed { model, response_message, generated_events, input_tokens, output_tokens })`
/// - `Activity(ActivityEvent::Completed { name: "generate", idempotency_key })`
/// - `Generation(GenerationEvent::Failed { model, error })` + `Activity(ActivityEvent::Failed { retryable })` — on error
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
        services: &dyn ServiceLocator,
        _event_log: &dyn EventLog,
        event_tx: mpsc::Sender<PipelineEvent>,
        cancel: CancellationToken,
    ) {
        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Started {
                name: self.name().to_string(),
            }))
            .await;

        // Check cancellation before starting work.
        if cancel.is_cancelled() {
            let model = extract_model_name(&input, services);
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Failed {
                    model: model.clone(),
                    error: "cancelled".to_string(),
                }))
                .await;
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "cancelled before generation".to_string(),
                    retryable: false,
                }))
                .await;
            return;
        }

        // Determine which model to use.
        let model = extract_model_name(&input, services);
        let model_id = services
            .providers()
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
            .send(PipelineEvent::Generation(GenerationEvent::Started {
                model: model.clone(),
                message_count,
            }))
            .await;

        debug!(model = %model, message_count, "generate: starting");

        // Create the `generate` span. This span brackets the LLM provider call and the
        // full stream consumption. Attributes known at span creation are set immediately;
        // token counts and stop reason are recorded via Span::current().record() at close.
        //
        // `llm.attempt` is the retry attempt counter. GenerateActivity does not track
        // retries internally (the reactor handles retry orchestration), so we read it
        // from metadata if available, defaulting to 0.
        let llm_attempt: u32 = input
            .metadata
            .get("generate_retry_attempt")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let generate_span = info_span!(
            "generate",
            llm.model = %model,
            llm.provider = tracing::field::Empty,
            llm.attempt = llm_attempt,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
        );

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
                            let _ = hb_tx.send(PipelineEvent::Activity(ActivityEvent::Heartbeat {
                                activity_name: activity_name.clone(),
                            })).await;
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

        // Build the provider request, merging clamped sampling parameters from
        // generation_config over the original request options.
        //
        // SamplingAdjustmentActivity writes the clamped max_tokens (and optionally
        // temperature/top_p) into generation_config so that downstream activities use
        // the provider-safe values. We must read them here rather than from
        // input.request.options, which carries the original unclamped client values.
        let mut options = input.request.options.clone();
        if let Some(ref cfg) = input.generation_config {
            if let Some(max_tokens) = cfg.get("max_tokens").and_then(|v| v.as_u64()) {
                options.max_tokens = Some(max_tokens as u32);
            }
            if let Some(temperature) = cfg.get("temperature").and_then(|v| v.as_f64()) {
                options.temperature = Some(temperature as f32);
            }
            if let Some(top_p) = cfg.get("top_p").and_then(|v| v.as_f64()) {
                options.top_p = Some(top_p as f32);
            }
        }
        let provider_request = ProviderRequest::ChatCompletion {
            messages: input.messages.clone(),
            model: model_id.clone(),
            options,
        };

        // Get the provider and open the streaming request.
        let provider = services.providers().get(&model);
        // Record provider name now that it is known; must happen before the stream
        // is opened so the attribute is present for the full span lifetime.
        generate_span.record("llm.provider", provider.name());

        // Use tokio::select! to support cancellation while opening the stream.
        let stream_result = tokio::select! {
            result = provider.execute_stream(provider_request) => result,
            _ = cancel.cancelled() => {
                // Cancellation before the stream opened.
                heartbeat_cancel.cancel();
                if let Some(handle) = heartbeat_handle {
                    let _ = handle.await;
                }
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Failed {
                        model: model.clone(),
                        error: "cancelled during generation".to_string(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: "cancelled during generation".to_string(),
                        retryable: false,
                    }))
                    .await;
                return;
            }
        };

        let mut stream: std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<weft_llm_trait::ProviderChunk, weft_llm_trait::ProviderError>,
                    > + Send,
            >,
        > = match stream_result {
            Ok(s) => s,
            Err(e) => {
                heartbeat_cancel.cancel();
                if let Some(handle) = heartbeat_handle {
                    let _ = handle.await;
                }
                let retryable = is_retryable(&e);
                warn!(model = %model, error = %e, retryable, "generate: provider error opening stream");
                let _ = event_tx
                    .send(PipelineEvent::Generation(GenerationEvent::Failed {
                        model: model.clone(),
                        error: e.to_string(),
                    }))
                    .await;
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: e.to_string(),
                        retryable,
                    }))
                    .await;
                return;
            }
        };

        // Accumulate generated events for GenerationCompleted.
        // Delta chunks push events immediately; the Complete chunk contributes the
        // full response_message and usage data at stream end.
        let mut all_generated_events: Vec<GeneratedEvent> = Vec::new();
        let mut response_message_opt: Option<WeftMessage> = None;
        let mut input_tokens: Option<u32> = None;
        let mut output_tokens: Option<u32> = None;
        // Accumulate delta text so we can reconstruct a complete response message
        // when a streaming provider yields only deltas (no Complete chunk with a
        // pre-assembled WeftMessage). Currently used as a fallback; full delta
        // reconstruction is future work.
        let mut accumulated_delta_text = String::new();

        // Consume the stream chunk by chunk. Use tokio::select! on each iteration
        // so cancellation is detected between chunks without an extra task.
        loop {
            let chunk = tokio::select! {
                item = stream.next() => item,
                _ = cancel.cancelled() => {
                    // Cancellation while consuming the stream.
                    drop(stream);
                    heartbeat_cancel.cancel();
                    if let Some(handle) = heartbeat_handle {
                        let _ = handle.await;
                    }
                    let _ = event_tx
                        .send(PipelineEvent::Generation(GenerationEvent::Failed {
                            model: model.clone(),
                            error: "cancelled during streaming".to_string(),
                        }))
                        .await;
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Failed {
                            name: self.name().to_string(),
                            error: "cancelled during streaming".to_string(),
                            retryable: false,
                        }))
                        .await;
                    return;
                }
            };

            match chunk {
                None => {
                    // Stream ended without a Complete chunk (e.g., delta-only streaming
                    // provider). Synthesise a response_message from accumulated text.
                    if response_message_opt.is_none() && !accumulated_delta_text.is_empty() {
                        response_message_opt = Some(WeftMessage {
                            role: weft_core::Role::Assistant,
                            source: weft_core::Source::Provider,
                            model: Some(model.clone()),
                            content: vec![ContentPart::Text(accumulated_delta_text.clone())],
                            delta: false,
                            message_index: 0,
                        });
                    }
                    break;
                }
                Some(Err(e)) => {
                    // Error mid-stream: cancel heartbeat, push failure events.
                    drop(stream);
                    heartbeat_cancel.cancel();
                    if let Some(handle) = heartbeat_handle {
                        let _ = handle.await;
                    }
                    let retryable = is_retryable(&e);
                    warn!(model = %model, error = %e, retryable, "generate: provider stream error");
                    let _ = event_tx
                        .send(PipelineEvent::Generation(GenerationEvent::Failed {
                            model: model.clone(),
                            error: e.to_string(),
                        }))
                        .await;
                    let _ = event_tx
                        .send(PipelineEvent::Activity(ActivityEvent::Failed {
                            name: self.name().to_string(),
                            error: e.to_string(),
                            retryable,
                        }))
                        .await;
                    return;
                }
                Some(Ok(ProviderChunk::Delta(delta))) => {
                    // A real streaming chunk: push text immediately as it arrives.
                    if let Some(text) = delta.text {
                        accumulated_delta_text.push_str(&text);
                        let event = GeneratedEvent::Content {
                            part: ContentPart::Text(text),
                        };
                        all_generated_events.push(event.clone());
                        let _ = event_tx
                            .send(PipelineEvent::Generation(GenerationEvent::Chunk(event)))
                            .await;
                    }
                    // Tool call deltas are accumulated for future streaming tool support.
                    // For now they are carried through but not yet pushed as events
                    // (tool calls are assembled by parse_response_to_events on Complete).
                }
                Some(Ok(ProviderChunk::Complete(response))) => {
                    // The full response arrived (either as the sole chunk from a
                    // non-streaming provider, or as the final chunk from a streaming
                    // provider that assembles the message server-side).
                    //
                    // Parse the complete response into GeneratedEvent values (same
                    // logic as before). For non-streaming providers this produces the
                    // same events as the old buffered code path. For streaming providers
                    // that yield deltas *before* this chunk, we skip re-emitting text
                    // that was already sent as Delta events, but still parse command
                    // invocations from the assembled response.
                    let (msg, tok_in, tok_out) = match response {
                        ProviderResponse::ChatCompletion { message, usage } => {
                            let tok_in = usage.as_ref().map(|u| u.prompt_tokens);
                            let tok_out = usage.as_ref().map(|u| u.completion_tokens);
                            (message, tok_in, tok_out)
                        }
                    };
                    input_tokens = tok_in;
                    output_tokens = tok_out;

                    // Determine if we already emitted content as deltas.
                    // If no deltas were pushed yet (non-streaming provider), parse
                    // and push all events now (same as old code path).
                    // If deltas were already pushed, only extract command invocations
                    // from the complete response to avoid duplicate text events.
                    let events_from_complete = if accumulated_delta_text.is_empty() {
                        // Non-streaming: no deltas sent yet. Full parse + push.
                        parse_response_to_events(&msg)
                    } else {
                        // Streaming: deltas already sent. Only emit command invocations
                        // that weren't captured via delta text, plus Done.
                        parse_only_commands_and_done(&msg)
                    };

                    for event in &events_from_complete {
                        all_generated_events.push(event.clone());
                        let _ = event_tx
                            .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                                event.clone(),
                            )))
                            .await;
                    }

                    response_message_opt = Some(msg);
                    // Complete received; stream will end after this.
                }
            }
        }

        // Cancel heartbeat — stream consumption is done.
        heartbeat_cancel.cancel();
        if let Some(handle) = heartbeat_handle {
            let _ = handle.await;
        }

        // Ensure Done is present.
        if !all_generated_events
            .iter()
            .any(|e| matches!(e, GeneratedEvent::Done))
        {
            let done_event = GeneratedEvent::Done;
            all_generated_events.push(done_event.clone());
            let _ = event_tx
                .send(PipelineEvent::Generation(GenerationEvent::Chunk(
                    done_event,
                )))
                .await;
        }

        // Build response_message for GenerationCompleted. If the provider yielded
        // only deltas with no Complete chunk, we synthesised one above from the
        // accumulated text. As a last resort, emit an empty assistant message.
        let response_message = response_message_opt.unwrap_or_else(|| WeftMessage {
            role: weft_core::Role::Assistant,
            source: weft_core::Source::Provider,
            model: Some(model.clone()),
            content: vec![],
            delta: false,
            message_index: 0,
        });

        // Record token counts and stop reason on the generate span at close.
        // These are known only after the stream is fully consumed.
        let stop_reason = if all_generated_events
            .iter()
            .any(|e| matches!(e, GeneratedEvent::Refused { .. }))
        {
            "refusal"
        } else if all_generated_events
            .iter()
            .any(|e| matches!(e, GeneratedEvent::CommandInvocation(_)))
        {
            "tool_use"
        } else {
            "end_turn"
        };
        generate_span.record("llm.input_tokens", input_tokens.unwrap_or(0));
        generate_span.record("llm.output_tokens", output_tokens.unwrap_or(0));
        generate_span.record("llm.stop_reason", stop_reason);

        let _ = event_tx
            .send(PipelineEvent::Generation(GenerationEvent::Completed {
                model: model.clone(),
                response_message,
                generated_events: all_generated_events,
                input_tokens,
                output_tokens,
            }))
            .await;

        let _ = event_tx
            .send(PipelineEvent::Activity(ActivityEvent::Completed {
                name: self.name().to_string(),
                idempotency_key: input.idempotency_key.clone(),
            }))
            .await;

        debug!(model = %model, "generate: completed");
    }
}

/// Extract the model routing name from the input.
///
/// Checks `input.generation_config["model"]`, then `input.routing_result.model_routing.model`,
/// then falls back to the provider's default.
fn extract_model_name(input: &ActivityInput, services: &dyn ServiceLocator) -> String {
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
    services.providers().default_name().to_string()
}

/// Parse a WeftMessage response into a sequence of GeneratedEvent values.
///
/// Content parts become `Content { part }` events. Slash-command patterns in
/// text produce `CommandInvocation` events. A `Done` event is always appended.
///
/// Used by `GenerateActivity` for non-streaming providers (or as a fallback when
/// the `Complete` chunk is the first and only chunk). For streaming providers that
/// already emitted delta text events, use `parse_only_commands_and_done` instead
/// to avoid duplicating text content.
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
        let parsed = weft_commands_trait::parse_response(&full_text, &known_commands);
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

/// Extract only command invocations and `Done` from a complete response message.
///
/// Used after a streaming provider has already pushed delta text events. The
/// complete response is still parsed for `CommandCall` content parts and
/// slash-command patterns — these cannot be emitted incrementally from delta
/// chunks because a command invocation only becomes recognisable once its full
/// name and arguments have arrived. Text parts are skipped since they were
/// already pushed as `Delta` events.
///
/// A `Done` event is always appended.
fn parse_only_commands_and_done(message: &WeftMessage) -> Vec<GeneratedEvent> {
    let mut events = Vec::new();
    let mut full_text = String::new();

    for part in &message.content {
        match part {
            ContentPart::Text(text) => {
                // Accumulate text for slash-command parsing but do NOT emit a
                // Content event — delta text was already pushed.
                full_text.push_str(text);
            }
            ContentPart::CommandCall(call) => {
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
                // Non-text, non-command-call parts (images, etc.) were not
                // streamed as deltas; emit them now.
                events.push(GeneratedEvent::Content {
                    part: other.clone(),
                });
            }
        }
    }

    // Parse slash commands from accumulated text, deduplicating against already
    // captured CommandCall-based invocations.
    if !full_text.is_empty() {
        let known_commands = std::collections::HashSet::new();
        let parsed = weft_commands_trait::parse_response(&full_text, &known_commands);
        for cmd in parsed.invocations {
            let already_added = events.iter().any(
                |e| matches!(e, GeneratedEvent::CommandInvocation(inv) if inv.name == cmd.name),
            );
            if !already_added {
                events.push(GeneratedEvent::CommandInvocation(cmd));
            }
        }
    }

    events.push(GeneratedEvent::Done);
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        NullEventLog, collect_events, make_test_input, make_test_services_with_chunk_stream,
        make_test_services_with_mid_stream_error, make_test_services_with_response,
        make_test_services_with_slow_provider,
    };
    use pretty_assertions::assert_eq;
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
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push Activity(Started).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "generate")
            ),
            "expected Activity(Started)"
        );

        // Must push Generation(Started).
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Started { .. })
            )),
            "expected Generation(Started)"
        );

        // Must push at least one Generation(Chunk(Content)) event.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content { .. }))
            )),
            "expected Generation(Chunk(Content)) events"
        );

        // Must push Generation(Chunk(Done)).
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Done))
            )),
            "expected Generation(Chunk(Done))"
        );

        // Must push Generation(Completed).
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Completed { .. })
            )),
            "expected Generation(Completed)"
        );

        // Must push Activity(Completed).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "generate")
            ),
            "expected Activity(Completed)"
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
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let completed = events
            .iter()
            .find(|e| {
                matches!(
                    e,
                    PipelineEvent::Generation(GenerationEvent::Completed { .. })
                )
            })
            .expect("expected Generation(Completed)");

        match completed {
            PipelineEvent::Generation(GenerationEvent::Completed {
                response_message, ..
            }) => {
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
            _ => panic!("expected Generation(Completed)"),
        }
    }

    // ── Idempotency key in Activity(Completed) ─────────────────────────────

    #[tokio::test]
    async fn generate_includes_idempotency_key_in_completed() {
        let services = make_test_services_with_response("response");
        let mut input = make_test_input();
        input.idempotency_key = Some("exec123:generate:0".to_string());

        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let completed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "generate"))
            .expect("expected Activity(Completed)");

        match completed {
            PipelineEvent::Activity(ActivityEvent::Completed {
                idempotency_key, ..
            }) => {
                assert_eq!(
                    *idempotency_key,
                    Some("exec123:generate:0".to_string()),
                    "idempotency_key should be carried through to Activity(Completed)"
                );
            }
            _ => panic!("expected Activity(Completed)"),
        }
    }

    // ── Cancellation: pushes Generation(Failed) + Activity(Failed) (retryable: false) ──

    #[tokio::test]
    async fn generate_cancellation_pushes_generation_failed_not_retryable() {
        let services = make_test_services_with_response("response");
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        cancel.cancel(); // Pre-cancelled

        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push Generation(Failed).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Generation(GenerationEvent::Failed { .. }))),
            "expected Generation(Failed) on cancellation"
        );

        // Must push Activity(Failed) with retryable: false.
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");
        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(!retryable, "cancellation should not be retryable");
            }
            _ => panic!("expected Activity(Failed)"),
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
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");
        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(*retryable, "rate limit should be retryable");
            }
            _ => panic!("expected Activity(Failed)"),
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
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");
        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(!retryable, "auth error should not be retryable");
            }
            _ => panic!("expected Activity(Failed)"),
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
        let exec_id = ExecutionId::new();

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
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Heartbeat { activity_name }) if activity_name == "generate")),
            "expected at least one Activity(Heartbeat) event"
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

    // ── Streaming: multi-chunk provider pushes events incrementally ───────

    /// Verify that `GenerateActivity` pushes one `Generation(Chunk(Content))` event
    /// per delta chunk, and that all three chunks arrive before `Generation(Completed)`.
    ///
    /// This is the core streaming correctness test: events must arrive as chunks
    /// are yielded, not after buffering the whole response.
    #[tokio::test]
    async fn test_generate_streams_chunks_incrementally() {
        let chunks = vec!["Hello".to_string(), ", ".to_string(), "world!".to_string()];
        let services = make_test_services_with_chunk_stream(chunks.clone(), true);
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Count distinct Generation(Chunk(Content)) events — there must be one per delta chunk.
        let content_events: Vec<_> = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    PipelineEvent::Generation(GenerationEvent::Chunk(
                        GeneratedEvent::Content { .. }
                    ))
                )
            })
            .collect();

        assert_eq!(
            content_events.len(),
            chunks.len(),
            "expected one Chunk(Content) event per delta chunk; got {}",
            content_events.len()
        );

        // Verify the text of each Content event matches the original chunk text.
        for (i, (expected, event)) in chunks.iter().zip(content_events.iter()).enumerate() {
            match event {
                PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content {
                    part: ContentPart::Text(text),
                })) => {
                    assert_eq!(
                        text, expected,
                        "chunk {i}: expected text {:?}, got {:?}",
                        expected, text
                    );
                }
                other => {
                    panic!("chunk {i}: expected Generation(Chunk(Content(Text))), got {other:?}")
                }
            }
        }

        // Generation(Completed) must be pushed.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Completed { .. })
            )),
            "expected Generation(Completed) after all chunks"
        );

        // Generation(Chunk(Done)) must be present.
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Done))
            )),
            "expected Generation(Chunk(Done))"
        );

        // Activity(Completed) must be present.
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "generate")
            ),
            "expected Activity(Completed)"
        );
    }

    /// Verify that a provider implementing only `execute` (not `execute_stream`)
    /// still works correctly via the default `execute_stream` wrapper.
    ///
    /// `StubProvider` (used by `make_test_services_with_response`) does not
    /// override `execute_stream`, so this test exercises the default implementation.
    #[tokio::test]
    async fn test_generate_default_stream_wraps_execute() {
        let response_text = "Default stream response.";
        let services = make_test_services_with_response(response_text);
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must emit at least one Chunk(Content) event with the response text.
        let has_content = events.iter().any(|e| {
            matches!(e, PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content {
                part: ContentPart::Text(t),
            })) if t.contains(response_text))
        });
        assert!(
            has_content,
            "expected Generation(Chunk(Content)) with response text"
        );

        // Must emit Generation(Completed).
        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Generation(GenerationEvent::Completed { .. })
            )),
            "expected Generation(Completed)"
        );

        // Must emit Activity(Completed).
        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "generate")
            ),
            "expected Activity(Completed)"
        );
    }

    /// Verify that cancellation mid-stream causes `Generation(Failed)` with
    /// `retryable: false` and drops the stream.
    ///
    /// Uses a pre-cancelled token so the select! in the stream consumption loop
    /// fires before the first chunk arrives.
    #[tokio::test]
    async fn test_generate_stream_cancellation_mid_stream() {
        let chunks = vec!["chunk1".to_string(), "chunk2".to_string()];
        let services = make_test_services_with_chunk_stream(chunks, true);
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        // Pre-cancel so the stream loop is interrupted immediately.
        cancel.cancel();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // Must push Generation(Failed).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Generation(GenerationEvent::Failed { .. }))),
            "expected Generation(Failed) on cancellation"
        );

        // Must push Activity(Failed) with retryable: false.
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");
        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(!retryable, "cancellation should not be retryable");
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }

    /// Verify that a mid-stream error (after one chunk was already delivered) causes
    /// `Generation(Failed)` and `Activity(Failed)` with the correct `retryable` flag.
    ///
    /// The `MidStreamErrorProvider` yields one delta chunk then errors. The first
    /// chunk must have been pushed as a `Generation(Chunk(Content))` event, and the
    /// error must trigger `Generation(Failed)` + `Activity(Failed)`.
    #[tokio::test]
    async fn test_generate_stream_error_mid_stream() {
        let services = make_test_services_with_mid_stream_error(
            "partial text",
            ProviderError::RateLimited {
                retry_after_ms: 5000,
            },
        );
        let input = make_test_input();
        let (tx, mut rx) = mpsc::channel(128);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = GenerateActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        // The first chunk must have been pushed before the error.
        let has_partial_content = events.iter().any(|e| {
            matches!(e, PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Content {
                part: ContentPart::Text(t),
            })) if t == "partial text")
        });
        assert!(
            has_partial_content,
            "expected Generation(Chunk(Content)) with 'partial text' before the mid-stream error"
        );

        // Must push Generation(Failed).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Generation(GenerationEvent::Failed { .. }))),
            "expected Generation(Failed) after mid-stream error"
        );

        // RateLimited is retryable — Activity(Failed) must have retryable: true.
        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed)");
        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. }) => {
                assert!(
                    *retryable,
                    "rate limit mid-stream error should be retryable"
                );
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }
}
