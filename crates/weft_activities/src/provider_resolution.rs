//! ProviderResolution activity: resolve provider and capabilities for the selected model.
//!
//! Reads the selected model name from `ActivityInput.metadata` (placed by the reactor),
//! looks up the model_id, capabilities, and max_tokens from the provider service, and
//! finds the provider name from config. Emits `ProviderResolved` with all resolved data
//! so downstream activities can act on provider capabilities without re-querying.
//!
//! **Fail mode: CLOSED.** If no model_id can be resolved, pushes `ActivityFailed`.
//! Generation requires a resolved provider — proceeding without one is undefined behaviour.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use weft_reactor_trait::{
    Activity, ActivityEvent, ActivityInput, EventLog, ExecutionId, FailureDetail, PipelineEvent,
    SelectionEvent, ServiceLocator,
};

/// Resolves provider and model capabilities for the model selected by `ModelSelectionActivity`.
///
/// **Name:** `"provider_resolution"`
///
/// **Events pushed:**
/// - `Activity(ActivityEvent::Started { name: "provider_resolution" })`
/// - `Selection(SelectionEvent::ProviderResolved { model_name, model_id, provider_name, capabilities, max_tokens })` — on success
/// - `Activity(ActivityEvent::Completed { name: "provider_resolution", idempotency_key: None })`
/// - `Activity(ActivityEvent::Failed { name: "provider_resolution", error, retryable: false })` — on failure
pub struct ProviderResolutionActivity;

impl ProviderResolutionActivity {
    /// Construct a new `ProviderResolutionActivity`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ProviderResolutionActivity {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Activity for ProviderResolutionActivity {
    fn name(&self) -> &str {
        "provider_resolution"
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

        if cancel.is_cancelled() {
            let _ = event_tx
                .send(PipelineEvent::Activity(ActivityEvent::Failed {
                    name: self.name().to_string(),
                    error: "cancelled before provider resolution".to_string(),
                    retryable: false,
                    detail: FailureDetail {
                        error_code: "cancelled".to_string(),
                        detail: serde_json::Value::Null,
                        cause: Some(
                            "cancellation token was set before activity started".to_string(),
                        ),
                        attempted: Some("resolve provider for selected model".to_string()),
                        fallback: None,
                    },
                }))
                .await;
            return;
        }

        // Step 1: Extract selected_model from metadata. Fail closed if missing.
        let selected_model = match input
            .metadata
            .get("selected_model")
            .and_then(|v| v.as_str())
        {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: "selected_model missing from metadata — reactor wiring error"
                            .to_string(),
                        retryable: false,
                        detail: FailureDetail {
                            error_code: "model_not_found".to_string(),
                            detail: serde_json::json!({
                                "model_name": "(missing from metadata)",
                                "providers_checked": [],
                            }),
                            cause: Some("selected_model field absent from activity metadata — reactor wiring error".to_string()),
                            attempted: Some("extract selected_model from activity metadata".to_string()),
                            fallback: None,
                        },
                    }))
                    .await;
                return;
            }
        };

        // Step 2: Resolve model_id. Fail closed if None.
        let model_id = match services.providers().model_id(&selected_model) {
            Some(id) => id.to_string(),
            None => {
                let _ = event_tx
                    .send(PipelineEvent::Activity(ActivityEvent::Failed {
                        name: self.name().to_string(),
                        error: format!(
                            "model '{selected_model}' has no model_id mapping — configuration error"
                        ),
                        retryable: false,
                        detail: FailureDetail {
                            error_code: "model_not_found".to_string(),
                            detail: serde_json::json!({
                                "model_name": selected_model,
                                "providers_checked": [],
                            }),
                            cause: Some(format!(
                                "model '{selected_model}' is not registered in any provider"
                            )),
                            attempted: Some(format!("resolve model_id for '{selected_model}'",)),
                            fallback: None,
                        },
                    }))
                    .await;
                return;
            }
        };

        // Step 3: Resolve capabilities. Default to ["chat_completions"] if None.
        let capabilities: Vec<String> = match services
            .providers()
            .model_capabilities(&selected_model)
        {
            Some(caps) => caps.iter().map(|c| c.as_str().to_string()).collect(),
            None => {
                warn!(
                    model = %selected_model,
                    "provider_resolution: no capabilities registered for model, defaulting to chat_completions"
                );
                vec!["chat_completions".to_string()]
            }
        };

        // Step 4: Resolve max_tokens. Default to 4096 if None.
        let max_tokens = match services.providers().max_tokens_for(&selected_model) {
            Some(t) => t,
            None => {
                warn!(
                    model = %selected_model,
                    "provider_resolution: no max_tokens registered for model, defaulting to 4096"
                );
                4096
            }
        };

        // Step 5: Look up provider_name from the per-request config snapshot.
        // provider_name_for searches the pre-computed provider_names mapping.
        let provider_name = {
            let name = input.config.provider_name_for(&selected_model);
            if name == "unknown" {
                warn!(
                    model = %selected_model,
                    "provider_resolution: no provider found in config for model, using 'unknown'"
                );
            }
            name.to_string()
        };

        // Step 6: Push ProviderResolved.
        let _ = event_tx
            .send(PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                model_name: selected_model,
                model_id,
                provider_name,
                capabilities,
                max_tokens,
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{NullEventLog, collect_events, make_test_input, make_test_services};
    use pretty_assertions::assert_eq;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    /// Run the activity with the given input and return all emitted events.
    async fn run_provider_resolution(input: ActivityInput) -> Vec<PipelineEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ProviderResolutionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        collect_events(&mut rx)
    }

    /// Build a test input with the given metadata.
    fn make_input_with_metadata(metadata: serde_json::Value) -> ActivityInput {
        let mut input = make_test_input();
        input.metadata = metadata;
        input
    }

    // ── Name ────────────────────────────────────────────────────────────────

    #[test]
    fn provider_resolution_name() {
        assert_eq!(
            ProviderResolutionActivity::new().name(),
            "provider_resolution"
        );
    }

    // ── Happy path: emits ProviderResolved ────────────────────────────────────

    #[tokio::test]
    async fn provider_resolution_emits_provider_resolved() {
        // The test config has "stub-model" registered in StubProviderServiceV2.
        let input = make_input_with_metadata(serde_json::json!({ "selected_model": "stub-model" }));
        let events = run_provider_resolution(input).await;

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Started { name }) if name == "provider_resolution")
            ),
            "expected Activity(Started)"
        );

        assert!(
            events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ProviderResolved { .. })
            )),
            "expected Selection(ProviderResolved) event"
        );

        assert!(
            events.iter().any(
                |e| matches!(e, PipelineEvent::Activity(ActivityEvent::Completed { name, .. }) if name == "provider_resolution")
            ),
            "expected Activity(Completed)"
        );

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "did not expect Activity(Failed)"
        );
    }

    // ── ProviderResolved carries correct model_id ─────────────────────────────

    #[tokio::test]
    async fn provider_resolved_has_correct_model_id() {
        let input = make_input_with_metadata(serde_json::json!({ "selected_model": "stub-model" }));
        let events = run_provider_resolution(input).await;

        let resolved = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                model_name,
                model_id,
                provider_name,
                capabilities,
                max_tokens,
            }) = e
            {
                Some((
                    model_name.clone(),
                    model_id.clone(),
                    provider_name.clone(),
                    capabilities.clone(),
                    *max_tokens,
                ))
            } else {
                None
            }
        });

        let (model_name, model_id, provider_name, capabilities, max_tokens) =
            resolved.expect("Selection(ProviderResolved) must be present");

        assert_eq!(model_name, "stub-model");
        // StubProviderService::model_id("stub-model") returns "stub-model-v1"
        assert_eq!(model_id, "stub-model-v1");
        // Provider name is "anthropic" from the test config
        assert_eq!(provider_name, "anthropic");
        // StubProviderService registers "chat_completions" for "stub-model"
        assert!(
            capabilities.contains(&"chat_completions".to_string()),
            "expected chat_completions in capabilities"
        );
        // StubProviderService::max_tokens_for("stub-model") returns 4096
        assert_eq!(max_tokens, 4096);
    }

    // ── Fail-closed: unknown model_id returns None ────────────────────────────

    #[tokio::test]
    async fn provider_resolution_fails_when_model_id_none() {
        // "nonexistent-model" is not in StubProviderService, so model_id returns None.
        let input =
            make_input_with_metadata(serde_json::json!({ "selected_model": "nonexistent-model" }));
        let events = run_provider_resolution(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) when model_id returns None"
        );

        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ProviderResolved { .. })
            )),
            "Selection(ProviderResolved) must not be pushed when model_id is None"
        );
    }

    // ── Fail-closed: ActivityFailed is not retryable ──────────────────────────

    #[tokio::test]
    async fn provider_resolution_activity_failed_not_retryable() {
        let input =
            make_input_with_metadata(serde_json::json!({ "selected_model": "nonexistent-model" }));
        let events = run_provider_resolution(input).await;

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })));

        if let Some(PipelineEvent::Activity(ActivityEvent::Failed { retryable, .. })) = failed {
            assert!(
                !retryable,
                "provider resolution failures must not be retryable"
            );
        } else {
            panic!("expected Activity(Failed)");
        }
    }

    // ── Fail-closed: missing metadata ────────────────────────────────────────

    #[tokio::test]
    async fn provider_resolution_fails_when_metadata_missing_selected_model() {
        // Metadata is Null — no selected_model key.
        let input = make_input_with_metadata(serde_json::Value::Null);
        let events = run_provider_resolution(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) when selected_model is missing from metadata"
        );

        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ProviderResolved { .. })
            )),
            "Selection(ProviderResolved) must not be pushed when metadata is missing"
        );
    }

    // ── Fail-closed: empty model name in metadata ─────────────────────────────

    #[tokio::test]
    async fn provider_resolution_fails_when_selected_model_empty_string() {
        let input = make_input_with_metadata(serde_json::json!({ "selected_model": "" }));
        let events = run_provider_resolution(input).await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) when selected_model is empty string"
        );
    }

    // ── Cancellation ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn provider_resolution_handles_cancellation() {
        let input = make_input_with_metadata(serde_json::json!({ "selected_model": "stub-model" }));
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancel

        let services = make_test_services();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ProviderResolutionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);

        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. }))),
            "expected Activity(Failed) on cancellation"
        );

        assert!(
            !events.iter().any(|e| matches!(
                e,
                PipelineEvent::Selection(SelectionEvent::ProviderResolved { .. })
            )),
            "Selection(ProviderResolved) must not be pushed when cancelled"
        );
    }

    // ── Capabilities mapping ──────────────────────────────────────────────────

    #[tokio::test]
    async fn provider_resolved_capabilities_are_strings() {
        let input = make_input_with_metadata(serde_json::json!({ "selected_model": "stub-model" }));
        let events = run_provider_resolution(input).await;

        let caps = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                capabilities, ..
            }) = e
            {
                Some(capabilities.clone())
            } else {
                None
            }
        });

        let caps = caps.expect("Selection(ProviderResolved) must be present");
        // All entries must be non-empty strings.
        for cap in &caps {
            assert!(!cap.is_empty(), "capability string must not be empty");
        }
        // The event is serializable (capabilities are Vec<String>).
        let json = serde_json::to_string(&caps).expect("capabilities must serialize");
        let back: Vec<String> = serde_json::from_str(&json).expect("capabilities must deserialize");
        assert_eq!(back, caps);
    }

    // ── provider_name_for via ResolvedConfig ──────────────────────────────────

    #[test]
    fn resolved_config_provider_name_for_known_model() {
        // Verify the pre-computed provider_names mapping finds the correct provider.
        let config: weft_core::WeftConfig = toml::from_str(
            r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are helpful."

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "claude-opus"
  model = "claude-opus-4-20250514"
  examples = ["write code"]
"#,
        )
        .expect("test config must parse");

        let resolved = weft_core::ResolvedConfig::from_operator(&config);
        assert_eq!(resolved.provider_name_for("claude-opus"), "anthropic");
    }

    #[test]
    fn resolved_config_provider_name_for_missing_model_returns_unknown() {
        let config: weft_core::WeftConfig = toml::from_str(
            r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "claude-opus"
  model = "claude-opus-4-20250514"
  examples = ["write code"]
"#,
        )
        .expect("test config must parse");

        let resolved = weft_core::ResolvedConfig::from_operator(&config);
        assert_eq!(resolved.provider_name_for("nonexistent-model"), "unknown");
    }

    // ── Default capabilities when none registered ─────────────────────────────

    #[tokio::test]
    async fn provider_resolved_default_capabilities_when_none_registered() {
        use crate::test_support::MockServiceLocator;
        use std::collections::HashSet;
        use std::sync::Arc;

        // Build a custom provider service where the model has no capabilities registered.
        struct NoCapProviderService {
            provider: Arc<dyn weft_llm::Provider>,
        }

        impl weft_llm::ProviderService for NoCapProviderService {
            fn get(&self, _name: &str) -> &Arc<dyn weft_llm::Provider> {
                &self.provider
            }

            fn model_id(&self, name: &str) -> Option<&str> {
                if name == "stub-model" {
                    Some("stub-model-v1")
                } else {
                    None
                }
            }

            fn max_tokens_for(&self, name: &str) -> Option<u32> {
                if name == "stub-model" {
                    Some(4096)
                } else {
                    None
                }
            }

            fn default_provider(&self) -> &Arc<dyn weft_llm::Provider> {
                &self.provider
            }

            fn default_name(&self) -> &str {
                "stub-model"
            }

            fn models_with_capability(
                &self,
                _capability: &weft_llm::Capability,
            ) -> &HashSet<String> {
                // Return a reference to a leaked empty set for simplicity.
                Box::leak(Box::new(HashSet::new()))
            }

            fn model_has_capability(
                &self,
                _model_name: &str,
                _capability: &weft_llm::Capability,
            ) -> bool {
                false
            }

            fn model_capabilities(
                &self,
                _model_name: &str,
            ) -> Option<&HashSet<weft_llm::Capability>> {
                // None means no capabilities registered — activity should default.
                None
            }
        }

        // Build a stub provider.
        struct StubProv;
        #[async_trait::async_trait]
        impl weft_llm::Provider for StubProv {
            async fn execute(
                &self,
                _req: weft_llm::ProviderRequest,
            ) -> Result<weft_llm::ProviderResponse, weft_llm::ProviderError> {
                Err(weft_llm::ProviderError::Unsupported("stub".to_string()))
            }
            fn name(&self) -> &str {
                "stub"
            }
        }

        let config: weft_core::WeftConfig = toml::from_str(
            r#"
[server]
bind_address = "0.0.0.0:8080"
[gateway]
system_prompt = "test"
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
  examples = ["test"]
"#,
        )
        .expect("config parse");

        let provider_arc: Arc<dyn weft_llm::Provider> = Arc::new(StubProv);
        let svc_impl = NoCapProviderService {
            provider: provider_arc,
        };

        let base = make_test_services();
        let services = MockServiceLocator {
            resolved_config: Arc::new(weft_core::ResolvedConfig::from_operator(&config)),
            providers: Arc::new(svc_impl),
            router: base.router,
            commands: base.commands,
            hooks: base.hooks,
            memory: None,
            request_end_semaphore: base.request_end_semaphore,
        };

        let input = make_input_with_metadata(serde_json::json!({ "selected_model": "stub-model" }));
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let event_log = NullEventLog;
        let exec_id = ExecutionId::new();

        let activity = ProviderResolutionActivity::new();
        activity
            .execute(&exec_id, input, &services, &event_log, tx, cancel)
            .await;

        let events = collect_events(&mut rx);
        let caps = events.iter().find_map(|e| {
            if let PipelineEvent::Selection(SelectionEvent::ProviderResolved {
                capabilities, ..
            }) = e
            {
                Some(capabilities.clone())
            } else {
                None
            }
        });

        let caps = caps.expect("Selection(ProviderResolved) must be present");
        assert_eq!(
            caps,
            vec!["chat_completions".to_string()],
            "should default to chat_completions when no capabilities registered"
        );
    }

    // ── FailureDetail enrichment ──────────────────────────────────────────────

    #[tokio::test]
    async fn provider_resolution_missing_model_failure_detail_is_model_not_found() {
        use pretty_assertions::assert_ne;

        // No selected_model in metadata triggers the wiring error path.
        let input = make_input_with_metadata(serde_json::json!({}));
        let events = run_provider_resolution(input).await;

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed) when model missing from metadata");

        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { detail, .. }) => {
                assert_ne!(
                    detail.error_code, "unknown",
                    "missing model failure must have non-unknown error_code"
                );
                assert_eq!(detail.error_code, "model_not_found");
                assert!(
                    detail.cause.is_some(),
                    "cause must be populated for model_not_found failure"
                );
                assert!(
                    detail.attempted.is_some(),
                    "attempted must be populated for model_not_found failure"
                );
                assert!(
                    detail.detail.get("model_name").is_some(),
                    "detail must include model_name"
                );
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }

    #[tokio::test]
    async fn provider_resolution_unknown_model_failure_detail_is_model_not_found() {
        use pretty_assertions::assert_ne;

        // A model name that doesn't exist in the provider registry.
        let input = make_input_with_metadata(
            serde_json::json!({ "selected_model": "nonexistent-model-xyz" }),
        );
        let events = run_provider_resolution(input).await;

        let failed = events
            .iter()
            .find(|e| matches!(e, PipelineEvent::Activity(ActivityEvent::Failed { .. })))
            .expect("expected Activity(Failed) for unknown model");

        match failed {
            PipelineEvent::Activity(ActivityEvent::Failed { detail, .. }) => {
                assert_ne!(
                    detail.error_code, "unknown",
                    "unknown model failure must have non-unknown error_code"
                );
                assert_eq!(detail.error_code, "model_not_found");
                assert!(
                    detail.cause.is_some(),
                    "cause must be populated for model_not_found failure"
                );
                // detail must include model_name
                let model_name = detail
                    .detail
                    .get("model_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                assert_eq!(
                    model_name, "nonexistent-model-xyz",
                    "detail.model_name must match the requested model"
                );
            }
            _ => panic!("expected Activity(Failed)"),
        }
    }
}
