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
    Activity, ActivityEvent, ActivityInput, EventLog, ExecutionId, PipelineEvent, SelectionEvent,
    ServiceLocator,
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

        // Step 5: Look up provider_name from config.
        // Config is the source of truth: iterate providers to find which one owns this model.
        let provider_name = find_provider_name(services.config(), &selected_model);

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

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the provider name for a model routing name by scanning config.
///
/// Iterates `config.router.providers` and returns the provider `name` for the
/// first entry containing a model whose `name` matches `selected_model`.
/// Returns `"unknown"` if no match found (config/wiring error).
fn find_provider_name(config: &weft_core::WeftConfig, selected_model: &str) -> String {
    for provider in &config.router.providers {
        if provider.models.iter().any(|m| m.name == selected_model) {
            return provider.name.clone();
        }
    }
    warn!(
        model = %selected_model,
        "provider_resolution: no provider found in config for model, using 'unknown'"
    );
    "unknown".to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{NullEventLog, collect_events, make_test_input, make_test_services};
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

    // ── find_provider_name helper ─────────────────────────────────────────────

    #[test]
    fn find_provider_name_returns_provider_for_known_model() {
        // The test config has provider "anthropic" with model "stub-model".
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

        let name = find_provider_name(&config, "claude-opus");
        assert_eq!(name, "anthropic");
    }

    #[test]
    fn find_provider_name_returns_unknown_for_missing_model() {
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

        let name = find_provider_name(&config, "nonexistent-model");
        assert_eq!(name, "unknown");
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
            config: Arc::new(config),
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
}
