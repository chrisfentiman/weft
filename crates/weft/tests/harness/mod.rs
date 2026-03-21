//! Shared test harness for `weft` integration tests.
//!
//! Provides builder helpers for constructing `WeftService` instances with
//! stub services, and HTTP test utilities for exercising the full axum router.
//!
//! # Usage
//!
//! ```ignore
//! use harness::{make_router, make_weft_service, post_json};
//!
//! let router = make_router(StubProvider::new("Hello!"));
//! let (status, resp) = post_json(router, body).await;
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use axum::{Router, body::Body, http::Request};
use serde_json::Value;
use weft::WeftService;
use weft::server::build_router;
use weft_commands::CommandRegistry;
use weft_core::{
    ClassifierConfig, ContentPart, DomainsConfig, GatewayConfig, HookEvent, ModelEntry,
    ProviderConfig, Role, RouterConfig, ServerConfig, Source, WeftConfig, WeftMessage, WireFormat,
};
use weft_llm::{
    Capability, Provider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
    TokenUsage,
};
use weft_reactor::{
    ActivityRegistry, Reactor, ReactorConfig,
    activities::{
        AssembleResponseActivity, CommandFormattingActivity, CommandSelectionActivity,
        ExecuteCommandActivity, GenerateActivity, HookActivity, ModelSelectionActivity,
        ProviderResolutionActivity, SamplingAdjustmentActivity, SystemPromptAssemblyActivity,
        ValidateActivity,
    },
    config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, RetryPolicy},
    services::{ReactorHandle, Services},
};
use weft_router::test_support::StubRouter;

// ── Local test providers ───────────────────────────────────────────────────────

/// A provider that returns a fixed text response with known usage values.
///
/// Returns `TokenUsage { prompt_tokens: 5, completion_tokens: 3 }`, which
/// integration tests use to verify usage propagation through the response path.
pub struct TestProvider {
    response_text: String,
}

impl TestProvider {
    /// Construct a provider returning the given text with (5 prompt, 3 completion) tokens.
    pub fn ok(s: &str) -> Self {
        Self {
            response_text: s.to_string(),
        }
    }
}

#[async_trait]
impl Provider for TestProvider {
    async fn execute(&self, _request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        Ok(ProviderResponse::ChatCompletion {
            message: WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: None,
                content: vec![ContentPart::Text(self.response_text.clone())],
                delta: false,
                message_index: 0,
            },
            usage: Some(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 3,
            }),
        })
    }

    fn name(&self) -> &str {
        "test-provider"
    }
}

// ── Builder helpers ────────────────────────────────────────────────────────────

/// Build the standard test `WeftConfig`.
pub fn test_config() -> Arc<WeftConfig> {
    Arc::new(WeftConfig {
        server: ServerConfig {
            bind_address: "127.0.0.1:0".to_string(),
        },
        router: RouterConfig {
            classifier: ClassifierConfig {
                model_path: "models/test.onnx".to_string(),
                tokenizer_path: "models/tokenizer.json".to_string(),
                threshold: 0.0,
                max_commands: 20,
            },
            default_model: Some("test-model".to_string()),
            providers: vec![ProviderConfig {
                name: "test-provider".to_string(),
                wire_format: WireFormat::Anthropic,
                api_key: "test-key".to_string(),
                base_url: None,
                wire_script: None,
                models: vec![ModelEntry {
                    name: "test-model".to_string(),
                    model: "claude-test".to_string(),
                    max_tokens: 1024,
                    examples: vec!["test query".to_string()],
                    capabilities: vec!["chat_completions".to_string()],
                }],
            }],
            skip_tools_when_unnecessary: true,
            domains: DomainsConfig::default(),
        },
        tool_registry: None,
        memory: None,
        hooks: vec![],
        max_pre_response_retries: 2,
        request_end_concurrency: 64,
        event_log: None,
        gateway: GatewayConfig {
            system_prompt: "You are a test assistant.".to_string(),
            max_command_iterations: 10,
            request_timeout_secs: 30,
        },
    })
}

/// Build a `ProviderRegistry` backed by the given provider.
pub fn make_provider_registry(llm: impl Provider + 'static) -> Arc<ProviderRegistry> {
    let mut providers = HashMap::new();
    providers.insert(
        "test-model".to_string(),
        Arc::new(llm) as Arc<dyn weft_llm::Provider>,
    );
    let mut model_ids = HashMap::new();
    model_ids.insert("test-model".to_string(), "claude-test".to_string());
    let mut max_tokens = HashMap::new();
    max_tokens.insert("test-model".to_string(), 1024u32);
    let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
    caps.insert(
        "test-model".to_string(),
        [Capability::new(Capability::CHAT_COMPLETIONS)]
            .into_iter()
            .collect(),
    );
    Arc::new(ProviderRegistry::new(
        providers,
        model_ids,
        max_tokens,
        caps,
        "test-model".to_string(),
    ))
}

/// Build an `ActivityRegistry` with all standard activities registered.
pub fn build_test_activity_registry() -> ActivityRegistry {
    let mut registry = ActivityRegistry::new();
    registry.register(Arc::new(ValidateActivity)).unwrap();
    registry.register(Arc::new(ModelSelectionActivity)).unwrap();
    registry
        .register(Arc::new(CommandSelectionActivity))
        .unwrap();
    registry
        .register(Arc::new(ProviderResolutionActivity))
        .unwrap();
    registry
        .register(Arc::new(SystemPromptAssemblyActivity))
        .unwrap();
    registry
        .register(Arc::new(CommandFormattingActivity))
        .unwrap();
    registry
        .register(Arc::new(SamplingAdjustmentActivity))
        .unwrap();
    registry.register(Arc::new(GenerateActivity)).unwrap();
    registry.register(Arc::new(ExecuteCommandActivity)).unwrap();
    registry
        .register(Arc::new(AssembleResponseActivity))
        .unwrap();
    registry
        .register(Arc::new(HookActivity::new(HookEvent::RequestStart)))
        .unwrap();
    registry
        .register(Arc::new(HookActivity::new(HookEvent::RequestEnd)))
        .unwrap();
    registry
        .register(Arc::new(HookActivity::new(HookEvent::PreResponse)))
        .unwrap();
    registry
        .register(Arc::new(HookActivity::new(HookEvent::PreToolUse)))
        .unwrap();
    registry
        .register(Arc::new(HookActivity::new(HookEvent::PostToolUse)))
        .unwrap();
    registry
}

/// Build a `ReactorConfig` suitable for HTTP integration tests.
///
/// Uses `max_retries: 0` to avoid multi-second backoff delays in tests.
pub fn build_test_reactor_config(config: &WeftConfig) -> ReactorConfig {
    ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![
                ActivityRef::Name("validate".to_string()),
                ActivityRef::Name("hook_request_start".to_string()),
                ActivityRef::Name("model_selection".to_string()),
                ActivityRef::Name("command_selection".to_string()),
                ActivityRef::Name("provider_resolution".to_string()),
                ActivityRef::Name("system_prompt_assembly".to_string()),
                ActivityRef::Name("command_formatting".to_string()),
                ActivityRef::Name("sampling_adjustment".to_string()),
            ],
            post_loop: vec![
                ActivityRef::Name("assemble_response".to_string()),
                ActivityRef::Name("hook_request_end".to_string()),
            ],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                // No retries in tests: retries add multi-second backoff delays
                // and are tested in weft_reactor unit tests, not here.
                retry: Some(RetryPolicy {
                    max_retries: 0,
                    initial_backoff_ms: 0,
                    max_backoff_ms: 0,
                    backoff_multiplier: 1.0,
                }),
                timeout_secs: Some(config.gateway.request_timeout_secs),
                heartbeat_interval_secs: Some(15),
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 20,
            max_iterations: config.gateway.max_command_iterations,
            max_depth: 5,
            timeout_secs: config.gateway.request_timeout_secs,
            generation_timeout_secs: config.gateway.request_timeout_secs,
            command_timeout_secs: 10,
        },
    }
}

/// Build a fully wired `Arc<WeftService>` backed by the given provider.
///
/// Uses `StubRouter` and `StubCommandRegistry` from the trait crates' test-support
/// modules. The provider determines what the LLM response looks like.
pub fn make_weft_service(llm: impl Provider + 'static) -> Arc<WeftService> {
    let config = test_config();

    let provider_registry = make_provider_registry(llm);

    let services = Arc::new(Services {
        config: Arc::clone(&config),
        providers: provider_registry as Arc<dyn weft_llm::ProviderService + Send + Sync>,
        router: Arc::new(StubRouter) as Arc<dyn weft_router::SemanticRouter + Send + Sync>,
        commands: Arc::new(weft_commands::test_support::StubCommandRegistry::new())
            as Arc<dyn CommandRegistry + Send + Sync>,
        memory: None,
        hooks: Arc::new(weft_hooks::HookRegistry::empty())
            as Arc<dyn weft_hooks::HookRunner + Send + Sync>,
        reactor_handle: std::sync::OnceLock::new(),
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    });

    let registry = Arc::new(build_test_activity_registry());
    let event_log: Arc<dyn weft_reactor::event_log::EventLog> =
        Arc::new(weft_eventlog_memory::InMemoryEventLog::new());
    let reactor_config = build_test_reactor_config(&config);

    let reactor = Arc::new(
        Reactor::new(Arc::clone(&services), event_log, registry, &reactor_config)
            .expect("test reactor must construct"),
    );

    // Wire ReactorHandle to break the circular dependency.
    let handle = Arc::new(ReactorHandle::new(Arc::clone(&reactor)));
    services
        .reactor_handle
        .set(handle)
        .expect("OnceLock must be unset");

    Arc::new(WeftService::new(reactor, config))
}

/// Build a fully wired axum `Router` backed by the given provider.
pub fn make_router(llm: impl Provider + 'static) -> Router {
    build_router(make_weft_service(llm))
}

/// POST a JSON body to `/v1/chat/completions` and return the (status, parsed body).
pub async fn post_json(router: Router, body: Value) -> (axum::http::StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = tower::ServiceExt::oneshot(router, req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}
