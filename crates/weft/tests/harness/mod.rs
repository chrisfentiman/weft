// Harness utilities are designed for use across multiple test binaries (pipeline_integration,
// command_loop, error_paths). Not all items are used by every binary that includes this module.
// Suppressing dead_code warnings at the module level avoids requiring each test binary to
// import every helper, while keeping the harness functions readily discoverable.
#![allow(dead_code)]

//! Shared test harness for `weft` integration tests.
//!
//! Provides builder helpers for constructing `WeftService` instances with
//! stub services, and HTTP test utilities for exercising the full axum router.
//!
//! # New types (Phase 1 additions)
//!
//! - [`SequencedProvider`] — mock LLM provider that returns a different response on each
//!   successive call. Required for multi-turn command loop tests.
//! - [`MockResponse`] — response variant for `SequencedProvider`: plain text or text with
//!   structured command calls.
//! - [`ScriptedCommandRegistry`] — mock command registry with configurable named commands
//!   and per-command outcomes. Required for tests that need multiple distinct commands.
//! - [`make_weft_service_with_event_log`] — builds a fully wired `WeftService` and returns
//!   the `Arc<InMemoryEventLog>` so tests can inspect the event trace after a request.
//! - [`make_router_with_event_log`] — same as above but returns the axum `Router`.
//! - [`test_config_with_gateway`] — builds a `WeftConfig` with a custom gateway system prompt
//!   and `max_command_iterations`.
//! - [`make_weft_service_with_config`] — builds a `WeftService` with custom config, provider,
//!   and command registry.
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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{Router, body::Body, http::Request};
use serde_json::Value;
use weft::WeftService;
use weft::server::build_router;
use weft_activities::{
    AssembleResponseActivity, CommandFormattingActivity, CommandSelectionActivity,
    ExecuteCommandActivity, GenerateActivity, HookActivity, ModelSelectionActivity,
    ProviderResolutionActivity, SamplingAdjustmentActivity, SystemPromptAssemblyActivity,
    ValidateActivity,
};
use weft_commands::{CommandError, CommandRegistry};
use weft_core::{
    ClassifierConfig, CommandCallContent, CommandDescription, CommandInvocation, CommandResult,
    CommandStub, ContentPart, DomainsConfig, GatewayConfig, HookEvent, ModelEntry, ProviderConfig,
    Role, RouterConfig, ServerConfig, Source, WeftConfig, WeftMessage, WireFormat,
};
use weft_llm::{
    Capability, Provider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
    TokenUsage,
};
use weft_reactor::{
    ActivityRegistry, Reactor, ReactorConfig,
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
pub fn build_test_activity_registry(services: &Services) -> ActivityRegistry {
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
    for event in [
        HookEvent::RequestStart,
        HookEvent::RequestEnd,
        HookEvent::PreResponse,
        HookEvent::PreToolUse,
        HookEvent::PostToolUse,
    ] {
        registry
            .register(Arc::new(HookActivity::new(
                event,
                Arc::clone(&services.hooks),
                Arc::clone(&services.request_end_semaphore),
            )))
            .unwrap();
    }
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

    let registry = Arc::new(build_test_activity_registry(&services));
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

// ── SequencedProvider ──────────────────────────────────────────────────────────

/// A single response slot for [`SequencedProvider`].
///
/// Use [`MockResponse::Text`] for simple single-turn responses. Use
/// [`MockResponse::WithCommands`] to produce structured `ContentPart::CommandCall`
/// entries in the provider response, bypassing the text-based slash-command parser.
///
/// # Why `WithCommands` is needed
///
/// `GenerateActivity::parse_response` receives an empty `known_commands` set today
/// (see `generate.rs` comment: "known_commands population is Phase 3"). Without the
/// command name in `known_commands`, slash-command text like `/web_search query: x`
/// is NOT parsed as a command invocation. `ContentPart::CommandCall` bypasses the
/// text parser entirely and is the reliable path for command-loop integration tests.
pub enum MockResponse {
    /// Plain text response. Produces a `ProviderResponse::ChatCompletion` with a
    /// single `ContentPart::Text` content part.
    Text(String),

    /// Text response with one or more structured command calls.
    ///
    /// If `text` is non-empty, a `ContentPart::Text` part is prepended.
    /// Each `(command_name, arguments)` tuple becomes a `ContentPart::CommandCall`.
    WithCommands {
        /// Preceding prose text (may be empty).
        text: String,
        /// Commands to invoke: `(command_name, arguments_json)`.
        commands: Vec<(String, serde_json::Value)>,
    },
}

/// A mock LLM provider that returns a different response for each successive call.
///
/// Responses are consumed in index order. If more calls are made than responses
/// provided, the last response is repeated (clamped). An empty response list
/// returns `ProviderError::Unsupported`.
///
/// # Usage
///
/// ```ignore
/// let provider = Arc::new(SequencedProvider::new(vec![
///     MockResponse::WithCommands {
///         text: "Searching…".into(),
///         commands: vec![("web_search".into(), json!({"query": "Rust"}))],
///     },
///     MockResponse::Text("Here is what I found.".into()),
/// ]));
/// let provider_clone = Arc::clone(&provider);
/// let (svc, log) = make_weft_service_with_event_log(provider, commands);
/// // … run request …
/// assert_eq!(provider_clone.call_count(), 2);
/// ```
pub struct SequencedProvider {
    responses: Vec<MockResponse>,
    /// Monotonically increasing call index. `Provider::execute` takes `&self`, so
    /// atomic is the correct tool here — no lock contention, no poisoning risk.
    call_index: AtomicU32,
    /// Last `ProviderRequest` received. Stored behind a `Mutex` because
    /// `ProviderRequest` is not `Copy` and must be owned. Tests read this once
    /// after the request completes, so contention is negligible.
    last_request: Mutex<Option<ProviderRequest>>,
}

impl SequencedProvider {
    /// Create a provider with the given response sequence.
    ///
    /// On overflow (more calls than responses), the last response is repeated.
    /// An empty `responses` list causes every call to return `ProviderError::Unsupported`.
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses,
            call_index: AtomicU32::new(0),
            last_request: Mutex::new(None),
        }
    }

    /// Create a provider that returns the same text on every call.
    ///
    /// Equivalent to `SequencedProvider::new(vec![MockResponse::Text(text.into())])` but
    /// clamping means the single response is returned for any number of calls.
    pub fn single(text: impl Into<String>) -> Self {
        Self::new(vec![MockResponse::Text(text.into())])
    }

    /// Return the number of times `execute` has been called.
    pub fn call_count(&self) -> u32 {
        self.call_index.load(Ordering::Relaxed)
    }

    /// Return the last `ProviderRequest` received, if any.
    ///
    /// Returns `None` if `execute` has not been called yet.
    pub fn last_request(&self) -> Option<ProviderRequest> {
        self.last_request
            .lock()
            .expect("last_request lock poisoned")
            .clone()
    }
}

#[async_trait]
impl Provider for SequencedProvider {
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        // Store the request for post-call assertions.
        {
            let mut guard = self
                .last_request
                .lock()
                .expect("last_request lock poisoned");
            *guard = Some(request.clone());
        }

        // Atomically advance the call counter and determine which response to return.
        let index = self.call_index.fetch_add(1, Ordering::Relaxed) as usize;

        if self.responses.is_empty() {
            return Err(ProviderError::Unsupported(
                "SequencedProvider has no responses configured".to_string(),
            ));
        }

        // Clamp: if more calls than responses, repeat the last one.
        let clamped = index.min(self.responses.len() - 1);
        let response = &self.responses[clamped];

        let content = match response {
            MockResponse::Text(text) => vec![ContentPart::Text(text.clone())],
            MockResponse::WithCommands { text, commands } => {
                let mut parts = Vec::new();
                if !text.is_empty() {
                    parts.push(ContentPart::Text(text.clone()));
                }
                for (name, arguments) in commands {
                    // CommandCallContent.arguments_json is a JSON *string*, not a Value.
                    let arguments_json =
                        serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string());
                    parts.push(ContentPart::CommandCall(CommandCallContent {
                        command: name.clone(),
                        arguments_json,
                    }));
                }
                parts
            }
        };

        Ok(ProviderResponse::ChatCompletion {
            message: WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: None,
                content,
                delta: false,
                message_index: 0,
            },
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        })
    }

    fn name(&self) -> &str {
        "sequenced-provider"
    }
}

// ProviderRequest must be Clone for last_request storage.
// It already derives Clone per its definition in weft_llm::provider.

// ── SharedSequencedProvider ────────────────────────────────────────────────────
//
// Tests that assert on `call_count()` or `last_request()` after the request
// completes need to share ownership of the `SequencedProvider` with the harness.
// The orphan rule prevents `impl Provider for Arc<SequencedProvider>`, so we
// use a thin newtype wrapper that implements `Provider` by delegating to the
// inner `Arc`.
//
// # Usage
//
// ```ignore
// let shared = SharedSequencedProvider::new(SequencedProvider::new(responses));
// let handle = shared.clone(); // keep for assertions
// let (svc, log) = make_weft_service_with_event_log(shared, commands);
// // … run request …
// assert_eq!(handle.call_count(), 2);
// ```

/// A newtype that wraps `Arc<SequencedProvider>` and implements `Provider`.
///
/// Needed because the orphan rule prevents implementing an external trait
/// (`Provider`) for an external generic type (`Arc<_>`). The newtype is local
/// to the test harness, so the impl is legal.
///
/// Use `SharedSequencedProvider::clone()` to retain a handle for post-call
/// assertions on `call_count()` and `last_request()`.
#[derive(Clone)]
pub struct SharedSequencedProvider(Arc<SequencedProvider>);

impl SharedSequencedProvider {
    /// Wrap a `SequencedProvider` in a shared handle.
    pub fn new(inner: SequencedProvider) -> Self {
        Self(Arc::new(inner))
    }

    /// Return the number of times `execute` has been called.
    pub fn call_count(&self) -> u32 {
        self.0.call_count()
    }

    /// Return the last `ProviderRequest` received, if any.
    pub fn last_request(&self) -> Option<ProviderRequest> {
        self.0.last_request()
    }
}

#[async_trait]
impl Provider for SharedSequencedProvider {
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        self.0.execute(request).await
    }

    fn name(&self) -> &str {
        self.0.name()
    }
}

// ── ScriptedCommandRegistry ───────────────────────────────────────────────────

/// A single command entry in a [`ScriptedCommandRegistry`].
struct ScriptedCommand {
    stub: CommandStub,
    description: CommandDescription,
    result: CommandResult,
}

/// A command registry with configurable named commands and pre-set outcomes.
///
/// Unlike [`weft_commands::test_support::StubCommandRegistry`] which has one hardcoded
/// `test_command`, `ScriptedCommandRegistry` supports arbitrary commands with distinct
/// names, descriptions, and outputs. This lets integration tests verify that the
/// correct command was invoked and that its specific output was injected back into
/// the conversation.
///
/// # Usage
///
/// ```ignore
/// // All commands succeed:
/// let registry = ScriptedCommandRegistry::new(vec![
///     ("web_search", "Search the web", "Results: Rust async guide"),
///     ("calculator", "Evaluate math", "42"),
/// ]);
///
/// // One command fails (success: false), others succeed:
/// let registry = ScriptedCommandRegistry::with_failing_command(
///     vec![("search", "Search", "")],
///     "search",
///     "service unavailable",
/// );
/// ```
pub struct ScriptedCommandRegistry {
    commands: Vec<ScriptedCommand>,
}

impl ScriptedCommandRegistry {
    /// Create a registry from a list of `(name, description, output)` tuples.
    ///
    /// Every command returns `CommandResult { success: true, output }`.
    pub fn new(commands: Vec<(&str, &str, &str)>) -> Self {
        let scripted = commands
            .into_iter()
            .map(|(name, desc, output)| ScriptedCommand {
                stub: CommandStub {
                    name: name.to_string(),
                    description: desc.to_string(),
                },
                description: CommandDescription {
                    name: name.to_string(),
                    description: desc.to_string(),
                    usage: format!("/{name}"),
                    parameters_schema: None,
                },
                result: CommandResult {
                    command_name: name.to_string(),
                    success: true,
                    output: output.to_string(),
                    error: None,
                },
            })
            .collect();
        Self { commands: scripted }
    }

    /// Create a registry where one named command returns `success: false`.
    ///
    /// `failing_name` must appear in `commands`. All other commands succeed normally.
    /// The failing command returns `CommandResult { success: false, error: Some(error_msg) }`.
    ///
    /// This models a logical command failure (e.g., "service unavailable"), which the pipeline
    /// injects as context and continues — as opposed to a `CommandError` (infrastructure failure),
    /// which would abort the current command execution attempt.
    pub fn with_failing_command(
        commands: Vec<(&str, &str, &str)>,
        failing_name: &str,
        error_msg: &str,
    ) -> Self {
        let scripted = commands
            .into_iter()
            .map(|(name, desc, output)| {
                let (success, out, err) = if name == failing_name {
                    (false, String::new(), Some(error_msg.to_string()))
                } else {
                    (true, output.to_string(), None)
                };
                ScriptedCommand {
                    stub: CommandStub {
                        name: name.to_string(),
                        description: desc.to_string(),
                    },
                    description: CommandDescription {
                        name: name.to_string(),
                        description: desc.to_string(),
                        usage: format!("/{name}"),
                        parameters_schema: None,
                    },
                    result: CommandResult {
                        command_name: name.to_string(),
                        success,
                        output: out,
                        error: err,
                    },
                }
            })
            .collect();
        Self { commands: scripted }
    }
}

#[async_trait]
impl CommandRegistry for ScriptedCommandRegistry {
    async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
        Ok(self.commands.iter().map(|c| c.stub.clone()).collect())
    }

    async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
        self.commands
            .iter()
            .find(|c| c.stub.name == name)
            .map(|c| c.description.clone())
            .ok_or_else(|| CommandError::NotFound(name.to_string()))
    }

    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, CommandError> {
        self.commands
            .iter()
            .find(|c| c.stub.name == invocation.name)
            .map(|c| c.result.clone())
            .ok_or_else(|| CommandError::NotFound(invocation.name.clone()))
    }
}

// ── Extended builder functions ─────────────────────────────────────────────────

/// Build a `WeftConfig` with a custom gateway system prompt and `max_command_iterations`.
///
/// All other settings use the [`test_config`] defaults.
pub fn test_config_with_gateway(
    system_prompt: &str,
    max_command_iterations: u32,
) -> Arc<WeftConfig> {
    let base = test_config();
    Arc::new(WeftConfig {
        gateway: GatewayConfig {
            system_prompt: system_prompt.to_string(),
            max_command_iterations,
            request_timeout_secs: base.gateway.request_timeout_secs,
        },
        server: base.server.clone(),
        router: base.router.clone(),
        tool_registry: base.tool_registry.clone(),
        memory: base.memory.clone(),
        hooks: base.hooks.clone(),
        max_pre_response_retries: base.max_pre_response_retries,
        request_end_concurrency: base.request_end_concurrency,
        event_log: base.event_log.clone(),
    })
}

/// Build a fully wired `Arc<WeftService>` with a custom provider and command registry.
///
/// Returns both the service and the `Arc<InMemoryEventLog>` so tests can call
/// `all_events()` to inspect the full event trace after a request completes.
///
/// Uses the standard [`test_config`], [`StubRouter`], and `HookRegistry::empty()`.
pub fn make_weft_service_with_event_log(
    llm: impl Provider + 'static,
    commands: impl CommandRegistry + 'static,
) -> (
    Arc<WeftService>,
    Arc<weft_eventlog_memory::InMemoryEventLog>,
) {
    let config = test_config();
    make_weft_service_with_config(config, llm, commands)
}

/// Build a fully wired axum `Router` with a custom provider and command registry.
///
/// Returns both the router and the `Arc<InMemoryEventLog>` so tests can inspect
/// the event trace after a request completes.
pub fn make_router_with_event_log(
    llm: impl Provider + 'static,
    commands: impl CommandRegistry + 'static,
) -> (Router, Arc<weft_eventlog_memory::InMemoryEventLog>) {
    let (svc, log) = make_weft_service_with_event_log(llm, commands);
    (build_router(svc), log)
}

/// Build a fully wired `Arc<WeftService>` with a custom config, provider, and command registry.
///
/// Returns both the service and the `Arc<InMemoryEventLog>`. Use this when the test needs
/// to override gateway settings (system prompt, iteration limits) via [`test_config_with_gateway`].
pub fn make_weft_service_with_config(
    config: Arc<WeftConfig>,
    llm: impl Provider + 'static,
    commands: impl CommandRegistry + 'static,
) -> (
    Arc<WeftService>,
    Arc<weft_eventlog_memory::InMemoryEventLog>,
) {
    let provider_registry = make_provider_registry(llm);

    let event_log = Arc::new(weft_eventlog_memory::InMemoryEventLog::new());
    let event_log_dyn: Arc<dyn weft_reactor::event_log::EventLog> = Arc::clone(&event_log) as _;

    let services = Arc::new(Services {
        config: Arc::clone(&config),
        providers: provider_registry as Arc<dyn weft_llm::ProviderService + Send + Sync>,
        router: Arc::new(StubRouter) as Arc<dyn weft_router::SemanticRouter + Send + Sync>,
        commands: Arc::new(commands) as Arc<dyn CommandRegistry + Send + Sync>,
        memory: None,
        hooks: Arc::new(weft_hooks::HookRegistry::empty())
            as Arc<dyn weft_hooks::HookRunner + Send + Sync>,
        reactor_handle: std::sync::OnceLock::new(),
        request_end_semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
    });

    let registry = Arc::new(build_test_activity_registry(&services));
    let reactor_config = build_test_reactor_config(&config);

    let reactor = Arc::new(
        Reactor::new(
            Arc::clone(&services),
            event_log_dyn,
            registry,
            &reactor_config,
        )
        .expect("test reactor must construct"),
    );

    let handle = Arc::new(ReactorHandle::new(Arc::clone(&reactor)));
    services
        .reactor_handle
        .set(handle)
        .expect("OnceLock must be unset");

    let svc = Arc::new(WeftService::new(reactor, config));
    (svc, event_log)
}
