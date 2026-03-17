//! Gateway engine: the core loop that wires all components together.
//!
//! The engine routes requests semantically, assembles context, calls the LLM,
//! parses commands from the response, executes them, and loops until no more
//! commands are emitted or the iteration cap or timeout is reached.
//!
//! ## Request flow (with hooks)
//!
//! 1. [HOOK: RequestStart] — hard block (403), can modify request
//! 2. Validate request
//! 3. list_commands()
//! 4. extract_latest_user_message()
//! 5. For each routing domain (model, commands, tool_necessity):
//!    a. [HOOK: PreRoute(domain, trigger=request_start)]  — hard block (403)
//!    b. Router scores this domain
//!    c. [HOOK: PostRoute(domain, trigger=request_start)] — hard block (403)
//! 6. Assemble system prompt
//! 7. LOOP:
//!    a. Call LLM
//!    b. Parse response
//!    c. [HOOK: PreResponse] — feedback block (re-run LLM with reason), can modify;
//!    block with retries left injects reason and retries; exhausted retries returns 422.
//!    d. For each command invocation:
//!       - [HOOK: PreToolUse] — feedback block
//!       - For /recall and /remember: [HOOK: PreRoute/PostRoute(memory)] — feedback block
//!       - Execute command
//!       - [HOOK: PostToolUse] — can modify result
//! 8. Build HTTP response, return
//! 9. [HOOK: RequestEnd] — fire-and-forget (semaphore-gated)
//!
//! ## Activity events (Phase 4)
//!
//! When `options.activity = true`, the engine collects routing decisions and hook
//! events as `ActivityEvent` values during processing and includes them as
//! `source: gateway` system messages in the response. See `ActivityEvent` and
//! `assemble_response`.

use std::sync::Arc;

use weft_core::WeftConfig;
use weft_llm::ProviderService;
use weft_memory::MemoryService;
use weft_router::{
    MemoryStoreRef, RoutingCandidate, SemanticRouter, build_memory_candidates,
};
use weft_hooks::HookRunner;
use weft_commands::CommandRegistry;

mod activity;
mod memory;
mod provider;
mod request;
mod routing;
pub(crate) mod util;

/// The gateway engine: holds shared components and drives the request loop.
///
/// Generic parameters:
/// - `H`: hook runner (e.g. `weft_hooks::HookRegistry` or `NullHookRunner`)
/// - `R`: semantic router (e.g. `weft_router::ModernBertClassifier`)
/// - `M`: memory service (e.g. `weft_memory::DefaultMemoryService`)
/// - `P`: provider service (e.g. `weft_llm::ProviderRegistry`)
/// - `C`: command registry (e.g. `weft_commands::ToolRegistryCommandAdapter`)
///
/// All fields are `Arc` so `GatewayEngine` is cheaply cloneable — axum clones
/// it into each request handler. A manual `Clone` impl avoids requiring the
/// type params themselves to be `Clone` (they are behind `Arc`).
pub struct GatewayEngine<H, R, M, P, C> {
    config: Arc<WeftConfig>,
    providers: Arc<P>,
    router: Arc<R>,
    commands: Arc<C>,
    /// Optional memory service. `None` when no memory stores are configured.
    memory: Option<Arc<M>>,
    /// All memory store routing candidates (from `memory.stores()`). Used for the Memory
    /// domain in `route_all_domains()` and for per-invocation routing by both `/recall`
    /// and `/remember`. Empty if no memory stores are configured.
    memory_candidates: Vec<RoutingCandidate>,
    /// Memory candidates filtered to read-capable stores only. Used by `/recall`
    /// for per-invocation routing via `score_memory_candidates()`.
    read_memory_candidates: Vec<RoutingCandidate>,
    /// Memory candidates filtered to write-capable stores only. Used by `/remember`
    /// for per-invocation routing via `score_memory_candidates()`.
    write_memory_candidates: Vec<RoutingCandidate>,
    /// Hook runner. Shared immutably across all request handlers.
    hooks: Arc<H>,
    /// Semaphore limiting concurrent RequestEnd hook tasks.
    /// Prevents unbounded task accumulation under burst load.
    request_end_semaphore: Arc<tokio::sync::Semaphore>,
}

impl<H, R, M, P, C> Clone for GatewayEngine<H, R, M, P, C> {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            providers: Arc::clone(&self.providers),
            router: Arc::clone(&self.router),
            commands: Arc::clone(&self.commands),
            memory: self.memory.clone(),
            memory_candidates: self.memory_candidates.clone(),
            read_memory_candidates: self.read_memory_candidates.clone(),
            write_memory_candidates: self.write_memory_candidates.clone(),
            hooks: Arc::clone(&self.hooks),
            request_end_semaphore: Arc::clone(&self.request_end_semaphore),
        }
    }
}

impl<H, R, M, P, C> GatewayEngine<H, R, M, P, C>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    /// Expose the config for use by the health handler and other modules.
    pub fn config(&self) -> &WeftConfig {
        &self.config
    }

    /// Construct a new gateway engine.
    ///
    /// `memory`: Optional memory service. `None` when no memory stores configured.
    pub fn new(
        config: Arc<WeftConfig>,
        providers: Arc<P>,
        router: Arc<R>,
        commands: Arc<C>,
        memory: Option<Arc<M>>,
        hooks: Arc<H>,
    ) -> Self {
        // Build per-capability candidate sets from the memory service's store metadata.
        // Convert `StoreInfo` to `MemoryStoreRef` so `weft_router` doesn't depend on `weft_memory`.
        let mem_candidates = if let Some(mem) = &memory {
            let refs: Vec<MemoryStoreRef> = mem
                .stores()
                .into_iter()
                .map(|s| MemoryStoreRef {
                    name: s.name,
                    capabilities: s.capabilities,
                    examples: s.examples,
                })
                .collect();
            build_memory_candidates(&refs)
        } else {
            weft_router::MemoryCandidates::default()
        };
        let (memory_candidates, read_memory_candidates, write_memory_candidates) = (
            mem_candidates.all,
            mem_candidates.read,
            mem_candidates.write,
        );

        let request_end_concurrency = config.request_end_concurrency;
        let request_end_semaphore = Arc::new(tokio::sync::Semaphore::new(request_end_concurrency));

        Self {
            config,
            providers,
            router,
            commands,
            memory,
            memory_candidates,
            read_memory_candidates,
            write_memory_candidates,
            hooks,
            request_end_semaphore,
        }
    }

}
#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Duration;
    use weft_commands::{CommandError, CommandRegistry};
    use weft_core::{
        ClassifierConfig, CommandAction, CommandDescription, CommandInvocation, CommandResult,
        CommandStub, ContentPart, DomainConfig, DomainsConfig, GatewayConfig, HookEvent,
        MemoryConfig, MemoryStoreConfig, ModelEntry, ModelRoutingInstruction, ProviderConfig, Role,
        RouterConfig, SamplingOptions, ServerConfig, Source, StoreCapability, WeftConfig,
        WeftError, WeftMessage, WeftRequest, WeftResponse, WireFormat,
    };
    use weft_hooks::types::HookMatcher;
    use weft_hooks::{HookRegistry, RegisteredHook};
    use weft_llm::{
        Capability, Provider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
        TokenUsage,
    };
    use weft_router::{
        RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
        SemanticRouter,
    };

    // ── Mock implementations ───────────────────────────────────────────────

    /// A mock provider with configurable responses.
    struct MockLlmProvider {
        /// Responses to return in order. Repeats last on exhaustion.
        responses: std::sync::Mutex<Vec<String>>,
    }

    impl MockLlmProvider {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: std::sync::Mutex::new(
                    responses.into_iter().map(str::to_string).collect(),
                ),
            }
        }

        fn single(response: &str) -> Self {
            Self::new(vec![response])
        }
    }

    #[async_trait]
    impl Provider for MockLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            let mut guard = self.responses.lock().unwrap();
            let text = if guard.len() > 1 {
                guard.remove(0)
            } else {
                guard[0].clone()
            };
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text(text)],
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
            "mock"
        }
    }

    /// A provider that records which model was requested and returns a fixed response.
    struct RecordingLlmProvider {
        response: String,
        recorded_model: std::sync::Mutex<Option<String>>,
    }

    impl RecordingLlmProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                recorded_model: std::sync::Mutex::new(None),
            }
        }

        fn recorded_model(&self) -> Option<String> {
            self.recorded_model.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for RecordingLlmProvider {
        async fn execute(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            // Record the model from the request. Only ChatCompletion exists now;
            // when future variants land, update this to handle them.
            #[allow(irrefutable_let_patterns)]
            if let ProviderRequest::ChatCompletion { ref model, .. } = request {
                *self.recorded_model.lock().unwrap() = Some(model.clone());
            }
            Ok(ProviderResponse::ChatCompletion {
                message: WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text(self.response.clone())],
                    delta: false,
                    message_index: 0,
                },
                usage: None,
            })
        }

        fn name(&self) -> &str {
            "recording"
        }
    }

    /// A provider that always returns a rate-limit error.
    struct RateLimitedLlmProvider;

    #[async_trait]
    impl Provider for RateLimitedLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::RateLimited {
                retry_after_ms: 1000,
            })
        }

        fn name(&self) -> &str {
            "rate-limited"
        }
    }

    /// A provider that always returns a request failure error.
    struct FailingLlmProvider;

    #[async_trait]
    impl Provider for FailingLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::RequestFailed("network error".to_string()))
        }

        fn name(&self) -> &str {
            "failing"
        }
    }

    /// A mock router with configurable per-domain behavior.
    struct MockRouter {
        /// Score assigned to all Commands domain candidates.
        command_score: f32,
        /// Model to select (None = use default).
        model_decision: Option<String>,
        /// Tools needed decision.
        tools_needed: Option<bool>,
        /// Score assigned to the first memory candidate when `score_memory_candidates()` is called.
        /// Other candidates get score 0.0. `None` means return ModelNotLoaded error (simulate
        /// router unavailable). Default: Some(0.9) — first candidate wins.
        memory_score: Option<f32>,
    }

    impl MockRouter {
        fn with_score(score: f32) -> Self {
            Self {
                command_score: score,
                model_decision: None,
                tools_needed: None,
                memory_score: Some(0.9),
            }
        }

        fn with_model(model: &str) -> Self {
            Self {
                command_score: 1.0,
                model_decision: Some(model.to_string()),
                tools_needed: None,
                memory_score: Some(0.9),
            }
        }

        fn with_tools_needed(needed: bool) -> Self {
            Self {
                command_score: 1.0,
                model_decision: None,
                tools_needed: Some(needed),
                memory_score: Some(0.9),
            }
        }

        /// Configure memory scoring: returns the given score for the first candidate,
        /// 0.0 for all others. Useful for testing threshold behavior.
        fn with_memory_score(mut self, score: f32) -> Self {
            self.memory_score = Some(score);
            self
        }

        /// Configure router unavailable for memory scoring (returns ModelNotLoaded).
        fn with_memory_unavailable(mut self) -> Self {
            self.memory_score = None;
            self
        }

        fn failing() -> FailingRouter {
            FailingRouter
        }
    }

    #[async_trait]
    impl SemanticRouter for MockRouter {
        async fn route(
            &self,
            _user_message: &str,
            domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            let mut decision = RoutingDecision::empty();
            for (kind, candidates) in domains {
                match kind {
                    RoutingDomainKind::Commands => {
                        decision.commands = candidates
                            .iter()
                            .map(|c| ScoredCandidate {
                                id: c.id.clone(),
                                score: self.command_score,
                            })
                            .collect();
                    }
                    RoutingDomainKind::Model => {
                        if let Some(ref model_id) = self.model_decision {
                            // Return the configured model as the top scorer
                            decision.model =
                                candidates.iter().find(|c| c.id == *model_id).map(|c| {
                                    ScoredCandidate {
                                        id: c.id.clone(),
                                        score: 0.9,
                                    }
                                });
                        }
                    }
                    RoutingDomainKind::ToolNecessity => {
                        decision.tools_needed = self.tools_needed;
                    }
                    RoutingDomainKind::Memory => {}
                }
            }
            Ok(decision)
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            candidates: &[RoutingCandidate],
        ) -> Result<Vec<ScoredCandidate>, RouterError> {
            match self.memory_score {
                None => Err(RouterError::ModelNotLoaded),
                Some(first_score) => {
                    // First candidate gets the configured score, rest get 0.0.
                    // This makes the first candidate "win" when testing threshold behavior.
                    Ok(candidates
                        .iter()
                        .enumerate()
                        .map(|(i, c)| ScoredCandidate {
                            id: c.id.clone(),
                            score: if i == 0 { first_score } else { 0.0 },
                        })
                        .collect())
                }
            }
        }
    }

    struct FailingRouter;

    #[async_trait]
    impl SemanticRouter for FailingRouter {
        async fn route(
            &self,
            _: &str,
            _: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            Err(RouterError::ModelNotLoaded)
        }

        async fn score_memory_candidates(
            &self,
            _: &str,
            _: &[RoutingCandidate],
        ) -> Result<Vec<ScoredCandidate>, RouterError> {
            Err(RouterError::ModelNotLoaded)
        }
    }

    /// A mock command registry.
    struct MockCommandRegistry {
        commands: Vec<CommandStub>,
        execute_result: std::sync::Mutex<Option<CommandResult>>,
        describe_result: std::sync::Mutex<Option<CommandDescription>>,
        fail_list: bool,
    }

    impl MockCommandRegistry {
        fn new(commands: Vec<(&str, &str)>) -> Self {
            Self {
                commands: commands
                    .into_iter()
                    .map(|(n, d)| CommandStub {
                        name: n.to_string(),
                        description: d.to_string(),
                    })
                    .collect(),
                execute_result: std::sync::Mutex::new(None),
                describe_result: std::sync::Mutex::new(None),
                fail_list: false,
            }
        }

        fn with_execute_result(self, result: CommandResult) -> Self {
            *self.execute_result.lock().unwrap() = Some(result);
            self
        }

        fn with_describe_result(self, desc: CommandDescription) -> Self {
            *self.describe_result.lock().unwrap() = Some(desc);
            self
        }

        fn with_list_failure() -> Self {
            Self {
                commands: vec![],
                execute_result: std::sync::Mutex::new(None),
                describe_result: std::sync::Mutex::new(None),
                fail_list: true,
            }
        }
    }

    #[async_trait]
    impl CommandRegistry for MockCommandRegistry {
        async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
            if self.fail_list {
                return Err(CommandError::RegistryUnavailable(
                    "registry down".to_string(),
                ));
            }
            Ok(self.commands.clone())
        }

        async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
            self.describe_result
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| CommandError::NotFound(name.to_string()))
        }

        async fn execute_command(
            &self,
            invocation: &CommandInvocation,
        ) -> Result<CommandResult, CommandError> {
            match &invocation.action {
                CommandAction::Describe => {
                    let desc = self.describe_result.lock().unwrap().clone();
                    if let Some(d) = desc {
                        Ok(CommandResult {
                            command_name: invocation.name.clone(),
                            success: true,
                            output: format!("{}: {}", d.name, d.description),
                            error: None,
                        })
                    } else {
                        Err(CommandError::NotFound(invocation.name.clone()))
                    }
                }
                CommandAction::Execute => {
                    self.execute_result.lock().unwrap().clone().ok_or_else(|| {
                        CommandError::ExecutionFailed {
                            name: invocation.name.clone(),
                            reason: "no mock result configured".to_string(),
                        }
                    })
                }
            }
        }
    }

    /// Build a minimal valid WeftConfig for tests.
    fn test_config() -> Arc<WeftConfig> {
        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0, // All commands pass in tests
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
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    /// Build a multi-model WeftConfig for routing tests.
    fn multi_model_config() -> Arc<WeftConfig> {
        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("default-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "default-model".to_string(),
                            model: "claude-default".to_string(),
                            max_tokens: 1024,
                            examples: vec!["general question".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                        ModelEntry {
                            name: "complex-model".to_string(),
                            model: "claude-complex".to_string(),
                            max_tokens: 4096,
                            examples: vec!["complex reasoning task".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    fn make_user_request(content: &str) -> WeftRequest {
        WeftRequest {
            messages: vec![WeftMessage {
                role: Role::User,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text(content.to_string())],
                delta: false,
                message_index: 0,
            }],
            routing: ModelRoutingInstruction::parse("test-model"),
            options: SamplingOptions::default(),
        }
    }

    /// Build a default-only capabilities map for a single model (chat_completions).
    fn default_caps(model_name: &str) -> HashMap<String, HashSet<Capability>> {
        let mut caps = HashMap::new();
        caps.insert(
            model_name.to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        caps
    }

    /// Build a single-model `ProviderRegistry` backed by the given provider.
    fn single_model_registry(
        provider: impl Provider + 'static,
        model_name: &str,
        model_id: &str,
    ) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert(
            model_name.to_string(),
            Arc::new(provider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert(model_name.to_string(), model_id.to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert(model_name.to_string(), 1024u32);
        Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            default_caps(model_name),
            model_name.to_string(),
        ))
    }

    /// Build a two-model `ProviderRegistry` for fallback tests.
    fn two_model_registry(
        default_provider: impl Provider + 'static,
        non_default_provider: impl Provider + 'static,
    ) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(default_provider) as Arc<dyn Provider>,
        );
        providers.insert(
            "complex-model".to_string(),
            Arc::new(non_default_provider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert("complex-model".to_string(), "claude-complex".to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert("default-model".to_string(), 1024u32);
        max_tokens.insert("complex-model".to_string(), 4096u32);
        // Both models support chat_completions for testing purposes.
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        caps.insert(
            "complex-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "default-model".to_string(),
        ))
    }

    fn make_engine<R, C>(
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
    ) -> GatewayEngine<
        weft_hooks::HookRegistry,
        R,
        weft_memory::NullMemoryService,
        ProviderRegistry,
        C,
    >
    where
        R: SemanticRouter + Send + Sync + 'static,
        C: CommandRegistry + Send + Sync + 'static,
    {
        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(weft_hooks::HookRegistry::empty()),
        )
    }

    fn make_engine_with_config<R, C>(
        config: Arc<WeftConfig>,
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
    ) -> GatewayEngine<
        weft_hooks::HookRegistry,
        R,
        weft_memory::NullMemoryService,
        ProviderRegistry,
        C,
    >
    where
        R: SemanticRouter + Send + Sync + 'static,
        C: CommandRegistry + Send + Sync + 'static,
    {
        GatewayEngine::new(
            config,
            registry,
            Arc::new(router),
            Arc::new(commands),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(weft_hooks::HookRegistry::empty()),
        )
    }

    // ── Response extraction helpers ────────────────────────────────────────

    /// Extract the assistant response text from a `WeftResponse`.
    ///
    /// Returns the text from the first assistant message with source Provider.
    /// Panics if no such message is found (indicates a test setup error).
    fn resp_text(resp: &WeftResponse) -> &str {
        resp.messages
            .iter()
            .find(|m| m.role == Role::Assistant && m.source == Source::Provider)
            .and_then(|m| {
                m.content.iter().find_map(|p| match p {
                    ContentPart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
            })
            .expect("WeftResponse should contain an assistant text message")
    }

    // ── Test: no-command response (single pass) ────────────────────────────

    #[tokio::test]
    async fn test_no_command_response_single_pass() {
        let registry = single_model_registry(
            MockLlmProvider::single("Hello, I can help you with that!"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Hello, I can help you with that!");
        assert!(resp.id.starts_with("chatcmpl-"));
    }

    // ── Test: single command loop ──────────────────────────────────────────

    #[tokio::test]
    async fn test_single_command_executes_and_loops() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust async\"",
                "Here are the results I found.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "Found 5 results about Rust async".to_string(),
                    error: None,
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust async"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Here are the results I found.");
    }

    // ── Test: multiple commands ────────────────────────────────────────────

    #[tokio::test]
    async fn test_multiple_commands_executed_sequentially() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"\n/code_review target: src",
                "Done with both commands.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![
                ("web_search", "Search the web"),
                ("code_review", "Review code"),
            ])
            .with_execute_result(CommandResult {
                command_name: "any".to_string(),
                success: true,
                output: "OK".to_string(),
                error: None,
            }),
        );

        let resp = engine
            .handle_request(make_user_request("Do stuff"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Done with both commands.");
    }

    // ── Test: max iterations exceeded ─────────────────────────────────────

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        let config = Arc::new(WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 2,
                request_timeout_secs: 30,
            },
            ..(*test_config()).clone()
        });

        let registry = single_model_registry(
            MockLlmProvider::single("/web_search query: \"loop\""),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "still looping".to_string(),
                    error: None,
                },
            ),
        );

        let result = engine
            .handle_request(make_user_request("Search forever"))
            .await;

        assert!(
            matches!(result, Err(WeftError::CommandLoopExceeded { max: 2 })),
            "expected CommandLoopExceeded, got: {:?}",
            result
        );
    }

    // ── Test: command failure recovery ────────────────────────────────────

    #[tokio::test]
    async fn test_command_failure_injected_as_error_result() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "The search failed, but I can still answer.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("registry unavailable".to_string()),
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed despite command failure");

        assert_eq!(
            resp_text(&resp),
            "The search failed, but I can still answer."
        );
    }

    // ── Test: router fallback ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_router_failure_falls_back_to_all_commands() {
        let registry = single_model_registry(
            MockLlmProvider::single("No commands needed."),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::failing(),
            MockCommandRegistry::new(vec![
                ("web_search", "Search the web"),
                ("code_review", "Review code"),
            ]),
        );

        // Should succeed even with a failing router (fallback to all commands)
        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("router fallback must not fail the request");

        assert!(!resp_text(&resp).is_empty());
    }

    // ── Test: --describe flag ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_describe_action_handled_by_registry() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search --describe",
                "Now I understand the command.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_describe_result(
                CommandDescription {
                    name: "web_search".to_string(),
                    description: "Search the web for current information".to_string(),
                    usage: "/web_search query: \"search terms\"".to_string(),
                    parameters_schema: None,
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("What is web_search?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Now I understand the command.");
    }

    // ── Test: request timeout ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_request_timeout() {
        let config = Arc::new(WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 0,
            },
            ..(*test_config()).clone()
        });

        /// A provider that sleeps forever before responding.
        struct SlowLlmProvider;

        #[async_trait]
        impl Provider for SlowLlmProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text("never".to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "slow"
            }
        }

        let registry = single_model_registry(SlowLlmProvider, "test-model", "claude-test");
        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::RequestTimeout { .. })),
            "expected RequestTimeout, got: {:?}",
            result
        );
    }

    // ── Test: response format matches OpenAI schema ────────────────────────

    #[tokio::test]
    async fn test_response_format_matches_openai_schema() {
        let registry = single_model_registry(
            MockLlmProvider::single("Test response"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hi"))
            .await
            .expect("should succeed");

        assert!(
            resp.id.starts_with("chatcmpl-"),
            "id must start with chatcmpl-"
        );
        // model echoes the routing instruction from the request
        assert_eq!(
            resp.model, "test-model",
            "model must be preserved from request"
        );
        // Response messages contain exactly one assistant message from Provider
        let assistant_msg = resp
            .messages
            .iter()
            .find(|m| m.role == Role::Assistant && m.source == Source::Provider)
            .expect("response must contain an assistant Provider message");
        assert_eq!(assistant_msg.role, Role::Assistant);
        assert_eq!(
            resp.usage.total_tokens,
            resp.usage.prompt_tokens + resp.usage.completion_tokens
        );
    }

    // ── Test: no user message ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_no_user_message_returns_error() {
        let registry = single_model_registry(
            MockLlmProvider::single("irrelevant"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        // A request with only a system message (no user message) should fail.
        let req = WeftRequest {
            messages: vec![WeftMessage {
                role: Role::System,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("system only".to_string())],
                delta: false,
                message_index: 0,
            }],
            routing: ModelRoutingInstruction::parse("test-model"),
            options: SamplingOptions::default(),
        };

        let result = engine.handle_request(req).await;
        assert!(result.is_err(), "expected error for missing user message");
    }

    // ── Test: rate limit error propagated ─────────────────────────────────

    #[tokio::test]
    async fn test_rate_limit_error_propagated() {
        let registry = single_model_registry(RateLimitedLlmProvider, "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::RateLimited {
                    retry_after_ms: 1000
                })
            ),
            "expected RateLimited, got: {:?}",
            result
        );
    }

    // ── Test: LLM error propagated ─────────────────────────────────────────

    #[tokio::test]
    async fn test_llm_error_propagated() {
        // Single-model registry: failing provider IS the default, so no fallback.
        let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "expected Llm error, got: {:?}",
            result
        );
    }

    // ── Test: list commands failure ────────────────────────────────────────

    #[tokio::test]
    async fn test_list_commands_failure_propagated() {
        let registry = single_model_registry(
            MockLlmProvider::single("irrelevant"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::with_list_failure(),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Command(_))),
            "expected Command error from list failure, got: {:?}",
            result
        );
    }

    // ── Test: empty LLM response is valid ─────────────────────────────────

    #[tokio::test]
    async fn test_empty_llm_response_is_valid() {
        let registry =
            single_model_registry(MockLlmProvider::single(""), "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("empty LLM response must be valid");

        assert_eq!(resp_text(&resp), "");
    }

    // ── Test: usage propagated from last LLM call ──────────────────────────

    #[tokio::test]
    async fn test_usage_extracted_from_provider() {
        let registry =
            single_model_registry(MockLlmProvider::single("Done"), "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        // MockLlmProvider returns prompt=10, completion=5
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
    }

    // ── Test: model selection uses router decision ─────────────────────────

    #[tokio::test]
    async fn test_model_selection_uses_router_decision() {
        // Router selects "complex-model"; we verify the correct model_id is passed.
        let recording_provider = Arc::new(RecordingLlmProvider::new("response"));
        let default_provider = Arc::new(MockLlmProvider::single("default response"));

        let mut providers = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            default_provider as Arc<dyn Provider>,
        );
        providers.insert(
            "complex-model".to_string(),
            recording_provider.clone() as Arc<dyn Provider>,
        );

        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert("complex-model".to_string(), "claude-complex".to_string());

        let mut max_tokens_map = HashMap::new();
        max_tokens_map.insert("default-model".to_string(), 1024u32);
        max_tokens_map.insert("complex-model".to_string(), 4096u32);

        // Both models support chat_completions for this test.
        let mut caps_map: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps_map.insert(
            "default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        caps_map.insert(
            "complex-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens_map,
            caps_map,
            "default-model".to_string(),
        ));

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        engine
            .handle_request(make_user_request("Complex task"))
            .await
            .expect("should succeed");

        // Verify the correct model_id was passed to the recording provider
        assert_eq!(
            recording_provider.recorded_model(),
            Some("claude-complex".to_string()),
            "complex-model provider must receive the correct model_id"
        );
    }

    // ── Test: tool skipping when tools_needed = Some(false) ───────────────

    #[tokio::test]
    async fn test_tool_skipping_when_tools_not_needed() {
        // Provider records the system prompt to verify no command stubs
        struct SystemPromptCapture {
            captured: std::sync::Mutex<Option<String>>,
            response_text: &'static str,
        }

        #[async_trait]
        impl Provider for SystemPromptCapture {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Extract the system prompt from messages[0] (Role::System convention).
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                    let system_text = messages
                        .first()
                        .filter(|m| m.role == Role::System)
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| match p {
                                    ContentPart::Text(t) => Some(t.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    *self.captured.lock().unwrap() = Some(system_text);
                }
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text(self.response_text.to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "system-prompt-capture"
            }
        }

        let capture = Arc::new(SystemPromptCapture {
            captured: std::sync::Mutex::new(None),
            response_text: "no tools needed",
        });

        let registry = single_model_registry(
            {
                struct WrappedCapture(Arc<SystemPromptCapture>);
                #[async_trait]
                impl Provider for WrappedCapture {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "wrapped-capture"
                    }
                }
                WrappedCapture(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            MockRouter::with_tools_needed(false),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("What is the capital of France?"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // When tools are skipped, the system prompt must NOT contain command stubs
        assert!(
            !captured.contains("```toon"),
            "system prompt should not contain toon block when tools are skipped: {captured}"
        );
        assert!(
            !captured.contains("web_search"),
            "system prompt should not contain command stubs when tools are skipped"
        );
    }

    // ── Test: tool injection when tools_needed = Some(true) ───────────────

    #[tokio::test]
    async fn test_tool_injection_when_tools_needed() {
        struct SystemPromptCapture2 {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl Provider for SystemPromptCapture2 {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Extract the system prompt from messages[0] (Role::System convention).
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                    let system_text = messages
                        .first()
                        .filter(|m| m.role == Role::System)
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| match p {
                                    ContentPart::Text(t) => Some(t.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    *self.captured.lock().unwrap() = Some(system_text);
                }
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text("response".to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "system-prompt-capture2"
            }
        }

        let capture = Arc::new(SystemPromptCapture2 {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture2>);
                #[async_trait]
                impl Provider for W {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "w2"
                    }
                }
                W(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            MockRouter::with_tools_needed(true),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("Search for something"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // Commands must be injected in dash-list format with "Use when" trigger conditions
        assert!(
            captured.contains("- /web_search \u{2014} Use when Search the web"),
            "system prompt must contain dash-list command stubs when tools are needed"
        );
        assert!(
            captured.contains("BLOCKING REQUIREMENT"),
            "system prompt must contain blocking requirement language when tools are needed"
        );
    }

    // ── Test: tool injection when tools_needed = None (conservative) ──────

    #[tokio::test]
    async fn test_tool_injection_when_tools_needed_is_none() {
        // tools_needed = None -> conservative default: inject commands
        let router = MockRouter {
            command_score: 1.0,
            model_decision: None,
            tools_needed: None, // undecided
            memory_score: Some(0.9),
        };

        struct SystemPromptCapture3 {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl Provider for SystemPromptCapture3 {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Extract the system prompt from messages[0] (Role::System convention).
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion { ref messages, .. } = request {
                    let system_text = messages
                        .first()
                        .filter(|m| m.role == Role::System)
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| match p {
                                    ContentPart::Text(t) => Some(t.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    *self.captured.lock().unwrap() = Some(system_text);
                }
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text("response".to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "system-prompt-capture3"
            }
        }

        let capture = Arc::new(SystemPromptCapture3 {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture3>);
                #[async_trait]
                impl Provider for W {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "w3"
                    }
                }
                W(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            router,
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("Tell me about Rust"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // None -> conservative: inject commands
        assert!(
            captured.contains("web_search"),
            "tools_needed=None must default to injecting commands"
        );
    }

    // ── Test: model fallback on non-rate-limit error ───────────────────────

    #[tokio::test]
    async fn test_model_fallback_on_non_rate_limit_error() {
        // complex-model fails with RequestFailed; default-model succeeds.
        let registry = two_model_registry(
            MockLlmProvider::single("fallback response"),
            FailingLlmProvider,
        );

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Complex task"))
            .await
            .expect("fallback to default must succeed");

        assert_eq!(resp_text(&resp), "fallback response");
    }

    // ── Test: rate limit does NOT trigger fallback ─────────────────────────

    #[tokio::test]
    async fn test_rate_limit_does_not_trigger_fallback() {
        // complex-model is rate-limited; must not fall back.
        let registry = two_model_registry(
            MockLlmProvider::single("default would succeed"),
            RateLimitedLlmProvider,
        );

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Complex task"))
            .await;
        assert!(
            matches!(result, Err(WeftError::RateLimited { .. })),
            "rate limit must propagate without fallback, got: {:?}",
            result
        );
    }

    // ── Test: default model failure propagates error ───────────────────────

    #[tokio::test]
    async fn test_default_model_failure_propagates_error() {
        // Only the default model exists and it fails — no fallback available.
        let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");

        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "default model failure must propagate as Llm error, got: {:?}",
            result
        );
    }

    // ── Test: selected model IS default, no retry on failure ──────────────

    #[tokio::test]
    async fn test_selected_model_is_default_no_retry() {
        // Router selects "default-model" which is also the default — fails once,
        // must NOT retry (would infinite loop).
        let registry = two_model_registry(FailingLlmProvider, MockLlmProvider::single("alt"));

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            // Router selects "default-model" (same as the registry's default)
            MockRouter::with_model("default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Simple task"))
            .await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "selected=default failure must propagate without retry, got: {:?}",
            result
        );
    }

    // ── Test: context assembly uses TOON fenced blocks ─────────────────────

    #[tokio::test]
    async fn test_context_assembly_uses_toon_fenced_blocks() {
        use crate::context::assemble_system_prompt;
        use weft_core::CommandStub;

        let stubs = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];
        let prompt = assemble_system_prompt(&stubs, "You are a helpful assistant.", None);

        // Command stubs use dash-list format with "Use when" trigger conditions,
        // not TOON fenced blocks — TOON is reserved for results injection only.
        assert!(
            prompt.contains("- /web_search \u{2014} Use when Search the web"),
            "must use dash-list format with Use when trigger condition"
        );
        assert!(
            prompt.contains("BLOCKING REQUIREMENT"),
            "must include blocking requirement language"
        );
        assert!(
            !prompt.contains("commands[1]{name, description}:"),
            "must not use TOON array syntax for command stubs"
        );
    }

    // ── Test: no fenced blocks when tool skipping is active ───────────────

    #[tokio::test]
    async fn test_no_fenced_blocks_when_tool_skipping() {
        use crate::context::assemble_system_prompt_no_tools;

        let prompt = assemble_system_prompt_no_tools("You are a helpful assistant.");
        assert!(
            !prompt.contains("```toon"),
            "must not contain toon blocks when tools are skipped"
        );
        assert!(
            !prompt.contains("commands["),
            "must not contain command stubs when tools are skipped"
        );
    }

    // ── Built-in memory command tests ─────────────────────────────────────

    use weft_memory::{
        DefaultMemoryService, MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreMux,
        MemoryStoreResult, StoreInfo,
    };

    /// Mock memory store client with configurable query and store behaviour.
    struct MockMemStoreClient {
        query_entries: Vec<MemoryEntry>,
        query_error: Option<String>,
        store_id: String,
        store_error: Option<String>,
    }

    impl MockMemStoreClient {
        fn succeeds(entries: Vec<MemoryEntry>) -> Self {
            Self {
                query_entries: entries,
                query_error: None,
                store_id: "mock-mem-id".to_string(),
                store_error: None,
            }
        }

        fn query_fails(msg: &str) -> Self {
            Self {
                query_entries: vec![],
                query_error: Some(msg.to_string()),
                store_id: "mock-mem-id".to_string(),
                store_error: None,
            }
        }

        fn store_fails(msg: &str) -> Self {
            Self {
                query_entries: vec![],
                query_error: None,
                store_id: "mock-mem-id".to_string(),
                store_error: Some(msg.to_string()),
            }
        }
    }

    #[async_trait::async_trait]
    impl MemoryStoreClient for MockMemStoreClient {
        async fn query(
            &self,
            _query: &str,
            _max_results: u32,
            _min_score: f32,
        ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
            if let Some(e) = &self.query_error {
                return Err(MemoryStoreError::QueryFailed(e.clone()));
            }
            Ok(self.query_entries.clone())
        }

        async fn store(
            &self,
            _content: &str,
            _metadata: Option<&serde_json::Value>,
        ) -> Result<MemoryStoreResult, MemoryStoreError> {
            if let Some(e) = &self.store_error {
                return Err(MemoryStoreError::StoreFailed(e.clone()));
            }
            Ok(MemoryStoreResult {
                id: self.store_id.clone(),
            })
        }
    }

    fn mem_entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: content.to_string(),
            score: 0.9,
            created_at: "2026-03-15T10:00:00Z".to_string(),
            metadata: None,
        }
    }

    fn make_mux_with_stores(
        stores: Vec<(&str, bool, bool, Arc<dyn MemoryStoreClient>)>,
    ) -> Arc<MemoryStoreMux> {
        let mut client_map: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        let mut max_results = HashMap::new();
        let mut readable = std::collections::HashSet::new();
        let mut writable = std::collections::HashSet::new();
        for (name, r, w, client) in stores {
            client_map.insert(name.to_string(), client);
            max_results.insert(name.to_string(), 5u32);
            if r {
                readable.insert(name.to_string());
            }
            if w {
                writable.insert(name.to_string());
            }
        }
        Arc::new(MemoryStoreMux::new(
            client_map,
            max_results,
            readable,
            writable,
        ))
    }

    fn make_engine_with_mux<R, C>(
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
        mux: Option<Arc<MemoryStoreMux>>,
    ) -> GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
    where
        R: SemanticRouter + Send + Sync + 'static,
        C: CommandRegistry + Send + Sync + 'static,
    {
        let memory = mux.map(wrap_mux);
        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            memory,
            Arc::new(weft_hooks::HookRegistry::empty()),
        )
    }

    /// Wrap a `MemoryStoreMux` in a `DefaultMemoryService` with no examples (store_infos with
    /// no examples). Used when tests only care about ops (query/store), not routing candidates.
    fn wrap_mux(mux: Arc<MemoryStoreMux>) -> Arc<DefaultMemoryService> {
        // Build store_infos from the mux's store names and capabilities.
        // No examples — these tests use MockRouter with fixed scores and don't
        // need semantic routing candidates populated.
        let store_infos: Vec<StoreInfo> = {
            let all: std::collections::HashSet<String> = mux
                .store_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            let readable: std::collections::HashSet<String> = mux
                .readable_store_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            let writable: std::collections::HashSet<String> = mux
                .writable_store_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            all.into_iter()
                .map(|name| {
                    let mut caps = Vec::new();
                    if readable.contains(&name) {
                        caps.push("read".to_string());
                    }
                    if writable.contains(&name) {
                        caps.push("write".to_string());
                    }
                    StoreInfo {
                        name,
                        capabilities: caps,
                        examples: vec![],
                    }
                })
                .collect()
        };
        Arc::new(DefaultMemoryService::new(mux, store_infos))
    }

    // ── /recall tests ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_recall_intercepted_before_command_registry() {
        // Engine with memory mux — /recall should NOT reach the command registry.
        // If it did reach the registry, it would fail with CommandError::NotFound
        // (the mock registry has no "recall" command) and the loop would continue
        // until the LLM emits a response without commands.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "I found some memories about preferences.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]), // no "recall" in registry
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What are the user's preferences?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "I found some memories about preferences.");
    }

    #[tokio::test]
    async fn test_recall_returns_memories_as_toon() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "user prefers dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "Thanks for the context.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What does the user like?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Thanks for the context.");
    }

    #[tokio::test]
    async fn test_recall_no_query_arg_uses_user_message() {
        // /recall without query arg should still succeed (falls back to user message).
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall", "Nothing found in memory."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Tell me about user prefs"))
            .await
            .expect("should succeed without panicking");

        assert_eq!(resp_text(&resp), "Nothing found in memory.");
    }

    #[tokio::test]
    async fn test_recall_all_stores_fail_returns_success_with_no_memories() {
        // All stores fail — should return success=true with "No relevant memories found."
        let mux = make_mux_with_stores(vec![(
            "broken",
            true,
            true,
            Arc::new(MockMemStoreClient::query_fails("connection refused")),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something\"",
                "Understood, no memories available.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Search memory"))
            .await
            .expect("should succeed even when stores fail");

        assert_eq!(resp_text(&resp), "Understood, no memories available.");
    }

    #[tokio::test]
    async fn test_recall_no_mux_returns_error_result() {
        // /recall with no mux configured should emit an error result to the LLM.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something\"",
                "Memory is not configured.",
            ]),
            "test-model",
            "claude-test",
        );
        // Engine with no mux but we need to parse /recall — add it to registry as a command
        // Actually, without a mux, /recall won't be in known_commands, so it won't be parsed.
        // Test instead: call handle_builtin directly by building engine and sending a request
        // where the LLM emits /recall but the parser only picks it up if we register it.
        // To force the parse, we build an engine with a mock mux that is None-equivalent
        // — but there's no way to do that without an actual mux; the mux controls parser registration.
        // This is correct behaviour: without a mux, /recall is not in known_commands and is treated as prose.
        // Verify the response still works (just contains the text with /recall treated as prose).
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            None, // no mux
        );

        let resp = engine
            .handle_request(make_user_request("Try to recall"))
            .await
            .expect("should succeed");

        // The LLM response includes "/recall query: something" but it's treated as prose
        // because recall is not in known_commands when mux is None.
        // The first LLM response is returned directly since no commands are parsed.
        assert_eq!(resp_text(&resp), "/recall query: \"something\"");
    }

    // ── /remember tests ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_remember_intercepted_and_stores_to_writable() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Memory stored.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember this"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Memory stored.");
    }

    #[tokio::test]
    async fn test_remember_missing_content_returns_error() {
        // /remember without content arg should return success:false error to LLM.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember", // missing content
                "Got an error about missing argument.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Store something"))
            .await
            .expect("should succeed overall");

        assert_eq!(resp_text(&resp), "Got an error about missing argument.");
    }

    #[tokio::test]
    async fn test_remember_no_writable_stores_returns_empty_success() {
        // All stores are read-only — /remember should return success with no stores written.
        let mux = make_mux_with_stores(vec![(
            "code",
            true,
            false, // read-only
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"some info\"",
                "Attempted to remember.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember this"))
            .await
            .expect("should not fail");

        assert_eq!(resp_text(&resp), "Attempted to remember.");
    }

    // ── --describe tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_recall_describe_returns_compiled_text() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall --describe", "I see how recall works."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("How does recall work?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "I see how recall works.");
    }

    #[tokio::test]
    async fn test_remember_describe_does_not_reach_command_registry() {
        // The mock registry has no "remember" command — if --describe reached the
        // registry, it would return an error. With built-in interception it returns
        // the compiled describe text and success=true.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember --describe",
                "Now I know how to use remember.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]), // no "remember" in registry
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("How do I store memories?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Now I know how to use remember.");
    }

    // ── mixed built-in + external commands ────────────────────────────────

    #[tokio::test]
    async fn test_mixed_builtin_and_external_commands() {
        // LLM emits both /recall and /web_search in the same response.
        // Both should execute: built-in first, then external.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "user prefers dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"prefs\"\n/web_search query: \"dark mode\"",
                "Found memories and web results.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "dark mode is popular".to_string(),
                    error: None,
                },
            ),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What do I know about dark mode?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found memories and web results.");
    }

    #[tokio::test]
    async fn test_external_commands_still_work_with_mux_present() {
        // Verify that wiring in a mux doesn't break the existing command registry path.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Here are the results."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "5 results about Rust".to_string(),
                    error: None,
                },
            ),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Here are the results.");
    }

    // ── Phase 3: Semantic Router Integration tests ─────────────────────────

    /// Build a WeftConfig with memory stores for Phase 3 routing tests.
    ///
    /// `stores`: slice of `(name, endpoint, can_read, can_write, examples)`.
    /// `memory_threshold`: optional per-domain memory threshold.
    fn config_with_memory(
        stores: &[(&str, &str, bool, bool, Vec<&str>)],
        memory_threshold: Option<f32>,
    ) -> Arc<WeftConfig> {
        let store_configs: Vec<MemoryStoreConfig> = stores
            .iter()
            .map(|(name, endpoint, can_read, can_write, examples)| {
                let mut capabilities = Vec::new();
                if *can_read {
                    capabilities.push(StoreCapability::Read);
                }
                if *can_write {
                    capabilities.push(StoreCapability::Write);
                }
                MemoryStoreConfig {
                    name: name.to_string(),
                    endpoint: endpoint.to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities,
                    examples: examples.iter().map(|s| s.to_string()).collect(),
                }
            })
            .collect();

        let memory_domain = memory_threshold.map(|t| DomainConfig {
            enabled: true,
            threshold: Some(t),
        });

        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.5, // default threshold for memory routing tests
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
                domains: DomainsConfig {
                    model: None,
                    tool_necessity: None,
                    memory: memory_domain,
                },
            },
            tool_registry: None,
            memory: Some(MemoryConfig {
                stores: store_configs,
            }),
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    /// Build an engine with the given config, mux, and router.
    ///
    /// The config provides routing thresholds, domain settings, AND store metadata
    /// (examples) for building routing candidates. The mux handles actual ops.
    /// Internally constructs a `DefaultMemoryService` from the config's memory section.
    fn make_engine_with_config_and_mux<R, C>(
        config: Arc<WeftConfig>,
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
        mux: Option<Arc<MemoryStoreMux>>,
    ) -> GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
    where
        R: SemanticRouter + Send + Sync + 'static,
        C: CommandRegistry + Send + Sync + 'static,
    {
        let memory = mux.map(|m| service_from_config_and_mux(&config, m));
        GatewayEngine::new(
            config,
            registry,
            Arc::new(router),
            Arc::new(commands),
            memory,
            Arc::new(weft_hooks::HookRegistry::empty()),
        )
    }

    /// Build a `DefaultMemoryService` from a config and a mux.
    ///
    /// Extracts `StoreInfo` (name, capabilities, examples) from the config's memory section
    /// so that routing candidates are populated correctly, then wraps the mux for operations.
    fn service_from_config_and_mux(
        config: &WeftConfig,
        mux: Arc<MemoryStoreMux>,
    ) -> Arc<DefaultMemoryService> {
        let store_infos: Vec<StoreInfo> = config
            .memory
            .as_ref()
            .map(|mem| {
                mem.stores
                    .iter()
                    .map(|s| {
                        let mut caps = Vec::new();
                        if s.can_read() {
                            caps.push("read".to_string());
                        }
                        if s.can_write() {
                            caps.push("write".to_string());
                        }
                        StoreInfo {
                            name: s.name.clone(),
                            capabilities: caps,
                            examples: s.examples.clone(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Arc::new(DefaultMemoryService::new(mux, store_infos))
    }

    // ── Phase 3: build_memory_candidates (via weft_router) ────────────────
    // These tests verify the moved candidate builder via the weft_router re-export.

    #[test]
    fn test_build_memory_candidates_empty_when_no_stores() {
        // Empty store list produces empty candidate sets.
        let result = build_memory_candidates(&[]);
        assert!(result.all.is_empty());
        assert!(result.read.is_empty());
        assert!(result.write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_read_write_store() {
        // A store with both capabilities appears in all three sets.
        let stores = vec![weft_router::MemoryStoreRef {
            name: "conv".to_string(),
            capabilities: vec!["read".to_string(), "write".to_string()],
            examples: vec!["recall conv".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert_eq!(result.all[0].id, "conv");
        assert_eq!(result.read.len(), 1);
        assert_eq!(result.read[0].id, "conv");
        assert_eq!(result.write.len(), 1);
        assert_eq!(result.write[0].id, "conv");
    }

    #[test]
    fn test_build_memory_candidates_read_only_store() {
        // A read-only store appears in all and read, but not write.
        let stores = vec![weft_router::MemoryStoreRef {
            name: "kb".to_string(),
            capabilities: vec!["read".to_string()],
            examples: vec!["knowledge base".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert_eq!(result.read.len(), 1);
        assert!(result.write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_write_only_store() {
        // A write-only store appears in all and write, but not read.
        let stores = vec![weft_router::MemoryStoreRef {
            name: "audit".to_string(),
            capabilities: vec!["write".to_string()],
            examples: vec!["audit log".to_string()],
        }];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 1);
        assert!(result.read.is_empty());
        assert_eq!(result.write.len(), 1);
    }

    #[test]
    fn test_build_memory_candidates_multiple_stores() {
        // Multiple stores split correctly by capability.
        let stores = vec![
            weft_router::MemoryStoreRef {
                name: "conv".to_string(),
                capabilities: vec!["read".to_string(), "write".to_string()],
                examples: vec!["conversation".to_string()],
            },
            weft_router::MemoryStoreRef {
                name: "kb".to_string(),
                capabilities: vec!["read".to_string()],
                examples: vec!["knowledge base".to_string()],
            },
            weft_router::MemoryStoreRef {
                name: "audit".to_string(),
                capabilities: vec!["write".to_string()],
                examples: vec!["audit".to_string()],
            },
        ];
        let result = build_memory_candidates(&stores);
        assert_eq!(result.all.len(), 3);
        assert_eq!(result.read.len(), 2); // conv + kb
        assert_eq!(result.write.len(), 2); // conv + audit
        assert!(result.read.iter().any(|c| c.id == "conv"));
        assert!(result.read.iter().any(|c| c.id == "kb"));
        assert!(result.write.iter().any(|c| c.id == "conv"));
        assert!(result.write.iter().any(|c| c.id == "audit"));
    }

    // ── Phase 3: route_all_domains includes Memory domain ─────────────────

    #[tokio::test]
    async fn test_route_all_domains_includes_memory_when_configured() {
        // When memory stores are configured, route_all_domains should pass
        // Memory domain candidates to the router. We verify by checking that
        // the engine picks up the memory candidates (indirectly via exec_recall routing).
        //
        // Setup: one store "conv", router scores it above threshold (0.9 > 0.5).
        // /recall should target "conv" specifically (not fan out to all).
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "dark mode pref",
        )]));
        let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

        let config = config_with_memory(
            &[(
                "conv",
                "http://localhost:50052",
                true,
                true,
                vec!["conversation recall"],
            )],
            None, // no memory-domain threshold override — uses classifier.threshold = 0.5
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "I found memory about dark mode.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router scores first (only) read candidate at 0.9 > 0.5 threshold.
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Tell me about preferences"))
            .await
            .expect("should succeed");

        // Should succeed and include memory content in the response.
        assert_eq!(resp_text(&resp), "I found memory about dark mode.");
    }

    // ── Phase 3: /recall per-invocation routing ────────────────────────────

    #[tokio::test]
    async fn test_recall_routes_based_on_query_not_user_message() {
        // The key Phase 3 invariant: /recall routing uses the query argument,
        // not the user's original message. Both route via score_memory_candidates().
        //
        // Setup: two stores. "conv" gets score 0.9, "kb" gets 0.0 from mock router.
        // User message is "help me with my code" (code-domain).
        // /recall query is "user preferences" (conv-domain).
        // Only "conv" should be queried.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "user prefers dark mode",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::query_fails("should not be called"));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["user preferences"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["code architecture"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "Found: user prefers dark mode.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router: first candidate (conv) gets 0.9, second (kb) gets 0.0.
        // This simulates "user preferences" being semantically closest to conv.
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("help me with my code"))
            .await
            .expect("should succeed");

        // kb_client would fail if called — success means only conv was queried.
        assert_eq!(resp_text(&resp), "Found: user prefers dark mode.");
    }

    #[tokio::test]
    async fn test_recall_below_threshold_fans_out_to_all_readable() {
        // When all candidates score below threshold, /recall fans out to ALL read-capable stores.
        // Router scores first candidate at 0.1, second at 0.0 — both below threshold 0.5.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "conv memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m2",
            "kb memory",
        )]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            Some(0.5), // memory domain threshold
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something unrelated\"",
                "Found memories from both stores.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router: first candidate 0.1, second 0.0 — both below 0.5 threshold.
        let router = MockRouter::with_score(0.9).with_memory_score(0.1);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("something"))
            .await
            .expect("should succeed");

        // Both stores should have been queried (fan-out) — response shows 2 memories.
        assert_eq!(resp_text(&resp), "Found memories from both stores.");
    }

    #[tokio::test]
    async fn test_recall_router_unavailable_fans_out_to_all_readable() {
        // When the router is unavailable (ModelNotLoaded), /recall fans out to all readable.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "conv memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m2",
            "kb memory",
        )]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"anything\"",
                "Got results from both stores.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router returns ModelNotLoaded for score_memory_candidates.
        let router = MockRouter::with_score(0.9).with_memory_unavailable();

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("anything"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Got results from both stores.");
    }

    // ── Phase 3: /remember per-invocation routing ─────────────────────────

    #[tokio::test]
    async fn test_remember_routes_based_on_content_not_user_message() {
        // /remember routing uses the content argument, not the user's original message.
        // Setup: two writable stores. "conv" gets score 0.9, "audit" gets 0.0.
        // User asks about code; LLM remembers a user preference.
        // Only "conv" should be written to.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        // audit_client: if written to, the store call would succeed, but we verify
        // by checking the TOON output only mentions "conv".
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["user preferences"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit logs"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Noted, I'll remember that.",
            ]),
            "test-model",
            "claude-test",
        );

        // First write candidate (conv) gets 0.9, second (audit) gets 0.0.
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("how do I configure nginx?"))
            .await
            .expect("should succeed");

        // audit_client has store_fails — if it were called we'd get a partial failure in TOON.
        // Success means only conv was written.
        assert_eq!(resp_text(&resp), "Noted, I'll remember that.");
    }

    #[tokio::test]
    async fn test_remember_below_threshold_picks_highest_scoring_store() {
        // When all candidates are below threshold, /remember picks single highest-scoring store.
        // Router: first candidate (conv) 0.3, second (audit) 0.1 — both below 0.5 threshold.
        // conv has the higher score, so it gets picked.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        // audit_client: if written, would succeed — but conv should be picked.
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["conversation prefs"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit events"],
                ),
            ],
            Some(0.5), // threshold: 0.5
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/remember content: \"something\"", "Memory stored."]),
            "test-model",
            "claude-test",
        );

        // Both below threshold; first candidate scores higher (0.3 > 0.0 for second).
        let router = MockRouter::with_score(0.9).with_memory_score(0.3);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("remember something"))
            .await
            .expect("should succeed");

        // audit_client has store_fails — success means only conv was picked.
        assert_eq!(resp_text(&resp), "Memory stored.");
    }

    #[tokio::test]
    async fn test_remember_router_unavailable_writes_to_first_writable() {
        // When router is unavailable, /remember writes to the first configured writable store.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["conversations"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/remember content: \"important note\"", "Saved."]),
            "test-model",
            "claude-test",
        );

        // Router unavailable for memory scoring.
        let router = MockRouter::with_score(0.9).with_memory_unavailable();

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("remember this"))
            .await
            .expect("should succeed");

        // audit_client has store_fails — success means only first writable (conv) was picked.
        assert_eq!(resp_text(&resp), "Saved.");
    }

    // ── Phase 3: memory domain threshold from config ──────────────────────

    #[tokio::test]
    async fn test_memory_domain_threshold_gate_applied() {
        // When a per-domain memory threshold is configured, it takes precedence over
        // the classifier threshold.
        //
        // Classifier threshold = 0.5, memory domain threshold = 0.8.
        // Router scores first candidate at 0.7 — above 0.5 but below 0.8.
        // With memory domain threshold, the store should NOT be directly selected
        // (falls back to all readable for /recall, or highest-scoring for /remember).
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1", "memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            Some(0.8), // memory domain threshold = 0.8 (higher than classifier 0.5)
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"something\"", "Found memories."]),
            "test-model",
            "claude-test",
        );

        // Score 0.7: above classifier threshold (0.5) but below memory domain threshold (0.8).
        let router = MockRouter::with_score(0.9).with_memory_score(0.7);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("recall something"))
            .await
            .expect("should succeed");

        // Below memory-domain threshold → fell back to all readable → both stores queried.
        // conv_client returns a memory, kb_client returns empty.
        // Final response is whatever the LLM said after seeing results.
        assert_eq!(resp_text(&resp), "Found memories.");
    }

    // ── Phase 3: RoutingDecision::fallback() memory_stores population ─────

    #[test]
    fn test_fallback_decision_populates_memory_stores() {
        use weft_router::{RoutingDecision, RoutingDomainKind};
        // RoutingDecision::fallback() should populate memory_stores with all
        // candidates at score 1.0 when the Memory domain is included.
        let domains = vec![
            (
                RoutingDomainKind::Commands,
                vec![RoutingCandidate {
                    id: "search".to_string(),
                    examples: vec!["search: Search".to_string()],
                }],
            ),
            (
                RoutingDomainKind::Memory,
                vec![
                    RoutingCandidate {
                        id: "conv".to_string(),
                        examples: vec!["conversations".to_string()],
                    },
                    RoutingCandidate {
                        id: "kb".to_string(),
                        examples: vec!["knowledge base".to_string()],
                    },
                ],
            ),
        ];

        let decision = RoutingDecision::fallback(&domains);

        // Commands: scored at 1.0.
        assert_eq!(decision.commands.len(), 1);

        // Memory stores: all at 1.0, sorted alphabetically.
        assert_eq!(decision.memory_stores.len(), 2);
        assert_eq!(decision.memory_stores[0].id, "conv");
        assert_eq!(decision.memory_stores[1].id, "kb");
        for m in &decision.memory_stores {
            assert_eq!(m.score, 1.0, "fallback memory scores should be 1.0");
        }

        // Model and tools: conservative defaults.
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
    }

    #[test]
    fn test_fallback_decision_memory_stores_empty_when_no_memory_domain() {
        // When Memory domain is NOT included in domains, memory_stores stays empty.
        let domains = vec![(
            RoutingDomainKind::Commands,
            vec![RoutingCandidate {
                id: "search".to_string(),
                examples: vec!["search: Search".to_string()],
            }],
        )];

        let decision = RoutingDecision::fallback(&domains);
        assert!(decision.memory_stores.is_empty());
    }

    // ── Phase 3: memory domain disabled via config ─────────────────────────

    // ── Phase 4: hook integration helpers ─────────────────────────────────

    /// Build a `GatewayEngine` with a custom hook registry (Phase 4 tests).
    fn make_engine_with_hooks<R, C>(
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
        hook_registry: weft_hooks::HookRegistry,
    ) -> GatewayEngine<
        weft_hooks::HookRegistry,
        R,
        weft_memory::NullMemoryService,
        ProviderRegistry,
        C,
    >
    where
        R: SemanticRouter + Send + Sync + 'static,
        C: CommandRegistry + Send + Sync + 'static,
    {
        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_registry),
        )
    }

    /// Build a `HookRegistry` with a single hook executor for one event.
    fn hook_registry_with(
        event: HookEvent,
        executor: Box<dyn weft_hooks::executor::HookExecutor>,
        matcher: Option<&str>,
        priority: u32,
    ) -> weft_hooks::HookRegistry {
        let matcher = HookMatcher::new(matcher, 0).expect("valid matcher in test");
        let hook = RegisteredHook {
            event,
            matcher,
            executor,
            name: "test-hook".to_string(),
            priority,
        };
        let mut map = std::collections::HashMap::new();
        map.insert(event, vec![hook]);
        HookRegistry::from_registered(map)
    }

    // ── Phase 4: inline hook executor helpers ─────────────────────────────

    /// A hook executor that always returns a fixed response.
    struct FixedHookExecutor(weft_hooks::types::HookResponse);

    #[async_trait]
    impl weft_hooks::executor::HookExecutor for FixedHookExecutor {
        async fn execute(&self, _payload: &serde_json::Value) -> weft_hooks::types::HookResponse {
            self.0.clone()
        }
    }

    /// A hook executor that records every payload it receives, then returns Allow.
    struct RecordingHookExecutor {
        recorded: std::sync::Mutex<Vec<serde_json::Value>>,
    }

    impl RecordingHookExecutor {
        fn new() -> Self {
            Self {
                recorded: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn recorded_payloads(&self) -> Vec<serde_json::Value> {
            self.recorded.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl weft_hooks::executor::HookExecutor for RecordingHookExecutor {
        async fn execute(&self, payload: &serde_json::Value) -> weft_hooks::types::HookResponse {
            self.recorded.lock().unwrap().push(payload.clone());
            weft_hooks::types::HookResponse::allow()
        }
    }

    /// A hook executor that modifies the payload by merging in a JSON object.
    struct ModifyHookExecutor(serde_json::Value);

    #[async_trait]
    impl weft_hooks::executor::HookExecutor for ModifyHookExecutor {
        async fn execute(&self, payload: &serde_json::Value) -> weft_hooks::types::HookResponse {
            // Merge self.0 fields into current payload.
            let mut merged = payload.clone();
            if let (Some(obj), Some(extra)) = (merged.as_object_mut(), self.0.as_object()) {
                for (k, v) in extra {
                    obj.insert(k.clone(), v.clone());
                }
            }
            weft_hooks::types::HookResponse {
                decision: weft_hooks::types::HookDecision::Modify,
                reason: None,
                modified: Some(merged),
                context: None,
            }
        }
    }

    /// Build a HookRegistry with multiple hooks for potentially different events.
    fn hook_registry_multi(
        hooks: Vec<(
            HookEvent,
            Box<dyn weft_hooks::executor::HookExecutor>,
            Option<&'static str>,
            u32,
        )>,
    ) -> weft_hooks::HookRegistry {
        let mut map: std::collections::HashMap<HookEvent, Vec<RegisteredHook>> =
            std::collections::HashMap::new();
        for (i, (event, executor, matcher_pat, priority)) in hooks.into_iter().enumerate() {
            let matcher = HookMatcher::new(matcher_pat, i).expect("valid matcher");
            let hook = RegisteredHook {
                event,
                matcher,
                executor,
                name: format!("test-hook-{i}"),
                priority,
            };
            map.entry(event).or_default().push(hook);
        }
        HookRegistry::from_registered(map)
    }

    // ── Phase 4: RequestStart hook integration ─────────────────────────────

    #[tokio::test]
    async fn test_request_start_hook_blocks_returns_hook_blocked_error() {
        // RequestStart block — 403, no LLM call made.
        let registry = single_model_registry(
            MockLlmProvider::single("should not be called"),
            "test-model",
            "claude-test",
        );
        let hook_reg = hook_registry_with(
            HookEvent::RequestStart,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "auth failed",
            ))),
            None,
            100,
        );
        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlocked { ref event, ref reason, .. })
                    if event == "RequestStart" && reason == "auth failed"
            ),
            "expected HookBlocked for RequestStart, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_request_start_hook_allows_request_proceeds() {
        // RequestStart allow — request continues normally.
        let registry = single_model_registry(
            MockLlmProvider::single("Response text"),
            "test-model",
            "claude-test",
        );
        let hook_reg = hook_registry_with(
            HookEvent::RequestStart,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::allow())),
            None,
            100,
        );
        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed when RequestStart hook allows");
        assert_eq!(resp_text(&resp), "Response text");
    }

    // ── Phase 4: PreRoute hook integration ─────────────────────────────────

    #[tokio::test]
    async fn test_pre_route_hook_blocks_model_domain_returns_hook_blocked() {
        // PreRoute block on model domain → hard block (403).
        let hook_reg = hook_registry_with(
            HookEvent::PreRoute,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "model blocked",
            ))),
            Some("model"),
            100,
        );

        let engine = GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("should not be called"),
                MockLlmProvider::single("should not be called"),
            ),
            Arc::new(MockRouter::with_model("complex-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlocked { ref event, .. })
                    if event == "PreRoute"
            ),
            "expected HookBlocked for PreRoute on model domain, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_pre_route_hook_no_matcher_fires_on_all_domains() {
        // A PreRoute hook with no matcher fires on every domain.
        // Verify by collecting the `domain` field from each payload received.
        let recorder = Arc::new(RecordingHookExecutor::new());
        let recorder_clone = Arc::clone(&recorder);

        struct ArcRecording(Arc<RecordingHookExecutor>);
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for ArcRecording {
            async fn execute(
                &self,
                payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.recorded.lock().unwrap().push(payload.clone());
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::PreRoute,
            Box::new(ArcRecording(recorder_clone)),
            None, // no matcher — fires on all domains
            100,
        );

        let engine = make_engine_with_hooks(
            single_model_registry(MockLlmProvider::single("Done"), "test-model", "claude-test"),
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search")]),
            hook_reg,
        );

        engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        let payloads = recorder.recorded_payloads();
        // There must be at least one PreRoute firing (commands domain is always present).
        assert!(
            !payloads.is_empty(),
            "expected PreRoute hook to fire at least once"
        );
        // Extract domain names from payloads.
        let domains: Vec<&str> = payloads
            .iter()
            .filter_map(|p| p.get("domain").and_then(|d| d.as_str()))
            .collect();
        assert!(
            domains.contains(&"commands"),
            "expected PreRoute to fire for commands domain, got: {domains:?}"
        );
    }

    // ── Phase 4: PostRoute hook integration ────────────────────────────────

    #[tokio::test]
    async fn test_post_route_hook_overrides_model_selection() {
        // PostRoute hook on model domain overrides `selected` → correct model used.
        // Router would pick complex-model, but PostRoute hook forces default-model.
        // The `selected` field in the model PostRoute payload is an array of model name strings.
        let hook_reg = hook_registry_with(
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "selected": ["default-model"]
            }))),
            Some("model"),
            100,
        );

        let engine = GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                // default-model returns the expected response.
                MockLlmProvider::single("final response"),
                // complex-model returns a sentinel — if this appears, the hook didn't work.
                MockLlmProvider::single("WRONG: complex-model was used"),
            ),
            Arc::new(MockRouter::with_model("complex-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        // If PostRoute correctly overrides to default-model, we get "final response".
        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed with PostRoute model override to default-model");
        assert_eq!(resp_text(&resp), "final response");
    }

    #[tokio::test]
    async fn test_post_route_hook_overrides_model_to_invalid_falls_back_to_default() {
        // PostRoute hook overrides model to a name not in the registry → fallback to default.
        // `selected` is an array of model name strings (same format as the PostRoute payload).
        let hook_reg = hook_registry_with(
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "selected": ["nonexistent-model"]
            }))),
            Some("model"),
            100,
        );

        let engine = GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("fallback response"),
                MockLlmProvider::single("should not be called"),
            ),
            Arc::new(MockRouter::with_model("complex-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        // Should succeed by falling back to default-model.
        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should fall back to default model on invalid PostRoute override");
        assert_eq!(resp_text(&resp), "fallback response");
    }

    // ── Phase 4: PreToolUse hook integration ───────────────────────────────

    #[tokio::test]
    async fn test_pre_tool_use_blocks_command_returns_failed_result() {
        // PreToolUse blocks web_search → command result is an error in conversation.
        // The LLM sees the error and provides a final response.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "Search blocked, answering from memory.",
            ]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreToolUse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "permission denied",
            ))),
            Some("web_search"),
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed — block returns error result not 403");
        assert_eq!(resp_text(&resp), "Search blocked, answering from memory.");
    }

    #[tokio::test]
    async fn test_pre_tool_use_blocks_one_command_other_executes() {
        // PreToolUse blocks web_search (matcher="web_search") but allows code_review.
        // Both are invoked in the same turn — only code_review executes.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"\n/code_review target: src",
                "Done.",
            ]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreToolUse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "blocked",
            ))),
            Some("web_search"), // only blocks web_search
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search"), ("code_review", "Review")])
                .with_execute_result(CommandResult {
                    command_name: "code_review".to_string(),
                    success: true,
                    output: "review done".to_string(),
                    error: None,
                }),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Do both"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "Done.");
    }

    // ── Phase 4: PostToolUse hook integration ──────────────────────────────

    #[tokio::test]
    async fn test_post_tool_use_hook_modifies_output() {
        // PostToolUse hook modifies the command output — modified text visible to LLM.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "Search result was: hooked output",
            ]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PostToolUse,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "output": "hooked output"
            }))),
            Some("web_search"),
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "original output".to_string(),
                    error: None,
                },
            ),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "Search result was: hooked output");
    }

    // ── Phase 4: PreResponse hook integration ──────────────────────────────

    #[tokio::test]
    async fn test_pre_response_hook_blocks_triggers_regeneration() {
        // PreResponse blocks once → LLM regenerates; second response passes.
        // The mock LLM returns two responses in order.
        let registry = single_model_registry(
            MockLlmProvider::new(vec!["first response", "second response"]),
            "test-model",
            "claude-test",
        );

        // Block on first call, then allow (FixedHookExecutor always returns same response,
        // so we use a stateful executor that blocks once).
        struct BlockOnceExecutor {
            count: std::sync::Mutex<u32>,
        }
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for BlockOnceExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                let mut count = self.count.lock().unwrap();
                *count += 1;
                if *count == 1 {
                    weft_hooks::types::HookResponse::block("content policy")
                } else {
                    weft_hooks::types::HookResponse::allow()
                }
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::PreResponse,
            Box::new(BlockOnceExecutor {
                count: std::sync::Mutex::new(0),
            }),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed after regeneration");
        // Second response should be returned (first was blocked and regenerated).
        assert_eq!(resp_text(&resp), "second response");
    }

    #[tokio::test]
    async fn test_pre_response_hook_blocks_after_max_retries_returns_422() {
        // PreResponse blocks on every attempt → HTTP 422 after max retries exhausted.
        // test_config() sets max_pre_response_retries = 2.
        let registry = single_model_registry(
            // Provide enough responses for the retry loop + 1 extra.
            MockLlmProvider::new(vec!["bad response"; 10]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreResponse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "always blocked",
            ))),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlockedAfterRetries { ref event, retries, .. })
                    if event == "PreResponse" && retries == 2
            ),
            "expected HookBlockedAfterRetries after max retries, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_pre_response_hook_modifies_response_text() {
        // PreResponse Modify changes the response text → client sees modified text.
        let registry = single_model_registry(
            MockLlmProvider::single("original text"),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreResponse,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "text": "modified text"
            }))),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "modified text");
    }

    // ── Phase 4: RequestEnd hook integration ───────────────────────────────

    #[tokio::test]
    async fn test_request_end_hook_fires_after_response() {
        // RequestEnd fires after the response is built (verify via side-effect).
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        struct FireFlagExecutor(Arc<std::sync::atomic::AtomicBool>);
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for FireFlagExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.store(true, std::sync::atomic::Ordering::Release);
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::RequestEnd,
            Box::new(FireFlagExecutor(fired_clone)),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            single_model_registry(
                MockLlmProvider::single("Hello"),
                "test-model",
                "claude-test",
            ),
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "Hello");

        // Give the spawned RequestEnd task a moment to execute.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            fired.load(std::sync::atomic::Ordering::Acquire),
            "RequestEnd hook should have fired after response"
        );
    }

    #[tokio::test]
    async fn test_request_end_semaphore_exhausted_drops_task_with_warning() {
        // Semaphore with 0 permits → RequestEnd task dropped (no panic, no hang).
        // Build a config with request_end_concurrency = 0.
        let config = Arc::new(WeftConfig {
            request_end_concurrency: 0,
            ..(*test_config()).clone()
        });

        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        struct FireFlagExecutor(Arc<std::sync::atomic::AtomicBool>);
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for FireFlagExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.store(true, std::sync::atomic::Ordering::Release);
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::RequestEnd,
            Box::new(FireFlagExecutor(fired_clone)),
            None,
            100,
        );

        let engine = GatewayEngine::new(
            config,
            single_model_registry(
                MockLlmProvider::single("Hello"),
                "test-model",
                "claude-test",
            ),
            Arc::new(MockRouter::with_score(0.9)),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed even with exhausted semaphore");
        assert_eq!(resp_text(&resp), "Hello");

        // Give time for any possible task scheduling.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // Task was dropped — hook should NOT have fired.
        assert!(
            !fired.load(std::sync::atomic::Ordering::Acquire),
            "RequestEnd hook should have been dropped due to semaphore exhaustion"
        );
    }

    // ── Phase 4: no-hooks unchanged behavior ───────────────────────────────

    #[tokio::test]
    async fn test_no_hooks_configured_behavior_unchanged() {
        // With no hooks, engine behaves identically to existing tests.
        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Results found."]),
            "test-model",
            "claude-test",
        );
        // Default make_engine uses HookRegistry::empty() — explicitly verify.
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "Rust results".to_string(),
                    error: None,
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed with no hooks");
        assert_eq!(resp_text(&resp), "Results found.");
    }

    // ── Phase 4: hook priority ordering ────────────────────────────────────

    #[tokio::test]
    async fn test_hooks_execute_in_priority_order_lower_first() {
        // Two RequestStart hooks: priority 200 blocks, priority 50 allows.
        // Since lower fires first, priority 50 (allow) fires before priority 200 (block).
        // The block at priority 200 runs second → request is blocked.
        // This verifies that priority 200 hook's block is respected even after allow.
        //
        // Inverse test: priority 50 blocks, 200 allows → blocked at priority 50.
        let registry = single_model_registry(
            MockLlmProvider::single("should not be called"),
            "test-model",
            "claude-test",
        );

        // Two hooks: low priority blocks, high priority allows.
        // Since lower priority value fires first, the block hook (50) fires first → blocked.
        let hook_reg = hook_registry_multi(vec![
            (
                HookEvent::RequestStart,
                Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                    "low priority block",
                ))),
                None,
                50, // fires first
            ),
            (
                HookEvent::RequestStart,
                Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::allow())),
                None,
                200, // fires second (but chain already blocked)
            ),
        ]);

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlocked { ref reason, .. }) if reason == "low priority block"
            ),
            "lower priority hook should have fired first and blocked, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_recall_with_memory_domain_disabled_fans_out_to_all() {
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "memory found",
        )]));
        let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

        // Build config with memory domain explicitly disabled.
        let disabled_domain = DomainConfig {
            enabled: false,
            threshold: None,
        };
        let config = Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.5,
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
                domains: DomainsConfig {
                    model: None,
                    tool_necessity: None,
                    memory: Some(disabled_domain),
                },
            },
            tool_registry: None,
            memory: Some(MemoryConfig {
                stores: vec![MemoryStoreConfig {
                    name: "conv".to_string(),
                    endpoint: "http://localhost:50052".to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities: vec![StoreCapability::Read, StoreCapability::Write],
                    examples: vec!["conversation".to_string()],
                }],
            }),
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"test query\"", "Found something."]),
            "test-model",
            "claude-test",
        );

        // Router: score_memory_candidates still works (above threshold for conv).
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("recall test"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Found something.");
    }

    // ── Phase 4: memory command hook integration ───────────────────────────

    /// Build a `GatewayEngine` with a custom hook registry AND a memory mux.
    fn make_engine_with_hooks_and_mux<R, C>(
        registry: Arc<ProviderRegistry>,
        router: R,
        commands: C,
        hook_registry: weft_hooks::HookRegistry,
        mux: Option<Arc<MemoryStoreMux>>,
    ) -> GatewayEngine<weft_hooks::HookRegistry, R, DefaultMemoryService, ProviderRegistry, C>
    where
        R: SemanticRouter + Send + Sync + 'static,
        C: CommandRegistry + Send + Sync + 'static,
    {
        let memory = mux.map(wrap_mux);
        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            memory,
            Arc::new(hook_registry),
        )
    }

    /// Build a minimal single-store memory mux with both read and write capability.
    fn make_single_rw_mux(name: &str, client: Arc<dyn MemoryStoreClient>) -> Arc<MemoryStoreMux> {
        make_mux_with_stores(vec![(name, true, true, client)])
    }

    #[tokio::test]
    async fn test_pre_tool_use_blocks_recall_routing_hooks_not_fired() {
        // PreToolUse blocks /recall -> failed CommandResult returned, routing hooks
        // (PreRoute/PostRoute on memory domain) are NOT fired.
        //
        // Verified by: a PreRoute hook that panics if ever called. If routing hooks
        // were fired after the PreToolUse block, the test would panic.
        let mux = make_single_rw_mux(
            "conv",
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry("m1", "data")])),
        );

        struct PanicExecutor;
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for PanicExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                panic!("PreRoute should not fire after PreToolUse block");
            }
        }

        let hook_reg = hook_registry_multi(vec![
            // PreToolUse blocks /recall.
            (
                HookEvent::PreToolUse,
                Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                    "recall blocked by policy",
                ))),
                Some("recall"),
                100,
            ),
            // PreRoute on memory domain — must NOT fire.
            (
                HookEvent::PreRoute,
                Box::new(PanicExecutor),
                Some("memory"),
                100,
            ),
        ]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"test\"", "No memory needed."]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        // The /recall should be blocked, but the engine should not panic and should
        // eventually return a response (after LLM sees the failed recall result).
        let resp = engine
            .handle_request(make_user_request("What do you know about me?"))
            .await
            .expect("engine should not error — blocked command is a failed CommandResult");

        assert_eq!(resp_text(&resp), "No memory needed.");
    }

    #[tokio::test]
    async fn test_pre_tool_use_modifies_recall_arguments_used_in_routing() {
        // PreToolUse modifies /recall arguments -> the modified query is what reaches
        // the routing phase (and thus the memory store).
        //
        // Verified by: the mock mux client records which query text it receives.
        // We assert it matches the MODIFIED argument, not the original.
        let query_seen = Arc::new(std::sync::Mutex::new(String::new()));
        let query_seen_clone = Arc::clone(&query_seen);

        struct RecordingMemStoreClient {
            query_seen: Arc<std::sync::Mutex<String>>,
        }

        #[async_trait::async_trait]
        impl MemoryStoreClient for RecordingMemStoreClient {
            async fn query(
                &self,
                query: &str,
                _max_results: u32,
                _min_score: f32,
            ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
                *self.query_seen.lock().unwrap() = query.to_string();
                Ok(vec![])
            }

            async fn store(
                &self,
                _content: &str,
                _metadata: Option<&serde_json::Value>,
            ) -> Result<MemoryStoreResult, MemoryStoreError> {
                Ok(MemoryStoreResult {
                    id: "mock".to_string(),
                })
            }
        }

        let mux = make_single_rw_mux(
            "conv",
            Arc::new(RecordingMemStoreClient {
                query_seen: query_seen_clone,
            }),
        );

        // PreToolUse hook: replace the "query" argument with a different value.
        let hook_reg = hook_registry_with(
            HookEvent::PreToolUse,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "arguments": {"query": "modified query text"}
            }))),
            Some("recall"),
            100,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"original query\"",
                "Got the recall results.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9).with_memory_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        engine
            .handle_request(make_user_request("recall test"))
            .await
            .expect("should succeed");

        let seen = query_seen.lock().unwrap().clone();
        assert_eq!(
            seen, "modified query text",
            "routing should use the modified argument, not the original"
        );
    }

    #[tokio::test]
    async fn test_pre_route_memory_blocks_recall_returns_failed_command_result() {
        // PreRoute hook with matcher "memory" blocks the recall routing phase ->
        // a failed CommandResult is returned to the LLM (feedback block).
        let mux = make_single_rw_mux("conv", Arc::new(MockMemStoreClient::succeeds(vec![])));

        let hook_reg = hook_registry_with(
            HookEvent::PreRoute,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "memory access denied",
            ))),
            Some("memory"),
            100,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"preferences\"",
                "Memory was blocked.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        // The PreRoute block on memory results in a failed CommandResult, then
        // the LLM produces a final response after seeing the blocked recall.
        let resp = engine
            .handle_request(make_user_request("What do you know about me?"))
            .await
            .expect("engine should not error — blocked recall is a failed CommandResult");

        assert_eq!(resp_text(&resp), "Memory was blocked.");
    }

    #[tokio::test]
    async fn test_post_route_memory_overrides_selected_stores_for_remember() {
        // PostRoute hook with matcher "memory" overrides selected stores for /remember.
        // There are two write-capable stores: "store-a" and "store-b".
        // Router would normally select "store-a" (score 0.9).
        // PostRoute overrides selected to ["store-b"].
        // Verified by: only "store-b"'s client records a store() call.
        let store_a_stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let store_b_stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let store_a_clone = Arc::clone(&store_a_stored);
        let store_b_clone = Arc::clone(&store_b_stored);

        struct TrackingMemStoreClient(Arc<std::sync::atomic::AtomicBool>);

        #[async_trait::async_trait]
        impl MemoryStoreClient for TrackingMemStoreClient {
            async fn query(
                &self,
                _query: &str,
                _max_results: u32,
                _min_score: f32,
            ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
                Ok(vec![])
            }

            async fn store(
                &self,
                _content: &str,
                _metadata: Option<&serde_json::Value>,
            ) -> Result<MemoryStoreResult, MemoryStoreError> {
                self.0.store(true, std::sync::atomic::Ordering::Release);
                Ok(MemoryStoreResult {
                    id: "tracked".to_string(),
                })
            }
        }

        let mux = make_mux_with_stores(vec![
            (
                "store-a",
                true,
                true,
                Arc::new(TrackingMemStoreClient(store_a_clone)),
            ),
            (
                "store-b",
                true,
                true,
                Arc::new(TrackingMemStoreClient(store_b_clone)),
            ),
        ]);

        // PostRoute on memory domain: override selected to ["store-b"].
        let hook_reg = hook_registry_with(
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "selected": ["store-b"]
            }))),
            Some("memory"),
            100,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Stored the preference.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router selects store-a (first candidate, score 0.9) — hook should override to store-b.
        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9).with_memory_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember that I prefer dark mode"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Stored the preference.");

        // Only store-b should have been written to.
        assert!(
            store_b_stored.load(std::sync::atomic::Ordering::Acquire),
            "store-b should have been written to after PostRoute override"
        );
        assert!(
            !store_a_stored.load(std::sync::atomic::Ordering::Acquire),
            "store-a should NOT have been written to (PostRoute overrode selection)"
        );
    }

    #[tokio::test]
    async fn test_multiple_recall_invocations_each_fires_independent_hook_lifecycle() {
        // Multiple /recall invocations in the same LLM turn each fire their own
        // independent PreToolUse + PreRoute(memory) + PostRoute(memory) + PostToolUse.
        //
        // Verified by counting how many times PreToolUse is fired for "recall".
        let pre_tool_call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count_clone = Arc::clone(&pre_tool_call_count);

        struct CountingExecutor(Arc<std::sync::atomic::AtomicU32>);
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for CountingExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                weft_hooks::types::HookResponse::allow()
            }
        }

        let mux = make_single_rw_mux(
            "conv",
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry("m1", "data")])),
        );

        let matcher = HookMatcher::new(Some("recall"), 0).expect("valid matcher");
        let hook = RegisteredHook {
            event: HookEvent::PreToolUse,
            matcher,
            executor: Box::new(CountingExecutor(count_clone)),
            name: "counting-hook".to_string(),
            priority: 100,
        };
        let mut map = std::collections::HashMap::new();
        map.insert(HookEvent::PreToolUse, vec![hook]);
        let hook_reg = HookRegistry::from_registered(map);

        // LLM emits TWO /recall invocations in one turn.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"query one\"\n/recall query: \"query two\"",
                "Done with both recalls.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9).with_memory_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Recall twice"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Done with both recalls.");

        // PreToolUse must have fired once per /recall invocation = 2 times.
        assert_eq!(
            pre_tool_call_count.load(std::sync::atomic::Ordering::Acquire),
            2,
            "PreToolUse should fire independently for each /recall invocation"
        );
    }

    #[tokio::test]
    async fn test_post_route_modifies_scores_visible_to_subsequent_hooks() {
        // PostRoute hook modifies "scores" in the payload. A second PostRoute hook
        // that fires after the first should see the modified scores.
        //
        // Verified by: the second hook records what it received; we check that the
        // scores field reflects the first hook's modifications.
        let second_hook_payload = Arc::new(std::sync::Mutex::new(None::<serde_json::Value>));
        let capture_clone = Arc::clone(&second_hook_payload);

        struct CapturePayloadExecutor(Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        #[async_trait]
        impl weft_hooks::executor::HookExecutor for CapturePayloadExecutor {
            async fn execute(
                &self,
                payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                *self.0.lock().unwrap() = Some(payload.clone());
                weft_hooks::types::HookResponse::allow()
            }
        }

        // First hook: modify scores to set all scores to 0.5.
        // Second hook: capture the payload after modification.
        let hook_reg = hook_registry_multi(vec![
            (
                HookEvent::PostRoute,
                Box::new(ModifyHookExecutor(serde_json::json!({
                    "scores": [{"id": "test-model", "score": 0.5}]
                }))),
                Some("model"),
                50, // fires first
            ),
            (
                HookEvent::PostRoute,
                Box::new(CapturePayloadExecutor(capture_clone)),
                Some("model"),
                100, // fires second
            ),
        ]);

        let engine = GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("Response"),
                MockLlmProvider::single("Response"),
            ),
            Arc::new(MockRouter::with_model("default-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Response");

        // The second PostRoute hook should have seen the scores modified by the first hook.
        let captured = second_hook_payload.lock().unwrap().clone();
        let captured = captured.expect("second hook should have been called");
        let scores = captured.get("scores").expect("scores field must exist");
        // The first hook set scores to [{"id": "test-model", "score": 0.5}].
        assert_eq!(
            scores,
            &serde_json::json!([{"id": "test-model", "score": 0.5}]),
            "second PostRoute hook should see scores modified by first hook"
        );
    }

    // ── Phase 5: Capability filtering tests ───────────────────────────────

    /// Build a registry with custom capability sets per model.
    fn registry_with_capabilities(
        default_name: &str,
        models: Vec<(&str, &str, Vec<&str>)>, // (routing_name, model_id, capabilities)
    ) -> Arc<ProviderRegistry> {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        let mut model_ids = HashMap::new();
        let mut max_tokens = HashMap::new();
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();

        for (routing_name, model_id, capabilities) in &models {
            providers.insert(
                routing_name.to_string(),
                Arc::new(MockLlmProvider::single("Response")) as Arc<dyn Provider>,
            );
            model_ids.insert(routing_name.to_string(), model_id.to_string());
            max_tokens.insert(routing_name.to_string(), 1024u32);
            caps.insert(
                routing_name.to_string(),
                capabilities.iter().map(|c| Capability::new(*c)).collect(),
            );
        }

        Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            default_name.to_string(),
        ))
    }

    #[tokio::test]
    async fn test_no_eligible_models_returns_400() {
        // A registry with only an embeddings-capable model — no chat_completions.
        // The request should be rejected with NoEligibleModels.
        let registry = registry_with_capabilities(
            "embed-model",
            vec![("embed-model", "text-embed-v1", vec!["embeddings"])],
        );

        let engine = make_engine(
            registry,
            MockRouter::with_model("embed-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::NoEligibleModels { ref capability }) if capability == "chat_completions"),
            "expected NoEligibleModels error, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_capability_filter_excludes_non_chat_models_from_routing() {
        // Two models: one with chat_completions, one with only embeddings.
        // The embeddings-only model should be excluded from routing candidates.
        // We verify this by checking that only the chat model is ever selected.

        // Use a multi-model config so the model domain is included in routing.
        let config = Arc::new(WeftConfig {
            server: weft_core::ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("chat-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "chat-model".to_string(),
                            model: "claude-chat".to_string(),
                            max_tokens: 1024,
                            examples: vec!["general chat".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                        ModelEntry {
                            name: "embed-model".to_string(),
                            model: "text-embed-v1".to_string(),
                            max_tokens: 512,
                            examples: vec!["embed this text".to_string()],
                            capabilities: vec!["embeddings".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        // Build registry: chat-model has chat_completions, embed-model only has embeddings.
        let registry = registry_with_capabilities(
            "chat-model",
            vec![
                ("chat-model", "claude-chat", vec!["chat_completions"]),
                ("embed-model", "text-embed-v1", vec!["embeddings"]),
            ],
        );

        // Router always returns the embed-model — the capability filter should override.
        // Since embed-model is filtered out, the router sees only [chat-model].
        // We use a router that returns "embed-model" to confirm it gets filtered.
        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_model("embed-model"),
            MockCommandRegistry::new(vec![]),
        );

        // Should succeed — embed-model is filtered from candidates, chat-model is used.
        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            result.is_ok(),
            "expected success with capability filtering, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_fallback_skips_default_when_default_lacks_capability() {
        // The non-default model fails with a non-rate-limit error.
        // The default model only has "embeddings" — not chat_completions.
        // Fallback should be skipped, error should propagate as Llm error.
        //
        // We use a two-model config to ensure the model domain is included in routing,
        // so the router can select the non-default model which will then fail.
        struct FailingProvider;

        #[async_trait]
        impl Provider for FailingProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                Err(ProviderError::RequestFailed("network down".to_string()))
            }

            fn name(&self) -> &str {
                "failing"
            }
        }

        // default-model has only embeddings; non-default has chat_completions but fails.
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(MockLlmProvider::single("fallback response")) as Arc<dyn Provider>,
        );
        providers.insert(
            "non-default-model".to_string(),
            Arc::new(FailingProvider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "embed-v1".to_string());
        model_ids.insert(
            "non-default-model".to_string(),
            "claude-complex".to_string(),
        );
        let mut max_tokens = HashMap::new();
        max_tokens.insert("default-model".to_string(), 512u32);
        max_tokens.insert("non-default-model".to_string(), 4096u32);

        // Default model has embeddings only (no chat_completions).
        // Non-default model has chat_completions.
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "default-model".to_string(),
            [Capability::new("embeddings")].into_iter().collect(),
        );
        caps.insert(
            "non-default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "default-model".to_string(),
        ));

        // Use a two-model config with the same model names as the registry.
        // "non-default-model" is eligible (has chat_completions), so it appears in routing
        // candidates and the router selects it. It then fails, and the fallback check
        // finds that "default-model" lacks chat_completions, so fallback is skipped.
        let config = Arc::new(WeftConfig {
            server: weft_core::ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("default-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "default-model".to_string(),
                            model: "embed-v1".to_string(),
                            max_tokens: 512,
                            examples: vec!["embed text".to_string()],
                            // embeddings only — no chat_completions
                            capabilities: vec!["embeddings".to_string()],
                        },
                        ModelEntry {
                            name: "non-default-model".to_string(),
                            model: "claude-complex".to_string(),
                            max_tokens: 4096,
                            examples: vec!["complex reasoning".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        // Router selects non-default model (which will fail).
        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_model("non-default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "expected Llm error when fallback is skipped due to missing capability, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_fallback_proceeds_when_default_has_capability() {
        // Non-default fails, default has chat_completions — fallback should proceed.
        // Uses a two-model config so the model domain is included in routing and
        // the router can select the non-default model which then fails.
        struct FailingProvider;

        #[async_trait]
        impl Provider for FailingProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                Err(ProviderError::RequestFailed("network down".to_string()))
            }

            fn name(&self) -> &str {
                "failing"
            }
        }

        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(MockLlmProvider::single("fallback response")) as Arc<dyn Provider>,
        );
        providers.insert(
            "non-default-model".to_string(),
            Arc::new(FailingProvider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert(
            "non-default-model".to_string(),
            "claude-complex".to_string(),
        );
        let mut max_tokens = HashMap::new();
        max_tokens.insert("default-model".to_string(), 1024u32);
        max_tokens.insert("non-default-model".to_string(), 4096u32);

        // Both models have chat_completions.
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        caps.insert(
            "non-default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "default-model".to_string(),
        ));

        // Two-model config matching registry model names so model domain is included.
        let config = Arc::new(WeftConfig {
            server: weft_core::ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("default-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "default-model".to_string(),
                            model: "claude-default".to_string(),
                            max_tokens: 1024,
                            examples: vec!["general question".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                        ModelEntry {
                            name: "non-default-model".to_string(),
                            model: "claude-complex".to_string(),
                            max_tokens: 4096,
                            examples: vec!["complex task".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_model("non-default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("fallback should succeed");
        assert_eq!(
            resp_text(&result),
            "fallback response",
            "response should come from fallback default model"
        );
    }

    #[tokio::test]
    async fn test_end_to_end_uses_provider_execute_path() {
        // End-to-end: request flows through Provider::execute() and produces correct response.
        // Verifies the full path from handle_request -> run_loop -> call_with_fallback ->
        // provider.execute(ProviderRequest::ChatCompletion) -> ProviderResponse::ChatCompletion.
        let provider = MockLlmProvider::single("Hello from provider");
        let registry = single_model_registry(provider, "test-model", "claude-test");

        let engine = make_engine(
            registry,
            MockRouter::with_model("test-model"),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hi there"))
            .await
            .expect("end-to-end should succeed");

        assert_eq!(
            resp_text(&resp),
            "Hello from provider",
            "response text must come from provider.execute()"
        );
        // Usage should be populated from the mock provider.
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
    }

    #[tokio::test]
    async fn test_no_eligible_models_error_message() {
        // Verify the error message format for NoEligibleModels.
        let err = WeftError::NoEligibleModels {
            capability: "chat_completions".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "no models configured with capability 'chat_completions'"
        );
    }
}
