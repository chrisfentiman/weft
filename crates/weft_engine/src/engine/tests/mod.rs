/// Shared test infrastructure: mocks, helpers, config builders.
///
/// Sub-modules import via `use super::*` to get all helpers and mocks in scope.
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use weft_commands::{CommandError, CommandRegistry};
use weft_core::{
    ClassifierConfig, CommandAction, CommandDescription, CommandInvocation, CommandResult,
    CommandStub, ContentPart, DomainsConfig, GatewayConfig,
    ModelEntry, ModelRoutingInstruction, ProviderConfig, Role,
    RouterConfig, SamplingOptions, ServerConfig, Source, WeftConfig, WeftError,
    WeftMessage, WeftRequest, WeftResponse, WireFormat,
};
use weft_llm::{
    Capability, Provider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
    TokenUsage,
};
use weft_router::{
    RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
    SemanticRouter, build_memory_candidates,
};

// Re-export GatewayEngine from the parent engine module so sub-modules can use it.
pub use super::GatewayEngine;

pub mod hook_tests;
pub mod memory_tests;
pub mod provider_tests;
pub mod request_tests;
pub mod routing_tests;

// ── Mock implementations ───────────────────────────────────────────────

/// A mock provider with configurable responses.
pub struct MockLlmProvider {
    /// Responses to return in order. Repeats last on exhaustion.
    pub responses: std::sync::Mutex<Vec<String>>,
}

impl MockLlmProvider {
    pub fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: std::sync::Mutex::new(
                responses.into_iter().map(str::to_string).collect(),
            ),
        }
    }

    pub fn single(response: &str) -> Self {
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
pub struct RecordingLlmProvider {
    pub response: String,
    pub recorded_model: std::sync::Mutex<Option<String>>,
}

impl RecordingLlmProvider {
    pub fn new(response: &str) -> Self {
        Self {
            response: response.to_string(),
            recorded_model: std::sync::Mutex::new(None),
        }
    }

    pub fn recorded_model(&self) -> Option<String> {
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
pub struct RateLimitedLlmProvider;

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
pub struct FailingLlmProvider;

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
pub struct MockRouter {
    /// Score assigned to all Commands domain candidates.
    pub command_score: f32,
    /// Model to select (None = use default).
    pub model_decision: Option<String>,
    /// Tools needed decision.
    pub tools_needed: Option<bool>,
    /// Score assigned to the first memory candidate when `score_memory_candidates()` is called.
    /// Other candidates get score 0.0. `None` means return ModelNotLoaded error (simulate
    /// router unavailable). Default: Some(0.9) — first candidate wins.
    pub memory_score: Option<f32>,
}

impl MockRouter {
    pub fn with_score(score: f32) -> Self {
        Self {
            command_score: score,
            model_decision: None,
            tools_needed: None,
            memory_score: Some(0.9),
        }
    }

    pub fn with_model(model: &str) -> Self {
        Self {
            command_score: 1.0,
            model_decision: Some(model.to_string()),
            tools_needed: None,
            memory_score: Some(0.9),
        }
    }

    pub fn with_tools_needed(needed: bool) -> Self {
        Self {
            command_score: 1.0,
            model_decision: None,
            tools_needed: Some(needed),
            memory_score: Some(0.9),
        }
    }

    /// Configure memory scoring: returns the given score for the first candidate,
    /// 0.0 for all others. Useful for testing threshold behavior.
    pub fn with_memory_score(mut self, score: f32) -> Self {
        self.memory_score = Some(score);
        self
    }

    /// Configure router unavailable for memory scoring (returns ModelNotLoaded).
    pub fn with_memory_unavailable(mut self) -> Self {
        self.memory_score = None;
        self
    }

    pub fn failing() -> FailingRouter {
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

pub struct FailingRouter;

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
pub struct MockCommandRegistry {
    pub commands: Vec<CommandStub>,
    pub execute_result: std::sync::Mutex<Option<CommandResult>>,
    pub describe_result: std::sync::Mutex<Option<CommandDescription>>,
    pub fail_list: bool,
}

impl MockCommandRegistry {
    pub fn new(commands: Vec<(&str, &str)>) -> Self {
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

    pub fn with_execute_result(self, result: CommandResult) -> Self {
        *self.execute_result.lock().unwrap() = Some(result);
        self
    }

    pub fn with_describe_result(self, desc: CommandDescription) -> Self {
        *self.describe_result.lock().unwrap() = Some(desc);
        self
    }

    pub fn with_list_failure() -> Self {
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
pub fn test_config() -> Arc<WeftConfig> {
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
pub fn multi_model_config() -> Arc<WeftConfig> {
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

pub fn make_user_request(content: &str) -> WeftRequest {
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
pub fn default_caps(model_name: &str) -> HashMap<String, HashSet<Capability>> {
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
pub fn single_model_registry(
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
pub fn two_model_registry(
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

pub fn make_engine<R, C>(
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

pub fn make_engine_with_config<R, C>(
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
pub fn resp_text(resp: &WeftResponse) -> &str {
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
