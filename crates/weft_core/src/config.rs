//! Configuration types for the Weft gateway.
//!
//! Configuration is loaded from a TOML file. The `api_key` field supports
//! an `env:VAR_NAME` prefix for loading secrets from environment variables.

/// Top-level configuration, deserialized from TOML.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeftConfig {
    pub server: ServerConfig,
    pub tool_registry: Option<ToolRegistryConfig>,
    pub gateway: GatewayConfig,
    /// Router configuration — the single source of truth for model routing
    /// and classification. Replaces the old `[llm]` and `[classifier]` sections.
    pub router: RouterConfig,
    /// Memory store configuration. Optional -- absent means no memory stores.
    pub memory: Option<MemoryConfig>,
    /// Hook configuration. Empty vec means no hooks.
    #[serde(default)]
    pub hooks: Vec<HookConfig>,
    /// Maximum number of LLM regeneration attempts when a PreResponse hook blocks.
    /// Default: 2. Max: 5.
    #[serde(default = "default_max_pre_response_retries")]
    pub max_pre_response_retries: u32,
    /// Maximum concurrent RequestEnd hook tasks. Excess tasks are dropped with a warning.
    /// Default: 64.
    #[serde(default = "default_request_end_concurrency")]
    pub request_end_concurrency: usize,
    /// Event log backend selection. Optional — absent defaults to in-memory.
    pub event_log: Option<EventLogConfig>,
}

/// Event log backend configuration.
///
/// Controls which `EventLog` implementation the binary constructs at startup.
/// Absent or `backend = "memory"` uses `InMemoryEventLog` (no persistence).
/// `backend = "postgres"` requires `database_url` and uses `PostgresEventLog`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EventLogConfig {
    /// Which backend to use: `"memory"` (default) or `"postgres"`.
    #[serde(default = "default_event_log_backend")]
    pub backend: String,
    /// PostgreSQL connection URL. Required when `backend = "postgres"`.
    /// Supports `env:VAR_NAME` prefix for loading from the environment.
    pub database_url: Option<String>,
}

fn default_event_log_backend() -> String {
    "memory".to_string()
}

fn default_max_pre_response_retries() -> u32 {
    2
}

fn default_request_end_concurrency() -> usize {
    64
}

impl WeftConfig {
    /// Resolve environment variable references in sensitive fields.
    ///
    /// Call this once after deserializing the config. Fields with the prefix
    /// `env:` are replaced with the value of the named environment variable.
    ///
    /// Returns an error if any referenced environment variable is not set.
    pub fn resolve(&mut self) -> Result<(), String> {
        for provider in &mut self.router.providers {
            provider.api_key = resolve_env_var(&provider.api_key)
                .map_err(|e| format!("router.providers[{}].api_key: {e}", provider.name))?;
        }
        // Resolve env: references in hook secrets.
        for (i, hook) in self.hooks.iter_mut().enumerate() {
            if let Some(ref secret) = hook.secret.clone()
                && secret.starts_with("env:")
            {
                let resolved = resolve_env_var(secret).map_err(|e| {
                    let var_name = secret.strip_prefix("env:").unwrap_or(secret);
                    format!(
                        "hooks[{i}]: secret references env var '{var_name}' which is not set: {e}"
                    )
                })?;
                hook.secret = Some(resolved);
            }
        }
        // Resolve env: references in event_log.database_url.
        if let Some(ref mut el) = self.event_log
            && let Some(ref url) = el.database_url.clone()
            && url.starts_with("env:")
        {
            let resolved =
                resolve_env_var(url).map_err(|e| format!("event_log.database_url: {e}"))?;
            el.database_url = Some(resolved);
        }
        Ok(())
    }

    /// Validate the configuration after resolving environment variables.
    ///
    /// Checks all consistency constraints that cannot be expressed in the type system.
    /// Must be called after `resolve()`.
    pub fn validate(&self) -> Result<(), String> {
        self.router.validate()?;
        if let Some(ref memory) = self.memory {
            memory.validate()?;
        }
        self.validate_hooks()?;
        if let Some(ref el) = self.event_log {
            if el.backend == "postgres" && el.database_url.is_none() {
                return Err(
                    "event_log.database_url is required when event_log.backend = \"postgres\""
                        .to_string(),
                );
            }
            if el.backend != "memory" && el.backend != "postgres" {
                return Err(format!(
                    "event_log.backend must be \"memory\" or \"postgres\", got \"{}\"",
                    el.backend
                ));
            }
        }
        Ok(())
    }

    /// Returns the effective `max_pre_response_retries` value, clamped to 5.
    ///
    /// Values above 5 are silently clamped here. The warning is emitted by
    /// `validate_hooks` at config validation time.
    pub fn effective_max_pre_response_retries(&self) -> u32 {
        self.max_pre_response_retries.min(5)
    }

    fn validate_hooks(&self) -> Result<(), String> {
        // Enforce request_end_concurrency > 0.
        if self.request_end_concurrency == 0 {
            return Err("request_end_concurrency must be > 0".to_string());
        }

        // Clamp max_pre_response_retries to 5. Values above 5 are clamped with a warning.
        if self.max_pre_response_retries > 5 {
            tracing::warn!(
                value = self.max_pre_response_retries,
                "max_pre_response_retries exceeds maximum of 5 — clamping to 5"
            );
        }

        for (i, hook) in self.hooks.iter().enumerate() {
            match hook.hook_type {
                HookType::Rhai => {
                    if hook.script.is_none() {
                        return Err(format!("hooks[{i}]: rhai hook requires 'script' field"));
                    }
                    // Clamp Rhai timeout to 5000ms (no hard error, just cap).
                    // Capping is applied at registry construction time.
                }
                HookType::Http => {
                    let url = hook.url.as_deref().ok_or_else(|| {
                        format!("hooks[{i}]: http hook requires valid 'url' field")
                    })?;
                    // Validate URL starts with a valid scheme (http:// or https://).
                    // Full URL parsing is done by the reqwest client at registry construction time.
                    if !url.starts_with("http://") && !url.starts_with("https://") {
                        return Err(format!(
                            "hooks[{i}]: http hook requires valid 'url' field (must start with http:// or https://)"
                        ));
                    }
                    // Clamp HTTP timeout to 30000ms (applied at registry construction).
                }
                HookType::Weft => {
                    // agent field validated but hook is skipped at registry time.
                    if hook.agent.is_none() {
                        tracing::warn!(
                            "hooks[{i}]: weft hook missing 'agent' field — hook will be skipped"
                        );
                    }
                    tracing::warn!(
                        "hooks[{i}]: weft-type hooks are reserved for future implementation"
                    );
                }
            }
        }

        Ok(())
    }
}

/// Resolve a single value that may contain an `env:VAR_NAME` prefix.
///
/// This is an internal utility. It is `pub(crate)` so that other modules
/// within `weft_core` can reuse it, but it is not part of the public API.
pub(crate) fn resolve_env_var(value: &str) -> Result<String, String> {
    if let Some(var_name) = value.strip_prefix("env:") {
        std::env::var(var_name).map_err(|_| format!("environment variable '{var_name}' is not set"))
    } else {
        Ok(value.to_string())
    }
}

/// Router configuration — the single source of truth for model routing and classification.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RouterConfig {
    /// Classifier (ONNX bi-encoder) configuration. Previously a top-level [classifier] section.
    pub classifier: ClassifierConfig,
    /// Default model name. Must reference a model name from one of the providers.
    /// If absent, the first model of the first provider is the default.
    pub default_model: Option<String>,
    /// Provider definitions. Each provider is an API endpoint with one or more models.
    /// Defaults to empty — validation enforces at least one provider.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// Whether to skip command injection when the router determines tools aren't needed.
    /// Default: true.
    #[serde(default = "default_skip_tools_when_unnecessary")]
    pub skip_tools_when_unnecessary: bool,
    /// Per-domain configuration overrides.
    #[serde(default)]
    pub domains: DomainsConfig,
}

fn default_skip_tools_when_unnecessary() -> bool {
    true
}

impl RouterConfig {
    /// Validate router config constraints.
    pub fn validate(&self) -> Result<(), String> {
        // 1. At least one provider.
        if self.providers.is_empty() {
            return Err("router.providers must have at least one entry".to_string());
        }

        // 2. Each provider must have at least one model.
        for provider in &self.providers {
            if provider.models.is_empty() {
                return Err(format!(
                    "provider '{}' must have at least one model",
                    provider.name
                ));
            }
        }

        // 3. Provider names must be unique.
        let mut provider_names = std::collections::HashSet::new();
        for provider in &self.providers {
            if !provider_names.insert(provider.name.as_str()) {
                return Err(format!("duplicate provider name: '{}'", provider.name));
            }
        }

        // 4. Model names must be globally unique across all providers.
        let mut model_names = std::collections::HashSet::new();
        for provider in &self.providers {
            for model in &provider.models {
                if !model_names.insert(model.name.as_str()) {
                    return Err(format!(
                        "duplicate model name: '{}' (model names must be globally unique across all providers)",
                        model.name
                    ));
                }
            }
        }

        // 5. If default_model is specified, it must match an existing model name.
        if let Some(ref default) = self.default_model
            && !model_names.contains(default.as_str())
        {
            return Err(format!(
                "router.default_model '{}' does not match any model name",
                default
            ));
        }

        // 6. Each model must have at least one example.
        for provider in &self.providers {
            for model in &provider.models {
                if model.examples.is_empty() {
                    return Err(format!(
                        "model '{}' under provider '{}' must have at least one example",
                        model.name, provider.name
                    ));
                }
            }
        }

        // 7. wire_format = custom requires wire_script; non-custom must not have wire_script.
        for provider in &self.providers {
            match &provider.wire_format {
                WireFormat::Custom => {
                    if provider.wire_script.is_none() {
                        return Err(format!(
                            "provider '{}': custom wire_format requires 'wire_script' field",
                            provider.name
                        ));
                    }
                }
                WireFormat::OpenAI | WireFormat::Anthropic => {
                    if provider.wire_script.is_some() {
                        return Err(format!(
                            "provider '{}': wire_script is only valid with wire_format = 'custom'",
                            provider.name
                        ));
                    }
                }
            }
        }

        // 8. Each model's capabilities must have at least one entry.
        for provider in &self.providers {
            for model in &provider.models {
                if model.capabilities.is_empty() {
                    return Err(format!(
                        "model '{}': capabilities must have at least one entry",
                        model.name
                    ));
                }
            }
        }

        Ok(())
    }

    /// Resolve the effective default model name.
    ///
    /// Returns the configured `default_model` if set, otherwise the name of the
    /// first model of the first provider. Panics if providers is empty (call
    /// `validate()` first).
    pub fn effective_default_model(&self) -> &str {
        if let Some(ref default) = self.default_model {
            default.as_str()
        } else {
            // Validated: at least one provider with at least one model.
            self.providers[0].models[0].name.as_str()
        }
    }

    /// Flatten the nested provider/model structure into a list of `ResolvedModel`.
    ///
    /// Combines each provider's connection info with each of its model entries.
    /// Called once at startup after `resolve()` and `validate()`.
    pub fn resolve_models(&self) -> Vec<ResolvedModel> {
        let mut resolved = Vec::new();
        for provider in &self.providers {
            for model in &provider.models {
                resolved.push(ResolvedModel {
                    name: model.name.clone(),
                    model: model.model.clone(),
                    max_tokens: model.max_tokens,
                    examples: model.examples.clone(),
                    wire_format: provider.wire_format.clone(),
                    api_key: provider.api_key.clone(),
                    base_url: provider.base_url.clone(),
                    provider_name: provider.name.clone(),
                    wire_script: provider.wire_script.clone(),
                    capabilities: model.capabilities.clone(),
                });
            }
        }
        resolved
    }
}

/// Per-domain threshold and enablement overrides.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct DomainsConfig {
    /// Model routing domain config.
    pub model: Option<DomainConfig>,
    /// Tool necessity domain config.
    pub tool_necessity: Option<DomainConfig>,
    /// Memory routing domain config (interface only -- no memory stores yet).
    pub memory: Option<DomainConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DomainConfig {
    /// Score threshold for this domain.
    /// For the Commands domain: overrides `classifier.threshold` for filtering.
    /// For the Model domain: acts as a confidence gate -- if no model scores above this
    /// threshold, the default model is used instead. If absent, always pick the highest-scoring model.
    /// For ToolNecessity/Memory: overrides classifier.threshold.
    pub threshold: Option<f32>,
    /// Whether this domain is enabled. Default: true.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

/// An LLM provider -- an API endpoint that serves one or more models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ProviderConfig {
    /// Unique name for this provider (e.g., "anthropic", "local-ollama").
    pub name: String,
    /// Wire format -- determines the API protocol used.
    pub wire_format: WireFormat,
    /// API key for this provider. Supports `env:VAR_NAME` prefix.
    /// For local providers (e.g., Ollama) that don't require authentication,
    /// use a literal string like `"unused"`. The value is passed through to the
    /// provider's HTTP client (e.g., `Authorization: Bearer unused`). Local
    /// providers typically ignore this header, so any non-empty string is acceptable.
    /// The field is required (not Optional) to keep the config schema simple.
    pub api_key: String,
    /// Optional base URL override. Required for local/custom endpoints (e.g., Ollama).
    pub base_url: Option<String>,
    /// Path to Rhai wire format script. Required when `wire_format` is `custom`.
    /// Relative paths are resolved from the current working directory at startup.
    pub wire_script: Option<String>,
    /// Models served by this provider. Must have at least one.
    pub models: Vec<ModelEntry>,
}

/// A model available for routing, nested under a provider.
///
/// The model inherits the provider's API endpoint (wire_format, api_key, base_url).
/// The `model` field is sent to the provider API to select which model to use.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelEntry {
    /// Unique routing name for this model (e.g., "complex", "fast").
    /// Used as the routing candidate ID. Must be globally unique across all providers.
    pub name: String,
    /// Model identifier sent to the provider API.
    /// e.g., "claude-sonnet-4-20250514", "gpt-4o", "llama3.2"
    pub model: String,
    /// Maximum tokens in response.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Example queries that this model is best suited for.
    /// The router embeds all examples, averages them into a centroid vector,
    /// and routes based on distance to the centroid.
    /// Must contain at least one example.
    pub examples: Vec<String>,
    /// Capabilities this model supports.
    /// Default: `["chat_completions"]` -- a model that doesn't declare capabilities
    /// is assumed to support chat completions only.
    #[serde(default = "default_model_capabilities")]
    pub capabilities: Vec<String>,
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_model_capabilities() -> Vec<String> {
    vec!["chat_completions".to_string()]
}

/// A fully-resolved model combining provider endpoint info with model-specific config.
/// Not deserialized from TOML -- constructed during config resolution via
/// `RouterConfig::resolve_models()`.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The model's routing name (from ModelEntry.name).
    pub name: String,
    /// The model identifier sent to the API (from ModelEntry.model).
    pub model: String,
    /// Max tokens (from ModelEntry.max_tokens).
    pub max_tokens: u32,
    /// Routing examples (from ModelEntry.examples).
    pub examples: Vec<String>,
    /// Wire format (from ProviderConfig.wire_format).
    pub wire_format: WireFormat,
    /// Resolved API key (from ProviderConfig.api_key, after env resolution).
    pub api_key: String,
    /// Base URL (from ProviderConfig.base_url).
    pub base_url: Option<String>,
    /// Provider name (from ProviderConfig.name) -- for logging/diagnostics.
    pub provider_name: String,
    /// Path to Rhai wire script (only for Custom wire format).
    pub wire_script: Option<String>,
    /// Capabilities this model supports, as strings.
    pub capabilities: Vec<String>,
}

/// Wire format -- determines how requests are serialized and responses are parsed
/// for a provider's API.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WireFormat {
    /// OpenAI Chat Completions API format.
    /// Used by OpenAI, Azure OpenAI, Ollama, vLLM, and most OpenAI-compatible providers.
    #[serde(rename = "openai")]
    OpenAI,
    /// Anthropic Messages API format.
    Anthropic,
    /// Custom wire format defined by a Rhai script.
    Custom,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClassifierConfig {
    /// Path to the ONNX model file.
    pub model_path: String,
    /// Path to the tokenizer JSON file.
    pub tokenizer_path: String,
    /// Relevance score threshold. Commands scoring below this are excluded.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Maximum number of commands to include regardless of threshold.
    #[serde(default = "default_max_commands")]
    pub max_commands: usize,
}

fn default_threshold() -> f32 {
    0.3
}

fn default_max_commands() -> usize {
    20
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ServerConfig {
    /// Bind address, e.g. "0.0.0.0:8080"
    pub bind_address: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ToolRegistryConfig {
    /// gRPC endpoint, e.g. "http://localhost:50051"
    pub endpoint: String,
    /// Connection timeout in milliseconds.
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    /// Request timeout in milliseconds.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
}

fn default_connect_timeout_ms() -> u64 {
    5000
}

fn default_request_timeout_ms() -> u64 {
    30000
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GatewayConfig {
    /// System prompt prepended to every conversation.
    pub system_prompt: String,
    /// Maximum iterations of the command loop before giving up.
    #[serde(default = "default_max_iterations")]
    pub max_command_iterations: u32,
    /// Overall request timeout in seconds. Covers the entire gateway loop
    /// including all LLM calls and command executions.
    /// Default: 300 (5 minutes).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

fn default_max_iterations() -> u32 {
    10
}

fn default_request_timeout_secs() -> u64 {
    300
}

// ── Memory configuration ─────────────────────────────────────────────────────

/// Memory configuration. Optional -- absent means no memory stores.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MemoryConfig {
    /// Memory stores to connect to.
    pub stores: Vec<MemoryStoreConfig>,
}

impl MemoryConfig {
    /// Validate memory configuration.
    ///
    /// Checks store name uniqueness, example presence, capability presence,
    /// and endpoint URI validity.
    pub fn validate(&self) -> Result<(), String> {
        // 1. Non-empty stores list is not required (memory section can have zero stores).
        // 2. Store names must be unique.
        let mut names = std::collections::HashSet::new();
        for store in &self.stores {
            if !names.insert(store.name.as_str()) {
                return Err(format!("duplicate memory store name: '{}'", store.name));
            }
        }

        for store in &self.stores {
            // 3. Each store must have at least one example.
            if store.examples.is_empty() {
                return Err(format!(
                    "memory store '{}' must have at least one example",
                    store.name
                ));
            }

            // 4. Each store's capabilities must be non-empty.
            if store.capabilities.is_empty() {
                return Err(format!(
                    "memory store '{}' must have at least one capability (read, write)",
                    store.name
                ));
            }

            // 5. Validate endpoint URI using tonic.
            tonic::transport::Endpoint::new(store.endpoint.clone())
                .map_err(|e| format!("invalid endpoint URI for store '{}': {}", store.name, e))?;
        }

        // 6. Warn if no stores have read or write capability.
        if !self.stores.is_empty() {
            let any_readable = self.stores.iter().any(|s| s.can_read());
            let any_writable = self.stores.iter().any(|s| s.can_write());
            if !any_readable {
                tracing::warn!("no memory stores have read capability — /recall has no targets");
            }
            if !any_writable {
                tracing::warn!("no memory stores have write capability — /remember has no targets");
            }
        }

        Ok(())
    }
}

/// What a memory store can do. Config-level declaration by the admin --
/// not part of the gRPC contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoreCapability {
    /// Store supports querying/retrieving memories.
    Read,
    /// Store supports persisting new memories.
    Write,
}

/// A single memory store endpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MemoryStoreConfig {
    /// Unique name for this store (e.g., "conversations", "code_knowledge").
    /// Used as the routing candidate ID in the Memory domain.
    pub name: String,
    /// gRPC endpoint (e.g., "http://localhost:50052").
    pub endpoint: String,
    /// Connection timeout in milliseconds.
    #[serde(default = "default_memory_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    /// Request timeout in milliseconds.
    ///
    /// **Deferred (v1):** This field is parsed and stored but NOT enforced
    /// per-request on the gRPC client. The overall gateway request timeout
    /// (`gateway.request_timeout_secs`) is the safety net. The field exists
    /// for forward compatibility: when per-store timeouts are implemented
    /// (via tonic's `timeout()` on the channel or per-request deadline),
    /// existing configs will already have the value.
    #[serde(default = "default_memory_request_timeout_ms")]
    pub request_timeout_ms: u64,
    /// Maximum number of memories to retrieve per query.
    #[serde(default = "default_max_results")]
    pub max_results: u32,
    /// What this store can do: `read`, `write`, or both.
    /// Determines whether the store is eligible for `/recall` (read) and
    /// `/remember` (write) operations. Defaults to `["read", "write"]` if omitted.
    #[serde(default = "default_capabilities")]
    pub capabilities: Vec<StoreCapability>,
    /// Example queries that this store is best suited for.
    /// Used by the semantic router to build centroid embeddings for the Memory domain.
    /// Must contain at least one example.
    pub examples: Vec<String>,
}

impl MemoryStoreConfig {
    /// Whether this store supports read (query) operations.
    pub fn can_read(&self) -> bool {
        self.capabilities.contains(&StoreCapability::Read)
    }

    /// Whether this store supports write (store) operations.
    pub fn can_write(&self) -> bool {
        self.capabilities.contains(&StoreCapability::Write)
    }
}

fn default_memory_connect_timeout_ms() -> u64 {
    5000
}

fn default_memory_request_timeout_ms() -> u64 {
    10000
}

fn default_max_results() -> u32 {
    5
}

fn default_capabilities() -> Vec<StoreCapability> {
    vec![StoreCapability::Read, StoreCapability::Write]
}

// ── Hook configuration ────────────────────────────────────────────────────────

/// Lifecycle events that hooks can attach to.
///
/// Lives in `weft_core::config` so `HookConfig` can reference it for
/// deserialization. Placing it in the `weft` binary crate would create a
/// circular dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    /// Request arrives at gateway, before any processing.
    RequestStart,
    /// Before each routing decision (fires per domain).
    PreRoute,
    /// After each routing decision (fires per domain).
    PostRoute,
    /// Before executing a single command invocation.
    PreToolUse,
    /// After command returns, before result injection. Cannot block.
    PostToolUse,
    /// Before building the final HTTP response.
    PreResponse,
    /// After response sent to client. Cannot block. Fire-and-forget.
    RequestEnd,
}

impl HookEvent {
    /// Whether this event supports blocking decisions.
    pub fn can_block(&self) -> bool {
        matches!(
            self,
            Self::RequestStart
                | Self::PreRoute
                | Self::PostRoute
                | Self::PreToolUse
                | Self::PreResponse
        )
    }

    /// Whether this event uses feedback-loop blocking (reason fed back to LLM)
    /// as opposed to hard blocking (HTTP error to client).
    ///
    /// For PreRoute/PostRoute, the blocking behavior depends on the trigger:
    /// - `request_start` trigger: hard block (403)
    /// - `recall_command` or `remember_command` trigger: feedback block (failed CommandResult)
    ///
    /// This method returns the DEFAULT behavior. The engine overrides for
    /// PreRoute/PostRoute based on the trigger at call time.
    pub fn is_feedback_block(&self) -> bool {
        matches!(self, Self::PreToolUse | Self::PreResponse)
    }
}

/// The routing domain for PreRoute/PostRoute events.
///
/// Mirrors `RoutingDomainKind` from `weft_router` but lives in `weft_core::config`
/// to avoid a dependency on `weft_router` from config types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookRoutingDomain {
    /// Which model handles this request.
    Model,
    /// Which commands are relevant for this turn.
    Commands,
    /// Whether tools are needed at all.
    ToolNecessity,
    /// Which memory stores to query or write to.
    Memory,
}

impl HookRoutingDomain {
    /// String representation for matcher evaluation.
    pub fn as_matcher_target(&self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Commands => "commands",
            Self::ToolNecessity => "tool_necessity",
            Self::Memory => "memory",
        }
    }
}

/// What triggered this routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingTrigger {
    /// Initial routing at request start (model, commands, tool_necessity domains).
    RequestStart,
    /// Per-invocation routing for a `/recall` command.
    RecallCommand,
    /// Per-invocation routing for a `/remember` command.
    RememberCommand,
}

impl RoutingTrigger {
    /// Whether a block on this trigger should be a hard block (403) or a feedback block.
    ///
    /// Hard block: request terminated immediately.
    /// Feedback block: block reason fed back to LLM as failed CommandResult.
    pub fn is_hard_block(&self) -> bool {
        matches!(self, Self::RequestStart)
    }
}

/// Hook type: which execution engine handles this hook.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HookType {
    /// Rhai embedded scripting language hook.
    Rhai,
    /// HTTP webhook hook.
    Http,
    /// Weft-type hook (reserved for future implementation).
    Weft,
}

/// A single hook configuration entry.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HookConfig {
    /// Which lifecycle event this hook fires on.
    pub event: HookEvent,
    /// Optional regex matcher. Absence means "fire on all events of this type."
    pub matcher: Option<String>,
    /// Hook type: "rhai", "http", or "weft".
    #[serde(rename = "type")]
    pub hook_type: HookType,
    /// Path to Rhai script file. Required when `hook_type` is `rhai`.
    pub script: Option<String>,
    /// URL for HTTP webhook. Required when `hook_type` is `http`.
    pub url: Option<String>,
    /// Agent name for Weft hook. Required when `hook_type` is `weft`.
    pub agent: Option<String>,
    /// Execution timeout in milliseconds.
    /// Default: 100 for Rhai, 5000 for HTTP. Max: 5000 for Rhai, 30000 for HTTP.
    pub timeout_ms: Option<u64>,
    /// Optional shared secret for HTTP hook authentication.
    /// Sent as `Authorization: Bearer <secret>` header.
    /// Supports `env:VAR_NAME` syntax for environment variable resolution.
    /// Only applicable to HTTP hooks; ignored for other types.
    pub secret: Option<String>,
    /// Execution priority. Lower values run first. Default: 100.
    #[serde(default = "default_hook_priority")]
    pub priority: u32,
}

fn default_hook_priority() -> u32 {
    100
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid config TOML with router section.
    fn minimal_router_toml() -> &'static str {
        r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are a helpful assistant."
max_command_iterations = 10
request_timeout_secs = 300

[router]
default_model = "main"

[router.classifier]
model_path = "models/modernbert-classifier.onnx"
tokenizer_path = "models/tokenizer.json"
threshold = 0.3
max_commands = 20

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-ant-test-key"

  [[router.providers.models]]
  name = "main"
  model = "claude-sonnet-4-20250514"
  max_tokens = 4096
  examples = ["Explain how async/await works", "What is the capital of France?"]
"#
    }

    #[test]
    fn test_valid_toml_parses() {
        let config: WeftConfig = toml::from_str(minimal_router_toml()).unwrap();
        assert_eq!(config.server.bind_address, "0.0.0.0:8080");
        assert_eq!(config.router.providers.len(), 1);
        assert_eq!(config.router.providers[0].name, "anthropic");
        assert_eq!(config.router.providers[0].models.len(), 1);
        assert_eq!(config.router.providers[0].models[0].name, "main");
        assert_eq!(config.router.providers[0].models[0].max_tokens, 4096);
        assert_eq!(config.router.providers[0].models[0].examples.len(), 2);
        assert_eq!(config.router.classifier.threshold, 0.3);
        assert_eq!(config.router.classifier.max_commands, 20);
        assert_eq!(config.router.default_model.as_deref(), Some("main"));
    }

    #[test]
    fn test_multi_provider_multi_model_parses() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are helpful."

[router]
default_model = "complex"
skip_tools_when_unnecessary = true

[router.classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"
threshold = 0.3
max_commands = 20

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-ant-test"

  [[router.providers.models]]
  name = "complex"
  model = "claude-sonnet-4-20250514"
  max_tokens = 8192
  examples = ["Design a distributed system"]

  [[router.providers.models]]
  name = "fast"
  model = "claude-haiku-4-5-20251001"
  max_tokens = 2048
  examples = ["Hi, how are you?"]

[[router.providers]]
name = "local"
wire_format = "openai"
api_key = "unused"
base_url = "http://localhost:11434/v1/chat/completions"

  [[router.providers.models]]
  name = "general"
  model = "llama3.2"
  max_tokens = 4096
  examples = ["Draft an internal memo"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.router.providers.len(), 2);
        assert_eq!(config.router.providers[0].models.len(), 2);
        assert_eq!(config.router.providers[1].models.len(), 1);
        assert_eq!(
            config.router.providers[1].base_url.as_deref(),
            Some("http://localhost:11434/v1/chat/completions")
        );
        assert!(config.router.skip_tools_when_unnecessary);
    }

    #[test]
    fn test_classifier_config_parses() {
        let config: WeftConfig = toml::from_str(minimal_router_toml()).unwrap();
        assert_eq!(
            config.router.classifier.model_path,
            "models/modernbert-classifier.onnx"
        );
        assert_eq!(
            config.router.classifier.tokenizer_path,
            "models/tokenizer.json"
        );
        assert_eq!(config.router.classifier.threshold, 0.3);
        assert_eq!(config.router.classifier.max_commands, 20);
    }

    #[test]
    fn test_optional_tool_registry() {
        let config: WeftConfig = toml::from_str(minimal_router_toml()).unwrap();
        assert!(config.tool_registry.is_none());
    }

    #[test]
    fn test_defaults_applied() {
        let toml = r#"
[server]
bind_address = "127.0.0.1:8080"

[gateway]
system_prompt = "You are helpful."

[router]

[router.classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "main"
  model = "claude-sonnet-4-20250514"
  examples = ["example query"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.router.providers[0].models[0].max_tokens, 4096);
        assert_eq!(config.router.classifier.threshold, 0.3);
        assert_eq!(config.router.classifier.max_commands, 20);
        assert_eq!(config.gateway.max_command_iterations, 10);
        assert_eq!(config.gateway.request_timeout_secs, 300);
        // skip_tools_when_unnecessary defaults to true
        assert!(config.router.skip_tools_when_unnecessary);
    }

    // ---- Validation tests ----

    #[test]
    fn test_validate_zero_providers() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one entry"));
    }

    #[test]
    fn test_validate_zero_models_on_provider() {
        // toml crate will reject [[router.providers]] with no models array as valid
        // since models is a Vec that defaults to empty, but our validation catches it.
        let toml = r#"
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
models = []
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one model"));
    }

    #[test]
    fn test_validate_duplicate_model_names() {
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  examples = ["example"]

[[router.providers]]
name = "openai"
wire_format = "openai"
api_key = "sk-test2"

  [[router.providers.models]]
  name = "main"
  model = "gpt-4"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate model name"));
    }

    #[test]
    fn test_validate_duplicate_provider_names() {
        let toml = r#"
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
  name = "model1"
  model = "claude-1"
  examples = ["example"]

[[router.providers]]
name = "anthropic"
wire_format = "openai"
api_key = "sk-test2"

  [[router.providers.models]]
  name = "model2"
  model = "gpt-4"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate provider name"));
    }

    #[test]
    fn test_validate_missing_default_model() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"

[router]
default_model = "nonexistent"

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("default_model"));
    }

    #[test]
    fn test_validate_empty_examples_array() {
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  examples = []
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one example"));
    }

    #[test]
    fn test_validate_passes_for_valid_config() {
        let config: WeftConfig = toml::from_str(minimal_router_toml()).unwrap();
        assert!(config.validate().is_ok());
    }

    // ---- Env var resolution tests ----

    #[test]
    fn test_env_var_resolution_literal() {
        let toml = r#"
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
api_key = "sk-ant-literal-key"

  [[router.providers.models]]
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let mut config: WeftConfig = toml::from_str(toml).unwrap();
        config.resolve().unwrap();
        assert_eq!(config.router.providers[0].api_key, "sk-ant-literal-key");
    }

    #[test]
    fn test_env_var_resolution_from_env() {
        // SAFETY: test-only, single-threaded context. No other thread reads this var.
        unsafe { std::env::set_var("WEFT_TEST_ROUTER_API_KEY_456", "sk-ant-from-env") };
        let toml = r#"
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
api_key = "env:WEFT_TEST_ROUTER_API_KEY_456"

  [[router.providers.models]]
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let mut config: WeftConfig = toml::from_str(toml).unwrap();
        config.resolve().unwrap();
        assert_eq!(config.router.providers[0].api_key, "sk-ant-from-env");
        // SAFETY: test-only cleanup.
        unsafe { std::env::remove_var("WEFT_TEST_ROUTER_API_KEY_456") };
    }

    #[test]
    fn test_env_var_resolution_missing_env_var() {
        // Ensure var is not set.
        // SAFETY: test-only, single-threaded context.
        unsafe { std::env::remove_var("WEFT_TEST_MISSING_ROUTER_VAR_XYZ") };
        let toml = r#"
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
api_key = "env:WEFT_TEST_MISSING_ROUTER_VAR_XYZ"

  [[router.providers.models]]
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let mut config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.resolve();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("WEFT_TEST_MISSING_ROUTER_VAR_XYZ")
        );
    }

    #[test]
    fn test_wire_format_openai_parses() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "openai"
wire_format = "openai"
api_key = "sk-test"

  [[router.providers.models]]
  name = "main"
  model = "gpt-4o"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert!(matches!(
            config.router.providers[0].wire_format,
            WireFormat::OpenAI
        ));
    }

    // ---- resolve_models tests ----

    #[test]
    fn test_resolve_models_flattens_correctly() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"

[router]
default_model = "complex"

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-ant-key"

  [[router.providers.models]]
  name = "complex"
  model = "claude-sonnet-4-20250514"
  max_tokens = 8192
  examples = ["Design a distributed system"]

  [[router.providers.models]]
  name = "fast"
  model = "claude-haiku-4-5-20251001"
  max_tokens = 2048
  examples = ["Quick question"]

[[router.providers]]
name = "local"
wire_format = "openai"
api_key = "unused"
base_url = "http://localhost:11434/v1"

  [[router.providers.models]]
  name = "general"
  model = "llama3.2"
  max_tokens = 4096
  examples = ["Internal task"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let resolved = config.router.resolve_models();

        assert_eq!(resolved.len(), 3);

        let complex = resolved.iter().find(|m| m.name == "complex").unwrap();
        assert_eq!(complex.model, "claude-sonnet-4-20250514");
        assert_eq!(complex.max_tokens, 8192);
        assert_eq!(complex.provider_name, "anthropic");
        assert_eq!(complex.api_key, "sk-ant-key");
        assert!(complex.base_url.is_none());
        assert_eq!(complex.examples.len(), 1);
        assert!(matches!(complex.wire_format, WireFormat::Anthropic));

        let fast = resolved.iter().find(|m| m.name == "fast").unwrap();
        assert_eq!(fast.model, "claude-haiku-4-5-20251001");
        assert_eq!(fast.provider_name, "anthropic");
        assert_eq!(fast.api_key, "sk-ant-key");

        let general = resolved.iter().find(|m| m.name == "general").unwrap();
        assert_eq!(general.model, "llama3.2");
        assert_eq!(general.provider_name, "local");
        assert_eq!(general.api_key, "unused");
        assert_eq!(
            general.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn test_effective_default_model_explicit() {
        let config: WeftConfig = toml::from_str(minimal_router_toml()).unwrap();
        assert_eq!(config.router.effective_default_model(), "main");
    }

    #[test]
    fn test_effective_default_model_implicit() {
        let toml = r#"
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
  name = "first-model"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        // No default_model set: should use first model of first provider.
        assert_eq!(config.router.effective_default_model(), "first-model");
    }

    #[test]
    fn test_tool_registry_defaults() {
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  examples = ["example"]

[tool_registry]
endpoint = "http://localhost:50051"
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let registry = config.tool_registry.unwrap();
        assert_eq!(registry.connect_timeout_ms, 5000);
        assert_eq!(registry.request_timeout_ms, 30000);
    }

    #[test]
    fn test_missing_server_section_errors() {
        let bad_toml = r#"
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
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let result: Result<WeftConfig, _> = toml::from_str(bad_toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_domains_config_parses() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"

[router]

[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"

[router.domains.model]
threshold = 0.5

[router.domains.tool_necessity]
enabled = true

[router.domains.memory]
enabled = false

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let model_domain = config.router.domains.model.as_ref().unwrap();
        assert_eq!(model_domain.threshold, Some(0.5));
        assert!(model_domain.enabled);

        let tool_domain = config.router.domains.tool_necessity.as_ref().unwrap();
        assert!(tool_domain.enabled);
        assert!(tool_domain.threshold.is_none());

        let mem_domain = config.router.domains.memory.as_ref().unwrap();
        assert!(!mem_domain.enabled);
    }

    // ── Memory config tests ───────────────────────────────────────────────────

    fn minimal_router_toml_str() -> &'static str {
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
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#
    }

    #[test]
    fn test_weft_config_without_memory_section_parses() {
        let config: WeftConfig = toml::from_str(minimal_router_toml_str()).unwrap();
        assert!(
            config.memory.is_none(),
            "absent [memory] section should be None"
        );
    }

    #[test]
    fn test_weft_config_with_memory_section_parses() {
        let toml = format!(
            r#"{}
[[memory.stores]]
name = "conversations"
endpoint = "http://localhost:50052"
examples = ["What did we discuss yesterday?"]
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        let memory = config.memory.as_ref().unwrap();
        assert_eq!(memory.stores.len(), 1);
        assert_eq!(memory.stores[0].name, "conversations");
        assert_eq!(memory.stores[0].endpoint, "http://localhost:50052");
    }

    #[test]
    fn test_memory_store_config_defaults() {
        let toml = format!(
            r#"{}
[[memory.stores]]
name = "conversations"
endpoint = "http://localhost:50052"
examples = ["example"]
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        let store = &config.memory.as_ref().unwrap().stores[0];
        assert_eq!(store.connect_timeout_ms, 5000);
        assert_eq!(store.request_timeout_ms, 10000);
        assert_eq!(store.max_results, 5);
        // capabilities defaults to [read, write]
        assert!(store.can_read());
        assert!(store.can_write());
    }

    #[test]
    fn test_memory_store_read_only_capability() {
        let toml = format!(
            r#"{}
[[memory.stores]]
name = "knowledge"
endpoint = "http://localhost:50052"
capabilities = ["read"]
examples = ["How does X work?"]
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        let store = &config.memory.as_ref().unwrap().stores[0];
        assert!(store.can_read());
        assert!(!store.can_write());
    }

    #[test]
    fn test_memory_store_write_only_capability() {
        let toml = format!(
            r#"{}
[[memory.stores]]
name = "audit"
endpoint = "http://localhost:50052"
capabilities = ["write"]
examples = ["store user action"]
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        let store = &config.memory.as_ref().unwrap().stores[0];
        assert!(!store.can_read());
        assert!(store.can_write());
    }

    #[test]
    fn test_memory_store_invalid_capability_rejected_by_serde() {
        let toml = format!(
            r#"{}
[[memory.stores]]
name = "bad"
endpoint = "http://localhost:50052"
capabilities = ["execute"]
examples = ["example"]
"#,
            minimal_router_toml_str()
        );
        let result: Result<WeftConfig, _> = toml::from_str(&toml);
        assert!(
            result.is_err(),
            "unknown capability 'execute' should be rejected"
        );
    }

    #[test]
    fn test_memory_validate_duplicate_store_names() {
        let toml = format!(
            r#"{}
[[memory.stores]]
name = "conv"
endpoint = "http://localhost:50052"
examples = ["example 1"]

[[memory.stores]]
name = "conv"
endpoint = "http://localhost:50053"
examples = ["example 2"]
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate memory store name"));
    }

    #[test]
    fn test_memory_validate_empty_examples_rejected() {
        let memory_config = MemoryConfig {
            stores: vec![MemoryStoreConfig {
                name: "conv".to_string(),
                endpoint: "http://localhost:50052".to_string(),
                connect_timeout_ms: 5000,
                request_timeout_ms: 10000,
                max_results: 5,
                capabilities: vec![StoreCapability::Read, StoreCapability::Write],
                examples: vec![], // empty — should fail
            }],
        };
        let result = memory_config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one example"));
    }

    #[test]
    fn test_memory_validate_empty_capabilities_rejected() {
        let memory_config = MemoryConfig {
            stores: vec![MemoryStoreConfig {
                name: "conv".to_string(),
                endpoint: "http://localhost:50052".to_string(),
                connect_timeout_ms: 5000,
                request_timeout_ms: 10000,
                max_results: 5,
                capabilities: vec![], // empty — should fail
                examples: vec!["example".to_string()],
            }],
        };
        let result = memory_config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one capability"));
    }

    #[test]
    fn test_memory_validate_invalid_endpoint_rejected() {
        // An empty endpoint string is rejected by tonic::transport::Endpoint::new().
        // Tonic accepts relative-looking strings, but empty string is always invalid.
        let memory_config = MemoryConfig {
            stores: vec![MemoryStoreConfig {
                name: "conv".to_string(),
                endpoint: String::new(), // empty string — always invalid
                connect_timeout_ms: 5000,
                request_timeout_ms: 10000,
                max_results: 5,
                capabilities: vec![StoreCapability::Read],
                examples: vec!["example".to_string()],
            }],
        };
        let result = memory_config.validate();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("invalid endpoint URI"),
            "should report invalid endpoint URI"
        );
    }

    #[test]
    fn test_memory_validate_valid_config_passes() {
        let memory_config = MemoryConfig {
            stores: vec![
                MemoryStoreConfig {
                    name: "conv".to_string(),
                    endpoint: "http://localhost:50052".to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities: vec![StoreCapability::Read, StoreCapability::Write],
                    examples: vec!["What did we discuss?".to_string()],
                },
                MemoryStoreConfig {
                    name: "code".to_string(),
                    endpoint: "http://localhost:50053".to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 3,
                    capabilities: vec![StoreCapability::Read],
                    examples: vec!["How does the parser work?".to_string()],
                },
            ],
        };
        assert!(memory_config.validate().is_ok());
    }

    #[test]
    fn test_memory_store_config_can_read_can_write() {
        let rw_store = MemoryStoreConfig {
            name: "rw".to_string(),
            endpoint: "http://localhost:50052".to_string(),
            connect_timeout_ms: 5000,
            request_timeout_ms: 10000,
            max_results: 5,
            capabilities: vec![StoreCapability::Read, StoreCapability::Write],
            examples: vec!["example".to_string()],
        };
        assert!(rw_store.can_read());
        assert!(rw_store.can_write());

        let ro_store = MemoryStoreConfig {
            name: "ro".to_string(),
            endpoint: "http://localhost:50052".to_string(),
            connect_timeout_ms: 5000,
            request_timeout_ms: 10000,
            max_results: 5,
            capabilities: vec![StoreCapability::Read],
            examples: vec!["example".to_string()],
        };
        assert!(ro_store.can_read());
        assert!(!ro_store.can_write());

        let wo_store = MemoryStoreConfig {
            name: "wo".to_string(),
            endpoint: "http://localhost:50052".to_string(),
            connect_timeout_ms: 5000,
            request_timeout_ms: 10000,
            max_results: 5,
            capabilities: vec![StoreCapability::Write],
            examples: vec!["example".to_string()],
        };
        assert!(!wo_store.can_read());
        assert!(wo_store.can_write());
    }

    // ── HookEvent tests ───────────────────────────────────────────────────────

    #[test]
    fn test_hook_event_can_block_all_variants() {
        // Blocking events: RequestStart, PreRoute, PostRoute, PreToolUse, PreResponse.
        assert!(HookEvent::RequestStart.can_block());
        assert!(HookEvent::PreRoute.can_block());
        assert!(HookEvent::PostRoute.can_block());
        assert!(HookEvent::PreToolUse.can_block());
        assert!(HookEvent::PreResponse.can_block());
        // Non-blocking events: PostToolUse, RequestEnd.
        assert!(!HookEvent::PostToolUse.can_block());
        assert!(!HookEvent::RequestEnd.can_block());
    }

    #[test]
    fn test_hook_event_is_feedback_block_all_variants() {
        // Feedback-block events (reason fed back to LLM): PreToolUse, PreResponse.
        assert!(HookEvent::PreToolUse.is_feedback_block());
        assert!(HookEvent::PreResponse.is_feedback_block());
        // Hard-block events (immediate HTTP error): RequestStart, PreRoute, PostRoute.
        assert!(!HookEvent::RequestStart.is_feedback_block());
        assert!(!HookEvent::PreRoute.is_feedback_block());
        assert!(!HookEvent::PostRoute.is_feedback_block());
        // Non-blocking events do not feedback-block either.
        assert!(!HookEvent::PostToolUse.is_feedback_block());
        assert!(!HookEvent::RequestEnd.is_feedback_block());
    }

    // ── HookRoutingDomain tests ───────────────────────────────────────────────

    #[test]
    fn test_hook_routing_domain_serde_round_trip() {
        for (domain, expected_json) in [
            (HookRoutingDomain::Model, r#""model""#),
            (HookRoutingDomain::Commands, r#""commands""#),
            (HookRoutingDomain::ToolNecessity, r#""tool_necessity""#),
            (HookRoutingDomain::Memory, r#""memory""#),
        ] {
            let json = serde_json::to_string(&domain).unwrap();
            assert_eq!(json, expected_json, "serialize {domain:?}");
            let back: HookRoutingDomain = serde_json::from_str(&json).unwrap();
            assert_eq!(back, domain, "deserialize {domain:?}");
        }
    }

    #[test]
    fn test_hook_routing_domain_as_matcher_target() {
        assert_eq!(HookRoutingDomain::Model.as_matcher_target(), "model");
        assert_eq!(HookRoutingDomain::Commands.as_matcher_target(), "commands");
        assert_eq!(
            HookRoutingDomain::ToolNecessity.as_matcher_target(),
            "tool_necessity"
        );
        assert_eq!(HookRoutingDomain::Memory.as_matcher_target(), "memory");
    }

    // ── RoutingTrigger tests ──────────────────────────────────────────────────

    #[test]
    fn test_routing_trigger_serde_round_trip() {
        for (trigger, expected_json) in [
            (RoutingTrigger::RequestStart, r#""request_start""#),
            (RoutingTrigger::RecallCommand, r#""recall_command""#),
            (RoutingTrigger::RememberCommand, r#""remember_command""#),
        ] {
            let json = serde_json::to_string(&trigger).unwrap();
            assert_eq!(json, expected_json, "serialize {trigger:?}");
            let back: RoutingTrigger = serde_json::from_str(&json).unwrap();
            assert_eq!(back, trigger, "deserialize {trigger:?}");
        }
    }

    #[test]
    fn test_routing_trigger_is_hard_block() {
        // RequestStart: hard block (403).
        assert!(RoutingTrigger::RequestStart.is_hard_block());
        // RecallCommand and RememberCommand: feedback block (failed CommandResult).
        assert!(!RoutingTrigger::RecallCommand.is_hard_block());
        assert!(!RoutingTrigger::RememberCommand.is_hard_block());
    }

    // ── TOML parsing with [[hooks]] ───────────────────────────────────────────

    #[test]
    fn test_hooks_toml_parses_rhai_and_http() {
        let toml = format!(
            r#"{}
[[hooks]]
event = "request_start"
type = "rhai"
script = "/etc/hooks/auth.rhai"

[[hooks]]
event = "pre_tool_use"
type = "http"
url = "https://example.com/hook"
matcher = "web_search"
priority = 50
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        assert_eq!(config.hooks.len(), 2);

        let rhai_hook = &config.hooks[0];
        assert_eq!(rhai_hook.event, HookEvent::RequestStart);
        assert_eq!(rhai_hook.hook_type, HookType::Rhai);
        assert_eq!(rhai_hook.script.as_deref(), Some("/etc/hooks/auth.rhai"));
        // priority defaults to 100 when absent.
        assert_eq!(rhai_hook.priority, 100);

        let http_hook = &config.hooks[1];
        assert_eq!(http_hook.event, HookEvent::PreToolUse);
        assert_eq!(http_hook.hook_type, HookType::Http);
        assert_eq!(http_hook.url.as_deref(), Some("https://example.com/hook"));
        assert_eq!(http_hook.matcher.as_deref(), Some("web_search"));
        assert_eq!(http_hook.priority, 50);
    }

    #[test]
    fn test_hooks_priority_defaults_to_100() {
        let toml = format!(
            r#"{}
[[hooks]]
event = "request_start"
type = "http"
url = "https://example.com/hook"
"#,
            minimal_router_toml_str()
        );
        let config: WeftConfig = toml::from_str(&toml).unwrap();
        assert_eq!(config.hooks[0].priority, 100);
    }

    #[test]
    fn test_hooks_secret_with_env_prefix_parsed() {
        // SAFETY: test-only, single-threaded context.
        unsafe { std::env::set_var("WEFT_TEST_HOOK_SECRET_789", "my-shared-secret") };

        let toml = format!(
            r#"{}
[[hooks]]
event = "request_start"
type = "http"
url = "https://example.com/hook"
secret = "env:WEFT_TEST_HOOK_SECRET_789"
"#,
            minimal_router_toml_str()
        );
        let mut config: WeftConfig = toml::from_str(&toml).unwrap();
        // Before resolve: raw env: reference is stored.
        assert_eq!(
            config.hooks[0].secret.as_deref(),
            Some("env:WEFT_TEST_HOOK_SECRET_789")
        );
        // After resolve: the env var value is substituted.
        config.resolve().unwrap();
        assert_eq!(config.hooks[0].secret.as_deref(), Some("my-shared-secret"));

        // SAFETY: test-only cleanup.
        unsafe { std::env::remove_var("WEFT_TEST_HOOK_SECRET_789") };
    }

    #[test]
    fn test_max_pre_response_retries_defaults_to_2() {
        let config: WeftConfig = toml::from_str(minimal_router_toml_str()).unwrap();
        assert_eq!(config.max_pre_response_retries, 2);
    }

    #[test]
    fn test_request_end_concurrency_defaults_to_64() {
        let config: WeftConfig = toml::from_str(minimal_router_toml_str()).unwrap();
        assert_eq!(config.request_end_concurrency, 64);
    }

    #[test]
    fn test_max_pre_response_retries_clamped_at_5() {
        // Top-level fields must appear before section headers in TOML.
        let toml = r#"
max_pre_response_retries = 100

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
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        // Raw field stores the configured value.
        assert_eq!(config.max_pre_response_retries, 100);
        // Effective accessor returns clamped value.
        assert_eq!(config.effective_max_pre_response_retries(), 5);
    }

    #[test]
    fn test_max_pre_response_retries_within_limit_unchanged() {
        let toml = r#"
max_pre_response_retries = 3

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
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.max_pre_response_retries, 3);
        assert_eq!(config.effective_max_pre_response_retries(), 3);
    }

    // ── WireFormat, capabilities, wire_script tests ───────────────────────────

    #[test]
    fn test_wire_format_anthropic_parses() {
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert!(matches!(
            config.router.providers[0].wire_format,
            WireFormat::Anthropic
        ));
    }

    #[test]
    fn test_wire_format_custom_parses() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"
[gateway]
system_prompt = "test"
[router]
[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"
[[router.providers]]
name = "custom-provider"
wire_format = "custom"
wire_script = "providers/custom.rhai"
api_key = "sk-test"
base_url = "https://api.custom.ai/v1"
  [[router.providers.models]]
  name = "main"
  model = "custom-model"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert!(matches!(
            config.router.providers[0].wire_format,
            WireFormat::Custom
        ));
        assert_eq!(
            config.router.providers[0].wire_script.as_deref(),
            Some("providers/custom.rhai")
        );
    }

    #[test]
    fn test_model_capabilities_default_to_chat_completions() {
        // A model without a 'capabilities' field should default to ["chat_completions"].
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let caps = &config.router.providers[0].models[0].capabilities;
        assert_eq!(caps, &vec!["chat_completions".to_string()]);
    }

    #[test]
    fn test_model_explicit_capabilities_parsed() {
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  capabilities = ["chat_completions", "vision", "tool_calling"]
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let caps = &config.router.providers[0].models[0].capabilities;
        assert_eq!(
            caps,
            &vec![
                "chat_completions".to_string(),
                "vision".to_string(),
                "tool_calling".to_string(),
            ]
        );
    }

    #[test]
    fn test_validation_custom_wire_format_without_wire_script_fails() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"
[gateway]
system_prompt = "test"
[router]
[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"
[[router.providers]]
name = "custom-provider"
wire_format = "custom"
api_key = "sk-test"
base_url = "https://api.custom.ai/v1"
  [[router.providers.models]]
  name = "main"
  model = "custom-model"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("custom wire_format requires 'wire_script'"),
            "expected wire_script required error, got: {err}"
        );
    }

    #[test]
    fn test_validation_wire_script_with_non_custom_wire_format_fails() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"
[gateway]
system_prompt = "test"
[router]
[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"
[[router.providers]]
name = "openai-provider"
wire_format = "openai"
wire_script = "providers/extra.rhai"
api_key = "sk-test"
  [[router.providers.models]]
  name = "main"
  model = "gpt-4"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("wire_script is only valid with wire_format = 'custom'"),
            "expected wire_script only for custom error, got: {err}"
        );
    }

    #[test]
    fn test_validation_empty_capabilities_fails() {
        let toml = r#"
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
  name = "main"
  model = "claude-1"
  capabilities = []
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("capabilities must have at least one entry"),
            "expected empty capabilities error, got: {err}"
        );
    }

    #[test]
    fn test_resolve_models_includes_wire_format_and_capabilities() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"
[gateway]
system_prompt = "test"
[router]
default_model = "chat-model"
[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"
[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-ant-key"
  [[router.providers.models]]
  name = "chat-model"
  model = "claude-sonnet"
  max_tokens = 8192
  capabilities = ["chat_completions", "vision"]
  examples = ["example query"]

[[router.providers]]
name = "custom-provider"
wire_format = "custom"
wire_script = "providers/custom.rhai"
api_key = "sk-custom"
base_url = "https://api.custom.ai/v1"
  [[router.providers.models]]
  name = "image-model"
  model = "custom-image-v1"
  max_tokens = 1024
  capabilities = ["image_generations"]
  examples = ["generate an image"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let resolved = config.router.resolve_models();
        assert_eq!(resolved.len(), 2);

        let chat = resolved.iter().find(|m| m.name == "chat-model").unwrap();
        assert!(matches!(chat.wire_format, WireFormat::Anthropic));
        assert!(chat.wire_script.is_none());
        assert_eq!(
            chat.capabilities,
            vec!["chat_completions".to_string(), "vision".to_string()]
        );

        let image = resolved.iter().find(|m| m.name == "image-model").unwrap();
        assert!(matches!(image.wire_format, WireFormat::Custom));
        assert_eq!(image.wire_script.as_deref(), Some("providers/custom.rhai"));
        assert_eq!(image.capabilities, vec!["image_generations".to_string()]);
    }

    #[test]
    fn test_validation_custom_with_wire_script_passes() {
        // Custom wire_format with wire_script present should pass validation.
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"
[gateway]
system_prompt = "test"
[router]
[router.classifier]
model_path = "m.onnx"
tokenizer_path = "t.json"
[[router.providers]]
name = "custom-provider"
wire_format = "custom"
wire_script = "providers/custom.rhai"
api_key = "sk-test"
base_url = "https://api.custom.ai/v1"
  [[router.providers.models]]
  name = "main"
  model = "custom-model"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        // Wire script validation (file existence) is deferred to startup, not config parse time.
        // The config-level validation only checks that wire_script is present for custom.
        let result = config.validate();
        assert!(
            result.is_ok(),
            "custom with wire_script should pass config validation: {:?}",
            result
        );
    }
}
