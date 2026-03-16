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
        Ok(())
    }

    /// Validate the configuration after resolving environment variables.
    ///
    /// Checks all consistency constraints that cannot be expressed in the type system.
    /// Must be called after `resolve()`.
    pub fn validate(&self) -> Result<(), String> {
        self.router.validate()
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
                    provider_kind: provider.kind.clone(),
                    api_key: provider.api_key.clone(),
                    base_url: provider.base_url.clone(),
                    provider_name: provider.name.clone(),
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
    /// Unique name for this provider (e.g., "anthropic", "local").
    pub name: String,
    /// Provider type -- determines the API protocol used.
    pub kind: LlmProviderKind,
    /// API key for this provider. Supports `env:VAR_NAME` prefix.
    /// For local providers (e.g., Ollama) that don't require authentication,
    /// use a literal string like `"unused"`. The value is passed through to the
    /// provider's HTTP client (e.g., `Authorization: Bearer unused`). Local
    /// providers typically ignore this header, so any non-empty string is acceptable.
    /// The field is required (not Optional) to keep the config schema simple.
    pub api_key: String,
    /// Optional base URL override. Required for local/custom endpoints (e.g., Ollama).
    pub base_url: Option<String>,
    /// Models served by this provider. Must have at least one.
    pub models: Vec<ModelEntry>,
}

/// A model available for routing, nested under a provider.
///
/// The model inherits the provider's API endpoint (kind, api_key, base_url).
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
}

fn default_max_tokens() -> u32 {
    4096
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
    /// Provider kind (from ProviderConfig.kind).
    pub provider_kind: LlmProviderKind,
    /// Resolved API key (from ProviderConfig.api_key, after env resolution).
    pub api_key: String,
    /// Base URL (from ProviderConfig.base_url).
    pub base_url: Option<String>,
    /// Provider name (from ProviderConfig.name) -- for logging/diagnostics.
    pub provider_name: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmProviderKind {
    Anthropic,
    #[serde(rename = "openai")]
    OpenAI,
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "openai"
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "main"
  model = "claude-1"
  examples = ["example"]

[[router.providers]]
name = "openai"
kind = "openai"
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
kind = "anthropic"
api_key = "sk-test"

  [[router.providers.models]]
  name = "model1"
  model = "claude-1"
  examples = ["example"]

[[router.providers]]
name = "anthropic"
kind = "openai"
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "anthropic"
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
    fn test_llm_provider_kind_openai() {
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
kind = "openai"
api_key = "sk-test"

  [[router.providers.models]]
  name = "main"
  model = "gpt-4o"
  examples = ["example"]
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert!(matches!(
            config.router.providers[0].kind,
            LlmProviderKind::OpenAI
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
kind = "anthropic"
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
kind = "openai"
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "anthropic"
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
kind = "anthropic"
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
}
