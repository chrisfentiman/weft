//! Configuration types for the Weft gateway.
//!
//! Configuration is loaded from a TOML file. The `api_key` field supports
//! an `env:VAR_NAME` prefix for loading secrets from environment variables.

/// Top-level configuration, deserialized from TOML.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeftConfig {
    pub server: ServerConfig,
    pub llm: LlmConfig,
    pub classifier: ClassifierConfig,
    pub tool_registry: Option<ToolRegistryConfig>,
    pub gateway: GatewayConfig,
}

impl WeftConfig {
    /// Resolve environment variable references in sensitive fields.
    ///
    /// Call this once after deserializing the config. Fields with the prefix
    /// `env:` are replaced with the value of the named environment variable.
    ///
    /// Returns an error if any referenced environment variable is not set.
    pub fn resolve(&mut self) -> Result<(), String> {
        self.llm.api_key =
            resolve_env_var(&self.llm.api_key).map_err(|e| format!("llm.api_key: {e}"))?;
        Ok(())
    }
}

/// Resolve a single value that may contain an `env:VAR_NAME` prefix.
fn resolve_env_var(value: &str) -> Result<String, String> {
    if let Some(var_name) = value.strip_prefix("env:") {
        std::env::var(var_name).map_err(|_| format!("environment variable '{var_name}' is not set"))
    } else {
        Ok(value.to_string())
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ServerConfig {
    /// Bind address, e.g. "0.0.0.0:8080"
    pub bind_address: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmConfig {
    /// Which provider to use: "anthropic" or "openai"
    pub provider: LlmProviderKind,
    /// API key for the provider. Loaded from env var if prefixed with "env:".
    pub api_key: String,
    /// Model identifier to use, e.g. "claude-sonnet-4-20250514" or "gpt-4o"
    pub model: String,
    /// Maximum tokens in response (used as default when request doesn't specify).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Optional base URL override (for proxies or compatible APIs).
    pub base_url: Option<String>,
}

fn default_max_tokens() -> u32 {
    4096
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

    const VALID_TOML: &str = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "sk-ant-test-key"
model = "claude-sonnet-4-20250514"
max_tokens = 4096

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"
threshold = 0.3
max_commands = 20

[tool_registry]
endpoint = "http://localhost:50051"
connect_timeout_ms = 5000
request_timeout_ms = 30000

[gateway]
system_prompt = "You are a helpful assistant."
max_command_iterations = 10
request_timeout_secs = 300
"#;

    #[test]
    fn test_valid_toml_parses() {
        let config: WeftConfig = toml::from_str(VALID_TOML).unwrap();
        assert_eq!(config.server.bind_address, "0.0.0.0:8080");
        assert_eq!(config.llm.model, "claude-sonnet-4-20250514");
        assert_eq!(config.llm.max_tokens, 4096);
        assert!(matches!(config.llm.provider, LlmProviderKind::Anthropic));
        assert_eq!(config.classifier.threshold, 0.3);
        assert_eq!(config.classifier.max_commands, 20);
        assert!(config.tool_registry.is_some());
        assert_eq!(config.gateway.max_command_iterations, 10);
        assert_eq!(config.gateway.request_timeout_secs, 300);
    }

    #[test]
    fn test_optional_tool_registry() {
        let toml_no_registry = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "sk-ant-test-key"
model = "claude-sonnet-4-20250514"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let config: WeftConfig = toml::from_str(toml_no_registry).unwrap();
        assert!(config.tool_registry.is_none());
    }

    #[test]
    fn test_defaults_applied() {
        let minimal_toml = r#"
[server]
bind_address = "127.0.0.1:8080"

[llm]
provider = "openai"
api_key = "sk-test"
model = "gpt-4o"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let config: WeftConfig = toml::from_str(minimal_toml).unwrap();
        assert_eq!(config.llm.max_tokens, 4096);
        assert_eq!(config.classifier.threshold, 0.3);
        assert_eq!(config.classifier.max_commands, 20);
        assert_eq!(config.gateway.max_command_iterations, 10);
        assert_eq!(config.gateway.request_timeout_secs, 300);
    }

    #[test]
    fn test_missing_required_field_errors() {
        // Missing [server] section
        let bad_toml = r#"
[llm]
provider = "anthropic"
api_key = "sk-ant-test"
model = "claude-sonnet-4-20250514"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let result: Result<WeftConfig, _> = toml::from_str(bad_toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_env_var_resolution_literal() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "sk-ant-literal-key"
model = "claude-sonnet-4-20250514"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let mut config: WeftConfig = toml::from_str(toml).unwrap();
        config.resolve().unwrap();
        assert_eq!(config.llm.api_key, "sk-ant-literal-key");
    }

    #[test]
    fn test_env_var_resolution_from_env() {
        // SAFETY: test-only, single-threaded context. No other thread reads this var.
        unsafe { std::env::set_var("WEFT_TEST_API_KEY_123", "sk-ant-from-env") };
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "env:WEFT_TEST_API_KEY_123"
model = "claude-sonnet-4-20250514"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let mut config: WeftConfig = toml::from_str(toml).unwrap();
        config.resolve().unwrap();
        assert_eq!(config.llm.api_key, "sk-ant-from-env");
        // SAFETY: test-only cleanup.
        unsafe { std::env::remove_var("WEFT_TEST_API_KEY_123") };
    }

    #[test]
    fn test_env_var_resolution_missing_env_var() {
        // Ensure var is not set
        // SAFETY: test-only, single-threaded context.
        unsafe { std::env::remove_var("WEFT_TEST_MISSING_VAR_XYZ") };
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "env:WEFT_TEST_MISSING_VAR_XYZ"
model = "claude-sonnet-4-20250514"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let mut config: WeftConfig = toml::from_str(toml).unwrap();
        let result = config.resolve();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("WEFT_TEST_MISSING_VAR_XYZ"));
    }

    #[test]
    fn test_llm_provider_kind_openai() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "openai"
api_key = "sk-test"
model = "gpt-4o"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert!(matches!(config.llm.provider, LlmProviderKind::OpenAI));
    }

    #[test]
    fn test_tool_registry_defaults() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "sk-test"
model = "claude-sonnet-4-20250514"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[tool_registry]
endpoint = "http://localhost:50051"

[gateway]
system_prompt = "You are helpful."
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        let registry = config.tool_registry.unwrap();
        assert_eq!(registry.connect_timeout_ms, 5000);
        assert_eq!(registry.request_timeout_ms, 30000);
    }

    #[test]
    fn test_base_url_optional() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[llm]
provider = "anthropic"
api_key = "sk-test"
model = "claude-sonnet-4-20250514"
base_url = "https://proxy.example.com"

[classifier]
model_path = "models/model.onnx"
tokenizer_path = "models/tokenizer.json"

[gateway]
system_prompt = "You are helpful."
"#;
        let config: WeftConfig = toml::from_str(toml).unwrap();
        assert_eq!(
            config.llm.base_url.as_deref(),
            Some("https://proxy.example.com")
        );
    }
}
