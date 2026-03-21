//! Config loading using `config-rs` for layered source merging.
//!
//! Loads operator configuration from a TOML file with optional environment
//! variable overrides. Environment variables use the `WEFT_` prefix and `__`
//! (double underscore) as the nesting separator.
//!
//! # Environment variable override convention
//!
//! - `WEFT_GATEWAY__SYSTEM_PROMPT` overrides `gateway.system_prompt`
//! - `WEFT_ROUTER__CLASSIFIER__THRESHOLD` overrides `router.classifier.threshold`
//! - `WEFT_ROUTER__CLASSIFIER__MAX_COMMANDS` overrides `router.classifier.max_commands`
//!
//! Environment variables override TOML values. The loaded config still needs
//! `resolve()` (for `env:` prefixes in API keys) and `validate()` after this call.

use std::path::Path;

use config::{Config, Environment, File};
use weft_core::{ConfigStore, ResolvedConfig, ResolvedConfigError, WeftConfig};

/// Errors from config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigLoadError {
    #[error("failed to build config: {0}")]
    Build(config::ConfigError),
    #[error("failed to deserialize config: {0}")]
    Deserialize(config::ConfigError),
    #[error("config resolution failed: {0}")]
    Resolve(String),
    #[error("config validation failed: {0}")]
    Validate(String),
    #[error("resolved config validation failed: {0}")]
    ResolvedValidate(#[from] ResolvedConfigError),
}

/// Load `WeftConfig` from a TOML file + environment variables.
///
/// Environment variables override TOML values. The prefix is `WEFT_`
/// and nesting uses `__` (double underscore). For example:
/// - `WEFT_GATEWAY__SYSTEM_PROMPT` overrides `gateway.system_prompt`
/// - `WEFT_ROUTER__CLASSIFIER__THRESHOLD` overrides `router.classifier.threshold`
///
/// The loaded config still needs `resolve()` and `validate()` after this call.
pub fn load_config(path: &Path) -> Result<WeftConfig, ConfigLoadError> {
    let config = Config::builder()
        .add_source(File::from(path))
        .add_source(
            Environment::with_prefix("WEFT")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        )
        .build()
        .map_err(ConfigLoadError::Build)?;

    config
        .try_deserialize::<WeftConfig>()
        .map_err(ConfigLoadError::Deserialize)
}

/// Load, resolve, validate, and produce a `ConfigStore`.
///
/// Single entry point for the binary's startup path. Returns a fully
/// validated `ConfigStore` ready for sharing.
///
/// Steps:
/// 1. Load TOML + env var overrides via `load_config`
/// 2. Resolve `env:VAR_NAME` prefixes in API keys and secrets
/// 3. Validate `WeftConfig` constraints
/// 4. Project into `ResolvedConfig` and validate
/// 5. Construct `ConfigStore` with both configs
pub fn load_and_build_store(path: &Path) -> Result<ConfigStore, ConfigLoadError> {
    let mut config = load_config(path)?;
    config.resolve().map_err(ConfigLoadError::Resolve)?;
    config.validate().map_err(ConfigLoadError::Validate)?;

    let resolved = ResolvedConfig::from_operator(&config);
    resolved.validate()?;

    Ok(ConfigStore::new(config))
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::io::Write;
    use std::sync::Mutex;
    use tempfile::NamedTempFile;

    // Serialize all tests that mutate environment variables to prevent parallel
    // contamination. Rust tests run concurrently by default; env mutations are global.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    const MINIMAL_TOML: &str = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "You are a helpful assistant."
max_command_iterations = 5
request_timeout_secs = 30

[router]
[router.classifier]
model_path = "models/classifier"
tokenizer_path = "models/tokenizer"
threshold = 0.5
max_commands = 10

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

[[router.providers.models]]
name = "claude"
model = "claude-sonnet-4-20250514"
max_tokens = 4096
examples = ["Write a poem", "Explain something", "What is X?"]
capabilities = ["chat_completions"]
"#;

    fn write_toml(content: &str) -> NamedTempFile {
        // config-rs uses the file extension to detect format; must use .toml suffix.
        let mut file = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile must create");
        file.write_all(content.as_bytes())
            .expect("write must succeed");
        file
    }

    #[test]
    fn test_load_config_from_valid_toml() {
        // Acquire ENV_MUTEX to prevent contamination from env-var-setting tests
        // running concurrently in the same binary.
        let _guard = ENV_MUTEX.lock().expect("env mutex must not be poisoned");
        // Use unique field values not shared with any env var override test.
        let toml = r#"
[server]
bind_address = "127.0.0.1:9999"

[gateway]
system_prompt = "UNIQUE_PROMPT_FOR_LOAD_CONFIG_TEST"
max_command_iterations = 3
request_timeout_secs = 60

[router]
[router.classifier]
model_path = "models/classifier"
tokenizer_path = "models/tokenizer"
threshold = 0.4
max_commands = 5

[[router.providers]]
name = "test-provider"
wire_format = "openai"
api_key = "sk-unique"

[[router.providers.models]]
name = "test-model"
model = "gpt-4o"
max_tokens = 2048
examples = ["Test query", "Another test", "Third test"]
capabilities = ["chat_completions"]
"#;
        let file = write_toml(toml);
        let config = load_config(file.path()).expect("load must succeed");

        assert_eq!(
            config.gateway.system_prompt,
            "UNIQUE_PROMPT_FOR_LOAD_CONFIG_TEST"
        );
        assert_eq!(config.router.classifier.threshold, 0.4);
        assert_eq!(config.router.providers.len(), 1);
        assert_eq!(config.router.providers[0].name, "test-provider");
    }

    #[test]
    fn test_load_config_returns_error_for_missing_file() {
        let result = load_config(Path::new("/nonexistent/path/weft.toml"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigLoadError::Build(_)));
    }

    #[test]
    fn test_load_config_returns_error_for_malformed_toml() {
        let file = write_toml("this is not valid toml ][[[");
        let result = load_config(file.path());
        assert!(result.is_err());
        // config-rs may return Build or Deserialize depending on parsing stage.
        let err = result.unwrap_err();
        assert!(
            matches!(err, ConfigLoadError::Build(_))
                || matches!(err, ConfigLoadError::Deserialize(_)),
            "expected Build or Deserialize, got: {err}"
        );
    }

    #[test]
    fn test_load_and_build_store_produces_correct_snapshot() {
        // Acquire ENV_MUTEX to prevent contamination from env-var-setting tests
        // running concurrently in the same binary.
        let _guard = ENV_MUTEX.lock().expect("env mutex must not be poisoned");
        // Use unique field values not shared with any env var override test.
        let toml = r#"
[server]
bind_address = "127.0.0.1:8888"

[gateway]
system_prompt = "UNIQUE_SNAPSHOT_TEST_PROMPT"
max_command_iterations = 7
request_timeout_secs = 45

[router]
[router.classifier]
model_path = "models/classifier"
tokenizer_path = "models/tokenizer"
threshold = 0.35
max_commands = 8

[[router.providers]]
name = "snapshot-provider"
wire_format = "anthropic"
api_key = "sk-snapshot"

[[router.providers.models]]
name = "snapshot-model"
model = "claude-sonnet-4-20250514"
max_tokens = 4096
examples = ["Snapshot test", "Second example", "Third example"]
capabilities = ["chat_completions"]
"#;
        let file = write_toml(toml);
        let store = load_and_build_store(file.path()).expect("load_and_build_store must succeed");

        let snapshot = store.snapshot();
        assert_eq!(snapshot.system_prompt, "UNIQUE_SNAPSHOT_TEST_PROMPT");
        assert_eq!(snapshot.default_model, "snapshot-model");
        assert_eq!(snapshot.classifier_threshold, 0.35_f32);
    }

    #[test]
    fn test_load_and_build_store_returns_error_for_invalid_toml() {
        let file = write_toml("bad toml [[[");
        let result = load_and_build_store(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_env_var_overrides_toml_threshold() {
        let _guard = ENV_MUTEX.lock().expect("env mutex must not be poisoned");
        let env_key = "WEFT_ROUTER__CLASSIFIER__THRESHOLD";
        let file = write_toml(MINIMAL_TOML);

        // SAFETY: serialized by ENV_MUTEX; no other test holds this env var concurrently.
        unsafe {
            std::env::set_var(env_key, "0.9");
        }

        let result = load_config(file.path());

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var(env_key);
        }

        let config = result.expect("load must succeed with env override");
        assert_eq!(config.router.classifier.threshold, 0.9_f32);
    }

    #[test]
    fn test_env_var_overrides_max_command_iterations() {
        let _guard = ENV_MUTEX.lock().expect("env mutex must not be poisoned");
        // Use a field that's clearly numeric to verify try_parsing(true) works.
        let env_key = "WEFT_GATEWAY__MAX_COMMAND_ITERATIONS";
        let file = write_toml(MINIMAL_TOML);

        // SAFETY: serialized by ENV_MUTEX; no other test holds this env var concurrently.
        unsafe {
            std::env::set_var(env_key, "42");
        }

        let result = load_config(file.path());

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var(env_key);
        }

        let config = result.expect("load must succeed");
        assert_eq!(config.gateway.max_command_iterations, 42_u32);
    }
}
