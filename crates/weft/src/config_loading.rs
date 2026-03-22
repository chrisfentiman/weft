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
    #[error("{0}")]
    TomlValidation(String),
}

/// Pre-validate raw TOML for unknown fields before config-rs processing.
///
/// Deserializes the raw TOML string directly into `WeftConfig`. Since all config
/// structs have `deny_unknown_fields`, this catches typos in field names and
/// produces errors referencing the TOML source specifically.
///
/// Returns `Ok(())` if validation passes. Returns `Err` with an enhanced error
/// message including field suggestions if an unknown field is found.
fn pre_validate_toml(raw_toml: &str) -> Result<(), ConfigLoadError> {
    match toml::from_str::<WeftConfig>(raw_toml) {
        Ok(_) => Ok(()),
        Err(e) => {
            let enhanced = enhance_toml_error(&e);
            Err(ConfigLoadError::TomlValidation(enhanced))
        }
    }
}

/// Enhance a TOML deserialization error with field suggestions.
///
/// Parses the error message for "unknown field" patterns and adds
/// Levenshtein-distance-based suggestions for the closest known field.
/// This is best-effort: if the error message doesn't match the expected pattern,
/// the raw error string is returned unchanged.
///
/// The toml crate (v0.8+) formats deny_unknown_fields errors as:
/// ```text
/// TOML parse error at line N, column M
///   |
/// N | field_name = value
///   | ^^^^^^^^^^
/// unknown field `field_name`, expected one of `field1`, `field2`, ...
/// ```
/// The "unknown field" clause is at the END of the message, after the snippet.
fn enhance_toml_error(err: &toml::de::Error) -> String {
    let msg = err.to_string();

    // Find the "unknown field `...`" clause anywhere in the message.
    // It may appear at the end after TOML parse context lines.
    let unknown_prefix = "unknown field `";
    let Some(unknown_start) = msg.find(unknown_prefix) else {
        // Not an unknown-field error — return raw.
        return msg;
    };

    let after_unknown = &msg[unknown_start + unknown_prefix.len()..];

    let Some(backtick_end) = after_unknown.find('`') else {
        return msg;
    };

    let unknown_field = &after_unknown[..backtick_end];
    let rest = &after_unknown[backtick_end + 1..];

    // Parse expected fields from ", expected one of `field1`, `field2`, ..."
    // or ", expected `field1`"
    let expected_fields: Vec<&str> = if let Some(after_expected) = rest
        .find(", expected one of ")
        .map(|i| &rest[i + ", expected one of ".len()..])
        .or_else(|| {
            rest.find(", expected `")
                .map(|i| &rest[i + ", expected ".len()..])
        }) {
        // Extract all backtick-delimited field names
        let mut fields = Vec::new();
        let mut remaining = after_expected;
        while let Some(start) = remaining.find('`') {
            remaining = &remaining[start + 1..];
            if let Some(end) = remaining.find('`') {
                fields.push(&remaining[..end]);
                remaining = &remaining[end + 1..];
            } else {
                break;
            }
        }
        fields
    } else {
        Vec::new()
    };

    // Find the closest match using Levenshtein distance, threshold <= 3.
    let best_match = expected_fields
        .iter()
        .min_by_key(|&&field| strsim::levenshtein(unknown_field, field));

    // Extract line/column info from the first line of the toml error.
    // Toml errors start with "TOML parse error at line N, column M"
    let location = msg
        .lines()
        .next()
        .and_then(|line| {
            // "TOML parse error at line N, column M" — extract " at line N, column M"
            line.find(" at line ").map(|i| &line[i..])
        })
        .unwrap_or("");

    // Build the enhanced error message.
    let mut output = format!("unknown field '{unknown_field}' in configuration");

    if let Some(&closest) = best_match
        && strsim::levenshtein(unknown_field, closest) <= 3
    {
        output.push_str(&format!("\n  -> did you mean '{closest}'?"));
    }

    if !expected_fields.is_empty() {
        let fields_list = expected_fields.join(", ");
        output.push_str(&format!("\n  -> expected fields: {fields_list}"));
    }

    if !location.is_empty() {
        output.push_str(&format!("\n  -> location{location}"));
    }

    output
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
/// 1. Read raw TOML and pre-validate for unknown fields (fast fail with suggestions)
/// 2. Load TOML + env var overrides via `load_config`
/// 3. Resolve `env:VAR_NAME` prefixes in API keys and secrets
/// 4. Validate `WeftConfig` constraints
/// 5. Project into `ResolvedConfig` and validate
/// 6. Construct `ConfigStore` with both configs
pub fn load_and_build_store(path: &Path) -> Result<ConfigStore, ConfigLoadError> {
    // Pre-validate the raw TOML before config-rs processing to catch unknown
    // fields with enhanced error messages and "did you mean?" suggestions.
    let raw_toml = std::fs::read_to_string(path).map_err(|e| {
        ConfigLoadError::Build(config::ConfigError::Message(format!(
            "failed to read config file: {e}"
        )))
    })?;
    pre_validate_toml(&raw_toml)?;

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

    // ── Phase 1: pre_validate_toml and enhance_toml_error tests ────────────────

    #[test]
    fn test_pre_validate_catches_unknown_field() {
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"
max_command_iterations = 5
request_timeout_secs = 30
unknown_field = "oops"

[router]
[router.classifier]
model_path = "models/classifier"
tokenizer_path = "models/tokenizer"
threshold = 0.5
max_commands = 10

[[router.providers]]
name = "p"
wire_format = "openai"
api_key = "sk-test"

[[router.providers.models]]
name = "m"
model = "gpt-4"
max_tokens = 2048
examples = ["test"]
capabilities = ["chat_completions"]
"#;
        let result = pre_validate_toml(toml);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ConfigLoadError::TomlValidation(_)),
            "expected TomlValidation error"
        );
    }

    #[test]
    fn test_pre_validate_passes_valid_toml() {
        let result = pre_validate_toml(MINIMAL_TOML);
        assert!(
            result.is_ok(),
            "valid TOML must pass pre-validation: {result:?}"
        );
    }

    #[test]
    fn test_error_enhancement_produces_suggestion() {
        // "systemm_prompt" is within edit distance 1 of "system_prompt"
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
systemm_prompt = "test"

[router]
[router.classifier]
model_path = "x"
tokenizer_path = "x"

[[router.providers]]
name = "p"
wire_format = "openai"
api_key = "k"

[[router.providers.models]]
name = "m"
model = "gpt-4"
examples = ["test"]
"#;
        let result = pre_validate_toml(toml);
        let err_msg = match result {
            Err(ConfigLoadError::TomlValidation(msg)) => msg,
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => panic!("unexpected error variant: {e}"),
        };
        assert!(
            err_msg.contains("did you mean"),
            "expected 'did you mean' suggestion in: {err_msg}"
        );
        assert!(
            err_msg.contains("system_prompt"),
            "expected 'system_prompt' as suggestion in: {err_msg}"
        );
    }

    #[test]
    fn test_error_enhancement_no_suggestion_for_distant_field() {
        // "zzzzzzz" has edit distance >> 3 from all known fields
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"
zzzzzzz = "far away"

[router]
[router.classifier]
model_path = "x"
tokenizer_path = "x"

[[router.providers]]
name = "p"
wire_format = "openai"
api_key = "k"

[[router.providers.models]]
name = "m"
model = "gpt-4"
examples = ["test"]
"#;
        let result = pre_validate_toml(toml);
        let err_msg = match result {
            Err(ConfigLoadError::TomlValidation(msg)) => msg,
            Ok(_) => panic!("expected error, got Ok"),
            Err(e) => panic!("unexpected error variant: {e}"),
        };
        assert!(
            !err_msg.contains("did you mean"),
            "should NOT contain 'did you mean' for distant field in: {err_msg}"
        );
    }

    #[test]
    fn test_load_and_build_store_rejects_unknown_field() {
        let _guard = ENV_MUTEX.lock().expect("env mutex must not be poisoned");
        let toml = r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "test"
max_command_iterations = 5
request_timeout_secs = 30
nonexistent_field = true

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
        let file = write_toml(toml);
        let result = load_and_build_store(file.path());
        assert!(result.is_err(), "must fail with unknown field");
        match result {
            Err(ConfigLoadError::TomlValidation(_)) => {}
            Err(other) => panic!("expected TomlValidation, got: {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn test_env_var_unknown_field_rejected() {
        let _guard = ENV_MUTEX.lock().expect("env mutex must not be poisoned");
        // WEFT_UNKNOWNFIELD maps to top-level field "unknownfield" via config-rs.
        // With deny_unknown_fields on WeftConfig, this is rejected at try_deserialize.
        let env_key = "WEFT_UNKNOWNFIELD";
        let file = write_toml(MINIMAL_TOML);

        // SAFETY: serialized by ENV_MUTEX; no other test holds this env var concurrently.
        unsafe {
            std::env::set_var(env_key, "x");
        }

        let result = load_config(file.path());

        // SAFETY: cleanup under the same lock.
        unsafe {
            std::env::remove_var(env_key);
        }

        assert!(
            result.is_err(),
            "config-rs must reject unknown env var field"
        );
        assert!(
            matches!(result.unwrap_err(), ConfigLoadError::Deserialize(_)),
            "error must be Deserialize variant (config-rs enforces deny_unknown_fields)"
        );
    }
}
