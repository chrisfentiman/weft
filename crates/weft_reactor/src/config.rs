//! Pipeline configuration: PipelineConfig, ActivityRef, ReactorConfig, BudgetConfig, LoopHooks.
//!
//! Configuration is parsed from TOML. ActivityRef supports both simple name
//! references (`"validate"`) and table references with optional retry, timeout,
//! and heartbeat config (`{ name = "generate", retry = { max_retries = 3 }, timeout_secs = 60 }`).

use serde::Deserialize;

/// Retry policy for transient activity failures.
///
/// Applied by the Reactor when an activity pushes `ActivityFailed`.
/// The Reactor checks whether the error is retryable, respects the
/// budget and cancellation state, and re-invokes the activity.
///
/// Backoff is exponential with jitter: actual_backoff = min(
///   initial_backoff * backoff_multiplier^attempt, max_backoff
/// ) + random_jitter(0..25% of calculated backoff).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (not counting the initial attempt).
    /// Total attempts = 1 (initial) + max_retries.
    pub max_retries: u32,
    /// Initial backoff duration in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds. Caps exponential growth.
    pub max_backoff_ms: u64,
    /// Multiplier for exponential backoff. Typical value: 2.0.
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30_000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Pipeline configuration. Parsed from TOML.
///
/// Defines which activities run in what order for a named pipeline.
/// Multiple pipelines can be defined (e.g., "default", "judge",
/// "council-member"). The Reactor selects the pipeline by name
/// from the execution's `pipeline_name`.
///
/// Activities are referenced by name (matching ActivityRegistry keys).
/// The Reactor resolves names to Arc<dyn Activity> at startup and
/// returns an error if any name is not found in the registry.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name. Must be unique.
    pub name: String,

    /// Activities that run sequentially before the dispatch loop.
    /// Executed in order. If any fails (pushes ActivityFailed), the execution fails.
    /// Typical: ["validate", "hook_request_start", "route", "assemble_prompt"]
    pub pre_loop: Vec<ActivityRef>,

    /// Activities that run sequentially after the dispatch loop completes.
    /// Typical: ["assemble_response", "hook_request_end"]
    pub post_loop: Vec<ActivityRef>,

    /// The activity that performs generation. Spawned as an event
    /// producer inside the dispatch loop. It pushes Generated events
    /// (streamed token-by-token) and GenerationStarted/GenerationCompleted
    /// lifecycle events.
    ///
    /// Supports retry, timeout, and heartbeat config via ActivityRef::WithConfig.
    pub generate: ActivityRef,

    /// Activity to call for command execution.
    /// Called once per command invocation found in the generated events.
    ///
    /// Supports retry and timeout config via ActivityRef::WithConfig.
    pub execute_command: ActivityRef,

    /// Hook activities to run at specific lifecycle points within
    /// the dispatch loop.
    #[serde(default)]
    pub loop_hooks: LoopHooks,
}

/// Reference to an activity in the registry, with optional config.
///
/// Supports two TOML forms:
/// - Simple string: `"validate"` → `ActivityRef::Name("validate")`
/// - Table with config: `{ name = "generate", retry = { ... }, timeout_secs = 60 }`
///   → `ActivityRef::WithConfig { name, config, retry, timeout_secs, heartbeat_interval_secs }`
///
/// The `#[serde(untagged)]` attribute makes serde try `WithConfig` first
/// (table), then `Name` (string). A plain string always parses as `Name`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ActivityRef {
    /// Simple name reference: `"validate"`
    Name(String),

    /// Name with optional metadata and runtime config.
    ///
    /// All extra fields are optional. A table with only `name` (no retry,
    /// timeout, or heartbeat) is still parsed as `WithConfig`.
    WithConfig {
        name: String,
        /// Activity-specific configuration passed to the activity via
        /// `ActivityInput.metadata`. If not specified, defaults to `Value::Null`.
        #[serde(default)]
        config: serde_json::Value,
        /// Optional retry policy for transient failures.
        /// If None, the activity is not retried on failure.
        #[serde(default)]
        retry: Option<RetryPolicy>,
        /// Per-activity timeout in seconds. If the activity does not
        /// complete within this duration, it is cancelled and treated
        /// as a failure (subject to retry policy).
        /// For generation activities, this is a per-chunk timeout:
        /// the timer resets on each received chunk.
        #[serde(default)]
        timeout_secs: Option<u64>,
        /// Heartbeat interval in seconds. Activities with this set
        /// must push Heartbeat events at this interval. The Reactor
        /// cancels activities that miss 2x the interval.
        #[serde(default)]
        heartbeat_interval_secs: Option<u64>,
    },
}

impl ActivityRef {
    /// Returns the activity name regardless of variant.
    pub fn name(&self) -> &str {
        match self {
            ActivityRef::Name(n) => n,
            ActivityRef::WithConfig { name, .. } => name,
        }
    }

    /// Returns the activity-specific config value.
    ///
    /// For `Name` variants (no config), returns `Value::Null`.
    /// For `WithConfig`, returns the `config` field.
    pub fn config(&self) -> serde_json::Value {
        match self {
            ActivityRef::Name(_) => serde_json::Value::Null,
            ActivityRef::WithConfig { config, .. } => config.clone(),
        }
    }

    /// Returns the retry policy, if configured.
    ///
    /// `None` for `Name` variants and `WithConfig` variants without a retry policy.
    pub fn retry_policy(&self) -> Option<&RetryPolicy> {
        match self {
            ActivityRef::Name(_) => None,
            ActivityRef::WithConfig { retry, .. } => retry.as_ref(),
        }
    }

    /// Returns the per-activity timeout in seconds, if configured.
    ///
    /// `None` for `Name` variants and `WithConfig` variants without a timeout.
    pub fn timeout_secs(&self) -> Option<u64> {
        match self {
            ActivityRef::Name(_) => None,
            ActivityRef::WithConfig { timeout_secs, .. } => *timeout_secs,
        }
    }

    /// Returns the heartbeat interval in seconds, if configured.
    ///
    /// `None` for `Name` variants and `WithConfig` variants without a heartbeat interval.
    pub fn heartbeat_interval_secs(&self) -> Option<u64> {
        match self {
            ActivityRef::Name(_) => None,
            ActivityRef::WithConfig {
                heartbeat_interval_secs,
                ..
            } => *heartbeat_interval_secs,
        }
    }
}

/// Hook activities that fire at specific points in the dispatch loop.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoopHooks {
    /// Fires before each generation call. Can block (cancel execution).
    #[serde(default)]
    pub pre_generate: Vec<ActivityRef>,

    /// Fires when the generative source signals Done (before response assembly).
    /// Can block with feedback (inject message, retry generation).
    #[serde(default)]
    pub pre_response: Vec<ActivityRef>,

    /// Fires before each command execution.
    #[serde(default)]
    pub pre_tool_use: Vec<ActivityRef>,

    /// Fires after each command execution.
    #[serde(default)]
    pub post_tool_use: Vec<ActivityRef>,
}

/// Top-level reactor configuration. Contains all pipeline definitions
/// and global settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ReactorConfig {
    /// Named pipelines. At least "default" must be defined.
    pub pipelines: Vec<PipelineConfig>,

    /// Default budget values for new executions.
    pub budget: BudgetConfig,
}

/// Default budget configuration values.
#[derive(Debug, Clone, Deserialize)]
pub struct BudgetConfig {
    /// Maximum generation calls per execution tree (root + all children).
    #[serde(default = "default_max_generation_calls")]
    pub max_generation_calls: u32,

    /// Maximum iterations of the command loop per execution.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    /// Maximum recursion depth for child executions.
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,

    /// Request timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Default per-generation-call timeout in seconds.
    /// Applied when no per-activity timeout is configured.
    /// The timer resets on each received chunk during streaming.
    #[serde(default = "default_generation_timeout_secs")]
    pub generation_timeout_secs: u64,

    /// Default per-command-execution timeout in seconds.
    /// Applied when no per-activity timeout is configured.
    #[serde(default = "default_command_timeout_secs")]
    pub command_timeout_secs: u64,
}

fn default_max_generation_calls() -> u32 {
    20
}
fn default_max_iterations() -> u32 {
    10
}
fn default_max_depth() -> u32 {
    5
}
fn default_timeout_secs() -> u64 {
    300
}
fn default_generation_timeout_secs() -> u64 {
    60
}
fn default_command_timeout_secs() -> u64 {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RetryPolicy ─────────────────────────────────────────────────────────

    #[test]
    fn retry_policy_default_values() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.initial_backoff_ms, 1000);
        assert_eq!(policy.max_backoff_ms, 30_000);
        assert!((policy.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_policy_deserializes() {
        let toml = r#"
            max_retries = 5
            initial_backoff_ms = 500
            max_backoff_ms = 60000
            backoff_multiplier = 1.5
        "#;
        let policy: RetryPolicy = toml::from_str(toml).expect("should deserialize");
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.initial_backoff_ms, 500);
        assert_eq!(policy.max_backoff_ms, 60_000);
        assert!((policy.backoff_multiplier - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_policy_serializes_and_round_trips() {
        let policy = RetryPolicy {
            max_retries: 2,
            initial_backoff_ms: 250,
            max_backoff_ms: 10_000,
            backoff_multiplier: 3.0,
        };
        let json = serde_json::to_string(&policy).expect("should serialize");
        let back: RetryPolicy = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(back.max_retries, 2);
        assert_eq!(back.initial_backoff_ms, 250);
    }

    // ── ActivityRef::Name ───────────────────────────────────────────────────

    #[test]
    fn activity_ref_name_from_string() {
        // TOML doesn't allow a bare string value at the document level.
        // Parse via serde_json (which ActivityRef must also support as a JSON string).
        let r: ActivityRef = serde_json::from_str(r#""validate""#).expect("should parse from JSON");
        assert_eq!(r.name(), "validate");
    }

    #[test]
    fn activity_ref_name_config_is_null() {
        let r = ActivityRef::Name("validate".to_string());
        assert_eq!(r.config(), serde_json::Value::Null);
    }

    #[test]
    fn activity_ref_name_retry_policy_is_none() {
        let r = ActivityRef::Name("validate".to_string());
        assert!(r.retry_policy().is_none());
    }

    #[test]
    fn activity_ref_name_timeout_is_none() {
        let r = ActivityRef::Name("validate".to_string());
        assert!(r.timeout_secs().is_none());
    }

    #[test]
    fn activity_ref_name_heartbeat_is_none() {
        let r = ActivityRef::Name("validate".to_string());
        assert!(r.heartbeat_interval_secs().is_none());
    }

    // ── ActivityRef::WithConfig ─────────────────────────────────────────────

    #[test]
    fn activity_ref_with_config_name_and_timeout() {
        let toml = r#"
            name = "execute_command"
            timeout_secs = 10
        "#;
        let r: ActivityRef = toml::from_str(toml).expect("should parse");
        assert_eq!(r.name(), "execute_command");
        assert_eq!(r.timeout_secs(), Some(10));
        assert!(r.retry_policy().is_none());
        assert!(r.heartbeat_interval_secs().is_none());
    }

    #[test]
    fn activity_ref_with_config_retry_and_heartbeat() {
        let toml = r#"
            name = "generate"
            timeout_secs = 60
            heartbeat_interval_secs = 15
            [retry]
            max_retries = 3
            initial_backoff_ms = 1000
            max_backoff_ms = 30000
            backoff_multiplier = 2.0
        "#;
        let r: ActivityRef = toml::from_str(toml).expect("should parse");
        assert_eq!(r.name(), "generate");
        assert_eq!(r.timeout_secs(), Some(60));
        assert_eq!(r.heartbeat_interval_secs(), Some(15));
        let policy = r.retry_policy().expect("should have retry policy");
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.initial_backoff_ms, 1000);
        assert_eq!(policy.max_backoff_ms, 30_000);
        assert!((policy.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn activity_ref_with_config_returns_config_value() {
        let toml = r#"
            name = "hook_pre_route"
            [config]
            domain = "model"
        "#;
        let r: ActivityRef = toml::from_str(toml).expect("should parse");
        let config = r.config();
        assert_eq!(config["domain"], serde_json::json!("model"));
    }

    #[test]
    fn activity_ref_with_config_no_extra_fields_is_with_config_variant() {
        // A TOML table with only 'name' should still parse as WithConfig.
        let toml = r#"name = "route""#;
        let r: ActivityRef = toml::from_str(toml).expect("should parse");
        // Name comes through regardless of variant.
        assert_eq!(r.name(), "route");
    }

    // ── PipelineConfig deserialization ──────────────────────────────────────

    #[test]
    fn pipeline_config_with_string_refs() {
        let toml = r#"
            name = "judge"
            generate = "generate"
            execute_command = "execute_command"
            pre_loop = ["validate", "assemble_prompt"]
            post_loop = ["assemble_response"]
        "#;
        let config: PipelineConfig = toml::from_str(toml).expect("should parse");
        assert_eq!(config.name, "judge");
        assert_eq!(config.generate.name(), "generate");
        assert_eq!(config.execute_command.name(), "execute_command");
        assert_eq!(config.pre_loop.len(), 2);
        assert_eq!(config.pre_loop[0].name(), "validate");
        assert_eq!(config.pre_loop[1].name(), "assemble_prompt");
        assert_eq!(config.post_loop.len(), 1);
        assert_eq!(config.post_loop[0].name(), "assemble_response");
    }

    #[test]
    fn pipeline_config_with_table_generate() {
        let toml = r#"
            name = "default"
            pre_loop = ["validate", "route", "assemble_prompt"]
            post_loop = ["assemble_response"]
            execute_command = "execute_command"

            [generate]
            name = "generate"
            timeout_secs = 60
            heartbeat_interval_secs = 15
            [generate.retry]
            max_retries = 3
            initial_backoff_ms = 1000
            max_backoff_ms = 30000
            backoff_multiplier = 2.0
        "#;
        let config: PipelineConfig = toml::from_str(toml).expect("should parse");
        assert_eq!(config.generate.name(), "generate");
        assert_eq!(config.generate.timeout_secs(), Some(60));
        assert_eq!(config.generate.heartbeat_interval_secs(), Some(15));
        let policy = config.generate.retry_policy().expect("should have retry");
        assert_eq!(policy.max_retries, 3);
    }

    #[test]
    fn pipeline_config_loop_hooks_default_empty() {
        let toml = r#"
            name = "simple"
            generate = "generate"
            execute_command = "execute_command"
            pre_loop = ["validate"]
            post_loop = ["assemble_response"]
        "#;
        let config: PipelineConfig = toml::from_str(toml).expect("should parse");
        assert!(config.loop_hooks.pre_generate.is_empty());
        assert!(config.loop_hooks.pre_response.is_empty());
        assert!(config.loop_hooks.pre_tool_use.is_empty());
        assert!(config.loop_hooks.post_tool_use.is_empty());
    }

    #[test]
    fn pipeline_config_with_loop_hooks() {
        let toml = r#"
            name = "default"
            generate = "generate"
            execute_command = "execute_command"
            pre_loop = ["validate", "route", "assemble_prompt"]
            post_loop = ["assemble_response", "hook_request_end"]

            [loop_hooks]
            pre_response = ["hook_pre_response"]
            pre_tool_use = ["hook_pre_tool_use"]
            post_tool_use = ["hook_post_tool_use"]
        "#;
        let config: PipelineConfig = toml::from_str(toml).expect("should parse");
        assert_eq!(config.loop_hooks.pre_response.len(), 1);
        assert_eq!(
            config.loop_hooks.pre_response[0].name(),
            "hook_pre_response"
        );
        assert_eq!(config.loop_hooks.pre_tool_use.len(), 1);
        assert_eq!(config.loop_hooks.post_tool_use.len(), 1);
    }

    // ── Missing required fields produce parse errors ─────────────────────────
    // Spec test expectation (line 2744): test missing required fields produce parse errors.

    #[test]
    fn pipeline_config_missing_generate_field_fails() {
        // `generate` is required on PipelineConfig. Omitting it must produce a parse error.
        let toml = r#"
            name = "default"
            execute_command = "execute_command"
            pre_loop = ["validate"]
            post_loop = ["assemble_response"]
        "#;
        let result: Result<PipelineConfig, _> = toml::from_str(toml);
        assert!(
            result.is_err(),
            "expected parse error when `generate` is missing, got: {result:?}"
        );
    }

    #[test]
    fn pipeline_config_missing_name_field_fails() {
        // `name` is required on PipelineConfig. Omitting it must produce a parse error.
        let toml = r#"
            generate = "generate"
            execute_command = "execute_command"
            pre_loop = ["validate"]
            post_loop = ["assemble_response"]
        "#;
        let result: Result<PipelineConfig, _> = toml::from_str(toml);
        assert!(
            result.is_err(),
            "expected parse error when `name` is missing, got: {result:?}"
        );
    }

    #[test]
    fn pipeline_config_missing_pre_loop_field_fails() {
        // `pre_loop` is required (Vec<ActivityRef>, not Option). Omitting must fail.
        let toml = r#"
            name = "default"
            generate = "generate"
            execute_command = "execute_command"
            post_loop = ["assemble_response"]
        "#;
        let result: Result<PipelineConfig, _> = toml::from_str(toml);
        assert!(
            result.is_err(),
            "expected parse error when `pre_loop` is missing, got: {result:?}"
        );
    }

    // ── BudgetConfig ────────────────────────────────────────────────────────

    #[test]
    fn budget_config_defaults() {
        let toml = ""; // all defaults
        // BudgetConfig can't be deserialized from empty string directly without wrapping
        // in a struct, so test defaults individually via default functions.
        assert_eq!(default_max_generation_calls(), 20);
        assert_eq!(default_max_iterations(), 10);
        assert_eq!(default_max_depth(), 5);
        assert_eq!(default_timeout_secs(), 300);
        assert_eq!(default_generation_timeout_secs(), 60);
        assert_eq!(default_command_timeout_secs(), 10);
        let _ = toml; // suppress unused warning
    }

    #[test]
    fn budget_config_explicit_values() {
        let toml = r#"
            max_generation_calls = 50
            max_iterations = 20
            max_depth = 3
            timeout_secs = 600
            generation_timeout_secs = 120
            command_timeout_secs = 30
        "#;
        let config: BudgetConfig = toml::from_str(toml).expect("should parse");
        assert_eq!(config.max_generation_calls, 50);
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.max_depth, 3);
        assert_eq!(config.timeout_secs, 600);
        assert_eq!(config.generation_timeout_secs, 120);
        assert_eq!(config.command_timeout_secs, 30);
    }

    // ── ReactorConfig ───────────────────────────────────────────────────────

    #[test]
    fn reactor_config_full_example() {
        let toml = r#"
            [budget]
            max_generation_calls = 20
            max_iterations = 10
            max_depth = 5
            timeout_secs = 300
            generation_timeout_secs = 60
            command_timeout_secs = 10

            [[pipelines]]
            name = "default"
            pre_loop = ["validate", "hook_request_start", "route", "assemble_prompt"]
            post_loop = ["assemble_response", "hook_request_end"]
            execute_command = "execute_command"

            [pipelines.generate]
            name = "generate"
            timeout_secs = 60
            heartbeat_interval_secs = 15
            [pipelines.generate.retry]
            max_retries = 3
            initial_backoff_ms = 1000
            max_backoff_ms = 30000
            backoff_multiplier = 2.0

            [pipelines.loop_hooks]
            pre_response = ["hook_pre_response"]
            pre_tool_use = ["hook_pre_tool_use"]
            post_tool_use = ["hook_post_tool_use"]

            [[pipelines]]
            name = "judge"
            generate = "generate"
            execute_command = "execute_command"
            pre_loop = ["validate", "assemble_prompt"]
            post_loop = ["assemble_response"]
        "#;
        let config: ReactorConfig = toml::from_str(toml).expect("should parse");
        assert_eq!(config.pipelines.len(), 2);
        assert_eq!(config.pipelines[0].name, "default");
        assert_eq!(config.pipelines[1].name, "judge");
        assert_eq!(config.budget.max_generation_calls, 20);
        assert_eq!(config.budget.generation_timeout_secs, 60);
        assert_eq!(config.budget.command_timeout_secs, 10);

        // Check default pipeline generate config
        let generate_ref = &config.pipelines[0].generate;
        assert_eq!(generate_ref.name(), "generate");
        assert_eq!(generate_ref.timeout_secs(), Some(60));
        assert_eq!(generate_ref.heartbeat_interval_secs(), Some(15));
        assert_eq!(generate_ref.retry_policy().unwrap().max_retries, 3);

        // Check judge pipeline uses simple string refs
        let judge_generate = &config.pipelines[1].generate;
        assert_eq!(judge_generate.name(), "generate");
        assert!(judge_generate.retry_policy().is_none());
    }

    #[test]
    fn pre_loop_mixed_string_and_table_refs() {
        let toml = r#"
            name = "default"
            generate = "generate"
            execute_command = "execute_command"
            post_loop = ["assemble_response"]

            pre_loop = [
                "validate",
                { name = "hook_request_start", config = { event = "request_start" } },
                "route",
                "assemble_prompt",
            ]
        "#;
        let config: PipelineConfig = toml::from_str(toml).expect("should parse");
        assert_eq!(config.pre_loop.len(), 4);
        assert_eq!(config.pre_loop[0].name(), "validate");
        assert_eq!(config.pre_loop[1].name(), "hook_request_start");
        assert_eq!(config.pre_loop[2].name(), "route");
        assert_eq!(config.pre_loop[3].name(), "assemble_prompt");

        // The hook activity has a config value
        let hook_config = config.pre_loop[1].config();
        assert_eq!(hook_config["event"], serde_json::json!("request_start"));
    }

    // ── LoopHooks defaults ──────────────────────────────────────────────────

    #[test]
    fn loop_hooks_default_all_empty() {
        let hooks = LoopHooks::default();
        assert!(hooks.pre_generate.is_empty());
        assert!(hooks.pre_response.is_empty());
        assert!(hooks.pre_tool_use.is_empty());
        assert!(hooks.post_tool_use.is_empty());
    }
}
