//! Fully-resolved configuration for request processing.
//!
//! `ResolvedConfig` carries the "hot" config fields that activities read during
//! request processing, plus pre-computed provider-derived data (model candidates,
//! model infos, provider-to-model mapping). "Cold" config (provider endpoints,
//! ONNX paths, bind address, `request_end_concurrency`) lives in `WeftConfig` and
//! is used only at startup.
//!
//! Built by [`ResolvedConfig::from_operator`] at startup and on config reload.
//! Swapped atomically via [`crate::config_store::ConfigStore`].

use crate::config::{ProviderConfig, WeftConfig};
use crate::routing::ModelInfo;

// ── ModelCandidate ─────────────────────────────────────────────────────────────

/// A model routing candidate, pre-computed from provider config.
///
/// Equivalent shape to `weft_router_trait::RoutingCandidate` but defined
/// here to avoid a circular dependency between `weft_core` and `weft_router_trait`.
/// The router-facing `RoutingCandidate` is constructed from this type at the
/// `build_model_candidates` call site in `main.rs`.
#[derive(Debug, Clone)]
pub struct ModelCandidate {
    /// Model routing name (same as `RoutingCandidate::id`).
    pub name: String,
    /// Example queries for semantic scoring.
    pub examples: Vec<String>,
}

// ── TenantOverrides ────────────────────────────────────────────────────────────

/// Per-tenant overrides. All fields Optional -- None means "use operator default."
///
/// Deserialized from a sparse JSON/TOML document. Only specified fields
/// override the operator config. This type is defined but not wired to
/// any storage backend yet.
///
/// **Design decisions:**
/// - `has_memory_stores` is NOT included. Memory store availability is
///   infrastructure-level, not tenant-configurable (see spec D9).
/// - Threshold fields use `Option<f32>`, not `Clearable<f32>`. This means
///   tenants cannot explicitly clear a threshold to None -- they can only
///   override with a new value or inherit the operator default (see spec D11).
/// - Provider-derived data (`model_candidates`, `model_infos`, `provider_names`)
///   is not tenant-overridable.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TenantOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_command_iterations: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classifier_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classifier_max_commands: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_tools_when_unnecessary: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_domain_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_domain_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_necessity_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_necessity_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_domain_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_domain_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_pre_response_retries: Option<u32>,
}

// ── ResolvedConfigError ────────────────────────────────────────────────────────

/// Errors from resolved config validation.
#[derive(Debug, thiserror::Error)]
pub enum ResolvedConfigError {
    #[error("field '{field}': {reason}")]
    InvalidField { field: &'static str, reason: String },
}

// ── ResolvedConfig ─────────────────────────────────────────────────────────────

/// Fully-resolved configuration for request processing.
///
/// All scalar fields are concrete values -- no Options (except domain thresholds,
/// which are semantically optional). Built by merging operator defaults with
/// tenant overrides (when tenants exist). Activities access this via
/// `ActivityInput.config` or `ServiceLocator::resolved_config()`.
///
/// This struct contains "hot" config (values read during request processing)
/// plus pre-computed provider-derived data (model candidates, model infos,
/// provider-to-model mapping). "Cold" config (provider endpoints, ONNX paths,
/// bind address, `request_end_concurrency`) lives in `WeftConfig` and is used
/// only at startup.
///
/// # Construction
///
/// Use [`ResolvedConfig::from_operator`] to build from `WeftConfig`.
/// Use [`ResolvedConfig::from_operator_with_overrides`] when tenant overrides exist.
/// Always call [`ResolvedConfig::validate`] after construction.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    // ── Gateway ────────────────────────────────────────────────
    /// System prompt prepended to every conversation.
    pub system_prompt: String,
    /// Maximum iterations of the command loop.
    pub max_command_iterations: u32,
    /// Overall request timeout in seconds.
    pub request_timeout_secs: u64,

    // ── Router behavior ────────────────────────────────────────
    /// Classifier score threshold for command filtering.
    pub classifier_threshold: f32,
    /// Maximum commands to include regardless of threshold.
    pub classifier_max_commands: usize,
    /// Default model routing name.
    pub default_model: String,
    /// Whether to skip command injection when tools aren't needed.
    pub skip_tools_when_unnecessary: bool,

    // ── Domain thresholds ──────────────────────────────────────
    /// Model domain score threshold. None = always pick highest.
    pub model_domain_threshold: Option<f32>,
    /// Model domain enabled.
    pub model_domain_enabled: bool,
    /// Tool necessity domain threshold.
    pub tool_necessity_threshold: Option<f32>,
    /// Tool necessity domain enabled.
    pub tool_necessity_enabled: bool,
    /// Memory domain threshold.
    pub memory_domain_threshold: Option<f32>,
    /// Memory domain enabled.
    pub memory_domain_enabled: bool,

    // ── Hooks ──────────────────────────────────────────────────
    /// Maximum PreResponse hook retry attempts.
    pub max_pre_response_retries: u32,

    // ── Memory ─────────────────────────────────────────────────
    /// Whether any memory stores are configured. Activities use this
    /// to decide whether /recall and /remember commands are available.
    /// Not tenant-overridable (see spec D9).
    pub has_memory_stores: bool,

    // ── Pre-computed provider data (see spec D10) ──────────────
    /// Model routing candidates, pre-built from provider config.
    /// Used by `ModelSelectionActivity` for semantic scoring.
    /// Converts to `weft_router_trait::RoutingCandidate` at the call site.
    pub model_candidates: Vec<ModelCandidate>,
    /// Model info list, pre-built from provider config.
    /// Used by `ModelSelectionActivity` for filter resolution.
    pub model_infos: Vec<ModelInfo>,
    /// Model-to-provider name mapping.
    /// Vec of (model_routing_name, provider_name) pairs.
    /// Used by `ProviderResolutionActivity` to find which provider
    /// owns a given model without accessing `WeftConfig`.
    pub provider_names: Vec<(String, String)>,
}

impl ResolvedConfig {
    /// Build a `ResolvedConfig` from the operator's `WeftConfig`.
    ///
    /// Projects the hot-path fields from the full config tree into
    /// a flat struct and pre-computes provider-derived data.
    /// Called at startup and on config reload.
    pub fn from_operator(config: &WeftConfig) -> Self {
        let model_domain = config.router.domains.model.as_ref();
        let tool_domain = config.router.domains.tool_necessity.as_ref();
        let memory_domain = config.router.domains.memory.as_ref();

        // Pre-compute provider-derived data (spec D10).
        let model_candidates = Self::build_model_candidates(&config.router.providers);
        let model_infos = Self::build_model_infos(&config.router.providers);
        let provider_names = Self::build_provider_names(&config.router.providers);

        Self {
            system_prompt: config.gateway.system_prompt.clone(),
            max_command_iterations: config.gateway.max_command_iterations,
            request_timeout_secs: config.gateway.request_timeout_secs,
            classifier_threshold: config.router.classifier.threshold,
            classifier_max_commands: config.router.classifier.max_commands,
            default_model: config.router.effective_default_model().to_string(),
            skip_tools_when_unnecessary: config.router.skip_tools_when_unnecessary,
            model_domain_threshold: model_domain.and_then(|d| d.threshold),
            model_domain_enabled: model_domain.map(|d| d.enabled).unwrap_or(true),
            tool_necessity_threshold: tool_domain.and_then(|d| d.threshold),
            tool_necessity_enabled: tool_domain.map(|d| d.enabled).unwrap_or(true),
            memory_domain_threshold: memory_domain.and_then(|d| d.threshold),
            memory_domain_enabled: memory_domain.map(|d| d.enabled).unwrap_or(true),
            max_pre_response_retries: config.effective_max_pre_response_retries(),
            has_memory_stores: config.memory.as_ref().is_some_and(|m| !m.stores.is_empty()),
            model_candidates,
            model_infos,
            provider_names,
        }
    }

    /// Build a `ResolvedConfig` from operator config with tenant overrides applied.
    ///
    /// Each `Some` field in `overrides` replaces the operator default.
    /// `None` fields inherit the operator value.
    ///
    /// Infrastructure-level fields (`has_memory_stores`, `model_candidates`,
    /// `model_infos`, `provider_names`) are always taken from the operator config
    /// and cannot be overridden by tenants.
    pub fn from_operator_with_overrides(config: &WeftConfig, overrides: &TenantOverrides) -> Self {
        let base = Self::from_operator(config);
        Self {
            system_prompt: overrides
                .system_prompt
                .clone()
                .unwrap_or(base.system_prompt),
            max_command_iterations: overrides
                .max_command_iterations
                .unwrap_or(base.max_command_iterations),
            request_timeout_secs: overrides
                .request_timeout_secs
                .unwrap_or(base.request_timeout_secs),
            classifier_threshold: overrides
                .classifier_threshold
                .unwrap_or(base.classifier_threshold),
            classifier_max_commands: overrides
                .classifier_max_commands
                .unwrap_or(base.classifier_max_commands),
            default_model: overrides
                .default_model
                .clone()
                .unwrap_or(base.default_model),
            skip_tools_when_unnecessary: overrides
                .skip_tools_when_unnecessary
                .unwrap_or(base.skip_tools_when_unnecessary),
            model_domain_threshold: overrides
                .model_domain_threshold
                .or(base.model_domain_threshold),
            model_domain_enabled: overrides
                .model_domain_enabled
                .unwrap_or(base.model_domain_enabled),
            tool_necessity_threshold: overrides
                .tool_necessity_threshold
                .or(base.tool_necessity_threshold),
            tool_necessity_enabled: overrides
                .tool_necessity_enabled
                .unwrap_or(base.tool_necessity_enabled),
            memory_domain_threshold: overrides
                .memory_domain_threshold
                .or(base.memory_domain_threshold),
            memory_domain_enabled: overrides
                .memory_domain_enabled
                .unwrap_or(base.memory_domain_enabled),
            max_pre_response_retries: overrides
                .max_pre_response_retries
                .unwrap_or(base.max_pre_response_retries),
            // Infrastructure-level fields: not tenant-overridable (spec D9).
            has_memory_stores: base.has_memory_stores,
            // Provider-derived data: not tenant-overridable (cold config).
            model_candidates: base.model_candidates,
            model_infos: base.model_infos,
            provider_names: base.provider_names,
        }
    }

    /// Validate the resolved config.
    ///
    /// Called after construction (from operator or from operator+overrides).
    /// Checks value ranges and cross-field constraints.
    pub fn validate(&self) -> Result<(), ResolvedConfigError> {
        if self.max_command_iterations == 0 {
            return Err(ResolvedConfigError::InvalidField {
                field: "max_command_iterations",
                reason: "must be at least 1".to_string(),
            });
        }
        if self.request_timeout_secs < 5 {
            return Err(ResolvedConfigError::InvalidField {
                field: "request_timeout_secs",
                reason: "must be at least 5 seconds".to_string(),
            });
        }
        if self.classifier_threshold < 0.0 || self.classifier_threshold > 1.0 {
            return Err(ResolvedConfigError::InvalidField {
                field: "classifier_threshold",
                reason: format!(
                    "must be between 0.0 and 1.0, got {}",
                    self.classifier_threshold
                ),
            });
        }
        if self.classifier_max_commands == 0 {
            return Err(ResolvedConfigError::InvalidField {
                field: "classifier_max_commands",
                reason: "must be at least 1".to_string(),
            });
        }
        if self.default_model.is_empty() {
            return Err(ResolvedConfigError::InvalidField {
                field: "default_model",
                reason: "must not be empty".to_string(),
            });
        }
        Ok(())
    }

    /// Find the provider name for a model routing name.
    ///
    /// Searches the pre-computed `provider_names` mapping.
    /// Returns `"unknown"` if no match found (config/wiring error).
    pub fn provider_name_for(&self, model_name: &str) -> &str {
        self.provider_names
            .iter()
            .find(|(m, _)| m == model_name)
            .map(|(_, p)| p.as_str())
            .unwrap_or("unknown")
    }

    // ── Provider-derived data builders (spec D10) ──────────────────────────

    /// Build model routing candidates from provider config.
    ///
    /// Each model entry becomes a `ModelCandidate` with the model
    /// routing name as `name` and its examples array.
    fn build_model_candidates(providers: &[ProviderConfig]) -> Vec<ModelCandidate> {
        providers
            .iter()
            .flat_map(|p| {
                p.models.iter().map(|m| ModelCandidate {
                    name: m.name.clone(),
                    examples: m.examples.clone(),
                })
            })
            .collect()
    }

    /// Build `ModelInfo` list from provider config for filter resolution.
    fn build_model_infos(providers: &[ProviderConfig]) -> Vec<ModelInfo> {
        providers
            .iter()
            .flat_map(|p| {
                p.models.iter().map(|m| ModelInfo {
                    routing_name: m.name.clone(),
                    provider_name: p.name.clone(),
                    capabilities: vec![],
                })
            })
            .collect()
    }

    /// Build model-to-provider name mapping.
    ///
    /// Returns Vec of (model_routing_name, provider_name) pairs.
    fn build_provider_names(providers: &[ProviderConfig]) -> Vec<(String, String)> {
        providers
            .iter()
            .flat_map(|p| {
                p.models
                    .iter()
                    .map(move |m| (m.name.clone(), p.name.clone()))
            })
            .collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    /// Minimal valid TOML config for use in unit tests.
    ///
    /// Constructs a WeftConfig that passes validate() with one provider and one model.
    fn minimal_toml() -> WeftConfig {
        let toml = r#"
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
examples = ["Write a poem", "Explain quantum physics", "What is 2+2?"]
capabilities = ["chat_completions"]
"#;
        toml::from_str(toml).expect("test TOML must parse")
    }

    /// TOML config with two providers and multiple models.
    fn two_provider_toml() -> WeftConfig {
        let toml = r#"
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
name = "complex"
model = "claude-opus-4-20250514"
max_tokens = 8192
examples = ["Complex reasoning task", "Multi-step analysis", "Deep research"]
capabilities = ["chat_completions"]

[[router.providers]]
name = "openai"
wire_format = "openai"
api_key = "sk-test2"

[[router.providers.models]]
name = "fast"
model = "gpt-4o-mini"
max_tokens = 4096
examples = ["Quick question", "Simple lookup", "Fast summary"]
capabilities = ["chat_completions"]
"#;
        toml::from_str(toml).expect("test TOML must parse")
    }

    // ── from_operator tests ────────────────────────────────────────────────

    #[test]
    fn test_from_operator_projects_scalar_fields() {
        let config = minimal_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        assert_eq!(resolved.system_prompt, "You are a helpful assistant.");
        assert_eq!(resolved.max_command_iterations, 5);
        assert_eq!(resolved.request_timeout_secs, 30);
        assert_eq!(resolved.classifier_threshold, 0.5);
        assert_eq!(resolved.classifier_max_commands, 10);
        assert_eq!(resolved.default_model, "claude");
        assert!(resolved.skip_tools_when_unnecessary); // default true
    }

    #[test]
    fn test_from_operator_defaults_when_domains_absent() {
        let config = minimal_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        // All domains absent in minimal config => enabled defaults to true, threshold None.
        assert!(resolved.model_domain_enabled);
        assert!(resolved.model_domain_threshold.is_none());
        assert!(resolved.tool_necessity_enabled);
        assert!(resolved.tool_necessity_threshold.is_none());
        assert!(resolved.memory_domain_enabled);
        assert!(resolved.memory_domain_threshold.is_none());
    }

    #[test]
    fn test_from_operator_no_memory_config_sets_has_memory_stores_false() {
        let config = minimal_toml();
        let resolved = ResolvedConfig::from_operator(&config);
        assert!(!resolved.has_memory_stores);
    }

    #[test]
    fn test_from_operator_precomputes_model_candidates() {
        let config = two_provider_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        assert_eq!(resolved.model_candidates.len(), 2);
        let names: Vec<&str> = resolved
            .model_candidates
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert!(names.contains(&"complex"));
        assert!(names.contains(&"fast"));
    }

    #[test]
    fn test_from_operator_model_candidates_include_examples() {
        let config = two_provider_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        let complex = resolved
            .model_candidates
            .iter()
            .find(|c| c.name == "complex")
            .expect("complex candidate must exist");
        assert_eq!(complex.examples.len(), 3);
        assert!(
            complex
                .examples
                .contains(&"Complex reasoning task".to_string())
        );
    }

    #[test]
    fn test_from_operator_precomputes_model_infos() {
        let config = two_provider_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        assert_eq!(resolved.model_infos.len(), 2);

        let complex_info = resolved
            .model_infos
            .iter()
            .find(|i| i.routing_name == "complex")
            .expect("complex model info must exist");
        assert_eq!(complex_info.provider_name, "anthropic");

        let fast_info = resolved
            .model_infos
            .iter()
            .find(|i| i.routing_name == "fast")
            .expect("fast model info must exist");
        assert_eq!(fast_info.provider_name, "openai");
    }

    #[test]
    fn test_from_operator_precomputes_provider_names() {
        let config = two_provider_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        assert_eq!(resolved.provider_names.len(), 2);
        assert!(
            resolved
                .provider_names
                .contains(&("complex".to_string(), "anthropic".to_string()))
        );
        assert!(
            resolved
                .provider_names
                .contains(&("fast".to_string(), "openai".to_string()))
        );
    }

    // ── provider_name_for tests ────────────────────────────────────────────

    #[test]
    fn test_provider_name_for_known_model() {
        let config = two_provider_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        assert_eq!(resolved.provider_name_for("complex"), "anthropic");
        assert_eq!(resolved.provider_name_for("fast"), "openai");
    }

    #[test]
    fn test_provider_name_for_unknown_model_returns_unknown() {
        let config = minimal_toml();
        let resolved = ResolvedConfig::from_operator(&config);

        assert_eq!(resolved.provider_name_for("nonexistent-model"), "unknown");
    }

    // ── from_operator_with_overrides tests ────────────────────────────────

    #[test]
    fn test_from_operator_with_overrides_empty_matches_from_operator() {
        let config = minimal_toml();
        let base = ResolvedConfig::from_operator(&config);
        let overrides = TenantOverrides::default();
        let with_overrides = ResolvedConfig::from_operator_with_overrides(&config, &overrides);

        assert_eq!(with_overrides.system_prompt, base.system_prompt);
        assert_eq!(
            with_overrides.max_command_iterations,
            base.max_command_iterations
        );
        assert_eq!(
            with_overrides.request_timeout_secs,
            base.request_timeout_secs
        );
        assert_eq!(
            with_overrides.classifier_threshold,
            base.classifier_threshold
        );
        assert_eq!(
            with_overrides.classifier_max_commands,
            base.classifier_max_commands
        );
        assert_eq!(with_overrides.default_model, base.default_model);
        assert_eq!(
            with_overrides.skip_tools_when_unnecessary,
            base.skip_tools_when_unnecessary
        );
        assert_eq!(
            with_overrides.model_domain_threshold,
            base.model_domain_threshold
        );
        assert_eq!(
            with_overrides.model_domain_enabled,
            base.model_domain_enabled
        );
        assert_eq!(
            with_overrides.tool_necessity_threshold,
            base.tool_necessity_threshold
        );
        assert_eq!(
            with_overrides.tool_necessity_enabled,
            base.tool_necessity_enabled
        );
        assert_eq!(
            with_overrides.memory_domain_threshold,
            base.memory_domain_threshold
        );
        assert_eq!(
            with_overrides.memory_domain_enabled,
            base.memory_domain_enabled
        );
        assert_eq!(
            with_overrides.max_pre_response_retries,
            base.max_pre_response_retries
        );
    }

    #[test]
    fn test_from_operator_with_overrides_all_fields_set() {
        let config = minimal_toml();
        let overrides = TenantOverrides {
            system_prompt: Some("Override prompt".to_string()),
            max_command_iterations: Some(10),
            request_timeout_secs: Some(60),
            classifier_threshold: Some(0.7),
            classifier_max_commands: Some(5),
            default_model: Some("fast".to_string()),
            skip_tools_when_unnecessary: Some(false),
            model_domain_threshold: Some(0.8),
            model_domain_enabled: Some(false),
            tool_necessity_threshold: Some(0.6),
            tool_necessity_enabled: Some(false),
            memory_domain_threshold: Some(0.4),
            memory_domain_enabled: Some(false),
            max_pre_response_retries: Some(3),
        };

        let resolved = ResolvedConfig::from_operator_with_overrides(&config, &overrides);

        assert_eq!(resolved.system_prompt, "Override prompt");
        assert_eq!(resolved.max_command_iterations, 10);
        assert_eq!(resolved.request_timeout_secs, 60);
        assert_eq!(resolved.classifier_threshold, 0.7);
        assert_eq!(resolved.classifier_max_commands, 5);
        assert_eq!(resolved.default_model, "fast");
        assert!(!resolved.skip_tools_when_unnecessary);
        assert_eq!(resolved.model_domain_threshold, Some(0.8));
        assert!(!resolved.model_domain_enabled);
        assert_eq!(resolved.tool_necessity_threshold, Some(0.6));
        assert!(!resolved.tool_necessity_enabled);
        assert_eq!(resolved.memory_domain_threshold, Some(0.4));
        assert!(!resolved.memory_domain_enabled);
        assert_eq!(resolved.max_pre_response_retries, 3);
    }

    #[test]
    fn test_from_operator_with_overrides_partial_merge() {
        let config = minimal_toml();
        let overrides = TenantOverrides {
            system_prompt: Some("Custom prompt".to_string()),
            classifier_threshold: Some(0.9),
            ..TenantOverrides::default()
        };

        let base = ResolvedConfig::from_operator(&config);
        let resolved = ResolvedConfig::from_operator_with_overrides(&config, &overrides);

        // Overridden fields.
        assert_eq!(resolved.system_prompt, "Custom prompt");
        assert_ne!(resolved.system_prompt, base.system_prompt);
        assert_eq!(resolved.classifier_threshold, 0.9);

        // Inherited fields.
        assert_eq!(resolved.max_command_iterations, base.max_command_iterations);
        assert_eq!(resolved.default_model, base.default_model);
    }

    #[test]
    fn test_from_operator_with_overrides_does_not_override_has_memory_stores() {
        let config = minimal_toml();
        let overrides = TenantOverrides::default();
        let resolved = ResolvedConfig::from_operator_with_overrides(&config, &overrides);

        // has_memory_stores is infrastructure-level: always from operator.
        assert!(!resolved.has_memory_stores);
    }

    #[test]
    fn test_from_operator_with_overrides_does_not_override_provider_data() {
        let config = two_provider_toml();
        let overrides = TenantOverrides::default();
        let base = ResolvedConfig::from_operator(&config);
        let resolved = ResolvedConfig::from_operator_with_overrides(&config, &overrides);

        // Provider-derived data is always from operator config.
        assert_eq!(resolved.model_candidates.len(), base.model_candidates.len());
        assert_eq!(resolved.model_infos.len(), base.model_infos.len());
        assert_eq!(resolved.provider_names.len(), base.provider_names.len());
    }

    // ── validate tests ─────────────────────────────────────────────────────

    #[test]
    fn test_validate_accepts_valid_config() {
        let config = minimal_toml();
        let resolved = ResolvedConfig::from_operator(&config);
        assert!(resolved.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_max_command_iterations() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.max_command_iterations = 0;

        let err = resolved.validate().unwrap_err();
        assert!(matches!(
            err,
            ResolvedConfigError::InvalidField {
                field: "max_command_iterations",
                ..
            }
        ));
        assert!(err.to_string().contains("max_command_iterations"));
    }

    #[test]
    fn test_validate_rejects_request_timeout_below_5() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.request_timeout_secs = 4;

        let err = resolved.validate().unwrap_err();
        assert!(matches!(
            err,
            ResolvedConfigError::InvalidField {
                field: "request_timeout_secs",
                ..
            }
        ));
    }

    #[test]
    fn test_validate_accepts_request_timeout_exactly_5() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.request_timeout_secs = 5;

        assert!(resolved.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_classifier_threshold_above_1() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.classifier_threshold = 1.1;

        let err = resolved.validate().unwrap_err();
        assert!(matches!(
            err,
            ResolvedConfigError::InvalidField {
                field: "classifier_threshold",
                ..
            }
        ));
        assert!(err.to_string().contains("1.1"));
    }

    #[test]
    fn test_validate_rejects_classifier_threshold_below_0() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.classifier_threshold = -0.1;

        let err = resolved.validate().unwrap_err();
        assert!(matches!(
            err,
            ResolvedConfigError::InvalidField {
                field: "classifier_threshold",
                ..
            }
        ));
    }

    #[test]
    fn test_validate_rejects_zero_classifier_max_commands() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.classifier_max_commands = 0;

        let err = resolved.validate().unwrap_err();
        assert!(matches!(
            err,
            ResolvedConfigError::InvalidField {
                field: "classifier_max_commands",
                ..
            }
        ));
    }

    #[test]
    fn test_validate_rejects_empty_default_model() {
        let config = minimal_toml();
        let mut resolved = ResolvedConfig::from_operator(&config);
        resolved.default_model = String::new();

        let err = resolved.validate().unwrap_err();
        assert!(matches!(
            err,
            ResolvedConfigError::InvalidField {
                field: "default_model",
                ..
            }
        ));
    }
}
