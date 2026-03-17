//! Provider registry: maps model routing names to provider instances,
//! with capability indexing for capability-aware routing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Provider;
use crate::provider::Capability;

/// Registry of named providers with capability indexing.
///
/// Thread-safe: providers are Arc'd and all maps are immutable after construction.
///
/// Supports two access patterns:
/// 1. By model name (existing): `get("claude-sonnet")` -> provider
/// 2. By capability (new): `models_with_capability(&cap)` -> &HashSet<String>
pub struct ProviderRegistry {
    /// Named providers. Key is the model routing name.
    providers: HashMap<String, Arc<dyn Provider>>,
    /// Name of the default model.
    default_name: String,
    /// Model identifier for each routing name.
    model_ids: HashMap<String, String>,
    /// Max tokens per model routing name.
    max_tokens: HashMap<String, u32>,
    /// Capabilities per model routing name.
    capabilities: HashMap<String, HashSet<Capability>>,
    /// Reverse index: capability -> set of model routing names that support it.
    capability_index: HashMap<Capability, HashSet<String>>,
}

impl ProviderRegistry {
    /// Construct a new provider registry with capability indexing.
    ///
    /// `providers`: Map of model_routing_name -> provider. Must contain at least one entry.
    /// `model_ids`: Map of model_routing_name -> model API identifier.
    /// `max_tokens`: Map of model_routing_name -> max tokens.
    /// `capabilities`: Map of model_routing_name -> set of capabilities.
    /// `default_name`: Must be a key in `providers`.
    ///
    /// # Panics
    ///
    /// Panics if `providers` is empty or if `default_name` is not a key in `providers`.
    /// These panics are defensive assertions that should never fire if
    /// `WeftConfig::validate()` was called first. They exist to prevent silent
    /// misconfiguration if someone constructs a `ProviderRegistry` without
    /// going through the validated config path. This is a startup-time check,
    /// not a runtime condition.
    pub fn new(
        providers: HashMap<String, Arc<dyn Provider>>,
        model_ids: HashMap<String, String>,
        max_tokens: HashMap<String, u32>,
        capabilities: HashMap<String, HashSet<Capability>>,
        default_name: String,
    ) -> Self {
        assert!(
            !providers.is_empty(),
            "provider registry must have at least one provider"
        );
        assert!(
            providers.contains_key(&default_name),
            "default provider '{}' not found in registry",
            default_name
        );

        // Build the reverse capability index: capability -> set of model routing names.
        let mut capability_index: HashMap<Capability, HashSet<String>> = HashMap::new();
        for (model_name, caps) in &capabilities {
            for cap in caps {
                capability_index
                    .entry(cap.clone())
                    .or_default()
                    .insert(model_name.clone());
            }
        }

        Self {
            providers,
            default_name,
            model_ids,
            max_tokens,
            capabilities,
            capability_index,
        }
    }

    /// Get a provider by model routing name. Returns the default if name is not found.
    pub fn get(&self, name: &str) -> &Arc<dyn Provider> {
        self.providers
            .get(name)
            .unwrap_or_else(|| self.providers.get(&self.default_name).unwrap())
    }

    /// Get the model API identifier for a routing name.
    pub fn model_id(&self, name: &str) -> Option<&str> {
        self.model_ids.get(name).map(|s| s.as_str())
    }

    /// Get the max tokens for a routing name.
    pub fn max_tokens_for(&self, name: &str) -> Option<u32> {
        self.max_tokens.get(name).copied()
    }

    /// Get the default provider.
    pub fn default_provider(&self) -> &Arc<dyn Provider> {
        self.providers.get(&self.default_name).unwrap()
    }

    /// Get the default model name.
    pub fn default_name(&self) -> &str {
        &self.default_name
    }

    /// Get all model routing names that support a given capability.
    ///
    /// Returns an empty set if no models support the capability.
    pub fn models_with_capability(&self, capability: &Capability) -> &HashSet<String> {
        static EMPTY: std::sync::LazyLock<HashSet<String>> = std::sync::LazyLock::new(HashSet::new);
        self.capability_index.get(capability).unwrap_or(&EMPTY)
    }

    /// Check if a specific model supports a given capability.
    pub fn model_has_capability(&self, model_name: &str, capability: &Capability) -> bool {
        self.capabilities
            .get(model_name)
            .is_some_and(|caps| caps.contains(capability))
    }

    /// Get the capabilities of a specific model.
    pub fn model_capabilities(&self, model_name: &str) -> Option<&HashSet<Capability>> {
        self.capabilities.get(model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProviderError, ProviderRequest, ProviderResponse};
    use async_trait::async_trait;

    struct StubProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for StubProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::Unsupported(format!(
                "stub provider '{}' does not handle requests",
                self.name
            )))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    fn stub(name: &str) -> Arc<dyn Provider> {
        Arc::new(StubProvider {
            name: name.to_string(),
        })
    }

    fn chat_cap() -> Capability {
        Capability::new(Capability::CHAT_COMPLETIONS)
    }

    fn embed_cap() -> Capability {
        Capability::new(Capability::EMBEDDINGS)
    }

    fn vision_cap() -> Capability {
        Capability::new(Capability::VISION)
    }

    fn build_registry() -> ProviderRegistry {
        let mut providers = HashMap::new();
        providers.insert("complex".to_string(), stub("anthropic-complex"));
        providers.insert("fast".to_string(), stub("anthropic-fast"));
        providers.insert("general".to_string(), stub("local-general"));

        let mut model_ids = HashMap::new();
        model_ids.insert(
            "complex".to_string(),
            "claude-sonnet-4-20250514".to_string(),
        );
        model_ids.insert("fast".to_string(), "claude-haiku-4-5-20251001".to_string());
        model_ids.insert("general".to_string(), "llama3.2".to_string());

        let mut max_tokens = HashMap::new();
        max_tokens.insert("complex".to_string(), 8192u32);
        max_tokens.insert("fast".to_string(), 2048u32);
        max_tokens.insert("general".to_string(), 4096u32);

        // complex: chat + vision; fast: chat only; general: chat only
        let mut capabilities: HashMap<String, HashSet<Capability>> = HashMap::new();
        capabilities.insert(
            "complex".to_string(),
            [chat_cap(), vision_cap()].into_iter().collect(),
        );
        capabilities.insert("fast".to_string(), [chat_cap()].into_iter().collect());
        capabilities.insert("general".to_string(), [chat_cap()].into_iter().collect());

        ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            capabilities,
            "complex".to_string(),
        )
    }

    #[test]
    fn test_new_panics_on_empty_providers() {
        let result = std::panic::catch_unwind(|| {
            ProviderRegistry::new(
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                "anything".to_string(),
            )
        });
        assert!(result.is_err(), "should panic on empty providers");
    }

    #[test]
    fn test_new_panics_on_missing_default() {
        let mut providers = HashMap::new();
        providers.insert("fast".to_string(), stub("fast"));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ProviderRegistry::new(
                providers,
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                "nonexistent".to_string(),
            )
        }));
        assert!(
            result.is_err(),
            "should panic when default is not in providers"
        );
    }

    #[test]
    fn test_get_named_provider() {
        let registry = build_registry();
        // Just verify get returns something (can't compare Arc<dyn Provider> by value)
        let _ = registry.get("fast");
        let _ = registry.get("complex");
    }

    #[test]
    fn test_get_unknown_name_returns_default() {
        let registry = build_registry();
        // Unknown name should return default provider (complex)
        // We verify by checking default_name
        assert_eq!(registry.default_name(), "complex");
        // get("unknown") should not panic
        let _ = registry.get("unknown_model");
    }

    #[test]
    fn test_model_id_returns_correct_identifier() {
        let registry = build_registry();
        assert_eq!(
            registry.model_id("complex"),
            Some("claude-sonnet-4-20250514")
        );
        assert_eq!(registry.model_id("fast"), Some("claude-haiku-4-5-20251001"));
        assert_eq!(registry.model_id("general"), Some("llama3.2"));
        assert_eq!(registry.model_id("unknown"), None);
    }

    #[test]
    fn test_max_tokens_for() {
        let registry = build_registry();
        assert_eq!(registry.max_tokens_for("complex"), Some(8192));
        assert_eq!(registry.max_tokens_for("fast"), Some(2048));
        assert_eq!(registry.max_tokens_for("general"), Some(4096));
        assert_eq!(registry.max_tokens_for("unknown"), None);
    }

    #[test]
    fn test_default_provider_and_name() {
        let registry = build_registry();
        assert_eq!(registry.default_name(), "complex");
        // default_provider() should not panic
        let _ = registry.default_provider();
    }

    // ── Capability index tests ─────────────────────────────────────────────

    #[test]
    fn test_models_with_capability_chat_completions() {
        let registry = build_registry();
        let models = registry.models_with_capability(&chat_cap());
        // All three models support chat_completions
        assert_eq!(models.len(), 3);
        assert!(models.contains("complex"));
        assert!(models.contains("fast"));
        assert!(models.contains("general"));
    }

    #[test]
    fn test_models_with_capability_vision() {
        let registry = build_registry();
        let models = registry.models_with_capability(&vision_cap());
        // Only "complex" supports vision
        assert_eq!(models.len(), 1);
        assert!(models.contains("complex"));
        assert!(!models.contains("fast"));
        assert!(!models.contains("general"));
    }

    #[test]
    fn test_models_with_capability_unknown_returns_empty() {
        let registry = build_registry();
        let models = registry.models_with_capability(&embed_cap());
        // No model has embeddings capability in the test registry
        assert!(models.is_empty());
    }

    #[test]
    fn test_model_has_capability_true() {
        let registry = build_registry();
        assert!(registry.model_has_capability("complex", &chat_cap()));
        assert!(registry.model_has_capability("complex", &vision_cap()));
        assert!(registry.model_has_capability("fast", &chat_cap()));
        assert!(registry.model_has_capability("general", &chat_cap()));
    }

    #[test]
    fn test_model_has_capability_false_for_undeclared() {
        let registry = build_registry();
        // "fast" does not have vision
        assert!(!registry.model_has_capability("fast", &vision_cap()));
        // "general" does not have vision
        assert!(!registry.model_has_capability("general", &vision_cap()));
        // "complex" does not have embeddings
        assert!(!registry.model_has_capability("complex", &embed_cap()));
    }

    #[test]
    fn test_model_has_capability_unknown_model_returns_false() {
        let registry = build_registry();
        assert!(!registry.model_has_capability("nonexistent", &chat_cap()));
    }

    #[test]
    fn test_model_capabilities_returns_correct_set() {
        let registry = build_registry();
        let caps = registry
            .model_capabilities("complex")
            .expect("complex should have capabilities");
        assert!(caps.contains(&chat_cap()));
        assert!(caps.contains(&vision_cap()));
        assert_eq!(caps.len(), 2);
    }

    #[test]
    fn test_model_capabilities_returns_none_for_unknown() {
        let registry = build_registry();
        assert!(registry.model_capabilities("nonexistent").is_none());
    }

    #[test]
    fn test_capability_index_construction_is_correct() {
        // Build a registry where we know exactly who has what,
        // then verify the reverse index is constructed correctly.
        let mut providers = HashMap::new();
        providers.insert("a".to_string(), stub("pa"));
        providers.insert("b".to_string(), stub("pb"));

        let mut capabilities: HashMap<String, HashSet<Capability>> = HashMap::new();
        capabilities.insert(
            "a".to_string(),
            [Capability::new("x"), Capability::new("y")]
                .into_iter()
                .collect(),
        );
        capabilities.insert(
            "b".to_string(),
            [Capability::new("y"), Capability::new("z")]
                .into_iter()
                .collect(),
        );

        let registry = ProviderRegistry::new(
            providers,
            HashMap::new(),
            HashMap::new(),
            capabilities,
            "a".to_string(),
        );

        let x_models = registry.models_with_capability(&Capability::new("x"));
        assert_eq!(x_models.len(), 1);
        assert!(x_models.contains("a"));

        let y_models = registry.models_with_capability(&Capability::new("y"));
        assert_eq!(y_models.len(), 2);
        assert!(y_models.contains("a"));
        assert!(y_models.contains("b"));

        let z_models = registry.models_with_capability(&Capability::new("z"));
        assert_eq!(z_models.len(), 1);
        assert!(z_models.contains("b"));
    }
}
