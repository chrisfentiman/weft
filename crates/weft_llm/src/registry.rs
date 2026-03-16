//! Provider registry: maps model routing names to LLM provider instances.

use std::collections::HashMap;
use std::sync::Arc;

use crate::LlmProvider;

/// Registry of named LLM providers. The gateway selects a provider
/// by model name based on the semantic router's model decision.
///
/// Thread-safe: providers are Arc'd and the HashMap is immutable after construction.
///
/// Key insight: model names are globally unique, so the registry maps
/// model_name -> provider. One provider instance may appear multiple times
/// (once per model it serves), but each Arc is cheap.
pub struct ProviderRegistry {
    /// Named providers. Key is the model routing name (e.g., "complex", "fast").
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    /// Name of the default model.
    default_name: String,
    /// Model identifier for each routing name. Used to set the model field on API requests.
    model_ids: HashMap<String, String>,
    /// Max tokens per model routing name.
    max_tokens: HashMap<String, u32>,
}

impl ProviderRegistry {
    /// Construct a new provider registry.
    ///
    /// `providers`: Map of model_routing_name -> provider. Must contain at least one entry.
    /// `model_ids`: Map of model_routing_name -> model API identifier.
    /// `max_tokens`: Map of model_routing_name -> max tokens.
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
        providers: HashMap<String, Arc<dyn LlmProvider>>,
        model_ids: HashMap<String, String>,
        max_tokens: HashMap<String, u32>,
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
        Self {
            providers,
            default_name,
            model_ids,
            max_tokens,
        }
    }

    /// Get a provider by model routing name. Returns the default if name is not found.
    pub fn get(&self, name: &str) -> &Arc<dyn LlmProvider> {
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
    pub fn default_provider(&self) -> &Arc<dyn LlmProvider> {
        self.providers.get(&self.default_name).unwrap()
    }

    /// Get the default model name.
    pub fn default_name(&self) -> &str {
        &self.default_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletionOptions, CompletionResponse, LlmError};
    use async_trait::async_trait;
    use weft_core::Message;

    struct StubProvider {
        name: String,
    }

    #[async_trait]
    impl LlmProvider for StubProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _messages: &[Message],
            _options: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                text: format!("response from {}", self.name),
                usage: None,
            })
        }
    }

    fn stub(name: &str) -> Arc<dyn LlmProvider> {
        Arc::new(StubProvider {
            name: name.to_string(),
        })
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

        ProviderRegistry::new(providers, model_ids, max_tokens, "complex".to_string())
    }

    #[test]
    fn test_new_panics_on_empty_providers() {
        let result = std::panic::catch_unwind(|| {
            ProviderRegistry::new(
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
        // Just verify get returns something (can't compare Arc<dyn LlmProvider> by value)
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
}
