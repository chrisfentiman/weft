//! `ProviderService` trait: abstracts `ProviderRegistry` for the engine.

use std::collections::HashSet;
use std::sync::Arc;

use crate::Provider;
use crate::provider::Capability;
use crate::registry::ProviderRegistry;

/// Provider service trait. Abstracts provider registry for the engine.
///
/// The engine uses this to look up providers, model identifiers, token limits,
/// and capabilities. `ProviderRegistry` is the production implementation.
///
/// Send + Sync + 'static: shared via Arc across async request handlers.
pub trait ProviderService: Send + Sync + 'static {
    /// Get a provider by model routing name. Returns the default if name is not found.
    fn get(&self, name: &str) -> &Arc<dyn Provider>;

    /// Get the model API identifier for a routing name.
    fn model_id(&self, name: &str) -> Option<&str>;

    /// Get the max tokens for a routing name.
    fn max_tokens_for(&self, name: &str) -> Option<u32>;

    /// Get the default provider.
    fn default_provider(&self) -> &Arc<dyn Provider>;

    /// Get the default model name.
    fn default_name(&self) -> &str;

    /// Get all model routing names that support a given capability.
    fn models_with_capability(&self, capability: &Capability) -> &HashSet<String>;

    /// Check if a specific model supports a given capability.
    fn model_has_capability(&self, model_name: &str, capability: &Capability) -> bool;

    /// Get the capabilities of a specific model.
    fn model_capabilities(&self, model_name: &str) -> Option<&HashSet<Capability>>;
}

impl ProviderService for ProviderRegistry {
    fn get(&self, name: &str) -> &Arc<dyn Provider> {
        self.get(name)
    }

    fn model_id(&self, name: &str) -> Option<&str> {
        self.model_id(name)
    }

    fn max_tokens_for(&self, name: &str) -> Option<u32> {
        self.max_tokens_for(name)
    }

    fn default_provider(&self) -> &Arc<dyn Provider> {
        self.default_provider()
    }

    fn default_name(&self) -> &str {
        self.default_name()
    }

    fn models_with_capability(&self, capability: &Capability) -> &HashSet<String> {
        self.models_with_capability(capability)
    }

    fn model_has_capability(&self, model_name: &str, capability: &Capability) -> bool {
        self.model_has_capability(model_name, capability)
    }

    fn model_capabilities(&self, model_name: &str) -> Option<&HashSet<Capability>> {
        self.model_capabilities(model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProviderError, ProviderRequest, ProviderResponse};
    use async_trait::async_trait;
    use std::collections::HashMap;

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

    fn build_registry() -> ProviderRegistry {
        let mut providers = HashMap::new();
        providers.insert("default".to_string(), stub("default-provider"));
        providers.insert("fast".to_string(), stub("fast-provider"));

        let mut model_ids = HashMap::new();
        model_ids.insert("default".to_string(), "model-default-v1".to_string());
        model_ids.insert("fast".to_string(), "model-fast-v1".to_string());

        let mut max_tokens = HashMap::new();
        max_tokens.insert("default".to_string(), 8192u32);
        max_tokens.insert("fast".to_string(), 2048u32);

        let mut capabilities: HashMap<String, HashSet<Capability>> = HashMap::new();
        capabilities.insert("default".to_string(), [chat_cap()].into_iter().collect());
        capabilities.insert("fast".to_string(), [chat_cap()].into_iter().collect());

        ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            capabilities,
            "default".to_string(),
        )
    }

    #[test]
    fn test_provider_service_get_delegates() {
        let registry = build_registry();
        // Should not panic; delegates to ProviderRegistry::get
        let _ = <ProviderRegistry as ProviderService>::get(&registry, "fast");
        let _ = <ProviderRegistry as ProviderService>::get(&registry, "unknown");
    }

    #[test]
    fn test_provider_service_model_id_delegates() {
        let registry = build_registry();
        assert_eq!(
            <ProviderRegistry as ProviderService>::model_id(&registry, "default"),
            Some("model-default-v1")
        );
        assert_eq!(
            <ProviderRegistry as ProviderService>::model_id(&registry, "unknown"),
            None
        );
    }

    #[test]
    fn test_provider_service_max_tokens_delegates() {
        let registry = build_registry();
        assert_eq!(
            <ProviderRegistry as ProviderService>::max_tokens_for(&registry, "default"),
            Some(8192)
        );
        assert_eq!(
            <ProviderRegistry as ProviderService>::max_tokens_for(&registry, "fast"),
            Some(2048)
        );
        assert_eq!(
            <ProviderRegistry as ProviderService>::max_tokens_for(&registry, "missing"),
            None
        );
    }

    #[test]
    fn test_provider_service_default_name_delegates() {
        let registry = build_registry();
        assert_eq!(
            <ProviderRegistry as ProviderService>::default_name(&registry),
            "default"
        );
    }

    #[test]
    fn test_provider_service_default_provider_delegates() {
        let registry = build_registry();
        // Should not panic; delegates to ProviderRegistry::default_provider
        let _ = <ProviderRegistry as ProviderService>::default_provider(&registry);
    }

    #[test]
    fn test_provider_service_models_with_capability_delegates() {
        let registry = build_registry();
        let models =
            <ProviderRegistry as ProviderService>::models_with_capability(&registry, &chat_cap());
        assert_eq!(models.len(), 2);
        assert!(models.contains("default"));
        assert!(models.contains("fast"));
    }

    #[test]
    fn test_provider_service_model_has_capability_delegates() {
        let registry = build_registry();
        assert!(<ProviderRegistry as ProviderService>::model_has_capability(
            &registry,
            "default",
            &chat_cap()
        ));
        assert!(
            !<ProviderRegistry as ProviderService>::model_has_capability(
                &registry,
                "nonexistent",
                &chat_cap()
            )
        );
    }

    #[test]
    fn test_provider_service_model_capabilities_delegates() {
        let registry = build_registry();
        let caps = <ProviderRegistry as ProviderService>::model_capabilities(&registry, "default")
            .expect("default should have capabilities");
        assert!(caps.contains(&chat_cap()));

        assert!(
            <ProviderRegistry as ProviderService>::model_capabilities(&registry, "nonexistent")
                .is_none()
        );
    }

    #[test]
    fn test_provider_service_via_dyn_trait() {
        // Verify that ProviderRegistry can be used as &dyn ProviderService — the
        // object-safety requirement is satisfied by not returning Self or using
        // generic type parameters in method signatures.
        let registry = build_registry();
        let svc: &dyn ProviderService = &registry;
        assert_eq!(svc.default_name(), "default");
        assert_eq!(svc.model_id("default"), Some("model-default-v1"));
        assert_eq!(svc.max_tokens_for("fast"), Some(2048));
        let models = svc.models_with_capability(&chat_cap());
        assert_eq!(models.len(), 2);
    }
}
