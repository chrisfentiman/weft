//! Provider call logic for the gateway engine.
//!
//! Provides `call_with_fallback`, which calls the selected LLM provider and
//! falls back to the default provider on non-rate-limit errors.

use std::sync::Arc;

use tracing::warn;
use weft_commands::CommandRegistry;
use weft_core::{SamplingOptions, WeftError, WeftMessage};
use weft_hooks::HookRunner;
use weft_llm::{
    Capability, Provider, ProviderError, ProviderRequest, ProviderResponse, ProviderService,
    TokenUsage,
};
use weft_memory::MemoryService;
use weft_router::SemanticRouter;

use crate::GatewayEngine;

impl<H, R, M, P, C> GatewayEngine<H, R, M, P, C>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    /// Call the provider, falling back to the default provider on non-rate-limit failure.
    ///
    /// Fallback rules:
    /// - `RateLimited`: always propagate immediately, no fallback.
    /// - Any other error from a non-default model: retry with the default provider.
    /// - Any other error from the default model (or after fallback retry fails): propagate.
    pub(crate) async fn call_with_fallback(
        &self,
        provider: Arc<dyn Provider>,
        selected_model_name: &str,
        request: ProviderRequest,
        temperature: Option<f32>,
    ) -> Result<(WeftMessage, Option<TokenUsage>), WeftError> {
        // The required capability for this call path is always chat_completions.
        let required_capability = Capability::new(Capability::CHAT_COMPLETIONS);

        let result = provider.execute(request.clone()).await;
        #[allow(unreachable_patterns)]
        match result {
            Ok(ProviderResponse::ChatCompletion { message, usage }) => Ok((message, usage)),
            Ok(_) => Err(WeftError::Llm(
                "unexpected response type from provider".to_string(),
            )),
            Err(ProviderError::RateLimited { retry_after_ms }) => {
                // Rate limit: propagate immediately, no fallback.
                Err(WeftError::RateLimited { retry_after_ms })
            }
            Err(e) if selected_model_name != self.providers.default_name() => {
                // Non-default model failed with a non-rate-limit error: try the default,
                // but only if the default model supports the required capability.
                let default_name = self.providers.default_name();
                if !self
                    .providers
                    .model_has_capability(default_name, &required_capability)
                {
                    warn!(
                        model = selected_model_name,
                        default_model = default_name,
                        capability = %required_capability,
                        error = %e,
                        "fallback to default model skipped: default lacks required capability"
                    );
                    return Err(WeftError::Llm(e.to_string()));
                }
                warn!(
                    model = selected_model_name,
                    error = %e,
                    "model failed, falling back to default"
                );
                let default_provider = self.providers.default_provider();

                // Build a new request with the default model's identifiers.
                let fallback_request = match request {
                    ProviderRequest::ChatCompletion {
                        messages,
                        model: _,
                        options,
                    } => {
                        let fallback_model_id = self
                            .providers
                            .model_id(default_name)
                            .map(String::from)
                            .unwrap_or_default();
                        let fallback_max_tokens =
                            self.providers.max_tokens_for(default_name).unwrap_or(4096);
                        ProviderRequest::ChatCompletion {
                            messages,
                            model: fallback_model_id,
                            options: SamplingOptions {
                                max_tokens: Some(fallback_max_tokens),
                                temperature,
                                ..options
                            },
                        }
                    }
                    #[allow(unreachable_patterns)]
                    other => other,
                };

                #[allow(unreachable_patterns)]
                match default_provider.execute(fallback_request).await {
                    Ok(ProviderResponse::ChatCompletion { message, usage }) => Ok((message, usage)),
                    Ok(_) => Err(WeftError::Llm(
                        "unexpected response type from default provider".to_string(),
                    )),
                    Err(e) => Err(WeftError::Llm(e.to_string())),
                }
            }
            Err(e) => {
                // Default model failed (or selected was already default): propagate.
                Err(WeftError::Llm(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use std::collections::{HashMap, HashSet};
    use weft_core::{
        ClassifierConfig, DomainsConfig, GatewayConfig, ModelEntry, ProviderConfig, RouterConfig,
        ServerConfig, WeftConfig, WeftError, WireFormat,
    };
    use weft_llm::{Capability, ProviderRegistry};

    #[tokio::test]
    async fn test_no_eligible_models_returns_400() {
        // A registry with only an embeddings-capable model — no chat_completions.
        // The request should be rejected with NoEligibleModels.
        let registry = registry_with_capabilities(
            "embed-model",
            vec![("embed-model", "text-embed-v1", vec!["embeddings"])],
        );

        let engine = make_engine(
            registry,
            MockRouter::with_model("embed-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::NoEligibleModels { ref capability }) if capability == "chat_completions"),
            "expected NoEligibleModels error, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_capability_filter_excludes_non_chat_models_from_routing() {
        // Two models: one with chat_completions, one with only embeddings.
        // The embeddings-only model should be excluded from routing candidates.
        // We verify this by checking that only the chat model is ever selected.

        // Use a multi-model config so the model domain is included in routing.
        let config = Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("chat-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "chat-model".to_string(),
                            model: "claude-chat".to_string(),
                            max_tokens: 1024,
                            examples: vec!["general chat".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                        ModelEntry {
                            name: "embed-model".to_string(),
                            model: "text-embed-v1".to_string(),
                            max_tokens: 512,
                            examples: vec!["embed this text".to_string()],
                            capabilities: vec!["embeddings".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            event_log: None,
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        // Build registry: chat-model has chat_completions, embed-model only has embeddings.
        let registry = registry_with_capabilities(
            "chat-model",
            vec![
                ("chat-model", "claude-chat", vec!["chat_completions"]),
                ("embed-model", "text-embed-v1", vec!["embeddings"]),
            ],
        );

        // Router always returns the embed-model — the capability filter should override.
        // Since embed-model is filtered out, the router sees only [chat-model].
        // We use a router that returns "embed-model" to confirm it gets filtered.
        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_model("embed-model"),
            MockCommandRegistry::new(vec![]),
        );

        // Should succeed — embed-model is filtered from candidates, chat-model is used.
        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            result.is_ok(),
            "expected success with capability filtering, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_fallback_skips_default_when_default_lacks_capability() {
        // The non-default model fails with a non-rate-limit error.
        // The default model only has "embeddings" — not chat_completions.
        // Fallback should be skipped, error should propagate as Llm error.
        //
        // We use a two-model config to ensure the model domain is included in routing,
        // so the router can select the non-default model which will then fail.
        struct FailingProvider;

        #[async_trait::async_trait]
        impl Provider for FailingProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                Err(ProviderError::RequestFailed("network down".to_string()))
            }

            fn name(&self) -> &str {
                "failing"
            }
        }

        // default-model has only embeddings; non-default has chat_completions but fails.
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(MockLlmProvider::single("fallback response")) as Arc<dyn Provider>,
        );
        providers.insert(
            "non-default-model".to_string(),
            Arc::new(FailingProvider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "embed-v1".to_string());
        model_ids.insert(
            "non-default-model".to_string(),
            "claude-complex".to_string(),
        );
        let mut max_tokens = HashMap::new();
        max_tokens.insert("default-model".to_string(), 512u32);
        max_tokens.insert("non-default-model".to_string(), 4096u32);

        // Default model has embeddings only (no chat_completions).
        // Non-default model has chat_completions.
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "default-model".to_string(),
            [Capability::new("embeddings")].into_iter().collect(),
        );
        caps.insert(
            "non-default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "default-model".to_string(),
        ));

        // Use a two-model config with the same model names as the registry.
        // "non-default-model" is eligible (has chat_completions), so it appears in routing
        // candidates and the router selects it. It then fails, and the fallback check
        // finds that "default-model" lacks chat_completions, so fallback is skipped.
        let config = Arc::new(WeftConfig {
            server: weft_core::ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("default-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "default-model".to_string(),
                            model: "embed-v1".to_string(),
                            max_tokens: 512,
                            examples: vec!["embed text".to_string()],
                            // embeddings only — no chat_completions
                            capabilities: vec!["embeddings".to_string()],
                        },
                        ModelEntry {
                            name: "non-default-model".to_string(),
                            model: "claude-complex".to_string(),
                            max_tokens: 4096,
                            examples: vec!["complex reasoning".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            event_log: None,
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        // Router selects non-default model (which will fail).
        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_model("non-default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "expected Llm error when fallback is skipped due to missing capability, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_fallback_proceeds_when_default_has_capability() {
        // Non-default fails, default has chat_completions — fallback should proceed.
        // Uses a two-model config so the model domain is included in routing and
        // the router can select the non-default model which then fails.
        struct FailingProvider;

        #[async_trait::async_trait]
        impl Provider for FailingProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                Err(ProviderError::RequestFailed("network down".to_string()))
            }

            fn name(&self) -> &str {
                "failing"
            }
        }

        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(MockLlmProvider::single("fallback response")) as Arc<dyn Provider>,
        );
        providers.insert(
            "non-default-model".to_string(),
            Arc::new(FailingProvider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert(
            "non-default-model".to_string(),
            "claude-complex".to_string(),
        );
        let mut max_tokens = HashMap::new();
        max_tokens.insert("default-model".to_string(), 1024u32);
        max_tokens.insert("non-default-model".to_string(), 4096u32);

        // Both models have chat_completions.
        let mut caps: HashMap<String, HashSet<Capability>> = HashMap::new();
        caps.insert(
            "default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );
        caps.insert(
            "non-default-model".to_string(),
            [Capability::new(Capability::CHAT_COMPLETIONS)]
                .into_iter()
                .collect(),
        );

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            caps,
            "default-model".to_string(),
        ));

        // Two-model config matching registry model names so model domain is included.
        let config = Arc::new(WeftConfig {
            server: weft_core::ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("default-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    wire_format: WireFormat::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    wire_script: None,
                    models: vec![
                        ModelEntry {
                            name: "default-model".to_string(),
                            model: "claude-default".to_string(),
                            max_tokens: 1024,
                            examples: vec!["general question".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                        ModelEntry {
                            name: "non-default-model".to_string(),
                            model: "claude-complex".to_string(),
                            max_tokens: 4096,
                            examples: vec!["complex task".to_string()],
                            capabilities: vec!["chat_completions".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            event_log: None,
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_model("non-default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("fallback should succeed");
        assert_eq!(
            resp_text(&result),
            "fallback response",
            "response should come from fallback default model"
        );
    }
}
