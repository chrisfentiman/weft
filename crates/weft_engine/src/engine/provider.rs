//! Provider call logic for the gateway engine.
//!
//! Provides `call_with_fallback`, which calls the selected LLM provider and
//! falls back to the default provider on non-rate-limit errors.

use std::sync::Arc;

use tracing::warn;
use weft_core::{SamplingOptions, WeftError, WeftMessage};
use weft_llm::{Capability, Provider, ProviderError, ProviderRequest, ProviderResponse, ProviderService, TokenUsage};

use super::GatewayEngine;
use weft_memory::MemoryService;
use weft_hooks::HookRunner;
use weft_router::SemanticRouter;
use weft_commands::CommandRegistry;

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
    pub(super) async fn call_with_fallback(
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
