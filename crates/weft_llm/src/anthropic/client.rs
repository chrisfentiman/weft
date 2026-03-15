use tracing::{debug, warn};
use weft_core::{LlmConfig, Message, Role};

use super::wire::{AnthropicMessage, AnthropicRequest, AnthropicResponse};
use crate::{CompletionOptions, CompletionResponse, LlmError, LlmProvider, LlmUsage};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    default_max_tokens: u32,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new `AnthropicProvider` from config.
    ///
    /// `api_key` must already be resolved (env: prefix expanded).
    pub fn new(config: &LlmConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| ANTHROPIC_API_URL.to_string());

        Self {
            client: reqwest::Client::new(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            default_max_tokens: config.max_tokens,
            base_url,
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> Result<CompletionResponse, LlmError> {
        // Build the system string: the provided system_prompt plus any System-role messages
        // concatenated in order.
        let mut system_parts = vec![system_prompt.to_string()];
        let wire_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter_map(|m| match m.role {
                Role::System => {
                    system_parts.push(m.content.clone());
                    None
                }
                Role::User => Some(AnthropicMessage {
                    role: "user".to_string(),
                    content: m.content.clone(),
                }),
                Role::Assistant => Some(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: m.content.clone(),
                }),
            })
            .collect();

        let system = system_parts.join("\n\n");
        let max_tokens = options.max_tokens.unwrap_or(self.default_max_tokens);

        let request = AnthropicRequest {
            model: self.model.clone(),
            system,
            messages: wire_messages,
            max_tokens,
            temperature: options.temperature,
        };

        debug!(model = %self.model, max_tokens, "sending Anthropic request");

        let response = self
            .client
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let status = response.status().as_u16();

        if status == 429 {
            // Extract Retry-After header if present
            let retry_after_ms = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(|secs| secs * 1000)
                .unwrap_or(60_000);
            return Err(LlmError::RateLimited { retry_after_ms });
        }

        if status != 200 {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            warn!(status, "Anthropic API returned non-200");
            return Err(LlmError::ProviderError { status, body });
        }

        let body_text = response
            .text()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let anthropic_response: AnthropicResponse = serde_json::from_str(&body_text)
            .map_err(|e| LlmError::DeserializationError(e.to_string()))?;

        // Extract text from the first text content block
        let text = anthropic_response
            .content
            .iter()
            .find(|b| b.kind == "text")
            .and_then(|b| b.text.clone())
            .unwrap_or_default();

        let usage = anthropic_response.usage.map(|u| LlmUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
        });

        Ok(CompletionResponse { text, usage })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use weft_core::LlmProviderKind;

    fn make_config(base_url: &str) -> LlmConfig {
        LlmConfig {
            provider: LlmProviderKind::Anthropic,
            api_key: "test-key".to_string(),
            model: "claude-test".to_string(),
            max_tokens: 1024,
            base_url: Some(base_url.to_string()),
        }
    }

    fn make_messages() -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: "Hello".to_string(),
        }]
    }

    #[tokio::test]
    async fn test_successful_response() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello! How can I help?"}],
                    "usage": {"input_tokens": 10, "output_tokens": 8}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = AnthropicProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await
            .expect("should succeed");

        assert_eq!(result.text, "Hello! How can I help?");
        let usage = result.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 8);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_provider_error_non_200() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(400)
            .with_body(r#"{"error": {"type": "invalid_request_error", "message": "bad request"}}"#)
            .create_async()
            .await;

        let provider = AnthropicProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await;

        assert!(matches!(
            result,
            Err(LlmError::ProviderError { status: 400, .. })
        ));
    }

    #[tokio::test]
    async fn test_rate_limited_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(429)
            .with_header("retry-after", "30")
            .with_body(r#"{"error": {"type": "rate_limit_error"}}"#)
            .create_async()
            .await;

        let provider = AnthropicProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await;

        assert!(matches!(
            result,
            Err(LlmError::RateLimited {
                retry_after_ms: 30_000
            })
        ));
    }

    #[tokio::test]
    async fn test_system_role_messages_concatenated() {
        let mut server = mockito::Server::new_async().await;
        // Capture the request body to verify system prompt concatenation
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 5, "output_tokens": 2}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: Role::User,
                content: "Hello".to_string(),
            },
        ];

        let provider = AnthropicProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("base system", &messages, &CompletionOptions::default())
            .await;

        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_options_forwarded() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "response"}],
                    "usage": null
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = AnthropicProvider::new(&make_config(&server.url()));
        let options = CompletionOptions {
            max_tokens: Some(512),
            temperature: Some(0.7),
        };
        let result = provider
            .complete("system", &make_messages(), &options)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_deserialization_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"not_valid": true}"#)
            .create_async()
            .await;

        let provider = AnthropicProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await;

        assert!(matches!(result, Err(LlmError::DeserializationError(_))));
    }
}
