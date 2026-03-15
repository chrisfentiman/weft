use tracing::{debug, warn};
use weft_core::{LlmConfig, Message, Role};

use super::wire::{OpenAIMessage, OpenAIRequest, OpenAIResponse};
use crate::{CompletionOptions, CompletionResponse, LlmError, LlmProvider, LlmUsage};

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI Chat Completions API provider.
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    default_max_tokens: u32,
    base_url: String,
}

impl OpenAIProvider {
    /// Create a new `OpenAIProvider` from config.
    ///
    /// `api_key` must already be resolved (env: prefix expanded).
    pub fn new(config: &LlmConfig) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| OPENAI_API_URL.to_string());

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
impl LlmProvider for OpenAIProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> Result<CompletionResponse, LlmError> {
        // System prompt goes as first message with role "system".
        // Any Role::System messages in the messages slice are also mapped to role "system".
        let mut wire_messages = Vec::new();

        // Prepend the system_prompt as a system role message
        if !system_prompt.is_empty() {
            wire_messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            });
        }

        for m in messages {
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            wire_messages.push(OpenAIMessage {
                role: role.to_string(),
                content: m.content.clone(),
            });
        }

        let max_tokens = options.max_tokens.unwrap_or(self.default_max_tokens);

        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: wire_messages,
            max_tokens: Some(max_tokens),
            temperature: options.temperature,
        };

        debug!(model = %self.model, max_tokens, "sending OpenAI request");

        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let status = response.status().as_u16();

        if status == 429 {
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
            warn!(status, "OpenAI API returned non-200");
            return Err(LlmError::ProviderError { status, body });
        }

        let body_text = response
            .text()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let openai_response: OpenAIResponse = serde_json::from_str(&body_text)
            .map_err(|e| LlmError::DeserializationError(e.to_string()))?;

        let text = openai_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = openai_response.usage.map(|u| LlmUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
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
            provider: LlmProviderKind::OpenAI,
            api_key: "test-key".to_string(),
            model: "gpt-test".to_string(),
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
                    "id": "chatcmpl-01",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello! How can I help?"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 9}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await
            .expect("should succeed");

        assert_eq!(result.text, "Hello! How can I help?");
        let usage = result.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 9);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_provider_error_non_200() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(400)
            .with_body(r#"{"error": {"message": "bad request", "type": "invalid_request_error"}}"#)
            .create_async()
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.url()));
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
            .with_header("retry-after", "60")
            .with_body("{}")
            .create_async()
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await;

        assert!(matches!(
            result,
            Err(LlmError::RateLimited {
                retry_after_ms: 60_000
            })
        ));
    }

    #[tokio::test]
    async fn test_system_prompt_prepended_as_system_message() {
        let mut server = mockito::Server::new_async().await;
        // Verify that the system prompt appears as the first message with role "system"
        // in the serialized request body.
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": null
                })
                .to_string(),
            )
            .match_body(mockito::Matcher::PartialJsonString(
                json!({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ]
                })
                .to_string(),
            ))
            .create_async()
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.url()));
        let result = provider
            .complete(
                "You are a helpful assistant.",
                &make_messages(),
                &CompletionOptions::default(),
            )
            .await;

        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_options_forwarded() {
        let mut server = mockito::Server::new_async().await;
        // Verify that max_tokens and temperature from CompletionOptions appear in the
        // serialized request body sent to OpenAI.
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": null
                })
                .to_string(),
            )
            .match_body(mockito::Matcher::PartialJsonString(
                json!({
                    "max_tokens": 256,
                    "temperature": 0.5_f32
                })
                .to_string(),
            ))
            .create_async()
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.url()));
        let options = CompletionOptions {
            max_tokens: Some(256),
            temperature: Some(0.5),
        };
        let result = provider
            .complete("system", &make_messages(), &options)
            .await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_empty_system_prompt_not_added() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": null
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.url()));
        // Empty system prompt should not add a system message
        let result = provider
            .complete("", &make_messages(), &CompletionOptions::default())
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

        let provider = OpenAIProvider::new(&make_config(&server.url()));
        let result = provider
            .complete("system", &make_messages(), &CompletionOptions::default())
            .await;

        assert!(matches!(result, Err(LlmError::DeserializationError(_))));
    }
}
