use tracing::{debug, warn};
use weft_core::Message;
use weft_core::Role;

use super::wire::{AnthropicMessage, AnthropicRequest, AnthropicResponse};
use crate::{CompletionOptions, CompletionResponse, LlmError, LlmProvider, LlmUsage};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
///
/// One provider instance corresponds to one API endpoint (credentials + base_url).
/// The model identifier is passed per-request via `CompletionOptions.model`.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new `AnthropicProvider` from connection info.
    ///
    /// `api_key` must already be resolved (env: prefix expanded).
    /// `base_url` overrides the default Anthropic API URL if provided.
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let base_url = base_url.unwrap_or_else(|| ANTHROPIC_API_URL.to_string());
        Self {
            client: reqwest::Client::new(),
            api_key,
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
        // Providers MUST return an error if model is None (defensive check).
        let model = options.model.as_deref().ok_or_else(|| {
            LlmError::RequestFailed(
                "CompletionOptions.model is None -- engine misconfiguration".to_string(),
            )
        })?;

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
        let max_tokens = options.max_tokens.unwrap_or(4096);

        let request = AnthropicRequest {
            model: model.to_string(),
            system,
            messages: wire_messages,
            max_tokens,
            temperature: options.temperature,
        };

        debug!(model = %model, max_tokens, "sending Anthropic request");

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

    fn make_provider(base_url: &str) -> AnthropicProvider {
        AnthropicProvider::new("test-key".to_string(), Some(base_url.to_string()))
    }

    fn make_messages() -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: "Hello".to_string(),
        }]
    }

    fn options_with_model(model: &str) -> CompletionOptions {
        CompletionOptions {
            max_tokens: None,
            temperature: None,
            model: Some(model.to_string()),
        }
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

        let provider = make_provider(&server.url());
        let result = provider
            .complete(
                "system",
                &make_messages(),
                &options_with_model("claude-test"),
            )
            .await
            .expect("should succeed");

        assert_eq!(result.text, "Hello! How can I help?");
        let usage = result.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 8);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_model_none_returns_error() {
        let mut server = mockito::Server::new_async().await;
        // No mock needed — should fail before hitting the server
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let options = CompletionOptions::default(); // model is None
        let result = provider
            .complete("system", &make_messages(), &options)
            .await;

        assert!(
            matches!(result, Err(LlmError::RequestFailed(_))),
            "None model should return RequestFailed"
        );
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

        let provider = make_provider(&server.url());
        let result = provider
            .complete(
                "system",
                &make_messages(),
                &options_with_model("claude-test"),
            )
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

        let provider = make_provider(&server.url());
        let result = provider
            .complete(
                "system",
                &make_messages(),
                &options_with_model("claude-test"),
            )
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
            // Partial JSON match: verify the `system` field contains both parts.
            .match_body(mockito::Matcher::PartialJsonString(
                json!({
                    "system": "base system\n\nYou are a helpful assistant."
                })
                .to_string(),
            ))
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

        let provider = make_provider(&server.url());
        let result = provider
            .complete("base system", &messages, &options_with_model("claude-test"))
            .await;

        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_options_forwarded() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
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
            // Verify max_tokens appears in the request body.
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::PartialJsonString(json!({"max_tokens": 512}).to_string()),
                // temperature field is present
                mockito::Matcher::Regex(r#""temperature"\s*:"#.to_string()),
            ]))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let options = CompletionOptions {
            max_tokens: Some(512),
            temperature: Some(0.7),
            model: Some("claude-test".to_string()),
        };
        let result = provider
            .complete("system", &make_messages(), &options)
            .await;
        assert!(result.is_ok());
        mock.assert_async().await;
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

        let provider = make_provider(&server.url());
        let result = provider
            .complete(
                "system",
                &make_messages(),
                &options_with_model("claude-test"),
            )
            .await;

        assert!(matches!(result, Err(LlmError::DeserializationError(_))));
    }
}
