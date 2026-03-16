use tracing::{debug, warn};

use super::wire::{AnthropicMessage, AnthropicRequest, AnthropicResponse};
use crate::{
    ChatCompletionOutput, Provider, ProviderError, ProviderRequest, ProviderResponse, TokenUsage,
    provider::weft_messages_to_text,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
///
/// One provider instance corresponds to one API endpoint (credentials + base_url).
/// The model identifier is passed per-request via `ChatCompletionInput.model`.
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

    async fn chat_completion(
        &self,
        input: crate::ChatCompletionInput,
    ) -> Result<ChatCompletionOutput, ProviderError> {
        use weft_core::Role;

        // Extract text from WeftMessage content parts.
        // weft_messages_to_text filters out gateway activity (source: Gateway, role: System)
        // and extracts only Text content parts, skipping non-text content.
        let role_text_pairs = weft_messages_to_text(&input.messages);

        // Build the system string: the provided system_prompt plus any client-provided
        // System-role messages concatenated in order.
        let mut system_parts = vec![input.system_prompt.clone()];
        let wire_messages: Vec<AnthropicMessage> = role_text_pairs
            .into_iter()
            .filter_map(|(role, text)| match role {
                Role::System => {
                    system_parts.push(text);
                    None
                }
                Role::User => Some(AnthropicMessage {
                    role: "user".to_string(),
                    content: text,
                }),
                Role::Assistant => Some(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: text,
                }),
            })
            .collect();

        let system = system_parts.join("\n\n");

        let request = AnthropicRequest {
            model: input.model.clone(),
            system,
            messages: wire_messages,
            max_tokens: input.max_tokens,
            temperature: input.temperature,
        };

        debug!(model = %input.model, max_tokens = input.max_tokens, "sending Anthropic request");

        let response = self
            .client
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;

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
            return Err(ProviderError::RateLimited { retry_after_ms });
        }

        if status != 200 {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            warn!(status, "Anthropic API returned non-200");
            return Err(ProviderError::ProviderHttpError { status, body });
        }

        let body_text = response
            .text()
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;

        let anthropic_response: AnthropicResponse = serde_json::from_str(&body_text)
            .map_err(|e| ProviderError::DeserializationError(e.to_string()))?;

        // Extract text from the first text content block
        let text = anthropic_response
            .content
            .iter()
            .find(|b| b.kind == "text")
            .and_then(|b| b.text.clone())
            .unwrap_or_default();

        let usage = anthropic_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
        });

        Ok(ChatCompletionOutput { text, usage })
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        match request {
            ProviderRequest::ChatCompletion(input) => {
                let output = self.chat_completion(input).await?;
                Ok(ProviderResponse::ChatCompletion(output))
            }
            // For now, only ChatCompletion is implemented.
            #[allow(unreachable_patterns)]
            _ => Err(ProviderError::Unsupported(
                "Anthropic provider currently supports chat_completions only".to_string(),
            )),
        }
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    fn make_provider(base_url: &str) -> AnthropicProvider {
        AnthropicProvider::new("test-key".to_string(), Some(base_url.to_string()))
    }

    fn make_messages() -> Vec<WeftMessage> {
        vec![WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("Hello".to_string())],
            delta: false,
            message_index: 0,
        }]
    }

    fn make_request(model: &str) -> ProviderRequest {
        ProviderRequest::ChatCompletion(crate::ChatCompletionInput {
            system_prompt: "system".to_string(),
            messages: make_messages(),
            model: model.to_string(),
            max_tokens: 4096,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: vec![],
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            response_format: None,
        })
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
            .execute(make_request("claude-test"))
            .await
            .expect("should succeed");

        // Only ChatCompletion variant exists; when future variants land this becomes refutable.
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion(output) = result else {
            panic!("expected ChatCompletion response");
        };
        assert_eq!(output.text, "Hello! How can I help?");
        let usage = output.usage.expect("usage should be present");
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

        let provider = make_provider(&server.url());
        let result = provider.execute(make_request("claude-test")).await;

        assert!(matches!(
            result,
            Err(ProviderError::ProviderHttpError { status: 400, .. })
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
        let result = provider.execute(make_request("claude-test")).await;

        assert!(matches!(
            result,
            Err(ProviderError::RateLimited {
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

        // System messages with source Client are included and concatenated.
        let messages = vec![
            WeftMessage {
                role: Role::System,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text(
                    "You are a helpful assistant.".to_string(),
                )],
                delta: false,
                message_index: 0,
            },
            WeftMessage {
                role: Role::User,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("Hello".to_string())],
                delta: false,
                message_index: 0,
            },
        ];

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion(crate::ChatCompletionInput {
            system_prompt: "base system".to_string(),
            messages,
            model: "claude-test".to_string(),
            max_tokens: 4096,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: vec![],
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            response_format: None,
        });
        let result = provider.execute(request).await;

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
        let request = ProviderRequest::ChatCompletion(crate::ChatCompletionInput {
            system_prompt: "system".to_string(),
            messages: make_messages(),
            model: "claude-test".to_string(),
            max_tokens: 512,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop: vec![],
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            response_format: None,
        });
        let result = provider.execute(request).await;
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
        let result = provider.execute(make_request("claude-test")).await;

        assert!(matches!(
            result,
            Err(ProviderError::DeserializationError(_))
        ));
    }

    #[tokio::test]
    async fn test_execute_unsupported_variant_returns_unsupported() {
        // ChatCompletion should succeed — not unsupported.
        // When future variants are added, they must return Unsupported from AnthropicProvider.
        let mut server = mockito::Server::new_async().await;
        let _mock = server
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

        let provider = make_provider(&server.url());
        let result = provider.execute(make_request("claude-test")).await;
        assert!(
            result.is_ok(),
            "ChatCompletion should not return Unsupported"
        );
    }

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::new("key".to_string(), None);
        assert_eq!(provider.name(), "anthropic");
    }
}
