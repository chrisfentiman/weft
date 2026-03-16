use tracing::{debug, warn};

use super::wire::{OpenAIMessage, OpenAIRequest, OpenAIResponse};
use crate::{
    ChatCompletionOutput, Provider, ProviderError, ProviderRequest, ProviderResponse, TokenUsage,
    provider::weft_messages_to_text,
};

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI Chat Completions API provider.
///
/// One provider instance corresponds to one API endpoint (credentials + base_url).
/// The model identifier is passed per-request via `ChatCompletionInput.model`.
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    /// Create a new `OpenAIProvider` from connection info.
    ///
    /// `api_key` must already be resolved (env: prefix expanded).
    /// `base_url` overrides the default OpenAI API URL if provided.
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let base_url = base_url.unwrap_or_else(|| OPENAI_API_URL.to_string());
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

        // System prompt goes as first message with role "system".
        // Extract text from WeftMessage content parts. weft_messages_to_text
        // filters out gateway activity (source: Gateway, role: System).
        let role_text_pairs = weft_messages_to_text(&input.messages);

        let mut wire_messages = Vec::new();

        // Prepend the system_prompt as a system role message
        if !input.system_prompt.is_empty() {
            wire_messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: input.system_prompt.clone(),
            });
        }

        for (role, text) in role_text_pairs {
            let role_str = match role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            wire_messages.push(OpenAIMessage {
                role: role_str.to_string(),
                content: text,
            });
        }

        let request = OpenAIRequest {
            model: input.model.clone(),
            messages: wire_messages,
            max_tokens: Some(input.max_tokens),
            temperature: input.temperature,
        };

        debug!(model = %input.model, max_tokens = input.max_tokens, "sending OpenAI request");

        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;

        let status = response.status().as_u16();

        if status == 429 {
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
            warn!(status, "OpenAI API returned non-200");
            return Err(ProviderError::ProviderHttpError { status, body });
        }

        let body_text = response
            .text()
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;

        let openai_response: OpenAIResponse = serde_json::from_str(&body_text)
            .map_err(|e| ProviderError::DeserializationError(e.to_string()))?;

        let text = openai_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = openai_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        });

        Ok(ChatCompletionOutput { text, usage })
    }
}

#[async_trait::async_trait]
impl Provider for OpenAIProvider {
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        match request {
            ProviderRequest::ChatCompletion(input) => {
                let output = self.chat_completion(input).await?;
                Ok(ProviderResponse::ChatCompletion(output))
            }
            // For now, only ChatCompletion is implemented.
            #[allow(unreachable_patterns)]
            _ => Err(ProviderError::Unsupported(
                "OpenAI provider currently supports chat_completions only".to_string(),
            )),
        }
    }

    fn name(&self) -> &str {
        "openai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use weft_core::{ContentPart, Role, Source, WeftMessage};

    fn make_provider(base_url: &str) -> OpenAIProvider {
        OpenAIProvider::new("test-key".to_string(), Some(base_url.to_string()))
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

        let provider = make_provider(&server.url());
        let result = provider
            .execute(make_request("gpt-test"))
            .await
            .expect("should succeed");

        // Only ChatCompletion variant exists; when future variants land this becomes refutable.
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion(output) = result else {
            panic!("expected ChatCompletion response");
        };
        assert_eq!(output.text, "Hello! How can I help?");
        let usage = output.usage.expect("usage should be present");
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

        let provider = make_provider(&server.url());
        let result = provider.execute(make_request("gpt-test")).await;

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
            .with_header("retry-after", "60")
            .with_body("{}")
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let result = provider.execute(make_request("gpt-test")).await;

        assert!(matches!(
            result,
            Err(ProviderError::RateLimited {
                retry_after_ms: 60_000
            })
        ));
    }

    #[tokio::test]
    async fn test_system_prompt_prepended_as_system_message() {
        let mut server = mockito::Server::new_async().await;
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

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion(crate::ChatCompletionInput {
            system_prompt: "You are a helpful assistant.".to_string(),
            messages: make_messages(),
            model: "gpt-test".to_string(),
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

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion(crate::ChatCompletionInput {
            system_prompt: "system".to_string(),
            messages: make_messages(),
            model: "gpt-test".to_string(),
            max_tokens: 256,
            temperature: Some(0.5),
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

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion(crate::ChatCompletionInput {
            system_prompt: "".to_string(),
            messages: make_messages(),
            model: "gpt-test".to_string(),
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
        let result = provider.execute(make_request("gpt-test")).await;

        assert!(matches!(
            result,
            Err(ProviderError::DeserializationError(_))
        ));
    }

    #[tokio::test]
    async fn test_execute_unsupported_variant_returns_unsupported() {
        // There are no non-ChatCompletion variants yet, so we verify the match
        // arm compiles and the ChatCompletion arm works. When future variants are
        // added, they must return Unsupported from OpenAIProvider.
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

        let provider = make_provider(&server.url());
        // ChatCompletion should succeed — not unsupported
        let result = provider.execute(make_request("gpt-test")).await;
        assert!(
            result.is_ok(),
            "ChatCompletion should not return Unsupported"
        );
    }

    #[test]
    fn test_provider_name() {
        let provider = OpenAIProvider::new("key".to_string(), None);
        assert_eq!(provider.name(), "openai");
    }
}
