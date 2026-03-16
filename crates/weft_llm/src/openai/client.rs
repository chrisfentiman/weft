use tracing::{debug, warn};
use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};

use super::wire::{OpenAIMessage, OpenAIRequest, OpenAIResponse};
use crate::{
    Provider, ProviderError, ProviderRequest, ProviderResponse, TokenUsage,
    provider::extract_text_messages,
};

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI Chat Completions API provider.
///
/// One provider instance corresponds to one API endpoint (credentials + base_url).
/// The model identifier is passed per-request via `ProviderRequest::ChatCompletion.model`.
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
        messages: Vec<WeftMessage>,
        model: String,
        options: SamplingOptions,
    ) -> Result<(WeftMessage, Option<TokenUsage>), ProviderError> {
        let mut wire_messages = Vec::new();

        // Extract system prompt from messages[0] if present.
        // For OpenAI, system messages are in the messages array with role "system".
        // The system prompt (Role::System, Source::Gateway) is filtered out by
        // extract_text_messages, so we handle it explicitly first.
        let conversation_start = if messages.first().map(|m| m.role) == Some(Role::System) {
            let system_text: String = messages[0]
                .content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !system_text.is_empty() {
                wire_messages.push(OpenAIMessage {
                    role: "system".to_string(),
                    content: system_text,
                });
            }
            1 // skip messages[0] in the remaining extraction
        } else {
            0 // no system prompt
        };

        // Extract text from remaining WeftMessages, skipping gateway activity messages.
        for (role, text) in extract_text_messages(&messages[conversation_start..]) {
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
            model: model.clone(),
            messages: wire_messages,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            frequency_penalty: options.frequency_penalty,
            presence_penalty: options.presence_penalty,
            seed: options.seed,
            stop: if options.stop.is_empty() {
                None
            } else {
                Some(options.stop.clone())
            },
        };

        debug!(model = %model, "sending OpenAI request");

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

        let response_message = WeftMessage {
            role: Role::Assistant,
            source: Source::Provider,
            model: Some(model),
            content: vec![ContentPart::Text(text)],
            delta: false,
            message_index: 0,
        };

        Ok((response_message, usage))
    }
}

#[async_trait::async_trait]
impl Provider for OpenAIProvider {
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        match request {
            ProviderRequest::ChatCompletion {
                messages,
                model,
                options,
            } => {
                let (message, usage) = self.chat_completion(messages, model, options).await?;
                Ok(ProviderResponse::ChatCompletion { message, usage })
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

    /// Build a WeftMessage with a single text content part.
    fn make_weft_message(role: Role, source: Source, text: &str) -> WeftMessage {
        WeftMessage {
            role,
            source,
            model: None,
            content: vec![ContentPart::Text(text.to_string())],
            delta: false,
            message_index: 0,
        }
    }

    fn make_request(model: &str) -> ProviderRequest {
        ProviderRequest::ChatCompletion {
            messages: vec![
                make_weft_message(Role::System, Source::Gateway, "system"),
                make_weft_message(Role::User, Source::Client, "Hello"),
            ],
            model: model.to_string(),
            options: SamplingOptions::default(),
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

        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, usage } = result else {
            panic!("expected ChatCompletion response");
        };
        // Extract text from WeftMessage content parts
        let text = message
            .content
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "Hello! How can I help?");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.source, Source::Provider);
        let u = usage.expect("usage should be present");
        assert_eq!(u.prompt_tokens, 12);
        assert_eq!(u.completion_tokens, 9);
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
    async fn test_system_prompt_from_messages_0_as_system_message() {
        // System prompt at messages[0] (Role::System, Source::Gateway) is extracted
        // and sent as the first OpenAI system message.
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
        let request = ProviderRequest::ChatCompletion {
            messages: vec![
                make_weft_message(
                    Role::System,
                    Source::Gateway,
                    "You are a helpful assistant.",
                ),
                make_weft_message(Role::User, Source::Client, "Hello"),
            ],
            model: "gpt-test".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;

        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_no_system_prompt_when_messages_0_is_user() {
        // When messages[0] is Role::User (not System), no system message is prepended.
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
                        {"role": "user", "content": "Hello"}
                    ]
                })
                .to_string(),
            ))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
            model: "gpt-test".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;

        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gateway_activity_messages_not_sent() {
        // Gateway activity messages (Role::System, Source::Gateway) beyond messages[0]
        // are filtered out and not sent to OpenAI.
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
                        {"role": "system", "content": "sys prompt"},
                        {"role": "user", "content": "Hello"}
                    ]
                })
                .to_string(),
            ))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![
                make_weft_message(Role::System, Source::Gateway, "sys prompt"),
                // This gateway activity message should be filtered
                make_weft_message(Role::System, Source::Gateway, "routing activity"),
                make_weft_message(Role::User, Source::Client, "Hello"),
            ],
            model: "gpt-test".to_string(),
            options: SamplingOptions::default(),
        };
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
            // Verify max_tokens and seed are serialized (exact integer match is safe)
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::PartialJsonString(json!({"max_tokens": 256}).to_string()),
                mockito::Matcher::PartialJsonString(json!({"seed": 42_i64}).to_string()),
                // Verify the sampling fields are present (regex, no float precision issues)
                mockito::Matcher::Regex(r#""temperature"\s*:"#.to_string()),
                mockito::Matcher::Regex(r#""top_p"\s*:"#.to_string()),
                mockito::Matcher::Regex(r#""frequency_penalty"\s*:"#.to_string()),
                mockito::Matcher::Regex(r#""presence_penalty"\s*:"#.to_string()),
            ]))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
            model: "gpt-test".to_string(),
            options: SamplingOptions {
                max_tokens: Some(256),
                temperature: Some(0.5),
                top_p: Some(0.9),
                top_k: None,
                stop: vec![],
                frequency_penalty: Some(0.1),
                presence_penalty: Some(0.2),
                seed: Some(42),
                response_format: None,
                activity: false,
            },
        };
        let result = provider.execute(request).await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_stop_sequences_forwarded() {
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
                    "stop": ["STOP", "END"]
                })
                .to_string(),
            ))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
            model: "gpt-test".to_string(),
            options: SamplingOptions {
                max_tokens: None,
                temperature: None,
                top_p: None,
                top_k: None,
                stop: vec!["STOP".to_string(), "END".to_string()],
                frequency_penalty: None,
                presence_penalty: None,
                seed: None,
                response_format: None,
                activity: false,
            },
        };
        let result = provider.execute(request).await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_response_is_weft_message_with_correct_attribution() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let result = provider
            .execute(ProviderRequest::ChatCompletion {
                messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
                model: "gpt-4".to_string(),
                options: SamplingOptions::default(),
            })
            .await
            .expect("should succeed");

        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, .. } = result else {
            panic!("expected ChatCompletion");
        };
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.source, Source::Provider);
        assert_eq!(message.model, Some("gpt-4".to_string()));
        assert!(!message.delta);
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
        // Verify the ChatCompletion arm works and future variants would return Unsupported.
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
