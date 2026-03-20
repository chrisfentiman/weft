use tracing::{debug, warn};
use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};

use super::wire::{AnthropicMessage, AnthropicRequest, AnthropicResponse};
use crate::{
    Provider, ProviderError, ProviderRequest, ProviderResponse, TokenUsage,
    provider::extract_text_messages,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
///
/// One provider instance corresponds to one API endpoint (credentials + base_url).
/// The model identifier is passed per-request via `ProviderRequest::ChatCompletion.model`.
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
        messages: Vec<WeftMessage>,
        model: String,
        options: SamplingOptions,
    ) -> Result<(WeftMessage, Option<TokenUsage>), ProviderError> {
        let mut system_parts: Vec<String> = Vec::new();
        let mut wire_messages = Vec::new();

        // Extract system prompt from messages[0] if present.
        // Anthropic requires the system prompt in a dedicated top-level `system` field,
        // NOT in the messages array.
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
                system_parts.push(system_text);
            }
            1 // skip messages[0] in the remaining extraction
        } else {
            0 // no system prompt
        };

        // Process remaining messages. System-role messages that survive the gateway
        // activity filter are concatenated into the system field (Anthropic does not
        // allow system role in the messages array).
        for (role, text) in extract_text_messages(&messages[conversation_start..]) {
            match role {
                Role::System => system_parts.push(text),
                Role::User => wire_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: text,
                }),
                Role::Assistant => wire_messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: text,
                }),
            }
        }

        let system = system_parts.join("\n\n");

        // Anthropic requires max_tokens; default to 4096 if not specified.
        let max_tokens = options.max_tokens.unwrap_or(4096);

        let request = AnthropicRequest {
            model: model.clone(),
            system,
            messages: wire_messages,
            max_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            top_k: options.top_k,
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
impl Provider for AnthropicProvider {
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
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 8);
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
    async fn test_system_prompt_from_messages_0_extracted_to_system_field() {
        // System prompt at messages[0] (Role::System, Source::Gateway) is extracted
        // to Anthropic's top-level `system` field.
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
            .match_body(mockito::Matcher::PartialJsonString(
                json!({
                    "system": "You are a helpful assistant."
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
            model: "claude-test".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_no_system_prompt_produces_empty_system_field() {
        // When messages[0] is Role::User (not System), system field is empty string.
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1}
                })
                .to_string(),
            )
            .match_body(mockito::Matcher::PartialJsonString(
                json!({
                    "system": ""
                })
                .to_string(),
            ))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
            model: "claude-test".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_additional_client_system_messages_concatenated() {
        // After extracting messages[0] as system prompt, additional Role::System
        // messages (e.g. Source::Client) are concatenated into the system field.
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
            // System prompt at messages[0]
            make_weft_message(Role::System, Source::Gateway, "base system"),
            // Additional system message from client (not gateway activity -- passes filter)
            make_weft_message(Role::System, Source::Client, "You are a helpful assistant."),
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages,
            model: "claude-test".to_string(),
            options: SamplingOptions::default(),
        };
        let result = provider.execute(request).await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gateway_activity_messages_not_in_system_field() {
        // Gateway activity messages (Role::System, Source::Gateway) with non-text content
        // only are filtered out by extract_text_messages and do NOT pollute system field.
        // Activity telemetry (routing events, hook results) carries no text parts.
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
            .match_body(mockito::Matcher::PartialJsonString(
                json!({
                    "system": "sys prompt"
                })
                .to_string(),
            ))
            .create_async()
            .await;

        // Activity telemetry message: System+Gateway with NO text content parts.
        let activity_msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![], // no text -- pure telemetry, must be filtered
            delta: false,
            message_index: 0,
        };

        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "sys prompt"),
            // This gateway activity message (no text) must NOT appear in system field
            activity_msg,
            make_weft_message(Role::User, Source::Client, "Hello"),
        ];

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages,
            model: "claude-test".to_string(),
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
                    "content": [{"type": "text", "text": "response"}],
                    "usage": null
                })
                .to_string(),
            )
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::PartialJsonString(json!({"max_tokens": 512}).to_string()),
                // top_k is an integer, safe for PartialJsonString
                mockito::Matcher::PartialJsonString(json!({"top_k": 50_u32}).to_string()),
                // temperature and top_p are floats -- use regex to avoid precision issues
                mockito::Matcher::Regex(r#""temperature"\s*:"#.to_string()),
                mockito::Matcher::Regex(r#""top_p"\s*:"#.to_string()),
            ]))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
            model: "claude-test".to_string(),
            options: SamplingOptions {
                max_tokens: Some(512),
                temperature: Some(0.7),
                top_p: Some(0.95),
                top_k: Some(50),
                stop: vec![],
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
    async fn test_max_tokens_defaults_to_4096_when_none() {
        // Anthropic requires max_tokens; when SamplingOptions.max_tokens is None,
        // the provider defaults to 4096.
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": null
                })
                .to_string(),
            )
            .match_body(mockito::Matcher::PartialJsonString(
                json!({"max_tokens": 4096}).to_string(),
            ))
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
            model: "claude-test".to_string(),
            options: SamplingOptions {
                max_tokens: None, // None => default 4096
                ..SamplingOptions::default()
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
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let provider = make_provider(&server.url());
        let result = provider
            .execute(ProviderRequest::ChatCompletion {
                messages: vec![make_weft_message(Role::User, Source::Client, "Hello")],
                model: "claude-3-5-sonnet".to_string(),
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
        assert_eq!(message.model, Some("claude-3-5-sonnet".to_string()));
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
        let result = provider.execute(make_request("claude-test")).await;

        assert!(matches!(
            result,
            Err(ProviderError::DeserializationError(_))
        ));
    }

    #[tokio::test]
    async fn test_execute_unsupported_variant_returns_unsupported() {
        // ChatCompletion should succeed — not unsupported.
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
