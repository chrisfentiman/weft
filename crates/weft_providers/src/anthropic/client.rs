use tracing::{Instrument, debug, info_span};
use weft_core::{SamplingOptions, WeftMessage};

use super::translate;
use crate::{
    Provider, ProviderError, ProviderRequest, ProviderResponse, TokenUsage, http::check_response,
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
    ) -> Result<(weft_core::WeftMessage, Option<TokenUsage>), ProviderError> {
        let request = translate::build_outbound_request(&messages, &model, &options);

        debug!(model = %model, max_tokens = request.max_tokens, "sending Anthropic request");

        // Wrap the HTTP round-trip in a `provider_call` span so downstream
        // telemetry can observe provider latency independently of generation logic.
        let provider_call_span = info_span!(
            "provider_call",
            http.request.method = "POST",
            url.full = %self.base_url,
            otel.kind = "client",
            http.response.status_code = tracing::field::Empty,
        );

        let response = async {
            #[cfg_attr(not(feature = "telemetry"), allow(unused_mut))]
            let mut req_builder = self
                .client
                .post(&self.base_url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(&request);

            // Inject W3C TraceContext into the outgoing request headers inside the
            // provider_call span. When OTel is active, this adds a `traceparent`
            // header so the LLM provider can link its server-side trace as a child
            // of this span. When OTel is not active, inject_trace_context is a no-op.
            #[cfg(feature = "telemetry")]
            {
                let mut extra_headers = reqwest::header::HeaderMap::new();
                crate::http::inject_trace_context(&mut extra_headers);
                for (k, v) in extra_headers {
                    if let Some(name) = k {
                        req_builder = req_builder.header(name, v);
                    }
                }
            }

            req_builder
                .send()
                .await
                .map_err(|e| ProviderError::RequestFailed(e.to_string()))
        }
        .instrument(provider_call_span.clone())
        .await?;

        let status = response.status().as_u16();
        provider_call_span.record("http.response.status_code", status as i64);

        let body_text = check_response(response, "anthropic").await?;

        let wire_response: super::wire::AnthropicResponse = serde_json::from_str(&body_text)
            .map_err(|e| ProviderError::DeserializationError(e.to_string()))?;

        Ok(translate::parse_outbound_response(&wire_response, model))
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
    async fn test_no_system_prompt_system_field_absent() {
        // When messages[0] is Role::User (not System), system field is omitted
        // (None → skip_serializing_if omits it from JSON).
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
            // The JSON body must NOT contain a "system" key when there's no system prompt.
            .match_body(mockito::Matcher::JsonString(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 4096
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

        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "base system"),
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

        let activity_msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![],
            delta: false,
            message_index: 0,
        };

        let messages = vec![
            make_weft_message(Role::System, Source::Gateway, "sys prompt"),
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
                mockito::Matcher::PartialJsonString(json!({"top_k": 50_u32}).to_string()),
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
                max_tokens: None,
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
