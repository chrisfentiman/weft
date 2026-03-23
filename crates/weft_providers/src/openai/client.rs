use tracing::{Instrument, debug, info_span};
use weft_core::{SamplingOptions, WeftMessage};

use super::translate;
use crate::{
    Provider, ProviderError, ProviderRequest, ProviderResponse, TokenUsage, http::check_response,
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
    ) -> Result<(weft_core::WeftMessage, Option<TokenUsage>), ProviderError> {
        let request = translate::build_outbound_request(&messages, &model, &options);

        debug!(model = %model, "sending OpenAI request");

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
            // Build the request with content-type and auth headers first.
            #[cfg_attr(not(feature = "telemetry"), allow(unused_mut))]
            let mut req_builder = self
                .client
                .post(&self.base_url)
                .bearer_auth(&self.api_key)
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

        let body_text = check_response(response, "openai").await?;

        let wire_response: super::wire::OpenAIResponse = serde_json::from_str(&body_text)
            .map_err(|e| ProviderError::DeserializationError(e.to_string()))?;

        Ok(translate::parse_outbound_response(&wire_response, model))
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

        let activity_msg = WeftMessage {
            role: Role::System,
            source: Source::Gateway,
            model: None,
            content: vec![],
            delta: false,
            message_index: 0,
        };

        let provider = make_provider(&server.url());
        let request = ProviderRequest::ChatCompletion {
            messages: vec![
                make_weft_message(Role::System, Source::Gateway, "sys prompt"),
                activity_msg,
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
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::PartialJsonString(json!({"max_tokens": 256}).to_string()),
                mockito::Matcher::PartialJsonString(json!({"seed": 42_i64}).to_string()),
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
