//! HTTP webhook hook executor.
//!
//! Sends a POST request to a configured URL with the event payload as JSON body.
//! Expects a JSON response conforming to `HookResponse`.
//!
//! # Phase 1 note
//!
//! Phase 1 provides the type skeleton and registry wiring. Actual HTTP request
//! execution is implemented in Phase 3. In Phase 1, `execute()` returns
//! `HookResponse::allow()` unconditionally. The constructor validates the URL
//! but does not make any network connections.

use std::sync::Arc;

use tracing::warn;
use weft_core::HookEvent;

use crate::hooks::HookError;
use crate::hooks::executor::HookExecutor;
use crate::hooks::types::HookResponse;

/// HTTP webhook hook executor.
///
/// Sends event payloads as JSON POST requests to a remote endpoint.
/// Uses a shared `reqwest::Client` for connection pooling.
pub(crate) struct HttpHookExecutor {
    /// The webhook URL.
    url: String,
    /// Execution timeout in milliseconds (capped at 30000).
    timeout_ms: u64,
    /// Resolved shared secret for the `Authorization: Bearer` header.
    /// `None` means no authorization header is sent.
    secret: Option<String>,
    /// Shared HTTP client (connection pooling). Arc because multiple hooks share one client.
    /// Used in Phase 3 when HTTP execution is implemented.
    #[allow(dead_code)]
    client: Arc<reqwest::Client>,
    /// Which lifecycle event this executor handles (for diagnostics).
    event: HookEvent,
}

impl HttpHookExecutor {
    /// Construct an HTTP executor.
    ///
    /// Validates that the URL is syntactically valid. Does not make any
    /// network connections at construction time.
    ///
    /// The `secret` parameter should already be resolved (env: prefix removed).
    ///
    /// # Errors
    ///
    /// Returns `HookError::HttpError` if the URL is syntactically invalid.
    pub(crate) fn new(
        url: &str,
        timeout_ms: Option<u64>,
        secret: Option<String>,
        client: Arc<reqwest::Client>,
        event: HookEvent,
    ) -> Result<Self, HookError> {
        // Cap timeout at 30000ms as per spec.
        let timeout_ms = timeout_ms.unwrap_or(5000).min(30_000);

        // Validate the URL is parseable (fail fast at startup).
        reqwest::Url::parse(url).map_err(|e| HookError::HttpError {
            url: url.to_string(),
            message: format!("invalid URL: {e}"),
        })?;

        Ok(Self {
            url: url.to_string(),
            timeout_ms,
            secret,
            client,
            event,
        })
    }
}

#[async_trait::async_trait]
impl HookExecutor for HttpHookExecutor {
    async fn execute(&self, _payload: &serde_json::Value) -> HookResponse {
        // Phase 1 stub: HTTP hook execution is implemented in Phase 3.
        // Log a warning so operators know hooks are not yet executing.
        warn!(
            url = %self.url,
            event = ?self.event,
            timeout_ms = self.timeout_ms,
            "http hook execution not yet implemented (Phase 1 stub) — returning allow"
        );
        HookResponse::allow()
    }
}

impl std::fmt::Debug for HttpHookExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpHookExecutor")
            .field("url", &self.url)
            .field("timeout_ms", &self.timeout_ms)
            // Never log the secret.
            .field("secret", &self.secret.as_ref().map(|_| "<redacted>"))
            .field("event", &self.event)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> Arc<reqwest::Client> {
        Arc::new(reqwest::Client::new())
    }

    #[test]
    fn test_invalid_url_returns_error() {
        let result = HttpHookExecutor::new(
            "not-a-url",
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid URL"));
    }

    #[test]
    fn test_valid_url_constructs_ok() {
        let exec = HttpHookExecutor::new(
            "http://example.com/hook",
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        );
        assert!(exec.is_ok());
    }

    #[test]
    fn test_timeout_capped_at_30000ms() {
        let exec = HttpHookExecutor::new(
            "http://example.com/hook",
            Some(999_999),
            None,
            test_client(),
            HookEvent::PreToolUse,
        );
        assert!(exec.is_ok());
        // Timeout is capped internally — construction succeeds.
    }

    #[tokio::test]
    async fn test_executor_returns_allow() {
        let exec = HttpHookExecutor::new(
            "http://example.com/hook",
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();
        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, crate::hooks::types::HookDecision::Allow);
    }
}
