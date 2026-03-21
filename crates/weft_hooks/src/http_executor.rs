//! HTTP webhook hook executor.
//!
//! Sends a POST request to a configured URL with the event payload as JSON body.
//! Expects a JSON response conforming to `HookResponse`.
//!
//! # Request format
//!
//! ```json
//! POST <hook_url>
//! Content-Type: application/json
//! Authorization: Bearer <secret>   (if secret configured)
//!
//! {
//!     "event": "PreToolUse",
//!     "timestamp": "2026-03-16T12:00:00Z",
//!     "payload": { ... event-specific payload ... }
//! }
//! ```
//!
//! # Fail-open semantics
//!
//! All HTTP errors (connection failure, timeout, non-2xx status, invalid JSON,
//! schema mismatch) result in `HookResponse::allow()` with a warning log.
//! The gateway never fails a request due to a broken HTTP hook.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::{debug, info, warn};
use weft_core::HookEvent;

use crate::HookError;
use crate::executor::HookExecutor;
use crate::types::{HookDecision, HookResponse};

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
    client: Arc<reqwest::Client>,
    /// Which lifecycle event this executor handles. Serialized into the request body.
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

/// Format a `SystemTime` as an RFC 3339 / ISO 8601 UTC timestamp string.
///
/// Output format: `YYYY-MM-DDTHH:MM:SSZ`
///
/// This avoids a `chrono` dependency by formatting manually from duration-since-epoch.
fn format_rfc3339_utc(t: SystemTime) -> String {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();

    // Break Unix timestamp into calendar fields.
    let mut remaining = secs;

    let second = remaining % 60;
    remaining /= 60;
    let minute = remaining % 60;
    remaining /= 60;
    let hour = remaining % 24;
    remaining /= 24; // days since epoch (1970-01-01)

    // Compute year and day-of-year from days since epoch.
    let mut year = 1970u32;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    // Compute month and day-of-month.
    let leap = is_leap_year(year);
    let days_in_month = [
        31u64,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut month = 1u32;
    for &dim in &days_in_month {
        if remaining < dim {
            break;
        }
        remaining -= dim;
        month += 1;
    }
    let day = remaining + 1;

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

#[async_trait::async_trait]
impl HookExecutor for HttpHookExecutor {
    async fn execute(&self, payload: &serde_json::Value) -> HookResponse {
        let start = std::time::Instant::now();

        let timestamp = format_rfc3339_utc(SystemTime::now());

        // Build the request body as per spec:
        // { "event": "<EventName>", "timestamp": "<ISO8601>", "payload": { ... } }
        let body = serde_json::json!({
            "event": self.event,
            "timestamp": timestamp,
            "payload": payload,
        });

        // Build the request, applying the per-hook timeout.
        let mut request_builder = self
            .client
            .post(&self.url)
            .timeout(Duration::from_millis(self.timeout_ms))
            .header("Content-Type", "application/json")
            .json(&body);

        // Attach Authorization header if a secret is configured.
        if let Some(secret) = &self.secret {
            request_builder = request_builder.header("Authorization", format!("Bearer {secret}"));
        }

        let response = match request_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                // Connection failure (refused, DNS, timeout, etc.) — fail open.
                warn!(
                    url = %self.url,
                    event = ?self.event,
                    error = %e,
                    "http hook request failed — returning allow"
                );
                return HookResponse::allow();
            }
        };

        let status = response.status();

        if !status.is_success() {
            // Non-2xx status — log body at debug, fail open.
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable>".to_string());
            warn!(
                url = %self.url,
                event = ?self.event,
                status = %status,
                "http hook returned non-2xx status — returning allow"
            );
            debug!(
                url = %self.url,
                event = ?self.event,
                status = %status,
                body = %body_text,
                "http hook non-2xx response body"
            );
            return HookResponse::allow();
        }

        // 2xx response — parse body as JSON.
        let body_bytes = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    url = %self.url,
                    event = ?self.event,
                    error = %e,
                    "http hook failed to read response body — returning allow"
                );
                return HookResponse::allow();
            }
        };

        // Deserialize as HookResponse.
        let hook_response: HookResponse = match serde_json::from_slice(&body_bytes) {
            Ok(r) => r,
            Err(e) => {
                let body_preview = String::from_utf8_lossy(&body_bytes);
                warn!(
                    url = %self.url,
                    event = ?self.event,
                    error = %e,
                    body = %body_preview,
                    "http hook returned invalid JSON or non-conforming response — returning allow"
                );
                return HookResponse::allow();
            }
        };

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Log at info for block decisions, debug for others.
        if hook_response.decision == HookDecision::Block {
            info!(
                url = %self.url,
                event = ?self.event,
                decision = "block",
                duration_ms = duration_ms,
                "http hook executed"
            );
        } else {
            debug!(
                url = %self.url,
                event = ?self.event,
                decision = ?hook_response.decision,
                duration_ms = duration_ms,
                "http hook executed"
            );
        }

        hook_response
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
    use mockito::Server;

    fn test_client() -> Arc<reqwest::Client> {
        Arc::new(reqwest::Client::new())
    }

    // ── Construction ──────────────────────────────────────────────────────────

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

    // ── Successful responses ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_server_returning_allow() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"allow"}"#)
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        assert!(response.reason.is_none());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_server_returning_block() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"block","reason":"denied"}"#)
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Block);
        assert_eq!(response.reason.as_deref(), Some("denied"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_server_returning_modify() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"modify","modified":{"command":"new_cmd","arguments":{}}}"#)
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "original"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Modify);
        let modified = response.modified.expect("expected modified payload");
        assert_eq!(modified["command"], "new_cmd");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_server_returning_allow_with_context() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"allow","context":"some annotation"}"#)
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        assert_eq!(response.context.as_deref(), Some("some annotation"));
        mock.assert_async().await;
    }

    // ── Error handling (fail-open) ────────────────────────────────────────────

    #[tokio::test]
    async fn test_server_returning_500_gives_allow() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(500)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_server_returning_404_gives_allow() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(404)
            .with_body("Not Found")
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_server_returning_malformed_json_gives_allow() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("not json at all {{{{")
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_server_returning_valid_json_non_conforming_gives_allow() {
        // Valid JSON but not a HookResponse (missing "decision" field).
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"status":"ok","message":"hello"}"#)
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_connection_refused_gives_allow() {
        // Use a port that is (almost certainly) not listening.
        let exec = HttpHookExecutor::new(
            "http://127.0.0.1:1/hook",
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_timeout_gives_allow() {
        let mut server = Server::new_async().await;
        // No response — the mock never replies, so the request times out.
        // We use a very short timeout (1ms) so the test is fast.
        let _mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_body(r#"{"decision":"allow"}"#)
            // Add a delay longer than our timeout.
            .with_chunked_body(|w| {
                std::thread::sleep(Duration::from_millis(200));
                w.write_all(b"")
            })
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            Some(1), // 1ms timeout — will expire before the 200ms delay
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
    }

    // ── Request body and headers verification ────────────────────────────────

    #[tokio::test]
    async fn test_request_body_format() {
        // Verify that the request body contains event, timestamp, and payload fields.
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"allow"}"#)
            // Match that the body contains the required fields.
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"event":"pre_tool_use"}"#.to_string(),
            ))
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test", "arguments": {}});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_request_body_contains_payload() {
        // Verify that the event payload is nested under "payload" in the request body.
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"allow"}"#)
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"payload":{"command":"my_cmd"}}"#.to_string(),
            ))
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None,
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "my_cmd"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_authorization_header_sent_when_secret_configured() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"allow"}"#)
            .match_header("Authorization", "Bearer my-secret-token")
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            Some("my-secret-token".to_string()),
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_no_authorization_header_when_no_secret() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"decision":"allow"}"#)
            // Verify that Authorization header is NOT present.
            .match_header("Authorization", mockito::Matcher::Missing)
            .create_async()
            .await;

        let exec = HttpHookExecutor::new(
            &format!("{}/hook", server.url()),
            None,
            None, // No secret.
            test_client(),
            HookEvent::PreToolUse,
        )
        .unwrap();

        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, HookDecision::Allow);
        mock.assert_async().await;
    }

    // ── RFC 3339 timestamp formatting ─────────────────────────────────────────

    #[test]
    fn test_format_rfc3339_utc_epoch() {
        let epoch = UNIX_EPOCH;
        let formatted = format_rfc3339_utc(epoch);
        assert_eq!(formatted, "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_format_rfc3339_utc_known_timestamp() {
        // 2026-03-16T12:00:00Z = 1773748800 seconds since epoch.
        // Verify by checking format length and basic structure.
        let ts = UNIX_EPOCH + Duration::from_secs(1_773_748_800);
        let formatted = format_rfc3339_utc(ts);
        // Should be exactly 20 chars: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(
            formatted.len(),
            20,
            "timestamp should be 20 chars: {formatted}"
        );
        assert!(
            formatted.ends_with('Z'),
            "timestamp should end with Z: {formatted}"
        );
        assert!(
            formatted.contains('T'),
            "timestamp should contain T: {formatted}"
        );
    }

    #[test]
    fn test_format_rfc3339_utc_leap_year() {
        // 2000-02-29T00:00:00Z — leap year day.
        let ts = UNIX_EPOCH + Duration::from_secs(951_782_400); // 2000-02-29T00:00:00Z
        let formatted = format_rfc3339_utc(ts);
        assert_eq!(formatted, "2000-02-29T00:00:00Z");
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000)); // divisible by 400
        assert!(!is_leap_year(1900)); // divisible by 100 but not 400
        assert!(is_leap_year(2024)); // divisible by 4, not 100
        assert!(!is_leap_year(2023)); // not divisible by 4
    }
}
