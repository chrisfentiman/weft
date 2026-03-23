//! Shared HTTP client helpers for provider implementations.
//!
//! Contains common HTTP boilerplate used by all built-in provider clients:
//! trace context injection, status code handling, response body extraction.

use tracing::warn;

use crate::ProviderError;

/// Inject W3C TraceContext into outgoing HTTP request headers.
///
/// Uses the current OTel context (propagated from the active tracing span by
/// the tracing-opentelemetry layer) to set the `traceparent` header. When OTel
/// is not configured, the propagator injects nothing — this is always safe to call.
///
/// Compiled only when the `telemetry` feature is enabled.
#[cfg(feature = "telemetry")]
pub fn inject_trace_context(headers: &mut reqwest::header::HeaderMap) {
    use opentelemetry::propagation::{Injector, TextMapPropagator};
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    struct HeaderInjector<'a>(&'a mut reqwest::header::HeaderMap);

    impl Injector for HeaderInjector<'_> {
        fn set(&mut self, key: &str, value: String) {
            if let Ok(name) = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                && let Ok(val) = reqwest::header::HeaderValue::from_str(&value)
            {
                self.0.insert(name, val);
            }
        }
    }

    let propagator = TraceContextPropagator::new();
    propagator.inject_context(
        &opentelemetry::Context::current(),
        &mut HeaderInjector(headers),
    );
}

/// Check the HTTP response status and return the appropriate `ProviderError`
/// for error statuses, or the response body as a `String` on success.
///
/// - 429 → `ProviderError::RateLimited` (extracts Retry-After header)
/// - Other non-2xx → `ProviderError::ProviderHttpError`
/// - 2xx → returns the response body as a string
pub async fn check_response(
    response: reqwest::Response,
    provider_name: &str,
) -> Result<String, ProviderError> {
    let status = response.status();
    let status_u16 = status.as_u16();

    // Check 429 first so we can read the Retry-After header before consuming the body.
    if status_u16 == 429 {
        let retry_after_ms = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|secs| secs * 1000)
            .unwrap_or(60_000);
        return Err(ProviderError::RateLimited { retry_after_ms });
    }

    // Accept any 2xx status code (200 OK, 201 Created, etc.), not just exactly 200.
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<failed to read body>".to_string());
        warn!(
            status = status_u16,
            provider = provider_name,
            "provider returned non-2xx"
        );
        return Err(ProviderError::ProviderHttpError {
            status: status_u16,
            body,
        });
    }

    response
        .text()
        .await
        .map_err(|e| ProviderError::RequestFailed(e.to_string()))
}
