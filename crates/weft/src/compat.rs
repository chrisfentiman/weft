//! Shared validation and error mapping for compat HTTP endpoints.
//!
//! Both the OpenAI-compatible (`/v1/chat/completions`) and the
//! Anthropic-compatible (`/v1/messages`) endpoints share identical
//! request validation rules and a common `WeftError → HTTP status` mapping.
//!
//! This module centralises that logic so it lives exactly once:
//! - `validate_compat_request` — shared pre-flight checks
//! - `CompatError` — normalized error with status, message, retry-after
//! - `From<CompatError> for ApiError` — OpenAI body shaping
//! - `From<CompatError> for AnthropicApiError` — Anthropic body shaping

use axum::http::StatusCode;
use weft_core::WeftError;

// ── Validation ───────────────────────────────────────────────────────────────

/// Validate common compat-endpoint request constraints.
///
/// Returns `Err(message)` when:
/// - `messages_empty` is `true` → messages array is empty
/// - `has_user_message` is `false` → no user-role message present
/// - `stream` is `Some(true)` → streaming is not supported in v1
///
/// On success returns `Ok(())`.
pub fn validate_compat_request(
    messages_empty: bool,
    has_user_message: bool,
    stream: Option<bool>,
) -> Result<(), String> {
    if messages_empty {
        return Err("messages array must not be empty".to_string());
    }
    if !has_user_message {
        return Err("messages must contain at least one user message".to_string());
    }
    if stream == Some(true) {
        return Err("streaming is not supported in v1".to_string());
    }
    Ok(())
}

// ── CompatError ──────────────────────────────────────────────────────────────

/// Normalized error produced by compat endpoint handlers.
///
/// Carries the HTTP status, a human-readable message, an optional Retry-After
/// value (milliseconds), and the Anthropic-style error type string (used only
/// when constructing Anthropic-format responses).
pub struct CompatError {
    pub status: StatusCode,
    pub message: String,
    /// Retry-After in milliseconds; present only for 429 responses.
    pub retry_after_ms: Option<u64>,
    /// Anthropic error type string (e.g. `"rate_limit_error"`, `"api_error"`).
    /// OpenAI responses use a fixed `"invalid_request_error"` type and ignore this.
    pub anthropic_error_type: String,
}

impl CompatError {
    /// Create a 400 Bad Request error.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            retry_after_ms: None,
            anthropic_error_type: "invalid_request_error".to_string(),
        }
    }

    /// Create a 500 Internal Server Error.
    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            retry_after_ms: None,
            anthropic_error_type: "api_error".to_string(),
        }
    }

    /// Map a `WeftError` to a `CompatError`.
    ///
    /// This is the single authoritative mapping from domain errors to HTTP
    /// status codes and Anthropic error-type strings. Both `ApiError` and
    /// `AnthropicApiError` delegate here so the mapping exists exactly once.
    pub fn from_weft_error(e: WeftError) -> Self {
        match e {
            WeftError::InvalidRequest(_) => Self::bad_request(e.to_string()),
            WeftError::StreamingNotSupported => Self::bad_request(e.to_string()),
            WeftError::NoEligibleModels { .. } => Self::bad_request(e.to_string()),
            WeftError::ProtoConversion(_) => Self::internal(e.to_string()),
            WeftError::Config(_) => Self::internal(e.to_string()),
            WeftError::CommandLoopExceeded { .. } => Self::internal(e.to_string()),
            WeftError::Llm(_) => Self::internal(e.to_string()),
            WeftError::Routing(_) => Self::internal(e.to_string()),
            WeftError::ModelNotFound { .. } => Self::internal(e.to_string()),
            WeftError::Command(_) => Self::internal(e.to_string()),
            WeftError::RateLimited { retry_after_ms } => Self {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: e.to_string(),
                retry_after_ms: Some(retry_after_ms),
                anthropic_error_type: "rate_limit_error".to_string(),
            },
            WeftError::RequestTimeout { .. } => Self {
                status: StatusCode::GATEWAY_TIMEOUT,
                message: e.to_string(),
                retry_after_ms: None,
                anthropic_error_type: "api_error".to_string(),
            },
            WeftError::ToolRegistry(_) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: e.to_string(),
                retry_after_ms: None,
                anthropic_error_type: "api_error".to_string(),
            },
            WeftError::MemoryStore(_) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: e.to_string(),
                retry_after_ms: None,
                anthropic_error_type: "api_error".to_string(),
            },
            // Hard block: hook terminated the request before LLM involvement.
            WeftError::HookBlocked { .. } => Self {
                status: StatusCode::FORBIDDEN,
                message: e.to_string(),
                retry_after_ms: None,
                anthropic_error_type: "permission_error".to_string(),
            },
            // Feedback block exhausted retries (PreResponse only).
            WeftError::HookBlockedAfterRetries { .. } => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                message: e.to_string(),
                retry_after_ms: None,
                anthropic_error_type: "invalid_request_error".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_compat_request ──────────────────────────────────────────────

    #[test]
    fn test_validate_empty_messages_returns_err() {
        let result = validate_compat_request(true, false, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("messages array must not be empty")
        );
    }

    #[test]
    fn test_validate_no_user_message_returns_err() {
        let result = validate_compat_request(false, false, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one user message"));
    }

    #[test]
    fn test_validate_stream_true_returns_err() {
        let result = validate_compat_request(false, true, Some(true));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("streaming is not supported"));
    }

    #[test]
    fn test_validate_stream_false_is_ok() {
        let result = validate_compat_request(false, true, Some(false));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_stream_none_is_ok() {
        let result = validate_compat_request(false, true, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_valid_request_is_ok() {
        let result = validate_compat_request(false, true, None);
        assert!(result.is_ok());
    }

    // ── CompatError::from_weft_error ─────────────────────────────────────────

    #[test]
    fn test_invalid_request_maps_to_400() {
        let err = CompatError::from_weft_error(WeftError::InvalidRequest("bad".to_string()));
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_rate_limited_maps_to_429_with_retry_after() {
        let err = CompatError::from_weft_error(WeftError::RateLimited {
            retry_after_ms: 30_000,
        });
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.retry_after_ms, Some(30_000));
        assert_eq!(err.anthropic_error_type, "rate_limit_error");
    }

    #[test]
    fn test_hook_blocked_maps_to_403() {
        let err = CompatError::from_weft_error(WeftError::HookBlocked {
            event: "request_start".to_string(),
            hook_name: "jailbreak_detector".to_string(),
            reason: "blocked".to_string(),
        });
        assert_eq!(err.status, StatusCode::FORBIDDEN);
        assert_eq!(err.anthropic_error_type, "permission_error");
    }

    #[test]
    fn test_hook_blocked_after_retries_maps_to_422() {
        let err = CompatError::from_weft_error(WeftError::HookBlockedAfterRetries {
            event: "pre_response".to_string(),
            hook_name: "feedback_hook".to_string(),
            reason: "policy violation".to_string(),
            retries: 3,
        });
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn test_config_error_maps_to_500() {
        let err = CompatError::from_weft_error(WeftError::Config("bad config".to_string()));
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.anthropic_error_type, "api_error");
    }

    #[test]
    fn test_no_eligible_models_maps_to_400() {
        let err = CompatError::from_weft_error(WeftError::NoEligibleModels {
            capability: "chat_completions".to_string(),
        });
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_tool_registry_maps_to_503() {
        let err =
            CompatError::from_weft_error(WeftError::ToolRegistry("registry down".to_string()));
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_timeout_maps_to_504() {
        let err = CompatError::from_weft_error(WeftError::RequestTimeout { timeout_secs: 30 });
        assert_eq!(err.status, StatusCode::GATEWAY_TIMEOUT);
    }
}
