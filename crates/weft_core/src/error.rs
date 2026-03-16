//! Top-level gateway error type.

/// Top-level gateway error type.
///
/// This aggregates domain errors via `String` payloads to avoid circular
/// crate dependencies. Domain crates define their own error enums
/// (`LlmError`, `RouterError`, `CommandError`, `ToolRegistryError`).
/// The binary crate converts domain errors to `WeftError` at the boundary.
#[derive(Debug, thiserror::Error)]
pub enum WeftError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("llm provider error: {0}")]
    Llm(String),
    #[error("command error: {0}")]
    Command(String),
    #[error("tool registry error: {0}")]
    ToolRegistry(String),
    #[error("max command loop iterations ({max}) exceeded")]
    CommandLoopExceeded { max: u32 },
    #[error("request timed out after {timeout_secs}s")]
    RequestTimeout { timeout_secs: u64 },
    #[error("streaming not supported in v1")]
    StreamingNotSupported,
    /// Rate-limited by the LLM provider.
    #[error("rate limited by provider, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("routing error: {0}")]
    Routing(String),
    #[error("model '{name}' not found in provider registry")]
    ModelNotFound { name: String },
    #[error("memory store error: {0}")]
    MemoryStore(String),
    /// Hard block — request terminated immediately (RequestStart, PreRoute, PostRoute).
    #[error("hook blocked at {event}: {reason} (hook: {hook_name})")]
    HookBlocked {
        event: String,
        reason: String,
        hook_name: String,
    },
    /// Feedback block exhausted retries (PreResponse only).
    #[error("hook blocked after {retries} retries at {event}: {reason} (hook: {hook_name})")]
    HookBlockedAfterRetries {
        event: String,
        reason: String,
        hook_name: String,
        retries: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_variants_constructible() {
        let _ = WeftError::Config("bad config".to_string());
        let _ = WeftError::Llm("provider failed".to_string());
        let _ = WeftError::Command("command failed".to_string());
        let _ = WeftError::ToolRegistry("registry down".to_string());
        let _ = WeftError::CommandLoopExceeded { max: 10 };
        let _ = WeftError::RequestTimeout { timeout_secs: 300 };
        let _ = WeftError::StreamingNotSupported;
        let _ = WeftError::RateLimited {
            retry_after_ms: 1000,
        };
        let _ = WeftError::Routing("routing failed".to_string());
        let _ = WeftError::ModelNotFound {
            name: "fast".to_string(),
        };
        let _ = WeftError::MemoryStore("store unavailable".to_string());
    }

    #[test]
    fn test_error_messages() {
        let err = WeftError::Config("missing field".to_string());
        assert_eq!(err.to_string(), "configuration error: missing field");

        let err = WeftError::CommandLoopExceeded { max: 5 };
        assert_eq!(err.to_string(), "max command loop iterations (5) exceeded");

        let err = WeftError::RequestTimeout { timeout_secs: 120 };
        assert_eq!(err.to_string(), "request timed out after 120s");

        let err = WeftError::StreamingNotSupported;
        assert_eq!(err.to_string(), "streaming not supported in v1");

        let err = WeftError::RateLimited {
            retry_after_ms: 2000,
        };
        assert_eq!(
            err.to_string(),
            "rate limited by provider, retry after 2000ms"
        );
    }

    #[test]
    fn test_hook_blocked_display_formatting() {
        let err = WeftError::HookBlocked {
            event: "request_start".to_string(),
            reason: "blocked by policy".to_string(),
            hook_name: "auth-hook".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "hook blocked at request_start: blocked by policy (hook: auth-hook)"
        );
    }

    #[test]
    fn test_hook_blocked_after_retries_display_formatting() {
        let err = WeftError::HookBlockedAfterRetries {
            event: "pre_response".to_string(),
            reason: "content policy violation".to_string(),
            hook_name: "content-filter".to_string(),
            retries: 2,
        };
        assert_eq!(
            err.to_string(),
            "hook blocked after 2 retries at pre_response: content policy violation (hook: content-filter)"
        );
    }
}
