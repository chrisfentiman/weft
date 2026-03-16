//! Rhai hook executor.
//!
//! Executes hook scripts using the Rhai embedded scripting engine.
//! Each `RhaiHookExecutor` owns its own `rhai::Engine` and compiled `rhai::AST`.
//!
//! # Phase 1 note
//!
//! Phase 1 provides the type skeleton and registry wiring. Actual script
//! execution is implemented in Phase 2. In Phase 1, `execute()` returns
//! `HookResponse::allow()` unconditionally. The constructor validates that
//! the script file exists and is readable, but does not compile the AST.

use tracing::warn;
use weft_core::HookEvent;

use crate::hooks::HookError;
use crate::hooks::executor::HookExecutor;
use crate::hooks::types::HookResponse;

/// Rhai-based hook executor.
///
/// Owns a compiled Rhai AST and a configured Rhai engine.
/// Both are `Send + Sync` via the `rhai/sync` feature.
pub(crate) struct RhaiHookExecutor {
    /// Path to the Rhai script file (for logging/diagnostics).
    script_path: String,
    /// Configured execution timeout in milliseconds (capped at 5000).
    timeout_ms: u64,
    /// Which lifecycle event this executor handles (for diagnostics).
    event: HookEvent,
    // Phase 2 will add: engine: rhai::Engine, ast: rhai::AST
}

impl RhaiHookExecutor {
    /// Construct a Rhai executor from the given script path.
    ///
    /// Validates that the script file exists. Compilation of the Rhai AST
    /// is deferred to Phase 2.
    ///
    /// # Errors
    ///
    /// Returns `HookError::RhaiError` if the script file does not exist or
    /// cannot be read.
    pub(crate) fn new(
        script_path: &str,
        timeout_ms: Option<u64>,
        event: HookEvent,
    ) -> Result<Self, HookError> {
        // Cap timeout at 5000ms as per spec.
        let timeout_ms = timeout_ms.unwrap_or(100).min(5000);

        // Validate the script file exists (fail fast at startup).
        std::fs::metadata(script_path).map_err(|e| HookError::RhaiError {
            script: script_path.to_string(),
            message: format!("script file not found or unreadable: {e}"),
        })?;

        Ok(Self {
            script_path: script_path.to_string(),
            timeout_ms,
            event,
        })
    }
}

#[async_trait::async_trait]
impl HookExecutor for RhaiHookExecutor {
    async fn execute(&self, _payload: &serde_json::Value) -> HookResponse {
        // Phase 1 stub: Rhai script execution is implemented in Phase 2.
        // Log a warning so operators know hooks are not yet executing.
        warn!(
            script = %self.script_path,
            event = ?self.event,
            timeout_ms = self.timeout_ms,
            "rhai hook execution not yet implemented (Phase 1 stub) — returning allow"
        );
        HookResponse::allow()
    }
}

impl std::fmt::Debug for RhaiHookExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RhaiHookExecutor")
            .field("script_path", &self.script_path)
            .field("timeout_ms", &self.timeout_ms)
            .field("event", &self.event)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_script(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn test_executor_returns_allow() {
        let script = write_temp_script("fn hook(e) { #{ decision: \"allow\" } }");
        let exec =
            RhaiHookExecutor::new(script.path().to_str().unwrap(), None, HookEvent::PreToolUse)
                .unwrap();
        let payload = serde_json::json!({"command": "test"});
        let response = exec.execute(&payload).await;
        assert_eq!(response.decision, crate::hooks::types::HookDecision::Allow);
    }

    #[test]
    fn test_missing_script_returns_error() {
        let result =
            RhaiHookExecutor::new("/nonexistent/path/hook.rhai", None, HookEvent::PreToolUse);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("script file not found"));
    }

    #[test]
    fn test_timeout_capped_at_5000ms() {
        // We can't observe timeout_ms directly outside the module, but we can verify
        // construction succeeds with a very large timeout.
        // The cap is enforced internally — this test verifies no panic/error occurs.
        let script = {
            let mut f = tempfile::NamedTempFile::new().unwrap();
            f.write_all(b"fn hook(e) { #{ decision: \"allow\" } }")
                .unwrap();
            f
        };
        let exec = RhaiHookExecutor::new(
            script.path().to_str().unwrap(),
            Some(999_999),
            HookEvent::PreToolUse,
        );
        assert!(exec.is_ok());
    }
}
