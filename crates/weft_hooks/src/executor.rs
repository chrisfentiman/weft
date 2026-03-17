//! `HookExecutor` trait definition.
//!
//! One implementation per hook type (Rhai, HTTP). Implementations live in
//! `rhai_executor.rs` and `http_executor.rs` respectively.

use crate::types::HookResponse;

/// Trait for hook executors. One implementation per hook type.
///
/// Implementations MUST NOT panic — all errors are caught internally and
/// converted to `HookResponse::allow()` with logging. Hooks fail open.
#[async_trait::async_trait]
pub trait HookExecutor: Send + Sync {
    /// Execute the hook with the given event payload.
    ///
    /// Returns `HookResponse`. Implementations MUST NOT panic -- all errors
    /// are caught internally and converted to `HookResponse::allow()` with logging.
    async fn execute(&self, payload: &serde_json::Value) -> HookResponse;
}
