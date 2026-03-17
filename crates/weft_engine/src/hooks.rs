//! Hook helper functions shared across engine modules.
//!
//! Centralises the repeated `run_chain` + `match HookChainResult` boilerplate
//! for hook events whose block/allow semantics are identical at every call
//! site.  Events with unique semantics (RequestStart, PreResponse, PreToolUse,
//! RequestEnd) remain inline in their respective modules.

use tracing::warn;
use weft_core::{CommandResult, HookEvent};
use weft_hooks::{HookChainResult, HookRunner};

/// Fire a `PostToolUse` hook and apply any payload modifications to `result`.
///
/// `PostToolUse` blocks are non-blocking per spec — the engine logs and
/// continues.  On `Allowed`, the hook may mutate the `output` and/or
/// `success` fields of the result.
pub(crate) async fn apply_post_tool_use<H: HookRunner>(
    hooks: &H,
    command_name: &str,
    action: &str,
    result: &mut CommandResult,
) {
    let payload = serde_json::json!({
        "command": command_name,
        "action": action,
        "success": result.success,
        "output": result.output,
        "error": result.error,
    });

    match hooks
        .run_chain(HookEvent::PostToolUse, payload, Some(command_name))
        .await
    {
        HookChainResult::Allowed { payload, .. } => {
            if let Some(output) = payload.get("output").and_then(|v| v.as_str()) {
                result.output = output.to_string();
            }
            if let Some(success) = payload.get("success").and_then(|v| v.as_bool()) {
                result.success = success;
            }
        }
        HookChainResult::Blocked { hook_name, reason } => {
            warn!(
                hook = %hook_name,
                reason = %reason,
                "PostToolUse hook returned Block (non-blocking event) — ignoring"
            );
        }
    }
}
