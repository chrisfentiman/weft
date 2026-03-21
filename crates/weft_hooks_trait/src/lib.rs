//! `weft_hooks_trait` — Hook runner trait and chain result type.
//!
//! Contains the `HookRunner` trait and `HookChainResult`. The implementations
//! live in `weft_hooks`, which depends on this crate.
//!
//! Consumers that need the trait boundary without the implementation (e.g.
//! `weft_reactor_trait`) depend on this crate directly.

use async_trait::async_trait;

// ── HookChainResult ────────────────────────────────────────────────────────

/// Result of running a hook chain.
///
/// Consumed by the engine to determine whether to block or pass the modified
/// payload to the next stage.
#[derive(Debug)]
pub enum HookChainResult {
    /// All hooks allowed (or no hooks registered for this event).
    Allowed {
        payload: serde_json::Value,
        context: Option<String>,
    },
    /// A hook blocked the operation.
    Blocked { reason: String, hook_name: String },
}

impl HookChainResult {
    /// Convenience constructor for an Allowed result with unmodified payload and no context.
    pub fn allow(payload: serde_json::Value) -> Self {
        Self::Allowed {
            payload,
            context: None,
        }
    }
}

// ── HookRunner trait ───────────────────────────────────────────────────────

/// Trait for hook chain execution.
///
/// The engine calls this; implementations determine how hooks are resolved and run.
///
/// `Send + Sync + 'static` because the engine holds `Arc<H>` and calls from
/// async context.
#[async_trait]
pub trait HookRunner: Send + Sync + 'static {
    async fn run_chain(
        &self,
        event: weft_core::HookEvent,
        payload: serde_json::Value,
        matcher_target: Option<&str>,
    ) -> HookChainResult;
}
