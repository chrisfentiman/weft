//! Hook system for the Weft gateway.
//!
//! Hooks are synchronous evaluation gates placed at lifecycle checkpoints in the
//! gateway request flow. They evaluate, validate, and redirect — they do not
//! fetch external state or store data.
//!
//! # Architecture
//!
//! - Config types (`HookEvent`, `HookRoutingDomain`, `RoutingTrigger`, `HookConfig`)
//!   live in `weft_core::config` to avoid circular dependencies.
//! - Hook implementation types (`HookDecision`, `HookResponse`, `HookMatcher`)
//!   live in `types.rs`.
//! - The `HookExecutor` trait lives in `executor.rs`.
//! - Executor implementations live in `rhai_executor.rs` and `http_executor.rs`.
//! - `HookRegistry` and chain execution logic live in this file (`lib.rs`).
//!
//! # Fail-open semantics
//!
//! Hook errors never fail requests. A broken hook returns `Allow` with a warning log.
//! Hooks are optional evaluation gates; the gateway's primary job is routing to LLMs.

pub mod executor;
pub(crate) mod http_executor;
pub(crate) mod rhai_executor;
pub mod types;

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{debug, warn};
use weft_core::{HookConfig, HookEvent, HookType};

use crate::executor::HookExecutor;
use crate::http_executor::HttpHookExecutor;
use crate::rhai_executor::RhaiHookExecutor;
use crate::types::{HookDecision, HookMatcher};

/// Error type for hook registry construction.
///
/// Used during startup only — these errors are fatal (the gateway does not
/// start with broken hook configs). During request processing, errors are
/// caught by executors and logged; they never propagate as `HookError`.
#[derive(Debug, thiserror::Error)]
#[allow(clippy::enum_variant_names)]
pub enum HookError {
    #[error("rhai script error: {script}: {message}")]
    RhaiError { script: String, message: String },
    #[error("http hook error: {url}: {message}")]
    HttpError { url: String, message: String },
    /// Returned when a hook returns a response that violates validation rules.
    #[allow(dead_code)]
    #[error("hook response validation error: {hook_name}: {message}")]
    ValidationError { hook_name: String, message: String },
    #[error("hook registry construction error: {0}")]
    RegistryError(String),
}

/// A registered hook with its configuration and compiled execution state.
pub struct RegisteredHook {
    /// Which lifecycle event this hook fires on.
    pub event: HookEvent,
    /// Optional compiled matcher.
    pub matcher: HookMatcher,
    /// The hook executor (Rhai, HTTP, or future Weft).
    pub executor: Box<dyn HookExecutor>,
    /// Display name for logging (derived from config: script path for Rhai, URL for HTTP).
    pub name: String,
    /// Execution priority. Lower values run first.
    pub priority: u32,
}

impl std::fmt::Debug for RegisteredHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredHook")
            .field("event", &self.event)
            .field("matcher", &self.matcher)
            .field("name", &self.name)
            .field("priority", &self.priority)
            .finish()
    }
}

/// Result of running a hook chain.
///
/// Consumed by the engine to determine whether to block or
/// pass the modified payload to the next stage.
#[derive(Debug)]
pub enum HookChainResult {
    /// All hooks allowed (or no hooks registered for this event).
    /// Contains the final payload (possibly modified by Modify hooks)
    /// and any accumulated context strings.
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

/// Trait for hook chain execution.
///
/// The engine calls this; implementations determine how hooks are resolved and run.
///
/// `Send + Sync + 'static` because the engine holds `Arc<H>` and calls
/// from async context.
#[async_trait::async_trait]
pub trait HookRunner: Send + Sync + 'static {
    /// Run all hooks for the given event, in priority order.
    async fn run_chain(
        &self,
        event: HookEvent,
        payload: serde_json::Value,
        matcher_target: Option<&str>,
    ) -> HookChainResult;
}

/// Null implementation. Always returns Allowed with unmodified payload.
///
/// Used in tests that do not exercise hook behavior and in contexts where
/// no hook runner is needed.
pub struct NullHookRunner;

#[async_trait::async_trait]
impl HookRunner for NullHookRunner {
    async fn run_chain(
        &self,
        _event: HookEvent,
        payload: serde_json::Value,
        _matcher_target: Option<&str>,
    ) -> HookChainResult {
        HookChainResult::Allowed {
            payload,
            context: None,
        }
    }
}

/// The hook registry: owns all registered hooks, grouped by event.
///
/// Constructed at startup from config. Immutable after construction.
/// Shared via `Arc` (all fields are `Send + Sync`).
pub struct HookRegistry {
    /// Hooks indexed by event type, in priority-sorted order.
    /// Read by `run_chain` during the engine request loop.
    hooks: HashMap<HookEvent, Vec<RegisteredHook>>,
}

impl HookRegistry {
    /// Construct a registry from hook configuration entries.
    ///
    /// This is called once at startup. Failures are fatal — the gateway does
    /// not start with broken hook configs.
    ///
    /// For each hook config entry:
    /// 1. Compiles the matcher regex (fails if invalid).
    /// 2. Constructs the executor (validates file/URL existence).
    /// 3. Weft-type hooks are skipped with a warning log.
    /// 4. Hooks are sorted by priority (ascending, stable for equal priorities).
    pub fn from_config(
        hooks_config: &[HookConfig],
        http_client: Arc<reqwest::Client>,
    ) -> Result<Self, HookError> {
        let mut hooks: HashMap<HookEvent, Vec<RegisteredHook>> = HashMap::new();

        for (i, config) in hooks_config.iter().enumerate() {
            // Skip Weft-type hooks with a warning — not yet implemented.
            if config.hook_type == HookType::Weft {
                warn!(
                    hook_index = i,
                    agent = ?config.agent,
                    "weft-type hooks are reserved for future implementation — skipping hook[{i}]"
                );
                continue;
            }

            // Compile the matcher.
            let matcher = HookMatcher::new(config.matcher.as_deref(), i)?;

            // Construct the executor.
            let (executor, name): (Box<dyn HookExecutor>, String) = match &config.hook_type {
                HookType::Rhai => {
                    let script = config.script.as_deref().ok_or_else(|| {
                        HookError::RegistryError(format!(
                            "hooks[{i}]: rhai hook requires 'script' field"
                        ))
                    })?;
                    let exec = RhaiHookExecutor::new(script, config.timeout_ms, config.event)?;
                    (Box::new(exec), script.to_string())
                }
                HookType::Http => {
                    let url = config.url.as_deref().ok_or_else(|| {
                        HookError::RegistryError(format!(
                            "hooks[{i}]: http hook requires 'url' field"
                        ))
                    })?;
                    let exec = HttpHookExecutor::new(
                        url,
                        config.timeout_ms,
                        config.secret.clone(),
                        Arc::clone(&http_client),
                        config.event,
                    )?;
                    (Box::new(exec), url.to_string())
                }
                HookType::Weft => unreachable!("weft hooks filtered above"),
            };

            let registered = RegisteredHook {
                event: config.event,
                matcher,
                executor,
                name,
                priority: config.priority,
            };

            hooks.entry(config.event).or_default().push(registered);
        }

        // Sort each event's hooks by priority (ascending, stable).
        for hooks_for_event in hooks.values_mut() {
            hooks_for_event.sort_by_key(|h| h.priority);
        }

        Ok(Self { hooks })
    }

    /// Construct an empty registry (no hooks configured).
    ///
    /// Intended for test use. No allocations at request time — `run_chain` returns
    /// `Allowed` immediately when no hooks are registered.
    #[cfg(any(test, feature = "test-support"))]
    pub fn empty() -> Self {
        Self {
            hooks: HashMap::new(),
        }
    }

    /// Construct a registry from a pre-built event-to-hooks map.
    ///
    /// Intended for test use to build registries with inline executor implementations
    /// without going through `from_config` (which requires real Rhai scripts or URLs).
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_registered(hooks: HashMap<HookEvent, Vec<RegisteredHook>>) -> Self {
        Self { hooks }
    }

    /// Returns the number of registered hooks for a given event.
    /// Intended for diagnostics and testing.
    #[cfg(any(test, feature = "test-support"))]
    pub fn hook_count(&self, event: HookEvent) -> usize {
        self.hooks.get(&event).map(|v| v.len()).unwrap_or(0)
    }
}

#[async_trait::async_trait]
impl HookRunner for HookRegistry {
    /// Run all hooks for the given event, in priority order.
    ///
    /// Returns the final payload (possibly modified) and whether the chain was blocked.
    ///
    /// - On `Block`: returns immediately with the blocking hook's reason.
    /// - On `Modify`: passes the modified payload to subsequent hooks.
    /// - On `Allow`: passes the original payload to subsequent hooks.
    ///
    /// `matcher_target` is event-specific:
    /// - `PreRoute`/`PostRoute`: routing domain name ("model", "commands", "memory", "tool_necessity")
    /// - `PreToolUse`/`PostToolUse`: command name
    /// - Other events: `None` (matcher ignored — hook always fires)
    async fn run_chain(
        &self,
        event: HookEvent,
        payload: serde_json::Value,
        matcher_target: Option<&str>,
    ) -> HookChainResult {
        let hooks = match self.hooks.get(&event) {
            Some(h) if !h.is_empty() => h,
            _ => {
                // No hooks registered for this event — fast path.
                return HookChainResult::Allowed {
                    payload,
                    context: None,
                };
            }
        };

        let mut current_payload = payload;
        // Accumulated context strings from Allow/Modify hooks.
        let mut context_parts: Vec<String> = Vec::new();

        for hook in hooks {
            // Evaluate matcher.
            if !hook.matcher.matches(matcher_target) {
                debug!(
                    hook = %hook.name,
                    event = ?event,
                    matcher_target = ?matcher_target,
                    "hook skipped: matcher did not match"
                );
                continue;
            }

            // Execute the hook.
            let response = hook.executor.execute(&current_payload).await;

            match response.decision {
                HookDecision::Allow => {
                    // Accumulate context if present.
                    if let Some(ctx) = response.context {
                        // Per-hook context capped at 1024 bytes.
                        let truncated = truncate_bytes(&ctx, 1024);
                        context_parts.push(truncated);
                    }
                    debug!(hook = %hook.name, event = ?event, "hook allowed");
                }

                HookDecision::Block => {
                    if event.can_block() {
                        // Block the chain — return immediately.
                        let reason = response
                            .reason
                            .unwrap_or_else(|| "hook blocked without reason".to_string());
                        return HookChainResult::Blocked {
                            reason,
                            hook_name: hook.name.clone(),
                        };
                    } else {
                        // Block on a non-blocking event: log and continue.
                        warn!(
                            hook = %hook.name,
                            event = ?event,
                            "hook returned Block on non-blocking event — treating as Allow"
                        );
                    }
                }

                HookDecision::Modify => {
                    if let Some(modified) = response.modified {
                        // Replace the current payload with the modified version.
                        current_payload = modified;
                        debug!(hook = %hook.name, event = ?event, "hook modified payload");
                    } else {
                        // Modify with no modified payload: treat as Allow with warning.
                        warn!(
                            hook = %hook.name,
                            event = ?event,
                            "hook returned Modify with no modified payload — treating as Allow"
                        );
                    }
                    // Accumulate context if present.
                    if let Some(ctx) = response.context {
                        let truncated = truncate_bytes(&ctx, 1024);
                        context_parts.push(truncated);
                    }
                }
            }
        }

        // Assemble accumulated context.
        let context = if context_parts.is_empty() {
            None
        } else {
            let joined = context_parts.join("\n");
            // Total context capped at 4096 bytes.
            Some(truncate_bytes_with_suffix(&joined, 4096, "...[truncated]"))
        };

        HookChainResult::Allowed {
            payload: current_payload,
            context,
        }
    }
}

/// Truncate a string to at most `max_bytes` bytes, preserving UTF-8 boundaries.
fn truncate_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    // Find the last valid UTF-8 boundary at or before max_bytes.
    let end = s
        .char_indices()
        .take_while(|(i, _)| *i < max_bytes)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    s[..end].to_string()
}

/// Truncate a string to at most `max_bytes` bytes, appending `suffix` if truncated.
fn truncate_bytes_with_suffix(s: &str, max_bytes: usize, suffix: &str) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let suffix_len = suffix.len();
    // Make room for the suffix.
    let target = max_bytes.saturating_sub(suffix_len);
    let truncated = truncate_bytes(s, target);
    format!("{truncated}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HookDecision, HookResponse};

    fn test_http_client() -> Arc<reqwest::Client> {
        Arc::new(reqwest::Client::new())
    }

    // ── Empty registry ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_empty_registry_returns_allowed() {
        let registry = HookRegistry::empty();
        let payload = serde_json::json!({"request": "test"});
        let result = registry
            .run_chain(HookEvent::RequestStart, payload.clone(), None)
            .await;
        assert!(matches!(result, HookChainResult::Allowed { .. }));
        if let HookChainResult::Allowed {
            payload: p,
            context,
        } = result
        {
            assert_eq!(p, payload);
            assert!(context.is_none());
        }
    }

    #[tokio::test]
    async fn test_no_hooks_for_event_returns_allowed() {
        // Build registry from empty config.
        let registry = HookRegistry::from_config(&[], test_http_client()).unwrap();
        let payload = serde_json::json!({"command": "web_search"});
        let result = registry
            .run_chain(HookEvent::PreToolUse, payload.clone(), Some("web_search"))
            .await;
        assert!(matches!(result, HookChainResult::Allowed { .. }));
    }

    // ── NullHookRunner ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_null_hook_runner_always_allows() {
        let runner = NullHookRunner;
        let payload = serde_json::json!({"command": "test"});
        let result = runner
            .run_chain(HookEvent::RequestStart, payload.clone(), None)
            .await;
        assert!(matches!(result, HookChainResult::Allowed { .. }));
        if let HookChainResult::Allowed {
            payload: p,
            context,
        } = result
        {
            assert_eq!(p, payload);
            assert!(context.is_none());
        }
    }

    #[tokio::test]
    async fn test_null_hook_runner_preserves_payload() {
        let runner = NullHookRunner;
        let payload = serde_json::json!({"messages": [{"role": "user", "content": "hello"}]});
        let result = runner
            .run_chain(HookEvent::PreToolUse, payload.clone(), Some("web_search"))
            .await;
        if let HookChainResult::Allowed { payload: p, .. } = result {
            assert_eq!(p, payload);
        } else {
            panic!("expected Allowed");
        }
    }

    // ── from_config ───────────────────────────────────────────────────────────

    #[test]
    fn test_from_config_with_invalid_matcher_regex() {
        let config = vec![HookConfig {
            event: HookEvent::PreToolUse,
            matcher: Some("[invalid".to_string()),
            hook_type: HookType::Http,
            script: None,
            url: Some("http://example.com/hook".to_string()),
            agent: None,
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let result = HookRegistry::from_config(&config, test_http_client());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_config_with_weft_type_skipped() {
        let config = vec![HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            hook_type: HookType::Weft,
            script: None,
            url: None,
            agent: Some("quality-checker".to_string()),
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let registry = HookRegistry::from_config(&config, test_http_client()).unwrap();
        // Weft hook skipped — no hooks registered.
        assert_eq!(registry.hook_count(HookEvent::PreToolUse), 0);
    }

    #[test]
    fn test_from_config_invalid_rhai_path_returns_error() {
        let config = vec![HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            hook_type: HookType::Rhai,
            script: Some("/nonexistent/hook.rhai".to_string()),
            url: None,
            agent: None,
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let result = HookRegistry::from_config(&config, test_http_client());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_config_rhai_hook_with_valid_script_registered() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"fn hook(e) { #{ decision: \"allow\" } }")
            .unwrap();
        let config = vec![HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            hook_type: HookType::Rhai,
            script: Some(f.path().to_str().unwrap().to_string()),
            url: None,
            agent: None,
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let registry = HookRegistry::from_config(&config, test_http_client()).unwrap();
        assert_eq!(registry.hook_count(HookEvent::PreToolUse), 1);
    }

    #[test]
    fn test_from_config_rhai_hook_with_syntax_error_fails() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"fn hook(e) { let x = ; }").unwrap(); // syntax error
        let config = vec![HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            hook_type: HookType::Rhai,
            script: Some(f.path().to_str().unwrap().to_string()),
            url: None,
            agent: None,
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let result = HookRegistry::from_config(&config, test_http_client());
        assert!(result.is_err(), "syntax error should fail construction");
    }

    #[tokio::test]
    async fn test_rhai_hook_executes_via_run_chain() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"fn hook(e) { #{ decision: \"block\", reason: \"blocked by rhai\" } }")
            .unwrap();
        let config = vec![HookConfig {
            event: HookEvent::RequestStart,
            matcher: None,
            hook_type: HookType::Rhai,
            script: Some(f.path().to_str().unwrap().to_string()),
            url: None,
            agent: None,
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let registry = HookRegistry::from_config(&config, test_http_client()).unwrap();
        let payload = serde_json::json!({"messages": []});
        let result = registry
            .run_chain(HookEvent::RequestStart, payload, None)
            .await;
        assert!(matches!(
            result,
            HookChainResult::Blocked { reason, .. } if reason == "blocked by rhai"
        ));
    }

    #[test]
    fn test_from_config_http_hook_registered() {
        let config = vec![HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            hook_type: HookType::Http,
            script: None,
            url: Some("http://example.com/hook".to_string()),
            agent: None,
            timeout_ms: None,
            secret: None,
            priority: 100,
        }];
        let registry = HookRegistry::from_config(&config, test_http_client()).unwrap();
        assert_eq!(registry.hook_count(HookEvent::PreToolUse), 1);
    }

    // ── Chain execution with inline executors ─────────────────────────────────

    /// A hook executor that always returns a fixed response.
    struct FixedExecutor(HookResponse);

    #[async_trait::async_trait]
    impl HookExecutor for FixedExecutor {
        async fn execute(&self, _payload: &serde_json::Value) -> HookResponse {
            self.0.clone()
        }
    }

    /// A hook executor that echoes modified payload from the response.
    struct ModifyExecutor {
        modified: serde_json::Value,
        context: Option<String>,
    }

    #[async_trait::async_trait]
    impl HookExecutor for ModifyExecutor {
        async fn execute(&self, _payload: &serde_json::Value) -> HookResponse {
            HookResponse {
                decision: HookDecision::Modify,
                reason: None,
                modified: Some(self.modified.clone()),
                context: self.context.clone(),
            }
        }
    }

    fn make_registry_with_hooks(
        event: HookEvent,
        hooks: Vec<(Box<dyn HookExecutor>, Option<String>, u32)>,
    ) -> HookRegistry {
        let registered: Vec<RegisteredHook> = hooks
            .into_iter()
            .enumerate()
            .map(|(i, (executor, matcher_pattern, priority))| {
                let matcher = HookMatcher::new(matcher_pattern.as_deref(), i).unwrap();
                RegisteredHook {
                    event,
                    matcher,
                    executor,
                    name: format!("test-hook-{i}"),
                    priority,
                }
            })
            .collect();

        let mut hooks_map: HashMap<HookEvent, Vec<RegisteredHook>> = HashMap::new();
        if !registered.is_empty() {
            hooks_map.insert(event, registered);
        }
        HookRegistry { hooks: hooks_map }
    }

    #[tokio::test]
    async fn test_single_allow_hook() {
        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![(Box::new(FixedExecutor(HookResponse::allow())), None, 100)],
        );
        let payload = serde_json::json!({"command": "test"});
        let result = registry
            .run_chain(HookEvent::PreToolUse, payload.clone(), Some("test"))
            .await;
        assert!(matches!(result, HookChainResult::Allowed { .. }));
    }

    #[tokio::test]
    async fn test_single_block_hook_on_blocking_event() {
        let registry = make_registry_with_hooks(
            HookEvent::RequestStart,
            vec![(
                Box::new(FixedExecutor(HookResponse::block("denied"))),
                None,
                100,
            )],
        );
        let payload = serde_json::json!({"message": "hello"});
        let result = registry
            .run_chain(HookEvent::RequestStart, payload, None)
            .await;
        assert!(matches!(
            result,
            HookChainResult::Blocked { reason, .. } if reason == "denied"
        ));
    }

    #[tokio::test]
    async fn test_block_on_non_blocking_event_treated_as_allow() {
        // PostToolUse cannot block.
        let registry = make_registry_with_hooks(
            HookEvent::PostToolUse,
            vec![(
                Box::new(FixedExecutor(HookResponse::block("denied"))),
                None,
                100,
            )],
        );
        let payload = serde_json::json!({"command": "test"});
        let result = registry
            .run_chain(HookEvent::PostToolUse, payload.clone(), Some("test"))
            .await;
        // Block on non-blocking event is treated as Allow.
        assert!(matches!(result, HookChainResult::Allowed { .. }));
    }

    #[tokio::test]
    async fn test_modify_hook_passes_modified_payload_to_next() {
        // Hook 1: Modify, sets payload to {"modified": true}.
        // Hook 2: Capture what it receives.
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None::<serde_json::Value>));
        let captured_clone = std::sync::Arc::clone(&captured);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        #[async_trait::async_trait]
        impl HookExecutor for Capture {
            async fn execute(&self, payload: &serde_json::Value) -> HookResponse {
                *self.0.lock().unwrap() = Some(payload.clone());
                HookResponse::allow()
            }
        }

        let modified_payload = serde_json::json!({"modified": true});
        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![
                (
                    Box::new(ModifyExecutor {
                        modified: modified_payload.clone(),
                        context: None,
                    }),
                    None,
                    50,
                ),
                (Box::new(Capture(captured_clone)), None, 100),
            ],
        );

        let original = serde_json::json!({"original": true});
        let result = registry
            .run_chain(HookEvent::PreToolUse, original, Some("test"))
            .await;

        assert!(matches!(result, HookChainResult::Allowed { .. }));
        // The second hook received the modified payload.
        let seen = captured.lock().unwrap().clone();
        assert_eq!(seen, Some(modified_payload));
    }

    #[tokio::test]
    async fn test_modify_without_payload_treated_as_allow() {
        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![(
                Box::new(FixedExecutor(HookResponse {
                    decision: HookDecision::Modify,
                    reason: None,
                    modified: None, // No modified payload!
                    context: None,
                })),
                None,
                100,
            )],
        );
        let original = serde_json::json!({"original": true});
        let result = registry
            .run_chain(HookEvent::PreToolUse, original.clone(), Some("test"))
            .await;
        // Treated as Allow — original payload preserved.
        if let HookChainResult::Allowed { payload, .. } = result {
            assert_eq!(payload, original);
        } else {
            panic!("expected Allowed");
        }
    }

    #[tokio::test]
    async fn test_matcher_skips_non_matching_hook() {
        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![(
                Box::new(FixedExecutor(HookResponse::block("denied"))),
                Some("web_search".to_string()),
                100,
            )],
        );
        let payload = serde_json::json!({"command": "calculator"});
        // Target is "calculator" — does not match "web_search" matcher.
        let result = registry
            .run_chain(HookEvent::PreToolUse, payload, Some("calculator"))
            .await;
        // Hook skipped — result is Allowed.
        assert!(matches!(result, HookChainResult::Allowed { .. }));
    }

    #[tokio::test]
    async fn test_matcher_fires_for_matching_hook() {
        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![(
                Box::new(FixedExecutor(HookResponse::block("denied"))),
                Some("web_search".to_string()),
                100,
            )],
        );
        let payload = serde_json::json!({"command": "web_search"});
        let result = registry
            .run_chain(HookEvent::PreToolUse, payload, Some("web_search"))
            .await;
        assert!(matches!(result, HookChainResult::Blocked { .. }));
    }

    #[tokio::test]
    async fn test_context_accumulates_across_hooks() {
        let make_allow_with_context = |ctx: &str| -> Box<dyn HookExecutor> {
            Box::new(FixedExecutor(HookResponse {
                decision: HookDecision::Allow,
                reason: None,
                modified: None,
                context: Some(ctx.to_string()),
            }))
        };

        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![
                (make_allow_with_context("ctx1"), None, 50),
                (make_allow_with_context("ctx2"), None, 100),
            ],
        );

        let payload = serde_json::json!({"command": "test"});
        let result = registry
            .run_chain(HookEvent::PreToolUse, payload, Some("test"))
            .await;

        if let HookChainResult::Allowed { context, .. } = result {
            let ctx = context.unwrap();
            // Both context strings should be present.
            assert!(ctx.contains("ctx1"));
            assert!(ctx.contains("ctx2"));
        } else {
            panic!("expected Allowed");
        }
    }

    #[tokio::test]
    async fn test_block_without_reason_uses_fallback() {
        let registry = make_registry_with_hooks(
            HookEvent::RequestStart,
            vec![(
                Box::new(FixedExecutor(HookResponse {
                    decision: HookDecision::Block,
                    reason: None, // No reason!
                    modified: None,
                    context: None,
                })),
                None,
                100,
            )],
        );
        let payload = serde_json::json!({});
        let result = registry
            .run_chain(HookEvent::RequestStart, payload, None)
            .await;
        if let HookChainResult::Blocked { reason, .. } = result {
            assert_eq!(reason, "hook blocked without reason");
        } else {
            panic!("expected Blocked");
        }
    }

    // ── Context truncation ────────────────────────────────────────────────────

    #[test]
    fn test_truncate_bytes_short_string() {
        let s = "hello";
        assert_eq!(truncate_bytes(s, 100), "hello");
    }

    #[test]
    fn test_truncate_bytes_exact_length() {
        let s = "hello";
        assert_eq!(truncate_bytes(s, 5), "hello");
    }

    #[test]
    fn test_truncate_bytes_over_limit() {
        let s = "hello world";
        let t = truncate_bytes(s, 5);
        assert_eq!(t.len(), 5);
        assert_eq!(t, "hello");
    }

    #[test]
    fn test_truncate_bytes_with_suffix() {
        let s = "hello world";
        let t = truncate_bytes_with_suffix(s, 10, "...");
        // Must be at most 10 bytes total.
        assert!(t.len() <= 10);
        assert!(t.ends_with("..."));
    }

    #[test]
    fn test_truncate_bytes_with_suffix_no_truncation_needed() {
        let s = "hi";
        let t = truncate_bytes_with_suffix(s, 100, "...[truncated]");
        assert_eq!(t, "hi");
    }

    // ── Priority ordering ──────────────────────────────────────────────────────

    #[test]
    fn test_hooks_sorted_by_priority_ascending() {
        // Register hooks with priorities 200, 50, 100 (in that order) via from_config.
        // After sorting, the hooks should appear in ascending priority order: 50, 100, 200.
        let configs = vec![
            HookConfig {
                event: HookEvent::RequestStart,
                matcher: None,
                hook_type: HookType::Http,
                script: None,
                url: Some("http://example.com/hook-200".to_string()),
                agent: None,
                timeout_ms: None,
                secret: None,
                priority: 200,
            },
            HookConfig {
                event: HookEvent::RequestStart,
                matcher: None,
                hook_type: HookType::Http,
                script: None,
                url: Some("http://example.com/hook-50".to_string()),
                agent: None,
                timeout_ms: None,
                secret: None,
                priority: 50,
            },
            HookConfig {
                event: HookEvent::RequestStart,
                matcher: None,
                hook_type: HookType::Http,
                script: None,
                url: Some("http://example.com/hook-100".to_string()),
                agent: None,
                timeout_ms: None,
                secret: None,
                priority: 100,
            },
        ];

        let registry = HookRegistry::from_config(&configs, test_http_client()).unwrap();

        // The hooks for RequestStart should be sorted ascending: 50, 100, 200.
        let hooks_for_event = registry.hooks.get(&HookEvent::RequestStart).unwrap();
        let priorities: Vec<u32> = hooks_for_event.iter().map(|h| h.priority).collect();
        assert_eq!(priorities, vec![50, 100, 200]);
    }

    #[tokio::test]
    async fn test_total_context_capped_at_4096() {
        // Create a context string that would exceed 4096 bytes when combined.
        let large_ctx = "x".repeat(2000);

        struct LargeContext(String);
        #[async_trait::async_trait]
        impl HookExecutor for LargeContext {
            async fn execute(&self, _payload: &serde_json::Value) -> HookResponse {
                HookResponse {
                    decision: HookDecision::Allow,
                    reason: None,
                    modified: None,
                    context: Some(self.0.clone()),
                }
            }
        }

        let registry = make_registry_with_hooks(
            HookEvent::PreToolUse,
            vec![
                (Box::new(LargeContext(large_ctx.clone())), None, 50),
                (Box::new(LargeContext(large_ctx.clone())), None, 100),
                (Box::new(LargeContext(large_ctx.clone())), None, 150),
            ],
        );

        let payload = serde_json::json!({});
        let result = registry
            .run_chain(HookEvent::PreToolUse, payload, Some("test"))
            .await;

        if let HookChainResult::Allowed { context, .. } = result {
            let ctx = context.unwrap();
            // Total context must be at most 4096 bytes.
            assert!(
                ctx.len() <= 4096,
                "context exceeded 4096 bytes: {}",
                ctx.len()
            );
        } else {
            panic!("expected Allowed");
        }
    }
}
