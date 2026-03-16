//! Safe Rhai function execution.
//!
//! `safe_call_fn` is the canonical way to execute Rhai scripts in an async
//! context. It combines `tokio::task::spawn_blocking` (Rhai is CPU-bound) with
//! `std::panic::catch_unwind` (defense-in-depth against Rhai internal panics).

use std::sync::Arc;

use rhai::{Dynamic, Engine, FuncArgs, Scope};

use crate::error::ScriptError;
use crate::script::CompiledScript;

/// Result of a safe Rhai function call.
///
/// On success, contains the `Dynamic` return value.
/// On failure, contains a `ScriptError` covering all failure modes
/// (runtime error, panic, operation limit, task join failure).
pub type CallResult = Result<Dynamic, ScriptError>;

/// Call a Rhai function safely on a blocking thread with panic protection.
///
/// This is the canonical way to execute Rhai scripts in an async context.
/// It handles:
/// 1. `tokio::task::spawn_blocking` — Rhai is CPU-bound, must not block tokio workers.
/// 2. `std::panic::catch_unwind` — defense-in-depth against Rhai internal panics.
/// 3. Fresh `Scope` per call — no state bleeds between invocations.
///
/// # Arguments
///
/// - `engine`: Shared engine reference (cloned for the blocking task).
/// - `script`: Compiled script (AST cloned into the blocking task).
/// - `fn_name`: Name of the Rhai function to call.
/// - `args`: Arguments tuple. Must implement `rhai::FuncArgs`.
///
/// # Type Parameter
///
/// The `A` parameter must implement `rhai::FuncArgs + Send + 'static`.
/// Common argument types: `(Dynamic,)` for single-argument functions.
///
/// # Errors
///
/// Returns `ScriptError::ExecutionError` for Rhai runtime errors.
/// Returns `ScriptError::Panic` for caught panics.
/// Returns `ScriptError::TaskJoinError` for `spawn_blocking` join failures.
///
/// # Example
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use weft_rhai::{EngineBuilder, SandboxLimits, CompiledScript, safe_call_fn, Dynamic};
/// # async fn example() {
/// let engine = Arc::new(EngineBuilder::new(SandboxLimits::strict(100)).build());
/// let script = CompiledScript::load("hook.rhai", &engine).unwrap();
/// let payload = Dynamic::from(42_i64);
/// let result = safe_call_fn(engine, &script, "hook", (payload,)).await;
/// # }
/// ```
pub async fn safe_call_fn<A>(
    engine: Arc<Engine>,
    script: &CompiledScript,
    fn_name: &str,
    args: A,
) -> CallResult
where
    A: FuncArgs + Send + 'static,
{
    let path = script.path().to_string();
    let fn_name = fn_name.to_string();
    let ast = Arc::clone(script.ast());

    let result = tokio::task::spawn_blocking(move || {
        // SAFETY: We catch panics here for defense-in-depth against Rhai internals.
        // The engine's immutable config (registered functions, sandbox limits) is safe
        // to access concurrently from multiple tasks — `Engine` is Send + Sync with
        // the `rhai/sync` feature. Each call creates a fresh Scope so there is no
        // shared mutable state. After a caught panic, the engine itself remains valid
        // for subsequent calls because panics in Rhai typically originate in script
        // execution, not engine mutations.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut scope = Scope::new();
            engine.call_fn::<Dynamic>(&mut scope, &ast, &fn_name, args)
        }))
    })
    .await;

    match result {
        Err(join_err) => Err(ScriptError::TaskJoinError {
            path,
            message: join_err.to_string(),
        }),
        Ok(Err(panic_payload)) => {
            // catch_unwind caught a panic inside the Rhai execution.
            let panic_msg = panic_payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                .unwrap_or("unknown panic");
            Err(ScriptError::Panic {
                path,
                message: panic_msg.to_string(),
            })
        }
        Ok(Ok(Err(rhai_err))) => Err(ScriptError::ExecutionError {
            path,
            message: rhai_err.to_string(),
        }),
        Ok(Ok(Ok(dynamic))) => Ok(dynamic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineBuilder, SandboxLimits};
    use std::io::Write;

    fn write_temp_script(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    fn load_script(content: &str, engine: &Engine) -> CompiledScript {
        let f = write_temp_script(content);
        CompiledScript::load(f.path().to_str().unwrap(), engine).unwrap()
    }

    // ── Successful execution ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_safe_call_fn_returns_correct_value() {
        let engine = Arc::new(EngineBuilder::new(SandboxLimits::default()).build());
        let script = load_script("fn add(x) { x + 10 }", &engine);
        let result = safe_call_fn(engine, &script, "add", (Dynamic::from(5_i64),)).await;
        assert!(result.is_ok(), "expected success: {result:?}");
        let val: i64 = result.unwrap().cast();
        assert_eq!(val, 15);
    }

    #[tokio::test]
    async fn test_safe_call_fn_string_return() {
        let engine = Arc::new(EngineBuilder::new(SandboxLimits::default()).build());
        let script = load_script(r#"fn greet(name) { "hello, " + name }"#, &engine);
        let result = safe_call_fn(
            engine,
            &script,
            "greet",
            (Dynamic::from("world".to_string()),),
        )
        .await;
        assert!(result.is_ok());
        let s: String = result.unwrap().cast();
        assert_eq!(s, "hello, world");
    }

    #[tokio::test]
    async fn test_safe_call_fn_map_return() {
        let engine = Arc::new(EngineBuilder::new(SandboxLimits::default()).build());
        let script = load_script(r#"fn run(x) { #{ value: x, doubled: x * 2 } }"#, &engine);
        let result = safe_call_fn(engine, &script, "run", (Dynamic::from(21_i64),)).await;
        assert!(result.is_ok());
        // Just verify it's not unit.
        assert!(!result.unwrap().is_unit());
    }

    // ── Runtime errors ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_safe_call_fn_throw_returns_execution_error() {
        let engine = Arc::new(EngineBuilder::new(SandboxLimits::default()).build());
        let script = load_script(r#"fn run(x) { throw "intentional error"; }"#, &engine);
        let result = safe_call_fn(engine, &script, "run", (Dynamic::UNIT,)).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ScriptError::ExecutionError { .. }),
            "expected ExecutionError"
        );
    }

    #[tokio::test]
    async fn test_safe_call_fn_infinite_loop_returns_execution_error() {
        // 1ms timeout → 1000 max ops. An infinite loop will exceed this.
        let engine = Arc::new(
            EngineBuilder::new(SandboxLimits::strict(1))
                .with_time_helpers(false)
                .build(),
        );
        let script = load_script(r#"fn run(x) { let i = 0; loop { i += 1; } i }"#, &engine);
        let result = safe_call_fn(engine, &script, "run", (Dynamic::UNIT,)).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ScriptError::ExecutionError { .. }),
            "expected ExecutionError for operation limit"
        );
    }

    // ── Async boundary ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_safe_call_fn_crosses_spawn_blocking_boundary() {
        // Verifies that owned data round-trips correctly through spawn_blocking.
        let engine = Arc::new(EngineBuilder::new(SandboxLimits::default()).build());
        let script = load_script(r#"fn echo(x) { x }"#, &engine);
        let input = Dynamic::from(12345_i64);
        let result = safe_call_fn(engine, &script, "echo", (input,)).await;
        assert!(result.is_ok());
        let val: i64 = result.unwrap().cast();
        assert_eq!(val, 12345);
    }

    // ── Error path: script path is preserved ────────────────────────────────

    #[tokio::test]
    async fn test_safe_call_fn_error_contains_script_path() {
        let engine = Arc::new(EngineBuilder::new(SandboxLimits::default()).build());
        let script = load_script(r#"fn run(x) { throw "oops"; }"#, &engine);
        let path = script.path().to_string();
        let result = safe_call_fn(engine, &script, "run", (Dynamic::UNIT,)).await;
        let err = result.unwrap_err();
        assert_eq!(
            err.script_path(),
            Some(path.as_str()),
            "error should contain script path"
        );
    }
}
