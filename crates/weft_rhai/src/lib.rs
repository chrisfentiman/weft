//! `weft_rhai` — Shared Rhai scripting infrastructure for Weft.
//!
//! Provides sandboxed engine construction, script compilation, safe execution,
//! and Dynamic <-> JSON conversion. Domain-specific logic stays in consumers.
//!
//! This crate has no domain logic. It knows nothing about hooks, providers, or
//! gateway concepts. It operates on `serde_json::Value` and `rhai::Dynamic`.
//!
//! # Typical usage pattern
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use weft_rhai::{EngineBuilder, SandboxLimits, CompiledScript, safe_call_fn, json_to_dynamic};
//!
//! # async fn example() -> Result<(), weft_rhai::ScriptError> {
//! // 1. Build a sandboxed engine with the desired preset.
//! let mut engine = EngineBuilder::new(SandboxLimits::strict(100))
//!     .log_source("rhai_hook")
//!     .with_time_helpers(true)
//!     .build();
//!
//! // 2. Register domain-specific functions on the engine (if any).
//! // engine.register_fn("my_fn", |x: i64| x * 2);
//!
//! // 3. Wrap in Arc for sharing into spawn_blocking closures.
//! let engine = Arc::new(engine);
//!
//! // 4. Load and compile the script.
//! let script = CompiledScript::load("hook.rhai", &engine)?;
//! script.validate_functions(&["hook"])?;
//!
//! // 5. Convert input payload to Dynamic.
//! let payload = json_to_dynamic(&serde_json::json!({"command": "test"}));
//!
//! // 6. Call the script function safely.
//! let result = safe_call_fn(engine, &script, "hook", (payload,)).await?;
//! # Ok(())
//! # }
//! ```

pub mod convert;
pub mod engine;
pub mod error;
pub mod exec;
pub mod script;

// Re-export the main types at crate root for ergonomic imports.
pub use convert::{dynamic_to_json, json_to_dynamic};
pub use engine::{EngineBuilder, SandboxLimits};
pub use error::ScriptError;
pub use exec::{CallResult, safe_call_fn};
pub use script::CompiledScript;

// Re-export rhai types that consumers need for interop.
// This avoids consumers needing a direct `rhai` dependency for basic usage.
pub use rhai::{Dynamic, Engine, Scope};
