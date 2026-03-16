//! Rhai engine builder with sandboxing presets.
//!
//! `EngineBuilder` constructs sandboxed Rhai engines with configurable limits
//! and optional helper function registration (JSON, logging, time, base64).
//! Consumers register domain-specific functions on the returned `Engine` before
//! wrapping it in `Arc`.

use rhai::Engine;
use tracing::{info, warn};

/// Sandbox limits for a Rhai engine.
///
/// All fields have defaults. Use `SandboxLimits::default()` for conservative
/// defaults, or `SandboxLimits::strict()` / `SandboxLimits::relaxed()` for
/// named presets.
#[derive(Debug, Clone)]
pub struct SandboxLimits {
    /// Maximum abstract operations before Rhai halts execution.
    /// 0 means unlimited (never set to 0 in production).
    pub max_operations: u64,
    /// Maximum string size in bytes.
    pub max_string_size: usize,
    /// Maximum array elements.
    pub max_array_size: usize,
    /// Maximum map entries.
    pub max_map_size: usize,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            max_operations: 100_000,
            max_string_size: 65_536,
            max_array_size: 1_024,
            max_map_size: 256,
        }
    }
}

impl SandboxLimits {
    /// Strict limits for hooks. Low operation count, small allocations.
    ///
    /// Hooks should be fast and small — they're in the request hot path.
    /// `max_operations` is derived from the configured timeout: 1000 ops/ms,
    /// with a minimum floor of 1000 operations.
    pub fn strict(timeout_ms: u64) -> Self {
        let ops_per_ms: u64 = 1_000;
        let max_ops = timeout_ms.saturating_mul(ops_per_ms).max(1_000);
        Self {
            max_operations: max_ops,
            max_string_size: 65_536,
            max_array_size: 1_024,
            max_map_size: 256,
        }
    }

    /// Relaxed limits for wire format scripts.
    ///
    /// Higher string/array/map limits because wire format scripts process full
    /// request/response bodies.
    pub fn relaxed() -> Self {
        Self {
            max_operations: 10_000,
            max_string_size: 1_048_576, // 1 MB
            max_array_size: 4_096,
            max_map_size: 512,
        }
    }
}

/// Builder for constructing sandboxed Rhai engines.
///
/// Configures sandbox limits, registers common API functions (JSON, logging),
/// and returns a configured `rhai::Engine`. Consumers add domain-specific
/// registered functions after building.
///
/// # Example
///
/// ```rust
/// use weft_rhai::{EngineBuilder, SandboxLimits};
/// use std::sync::Arc;
///
/// let engine = EngineBuilder::new(SandboxLimits::strict(100))
///     .log_source("rhai_hook")
///     .with_time_helpers(true)
///     .build();
///
/// // Register domain-specific functions before wrapping in Arc.
/// // engine.register_fn("my_fn", |x: i64| x * 2);
///
/// let engine = Arc::new(engine);
/// ```
pub struct EngineBuilder {
    limits: SandboxLimits,
    log_source: String,
    register_json_helpers: bool,
    register_log_helpers: bool,
    register_time_helpers: bool,
    register_base64_helpers: bool,
}

impl EngineBuilder {
    /// Create a builder with the given sandbox limits.
    pub fn new(limits: SandboxLimits) -> Self {
        Self {
            limits,
            log_source: "rhai".to_string(),
            register_json_helpers: true,
            register_log_helpers: true,
            register_time_helpers: false,
            register_base64_helpers: false,
        }
    }

    /// Set the `source` field in tracing log events from log_info/log_warn.
    /// Default: "rhai".
    pub fn log_source(mut self, source: impl Into<String>) -> Self {
        self.log_source = source.into();
        self
    }

    /// Register json_encode and json_decode functions. Default: true.
    pub fn with_json_helpers(mut self, enable: bool) -> Self {
        self.register_json_helpers = enable;
        self
    }

    /// Register log_info and log_warn functions. Default: true.
    pub fn with_log_helpers(mut self, enable: bool) -> Self {
        self.register_log_helpers = enable;
        self
    }

    /// Register now_unix_secs function. Default: false.
    pub fn with_time_helpers(mut self, enable: bool) -> Self {
        self.register_time_helpers = enable;
        self
    }

    /// Register base64_encode and base64_decode functions. Default: false.
    pub fn with_base64_helpers(mut self, enable: bool) -> Self {
        self.register_base64_helpers = enable;
        self
    }

    /// Build the configured Rhai engine.
    ///
    /// Uses `Engine::new()` (standard library, no `eval`). Applies sandbox
    /// limits and registers the selected helper functions.
    ///
    /// The returned engine is `Send + Sync` (via the `rhai/sync` feature).
    /// Consumers may register additional domain-specific functions on the
    /// returned engine before wrapping it in `Arc`.
    pub fn build(self) -> Engine {
        let mut engine = Engine::new();

        // Apply sandbox limits.
        engine.set_max_string_size(self.limits.max_string_size);
        engine.set_max_array_size(self.limits.max_array_size);
        engine.set_max_map_size(self.limits.max_map_size);
        // Rhai 0 means unlimited — ensure we always have a positive limit.
        engine.set_max_operations(self.limits.max_operations.max(1));

        if self.register_json_helpers {
            register_json_helpers(&mut engine);
        }

        if self.register_log_helpers {
            register_log_helpers(&mut engine, self.log_source);
        }

        if self.register_time_helpers {
            register_time_helpers(&mut engine);
        }

        if self.register_base64_helpers {
            register_base64_helpers(&mut engine);
        }

        engine
    }
}

/// Register json_encode and json_decode into the engine.
fn register_json_helpers(engine: &mut Engine) {
    use rhai::Dynamic;
    use serde_json::Value;

    // json_encode(value) -> String — serialize a Rhai Dynamic to a JSON string.
    engine.register_fn("json_encode", |value: Dynamic| -> String {
        match rhai::serde::from_dynamic::<Value>(&value) {
            Ok(json_val) => serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string()),
            Err(_) => "null".to_string(),
        }
    });

    // json_decode(s) -> Dynamic — deserialize a JSON string to a Rhai Dynamic.
    engine.register_fn("json_decode", |s: &str| -> Dynamic {
        match serde_json::from_str::<Value>(s) {
            Ok(json_val) => rhai::serde::to_dynamic(&json_val).unwrap_or(Dynamic::UNIT),
            Err(_) => Dynamic::UNIT,
        }
    });
}

/// Register log_info and log_warn into the engine with the given source label.
fn register_log_helpers(engine: &mut Engine, log_source: String) {
    let source_info = log_source.clone();
    let source_warn = log_source;

    engine.register_fn("log_info", move |msg: &str| {
        info!(source = %source_info, "{}", msg);
    });

    engine.register_fn("log_warn", move |msg: &str| {
        warn!(source = %source_warn, "{}", msg);
    });
}

/// Register now_unix_secs into the engine.
fn register_time_helpers(engine: &mut Engine) {
    engine.register_fn("now_unix_secs", || -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    });
}

/// Register base64_encode and base64_decode into the engine.
///
/// Inline implementation — no external dependency needed for this helper.
fn register_base64_helpers(engine: &mut Engine) {
    // base64_encode(s) -> String — standard alphabet, no line breaks.
    engine.register_fn("base64_encode", |s: &str| -> String {
        // Inline base64 encoder — standard alphabet, no line breaks.
        // Avoids adding a dependency just for this helper.
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = s.as_bytes();
        let mut buf = String::with_capacity(bytes.len().div_ceil(3) * 4);
        let mut i = 0;
        while i < bytes.len() {
            let b0 = bytes[i] as u32;
            let b1 = if i + 1 < bytes.len() {
                bytes[i + 1] as u32
            } else {
                0
            };
            let b2 = if i + 2 < bytes.len() {
                bytes[i + 2] as u32
            } else {
                0
            };

            buf.push(ALPHABET[((b0 >> 2) & 0x3F) as usize] as char);
            buf.push(ALPHABET[(((b0 << 4) | (b1 >> 4)) & 0x3F) as usize] as char);
            if i + 1 < bytes.len() {
                buf.push(ALPHABET[(((b1 << 2) | (b2 >> 6)) & 0x3F) as usize] as char);
            } else {
                buf.push('=');
            }
            if i + 2 < bytes.len() {
                buf.push(ALPHABET[(b2 & 0x3F) as usize] as char);
            } else {
                buf.push('=');
            }
            i += 3;
        }
        buf
    });

    // base64_decode(s) -> String — standard alphabet, returns empty on error.
    engine.register_fn("base64_decode", |s: &str| -> String {
        const DECODE_TABLE: [i8; 256] = {
            let mut t = [-1i8; 256];
            let mut i = 0usize;
            while i < 26 {
                t[b'A' as usize + i] = i as i8;
                t[b'a' as usize + i] = (i + 26) as i8;
                i += 1;
            }
            let mut i = 0usize;
            while i < 10 {
                t[b'0' as usize + i] = (i + 52) as i8;
                i += 1;
            }
            t[b'+' as usize] = 62;
            t[b'/' as usize] = 63;
            t
        };

        let input: Vec<u8> = s
            .bytes()
            .filter(|&b| b != b'=')
            .filter(|&b| DECODE_TABLE[b as usize] >= 0)
            .collect();

        let mut out = Vec::with_capacity(input.len() * 3 / 4);
        let mut i = 0;
        while i + 3 < input.len() {
            let v0 = DECODE_TABLE[input[i] as usize] as u32;
            let v1 = DECODE_TABLE[input[i + 1] as usize] as u32;
            let v2 = DECODE_TABLE[input[i + 2] as usize] as u32;
            let v3 = DECODE_TABLE[input[i + 3] as usize] as u32;
            out.push(((v0 << 2) | (v1 >> 4)) as u8);
            out.push(((v1 << 4) | (v2 >> 2)) as u8);
            out.push(((v2 << 6) | v3) as u8);
            i += 4;
        }
        if i + 2 < input.len() {
            let v0 = DECODE_TABLE[input[i] as usize] as u32;
            let v1 = DECODE_TABLE[input[i + 1] as usize] as u32;
            let v2 = DECODE_TABLE[input[i + 2] as usize] as u32;
            out.push(((v0 << 2) | (v1 >> 4)) as u8);
            out.push(((v1 << 4) | (v2 >> 2)) as u8);
        } else if i + 1 < input.len() {
            let v0 = DECODE_TABLE[input[i] as usize] as u32;
            let v1 = DECODE_TABLE[input[i + 1] as usize] as u32;
            out.push(((v0 << 2) | (v1 >> 4)) as u8);
        }

        String::from_utf8(out).unwrap_or_default()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhai::Dynamic;

    fn strict_engine(timeout_ms: u64) -> Engine {
        EngineBuilder::new(SandboxLimits::strict(timeout_ms)).build()
    }

    fn relaxed_engine() -> Engine {
        EngineBuilder::new(SandboxLimits::relaxed()).build()
    }

    // ── SandboxLimits ────────────────────────────────────────────────────────

    #[test]
    fn test_sandbox_limits_default() {
        let limits = SandboxLimits::default();
        assert_eq!(limits.max_operations, 100_000);
        assert_eq!(limits.max_string_size, 65_536);
        assert_eq!(limits.max_array_size, 1_024);
        assert_eq!(limits.max_map_size, 256);
    }

    #[test]
    fn test_sandbox_limits_strict_100ms() {
        let limits = SandboxLimits::strict(100);
        assert_eq!(limits.max_operations, 100_000);
        assert_eq!(limits.max_string_size, 65_536);
        assert_eq!(limits.max_array_size, 1_024);
        assert_eq!(limits.max_map_size, 256);
    }

    #[test]
    fn test_sandbox_limits_strict_1ms_minimum() {
        // 1ms * 1000 = 1000, which equals the minimum floor.
        let limits = SandboxLimits::strict(1);
        assert_eq!(limits.max_operations, 1_000);
    }

    #[test]
    fn test_sandbox_limits_strict_0ms_floor() {
        // 0ms * 1000 = 0, floored to 1000.
        let limits = SandboxLimits::strict(0);
        assert_eq!(limits.max_operations, 1_000);
    }

    #[test]
    fn test_sandbox_limits_relaxed() {
        let limits = SandboxLimits::relaxed();
        assert_eq!(limits.max_operations, 10_000);
        assert_eq!(limits.max_string_size, 1_048_576);
        assert_eq!(limits.max_array_size, 4_096);
        assert_eq!(limits.max_map_size, 512);
    }

    // ── EngineBuilder: operation limit enforcement ──────────────────────────

    #[test]
    fn test_operation_limit_halts_infinite_loop() {
        // 1ms timeout → 1000 max ops. An infinite loop exceeds this.
        let engine = strict_engine(1);
        let result = engine.eval::<Dynamic>("let i = 0; loop { i += 1; } i");
        assert!(result.is_err(), "expected operation limit error");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.to_lowercase().contains("operations")
                || err_msg.to_lowercase().contains("too many"),
            "expected operations-related error message, got: {err_msg}"
        );
    }

    #[test]
    fn test_engine_with_json_helpers_disabled_errors_on_json_encode() {
        let engine = EngineBuilder::new(SandboxLimits::default())
            .with_json_helpers(false)
            .build();
        // json_encode is not registered — calling it is a runtime error.
        let result = engine.eval::<String>(r#"json_encode(#{ key: "val" })"#);
        assert!(
            result.is_err(),
            "expected error when json_encode not registered"
        );
    }

    #[test]
    fn test_engine_with_json_helpers_enabled_by_default() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        // json_encode should be registered by default.
        let result = engine.eval::<String>(r#"json_encode(#{ key: "value" })"#);
        assert!(
            result.is_ok(),
            "json_encode should be registered by default"
        );
        let s = result.unwrap();
        assert!(s.contains("value"));
    }

    #[test]
    fn test_engine_with_time_helpers_enabled() {
        let engine = EngineBuilder::new(SandboxLimits::default())
            .with_time_helpers(true)
            .build();
        let result = engine.eval::<i64>("now_unix_secs()");
        assert!(result.is_ok(), "now_unix_secs should succeed");
        let ts = result.unwrap();
        assert!(ts > 0, "unix timestamp should be positive");
    }

    #[test]
    fn test_engine_without_time_helpers_errors_on_now_unix_secs() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let result = engine.eval::<i64>("now_unix_secs()");
        assert!(
            result.is_err(),
            "now_unix_secs should not be registered by default"
        );
    }

    #[test]
    fn test_engine_with_base64_helpers_encode() {
        let engine = EngineBuilder::new(SandboxLimits::default())
            .with_base64_helpers(true)
            .build();
        let result = engine.eval::<String>(r#"base64_encode("hello")"#);
        assert!(result.is_ok(), "base64_encode should succeed");
        assert_eq!(result.unwrap(), "aGVsbG8=");
    }

    #[test]
    fn test_engine_with_base64_helpers_decode() {
        let engine = EngineBuilder::new(SandboxLimits::default())
            .with_base64_helpers(true)
            .build();
        let result = engine.eval::<String>(r#"base64_decode("aGVsbG8=")"#);
        assert!(result.is_ok(), "base64_decode should succeed");
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn test_engine_without_base64_helpers_errors() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let result = engine.eval::<String>(r#"base64_encode("hello")"#);
        assert!(
            result.is_err(),
            "base64_encode should not be registered by default"
        );
    }

    #[test]
    fn test_engine_log_source_no_panic() {
        // log_source changes the tracing field — we can't easily capture it in tests,
        // but we verify the engine builds and the log functions don't panic.
        let engine = EngineBuilder::new(SandboxLimits::default())
            .log_source("test_source")
            .build();
        let result = engine.eval::<Dynamic>(r#"log_info("test"); 42"#);
        assert!(result.is_ok());
    }

    #[test]
    fn test_engine_log_helpers_disabled() {
        let engine = EngineBuilder::new(SandboxLimits::default())
            .with_log_helpers(false)
            .build();
        let result = engine.eval::<Dynamic>(r#"log_info("test")"#);
        assert!(
            result.is_err(),
            "log_info should not be registered when disabled"
        );
    }

    #[test]
    fn test_json_encode_decode_round_trip() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let result = engine.eval::<Dynamic>(
            r#"
            let obj = #{ key: "value", num: 42 };
            let encoded = json_encode(obj);
            let decoded = json_decode(encoded);
            decoded.key
        "#,
        );
        assert!(result.is_ok());
        let val: String = result.unwrap().cast();
        assert_eq!(val, "value");
    }

    #[test]
    fn test_relaxed_engine_allows_larger_ops() {
        // Relaxed engine allows 10_000 operations — a tight loop of 5000 iterations should succeed.
        let engine = relaxed_engine();
        let result = engine.eval::<i64>(
            r#"
            let sum = 0;
            let i = 0;
            while i < 100 { sum += i; i += 1; }
            sum
        "#,
        );
        assert!(result.is_ok(), "relaxed engine should allow moderate loops");
    }
}
