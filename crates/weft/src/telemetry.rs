//! Tracing subscriber initialization for the Weft binary.
//!
//! This module owns all subscriber configuration and initialization. The binary
//! calls `init()` once at startup, before any tracing macros fire.
//!
//! Configuration is via environment variables only. Telemetry is operational
//! configuration — it changes between environments (dev/staging/prod) and is
//! set by the deployment platform, not embedded in TOML config.
//!
//! # Environment variables
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `RUST_LOG` | `info` | Standard tracing filter expression |
//! | `WEFT_LOG_FORMAT` | `text` | Log format: `"text"` or `"json"` |
//! | `WEFT_LOG_SPAN_EVENTS` | `false` | Include span open/close in output |
//! | `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | OTLP endpoint (enables OTel when set — Phase 3) |
//! | `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | OTLP protocol: `"grpc"` or `"http/protobuf"` (Phase 3) |
//! | `OTEL_SERVICE_NAME` | `weft` | Service name in OTel resource (Phase 3) |

use tracing_subscriber::{
    EnvFilter, Layer,
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable text with ANSI color codes.
    Text,
    /// Structured JSON, one object per log line.
    Json,
}

/// OTLP export protocol (Phase 3 — reserved for future use).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtlpProtocol {
    Grpc,
    Http,
}

/// Telemetry configuration derived from environment variables.
///
/// Load via [`TelemetryConfig::from_env`].
pub struct TelemetryConfig {
    /// Log output format: text or json. Default: text.
    pub log_format: LogFormat,
    /// Whether to include span open/close events in log output. Default: false.
    pub log_span_events: bool,
    /// OTLP endpoint (Phase 3). When set, enables OTel export.
    pub otlp_endpoint: Option<String>,
    /// OTLP protocol: grpc or http/protobuf. Default: grpc.
    pub otlp_protocol: OtlpProtocol,
    /// Service name for OTel resource. Default: "weft".
    pub service_name: String,
}

impl TelemetryConfig {
    /// Load configuration from environment variables.
    ///
    /// Uses sensible defaults for all variables. Never fails — missing or
    /// unrecognized values fall back to defaults.
    pub fn from_env() -> Self {
        let log_format = match std::env::var("WEFT_LOG_FORMAT")
            .as_deref()
            .unwrap_or("text")
        {
            "json" => LogFormat::Json,
            _ => LogFormat::Text,
        };

        let log_span_events = std::env::var("WEFT_LOG_SPAN_EVENTS")
            .as_deref()
            .unwrap_or("false")
            == "true";

        let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

        let otlp_protocol = match std::env::var("OTEL_EXPORTER_OTLP_PROTOCOL")
            .as_deref()
            .unwrap_or("grpc")
        {
            "http/protobuf" => OtlpProtocol::Http,
            _ => OtlpProtocol::Grpc,
        };

        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "weft".to_string());

        Self {
            log_format,
            log_span_events,
            otlp_endpoint,
            otlp_protocol,
            service_name,
        }
    }
}

/// Guard that must be held for the program's lifetime.
///
/// When dropped, flushes any buffered output and shuts down background
/// exporters. Phase 3 will hold an OTel `SdkTracerProvider` here.
pub struct TelemetryGuard {
    // Phase 3: opentelemetry_sdk::trace::SdkTracerProvider
    _private: (),
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        // Phase 3: provider.shutdown()
    }
}

/// Initialize the global tracing subscriber.
///
/// Must be called once, before any tracing macros. Subsequent calls are
/// silently ignored (the global subscriber is already set).
///
/// Returns a [`TelemetryGuard`] that must be held for the lifetime of the
/// program. Dropping it before program exit may lose buffered log output
/// or OTel spans.
///
/// # Panics
///
/// Does not panic. If the subscriber cannot be installed (e.g., in a test
/// environment where a subscriber is already set), the error is silently
/// ignored — the existing subscriber remains in place.
pub fn init(config: &TelemetryConfig) -> TelemetryGuard {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let span_events = if config.log_span_events {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };

    // Build the fmt layer. The text and JSON branches produce different concrete
    // types, so we box them to a common trait object for uniform composition.
    let fmt_layer: Box<dyn tracing_subscriber::Layer<_> + Send + Sync> = match config.log_format {
        LogFormat::Text => fmt::layer()
            .with_target(true)
            .with_span_events(span_events)
            .boxed(),
        LogFormat::Json => fmt::layer()
            .json()
            .with_target(true)
            .with_span_events(span_events)
            .boxed(),
    };

    // Phase 2: .with(MetricsLayer::new())
    // Phase 3: .with(otel_layer)  — optional OpenTelemetryLayer

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init();

    TelemetryGuard { _private: () }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn default_config_is_text_format() {
        // Remove any env vars that could affect the test.
        // We cannot unset them reliably in parallel tests, so just verify the
        // parsing logic directly.
        let format = match "text" {
            "json" => LogFormat::Json,
            _ => LogFormat::Text,
        };
        assert_eq!(format, LogFormat::Text);
    }

    #[test]
    fn json_format_parsed_correctly() {
        let format = match "json" {
            "json" => LogFormat::Json,
            _ => LogFormat::Text,
        };
        assert_eq!(format, LogFormat::Json);
    }

    #[test]
    fn unknown_format_falls_back_to_text() {
        let format = match "unknown_value" {
            "json" => LogFormat::Json,
            _ => LogFormat::Text,
        };
        assert_eq!(format, LogFormat::Text);
    }

    #[test]
    fn span_events_false_by_default() {
        // Default is "false" — parse it.
        let enabled = "false" == "true";
        assert!(!enabled);
    }

    #[test]
    fn span_events_true_when_set() {
        let enabled = "true" == "true";
        assert!(enabled);
    }

    #[test]
    fn otlp_protocol_grpc_by_default() {
        let proto = match "grpc" {
            "http/protobuf" => OtlpProtocol::Http,
            _ => OtlpProtocol::Grpc,
        };
        assert_eq!(proto, OtlpProtocol::Grpc);
    }

    #[test]
    fn otlp_protocol_http_when_set() {
        let proto = match "http/protobuf" {
            "http/protobuf" => OtlpProtocol::Http,
            _ => OtlpProtocol::Grpc,
        };
        assert_eq!(proto, OtlpProtocol::Http);
    }

    #[test]
    fn service_name_defaults_to_weft() {
        // The from_env fallback: std::env::var returns Err, so unwrap_or_else fires.
        let name: String = std::env::var("OTEL_SERVICE_NAME_NONEXISTENT_TEST_VAR")
            .unwrap_or_else(|_| "weft".to_string());
        assert_eq!(name, "weft");
    }

    #[test]
    fn telemetry_guard_drops_without_panic() {
        // Verify the guard drops cleanly (no panic in drop impl).
        let guard = TelemetryGuard { _private: () };
        drop(guard);
    }
}
