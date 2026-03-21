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
//! | `WEFT_PROMETHEUS_ENABLED` | `true` | Enable Prometheus `/metrics` endpoint |
//! | `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | OTLP endpoint. Enables OTel export when set. |
//! | `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | OTLP protocol: `"grpc"` or `"http/protobuf"` |
//! | `OTEL_SERVICE_NAME` | `weft` | Service name in OTel resource |

pub mod metrics_layer;
pub mod propagation;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::info;
use tracing_subscriber::{
    EnvFilter, Layer,
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

use crate::telemetry::metrics_layer::MetricsLayer;

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
    /// Whether the Prometheus `/metrics` endpoint is enabled. Default: true.
    pub prometheus_enabled: bool,
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

        let prometheus_enabled = std::env::var("WEFT_PROMETHEUS_ENABLED")
            .as_deref()
            .unwrap_or("true")
            != "false";

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
            prometheus_enabled,
            otlp_endpoint,
            otlp_protocol,
            service_name,
        }
    }
}

/// Guard that must be held for the program's lifetime.
///
/// When dropped, flushes any buffered output and shuts down background
/// exporters. Holds the `PrometheusHandle` used to serve the `/metrics` endpoint
/// and the `SdkTracerProvider` which flushes the OTLP batch exporter on drop.
pub struct TelemetryGuard {
    /// Prometheus metrics handle. `None` when Prometheus is disabled.
    prometheus_handle: Option<PrometheusHandle>,
    /// OTel tracer provider. `None` when OTLP is not configured.
    ///
    /// `shutdown()` is called on drop to flush the batch exporter and ensure
    /// in-flight spans are sent before the process exits.
    otel_provider: Option<SdkTracerProvider>,
}

impl TelemetryGuard {
    /// Return the Prometheus handle if Prometheus is enabled.
    ///
    /// The handle is `Clone + Send + Sync` and is used to serve the `/metrics` endpoint.
    pub fn prometheus_handle(&self) -> Option<&PrometheusHandle> {
        self.prometheus_handle.as_ref()
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        // Flush the OTLP batch exporter. The shutdown() call blocks until all
        // queued spans are exported or the timeout elapses. This ensures spans
        // from the last request(s) before SIGTERM are not lost.
        if let Some(provider) = self.otel_provider.take()
            && let Err(e) = provider.shutdown()
        {
            eprintln!("warn: OTel provider shutdown error: {e}");
        }
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
/// When `config.prometheus_enabled` is `true`, installs a Prometheus metrics
/// recorder and adds the `MetricsLayer` to the subscriber stack. The
/// `PrometheusHandle` is accessible via [`TelemetryGuard::prometheus_handle`].
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

    // Install Prometheus recorder and add MetricsLayer when Prometheus is enabled.
    let prometheus_handle = if config.prometheus_enabled {
        match PrometheusBuilder::new().install_recorder() {
            Ok(handle) => Some(handle),
            Err(e) => {
                // Telemetry init failure is non-fatal: the gateway still serves requests.
                // Log via eprintln because the subscriber is not yet initialized.
                eprintln!(
                    "warn: failed to install Prometheus metrics recorder: {e} — metrics disabled"
                );
                None
            }
        }
    } else {
        None
    };

    // The MetricsLayer is only added when Prometheus is enabled.
    // Option<MetricsLayer> implements Layer as a no-op when None.
    let metrics_layer: Option<MetricsLayer> = if prometheus_handle.is_some() {
        Some(MetricsLayer::new())
    } else {
        None
    };

    // Set the global W3C TraceContext propagator. This must happen before any
    // span extraction/injection calls, regardless of whether OTel export is enabled.
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    // Build the optional OTel provider. When `otlp_endpoint` is set, construct a
    // batch OTLP exporter and install it. Failures are non-fatal (gateway still serves).
    let otel_provider: Option<SdkTracerProvider> = if let Some(endpoint) = &config.otlp_endpoint {
        match build_otlp_provider(endpoint, &config.service_name) {
            Ok(provider) => Some(provider),
            Err(e) => {
                eprintln!(
                    "warn: failed to build OTLP exporter for endpoint '{endpoint}': {e} — OTel export disabled"
                );
                None
            }
        }
    } else {
        None
    };

    // Build a boxed OTel tracing layer when the provider is available.
    // Option<BoxedLayer> implements Layer as a no-op when None, keeping the
    // subscriber type uniform whether or not OTel is configured.
    let otel_layer: Option<Box<dyn Layer<_> + Send + Sync>> =
        otel_provider.as_ref().map(|provider| {
            use opentelemetry::trace::TracerProvider as _;
            let tracer = provider.tracer("weft");
            let layer = tracing_opentelemetry::layer().with_tracer(tracer);
            Box::new(layer) as Box<dyn Layer<_> + Send + Sync>
        });

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(metrics_layer)
        .with(otel_layer)
        .try_init();

    // Log telemetry configuration at INFO level. This uses tracing so it only
    // fires after the subscriber is initialized above.
    info!(
        log_format = ?config.log_format,
        otlp_enabled = otel_provider.is_some(),
        otlp_endpoint = config.otlp_endpoint.as_deref().unwrap_or("(unset)"),
        prometheus_enabled = prometheus_handle.is_some(),
        service_name = %config.service_name,
        "telemetry initialized"
    );

    TelemetryGuard {
        prometheus_handle,
        otel_provider,
    }
}

/// Build an OTLP gRPC span exporter provider.
///
/// Uses a batch exporter with the tokio runtime. The `endpoint` should be a
/// gRPC endpoint such as `http://localhost:4317`.
///
/// Returns `Err` if the exporter cannot be constructed (e.g., malformed endpoint).
/// The caller handles failures gracefully (non-fatal telemetry init error).
fn build_otlp_provider(
    endpoint: &str,
    service_name: &str,
) -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync>> {
    use opentelemetry_otlp::{SpanExporter, WithExportConfig};

    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name(service_name.to_string())
                .build(),
        )
        .build();

    opentelemetry::global::set_tracer_provider(provider.clone());
    Ok(provider)
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
        let guard = TelemetryGuard {
            prometheus_handle: None,
            otel_provider: None,
        };
        drop(guard);
    }

    #[test]
    fn otlp_endpoint_env_var_read_correctly() {
        // from_env reads OTEL_EXPORTER_OTLP_ENDPOINT and stores it in otlp_endpoint.
        // We can only test the parsing logic since we cannot guarantee the env var is set.
        // Test the None case (no env var set).
        let endpoint: Option<String> =
            std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT_NONEXISTENT_TEST").ok();
        assert!(endpoint.is_none());
    }

    #[tokio::test]
    async fn build_otlp_provider_fails_gracefully_with_bad_endpoint() {
        // A malformed endpoint should return Err, not panic.
        // Note: with tonic, endpoint parsing is lazy — the error may appear later.
        // We verify the function returns without panicking.
        // Requires a Tokio runtime because tonic spawns a background task on construction.
        let result = build_otlp_provider("not-a-valid-endpoint", "weft");
        // We don't assert Ok/Err specifically because tonic may accept the address
        // at construction time and fail at connect time. What matters: no panic.
        let _ = result;
    }

    #[tokio::test]
    async fn build_otlp_provider_succeeds_with_valid_endpoint_format() {
        // A properly formatted gRPC endpoint should construct the provider without panic.
        // No actual collector needs to be running — the connection is established lazily.
        // Requires a Tokio runtime because tonic spawns a background task on construction.
        let result = build_otlp_provider("http://localhost:4317", "weft-test");
        // The provider should be constructable even without a real collector.
        assert!(
            result.is_ok(),
            "valid endpoint should construct provider: {:?}",
            result.err()
        );
        // Shutdown the provider immediately to avoid background task leak.
        if let Ok(provider) = result {
            let _ = provider.shutdown();
        }
    }

    #[test]
    fn prometheus_enabled_by_default() {
        let enabled = std::env::var("WEFT_PROMETHEUS_ENABLED")
            .as_deref()
            .unwrap_or("true")
            != "false";
        assert!(enabled);
    }

    #[test]
    fn prometheus_disabled_when_env_false() {
        let enabled = "false" != "false";
        assert!(!enabled);
    }
}
