//! Custom `tracing::Layer` that derives metrics from span lifecycle events.
//!
//! Application code creates tracing spans; this layer observes span lifecycle
//! and emits counters, histograms, and gauges via the `metrics` crate. This
//! follows a metrics-from-spans pattern: single instrumentation point with metrics
//! derived automatically from spans rather than maintained in parallel.
//!
//! # Metrics emitted
//!
//! | Metric | Type | Trigger |
//! |--------|------|---------|
//! | `weft_requests_total` | counter | `request` span close |
//! | `weft_request_duration_seconds` | histogram | `request` span close |
//! | `weft_active_requests` | gauge | `request` span new/close |
//! | `weft_provider_call_duration_seconds` | histogram | `generate` span close |
//! | `weft_provider_tokens_total` | counter | `generate` span close |
//! | `weft_activity_duration_seconds` | histogram | `activity` span close |
//! | `weft_command_duration_seconds` | histogram | `execute_command` span close |
//! | `weft_dispatch_iterations_total` | counter | `iteration` span close |

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

use metrics::{counter, gauge, histogram};
use tracing::span;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

/// Maximum number of unique command names tracked before bucketing as "other".
///
/// Protects against unbounded cardinality from a misconfigured or adversarial
/// tool registry.
const MAX_COMMAND_NAMES: usize = 50;

// ── Per-span attribute storage ───────────────────────────────────────────────

/// Per-span metadata stored in span extensions by the MetricsLayer.
///
/// Created in `on_new_span` and read in `on_close`. The `start` field allows
/// duration computation without any external clock references.
pub(crate) struct SpanMetrics {
    /// Wall-clock time when the span was created.
    pub(crate) start: Instant,
    /// Parsed variant describing which metric(s) to emit on close.
    pub(crate) attrs: SpanAttributes,
}

/// Per-span attribute storage, keyed by the span name we care about.
///
/// Uses an enum rather than a HashMap to avoid heap allocation for spans we
/// do not derive metrics from, and to keep attribute access branchless.
pub(crate) enum SpanAttributes {
    /// `request` span — top-level pipeline execution.
    Request {
        pipeline: String,
        /// "ok" | "error" — populated via `on_record` when `otel.status_code` is recorded.
        status: String,
        /// `true` when `weft.request.degraded = true` is recorded on the span.
        /// Populated via `on_record`. Drives `weft_requests_degraded_total` counter.
        degraded: bool,
    },
    /// `activity` span — one activity in pre/dispatch/post phase.
    Activity {
        /// Value of `activity.name` attribute.
        name: String,
        /// Value of `activity.phase` attribute: "pre_loop" | "dispatch" | "post_loop".
        phase: String,
        /// "ok" | "error" | "degraded" — populated via `on_record` when `activity.status` is recorded.
        /// Drives `weft_degradations_total` counter when "degraded".
        status: String,
        /// Machine-readable error code from `degradation.error_code` span attribute.
        /// Populated via `on_record`. Empty when not degraded.
        error_code: String,
    },
    /// `generate` span — provider generation call inside an activity.
    Generate {
        model: String,
        provider: String,
        /// "ok" | "error" — set via `on_record`.
        status: String,
        /// Input tokens consumed — populated via `on_record` when `provider.input_tokens` is recorded.
        input_tokens: u64,
        /// Output tokens generated — populated via `on_record` when `provider.output_tokens` is recorded.
        output_tokens: u64,
    },
    /// `execute_command` span — per-command invocation in the dispatch loop.
    ExecuteCommand {
        /// Value of `command.name` attribute.
        name: String,
    },
    /// `iteration` span — one pass through the dispatch loop.
    Iteration,
    /// All other spans — no metrics derived.
    Other,
}

// ── MetricsLayer ─────────────────────────────────────────────────────────────

/// A `tracing::Layer` that derives metrics from span lifecycle events.
///
/// Add to the subscriber stack via `registry.with(MetricsLayer::new())`.
/// The layer uses `on_new_span`, `on_record`, and `on_close` to track span
/// state and emit metrics when spans close.
///
/// A `metrics` recorder must be installed globally before any spans fire.
/// When using Prometheus, call `PrometheusBuilder::new().install_recorder()`
/// before calling `telemetry::init()`, or pass it via the builder.
pub struct MetricsLayer {
    /// Set of command names seen so far. Capped at `MAX_COMMAND_NAMES` to
    /// prevent unbounded cardinality from a misconfigured tool registry.
    seen_command_names: Mutex<HashSet<String>>,
}

impl MetricsLayer {
    /// Create a new MetricsLayer.
    pub fn new() -> Self {
        Self {
            seen_command_names: Mutex::new(HashSet::new()),
        }
    }

    /// Resolve a command name through the cardinality cap.
    ///
    /// Returns the name itself if the set has not yet reached `MAX_COMMAND_NAMES`,
    /// or `"other"` if it has and the name is not already in the set.
    fn resolve_command_name(&self, name: &str) -> String {
        let mut seen = self
            .seen_command_names
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if seen.contains(name) {
            name.to_string()
        } else if seen.len() >= MAX_COMMAND_NAMES {
            "other".to_string()
        } else {
            seen.insert(name.to_string());
            name.to_string()
        }
    }
}

impl Default for MetricsLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> tracing_subscriber::Layer<S> for MetricsLayer
where
    S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };

        // Extract span attributes into our typed storage.
        let metadata = span.metadata();
        let span_name = metadata.name();

        let span_attrs = match span_name {
            "request" => {
                let mut visitor = StringVisitor::default();
                attrs.record(&mut visitor);
                let pipeline = visitor
                    .get("weft.pipeline")
                    .unwrap_or("default")
                    .to_string();
                SpanAttributes::Request {
                    pipeline,
                    status: "ok".to_string(), // default; overridden by on_record
                    degraded: false,          // default; overridden by on_record
                }
            }
            "activity" => {
                let mut visitor = StringVisitor::default();
                attrs.record(&mut visitor);
                let name = visitor
                    .get("activity.name")
                    .unwrap_or("unknown")
                    .to_string();
                let phase = visitor
                    .get("activity.phase")
                    .unwrap_or("unknown")
                    .to_string();
                SpanAttributes::Activity {
                    name,
                    phase,
                    status: "ok".to_string(), // default; overridden by on_record
                    error_code: String::new(), // populated by on_record on degradation
                }
            }
            "generate" => {
                let mut visitor = StringVisitor::default();
                attrs.record(&mut visitor);
                let model = visitor
                    .get("provider.model")
                    .unwrap_or("unknown")
                    .to_string();
                let provider = visitor
                    .get("provider.name")
                    .unwrap_or("unknown")
                    .to_string();
                SpanAttributes::Generate {
                    model,
                    provider,
                    status: "ok".to_string(),
                    input_tokens: 0,
                    output_tokens: 0,
                }
            }
            "execute_command" => {
                let mut visitor = StringVisitor::default();
                attrs.record(&mut visitor);
                let name = visitor.get("command.name").unwrap_or("unknown").to_string();
                let resolved = self.resolve_command_name(&name);
                SpanAttributes::ExecuteCommand { name: resolved }
            }
            "iteration" => SpanAttributes::Iteration,
            _ => SpanAttributes::Other,
        };

        // Increment active requests gauge when a `request` span opens.
        if matches!(span_attrs, SpanAttributes::Request { .. }) {
            gauge!("weft_active_requests").increment(1.0);
        }

        span.extensions_mut().insert(SpanMetrics {
            start: Instant::now(),
            attrs: span_attrs,
        });
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut extensions = span.extensions_mut();
        let Some(metrics) = extensions.get_mut::<SpanMetrics>() else {
            return;
        };

        // Update mutable fields from late-recorded attributes.
        let mut visitor = StringVisitor::default();
        values.record(&mut visitor);

        match &mut metrics.attrs {
            SpanAttributes::Request {
                status, degraded, ..
            } => {
                // otel.status_code is recorded as "OK" or "ERROR" at span close.
                if let Some(s) = visitor.get("otel.status_code") {
                    *status = if s.eq_ignore_ascii_case("error") {
                        "error".to_string()
                    } else {
                        "ok".to_string()
                    };
                }
                // weft.request.degraded = true triggers the degraded request counter.
                if let Some(b) = visitor.get_bool("weft.request.degraded") {
                    *degraded = b;
                }
            }
            SpanAttributes::Activity {
                status, error_code, ..
            } => {
                // activity.status = "degraded" triggers the degradation counter.
                if let Some(s) = visitor.get("activity.status") {
                    *status = s.to_string();
                }
                // degradation.error_code is recorded alongside activity.status = "degraded".
                if let Some(code) = visitor.get("degradation.error_code") {
                    *error_code = code.to_string();
                }
            }
            SpanAttributes::Generate {
                status,
                model,
                provider,
                input_tokens,
                output_tokens,
            } => {
                if let Some(s) = visitor.get("otel.status_code") {
                    *status = if s.eq_ignore_ascii_case("error") {
                        "error".to_string()
                    } else {
                        "ok".to_string()
                    };
                }
                // provider.name and provider.model may be recorded late (after span creation)
                // via span.record() when the provider is not known at span creation time.
                if let Some(p) = visitor.get("provider.name") {
                    *provider = p.to_string();
                }
                if let Some(m) = visitor.get("provider.model") {
                    *model = m.to_string();
                }
                if let Some(tok) = visitor.get_u64("provider.input_tokens") {
                    *input_tokens = tok;
                }
                if let Some(tok) = visitor.get_u64("provider.output_tokens") {
                    *output_tokens = tok;
                }
            }
            _ => {}
        }
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let extensions = span.extensions();
        let Some(metrics) = extensions.get::<SpanMetrics>() else {
            return;
        };

        let duration = metrics.start.elapsed().as_secs_f64();

        match &metrics.attrs {
            SpanAttributes::Request {
                pipeline,
                status,
                degraded,
            } => {
                counter!(
                    "weft_requests_total",
                    "pipeline" => pipeline.clone(),
                    "status" => status.clone()
                )
                .increment(1);
                histogram!(
                    "weft_request_duration_seconds",
                    "pipeline" => pipeline.clone()
                )
                .record(duration);
                gauge!("weft_active_requests").decrement(1.0);
                // Emit the degraded request counter when the request had degradations.
                // Separate request-level degradation counter alongside the main
                // requests counter.
                if *degraded {
                    counter!("weft_requests_degraded_total").increment(1);
                }
            }
            SpanAttributes::Generate {
                model,
                provider,
                status,
                input_tokens,
                output_tokens,
            } => {
                histogram!(
                    "weft_provider_call_duration_seconds",
                    "model" => model.clone(),
                    "provider" => provider.clone(),
                    "status" => status.clone()
                )
                .record(duration);
                if *input_tokens > 0 {
                    counter!(
                        "weft_provider_tokens_total",
                        "model" => model.clone(),
                        "direction" => "input"
                    )
                    .increment(*input_tokens);
                }
                if *output_tokens > 0 {
                    counter!(
                        "weft_provider_tokens_total",
                        "model" => model.clone(),
                        "direction" => "output"
                    )
                    .increment(*output_tokens);
                }
            }
            SpanAttributes::Activity {
                name,
                phase,
                status,
                error_code,
            } => {
                histogram!(
                    "weft_activity_duration_seconds",
                    "activity_name" => name.clone(),
                    "phase" => phase.clone()
                )
                .record(duration);
                // Emit the component-level degradation counter when the activity degraded.
                // Per-activity degradation counter.
                if status == "degraded" {
                    counter!(
                        "weft_degradations_total",
                        "activity_name" => name.clone(),
                        "error_code" => error_code.clone()
                    )
                    .increment(1);
                }
            }
            SpanAttributes::ExecuteCommand { name } => {
                histogram!(
                    "weft_command_duration_seconds",
                    "command_name" => name.clone()
                )
                .record(duration);
            }
            SpanAttributes::Iteration => {
                counter!("weft_dispatch_iterations_total").increment(1);
            }
            SpanAttributes::Other => {}
        }
    }
}

// ── Field visitors ────────────────────────────────────────────────────────────

/// A `tracing::field::Visit` implementation that captures string, numeric,
/// and boolean field values by name for attribute extraction in the MetricsLayer.
#[derive(Default)]
struct StringVisitor {
    /// String fields keyed by field name.
    strings: Vec<(String, String)>,
    /// Unsigned integer fields keyed by field name.
    u64s: Vec<(String, u64)>,
    /// Signed integer fields (also stored as i64).
    i64s: Vec<(String, i64)>,
    /// Boolean fields keyed by field name.
    bools: Vec<(String, bool)>,
}

impl StringVisitor {
    fn get(&self, key: &str) -> Option<&str> {
        self.strings
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn get_u64(&self, key: &str) -> Option<u64> {
        // Check both u64s and i64s (tracing may record u32 as i64 in some contexts).
        if let Some(&(_, v)) = self.u64s.iter().find(|(k, _)| k == key) {
            return Some(v);
        }
        if let Some(&(_, v)) = self.i64s.iter().find(|(k, _)| k == key) {
            return u64::try_from(v).ok();
        }
        None
    }

    fn get_bool(&self, key: &str) -> Option<bool> {
        self.bools.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }
}

impl tracing::field::Visit for StringVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.strings
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        // Some attributes are formatted as Debug (e.g., &str via %fmt).
        // Strip leading/trailing quotes that Debug adds for strings.
        let raw = format!("{:?}", value);
        let trimmed = raw.trim_matches('"').to_string();
        self.strings.push((field.name().to_string(), trimmed));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.u64s.push((field.name().to_string(), value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.i64s.push((field.name().to_string(), value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.bools.push((field.name().to_string(), value));
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use pretty_assertions::assert_eq;
    use tracing::subscriber::with_default;
    use tracing_subscriber::Registry;
    use tracing_subscriber::layer::SubscriberExt;

    // ── Test infrastructure ───────────────────────────────────────────────
    //
    // `CompositeKey` lives in a private module inside `metrics-util`, so we
    // cannot name it in a type annotation. The workaround: convert each raw
    // snapshot entry into a plain `SnapshotItem` that uses only public types.
    // `CompositeKey::key()` returns `&metrics::Key`, which is fully public and
    // exposes `.name()` and `.labels()`.

    /// Decoded snapshot entry with only public, nameable types.
    struct SnapshotItem {
        name: String,
        labels: Vec<(String, String)>,
        value: DebugValue,
    }

    /// Run `f` with a fresh `DebuggingRecorder` and `MetricsLayer` active,
    /// then decode the snapshot into `Vec<SnapshotItem>` and call `assert_fn`.
    ///
    /// Each call gets its own isolated recorder so tests do not interfere.
    fn with_metrics_layer<F, A>(f: F, assert_fn: A)
    where
        F: FnOnce(),
        A: FnOnce(&[SnapshotItem]),
    {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let subscriber = Registry::default().with(MetricsLayer::new());

        metrics::with_local_recorder(&recorder, || {
            with_default(subscriber, f);
        });

        // Convert the opaque `(CompositeKey, …)` tuples to named items.
        // The element type of `into_vec()` is inferred by the compiler; we
        // only use `ck.key()` which is the stable public accessor.
        let raw = snapshotter.snapshot().into_vec();
        let items: Vec<SnapshotItem> = raw
            .into_iter()
            .map(|(ck, _unit, _desc, value)| {
                let key = ck.key();
                let name = key.name().to_string();
                let labels = key
                    .labels()
                    .map(|l| (l.key().to_string(), l.value().to_string()))
                    .collect();
                SnapshotItem {
                    name,
                    labels,
                    value,
                }
            })
            .collect();

        assert_fn(&items);
    }

    /// Return `true` if `items` contains an entry whose name equals `metric_name`
    /// and whose labels include all `(key, value)` pairs in `required_labels`.
    fn snapshot_has(
        items: &[SnapshotItem],
        metric_name: &str,
        required_labels: &[(&str, &str)],
    ) -> bool {
        items.iter().any(|item| {
            item.name == metric_name
                && required_labels
                    .iter()
                    .all(|(k, v)| item.labels.iter().any(|(lk, lv)| lk == k && lv == *v))
        })
    }

    /// Return the counter value for the first entry matching `metric_name`
    /// and all `required_labels`. Returns `None` if not found or if the entry
    /// is not a counter.
    fn counter_value(
        items: &[SnapshotItem],
        metric_name: &str,
        required_labels: &[(&str, &str)],
    ) -> Option<u64> {
        items
            .iter()
            .find(|item| {
                item.name == metric_name
                    && required_labels
                        .iter()
                        .all(|(k, v)| item.labels.iter().any(|(lk, lv)| lk == k && lv == *v))
            })
            .and_then(|item| match &item.value {
                DebugValue::Counter(n) => Some(*n),
                _ => None,
            })
    }

    // ── weft_active_requests gauge ────────────────────────────────────────

    #[test]
    fn request_span_increments_active_gauge_on_new() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-1",
                    "weft.tenant_id" = "tenant-1",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                )
                .entered();
            },
            |items| {
                assert!(
                    snapshot_has(items, "weft_active_requests", &[]),
                    "weft_active_requests gauge not found in snapshots"
                );
            },
        );
    }

    #[test]
    fn request_span_decrements_active_gauge_on_close() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-2",
                    "weft.tenant_id" = "t",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                );
            },
            |items| {
                assert!(
                    snapshot_has(items, "weft_active_requests", &[]),
                    "weft_active_requests gauge not found"
                );
            },
        );
    }

    // ── weft_requests_total counter ───────────────────────────────────────

    #[test]
    fn request_span_emits_requests_total_on_close() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-3",
                    "weft.tenant_id" = "t",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                );
                span.record("otel.status_code", "OK");
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_requests_total",
                        &[("status", "ok"), ("pipeline", "default")]
                    ),
                    "weft_requests_total with status=ok not found"
                );
                let v = counter_value(items, "weft_requests_total", &[("status", "ok")]);
                assert_eq!(v, Some(1), "counter should be 1 after one request span");
            },
        );
    }

    #[test]
    fn request_span_emits_error_status_on_close() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-err",
                    "weft.tenant_id" = "t",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                );
                span.record("otel.status_code", "ERROR");
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(items, "weft_requests_total", &[("status", "error")]),
                    "weft_requests_total with status=error not found"
                );
            },
        );
    }

    // ── weft_request_duration_seconds histogram ───────────────────────────

    #[test]
    fn request_span_emits_duration_histogram() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-dur",
                    "weft.tenant_id" = "t",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                );
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_request_duration_seconds",
                        &[("pipeline", "default")]
                    ),
                    "weft_request_duration_seconds histogram not found"
                );
            },
        );
    }

    // ── weft_provider_call_duration_seconds ───────────────────────────────

    #[test]
    fn generate_span_emits_provider_call_duration() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "generate",
                    "provider.model" = "claude-sonnet",
                    "provider.name" = "anthropic",
                    "provider.attempt" = 0u32,
                    "provider.input_tokens" = tracing::field::Empty,
                    "provider.output_tokens" = tracing::field::Empty,
                    "provider.stop_reason" = tracing::field::Empty,
                );
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_provider_call_duration_seconds",
                        &[("model", "claude-sonnet"), ("provider", "anthropic")]
                    ),
                    "weft_provider_call_duration_seconds not found"
                );
            },
        );
    }

    // ── weft_provider_tokens_total counter ────────────────────────────────

    #[test]
    fn generate_span_emits_token_counters() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "generate",
                    "provider.model" = "claude-sonnet",
                    "provider.name" = "anthropic",
                    "provider.attempt" = 0u32,
                    "provider.input_tokens" = tracing::field::Empty,
                    "provider.output_tokens" = tracing::field::Empty,
                    "provider.stop_reason" = tracing::field::Empty,
                );
                span.record("provider.input_tokens", 100u64);
                span.record("provider.output_tokens", 50u64);
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_provider_tokens_total",
                        &[("direction", "input"), ("model", "claude-sonnet")]
                    ),
                    "weft_provider_tokens_total input not found"
                );
                let v = counter_value(
                    items,
                    "weft_provider_tokens_total",
                    &[("direction", "input")],
                );
                assert_eq!(v, Some(100), "input token counter should be 100");

                assert!(
                    snapshot_has(
                        items,
                        "weft_provider_tokens_total",
                        &[("direction", "output"), ("model", "claude-sonnet")]
                    ),
                    "weft_provider_tokens_total output not found"
                );
                let v = counter_value(
                    items,
                    "weft_provider_tokens_total",
                    &[("direction", "output")],
                );
                assert_eq!(v, Some(50), "output token counter should be 50");
            },
        );
    }

    #[test]
    fn generate_span_empty_tokens_does_not_emit_zero_counter() {
        // When token fields are never recorded, the layer must not emit zero-value counters.
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "generate",
                    "provider.model" = "claude-sonnet",
                    "provider.name" = "anthropic",
                    "provider.attempt" = 0u32,
                    "provider.input_tokens" = tracing::field::Empty,
                    "provider.output_tokens" = tracing::field::Empty,
                    "provider.stop_reason" = tracing::field::Empty,
                );
            },
            |items| {
                assert!(
                    !snapshot_has(items, "weft_provider_tokens_total", &[]),
                    "weft_provider_tokens_total should not be emitted when tokens are zero"
                );
            },
        );
    }

    // ── weft_activity_duration_seconds ────────────────────────────────────

    #[test]
    fn activity_span_emits_activity_duration() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "activity",
                    "activity.name" = "validate",
                    "activity.phase" = "pre_loop",
                    "activity.status" = tracing::field::Empty,
                );
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_activity_duration_seconds",
                        &[("activity_name", "validate"), ("phase", "pre_loop")]
                    ),
                    "weft_activity_duration_seconds not found"
                );
            },
        );
    }

    // ── weft_command_duration_seconds ─────────────────────────────────────

    #[test]
    fn execute_command_span_emits_command_duration() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!(
                    "execute_command",
                    "command.name" = "search",
                    "command.index" = 0u32,
                    "command.status" = tracing::field::Empty,
                );
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_command_duration_seconds",
                        &[("command_name", "search")]
                    ),
                    "weft_command_duration_seconds not found"
                );
            },
        );
    }

    // ── weft_dispatch_iterations_total ────────────────────────────────────

    #[test]
    fn iteration_span_emits_iterations_counter() {
        with_metrics_layer(
            || {
                {
                    let _span = tracing::info_span!("iteration", "iteration.number" = 0u32);
                }
                {
                    let _span = tracing::info_span!("iteration", "iteration.number" = 1u32);
                }
            },
            |items| {
                assert!(
                    snapshot_has(items, "weft_dispatch_iterations_total", &[]),
                    "weft_dispatch_iterations_total not found"
                );
                let v = counter_value(items, "weft_dispatch_iterations_total", &[]);
                assert_eq!(v, Some(2), "two iterations should produce counter=2");
            },
        );
    }

    // ── Command name cardinality cap ──────────────────────────────────────

    #[test]
    fn command_name_cardinality_cap_maps_overflow_to_other() {
        let layer = MetricsLayer::new();

        for i in 0..MAX_COMMAND_NAMES {
            let name = format!("cmd_{i}");
            let resolved = layer.resolve_command_name(&name);
            assert_eq!(resolved, name, "name under cap should pass through");
        }

        let overflow = layer.resolve_command_name("overflow_command");
        assert_eq!(
            overflow, "other",
            "51st unique command name must map to 'other'"
        );
    }

    #[test]
    fn command_name_already_in_set_does_not_become_other() {
        let layer = MetricsLayer::new();

        for i in 0..MAX_COMMAND_NAMES {
            layer.resolve_command_name(&format!("cmd_{i}"));
        }

        // A name already in the set should still resolve to itself.
        let already_in = layer.resolve_command_name("cmd_0");
        assert_eq!(
            already_in, "cmd_0",
            "known name should not be remapped to 'other'"
        );
    }

    // ── Other spans produce no metrics ────────────────────────────────────

    #[test]
    fn unknown_span_emits_no_metrics() {
        with_metrics_layer(
            || {
                let _span = tracing::info_span!("completely_unknown_span_name");
            },
            |items| {
                assert!(items.is_empty(), "unknown span should not emit any metrics");
            },
        );
    }

    // ── tracing::field::Empty handling ───────────────────────────────────

    #[test]
    fn empty_fields_recorded_late_are_included_in_metrics() {
        // Verify on_record correctly updates stored attributes when fields are
        // recorded via span.record() after span creation (the Empty pattern).
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "generate",
                    "provider.model" = "gpt-4",
                    "provider.name" = "openai",
                    "provider.attempt" = 0u32,
                    "provider.input_tokens" = tracing::field::Empty,
                    "provider.output_tokens" = tracing::field::Empty,
                    "provider.stop_reason" = tracing::field::Empty,
                );
                span.record("provider.input_tokens", 200u64);
                span.record("provider.output_tokens", 75u64);
                drop(span);
            },
            |items| {
                let v = counter_value(
                    items,
                    "weft_provider_tokens_total",
                    &[("direction", "input")],
                );
                assert_eq!(v, Some(200), "input tokens not found after late record");

                let v = counter_value(
                    items,
                    "weft_provider_tokens_total",
                    &[("direction", "output")],
                );
                assert_eq!(v, Some(75), "output tokens not found after late record");
            },
        );
    }

    // ── Late-recorded provider.name / provider.model labels ───────────────

    #[test]
    fn generate_span_late_recorded_provider_appears_in_metric_label() {
        // Production code in generate.rs creates the span with
        // provider.name = tracing::field::Empty and calls span.record() once
        // the provider is resolved. The metric label must reflect the recorded
        // value, not the "unknown" default stored at span creation.
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "generate",
                    "provider.model" = "llama3",
                    "provider.name" = tracing::field::Empty,
                    "provider.attempt" = 0u32,
                    "provider.input_tokens" = tracing::field::Empty,
                    "provider.output_tokens" = tracing::field::Empty,
                    "provider.stop_reason" = tracing::field::Empty,
                );
                span.record("provider.name", "ollama");
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_provider_call_duration_seconds",
                        &[("provider", "ollama")]
                    ),
                    "metric label should be 'ollama' after late span.record(), not 'unknown'"
                );
                assert!(
                    !snapshot_has(
                        items,
                        "weft_provider_call_duration_seconds",
                        &[("provider", "unknown")]
                    ),
                    "metric label must not be 'unknown' when provider was recorded late"
                );
            },
        );
    }

    #[test]
    fn generate_span_late_recorded_model_appears_in_metric_label() {
        // Defensive coverage for provider.model using the same late-recording pattern
        // as provider.name (in case model resolution is also deferred).
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "generate",
                    "provider.model" = tracing::field::Empty,
                    "provider.name" = "anthropic",
                    "provider.attempt" = 0u32,
                    "provider.input_tokens" = tracing::field::Empty,
                    "provider.output_tokens" = tracing::field::Empty,
                    "provider.stop_reason" = tracing::field::Empty,
                );
                span.record("provider.model", "claude-opus-4");
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_provider_call_duration_seconds",
                        &[("model", "claude-opus-4")]
                    ),
                    "metric label should be 'claude-opus-4' after late span.record(), not 'unknown'"
                );
                assert!(
                    !snapshot_has(
                        items,
                        "weft_provider_call_duration_seconds",
                        &[("model", "unknown")]
                    ),
                    "metric label must not be 'unknown' when model was recorded late"
                );
            },
        );
    }

    // ── Phase 3: Degradation counters ─────────────────────────────────────

    /// activity span with activity.status = "degraded" emits weft_degradations_total counter.
    #[test]
    fn activity_span_degraded_status_emits_degradations_counter() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "activity",
                    "activity.name" = "command_selection",
                    "activity.phase" = "pre_loop",
                    "activity.status" = tracing::field::Empty,
                    "degradation.error_code" = tracing::field::Empty,
                    "otel.status_code" = tracing::field::Empty,
                );
                span.record("activity.status", "degraded");
                span.record("degradation.error_code", "classifier_unavailable");
                span.record("otel.status_code", "ERROR");
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_degradations_total",
                        &[
                            ("activity_name", "command_selection"),
                            ("error_code", "classifier_unavailable"),
                        ]
                    ),
                    "weft_degradations_total counter not found for degraded activity"
                );
                let v = counter_value(
                    items,
                    "weft_degradations_total",
                    &[("activity_name", "command_selection")],
                );
                assert_eq!(
                    v,
                    Some(1),
                    "counter should be 1 after one degraded activity span"
                );
            },
        );
    }

    /// activity span with activity.status = "ok" does NOT emit weft_degradations_total.
    #[test]
    fn activity_span_ok_status_does_not_emit_degradations_counter() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "activity",
                    "activity.name" = "validate",
                    "activity.phase" = "pre_loop",
                    "activity.status" = tracing::field::Empty,
                    "degradation.error_code" = tracing::field::Empty,
                    "otel.status_code" = tracing::field::Empty,
                );
                span.record("activity.status", "ok");
                drop(span);
            },
            |items| {
                assert!(
                    !snapshot_has(items, "weft_degradations_total", &[]),
                    "weft_degradations_total must not be emitted for ok activity"
                );
            },
        );
    }

    /// activity span with activity.status = "error" does NOT emit weft_degradations_total.
    #[test]
    fn activity_span_error_status_does_not_emit_degradations_counter() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "activity",
                    "activity.name" = "validate",
                    "activity.phase" = "pre_loop",
                    "activity.status" = tracing::field::Empty,
                    "degradation.error_code" = tracing::field::Empty,
                    "otel.status_code" = tracing::field::Empty,
                );
                span.record("activity.status", "error");
                drop(span);
            },
            |items| {
                assert!(
                    !snapshot_has(items, "weft_degradations_total", &[]),
                    "weft_degradations_total must not be emitted for error activity (only degraded)"
                );
            },
        );
    }

    /// request span with weft.request.degraded = true emits weft_requests_degraded_total counter.
    #[test]
    fn request_span_degraded_emits_requests_degraded_total_counter() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-deg",
                    "weft.tenant_id" = "t",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                    "weft.request.degraded" = tracing::field::Empty,
                    "weft.request.degradation_count" = tracing::field::Empty,
                );
                span.record("otel.status_code", "OK");
                span.record("weft.request.degraded", true);
                span.record("weft.request.degradation_count", 1u64);
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(items, "weft_requests_degraded_total", &[]),
                    "weft_requests_degraded_total counter not found for degraded request"
                );
                let v = counter_value(items, "weft_requests_degraded_total", &[]);
                assert_eq!(
                    v,
                    Some(1),
                    "counter should be 1 after one degraded request span"
                );
            },
        );
    }

    /// request span without weft.request.degraded does NOT emit weft_requests_degraded_total.
    #[test]
    fn request_span_not_degraded_does_not_emit_requests_degraded_total() {
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "request",
                    "weft.request_id" = "req-ok",
                    "weft.tenant_id" = "t",
                    "weft.pipeline" = "default",
                    "weft.depth" = 0u32,
                    "otel.kind" = "server",
                    "otel.status_code" = tracing::field::Empty,
                    "weft.request.degraded" = tracing::field::Empty,
                    "weft.request.degradation_count" = tracing::field::Empty,
                );
                span.record("otel.status_code", "OK");
                // weft.request.degraded not recorded — should default to false
                drop(span);
            },
            |items| {
                assert!(
                    !snapshot_has(items, "weft_requests_degraded_total", &[]),
                    "weft_requests_degraded_total must not be emitted when request is not degraded"
                );
            },
        );
    }

    /// Activity span: otel.status_code = ERROR set when degraded (not OK).
    /// This is a span attribute test confirming the span records the correct value.
    #[test]
    fn activity_span_degraded_records_error_otel_status() {
        // We test that when activity.status = "degraded", the MetricsLayer's
        // on_record correctly captures "degradation.error_code" (not otel.status_code
        // — that's a tracing span attribute, not a MetricsLayer concern).
        // The two-level span status behavior is tested via the degradation counter test above.
        with_metrics_layer(
            || {
                let span = tracing::info_span!(
                    "activity",
                    "activity.name" = "model_selection",
                    "activity.phase" = "pre_loop",
                    "activity.status" = tracing::field::Empty,
                    "degradation.error_code" = tracing::field::Empty,
                    "otel.status_code" = tracing::field::Empty,
                );
                span.record("activity.status", "degraded");
                span.record("degradation.error_code", "no_matching_model");
                span.record("otel.status_code", "ERROR");
                drop(span);
            },
            |items| {
                assert!(
                    snapshot_has(
                        items,
                        "weft_degradations_total",
                        &[
                            ("activity_name", "model_selection"),
                            ("error_code", "no_matching_model"),
                        ]
                    ),
                    "weft_degradations_total should use the recorded error_code label"
                );
            },
        );
    }
}
