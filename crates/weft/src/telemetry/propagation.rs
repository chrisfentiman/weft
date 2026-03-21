//! W3C TraceContext propagation helpers.
//!
//! Provides `Extractor` and `Injector` adapters for both HTTP (`http::HeaderMap`)
//! and gRPC (`tonic::metadata::MetadataMap`) transports, enabling W3C `traceparent`
//! header extraction from inbound requests and injection into outbound calls.
//!
//! # Usage
//!
//! **Inbound HTTP (extract context):**
//! ```rust,ignore
//! let parent_ctx = extract_from_headers(request.headers());
//! let span = info_span!("request");
//! // Attach parent context so the span becomes a child of the incoming trace.
//! use tracing_opentelemetry::OpenTelemetrySpanExt;
//! span.set_parent(parent_ctx);
//! ```
//!
//! **Outbound HTTP (inject context):**
//! ```rust,ignore
//! let mut headers = HeaderMap::new();
//! inject_into_headers(&mut headers);
//! // headers now contains "traceparent" (and optionally "tracestate").
//! ```

use opentelemetry::Context;
use opentelemetry::propagation::{Extractor, Injector, TextMapPropagator};
use opentelemetry_sdk::propagation::TraceContextPropagator;

// axum re-exports the `http` crate — use it to avoid adding a direct `http` dep.
use axum::http;

// ── HTTP header adapters ─────────────────────────────────────────────────────

/// Adapts `http::HeaderMap` for use as an OpenTelemetry [`Extractor`].
pub(crate) struct HeaderExtractor<'a>(pub(crate) &'a http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0
            .get(key)
            .and_then(|v: &http::HeaderValue| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .map(|k: &http::header::HeaderName| k.as_str())
            .collect()
    }
}

/// Adapts a mutable `http::HeaderMap` for use as an OpenTelemetry [`Injector`].
pub(crate) struct HeaderInjector<'a>(pub(crate) &'a mut http::HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(name) = http::header::HeaderName::from_bytes(key.as_bytes())
            && let Ok(val) = http::header::HeaderValue::from_str(&value)
        {
            self.0.insert(name, val);
        }
    }
}

// ── gRPC metadata adapters ────────────────────────────────────────────────────

/// Adapts `tonic::metadata::MetadataMap` for use as an OpenTelemetry [`Extractor`].
pub(crate) struct MetadataExtractor<'a>(pub(crate) &'a tonic::metadata::MetadataMap);

impl Extractor for MetadataExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .filter_map(|k| match k {
                tonic::metadata::KeyRef::Ascii(k) => Some(k.as_str()),
                tonic::metadata::KeyRef::Binary(_) => None,
            })
            .collect()
    }
}

/// Adapts a mutable `tonic::metadata::MetadataMap` for use as an OpenTelemetry [`Injector`].
pub(crate) struct MetadataInjector<'a>(pub(crate) &'a mut tonic::metadata::MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        // tonic metadata keys must be ASCII.
        if let Ok(k) =
            tonic::metadata::MetadataKey::<tonic::metadata::Ascii>::from_bytes(key.as_bytes())
            && let Ok(v) = tonic::metadata::MetadataValue::try_from(value.as_str())
        {
            self.0.insert(k, v);
        }
    }
}

// ── Convenience functions ─────────────────────────────────────────────────────

/// Extract the W3C TraceContext from an HTTP `HeaderMap`.
///
/// Returns the parent [`Context`] when a valid `traceparent` header is present,
/// or an empty context when absent. The caller should call
/// `span.set_parent(ctx)` to make the request span a child of the incoming trace.
///
/// This function is a no-op when the `tracing-opentelemetry` layer is absent
/// (the returned context is empty and `set_parent` has no effect).
pub fn extract_from_headers(headers: &http::HeaderMap) -> Context {
    let propagator = TraceContextPropagator::new();
    propagator.extract(&HeaderExtractor(headers))
}

/// Extract the W3C TraceContext from a gRPC `MetadataMap`.
///
/// Identical semantics to [`extract_from_headers`] but operates on gRPC metadata.
pub fn extract_from_metadata(metadata: &tonic::metadata::MetadataMap) -> Context {
    let propagator = TraceContextPropagator::new();
    propagator.extract(&MetadataExtractor(metadata))
}

/// Inject the current span's W3C TraceContext into an HTTP `HeaderMap`.
///
/// Adds a `traceparent` header (and optionally `tracestate`) derived from the
/// current OTel context. When no OTel layer is active, the OTel context is
/// empty and the propagator injects nothing — this is always safe to call.
///
/// Call this inside a `provider_call` span (after entering it) so the injected
/// `traceparent` contains the `provider_call` span ID, linking the provider's
/// server-side trace as a child of this gateway's provider_call span.
pub fn inject_into_headers(headers: &mut http::HeaderMap) {
    let propagator = TraceContextPropagator::new();
    // opentelemetry::Context::current() returns the OTel context propagated
    // by the tracing-opentelemetry layer from the current tracing span.
    // When no OTel layer is active, this returns an empty context and inject
    // is a no-op.
    propagator.inject_context(
        &opentelemetry::Context::current(),
        &mut HeaderInjector(headers),
    );
}

/// Inject the current span's W3C TraceContext into a gRPC `MetadataMap`.
///
/// Identical semantics to [`inject_into_headers`] but operates on gRPC metadata.
pub fn inject_into_metadata(metadata: &mut tonic::metadata::MetadataMap) {
    let propagator = TraceContextPropagator::new();
    propagator.inject_context(
        &opentelemetry::Context::current(),
        &mut MetadataInjector(metadata),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    // ── HeaderExtractor ───────────────────────────────────────────────────────

    #[test]
    fn header_extractor_returns_value_for_known_key() {
        let mut headers = http::HeaderMap::new();
        headers.insert("traceparent", "00-abc123-def456-01".parse().unwrap());

        let extractor = HeaderExtractor(&headers);
        assert_eq!(extractor.get("traceparent"), Some("00-abc123-def456-01"));
    }

    #[test]
    fn header_extractor_returns_none_for_missing_key() {
        let headers = http::HeaderMap::new();
        let extractor = HeaderExtractor(&headers);
        assert_eq!(extractor.get("traceparent"), None);
    }

    #[test]
    fn header_extractor_keys_returns_all_header_names() {
        let mut headers = http::HeaderMap::new();
        headers.insert("traceparent", "value1".parse().unwrap());
        headers.insert("x-custom", "value2".parse().unwrap());

        let extractor = HeaderExtractor(&headers);
        let keys = extractor.keys();
        assert!(
            keys.contains(&"traceparent"),
            "keys should contain traceparent"
        );
        assert!(keys.contains(&"x-custom"), "keys should contain x-custom");
    }

    // ── HeaderInjector ────────────────────────────────────────────────────────

    #[test]
    fn header_injector_inserts_valid_header() {
        let mut headers = http::HeaderMap::new();
        {
            let mut injector = HeaderInjector(&mut headers);
            injector.set("traceparent", "00-abc-def-01".to_string());
        }
        assert_eq!(
            headers.get("traceparent").and_then(|v| v.to_str().ok()),
            Some("00-abc-def-01")
        );
    }

    #[test]
    fn header_injector_silently_ignores_invalid_header_name() {
        let mut headers = http::HeaderMap::new();
        {
            let mut injector = HeaderInjector(&mut headers);
            // Header names with spaces are invalid — should not panic.
            injector.set("invalid header name", "value".to_string());
        }
        assert!(
            headers.is_empty(),
            "invalid header name should be silently ignored"
        );
    }

    // ── MetadataExtractor ─────────────────────────────────────────────────────

    #[test]
    fn metadata_extractor_returns_value_for_known_key() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert("traceparent", "00-abc123-def456-01".parse().unwrap());

        let extractor = MetadataExtractor(&metadata);
        assert_eq!(extractor.get("traceparent"), Some("00-abc123-def456-01"));
    }

    #[test]
    fn metadata_extractor_returns_none_for_missing_key() {
        let metadata = tonic::metadata::MetadataMap::new();
        let extractor = MetadataExtractor(&metadata);
        assert_eq!(extractor.get("traceparent"), None);
    }

    // ── MetadataInjector ──────────────────────────────────────────────────────

    #[test]
    fn metadata_injector_inserts_valid_key() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        {
            let mut injector = MetadataInjector(&mut metadata);
            injector.set("traceparent", "00-abc-def-01".to_string());
        }
        assert_eq!(
            metadata.get("traceparent").and_then(|v| v.to_str().ok()),
            Some("00-abc-def-01")
        );
    }

    // ── extract_from_headers ──────────────────────────────────────────────────

    #[test]
    fn extract_from_headers_returns_empty_context_when_no_traceparent() {
        let headers = http::HeaderMap::new();
        let ctx = extract_from_headers(&headers);
        // Without a traceparent header, the extracted context has no remote span.
        use opentelemetry::trace::TraceContextExt;
        assert!(
            !ctx.span().span_context().is_valid(),
            "context without traceparent should have invalid span context"
        );
    }

    #[test]
    fn extract_from_headers_returns_valid_context_when_traceparent_present() {
        let mut headers = http::HeaderMap::new();
        // Valid W3C traceparent: version-traceid-parentid-flags
        headers.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
                .parse()
                .unwrap(),
        );
        let ctx = extract_from_headers(&headers);
        use opentelemetry::trace::TraceContextExt;
        assert!(
            ctx.span().span_context().is_valid(),
            "valid traceparent should produce valid span context"
        );
    }

    #[test]
    fn extract_from_metadata_returns_empty_context_when_no_traceparent() {
        let metadata = tonic::metadata::MetadataMap::new();
        let ctx = extract_from_metadata(&metadata);
        use opentelemetry::trace::TraceContextExt;
        assert!(
            !ctx.span().span_context().is_valid(),
            "context without traceparent should have invalid span context"
        );
    }

    #[test]
    fn extract_from_metadata_returns_valid_context_when_traceparent_present() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
                .parse()
                .unwrap(),
        );
        let ctx = extract_from_metadata(&metadata);
        use opentelemetry::trace::TraceContextExt;
        assert!(
            ctx.span().span_context().is_valid(),
            "valid traceparent in metadata should produce valid span context"
        );
    }

    // ── Round-trip test ───────────────────────────────────────────────────────

    #[test]
    fn extracted_trace_id_matches_original() {
        let trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
        let span_id = "00f067aa0ba902b7";
        let traceparent = format!("00-{trace_id}-{span_id}-01");

        let mut headers = http::HeaderMap::new();
        headers.insert("traceparent", traceparent.parse().unwrap());

        let ctx = extract_from_headers(&headers);
        use opentelemetry::trace::TraceContextExt;
        let span = ctx.span();
        let span_ctx = span.span_context();

        assert_eq!(
            span_ctx.trace_id().to_string(),
            trace_id,
            "extracted trace_id must match the original"
        );
    }

    // ── inject_into_headers no-op test ────────────────────────────────────────

    #[test]
    fn inject_into_headers_does_not_panic_without_active_span() {
        // Without an active OTel span, inject should be a no-op (not panic).
        let mut headers = http::HeaderMap::new();
        inject_into_headers(&mut headers);
        // May or may not insert headers depending on global propagator state.
        // The important thing: it does not panic.
    }

    #[test]
    fn inject_into_metadata_does_not_panic_without_active_span() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        inject_into_metadata(&mut metadata);
    }

    // ── Idempotency: extracted trace IDs are stable ───────────────────────────

    #[test]
    fn extract_twice_from_same_headers_produces_same_trace_id() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
                .parse()
                .unwrap(),
        );

        let ctx1 = extract_from_headers(&headers);
        let ctx2 = extract_from_headers(&headers);

        use opentelemetry::trace::TraceContextExt;
        assert_eq!(
            ctx1.span().span_context().trace_id(),
            ctx2.span().span_context().trace_id(),
            "extracting twice from the same headers must yield the same trace_id"
        );
    }

    // ── Different trace IDs produce different contexts ─────────────────────────

    #[test]
    fn different_traceparent_headers_produce_different_trace_ids() {
        let mut h1 = http::HeaderMap::new();
        h1.insert(
            "traceparent",
            "00-aaaabbbbccccddddaaaabbbbccccdddd-1111111111111111-01"
                .parse()
                .unwrap(),
        );
        let mut h2 = http::HeaderMap::new();
        h2.insert(
            "traceparent",
            "00-eeeeffffeeeeffffeeeeffffeeeeefff-2222222222222222-01"
                .parse()
                .unwrap(),
        );

        let ctx1 = extract_from_headers(&h1);
        let ctx2 = extract_from_headers(&h2);

        use opentelemetry::trace::TraceContextExt;
        assert_ne!(
            ctx1.span().span_context().trace_id(),
            ctx2.span().span_context().trace_id(),
            "different traceparent headers must produce different trace IDs"
        );
    }
}
