//! Observability integration tests for the `weft` gateway.
//!
//! Verifies the span tree produced by a request through the pipeline:
//! - Span hierarchy: `request > pre_loop > activity*`, `request > dispatch_loop > iteration > generate`, etc.
//! - Span attributes: `request_id`, `activity.name`, `llm.model` are populated.
//!
//! # Design
//!
//! A custom [`GlobalSpanCollector`] layer is installed as the global default subscriber once
//! per test binary. Tests serialize via a [`TEST_MUTEX`] so only one test writes to the global
//! `ACTIVE_STORE` at a time. This avoids parallel tests overwriting each other's span data.
//!
//! # Why global subscriber
//!
//! `tracing::subscriber::with_default` scopes the subscriber to the closure's synchronous
//! extent. A closure that returns a future has the subscriber uninstalled before any async
//! code executes. Installing once globally avoids this limitation.

mod harness;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::Mutex as TokioMutex;

use axum::http::StatusCode;
use serde_json::json;
use tracing::span::{Attributes, Record};
use tracing::{Id, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

use harness::{ScriptedCommandRegistry, TestProvider, make_weft_service_with_event_log, post_json};
use weft::server::build_router;

// ── SpanStore ─────────────────────────────────────────────────────────────────

/// A recorded span entry from the test subscriber.
#[derive(Debug, Clone)]
struct SpanRecord {
    id: u64,
    name: String,
    parent_id: Option<u64>,
    attrs: HashMap<String, String>,
}

/// Accumulated span records for one test request.
#[derive(Default, Debug)]
struct SpanStore {
    spans: HashMap<u64, SpanRecord>,
}

impl SpanStore {
    fn find_by_name(&self, name: &str) -> Vec<&SpanRecord> {
        self.spans.values().filter(|s| s.name == name).collect()
    }

    fn has_direct_child(&self, parent_name: &str, child_name: &str) -> bool {
        self.spans.values().any(|s| {
            s.name == child_name
                && s.parent_id
                    .and_then(|pid| self.spans.get(&pid))
                    .map(|p| p.name == parent_name)
                    .unwrap_or(false)
        })
    }

    fn has_descendant(&self, ancestor_name: &str, child_name: &str) -> bool {
        let ancestor_ids: Vec<u64> = self
            .spans
            .values()
            .filter(|s| s.name == ancestor_name)
            .map(|s| s.id)
            .collect();

        if ancestor_ids.is_empty() {
            return false;
        }

        for span in self.spans.values().filter(|s| s.name == child_name) {
            let mut current_id = span.parent_id;
            while let Some(pid) = current_id {
                if ancestor_ids.contains(&pid) {
                    return true;
                }
                current_id = self.spans.get(&pid).and_then(|p| p.parent_id);
            }
        }
        false
    }

    fn all_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .spans
            .values()
            .map(|s| s.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }
}

// ── Global subscriber infrastructure ─────────────────────────────────────────

/// Serializes tests so only one writes to `ACTIVE_STORE` at a time.
///
/// `tokio::sync::Mutex` is used so the guard can be held across `.await` points
/// without triggering the `clippy::await_holding_lock` lint (which fires for
/// `std::sync::Mutex` guards held across awaits).
static TEST_MUTEX: OnceLock<TokioMutex<()>> = OnceLock::new();

/// The active span store for the currently running test.
static ACTIVE_STORE: OnceLock<Mutex<Option<Arc<Mutex<SpanStore>>>>> = OnceLock::new();

/// Whether the global subscriber has been initialized.
static GLOBAL_SUB_INIT: AtomicBool = AtomicBool::new(false);

struct GlobalSpanCollector;

struct AttrVisitor {
    attrs: HashMap<String, String>,
}

impl AttrVisitor {
    fn new() -> Self {
        Self {
            attrs: HashMap::new(),
        }
    }
}

impl tracing::field::Visit for AttrVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.attrs
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.attrs
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.attrs
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.attrs
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.attrs
            .insert(field.name().to_string(), value.to_string());
    }
}

impl<S: Subscriber + for<'l> tracing_subscriber::registry::LookupSpan<'l>> Layer<S>
    for GlobalSpanCollector
{
    fn on_new_span(
        &self,
        attrs: &Attributes<'_>,
        id: &Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let slot = ACTIVE_STORE.get_or_init(|| Mutex::new(None));
        let guard = slot.lock().expect("ACTIVE_STORE lock poisoned");
        let Some(store_arc) = guard.as_ref().map(Arc::clone) else {
            return;
        };
        drop(guard);

        let mut visitor = AttrVisitor::new();
        attrs.record(&mut visitor);

        let parent_id = if let Some(parent) = attrs.parent() {
            Some(parent.into_u64())
        } else if attrs.is_contextual() {
            ctx.current_span().id().map(|id| id.into_u64())
        } else {
            None
        };

        store_arc
            .lock()
            .expect("SpanStore lock poisoned")
            .spans
            .insert(
                id.into_u64(),
                SpanRecord {
                    id: id.into_u64(),
                    name: attrs.metadata().name().to_string(),
                    parent_id,
                    attrs: visitor.attrs,
                },
            );
    }

    fn on_record(
        &self,
        id: &Id,
        values: &Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let slot = ACTIVE_STORE.get_or_init(|| Mutex::new(None));
        let guard = slot.lock().expect("ACTIVE_STORE lock poisoned");
        let Some(store_arc) = guard.as_ref().map(Arc::clone) else {
            return;
        };
        drop(guard);

        let mut visitor = AttrVisitor::new();
        values.record(&mut visitor);

        store_arc
            .lock()
            .expect("SpanStore lock poisoned")
            .spans
            .entry(id.into_u64())
            .and_modify(|span| span.attrs.extend(visitor.attrs));
    }
}

fn init_global_subscriber() {
    if GLOBAL_SUB_INIT
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        let subscriber = tracing_subscriber::registry().with(GlobalSpanCollector);
        // Ignore error: if another crate already set a global subscriber, we just skip.
        let _ = tracing::subscriber::set_global_default(subscriber);
    }
}

/// Run a request with span collection. Serializes via `TEST_MUTEX` to prevent
/// parallel tests from interfering with the global `ACTIVE_STORE`.
async fn run_with_span_collection(
    llm: impl weft_llm::Provider + 'static,
    commands: impl weft_commands::CommandRegistry + 'static,
    body: serde_json::Value,
) -> (
    Arc<Mutex<SpanStore>>,
    axum::http::StatusCode,
    serde_json::Value,
) {
    init_global_subscriber();

    // Acquire the test mutex to serialize span collection. Using tokio::sync::Mutex
    // avoids the clippy::await_holding_lock lint since this mutex is async-aware.
    let _guard = TEST_MUTEX.get_or_init(|| TokioMutex::new(())).lock().await;

    let (svc, _event_log) = make_weft_service_with_event_log(llm, commands);
    let router = build_router(svc);

    // Activate a fresh store.
    let store = Arc::new(Mutex::new(SpanStore::default()));
    {
        let slot = ACTIVE_STORE.get_or_init(|| Mutex::new(None));
        *slot.lock().expect("ACTIVE_STORE lock poisoned") = Some(Arc::clone(&store));
    }

    let (status, resp) = post_json(router, body).await;

    // Deactivate the store.
    {
        let slot = ACTIVE_STORE.get_or_init(|| Mutex::new(None));
        *slot.lock().expect("ACTIVE_STORE lock poisoned") = None;
    }

    (store, status, resp)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Verify the top-level span hierarchy for a simple (no-command) request.
///
/// Expected: `request > dispatch_loop > iteration > generate`.
/// The `generate` span is inside a `tokio::spawn`; if `.instrument()` is missing
/// there, it will be an orphan root instead of a child of `iteration`.
#[tokio::test]
async fn test_span_hierarchy_simple_request() {
    let commands = ScriptedCommandRegistry::new(vec![]);
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let (store, status, _resp) =
        run_with_span_collection(TestProvider::ok("Hello!"), commands, body).await;

    assert_eq!(status, StatusCode::OK);

    let store = store.lock().expect("store lock poisoned");

    // `request` must exist.
    assert!(
        !store.find_by_name("request").is_empty(),
        "expected at least one `request` span.\nAll span names: {:?}",
        store.all_names()
    );

    // `dispatch_loop` must be a direct child of `request`.
    assert!(
        store.has_direct_child("request", "dispatch_loop"),
        "expected `dispatch_loop` to be a direct child of `request`.\nAll span names: {:?}",
        store.all_names()
    );

    // `iteration` must be a descendant of `dispatch_loop`.
    assert!(
        store.has_descendant("dispatch_loop", "iteration"),
        "expected `iteration` to be a descendant of `dispatch_loop`.\nAll span names: {:?}",
        store.all_names()
    );

    // `generate` must be a descendant of `iteration`.
    // If this fails, the tokio::spawn in generate.rs is missing .instrument().
    assert!(
        store.has_descendant("iteration", "generate"),
        "expected `generate` to be a descendant of `iteration`.\n\
         This means tokio::spawn in generate.rs is missing .instrument().\n\
         All span names: {:?}",
        store.all_names()
    );
}

/// Verify that span attributes are populated.
///
/// - `request` span: `request_id` must be a non-empty string.
/// - `generate` span: `llm.model` must be populated.
/// - `activity` spans: `activity.name` must be populated.
#[tokio::test]
async fn test_span_attributes_populated() {
    let commands = ScriptedCommandRegistry::new(vec![]);
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let (store, status, _resp) =
        run_with_span_collection(TestProvider::ok("Hello!"), commands, body).await;

    assert_eq!(status, StatusCode::OK);

    let store = store.lock().expect("store lock poisoned");

    // The reactor's `request` span must have `weft.request_id`.
    // Note: tower-http also creates a span named "request" with HTTP attributes;
    // we look for a `request` span that has `weft.request_id` specifically.
    let request_spans = store.find_by_name("request");
    assert!(
        !request_spans.is_empty(),
        "no `request` span found.\nAll span names: {:?}",
        store.all_names()
    );
    let weft_request_span = request_spans
        .iter()
        .find(|s| s.attrs.contains_key("weft.request_id"));
    assert!(
        weft_request_span.is_some(),
        "expected at least one `request` span with `weft.request_id` attribute.\n\
         Found request spans: {:?}",
        request_spans.iter().map(|s| &s.attrs).collect::<Vec<_>>()
    );
    let weft_request_id = weft_request_span
        .unwrap()
        .attrs
        .get("weft.request_id")
        .map(|s| s.as_str())
        .unwrap_or("");
    assert!(
        !weft_request_id.is_empty(),
        "`request` span `weft.request_id` must be non-empty"
    );

    // `generate` span must have `llm.model`.
    let generate_spans = store.find_by_name("generate");
    assert!(
        !generate_spans.is_empty(),
        "no `generate` span found.\nAll span names: {:?}",
        store.all_names()
    );
    for span in &generate_spans {
        let llm_model = span
            .attrs
            .get("llm.model")
            .map(|s| s.as_str())
            .unwrap_or("");
        assert!(
            !llm_model.is_empty(),
            "`generate` span must have non-empty `llm.model`. attrs: {:?}",
            span.attrs
        );
    }

    // `activity` spans must have `activity.name`.
    let activity_spans = store.find_by_name("activity");
    assert!(
        !activity_spans.is_empty(),
        "no `activity` spans found.\nAll span names: {:?}",
        store.all_names()
    );
    for span in &activity_spans {
        let activity_name = span
            .attrs
            .get("activity.name")
            .map(|s| s.as_str())
            .unwrap_or("");
        assert!(
            !activity_name.is_empty(),
            "`activity` span must have non-empty `activity.name`. attrs: {:?}",
            span.attrs
        );
    }
}

/// Verify that pre_loop activities are descendants of `request` and `post_loop`
/// activities are also under `request`.
///
/// Expected: `request > pre_loop > activity*`, `request > post_loop > activity*`.
#[tokio::test]
async fn test_pre_post_loop_spans_under_request() {
    let commands = ScriptedCommandRegistry::new(vec![]);
    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let (store, status, _resp) =
        run_with_span_collection(TestProvider::ok("Hello!"), commands, body).await;

    assert_eq!(status, StatusCode::OK);

    let store = store.lock().expect("store lock poisoned");

    // `pre_loop` must be a descendant of `request`.
    assert!(
        store.has_descendant("request", "pre_loop"),
        "expected `pre_loop` to be a descendant of `request`.\nAll span names: {:?}",
        store.all_names()
    );

    // `post_loop` must be a descendant of `request`.
    assert!(
        store.has_descendant("request", "post_loop"),
        "expected `post_loop` to be a descendant of `request`.\nAll span names: {:?}",
        store.all_names()
    );

    // At least one `activity` span must be a descendant of `pre_loop`.
    assert!(
        store.has_descendant("pre_loop", "activity"),
        "expected at least one `activity` span under `pre_loop`.\nAll span names: {:?}",
        store.all_names()
    );
}

/// Verify the command execution span hierarchy.
///
/// For a request that invokes a command, `execute_command` must be a descendant
/// of `iteration`, so that iteration-level spans correctly bracket command execution.
#[tokio::test]
async fn test_command_span_under_iteration() {
    use harness::{MockResponse, SequencedProvider};

    let commands = ScriptedCommandRegistry::new(vec![(
        "web_search",
        "Search the web",
        "results for: rust tracing",
    )]);

    // First response invokes a command; second response is the final answer.
    let provider = SequencedProvider::new(vec![
        MockResponse::WithCommands {
            text: "Searching...".into(),
            commands: vec![("web_search".into(), json!({"query": "rust tracing"}))],
        },
        MockResponse::Text("Here are the results.".into()),
    ]);

    let body = json!({
        "model": "auto",
        "messages": [{"role": "user", "content": "Search for rust tracing"}]
    });

    let (store, status, _resp) = run_with_span_collection(provider, commands, body).await;

    assert_eq!(status, StatusCode::OK);

    let store = store.lock().expect("store lock poisoned");

    // `execute_command` must be a descendant of `iteration`.
    assert!(
        store.has_descendant("iteration", "execute_command"),
        "expected `execute_command` to be a descendant of `iteration`.\nAll span names: {:?}",
        store.all_names()
    );
}
