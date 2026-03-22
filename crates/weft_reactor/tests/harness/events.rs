//! Fluent event assertion helpers for Reactor integration tests.
//!
//! `EventAssertions` wraps a list of events from a `TestEventLog` and provides
//! a chainable assertion API. Assertions short-circuit on the first failure:
//! fix the first failure before chasing later ones.
//!
//! # Quick start
//!
//! ```rust,ignore
//! event_log.assert_events(&result.execution_id).await
//!     .contains("execution.started")
//!     .contains("generation.started")
//!     .does_not_contain("activity.retried")
//!     .in_order(&["execution.started", "generation.started", "execution.completed"]);
//! ```
//!
//! # Pre-loop ordering
//!
//! Multiple `activity.started` events share the same type string, so `in_order`
//! finds only the first occurrence of each type. Use `in_order_with` with
//! predicates to assert ordering between same-type events. Each closure has a
//! distinct Rust type, so bind them to named variables first:
//!
//! ```rust,ignore
//! let validate_pred = |e: &Event| {
//!     e.payload.pointer("/event/name") == Some(&serde_json::json!("validate"))
//! };
//! let model_pred = |e: &Event| {
//!     e.payload.pointer("/event/name") == Some(&serde_json::json!("model_selection"))
//! };
//! event_log.assert_events(&id).await
//!     .in_order_with(&[
//!         ("activity.started", &validate_pred as EventPredicate<'_>),
//!         ("activity.started", &model_pred),
//!     ]);
//! ```
#![allow(dead_code)]

use weft_reactor::Event;
use weft_reactor::EventLog;
use weft_reactor::execution::ExecutionId;
use weft_reactor::test_support::TestEventLog;

/// Type alias for a predicate on an `Event`, used by [`EventAssertions::in_order_with`].
pub type EventPredicate<'a> = &'a dyn Fn(&Event) -> bool;

/// Extension trait that adds `assert_events` to `TestEventLog`.
///
/// Import this trait to enable the ergonomic `.assert_events(&id).await` syntax:
///
/// ```rust,ignore
/// use harness::TestEventLogAssertExt;
/// event_log.assert_events(&result.execution_id).await
///     .contains("execution.started");
/// ```
#[allow(async_fn_in_trait)]
pub trait TestEventLogAssertExt {
    /// Build event assertions for a given execution.
    async fn assert_events(&self, execution_id: &ExecutionId) -> EventAssertions;
}

impl TestEventLogAssertExt for TestEventLog {
    async fn assert_events(&self, execution_id: &ExecutionId) -> EventAssertions {
        EventAssertions::for_execution(self, execution_id).await
    }
}

/// Fluent event assertion builder.
///
/// Constructed from a `TestEventLog` + `ExecutionId`. Reads all events once,
/// then provides assertion methods that operate on the cached event list.
/// Methods take `self` by value and return `Self` for chaining.
pub struct EventAssertions {
    events: Vec<Event>,
    event_types: Vec<String>,
}

impl EventAssertions {
    /// Create from a `TestEventLog` and execution ID.
    ///
    /// Reads all events for the execution. Panics (via `expect`) if the
    /// execution does not exist in the log — this indicates a test setup bug.
    pub async fn for_execution(event_log: &TestEventLog, execution_id: &ExecutionId) -> Self {
        let events = event_log
            .read(execution_id, None::<u64>)
            .await
            .expect("event log read should succeed in test");
        let event_types = events.iter().map(|e| e.event_type.clone()).collect();
        Self {
            events,
            event_types,
        }
    }

    /// Assert that an event type is present at least once.
    ///
    /// # Panics
    ///
    /// Panics with the full event type sequence if the type is not found.
    pub fn contains(self, event_type: &str) -> Self {
        assert!(
            self.event_types.iter().any(|t| t == event_type),
            "expected event type '{}' not found in: {:?}",
            event_type,
            self.event_types,
        );
        self
    }

    /// Assert that an event type is NOT present.
    ///
    /// # Panics
    ///
    /// Panics with the full event type sequence if the type is found.
    pub fn does_not_contain(self, event_type: &str) -> Self {
        assert!(
            !self.event_types.iter().any(|t| t == event_type),
            "unexpected event type '{}' found in: {:?}",
            event_type,
            self.event_types,
        );
        self
    }

    /// Assert that events appear in the given order (not necessarily adjacent).
    ///
    /// Finds the **first** occurrence of each event type. If a test needs to
    /// assert ordering between two events sharing the same type string (e.g.,
    /// two `activity.started` events for different activities), use
    /// [`in_order_with`] instead.
    ///
    /// # Panics
    ///
    /// Panics if any type is absent, or if any type appears before the
    /// previous one in the sequence.
    pub fn in_order(self, expected: &[&str]) -> Self {
        let mut last_pos: Option<usize> = None;
        for expected_type in expected {
            let pos = self.event_types.iter().position(|t| t == expected_type);
            assert!(
                pos.is_some(),
                "event type '{}' not found; available: {:?}",
                expected_type,
                self.event_types,
            );
            let pos = pos.unwrap();
            if let Some(prev) = last_pos {
                assert!(
                    pos > prev,
                    "'{}' (pos {}) should come after previous event (pos {}); full sequence: {:?}",
                    expected_type,
                    pos,
                    prev,
                    self.event_types,
                );
            }
            last_pos = Some(pos);
        }
        self
    }

    /// Assert ordering of events that may share the same event type.
    ///
    /// Each entry is `(event_type, predicate)`. Finds the first event matching
    /// both the type and the predicate, then asserts it comes after the
    /// previous match.
    ///
    /// This is critical for the pre-loop ordering tests where multiple
    /// `activity.started` events must appear in a specific order. Because each
    /// closure has a distinct Rust type, bind them to named variables first:
    ///
    /// ```rust,ignore
    /// let validate_pred = |e: &Event| {
    ///     e.payload.pointer("/event/name") == Some(&serde_json::json!("validate"))
    /// };
    /// let model_pred = |e: &Event| {
    ///     e.payload.pointer("/event/name") == Some(&serde_json::json!("model_selection"))
    /// };
    /// event_log.assert_events(&id).await
    ///     .in_order_with(&[
    ///         ("activity.started", &validate_pred as EventPredicate<'_>),
    ///         ("activity.started", &model_pred),
    ///     ]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any `(type, predicate)` pair matches no event, or if the
    /// matched event appears before the previous match.
    pub fn in_order_with(self, expected: &[(&str, EventPredicate<'_>)]) -> Self {
        let mut last_pos: Option<usize> = None;
        for (event_type, predicate) in expected {
            let pos = self
                .events
                .iter()
                .position(|e| e.event_type == *event_type && predicate(e));
            assert!(
                pos.is_some(),
                "no event matching type '{}' with predicate found; available: {:?}",
                event_type,
                self.event_types,
            );
            let pos = pos.unwrap();
            if let Some(prev) = last_pos {
                assert!(
                    pos > prev,
                    "'{}' (pos {}) should come after previous match (pos {}); full sequence: {:?}",
                    event_type,
                    pos,
                    prev,
                    self.event_types,
                );
            }
            last_pos = Some(pos);
        }
        self
    }

    /// Assert the exact count of a specific event type.
    ///
    /// # Panics
    ///
    /// Panics if the actual count differs from `expected`.
    pub fn count_of(self, event_type: &str, expected: usize) -> Self {
        let actual = self
            .event_types
            .iter()
            .filter(|t| t.as_str() == event_type)
            .count();
        assert_eq!(
            actual, expected,
            "expected {} occurrences of '{}', found {}; events: {:?}",
            expected, event_type, actual, self.event_types,
        );
        self
    }

    /// Get all events for manual assertions on payloads.
    ///
    /// Use this when the fluent API is not expressive enough. The returned
    /// slice is the full event sequence in log order.
    pub fn all(&self) -> &[Event] {
        &self.events
    }

    /// Get all event types as strings.
    ///
    /// Useful for snapshot testing with `insta::assert_debug_snapshot!`.
    pub fn types(&self) -> &[String] {
        &self.event_types
    }

    /// Find events matching a type and return their payloads.
    pub fn payloads_of(&self, event_type: &str) -> Vec<&serde_json::Value> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .map(|e| &e.payload)
            .collect()
    }

    /// Assert that at least one event of `event_type` has the given value at
    /// the JSON pointer path.
    ///
    /// Uses [RFC 6901 JSON Pointer](https://tools.ietf.org/html/rfc6901) syntax,
    /// where `/` separates path components. Example: `"/event/notice/activity_name"`.
    ///
    /// # Panics
    ///
    /// Panics if no events of the given type exist, or if none have the
    /// expected value at the pointer path.
    pub fn payload_contains(
        self,
        event_type: &str,
        json_pointer: &str,
        expected_value: &serde_json::Value,
    ) -> Self {
        let payloads = self.payloads_of(event_type);
        assert!(
            !payloads.is_empty(),
            "no events of type '{}' found; available: {:?}",
            event_type,
            self.event_types,
        );
        let found = payloads
            .iter()
            .any(|p| p.pointer(json_pointer) == Some(expected_value));
        assert!(
            found,
            "no '{}' event has {} == {}; payloads: {:?}",
            event_type, json_pointer, expected_value, payloads,
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};

    use weft_reactor::execution::{Execution, ExecutionId, ExecutionStatus};
    use weft_reactor::test_support::TestEventLog;
    use weft_reactor::{EventLog, RequestId, TenantId};

    use super::{EventAssertions, EventPredicate};

    // ── Helpers ───────────────────────────────────────────────────────────────

    async fn log_with_events(events: &[(&str, serde_json::Value)]) -> (TestEventLog, ExecutionId) {
        let log = TestEventLog::new();
        let id = ExecutionId::new();
        let exec = Execution {
            id: id.clone(),
            request_id: RequestId("test-req".to_string()),
            tenant_id: TenantId("test-tenant".to_string()),
            parent_id: None,
            pipeline_name: "default".to_string(),
            status: ExecutionStatus::Running,
            created_at: chrono::Utc::now(),
            depth: 0,
        };
        log.create_execution(&exec).await.unwrap();
        for (event_type, payload) in events {
            log.append(&id, event_type, payload.clone(), 1, None)
                .await
                .unwrap();
        }
        (log, id)
    }

    // ── contains / does_not_contain ───────────────────────────────────────────

    #[tokio::test]
    async fn contains_passes_when_event_present() {
        let (log, id) = log_with_events(&[("execution.started", serde_json::json!({}))]).await;
        EventAssertions::for_execution(&log, &id)
            .await
            .contains("execution.started");
    }

    #[tokio::test]
    #[should_panic(expected = "expected event type 'execution.started' not found")]
    async fn contains_fails_when_event_absent() {
        let (log, id) = log_with_events(&[]).await;
        EventAssertions::for_execution(&log, &id)
            .await
            .contains("execution.started");
    }

    #[tokio::test]
    async fn does_not_contain_passes_when_absent() {
        let (log, id) = log_with_events(&[]).await;
        EventAssertions::for_execution(&log, &id)
            .await
            .does_not_contain("activity.retried");
    }

    #[tokio::test]
    #[should_panic(expected = "unexpected event type 'activity.retried' found")]
    async fn does_not_contain_fails_when_present() {
        let (log, id) = log_with_events(&[("activity.retried", serde_json::json!({}))]).await;
        EventAssertions::for_execution(&log, &id)
            .await
            .does_not_contain("activity.retried");
    }

    // ── in_order ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn in_order_passes_with_correct_order() {
        let (log, id) = log_with_events(&[
            ("execution.started", serde_json::json!({})),
            ("generation.started", serde_json::json!({})),
            ("execution.completed", serde_json::json!({})),
        ])
        .await;
        EventAssertions::for_execution(&log, &id).await.in_order(&[
            "execution.started",
            "generation.started",
            "execution.completed",
        ]);
    }

    #[tokio::test]
    #[should_panic(expected = "should come after previous event")]
    async fn in_order_fails_with_wrong_order() {
        let (log, id) = log_with_events(&[
            ("generation.started", serde_json::json!({})),
            ("execution.started", serde_json::json!({})),
        ])
        .await;
        EventAssertions::for_execution(&log, &id)
            .await
            .in_order(&["execution.started", "generation.started"]);
    }

    // ── in_order_with ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn in_order_with_handles_same_type_events() {
        let (log, id) = log_with_events(&[
            (
                "activity.started",
                serde_json::json!({"event": {"name": "validate"}}),
            ),
            (
                "activity.started",
                serde_json::json!({"event": {"name": "model_selection"}}),
            ),
            ("generation.started", serde_json::json!({})),
        ])
        .await;
        let validate_pred = |e: &weft_reactor::Event| {
            e.payload.pointer("/event/name") == Some(&serde_json::json!("validate"))
        };
        let model_pred = |e: &weft_reactor::Event| {
            e.payload.pointer("/event/name") == Some(&serde_json::json!("model_selection"))
        };
        let gen_pred = |_: &weft_reactor::Event| true;
        EventAssertions::for_execution(&log, &id)
            .await
            .in_order_with(&[
                ("activity.started", &validate_pred as EventPredicate<'_>),
                ("activity.started", &model_pred),
                ("generation.started", &gen_pred),
            ]);
    }

    #[tokio::test]
    #[should_panic(expected = "should come after previous match")]
    async fn in_order_with_fails_when_order_wrong() {
        let (log, id) = log_with_events(&[
            (
                "activity.started",
                serde_json::json!({"event": {"name": "model_selection"}}),
            ),
            (
                "activity.started",
                serde_json::json!({"event": {"name": "validate"}}),
            ),
        ])
        .await;
        let validate_pred = |e: &weft_reactor::Event| {
            e.payload.pointer("/event/name") == Some(&serde_json::json!("validate"))
        };
        let model_pred = |e: &weft_reactor::Event| {
            e.payload.pointer("/event/name") == Some(&serde_json::json!("model_selection"))
        };
        EventAssertions::for_execution(&log, &id)
            .await
            .in_order_with(&[
                ("activity.started", &validate_pred as EventPredicate<'_>),
                ("activity.started", &model_pred),
            ]);
    }

    // ── count_of ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn count_of_passes_with_correct_count() {
        let (log, id) = log_with_events(&[
            ("execution.degraded", serde_json::json!({})),
            ("execution.degraded", serde_json::json!({})),
        ])
        .await;
        EventAssertions::for_execution(&log, &id)
            .await
            .count_of("execution.degraded", 2);
    }

    #[tokio::test]
    #[should_panic(expected = "expected 1 occurrences of 'execution.degraded', found 2")]
    async fn count_of_fails_with_wrong_count() {
        let (log, id) = log_with_events(&[
            ("execution.degraded", serde_json::json!({})),
            ("execution.degraded", serde_json::json!({})),
        ])
        .await;
        EventAssertions::for_execution(&log, &id)
            .await
            .count_of("execution.degraded", 1);
    }

    // ── payload_contains ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn payload_contains_passes_when_matching_payload_exists() {
        let (log, id) = log_with_events(&[(
            "execution.degraded",
            serde_json::json!({"event": {"notice": {"activity_name": "command_selection"}}}),
        )])
        .await;
        EventAssertions::for_execution(&log, &id)
            .await
            .payload_contains(
                "execution.degraded",
                "/event/notice/activity_name",
                &serde_json::json!("command_selection"),
            );
    }

    #[tokio::test]
    #[should_panic(expected = "no 'execution.degraded' event has /event/notice/activity_name")]
    async fn payload_contains_fails_when_no_matching_payload() {
        let (log, id) = log_with_events(&[(
            "execution.degraded",
            serde_json::json!({"event": {"notice": {"activity_name": "something_else"}}}),
        )])
        .await;
        EventAssertions::for_execution(&log, &id)
            .await
            .payload_contains(
                "execution.degraded",
                "/event/notice/activity_name",
                &serde_json::json!("command_selection"),
            );
    }

    // ── all / types ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn all_returns_all_events() {
        let (log, id) = log_with_events(&[
            ("execution.started", serde_json::json!({})),
            ("execution.completed", serde_json::json!({})),
        ])
        .await;
        let assertions = EventAssertions::for_execution(&log, &id).await;
        assert_eq!(assertions.all().len(), 2);
    }

    #[tokio::test]
    async fn types_returns_event_type_strings() {
        let (log, id) = log_with_events(&[
            ("execution.started", serde_json::json!({})),
            ("generation.started", serde_json::json!({})),
        ])
        .await;
        let assertions = EventAssertions::for_execution(&log, &id).await;
        let types = assertions.types();
        assert_eq!(types, &["execution.started", "generation.started"]);
    }

    // ── chaining ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn fluent_chain_combines_multiple_assertions() {
        let (log, id) = log_with_events(&[
            ("execution.started", serde_json::json!({})),
            ("generation.started", serde_json::json!({})),
            ("execution.completed", serde_json::json!({})),
        ])
        .await;
        EventAssertions::for_execution(&log, &id)
            .await
            .contains("execution.started")
            .contains("generation.started")
            .does_not_contain("activity.retried")
            .count_of("execution.started", 1)
            .in_order(&[
                "execution.started",
                "generation.started",
                "execution.completed",
            ]);
    }
}
