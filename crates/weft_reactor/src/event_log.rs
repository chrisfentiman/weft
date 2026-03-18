//! EventLog trait and associated error type.

use crate::event::Event;
use crate::execution::{Execution, ExecutionId, ExecutionStatus};

/// The durable event store.
///
/// Events are appended, never modified. The event log is the single
/// source of truth for execution state.
///
/// This trait lives in `weft_reactor`. Implementations live in separate crates:
/// - `weft_eventlog_memory::InMemoryEventLog`: Vec<Event> behind a RwLock. For tests and local dev.
/// - `weft_eventlog_postgres::PostgresEventLog`: Writes to `pipeline_events` table. For production.
///
/// Third parties can implement this trait without modifying Weft source.
///
/// The trait includes execution lifecycle methods (create, update status)
/// because the execution table and event table are tightly coupled --
/// creating an execution and appending its first event should be atomic.
#[async_trait::async_trait]
pub trait EventLog: Send + Sync + 'static {
    /// Create a new execution record.
    async fn create_execution(&self, execution: &Execution) -> Result<(), EventLogError>;

    /// Update execution status. Called when the execution completes,
    /// fails, or is cancelled.
    async fn update_execution_status(
        &self,
        execution_id: &ExecutionId,
        status: ExecutionStatus,
    ) -> Result<(), EventLogError>;

    /// Append an event to the log. Returns the assigned sequence number.
    ///
    /// The EventLog implementation assigns the sequence number (monotonically
    /// increasing per execution) and the current timestamp. The caller provides
    /// event_type, payload, schema_version, and optional idempotency_key.
    ///
    /// If `idempotency_key` is Some and an event with the same key already
    /// exists for this execution, the implementation returns the existing
    /// event's sequence number without inserting a duplicate.
    async fn append(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
        payload: serde_json::Value,
        schema_version: u32,
        idempotency_key: Option<&str>,
    ) -> Result<u64, EventLogError>;

    /// Read events for an execution, in sequence order.
    ///
    /// If `after_sequence` is Some(n), returns only events with sequence > n.
    /// If None, returns all events.
    async fn read(
        &self,
        execution_id: &ExecutionId,
        after_sequence: Option<u64>,
    ) -> Result<Vec<Event>, EventLogError>;

    /// Read the latest event of a specific type for an execution.
    async fn latest_of_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<Option<Event>, EventLogError>;

    /// Count events of a specific type for an execution.
    ///
    /// Used for budget queries: "how many generation.completed events?"
    async fn count_by_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<u64, EventLogError>;
}

#[derive(Debug, thiserror::Error)]
pub enum EventLogError {
    #[error("storage error: {0}")]
    Storage(String),
    #[error("execution not found: {0}")]
    ExecutionNotFound(ExecutionId),
    #[error("sequence conflict for execution {execution_id}: expected {expected}, got {actual}")]
    SequenceConflict {
        execution_id: ExecutionId,
        expected: u64,
        actual: u64,
    },
    #[error("serialization error: {0}")]
    Serialization(String),
}
