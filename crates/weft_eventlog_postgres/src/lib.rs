//! PostgreSQL-backed EventLog implementation for production use.
//!
//! This crate provides [`PostgresEventLog`], a durable implementation of the
//! [`weft_reactor_trait::EventLog`] trait backed by PostgreSQL.
//!
//! # Usage
//!
//! Apply `schema.sql` to your database before constructing this type:
//!
//! ```bash
//! psql "$DATABASE_URL" -f crates/weft_eventlog_postgres/schema.sql
//! ```
//!
//! Then wire it into a Reactor:
//!
//! ```rust,no_run
//! use sqlx::PgPool;
//! use weft_eventlog_postgres::PostgresEventLog;
//!
//! # async fn example() {
//! let pool = PgPool::connect("postgres://localhost/weft").await.unwrap();
//! let event_log = PostgresEventLog::new(pool);
//! // Pass event_log to Reactor::new(...)
//! # }
//! ```
//!
//! # Signal polling
//!
//! Includes a background task that polls `pipeline_signals` and pushes
//! `PipelineEvent::Signal(..)` onto the execution's event channel. Insert
//! rows into `pipeline_signals` from external systems to inject signals into
//! a running execution.
//!
//! # Schema
//!
//! Requires the schema from `schema.sql` to be applied before use.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::postgres::PgRow;
use sqlx::{PgPool, Row};
use tracing::{error, warn};
use uuid::Uuid;

use weft_reactor_trait::PipelineEvent;
use weft_reactor_trait::event::Event;
use weft_reactor_trait::event_log::{EventLog, EventLogError};
use weft_reactor_trait::execution::{Execution, ExecutionId, ExecutionStatus};
use weft_reactor_trait::signal::Signal;

/// PostgreSQL-backed event log for production use.
///
/// Persists all execution records and events to PostgreSQL using the schema
/// defined in `schema.sql`. Thread-safe: clone or wrap in `Arc` to share
/// across tasks. Clone is cheap — the inner `PgPool` is wrapped in `Arc`.
///
/// Construct with a `sqlx::PgPool` obtained from [`sqlx::PgPool::connect`]
/// or [`sqlx::PgPoolOptions`]. The schema must exist before any methods are
/// called.
#[derive(Clone)]
pub struct PostgresEventLog {
    pool: Arc<PgPool>,
}

impl PostgresEventLog {
    /// Create a new `PostgresEventLog` wrapping the given connection pool.
    ///
    /// The schema from `schema.sql` must be applied to the database before
    /// calling any trait methods.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }

    /// Start a background task that polls `pipeline_signals` for the given
    /// execution and pushes each unconsumed signal onto `event_tx`.
    ///
    /// The task runs until `event_tx` is closed (the channel is dropped on the
    /// receiver side). Signals are marked consumed after being pushed onto the
    /// channel.
    ///
    /// Returns a `tokio::task::JoinHandle` that can be aborted or awaited by
    /// the caller.
    pub fn start_signal_poller(
        &self,
        execution_id: ExecutionId,
        event_tx: tokio::sync::mpsc::Sender<PipelineEvent>,
    ) -> tokio::task::JoinHandle<()> {
        let pool = Arc::clone(&self.pool);
        tokio::spawn(async move {
            signal_poller(pool, execution_id, event_tx).await;
        })
    }
}

/// Background task: poll `pipeline_signals` and push onto the event channel.
///
/// Runs until the event channel is closed. Polls every 100ms.
async fn signal_poller(
    pool: Arc<PgPool>,
    execution_id: ExecutionId,
    event_tx: tokio::sync::mpsc::Sender<PipelineEvent>,
) {
    let exec_id = execution_id.0;

    loop {
        // Bail out if the Reactor has dropped the channel.
        if event_tx.is_closed() {
            break;
        }

        let rows: Vec<(i64, serde_json::Value)> = match sqlx::query(
            "SELECT signal_id, payload \
             FROM pipeline_signals \
             WHERE execution_id = $1 AND consumed = FALSE \
             ORDER BY signal_id ASC",
        )
        .bind(exec_id)
        .fetch_all(pool.as_ref())
        .await
        {
            Ok(rows) => rows
                .into_iter()
                .map(|r: PgRow| {
                    let id: i64 = r.get("signal_id");
                    let payload: serde_json::Value = r.get("payload");
                    (id, payload)
                })
                .collect(),
            Err(err) => {
                error!(
                    execution_id = %exec_id,
                    error = %err,
                    "signal_poller: failed to query pipeline_signals"
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };

        for (signal_id, payload) in rows {
            // Deserialize the signal payload.
            let signal: Signal = match serde_json::from_value(payload) {
                Ok(s) => s,
                Err(err) => {
                    warn!(
                        execution_id = %exec_id,
                        signal_id,
                        error = %err,
                        "signal_poller: failed to deserialize signal payload, skipping"
                    );
                    // Mark consumed so we don't loop forever on a bad row.
                    let _ = mark_signal_consumed(pool.as_ref(), signal_id).await;
                    continue;
                }
            };

            // Push onto the event channel. If the channel is closed, stop polling.
            if event_tx
                .send(PipelineEvent::Signal(weft_reactor_trait::SignalEvent::Received(
                    signal,
                )))
                .await
                .is_err()
            {
                // Receiver dropped — execution is over.
                return;
            }

            // Mark consumed only after successfully pushing onto the channel.
            if let Err(err) = mark_signal_consumed(pool.as_ref(), signal_id).await {
                error!(
                    execution_id = %exec_id,
                    signal_id,
                    error = %err,
                    "signal_poller: failed to mark signal consumed"
                );
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Mark a signal row as consumed in `pipeline_signals`.
async fn mark_signal_consumed(pool: &PgPool, signal_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE pipeline_signals SET consumed = TRUE WHERE signal_id = $1")
        .bind(signal_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Convert an [`ExecutionStatus`] to the string stored in the database.
fn status_to_str(status: ExecutionStatus) -> &'static str {
    match status {
        ExecutionStatus::Running => "running",
        ExecutionStatus::Completed => "completed",
        ExecutionStatus::Failed => "failed",
        ExecutionStatus::Cancelled => "cancelled",
    }
}

/// Map a `PgRow` from `pipeline_events` to an [`Event`].
///
/// Panics only if column names are wrong (programming error).
fn row_to_event(row: PgRow) -> Event {
    let execution_id: Uuid = row.get("execution_id");
    let sequence_num: i32 = row.get("sequence_num");
    let event_type: String = row.get("event_type");
    let payload: serde_json::Value = row.get("payload");
    let schema_version: i32 = row.get("schema_version");
    let created_at: DateTime<Utc> = row.get("created_at");

    Event {
        execution_id: ExecutionId(execution_id),
        sequence: sequence_num as u64,
        event_type,
        payload,
        timestamp: created_at,
        schema_version: schema_version as u32,
    }
}

/// Check whether an execution ID exists in `pipeline_executions`.
///
/// Returns `EventLogError::ExecutionNotFound` when missing.
async fn check_execution_exists(
    pool: &PgPool,
    execution_id: &ExecutionId,
) -> Result<(), EventLogError> {
    let exec_id = execution_id.0;
    let count: Option<i64> =
        sqlx::query_scalar("SELECT COUNT(*) FROM pipeline_executions WHERE execution_id = $1")
            .bind(exec_id)
            .fetch_one(pool)
            .await
            .map_err(|err| EventLogError::Storage(err.to_string()))?;

    if count.unwrap_or(0) == 0 {
        return Err(EventLogError::ExecutionNotFound(execution_id.clone()));
    }
    Ok(())
}

#[async_trait]
impl EventLog for PostgresEventLog {
    /// Create a new execution record in `pipeline_executions`.
    ///
    /// Returns [`EventLogError::Storage`] if the execution already exists.
    async fn create_execution(&self, execution: &Execution) -> Result<(), EventLogError> {
        let exec_id = execution.id.0;
        let tenant_id = &execution.tenant_id.0;
        let request_id = &execution.request_id.0;
        let parent_id: Option<Uuid> = execution.parent_id.as_ref().map(|p| p.0);
        let pipeline_name = &execution.pipeline_name;
        let status = status_to_str(execution.status);
        let depth = execution.depth as i32;
        let created_at = execution.created_at;
        // Budget is stored as an empty JSON object; full budget state travels
        // via the ExecutionStarted event payload.
        let budget_json = serde_json::json!({});

        sqlx::query(
            "INSERT INTO pipeline_executions \
             (execution_id, tenant_id, request_id, parent_id, pipeline_name, status, depth, created_at, budget_json) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(exec_id)
        .bind(tenant_id)
        .bind(request_id)
        .bind(parent_id)
        .bind(pipeline_name)
        .bind(status)
        .bind(depth)
        .bind(created_at)
        .bind(budget_json)
        .execute(self.pool.as_ref())
        .await
        .map_err(|err| {
            // Postgres unique_violation code = 23505
            if let sqlx::Error::Database(ref db_err) = err
                && db_err.code().as_deref() == Some("23505")
            {
                return EventLogError::Storage(format!(
                    "execution {} already exists",
                    execution.id
                ));
            }
            EventLogError::Storage(err.to_string())
        })?;

        Ok(())
    }

    /// Update execution status and set `completed_at` for terminal states.
    ///
    /// Returns [`EventLogError::ExecutionNotFound`] if no row matches.
    async fn update_execution_status(
        &self,
        execution_id: &ExecutionId,
        status: ExecutionStatus,
    ) -> Result<(), EventLogError> {
        let exec_id = execution_id.0;
        let status_str = status_to_str(status);

        // Set completed_at for terminal states.
        let completed_at: Option<DateTime<Utc>> = match status {
            ExecutionStatus::Completed | ExecutionStatus::Failed | ExecutionStatus::Cancelled => {
                Some(Utc::now())
            }
            ExecutionStatus::Running => None,
        };

        let result = sqlx::query(
            "UPDATE pipeline_executions \
             SET status = $2, \
                 completed_at = COALESCE($3, completed_at) \
             WHERE execution_id = $1",
        )
        .bind(exec_id)
        .bind(status_str)
        .bind(completed_at)
        .execute(self.pool.as_ref())
        .await
        .map_err(|err| EventLogError::Storage(err.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(EventLogError::ExecutionNotFound(execution_id.clone()));
        }

        Ok(())
    }

    /// Append an event to `pipeline_events`. Returns the assigned sequence number.
    ///
    /// Sequence numbers are atomically assigned via a subquery
    /// (`COALESCE(MAX(sequence_num), 0) + 1`) within the INSERT, ensuring
    /// monotonic ordering even under concurrent appends.
    ///
    /// When `idempotency_key` is provided and a row with that key already
    /// exists for this execution, returns the existing row's sequence number
    /// without inserting a duplicate.
    async fn append(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
        payload: serde_json::Value,
        schema_version: u32,
        idempotency_key: Option<&str>,
    ) -> Result<u64, EventLogError> {
        let exec_id = execution_id.0;
        let schema_version_i32 = schema_version as i32;

        if let Some(key) = idempotency_key {
            // Attempt idempotent insert: ON CONFLICT (execution_id, idempotency_key) DO NOTHING.
            // The unique index covers only non-null idempotency_key values.
            let result: Option<PgRow> = sqlx::query(
                "INSERT INTO pipeline_events \
                 (execution_id, sequence_num, event_type, payload, schema_version, idempotency_key) \
                 SELECT $1, (COALESCE(MAX(sequence_num), 0) + 1), $2, $3, $4, $5 \
                 FROM pipeline_events \
                 WHERE execution_id = $1 \
                 ON CONFLICT (execution_id, idempotency_key) DO NOTHING \
                 RETURNING sequence_num",
            )
            .bind(exec_id)
            .bind(event_type)
            .bind(&payload)
            .bind(schema_version_i32)
            .bind(key)
            .fetch_optional(self.pool.as_ref())
            .await
            .map_err(|err| {
                if let sqlx::Error::Database(ref db_err) = err
                    && db_err.code().as_deref() == Some("23503")
                {
                    return EventLogError::ExecutionNotFound(execution_id.clone());
                }
                EventLogError::Storage(err.to_string())
            })?;

            if let Some(row) = result {
                // Fresh insert succeeded.
                let seq: i32 = row.get("sequence_num");
                return Ok(seq as u64);
            }

            // Insert was a no-op due to conflict: fetch the existing sequence.
            let existing: Option<PgRow> = sqlx::query(
                "SELECT sequence_num \
                 FROM pipeline_events \
                 WHERE execution_id = $1 AND idempotency_key = $2",
            )
            .bind(exec_id)
            .bind(key)
            .fetch_optional(self.pool.as_ref())
            .await
            .map_err(|err| EventLogError::Storage(err.to_string()))?;

            match existing {
                Some(row) => {
                    let seq: i32 = row.get("sequence_num");
                    Ok(seq as u64)
                }
                None => Err(EventLogError::Storage(format!(
                    "idempotency conflict for key '{}' but row not found",
                    key
                ))),
            }
        } else {
            // No idempotency key: plain insert with atomic sequence assignment.
            let row: PgRow = sqlx::query(
                "INSERT INTO pipeline_events \
                 (execution_id, sequence_num, event_type, payload, schema_version) \
                 SELECT $1, (COALESCE(MAX(sequence_num), 0) + 1), $2, $3, $4 \
                 FROM pipeline_events \
                 WHERE execution_id = $1 \
                 RETURNING sequence_num",
            )
            .bind(exec_id)
            .bind(event_type)
            .bind(&payload)
            .bind(schema_version_i32)
            .fetch_one(self.pool.as_ref())
            .await
            .map_err(|err| {
                // Postgres FK violation = 23503
                if let sqlx::Error::Database(ref db_err) = err
                    && db_err.code().as_deref() == Some("23503")
                {
                    return EventLogError::ExecutionNotFound(execution_id.clone());
                }
                EventLogError::Storage(err.to_string())
            })?;

            let seq: i32 = row.get("sequence_num");
            Ok(seq as u64)
        }
    }

    /// Read events for an execution in sequence order.
    ///
    /// If `after_sequence` is `Some(n)`, returns only events with
    /// `sequence_num > n`. If `None`, returns all events.
    async fn read(
        &self,
        execution_id: &ExecutionId,
        after_sequence: Option<u64>,
    ) -> Result<Vec<Event>, EventLogError> {
        check_execution_exists(self.pool.as_ref(), execution_id).await?;

        let exec_id = execution_id.0;
        let after_seq = after_sequence.unwrap_or(0) as i32;

        let rows: Vec<PgRow> = sqlx::query(
            "SELECT execution_id, sequence_num, event_type, payload, schema_version, created_at \
             FROM pipeline_events \
             WHERE execution_id = $1 AND sequence_num > $2 \
             ORDER BY sequence_num ASC",
        )
        .bind(exec_id)
        .bind(after_seq)
        .fetch_all(self.pool.as_ref())
        .await
        .map_err(|err| EventLogError::Storage(err.to_string()))?;

        Ok(rows.into_iter().map(row_to_event).collect())
    }

    /// Read the latest event of a specific type for an execution.
    ///
    /// Returns `None` if no matching event exists.
    /// Returns [`EventLogError::ExecutionNotFound`] if the execution does not exist.
    async fn latest_of_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<Option<Event>, EventLogError> {
        check_execution_exists(self.pool.as_ref(), execution_id).await?;

        let exec_id = execution_id.0;

        let row: Option<PgRow> = sqlx::query(
            "SELECT execution_id, sequence_num, event_type, payload, schema_version, created_at \
             FROM pipeline_events \
             WHERE execution_id = $1 AND event_type = $2 \
             ORDER BY sequence_num DESC \
             LIMIT 1",
        )
        .bind(exec_id)
        .bind(event_type)
        .fetch_optional(self.pool.as_ref())
        .await
        .map_err(|err| EventLogError::Storage(err.to_string()))?;

        Ok(row.map(row_to_event))
    }

    /// Count events of a specific type for an execution.
    ///
    /// Returns 0 if no matching events exist.
    /// Returns [`EventLogError::ExecutionNotFound`] if the execution does not exist.
    async fn count_by_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<u64, EventLogError> {
        check_execution_exists(self.pool.as_ref(), execution_id).await?;

        let exec_id = execution_id.0;

        let count: Option<i64> = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pipeline_events \
             WHERE execution_id = $1 AND event_type = $2",
        )
        .bind(exec_id)
        .bind(event_type)
        .fetch_one(self.pool.as_ref())
        .await
        .map_err(|err| EventLogError::Storage(err.to_string()))?;

        Ok(count.unwrap_or(0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weft_reactor_trait::event::EVENT_SCHEMA_VERSION;
    use weft_reactor_trait::execution::{RequestId, TenantId};

    fn make_execution() -> Execution {
        Execution {
            id: ExecutionId::new(),
            tenant_id: TenantId("tenant-1".to_string()),
            request_id: RequestId("req-1".to_string()),
            parent_id: None,
            pipeline_name: "default".to_string(),
            status: ExecutionStatus::Running,
            created_at: Utc::now(),
            depth: 0,
        }
    }

    async fn connect_pool(database_url: &str) -> PgPool {
        PgPool::connect(database_url)
            .await
            .expect("failed to connect to Postgres")
    }

    // ─── Integration tests (skipped without DATABASE_URL) ───────────────────

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_create_execution() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec)
            .await
            .expect("create_execution failed");
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_create_execution_duplicate_fails() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec)
            .await
            .expect("first create failed");
        let err = log
            .create_execution(&exec)
            .await
            .expect_err("second create should fail");
        assert!(matches!(err, EventLogError::Storage(_)));
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_update_execution_status() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");
        log.update_execution_status(&exec.id, ExecutionStatus::Completed)
            .await
            .expect("update_execution_status failed");
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_update_execution_status_not_found() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let fake_id = ExecutionId::new();
        let err = log
            .update_execution_status(&fake_id, ExecutionStatus::Completed)
            .await
            .expect_err("should fail for unknown execution");
        assert!(matches!(err, EventLogError::ExecutionNotFound(_)));
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_append_returns_sequence() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        let seq1 = log
            .append(
                &exec.id,
                "execution.started",
                serde_json::Value::Null,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .expect("append 1 failed");
        assert_eq!(seq1, 1);

        let seq2 = log
            .append(
                &exec.id,
                "activity.started",
                serde_json::Value::Null,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .expect("append 2 failed");
        assert_eq!(seq2, 2);
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_append_idempotency_deduplication() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        let payload = serde_json::json!({"idempotency_key": "key-1"});

        let seq1 = log
            .append(
                &exec.id,
                "activity.completed",
                payload.clone(),
                EVENT_SCHEMA_VERSION,
                Some("key-1"),
            )
            .await
            .expect("first append failed");

        // Second append with same idempotency key must return the same sequence.
        let seq2 = log
            .append(
                &exec.id,
                "activity.completed",
                payload.clone(),
                EVENT_SCHEMA_VERSION,
                Some("key-1"),
            )
            .await
            .expect("second append failed");

        assert_eq!(seq1, seq2, "idempotent appends must return same sequence");

        // Verify only one row was inserted.
        let events = log.read(&exec.id, None).await.expect("read failed");
        assert_eq!(events.len(), 1, "only one row should exist");
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_read_all_events() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        for i in 0..3_i32 {
            log.append(
                &exec.id,
                "test.event",
                serde_json::json!({"n": i}),
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .expect("append failed");
        }

        let events = log.read(&exec.id, None).await.expect("read failed");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[1].sequence, 2);
        assert_eq!(events[2].sequence, 3);
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_read_after_sequence() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        for i in 0..5_i32 {
            log.append(
                &exec.id,
                "test.event",
                serde_json::json!({"n": i}),
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .expect("append failed");
        }

        let events = log.read(&exec.id, Some(2)).await.expect("read failed");
        assert_eq!(events.len(), 3, "should return sequences 3, 4, 5");
        assert_eq!(events[0].sequence, 3);
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_read_not_found_fails() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let fake_id = ExecutionId::new();
        let err = log.read(&fake_id, None).await.expect_err("should fail");
        assert!(matches!(err, EventLogError::ExecutionNotFound(_)));
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_latest_of_type() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        log.append(
            &exec.id,
            "generation.started",
            serde_json::json!({"model": "gpt-4", "call": 1}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .expect("append 1 failed");

        log.append(
            &exec.id,
            "generation.completed",
            serde_json::json!({"model": "gpt-4"}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .expect("append 2 failed");

        log.append(
            &exec.id,
            "generation.started",
            serde_json::json!({"model": "gpt-4", "call": 2}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .expect("append 3 failed");

        let latest = log
            .latest_of_type(&exec.id, "generation.started")
            .await
            .expect("latest_of_type failed")
            .expect("expected Some event");

        assert_eq!(latest.sequence, 3);
        assert_eq!(latest.payload["call"], 2);
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_latest_of_type_returns_none_when_absent() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        let result = log
            .latest_of_type(&exec.id, "generation.started")
            .await
            .expect("latest_of_type failed");
        assert!(result.is_none());
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_count_by_type() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        for _ in 0..3 {
            log.append(
                &exec.id,
                "generation.completed",
                serde_json::Value::Null,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .expect("append failed");
        }

        log.append(
            &exec.id,
            "activity.started",
            serde_json::Value::Null,
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .expect("append failed");

        let count = log
            .count_by_type(&exec.id, "generation.completed")
            .await
            .expect("count failed");
        assert_eq!(count, 3);

        let other = log
            .count_by_type(&exec.id, "activity.started")
            .await
            .expect("count failed");
        assert_eq!(other, 1);
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_event_schema_version_round_trips() {
        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        log.append(
            &exec.id,
            "test.event",
            serde_json::json!({"x": 42}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .expect("append failed");

        let events = log.read(&exec.id, None).await.expect("read failed");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].schema_version, EVENT_SCHEMA_VERSION);
        assert_eq!(events[0].schema_version, 1);
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_pipeline_event_jsonb_round_trip() {
        use weft_reactor_trait::event::{GeneratedEvent, GenerationEvent};

        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool);
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        // Serialize an actual PipelineEvent variant to JSON, store it, read it
        // back, and deserialize to verify the JSONB round-trip path works end-to-end:
        // PipelineEvent -> serde_json::Value -> Postgres JSONB -> serde_json::Value -> PipelineEvent.
        let original = PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Done));
        let payload = serde_json::to_value(&original).expect("serialize PipelineEvent");

        let seq = log
            .append(
                &exec.id,
                original.event_type_string(),
                payload,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .expect("append failed");
        assert_eq!(seq, 1);

        let events = log.read(&exec.id, None).await.expect("read failed");
        assert_eq!(events.len(), 1);

        // Deserialize the stored payload back to PipelineEvent.
        let round_tripped: PipelineEvent =
            serde_json::from_value(events[0].payload.clone()).expect("deserialize PipelineEvent");
        assert!(
            matches!(
                round_tripped,
                PipelineEvent::Generation(GenerationEvent::Chunk(GeneratedEvent::Done))
            ),
            "expected Generation(Chunk(Done)), got {:?}",
            round_tripped
        );
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL and schema.sql applied"]
    async fn test_signal_poller_pushes_signal() {
        use tokio::sync::mpsc;

        let url = match std::env::var("DATABASE_URL").ok() {
            Some(u) => u,
            None => return,
        };
        let pool = connect_pool(&url).await;
        let log = PostgresEventLog::new(pool.clone());
        let exec = make_execution();
        log.create_execution(&exec).await.expect("create failed");

        // Insert a signal directly via SQL.
        let exec_id = exec.id.0;
        let signal = Signal::Cancel {
            reason: "test".to_string(),
        };
        let payload = serde_json::to_value(&signal).expect("serialize signal");
        sqlx::query(
            "INSERT INTO pipeline_signals (execution_id, signal_type, payload) VALUES ($1, $2, $3)",
        )
        .bind(exec_id)
        .bind("cancel")
        .bind(payload)
        .execute(&pool)
        .await
        .expect("insert signal failed");

        // Start the signal poller and verify the signal arrives on the channel.
        let (tx, mut rx) = mpsc::channel(8);
        let _handle = log.start_signal_poller(exec.id.clone(), tx);

        let received = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for signal")
            .expect("channel closed");

        assert!(
            matches!(
                received,
                PipelineEvent::Signal(weft_reactor_trait::event::SignalEvent::Received(
                    Signal::Cancel { .. }
                ))
            ),
            "expected Cancel signal, got {:?}",
            received
        );
    }
}
