//! In-memory EventLog implementation for tests and local development.

use std::collections::HashMap;
use std::sync::RwLock;

use weft_reactor::event::Event;
use weft_reactor::event_log::{EventLog, EventLogError};
use weft_reactor::execution::{Execution, ExecutionId, ExecutionStatus};

/// In-memory event log for tests and local development.
///
/// All data is held in a `RwLock<HashMap>`. No persistence --
/// data is lost when the process exits.
///
/// Thread-safe: multiple Reactor instances can share one InMemoryEventLog
/// (each root execution gets its own event list in the HashMap).
pub struct InMemoryEventLog {
    /// Events keyed by execution ID.
    events: RwLock<HashMap<ExecutionId, Vec<Event>>>,
    /// Execution records keyed by execution ID.
    executions: RwLock<HashMap<ExecutionId, Execution>>,
}

impl InMemoryEventLog {
    pub fn new() -> Self {
        Self {
            events: RwLock::new(HashMap::new()),
            executions: RwLock::new(HashMap::new()),
        }
    }

    /// Read all events across all executions. Test utility only.
    pub fn all_events(&self) -> Vec<Event> {
        let guard = self.events.read().expect("poisoned lock");
        guard.values().flat_map(|v| v.iter().cloned()).collect()
    }

    /// Get the count of events for a specific execution. Test utility.
    pub fn event_count(&self, execution_id: &ExecutionId) -> usize {
        let guard = self.events.read().expect("poisoned lock");
        guard.get(execution_id).map(|v| v.len()).unwrap_or(0)
    }
}

impl Default for InMemoryEventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl EventLog for InMemoryEventLog {
    async fn create_execution(&self, execution: &Execution) -> Result<(), EventLogError> {
        let mut exec_map = self.executions.write().expect("poisoned lock");
        let mut event_map = self.events.write().expect("poisoned lock");

        if exec_map.contains_key(&execution.id) {
            return Err(EventLogError::Storage(format!(
                "execution {} already exists",
                execution.id
            )));
        }

        exec_map.insert(execution.id.clone(), execution.clone());
        event_map.insert(execution.id.clone(), Vec::new());
        Ok(())
    }

    async fn update_execution_status(
        &self,
        execution_id: &ExecutionId,
        status: ExecutionStatus,
    ) -> Result<(), EventLogError> {
        let mut exec_map = self.executions.write().expect("poisoned lock");
        match exec_map.get_mut(execution_id) {
            Some(exec) => {
                exec.status = status;
                Ok(())
            }
            None => Err(EventLogError::ExecutionNotFound(execution_id.clone())),
        }
    }

    async fn append(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
        payload: serde_json::Value,
        schema_version: u32,
        idempotency_key: Option<&str>,
    ) -> Result<u64, EventLogError> {
        let mut event_map = self.events.write().expect("poisoned lock");

        let events = event_map
            .get_mut(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;

        // Idempotency: check for an existing event with the same key
        if let Some(key) = idempotency_key {
            // Search for an existing event that has this idempotency key in its payload
            for existing in events.iter() {
                if let Some(existing_key) = existing.payload.get("idempotency_key")
                    && existing_key.as_str() == Some(key)
                {
                    return Ok(existing.sequence);
                }
            }
        }

        // Sequence is 1-indexed, monotonically increasing per execution
        let sequence = events.len() as u64 + 1;
        let event = Event {
            execution_id: execution_id.clone(),
            sequence,
            event_type: event_type.to_string(),
            payload,
            timestamp: chrono::Utc::now(),
            schema_version,
        };
        events.push(event);
        Ok(sequence)
    }

    async fn read(
        &self,
        execution_id: &ExecutionId,
        after_sequence: Option<u64>,
    ) -> Result<Vec<Event>, EventLogError> {
        let event_map = self.events.read().expect("poisoned lock");
        let events = event_map
            .get(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;

        let result = match after_sequence {
            None => events.clone(),
            Some(n) => events.iter().filter(|e| e.sequence > n).cloned().collect(),
        };
        Ok(result)
    }

    async fn latest_of_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<Option<Event>, EventLogError> {
        let event_map = self.events.read().expect("poisoned lock");
        let events = event_map
            .get(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;

        // Iterate in reverse to find the latest matching event
        let result = events
            .iter()
            .rev()
            .find(|e| e.event_type == event_type)
            .cloned();
        Ok(result)
    }

    async fn count_by_type(
        &self,
        execution_id: &ExecutionId,
        event_type: &str,
    ) -> Result<u64, EventLogError> {
        let event_map = self.events.read().expect("poisoned lock");
        let events = event_map
            .get(execution_id)
            .ok_or_else(|| EventLogError::ExecutionNotFound(execution_id.clone()))?;

        let count = events.iter().filter(|e| e.event_type == event_type).count() as u64;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use weft_reactor::event::EVENT_SCHEMA_VERSION;
    use weft_reactor::execution::{RequestId, TenantId};

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

    #[tokio::test]
    async fn test_create_execution() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();
        assert_eq!(log.event_count(&exec.id), 0);
    }

    #[tokio::test]
    async fn test_create_execution_duplicate_fails() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();
        let err = log.create_execution(&exec).await.unwrap_err();
        assert!(matches!(err, EventLogError::Storage(_)));
    }

    #[tokio::test]
    async fn test_update_execution_status() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();
        log.update_execution_status(&exec.id, ExecutionStatus::Completed)
            .await
            .unwrap();
        // Verify status was updated by checking the internal state
        let exec_map = log.executions.read().unwrap();
        let stored = exec_map.get(&exec.id).unwrap();
        assert_eq!(stored.status, ExecutionStatus::Completed);
    }

    #[tokio::test]
    async fn test_update_execution_status_not_found() {
        let log = InMemoryEventLog::new();
        let fake_id = ExecutionId::new();
        let err = log
            .update_execution_status(&fake_id, ExecutionStatus::Completed)
            .await
            .unwrap_err();
        assert!(matches!(err, EventLogError::ExecutionNotFound(_)));
    }

    #[tokio::test]
    async fn test_append_returns_sequence() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        let seq1 = log
            .append(
                &exec.id,
                "execution.started",
                serde_json::Value::Null,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .unwrap();
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
            .unwrap();
        assert_eq!(seq2, 2);
    }

    #[tokio::test]
    async fn test_append_not_found_fails() {
        let log = InMemoryEventLog::new();
        let fake_id = ExecutionId::new();
        let err = log
            .append(
                &fake_id,
                "execution.started",
                serde_json::Value::Null,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, EventLogError::ExecutionNotFound(_)));
    }

    #[tokio::test]
    async fn test_append_idempotency_deduplication() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        let payload = serde_json::json!({"idempotency_key": "exec-1:generate:0"});

        // First append
        let seq1 = log
            .append(
                &exec.id,
                "activity.completed",
                payload.clone(),
                EVENT_SCHEMA_VERSION,
                Some("exec-1:generate:0"),
            )
            .await
            .unwrap();

        // Second append with same idempotency key should return existing sequence
        let seq2 = log
            .append(
                &exec.id,
                "activity.completed",
                payload.clone(),
                EVENT_SCHEMA_VERSION,
                Some("exec-1:generate:0"),
            )
            .await
            .unwrap();

        assert_eq!(seq1, seq2);
        // Only one event should exist
        assert_eq!(log.event_count(&exec.id), 1);
    }

    #[tokio::test]
    async fn test_read_all_events() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        for i in 0..3 {
            log.append(
                &exec.id,
                "test.event",
                serde_json::json!({"n": i}),
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .unwrap();
        }

        let events = log.read(&exec.id, None).await.unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[1].sequence, 2);
        assert_eq!(events[2].sequence, 3);
    }

    #[tokio::test]
    async fn test_read_after_sequence() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        for i in 0..5 {
            log.append(
                &exec.id,
                "test.event",
                serde_json::json!({"n": i}),
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .unwrap();
        }

        let events = log.read(&exec.id, Some(2)).await.unwrap();
        assert_eq!(events.len(), 3); // sequences 3, 4, 5
        assert_eq!(events[0].sequence, 3);
    }

    #[tokio::test]
    async fn test_read_not_found_fails() {
        let log = InMemoryEventLog::new();
        let fake_id = ExecutionId::new();
        let err = log.read(&fake_id, None).await.unwrap_err();
        assert!(matches!(err, EventLogError::ExecutionNotFound(_)));
    }

    #[tokio::test]
    async fn test_latest_of_type() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        log.append(
            &exec.id,
            "generation.started",
            serde_json::json!({"model": "gpt-4", "call": 1}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();

        log.append(
            &exec.id,
            "generation.completed",
            serde_json::json!({"model": "gpt-4"}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();

        log.append(
            &exec.id,
            "generation.started",
            serde_json::json!({"model": "gpt-4", "call": 2}),
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();

        let latest = log
            .latest_of_type(&exec.id, "generation.started")
            .await
            .unwrap()
            .unwrap();

        // Should be the second "generation.started" event (sequence 3)
        assert_eq!(latest.sequence, 3);
        assert_eq!(latest.payload["call"], 2);
    }

    #[tokio::test]
    async fn test_latest_of_type_not_found() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        let result = log
            .latest_of_type(&exec.id, "generation.started")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_count_by_type() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        for _ in 0..3 {
            log.append(
                &exec.id,
                "generation.completed",
                serde_json::Value::Null,
                EVENT_SCHEMA_VERSION,
                None,
            )
            .await
            .unwrap();
        }

        log.append(
            &exec.id,
            "activity.started",
            serde_json::Value::Null,
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();

        let count = log
            .count_by_type(&exec.id, "generation.completed")
            .await
            .unwrap();
        assert_eq!(count, 3);

        let other_count = log
            .count_by_type(&exec.id, "activity.started")
            .await
            .unwrap();
        assert_eq!(other_count, 1);
    }

    #[tokio::test]
    async fn test_count_by_type_zero_for_missing() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        let count = log
            .count_by_type(&exec.id, "generation.completed")
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_all_events_across_executions() {
        let log = InMemoryEventLog::new();
        let exec1 = make_execution();
        let exec2 = make_execution();
        log.create_execution(&exec1).await.unwrap();
        log.create_execution(&exec2).await.unwrap();

        log.append(
            &exec1.id,
            "test.event",
            serde_json::Value::Null,
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();
        log.append(
            &exec2.id,
            "test.event",
            serde_json::Value::Null,
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();

        let all = log.all_events();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_event_schema_version_stored() {
        let log = InMemoryEventLog::new();
        let exec = make_execution();
        log.create_execution(&exec).await.unwrap();

        log.append(
            &exec.id,
            "test.event",
            serde_json::Value::Null,
            EVENT_SCHEMA_VERSION,
            None,
        )
        .await
        .unwrap();

        let events = log.read(&exec.id, None).await.unwrap();
        assert_eq!(events[0].schema_version, EVENT_SCHEMA_VERSION);
        assert_eq!(events[0].schema_version, 1);
    }
}
