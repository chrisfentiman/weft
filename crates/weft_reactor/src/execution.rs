//! Execution identity, lifecycle, and parent-child relationships.

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Strongly-typed execution identifier.
///
/// Every pipeline run -- root or child -- gets a unique ExecutionId.
/// This is the primary key in the event log and the handle for
/// signals, queries, and cancellation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExecutionId(pub Uuid);

impl ExecutionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ExecutionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ExecutionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strongly-typed tenant identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TenantId(pub String);

/// Strongly-typed request identifier.
///
/// A single HTTP request maps to one root Execution. The RequestId
/// links the Execution back to the originating request for correlation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RequestId(pub String);

/// Execution lifecycle status.
///
/// State machine: Running -> Completed | Failed | Cancelled
/// Transitions are one-way. A Completed execution cannot become Failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// A pipeline execution. The unit of identity, lifecycle, and cancellation.
///
/// An Execution is created when a request arrives (root) or when an
/// Activity spawns a child execution (recursive). All state is in the
/// EventLog -- the Execution struct holds identity and metadata only.
#[derive(Debug, Clone)]
pub struct Execution {
    pub id: ExecutionId,
    pub tenant_id: TenantId,
    pub request_id: RequestId,
    /// None for root executions. Some(parent_id) for child executions.
    pub parent_id: Option<ExecutionId>,
    /// Which pipeline config to use. Defaults to "default".
    pub pipeline_name: String,
    pub status: ExecutionStatus,
    pub created_at: DateTime<Utc>,
    /// The recursion depth. Root = 0, first child = 1, etc.
    pub depth: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_id_new_is_unique() {
        let a = ExecutionId::new();
        let b = ExecutionId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_execution_id_display() {
        let id = ExecutionId::new();
        let display = id.to_string();
        // UUID format: 8-4-4-4-12 hex chars
        assert_eq!(display.len(), 36);
    }

    #[test]
    fn test_execution_id_serde_round_trip() {
        let id = ExecutionId::new();
        let json = serde_json::to_string(&id).unwrap();
        let back: ExecutionId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn test_execution_status_serde() {
        for status in [
            ExecutionStatus::Running,
            ExecutionStatus::Completed,
            ExecutionStatus::Failed,
            ExecutionStatus::Cancelled,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ExecutionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }

    #[test]
    fn test_execution_constructible() {
        let exec = Execution {
            id: ExecutionId::new(),
            tenant_id: TenantId("tenant-1".to_string()),
            request_id: RequestId("req-1".to_string()),
            parent_id: None,
            pipeline_name: "default".to_string(),
            status: ExecutionStatus::Running,
            created_at: Utc::now(),
            depth: 0,
        };
        assert_eq!(exec.depth, 0);
        assert!(exec.parent_id.is_none());
    }

    #[test]
    fn test_execution_with_parent() {
        let parent_id = ExecutionId::new();
        let exec = Execution {
            id: ExecutionId::new(),
            tenant_id: TenantId("tenant-1".to_string()),
            request_id: RequestId("req-1".to_string()),
            parent_id: Some(parent_id.clone()),
            pipeline_name: "default".to_string(),
            status: ExecutionStatus::Running,
            created_at: Utc::now(),
            depth: 1,
        };
        assert_eq!(exec.parent_id.as_ref().unwrap(), &parent_id);
        assert_eq!(exec.depth, 1);
    }
}
