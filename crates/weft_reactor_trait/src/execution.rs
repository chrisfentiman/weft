//! Execution identity, lifecycle, and parent-child relationships.

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Strongly-typed execution identifier.
///
/// Every pipeline run — root or child — gets a unique ExecutionId.
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RequestId(pub String);

/// Execution lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// A pipeline execution.
#[derive(Debug, Clone)]
pub struct Execution {
    pub id: ExecutionId,
    pub tenant_id: TenantId,
    pub request_id: RequestId,
    pub parent_id: Option<ExecutionId>,
    pub pipeline_name: String,
    pub status: ExecutionStatus,
    pub created_at: DateTime<Utc>,
    pub depth: u32,
}
