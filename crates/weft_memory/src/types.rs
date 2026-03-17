//! Domain types for memory store operations.
//!
//! These are the gateway-internal types mapped from gRPC proto messages.
//! They decouple the rest of the codebase from generated proto types.

/// A memory entry returned from a store query.
///
/// Domain type — mapped from the gRPC `MemoryEntry` message.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: String,
    /// The memory content text.
    pub content: String,
    /// Relevance score (0.0-1.0).
    pub score: f32,
    /// ISO 8601 timestamp when stored.
    pub created_at: String,
    /// Optional metadata as parsed JSON. `None` if the `metadata_json` field
    /// was empty or contained invalid JSON (invalid JSON is logged as a warning
    /// but does not fail the query — the content and score are still valid).
    pub metadata: Option<serde_json::Value>,
}

/// Result of a memory query operation.
#[derive(Debug, Clone)]
pub struct MemoryQueryResult {
    /// The store name that returned these results.
    pub store_name: String,
    /// Matching memories, ordered by relevance (highest first).
    pub memories: Vec<MemoryEntry>,
}

/// Result of a successful memory store operation.
///
/// Failures are reported via `MemoryStoreError` (gRPC status codes),
/// not via fields on this struct. A value of this type means the store
/// succeeded.
#[derive(Debug, Clone)]
pub struct MemoryStoreResult {
    /// The ID assigned to the stored memory.
    pub id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_entry_construction() {
        let entry = MemoryEntry {
            id: "mem-001".to_string(),
            content: "User prefers dark mode".to_string(),
            score: 0.87,
            created_at: "2026-03-15T10:00:00Z".to_string(),
            metadata: Some(serde_json::json!({"source": "conversation"})),
        };
        assert_eq!(entry.id, "mem-001");
        assert_eq!(entry.content, "User prefers dark mode");
        assert!((entry.score - 0.87).abs() < 0.001);
        assert_eq!(entry.created_at, "2026-03-15T10:00:00Z");
        assert!(entry.metadata.is_some());
    }

    #[test]
    fn test_memory_entry_no_metadata() {
        let entry = MemoryEntry {
            id: "mem-002".to_string(),
            content: "Timezone: America/Vancouver".to_string(),
            score: 0.72,
            created_at: "2026-03-15T11:00:00Z".to_string(),
            metadata: None,
        };
        assert!(entry.metadata.is_none());
    }

    #[test]
    fn test_memory_query_result_empty() {
        let result = MemoryQueryResult {
            store_name: "conversations".to_string(),
            memories: vec![],
        };
        assert_eq!(result.store_name, "conversations");
        assert!(result.memories.is_empty());
    }

    #[test]
    fn test_memory_query_result_with_entries() {
        let entry = MemoryEntry {
            id: "mem-001".to_string(),
            content: "content".to_string(),
            score: 0.9,
            created_at: "2026-03-15T10:00:00Z".to_string(),
            metadata: None,
        };
        let result = MemoryQueryResult {
            store_name: "user_prefs".to_string(),
            memories: vec![entry],
        };
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.memories[0].id, "mem-001");
    }

    #[test]
    fn test_memory_store_result_construction() {
        let result = MemoryStoreResult {
            id: "mem-assigned-42".to_string(),
        };
        assert_eq!(result.id, "mem-assigned-42");
    }
}
