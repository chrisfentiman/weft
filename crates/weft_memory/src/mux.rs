//! Memory store multiplexer.
//!
//! Routes memory operations to one or more named stores, filtered by capability.
//! Both `/recall` and `/remember` perform per-invocation routing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::client::{MemoryStoreClient, MemoryStoreError};
use crate::types::{MemoryQueryResult, MemoryStoreResult};

/// Routes memory operations to N named stores, filtered by capability.
///
/// Both `/recall` and `/remember` perform per-invocation routing based on their
/// own argument content (`query` and `content` respectively) via
/// `score_memory_candidates()`. The pre-computed `RoutingDecision.memory_stores`
/// from the user's message is NOT used for store selection by either command.
/// The mux filters by capability: `/recall` only considers read-capable stores,
/// `/remember` only considers write-capable stores.
///
/// Thread-safe: clients are Arc'd, all HashMaps/HashSets are immutable after construction.
pub struct MemoryStoreMux {
    /// Named store clients. Key is the store name from config.
    stores: HashMap<String, Arc<dyn MemoryStoreClient>>,
    /// Max results per store (from config, per-store).
    max_results: HashMap<String, u32>,
    /// Store names that have read capability (eligible for `/recall`).
    readable: HashSet<String>,
    /// Store names that have write capability (eligible for `/remember`).
    writable: HashSet<String>,
}

impl MemoryStoreMux {
    /// Construct a new mux with the given stores and capability sets.
    ///
    /// All maps are moved in and become immutable after construction.
    pub fn new(
        stores: HashMap<String, Arc<dyn MemoryStoreClient>>,
        max_results: HashMap<String, u32>,
        readable: HashSet<String>,
        writable: HashSet<String>,
    ) -> Self {
        Self {
            stores,
            max_results,
            readable,
            writable,
        }
    }

    /// Query specific stores by name. Called after semantic routing determines
    /// which stores are relevant.
    ///
    /// - If `store_names` is empty, queries ALL read-capable stores (fallback behavior
    ///   when router fails or Memory domain is disabled).
    /// - Stores without read capability are silently filtered out, even if they
    ///   appear in `store_names`.
    /// - Individual store failures are logged and skipped (partial results are
    ///   better than total failure).
    ///
    /// Returns results from all queried stores, aggregated.
    pub async fn query(
        &self,
        store_names: &[String],
        query: &str,
        min_score: f32,
    ) -> Vec<MemoryQueryResult> {
        // Determine target set: empty store_names means fallback to all readable.
        let targets: Vec<&str> = if store_names.is_empty() {
            self.readable.iter().map(|s| s.as_str()).collect()
        } else {
            store_names
                .iter()
                .filter(|n| self.readable.contains(n.as_str()))
                .map(|n| n.as_str())
                .collect()
        };

        if targets.is_empty() {
            return vec![];
        }

        // Query targets concurrently.
        let futures: Vec<_> = targets
            .iter()
            .filter_map(|name| {
                let client = self.stores.get(*name)?;
                let max = *self.max_results.get(*name).unwrap_or(&5);
                Some((*name, Arc::clone(client), max))
            })
            .map(|(name, client, max)| async move {
                let result = client.query(query, max, min_score).await;
                (name.to_string(), result)
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        results
            .into_iter()
            .filter_map(|(store_name, result)| match result {
                Ok(memories) => Some(MemoryQueryResult {
                    store_name,
                    memories,
                }),
                Err(e) => {
                    tracing::warn!(store = %store_name, error = %e, "memory store query failed — skipping");
                    None
                }
            })
            .collect()
    }

    /// Store content to specific stores by name. Called after semantic routing
    /// determines which stores are appropriate for this content.
    ///
    /// - `store_names` must be non-empty — the engine is responsible for selecting
    ///   at least one store. An empty `store_names` is a caller bug; the mux logs
    ///   a `warn!` and returns an empty result set.
    /// - Stores without write capability are silently filtered out.
    /// - Returns per-store `Result`s — the caller can report partial success to the LLM.
    pub async fn store(
        &self,
        store_names: &[String],
        content: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Vec<(String, Result<MemoryStoreResult, MemoryStoreError>)> {
        if store_names.is_empty() {
            // Engine must always select a target for writes — this is a caller bug.
            tracing::warn!("MemoryStoreMux::store called with empty store_names — caller bug");
            return vec![];
        }

        // Intersect with write-capable stores.
        let targets: Vec<&str> = store_names
            .iter()
            .filter(|n| self.writable.contains(n.as_str()))
            .map(|n| n.as_str())
            .collect();

        if targets.is_empty() {
            return vec![];
        }

        // Store concurrently to targets.
        // Clone metadata for each concurrent call.
        let metadata_owned: Option<serde_json::Value> = metadata.cloned();

        let futures: Vec<_> = targets
            .iter()
            .filter_map(|name| {
                let client = self.stores.get(*name)?;
                Some((*name, Arc::clone(client)))
            })
            .map(|(name, client)| {
                let meta_ref = metadata_owned.as_ref();
                let content = content.to_string();
                async move {
                    let result = client.store(&content, meta_ref).await;
                    (name.to_string(), result)
                }
            })
            .collect();

        futures::future::join_all(futures).await
    }

    /// Whether any stores are configured.
    pub fn is_empty(&self) -> bool {
        self.stores.is_empty()
    }

    /// List store names that have read capability.
    ///
    /// Used for building routing candidates for `/recall` operations.
    pub fn readable_store_names(&self) -> Vec<&str> {
        self.readable.iter().map(|s| s.as_str()).collect()
    }

    /// List store names that have write capability.
    ///
    /// Used for building routing candidates for `/remember` operations.
    pub fn writable_store_names(&self) -> Vec<&str> {
        self.writable.iter().map(|s| s.as_str()).collect()
    }

    /// List all store names regardless of capability.
    ///
    /// Used for building routing candidates at startup — used for the Memory
    /// domain in `route_all_domains()` and for per-invocation routing by both
    /// `/recall` and `/remember`.
    pub fn store_names(&self) -> Vec<&str> {
        self.stores.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::MemoryStoreError;
    use crate::types::{MemoryEntry, MemoryStoreResult};
    use std::sync::Mutex;

    /// Configurable mock that returns preset results or errors.
    struct MockMemoryStoreClient {
        query_result: Arc<Mutex<Result<Vec<MemoryEntry>, String>>>,
        store_result: Arc<Mutex<Result<String, String>>>,
    }

    impl MockMemoryStoreClient {
        fn succeeds_query(entries: Vec<MemoryEntry>) -> Self {
            Self {
                query_result: Arc::new(Mutex::new(Ok(entries))),
                store_result: Arc::new(Mutex::new(Ok("mock-id".to_string()))),
            }
        }

        fn fails_query(msg: &str) -> Self {
            Self {
                query_result: Arc::new(Mutex::new(Err(msg.to_string()))),
                store_result: Arc::new(Mutex::new(Ok("mock-id".to_string()))),
            }
        }

        fn fails_store(msg: &str) -> Self {
            Self {
                query_result: Arc::new(Mutex::new(Ok(vec![]))),
                store_result: Arc::new(Mutex::new(Err(msg.to_string()))),
            }
        }
    }

    #[async_trait::async_trait]
    impl MemoryStoreClient for MockMemoryStoreClient {
        async fn query(
            &self,
            _query: &str,
            _max_results: u32,
            _min_score: f32,
        ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
            match &*self.query_result.lock().unwrap() {
                Ok(entries) => Ok(entries.clone()),
                Err(msg) => Err(MemoryStoreError::QueryFailed(msg.clone())),
            }
        }

        async fn store(
            &self,
            _content: &str,
            _metadata: Option<&serde_json::Value>,
        ) -> Result<MemoryStoreResult, MemoryStoreError> {
            match &*self.store_result.lock().unwrap() {
                Ok(id) => Ok(MemoryStoreResult { id: id.clone() }),
                Err(msg) => Err(MemoryStoreError::StoreFailed(msg.clone())),
            }
        }
    }

    fn entry(id: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: format!("content-{id}"),
            score: 0.8,
            created_at: "2026-03-15T10:00:00Z".to_string(),
            metadata: None,
        }
    }

    fn build_mux(
        configs: Vec<(&str, bool, bool)>, // (name, readable, writable)
        clients: HashMap<String, Arc<dyn MemoryStoreClient>>,
    ) -> MemoryStoreMux {
        let mut readable = HashSet::new();
        let mut writable = HashSet::new();
        let mut max_results = HashMap::new();
        for (name, r, w) in configs {
            if r {
                readable.insert(name.to_string());
            }
            if w {
                writable.insert(name.to_string());
            }
            max_results.insert(name.to_string(), 5u32);
        }
        MemoryStoreMux::new(clients, max_results, readable, writable)
    }

    // ── is_empty / store_names ─────────────────────────────────────────────────

    #[test]
    fn test_is_empty_with_no_stores() {
        let mux = MemoryStoreMux::new(
            HashMap::new(),
            HashMap::new(),
            HashSet::new(),
            HashSet::new(),
        );
        assert!(mux.is_empty());
    }

    #[test]
    fn test_is_empty_with_stores() {
        let client: Arc<dyn MemoryStoreClient> =
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![]));
        let mut stores = HashMap::new();
        stores.insert("conv".to_string(), client);
        let mux = build_mux(vec![("conv", true, true)], stores);
        assert!(!mux.is_empty());
    }

    #[test]
    fn test_readable_store_names() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "conv".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        stores.insert(
            "code".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        stores.insert(
            "write_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(
            vec![
                ("conv", true, true),
                ("code", true, false),
                ("write_only", false, true),
            ],
            stores,
        );
        let mut names = mux.readable_store_names();
        names.sort();
        assert_eq!(names, vec!["code", "conv"]);
    }

    #[test]
    fn test_writable_store_names() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "conv".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        stores.insert(
            "read_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(
            vec![("conv", true, true), ("read_only", true, false)],
            stores,
        );
        let names = mux.writable_store_names();
        assert_eq!(names, vec!["conv"]);
    }

    // ── query ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_query_with_specific_store_names() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "conv".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("m1")])),
        );
        stores.insert(
            "code".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("m2")])),
        );
        let mux = build_mux(vec![("conv", true, true), ("code", true, false)], stores);

        // Query only "conv"
        let results = mux.query(&["conv".to_string()], "query", 0.0).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].store_name, "conv");
        assert_eq!(results[0].memories[0].id, "m1");
    }

    #[tokio::test]
    async fn test_query_empty_store_names_queries_all_readable() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "conv".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("m1")])),
        );
        stores.insert(
            "code".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("m2")])),
        );
        stores.insert(
            "write_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("m3")])),
        );
        let mux = build_mux(
            vec![
                ("conv", true, true),
                ("code", true, false),
                ("write_only", false, true),
            ],
            stores,
        );

        // Empty names = fallback to all readable
        let results = mux.query(&[], "query", 0.0).await;
        // Should have results from "conv" and "code" only (write_only is not readable)
        assert_eq!(results.len(), 2);
        let store_names: HashSet<&str> = results.iter().map(|r| r.store_name.as_str()).collect();
        assert!(store_names.contains("conv"));
        assert!(store_names.contains("code"));
        assert!(!store_names.contains("write_only"));
    }

    #[tokio::test]
    async fn test_query_filters_write_only_stores_even_when_named() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "write_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("m1")])),
        );
        let mux = build_mux(vec![("write_only", false, true)], stores);

        // Explicitly naming a write-only store should be filtered out
        let results = mux.query(&["write_only".to_string()], "query", 0.0).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_query_with_no_readable_stores_returns_empty() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "write_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(vec![("write_only", false, true)], stores);

        let results = mux.query(&[], "query", 0.0).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_query_partial_failure_skips_failed_stores() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "ok".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![entry("good")])),
        );
        stores.insert(
            "broken".to_string(),
            Arc::new(MockMemoryStoreClient::fails_query("store is down")),
        );
        let mux = build_mux(vec![("ok", true, true), ("broken", true, true)], stores);

        let results = mux.query(&[], "query", 0.0).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].store_name, "ok");
        assert_eq!(results[0].memories[0].id, "good");
    }

    #[tokio::test]
    async fn test_query_all_stores_fail_returns_empty() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "broken1".to_string(),
            Arc::new(MockMemoryStoreClient::fails_query("down")),
        );
        stores.insert(
            "broken2".to_string(),
            Arc::new(MockMemoryStoreClient::fails_query("down")),
        );
        let mux = build_mux(
            vec![("broken1", true, true), ("broken2", true, true)],
            stores,
        );

        let results = mux.query(&[], "query", 0.0).await;
        assert!(
            results.is_empty(),
            "all failures -> empty results, no panic"
        );
    }

    // ── store ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_store_with_specific_store_names() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "conv".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        stores.insert(
            "code".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(vec![("conv", true, true), ("code", true, true)], stores);

        let results = mux.store(&["conv".to_string()], "content", None).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "conv");
        assert!(results[0].1.is_ok());
    }

    #[tokio::test]
    async fn test_store_empty_store_names_logs_warn_and_returns_empty() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "conv".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(vec![("conv", true, true)], stores);

        let results = mux.store(&[], "content", None).await;
        assert!(
            results.is_empty(),
            "empty store_names is a caller bug — returns empty"
        );
    }

    #[tokio::test]
    async fn test_store_filters_read_only_stores_even_when_named() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "read_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(vec![("read_only", true, false)], stores);

        // Explicitly naming a read-only store should be filtered out for writes
        let results = mux.store(&["read_only".to_string()], "content", None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_store_with_no_writable_stores_returns_empty() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "read_only".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        let mux = build_mux(vec![("read_only", true, false)], stores);

        let results = mux.store(&["read_only".to_string()], "content", None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_store_partial_failure_included_in_results() {
        let mut stores: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        stores.insert(
            "ok".to_string(),
            Arc::new(MockMemoryStoreClient::succeeds_query(vec![])),
        );
        stores.insert(
            "broken".to_string(),
            Arc::new(MockMemoryStoreClient::fails_store("write failed")),
        );
        let mux = build_mux(vec![("ok", true, true), ("broken", true, true)], stores);

        let results = mux
            .store(&["ok".to_string(), "broken".to_string()], "content", None)
            .await;
        assert_eq!(results.len(), 2);

        let ok_result = results.iter().find(|(name, _)| name == "ok").unwrap();
        assert!(ok_result.1.is_ok());

        let broken_result = results.iter().find(|(name, _)| name == "broken").unwrap();
        assert!(broken_result.1.is_err());
    }
}
