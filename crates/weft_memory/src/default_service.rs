//! `DefaultMemoryService` — concrete implementation backed by `MemoryStoreMux`.
//!
//! Owns an `Arc<MemoryStoreMux>` and implements the `MemoryService` trait.
//! Formats results using `weft_core::toon` helpers so the LLM receives
//! TOON-encoded output.

use std::sync::Arc;

use weft_core::{
    CommandResult,
    toon::{fenced_toon, serialize_table},
};

use crate::{
    client::MemoryStoreError,
    mux::MemoryStoreMux,
    service::{MemoryService, StoreInfo},
    types::{MemoryQueryResult, MemoryStoreResult},
};

/// Concrete `MemoryService` backed by a `MemoryStoreMux`.
///
/// Constructed in the binary bootstrap and injected into the reactor.
/// All recall/remember logic lives here — the engine only decides *which*
/// store_ids to pass in (via routing).
///
/// `store_infos` carries the static per-store metadata (name, capabilities, examples)
/// that the engine uses to build routing candidates. Examples come from config at
/// startup; they are not stored in the mux itself.
pub struct DefaultMemoryService {
    mux: Arc<MemoryStoreMux>,
    store_infos: Vec<StoreInfo>,
}

impl DefaultMemoryService {
    /// Construct a new service wrapping the given multiplexer.
    ///
    /// `store_infos` — static metadata for each configured store, populated at
    /// startup from config. Passed through to `stores()` so the engine can build
    /// routing candidates without importing config types.
    pub fn new(mux: Arc<MemoryStoreMux>, store_infos: Vec<StoreInfo>) -> Self {
        Self { mux, store_infos }
    }
}

#[async_trait::async_trait]
impl MemoryService for DefaultMemoryService {
    async fn recall(&self, store_ids: &[String], query: &str, user_message: &str) -> CommandResult {
        // Use the explicit query if non-empty, otherwise fall back to the user message.
        let effective_query = if query.is_empty() {
            user_message
        } else {
            query
        };

        let results = self.mux.query(store_ids, effective_query, 0.0).await;
        let output = format_memory_query_results(&results);

        CommandResult {
            command_name: "recall".to_string(),
            success: true,
            output,
            error: None,
        }
    }

    async fn remember(&self, store_ids: &[String], content: &str) -> CommandResult {
        // Determine target stores. If store_ids is empty, fall back to the first writable store.
        let target_stores: Vec<String> = if store_ids.is_empty() {
            self.mux
                .writable_store_names()
                .into_iter()
                .take(1)
                .map(|s| s.to_string())
                .collect()
        } else {
            store_ids.to_vec()
        };

        if target_stores.is_empty() {
            // No writable stores — succeed silently (no stores = nothing to write).
            return CommandResult {
                command_name: "remember".to_string(),
                success: true,
                output: String::new(),
                error: None,
            };
        }

        let results = self.mux.store(&target_stores, content, None).await;
        let output = format_memory_store_results(&results);

        CommandResult {
            command_name: "remember".to_string(),
            success: true,
            output,
            error: None,
        }
    }

    fn is_configured(&self) -> bool {
        !self.mux.is_empty()
    }

    fn stores(&self) -> Vec<StoreInfo> {
        self.store_infos.clone()
    }
}

/// Format memory query results as TOON output for the LLM.
///
/// Groups results by store with `{store, content, score}` columns.
/// Returns "No relevant memories found." when no memories were retrieved.
pub fn format_memory_query_results(results: &[MemoryQueryResult]) -> String {
    let all_memories: Vec<_> = results
        .iter()
        .flat_map(|r| {
            r.memories.iter().map(|m| {
                vec![
                    r.store_name.clone(),
                    m.content.clone(),
                    format!("{:.2}", m.score),
                ]
            })
        })
        .collect();

    if all_memories.is_empty() {
        return "No relevant memories found.".to_string();
    }

    let table = serialize_table("memories", &["store", "content", "score"], &all_memories);
    fenced_toon(&table)
}

/// Format memory store results as TOON output for the LLM.
///
/// Reports per-store success/failure with `{store, status}` or
/// `{store, status, error}` columns depending on whether errors occurred.
pub fn format_memory_store_results(
    results: &[(String, Result<MemoryStoreResult, MemoryStoreError>)],
) -> String {
    let has_errors = results.iter().any(|(_, r)| r.is_err());

    if has_errors {
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|(name, r)| match r {
                Ok(_) => vec![name.clone(), "success".to_string(), String::new()],
                Err(e) => vec![name.clone(), "error".to_string(), e.to_string()],
            })
            .collect();
        let table = serialize_table("stored", &["store", "status", "error"], &rows);
        fenced_toon(&table)
    } else {
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|(name, _)| vec![name.clone(), "success".to_string()])
            .collect();
        let table = serialize_table("stored", &["store", "status"], &rows);
        fenced_toon(&table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        client::MemoryStoreError,
        types::{MemoryEntry, MemoryQueryResult, MemoryStoreResult},
    };

    fn entry(id: &str, content: &str, score: f32) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: content.to_string(),
            score,
            created_at: "2026-03-16T00:00:00Z".to_string(),
            metadata: None,
        }
    }

    // ── format_memory_query_results ────────────────────────────────────────

    #[test]
    fn test_format_query_results_empty_returns_no_memories() {
        let output = format_memory_query_results(&[]);
        assert_eq!(output, "No relevant memories found.");
    }

    #[test]
    fn test_format_query_results_single_store() {
        let results = vec![MemoryQueryResult {
            store_name: "conv".to_string(),
            memories: vec![entry("m1", "user prefers dark mode", 0.9)],
        }];
        let output = format_memory_query_results(&results);
        // Should be TOON-fenced and contain the content.
        assert!(output.contains("user prefers dark mode"));
        assert!(output.contains("conv"));
        assert!(output.contains("0.90"));
    }

    #[test]
    fn test_format_query_results_multiple_stores() {
        let results = vec![
            MemoryQueryResult {
                store_name: "conv".to_string(),
                memories: vec![entry("m1", "dark mode", 0.9)],
            },
            MemoryQueryResult {
                store_name: "code".to_string(),
                memories: vec![entry("m2", "uses tokio", 0.8)],
            },
        ];
        let output = format_memory_query_results(&results);
        assert!(output.contains("conv"));
        assert!(output.contains("code"));
        assert!(output.contains("dark mode"));
        assert!(output.contains("uses tokio"));
    }

    // ── format_memory_store_results ────────────────────────────────────────

    #[test]
    fn test_format_store_results_all_success() {
        let results = vec![
            (
                "conv".to_string(),
                Ok(MemoryStoreResult {
                    id: "id1".to_string(),
                }),
            ),
            (
                "code".to_string(),
                Ok(MemoryStoreResult {
                    id: "id2".to_string(),
                }),
            ),
        ];
        let output = format_memory_store_results(&results);
        assert!(output.contains("success"));
        // No error column when all succeed.
        assert!(!output.contains("error"));
    }

    #[test]
    fn test_format_store_results_partial_failure() {
        let results = vec![
            (
                "conv".to_string(),
                Ok(MemoryStoreResult {
                    id: "id1".to_string(),
                }),
            ),
            (
                "broken".to_string(),
                Err(MemoryStoreError::StoreFailed(
                    "connection refused".to_string(),
                )),
            ),
        ];
        let output = format_memory_store_results(&results);
        assert!(output.contains("success"));
        assert!(output.contains("error"));
        assert!(output.contains("connection refused"));
    }

    #[test]
    fn test_format_store_results_all_failure() {
        let results = vec![(
            "broken".to_string(),
            Err(MemoryStoreError::StoreFailed("down".to_string())),
        )];
        let output = format_memory_store_results(&results);
        assert!(output.contains("error"));
        assert!(output.contains("down"));
    }
}
