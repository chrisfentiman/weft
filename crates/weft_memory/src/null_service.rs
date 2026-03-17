//! `NullMemoryService` — a no-op implementation for testing.
//!
//! Returns empty/unconfigured results for all operations. Use this in
//! engine tests that do not exercise memory commands, to avoid needing
//! a real `MemoryStoreMux` or gRPC infrastructure.

use weft_core::CommandResult;

use crate::service::{MemoryService, StoreInfo};

/// A do-nothing `MemoryService` for tests and dev scenarios with no memory stores.
///
/// - `is_configured()` returns `false`.
/// - `stores()` returns an empty vec.
/// - `recall`/`remember` return empty `CommandResult`s with no error.
pub struct NullMemoryService;

#[async_trait::async_trait]
impl MemoryService for NullMemoryService {
    async fn recall(
        &self,
        _store_ids: &[String],
        _query: &str,
        _user_message: &str,
    ) -> CommandResult {
        CommandResult {
            command_name: "recall".to_string(),
            success: false,
            output: String::new(),
            error: Some("no memory stores configured".to_string()),
        }
    }

    async fn remember(&self, _store_ids: &[String], _content: &str) -> CommandResult {
        CommandResult {
            command_name: "remember".to_string(),
            success: false,
            output: String::new(),
            error: Some("no memory stores configured".to_string()),
        }
    }

    fn is_configured(&self) -> bool {
        false
    }

    fn stores(&self) -> Vec<StoreInfo> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_null_service_is_not_configured() {
        assert!(!NullMemoryService.is_configured());
    }

    #[tokio::test]
    async fn test_null_service_stores_empty() {
        assert!(NullMemoryService.stores().is_empty());
    }

    #[tokio::test]
    async fn test_null_recall_returns_unconfigured_error() {
        let result = NullMemoryService.recall(&[], "query", "user message").await;
        assert!(!result.success);
        assert!(result.error.is_some());
        assert_eq!(result.command_name, "recall");
    }

    #[tokio::test]
    async fn test_null_remember_returns_unconfigured_error() {
        let result = NullMemoryService.remember(&[], "content").await;
        assert!(!result.success);
        assert!(result.error.is_some());
        assert_eq!(result.command_name, "remember");
    }
}
