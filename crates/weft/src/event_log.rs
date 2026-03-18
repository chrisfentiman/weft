//! Event log backend selection and construction.
//!
//! Reads `[event_log]` from `WeftConfig` and constructs the appropriate
//! `Arc<dyn EventLog>` implementation:
//!
//! - `backend = "memory"` (default): `InMemoryEventLog` — no persistence.
//! - `backend = "postgres"`: `PostgresEventLog` — durable, requires `database_url`.
//!
//! The Reactor receives `Arc<dyn EventLog>` regardless of which backend is selected.

use std::sync::Arc;

use weft_core::EventLogConfig;
use weft_reactor::event_log::EventLog;

/// Construct an `Arc<dyn EventLog>` from the optional `[event_log]` config section.
///
/// Returns `Err` if the Postgres backend is selected but the connection cannot be
/// established. On success, always returns an `Arc<dyn EventLog>` that the Reactor
/// can use immediately.
///
/// When `event_log_config` is `None`, falls back to the in-memory backend.
pub async fn build_event_log(
    event_log_config: Option<&EventLogConfig>,
) -> Result<Arc<dyn EventLog>, String> {
    let backend = event_log_config
        .map(|c| c.backend.as_str())
        .unwrap_or("memory");

    match backend {
        "memory" => {
            let log = weft_eventlog_memory::InMemoryEventLog::new();
            Ok(Arc::new(log))
        }
        "postgres" => build_postgres_event_log(event_log_config).await,
        other => Err(format!(
            "unknown event_log.backend: \"{other}\" — expected \"memory\" or \"postgres\""
        )),
    }
}

/// Construct a `PostgresEventLog`. Only compiled when the `weft_eventlog_postgres`
/// optional dependency is enabled.
#[cfg(feature = "weft_eventlog_postgres")]
async fn build_postgres_event_log(
    event_log_config: Option<&EventLogConfig>,
) -> Result<Arc<dyn EventLog>, String> {
    let config = event_log_config
        .ok_or_else(|| "event_log config required for postgres backend".to_string())?;
    let url = config
        .database_url
        .as_deref()
        .ok_or_else(|| "event_log.database_url is required for postgres backend".to_string())?;

    let pool = sqlx::PgPool::connect(url)
        .await
        .map_err(|e| format!("event_log: failed to connect to Postgres: {e}"))?;

    let log = weft_eventlog_postgres::PostgresEventLog::new(pool);
    Ok(Arc::new(log) as Arc<dyn EventLog>)
}

/// Fallback when `weft_eventlog_postgres` optional dependency is not compiled in.
///
/// If the config asks for postgres but the feature is not compiled in, fail fast
/// at startup rather than silently falling back to in-memory — silent fallback
/// would hide a misconfiguration in production.
#[cfg(not(feature = "weft_eventlog_postgres"))]
async fn build_postgres_event_log(
    _event_log_config: Option<&EventLogConfig>,
) -> Result<Arc<dyn EventLog>, String> {
    Err(
        "event_log.backend = \"postgres\" requires the weft_eventlog_postgres feature to be enabled. \
         Rebuild with --features weft_eventlog_postgres or change event_log.backend to \"memory\"."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_build_event_log_defaults_to_memory() {
        let log = build_event_log(None)
            .await
            .expect("memory build should succeed");
        // Verify it compiles and is usable as Arc<dyn EventLog>.
        let _: Arc<dyn EventLog> = log;
    }

    #[tokio::test]
    async fn test_build_event_log_explicit_memory() {
        let config = EventLogConfig {
            backend: "memory".to_string(),
            database_url: None,
        };
        let log = build_event_log(Some(&config))
            .await
            .expect("explicit memory build should succeed");
        let _: Arc<dyn EventLog> = log;
    }

    #[tokio::test]
    async fn test_build_event_log_unknown_backend_errors() {
        let config = EventLogConfig {
            backend: "redis".to_string(),
            database_url: None,
        };
        let result = build_event_log(Some(&config)).await;
        match result {
            Err(msg) => assert!(msg.contains("unknown event_log.backend"), "got: {msg}"),
            Ok(_) => panic!("unknown backend should have returned Err"),
        }
    }

    #[tokio::test]
    #[cfg(not(feature = "weft_eventlog_postgres"))]
    async fn test_build_event_log_postgres_without_feature_errors() {
        let config = EventLogConfig {
            backend: "postgres".to_string(),
            database_url: Some("postgres://localhost/test".to_string()),
        };
        let result = build_event_log(Some(&config)).await;
        match result {
            Err(msg) => {
                assert!(msg.contains("weft_eventlog_postgres feature"), "got: {msg}")
            }
            Ok(_) => panic!("postgres backend without feature should have returned Err"),
        }
    }
}
