//! ActivityRegistry: maps activity names to Arc<dyn Activity>.
//!
//! Constructed at startup. Immutable after construction. Shared via Arc.
//! The Reactor looks up activities by name when building compiled pipelines.

use std::collections::HashMap;
use std::sync::Arc;

use crate::activity::Activity;

/// Registry of named activities.
///
/// Activities register by name. The pipeline config references activities
/// by name. The Reactor looks up activities in the registry when building
/// the pipeline activity list.
///
/// Constructed at startup. Immutable after construction. Shared via Arc.
///
/// # Registration
///
/// Activity names must be unique. Attempting to register a second activity
/// under the same name returns `RegistryError::DuplicateName`.
///
/// By convention, names are lowercase with underscores:
/// `"validate"`, `"route"`, `"generate"`, `"hook_pre_response"`.
pub struct ActivityRegistry {
    activities: HashMap<String, Arc<dyn Activity>>,
}

impl ActivityRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            activities: HashMap::new(),
        }
    }

    /// Register an activity by name.
    ///
    /// The name is taken from `activity.name()`. It must match what the
    /// pipeline config uses to reference this activity.
    ///
    /// Returns `Err(RegistryError::DuplicateName)` if an activity with
    /// this name is already registered.
    pub fn register(&mut self, activity: Arc<dyn Activity>) -> Result<(), RegistryError> {
        let name = activity.name().to_string();
        if self.activities.contains_key(&name) {
            return Err(RegistryError::DuplicateName(name));
        }
        self.activities.insert(name, activity);
        Ok(())
    }

    /// Look up an activity by name.
    ///
    /// Returns `None` if no activity with the given name is registered.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Activity>> {
        self.activities.get(name)
    }

    /// List all registered activity names.
    ///
    /// The order is not guaranteed to be stable across calls.
    pub fn names(&self) -> Vec<&str> {
        self.activities.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ActivityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from the ActivityRegistry.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// An activity with this name was already registered.
    #[error("activity already registered: {0}")]
    DuplicateName(String),

    /// No activity with this name exists in the registry.
    #[error("activity not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activity::{Activity, ActivityInput};
    use crate::event::PipelineEvent;
    use crate::event_log::EventLog;
    use crate::execution::ExecutionId;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use weft_reactor_trait::ServiceLocator;

    // ── Test activity stubs ─────────────────────────────────────────────────

    struct StubActivity {
        name: String,
    }

    impl StubActivity {
        fn named(name: &str) -> Arc<dyn Activity> {
            Arc::new(Self {
                name: name.to_string(),
            })
        }
    }

    #[async_trait::async_trait]
    impl Activity for StubActivity {
        fn name(&self) -> &str {
            &self.name
        }

        async fn execute(
            &self,
            _execution_id: &ExecutionId,
            _input: ActivityInput,
            _services: &dyn ServiceLocator,
            _event_log: &dyn EventLog,
            _event_tx: mpsc::Sender<PipelineEvent>,
            _cancel: CancellationToken,
        ) {
        }
    }

    // ── ActivityRegistry::new ───────────────────────────────────────────────

    #[test]
    fn new_registry_is_empty() {
        let registry = ActivityRegistry::new();
        assert!(registry.names().is_empty());
    }

    #[test]
    fn default_registry_is_empty() {
        let registry = ActivityRegistry::default();
        assert!(registry.names().is_empty());
    }

    // ── ActivityRegistry::register ──────────────────────────────────────────

    #[test]
    fn register_single_activity_succeeds() {
        let mut registry = ActivityRegistry::new();
        let result = registry.register(StubActivity::named("validate"));
        assert!(result.is_ok());
        assert_eq!(registry.names().len(), 1);
    }

    #[test]
    fn register_multiple_distinct_activities_succeeds() {
        let mut registry = ActivityRegistry::new();
        registry.register(StubActivity::named("validate")).unwrap();
        registry.register(StubActivity::named("route")).unwrap();
        registry.register(StubActivity::named("generate")).unwrap();
        assert_eq!(registry.names().len(), 3);
    }

    #[test]
    fn register_duplicate_name_returns_error() {
        let mut registry = ActivityRegistry::new();
        registry.register(StubActivity::named("validate")).unwrap();
        let result = registry.register(StubActivity::named("validate"));
        assert!(
            matches!(result, Err(RegistryError::DuplicateName(ref name)) if name == "validate")
        );
    }

    #[test]
    fn register_duplicate_returns_correct_name_in_error() {
        let mut registry = ActivityRegistry::new();
        registry
            .register(StubActivity::named("hook_request_end"))
            .unwrap();
        let err = registry
            .register(StubActivity::named("hook_request_end"))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "activity already registered: hook_request_end"
        );
    }

    // ── ActivityRegistry::get ───────────────────────────────────────────────

    #[test]
    fn get_registered_activity_returns_some() {
        let mut registry = ActivityRegistry::new();
        registry.register(StubActivity::named("validate")).unwrap();
        let activity = registry.get("validate");
        assert!(activity.is_some());
        assert_eq!(activity.unwrap().name(), "validate");
    }

    #[test]
    fn get_unknown_name_returns_none() {
        let registry = ActivityRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn get_returns_correct_activity_by_name() {
        let mut registry = ActivityRegistry::new();
        registry.register(StubActivity::named("validate")).unwrap();
        registry.register(StubActivity::named("route")).unwrap();

        // Each lookup returns the correct activity.
        assert_eq!(registry.get("validate").unwrap().name(), "validate");
        assert_eq!(registry.get("route").unwrap().name(), "route");
        assert!(registry.get("generate").is_none());
    }

    // ── ActivityRegistry::names ─────────────────────────────────────────────

    #[test]
    fn names_returns_all_registered_names() {
        let mut registry = ActivityRegistry::new();
        registry.register(StubActivity::named("validate")).unwrap();
        registry.register(StubActivity::named("route")).unwrap();
        registry.register(StubActivity::named("generate")).unwrap();

        let mut names = registry.names();
        names.sort(); // order not guaranteed
        assert_eq!(names, vec!["generate", "route", "validate"]);
    }

    #[test]
    fn names_empty_after_new() {
        let registry = ActivityRegistry::new();
        assert!(registry.names().is_empty());
    }

    // ── RegistryError display ───────────────────────────────────────────────

    #[test]
    fn registry_error_duplicate_display() {
        let err = RegistryError::DuplicateName("validate".to_string());
        assert_eq!(err.to_string(), "activity already registered: validate");
    }

    #[test]
    fn registry_error_not_found_display() {
        let err = RegistryError::NotFound("nonexistent".to_string());
        assert_eq!(err.to_string(), "activity not found: nonexistent");
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn register_activity_with_spaces_in_name_succeeds() {
        // Unconventional name — registry accepts any non-duplicate string.
        let mut registry = ActivityRegistry::new();
        let result = registry.register(StubActivity::named("my custom activity"));
        assert!(result.is_ok());
    }

    #[test]
    fn register_activity_with_empty_name_succeeds() {
        // Registry doesn't enforce naming conventions — caller is responsible.
        let mut registry = ActivityRegistry::new();
        let result = registry.register(StubActivity::named(""));
        assert!(result.is_ok());
        assert!(registry.get("").is_some());
    }

    #[test]
    fn get_returns_arc_that_preserves_identity() {
        let activity = StubActivity::named("validate");
        // Get the pointer address before registering.
        let ptr = Arc::as_ptr(&activity);

        let mut registry = ActivityRegistry::new();
        registry.register(Arc::clone(&activity)).unwrap();

        let retrieved = registry.get("validate").unwrap();
        // The Arc in the registry points to the same allocation.
        assert_eq!(Arc::as_ptr(retrieved), ptr);
    }
}
