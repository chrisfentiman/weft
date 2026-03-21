//! Central config access point with atomic hot-swap support.
//!
//! `ConfigStore` owns a single `ArcSwap<ConfigSnapshot>` for lock-free reads
//! and atomic updates. The reactor calls `snapshot()` once at request entry
//! and threads the result through `ActivityInput`, ensuring all activities in
//! a single request see the same config version.

use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::config::WeftConfig;
use crate::resolved_config::ResolvedConfig;

// ── ConfigSnapshot ─────────────────────────────────────────────────────────────

/// Atomic bundle of resolved and operator config.
///
/// Both configs are swapped together in a single `ArcSwap::store()` call,
/// guaranteeing that no reader ever sees a mismatched pair (a `ResolvedConfig`
/// derived from a different `WeftConfig` than what `operator_config()` returns).
pub struct ConfigSnapshot {
    /// Hot config: pre-computed values for per-request access.
    pub resolved: ResolvedConfig,
    /// Full operator config: for startup-only access (service construction).
    pub operator: WeftConfig,
}

// ── ConfigStore ────────────────────────────────────────────────────────────────

/// Central config access point.
///
/// Owns a single `ArcSwap<ConfigSnapshot>` for lock-free reads and
/// atomic updates. The reactor calls `snapshot()` once at request entry
/// and threads the result through `ActivityInput`.
///
/// `operator_config()` is for startup-only use (provider construction,
/// router initialization, semaphore sizing). It must not be called
/// per-request by activities.
pub struct ConfigStore {
    inner: ArcSwap<ConfigSnapshot>,
}

impl ConfigStore {
    /// Create a new `ConfigStore` from the initial config.
    ///
    /// Computes `ResolvedConfig` from `config` and stores both together
    /// in a `ConfigSnapshot`.
    pub fn new(config: WeftConfig) -> Self {
        let resolved = ResolvedConfig::from_operator(&config);
        Self {
            inner: ArcSwap::from_pointee(ConfigSnapshot {
                resolved,
                operator: config,
            }),
        }
    }

    /// Take a snapshot of the current resolved config.
    ///
    /// Call once at request entry. The returned `Arc` is pinned to the
    /// config version at the time of this call. Config changes after this
    /// call do not affect the returned snapshot -- `Arc` reference counting
    /// keeps the old `ConfigSnapshot` alive until all holders drop it.
    pub fn snapshot(&self) -> Arc<ResolvedConfig> {
        let snap = self.inner.load_full();
        // Clone the ResolvedConfig out of the snapshot into its own Arc.
        // This is the per-request token that activities share.
        Arc::new(snap.resolved.clone())
    }

    /// Access the full operator config (startup-only fields).
    ///
    /// Used by the binary crate for service construction. Not for
    /// per-request access by activities.
    pub fn operator_config(&self) -> Arc<WeftConfig> {
        let snap = self.inner.load_full();
        Arc::new(snap.operator.clone())
    }

    /// Replace the config with a new version.
    ///
    /// Called on TOML file reload. The new config must already be
    /// validated. Both resolved and operator config are swapped
    /// atomically in a single store. In-flight requests continue
    /// using their snapshot; new requests see the updated config.
    pub fn swap(&self, new_config: WeftConfig) {
        let resolved = ResolvedConfig::from_operator(&new_config);
        self.inner.store(Arc::new(ConfigSnapshot {
            resolved,
            operator: new_config,
        }));
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    fn make_config(system_prompt: &str) -> WeftConfig {
        let toml = format!(
            r#"
[server]
bind_address = "0.0.0.0:8080"

[gateway]
system_prompt = "{system_prompt}"
max_command_iterations = 5
request_timeout_secs = 30

[router]
[router.classifier]
model_path = "models/classifier"
tokenizer_path = "models/tokenizer"
threshold = 0.5
max_commands = 10

[[router.providers]]
name = "anthropic"
wire_format = "anthropic"
api_key = "sk-test"

[[router.providers.models]]
name = "claude"
model = "claude-sonnet-4-20250514"
max_tokens = 4096
examples = ["Write a poem", "Explain something", "What is X?"]
capabilities = ["chat_completions"]
"#
        );
        toml::from_str(&toml).expect("test TOML must parse")
    }

    #[test]
    fn test_new_snapshot_matches_from_operator() {
        let config = make_config("Hello world");
        let expected_resolved = ResolvedConfig::from_operator(&config);
        let store = ConfigStore::new(config);

        let snapshot = store.snapshot();
        assert_eq!(snapshot.system_prompt, expected_resolved.system_prompt);
        assert_eq!(snapshot.default_model, expected_resolved.default_model);
    }

    #[test]
    fn test_operator_config_returns_full_config() {
        let config = make_config("Test prompt");
        let expected_system_prompt = config.gateway.system_prompt.clone();
        let store = ConfigStore::new(config);

        let operator = store.operator_config();
        assert_eq!(operator.gateway.system_prompt, expected_system_prompt);
    }

    #[test]
    fn test_swap_changes_snapshot() {
        let config_v1 = make_config("Version 1");
        let config_v2 = make_config("Version 2");
        let store = ConfigStore::new(config_v1);

        let snap_before = store.snapshot();
        assert_eq!(snap_before.system_prompt, "Version 1");

        store.swap(config_v2);

        let snap_after = store.snapshot();
        assert_eq!(snap_after.system_prompt, "Version 2");
        assert_ne!(snap_before.system_prompt, snap_after.system_prompt);
    }

    #[test]
    fn test_swap_updates_operator_config() {
        let config_v1 = make_config("Version 1");
        let config_v2 = make_config("Version 2");
        let store = ConfigStore::new(config_v1);

        store.swap(config_v2);

        let operator = store.operator_config();
        assert_eq!(operator.gateway.system_prompt, "Version 2");
    }

    #[test]
    fn test_swap_atomicity_snapshot_and_operator_config_consistent() {
        let config_v2 = make_config("Consistent");
        let store = ConfigStore::new(make_config("Initial"));

        store.swap(config_v2);

        let snapshot = store.snapshot();
        let operator = store.operator_config();

        // Both resolved and operator must reflect the same config version.
        assert_eq!(snapshot.system_prompt, operator.gateway.system_prompt);
    }

    #[test]
    fn test_snapshot_pinned_after_swap() {
        let config_v1 = make_config("Pinned version");
        let config_v2 = make_config("New version");
        let store = ConfigStore::new(config_v1);

        // Take snapshot BEFORE swap.
        let pinned = store.snapshot();
        assert_eq!(pinned.system_prompt, "Pinned version");

        // Swap to new config.
        store.swap(config_v2);

        // Pinned snapshot must NOT reflect the swap.
        assert_eq!(pinned.system_prompt, "Pinned version");

        // New snapshot DOES reflect the swap.
        let new_snap = store.snapshot();
        assert_eq!(new_snap.system_prompt, "New version");
    }
}
