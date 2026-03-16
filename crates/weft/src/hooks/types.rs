//! Hook response types and matcher.
//!
//! `HookDecision`, `HookResponse`, and `HookMatcher` live here.
//! `HookEvent`, `HookRoutingDomain`, and `RoutingTrigger` live in
//! `weft_core::config` alongside `HookConfig`, to avoid circular dependencies.

/// The decision a hook makes about a lifecycle event.
// Used by HookResponse (returned from executors) and run_chain (Phase 4 wiring).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HookDecision {
    /// Allow the operation to proceed unchanged.
    Allow,
    /// Block the operation. Only effective on blocking events.
    /// On non-blocking events, logged and treated as Allow.
    Block,
    /// Modify the event payload and continue.
    Modify,
}

/// Response from a hook execution.
///
/// Rhai scripts return this directly (via registered type).
/// HTTP hooks return this as JSON.
// Used by HookExecutor::execute implementations (Phase 2/3 wiring).
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HookResponse {
    /// The decision: allow, block, or modify.
    pub decision: HookDecision,
    /// Why the hook made this decision. Required when `decision` is `Block`.
    /// Fed back to the LLM or client as the error reason.
    pub reason: Option<String>,
    /// The modified payload. Required when `decision` is `Modify`.
    /// Must conform to the event's payload schema.
    pub modified: Option<serde_json::Value>,
    /// Additional context to inject into the conversation.
    /// Appended as a system-level annotation visible to the LLM on the next turn.
    pub context: Option<String>,
}

// allow() and block() are used by executor implementations (Phase 2/3) and tests.
#[allow(dead_code)]
impl HookResponse {
    /// Convenience constructor for an Allow response with no modifications.
    pub fn allow() -> Self {
        Self {
            decision: HookDecision::Allow,
            reason: None,
            modified: None,
            context: None,
        }
    }

    /// Convenience constructor for a Block response.
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            decision: HookDecision::Block,
            reason: Some(reason.into()),
            modified: None,
            context: None,
        }
    }
}

/// Compiled matcher for a hook. Determines whether a hook fires for a given event instance.
pub struct HookMatcher {
    /// Compiled regex. `None` means the hook fires for all events of its type.
    // Used by HookRegistry::run_chain (Phase 4 wiring).
    #[allow(dead_code)]
    regex: Option<regex::Regex>,
    /// Original pattern string, for error messages and debug output.
    pattern: Option<String>,
}

impl HookMatcher {
    /// Construct a matcher from an optional regex pattern string.
    ///
    /// Returns an error if the pattern is provided but invalid.
    /// The error message includes the hook index for operator diagnostics.
    pub fn new(pattern: Option<&str>, hook_index: usize) -> Result<Self, crate::hooks::HookError> {
        match pattern {
            None => Ok(Self {
                regex: None,
                pattern: None,
            }),
            Some(p) => {
                // Anchor the pattern: full-string match semantics (^pattern$).
                let anchored = format!("^(?:{p})$");
                let regex = regex::Regex::new(&anchored).map_err(|e| {
                    crate::hooks::HookError::RegistryError(format!(
                        "hooks[{hook_index}]: invalid matcher regex '{p}': {e}"
                    ))
                })?;
                Ok(Self {
                    regex: Some(regex),
                    pattern: Some(p.to_string()),
                })
            }
        }
    }

    /// Whether this hook matches the given target string.
    ///
    /// A `None` regex always matches (the hook fires for all events of its type).
    /// `target` is event-specific: the routing domain name, command name, etc.
    /// Pass `None` for events where matchers are ignored (RequestStart, PreResponse, RequestEnd).
    // Used by HookRegistry::run_chain and tests.
    #[allow(dead_code)]
    pub fn matches(&self, target: Option<&str>) -> bool {
        match (&self.regex, target) {
            // No regex: always fires.
            (None, _) => true,
            // Has regex, target provided: evaluate.
            (Some(re), Some(t)) => re.is_match(t),
            // Has regex, no target: fire only if no matcher was configured.
            // This branch is unreachable given how we construct matchers (we always
            // provide a target for events that support matchers). If we somehow reach
            // it, treat as: matcher was configured but no target to match against,
            // so the hook does NOT fire.
            (Some(_), None) => false,
        }
    }

    /// Returns the original pattern string, if any.
    // Used for diagnostics and debug output.
    #[allow(dead_code)]
    pub fn pattern(&self) -> Option<&str> {
        self.pattern.as_deref()
    }
}

impl std::fmt::Debug for HookMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookMatcher")
            .field("pattern", &self.pattern)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allow_constructor() {
        let r = HookResponse::allow();
        assert_eq!(r.decision, HookDecision::Allow);
        assert!(r.reason.is_none());
        assert!(r.modified.is_none());
        assert!(r.context.is_none());
    }

    #[test]
    fn test_block_constructor() {
        let r = HookResponse::block("not allowed");
        assert_eq!(r.decision, HookDecision::Block);
        assert_eq!(r.reason.as_deref(), Some("not allowed"));
        assert!(r.modified.is_none());
        assert!(r.context.is_none());
    }

    #[test]
    fn test_decision_serde_round_trip() {
        for (decision, expected_json) in [
            (HookDecision::Allow, r#""allow""#),
            (HookDecision::Block, r#""block""#),
            (HookDecision::Modify, r#""modify""#),
        ] {
            let json = serde_json::to_string(&decision).unwrap();
            assert_eq!(json, expected_json);
            let back: HookDecision = serde_json::from_str(&json).unwrap();
            assert_eq!(back, decision);
        }
    }

    #[test]
    fn test_hook_response_serde_round_trip() {
        let response = HookResponse {
            decision: HookDecision::Block,
            reason: Some("blocked".to_string()),
            modified: None,
            context: Some("ctx".to_string()),
        };
        let json = serde_json::to_string(&response).unwrap();
        let back: HookResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.decision, HookDecision::Block);
        assert_eq!(back.reason.as_deref(), Some("blocked"));
        assert_eq!(back.context.as_deref(), Some("ctx"));
    }

    #[test]
    fn test_matcher_no_regex_always_matches() {
        let m = HookMatcher::new(None, 0).unwrap();
        assert!(m.matches(None));
        assert!(m.matches(Some("model")));
        assert!(m.matches(Some("anything")));
    }

    #[test]
    fn test_matcher_exact_match() {
        let m = HookMatcher::new(Some("model"), 0).unwrap();
        assert!(m.matches(Some("model")));
        assert!(!m.matches(Some("commands")));
        assert!(!m.matches(Some("memory")));
    }

    #[test]
    fn test_matcher_alternation() {
        let m = HookMatcher::new(Some("model|memory"), 0).unwrap();
        assert!(m.matches(Some("model")));
        assert!(m.matches(Some("memory")));
        assert!(!m.matches(Some("commands")));
        assert!(!m.matches(Some("tool_necessity")));
    }

    #[test]
    fn test_matcher_routing_domains() {
        // "model|commands" matches both model and commands, not others.
        let m = HookMatcher::new(Some("model|commands"), 0).unwrap();
        assert!(m.matches(Some("model")));
        assert!(m.matches(Some("commands")));
        assert!(!m.matches(Some("memory")));
        assert!(!m.matches(Some("tool_necessity")));
    }

    #[test]
    fn test_matcher_no_match() {
        let m = HookMatcher::new(Some("web_search"), 0).unwrap();
        assert!(!m.matches(Some("model")));
        assert!(m.matches(Some("web_search")));
    }

    #[test]
    fn test_matcher_no_target_with_regex_does_not_match() {
        let m = HookMatcher::new(Some("model"), 0).unwrap();
        // No target provided: matcher configured but nothing to match against.
        assert!(!m.matches(None));
    }

    #[test]
    fn test_matcher_invalid_regex_returns_error() {
        let result = HookMatcher::new(Some("[invalid"), 3);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("hooks[3]"));
        assert!(err.contains("invalid"));
    }
}
