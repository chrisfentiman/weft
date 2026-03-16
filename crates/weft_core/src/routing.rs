//! Model routing instruction parser and filter resolution.
//!
//! The routing instruction is a slash-separated string parsed from the request's `model` field.
//! It specifies the routing mode and optional filter segments used to narrow the model candidate set.

use crate::error::WeftError;

// ── Routing Instruction ───────────────────────────────────────────────────────

/// A parsed model routing instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoutingInstruction {
    /// The routing mode.
    pub mode: RoutingMode,
    /// Filter segments applied left-to-right to narrow candidates.
    pub filters: Vec<String>,
    /// The original string, for echo in responses.
    pub raw: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMode {
    /// Semantic router picks the best model.
    Auto,
    /// Council: parallel multi-model with judge synthesis.
    Council,
    /// Direct: filters applied, best match wins.
    Direct,
}

impl ModelRoutingInstruction {
    /// Parse a model routing string into a structured instruction.
    ///
    /// Rules:
    /// - Empty string or "auto" -> Auto mode, no filters
    /// - "auto/X/Y" -> Auto mode with filters [X, Y]
    /// - "council" -> Council mode, no filters
    /// - "council/X/Y" -> Council mode with filters [X, Y]
    /// - "X/Y" -> Direct mode with filters [X, Y]
    /// - "X" -> Direct mode with filter [X]
    ///
    /// Empty segments (from trailing or consecutive slashes) are ignored.
    pub fn parse(input: &str) -> Self {
        let raw = input.to_string();
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return Self {
                mode: RoutingMode::Auto,
                filters: vec![],
                raw,
            };
        }

        let segments: Vec<&str> = trimmed.split('/').collect();
        let (mode, filter_start) = match segments[0] {
            "auto" => (RoutingMode::Auto, 1),
            "council" => (RoutingMode::Council, 1),
            _ => (RoutingMode::Direct, 0),
        };

        // Filter out empty segments (from trailing/consecutive slashes)
        let filters: Vec<String> = segments[filter_start..]
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        Self { mode, filters, raw }
    }
}

// ── Filter Resolution ─────────────────────────────────────────────────────────

/// Lightweight model info for filter resolution. Constructed from ProviderRegistry.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub routing_name: String,
    pub provider_name: String,
    pub capabilities: Vec<String>,
}

/// Resolve filters against the provider registry to produce a candidate set.
///
/// Each filter narrows the surviving set. An empty filter list means all models.
/// Returns the set of model routing names that survive all filters.
///
/// Resolution order per filter: model name > provider name > capability.
/// Model groups are a future addition (not yet in config).
pub fn resolve_filters(filters: &[String], all_models: &[ModelInfo]) -> Vec<String> {
    let mut survivors: Vec<String> = all_models.iter().map(|m| m.routing_name.clone()).collect();

    for filter in filters {
        let next: Vec<String> = survivors
            .iter()
            .filter(|name| {
                // SAFETY: all survivor names came from all_models so find always returns Some.
                let info = all_models
                    .iter()
                    .find(|m| &m.routing_name == *name)
                    .expect("survivor must be in all_models");

                // 1. Exact model name match
                info.routing_name == *filter
                // 2. Provider name match
                || info.provider_name == *filter
                // 3. Capability match
                || info.capabilities.iter().any(|c| c == filter)
            })
            .cloned()
            .collect();
        survivors = next;
    }

    survivors
}

// ── Council Validation ────────────────────────────────────────────────────────

/// Validate the routing instruction against resolved candidates.
///
/// Fails if no models survive. Council mode additionally requires at least 2 survivors.
pub fn validate_routing(
    instruction: &ModelRoutingInstruction,
    survivors: &[String],
) -> Result<(), WeftError> {
    if survivors.is_empty() {
        return Err(WeftError::Routing(format!(
            "no models match routing instruction '{}'",
            instruction.raw
        )));
    }
    if instruction.mode == RoutingMode::Council && survivors.len() < 2 {
        return Err(WeftError::Routing(format!(
            "council mode requires at least 2 models, but '{}' resolves to {}",
            instruction.raw,
            survivors.len()
        )));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Parser tests ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_string_is_auto() {
        let inst = ModelRoutingInstruction::parse("");
        assert_eq!(inst.mode, RoutingMode::Auto);
        assert!(inst.filters.is_empty());
        assert_eq!(inst.raw, "");
    }

    #[test]
    fn test_parse_auto_is_auto_no_filters() {
        let inst = ModelRoutingInstruction::parse("auto");
        assert_eq!(inst.mode, RoutingMode::Auto);
        assert!(inst.filters.is_empty());
        assert_eq!(inst.raw, "auto");
    }

    #[test]
    fn test_parse_auto_with_single_filter() {
        let inst = ModelRoutingInstruction::parse("auto/anthropic");
        assert_eq!(inst.mode, RoutingMode::Auto);
        assert_eq!(inst.filters, vec!["anthropic"]);
    }

    #[test]
    fn test_parse_auto_with_multiple_filters() {
        let inst = ModelRoutingInstruction::parse("auto/anthropic/vision");
        assert_eq!(inst.mode, RoutingMode::Auto);
        assert_eq!(inst.filters, vec!["anthropic", "vision"]);
    }

    #[test]
    fn test_parse_council_no_filters() {
        let inst = ModelRoutingInstruction::parse("council");
        assert_eq!(inst.mode, RoutingMode::Council);
        assert!(inst.filters.is_empty());
        assert_eq!(inst.raw, "council");
    }

    #[test]
    fn test_parse_council_with_filters() {
        let inst = ModelRoutingInstruction::parse("council/complex");
        assert_eq!(inst.mode, RoutingMode::Council);
        assert_eq!(inst.filters, vec!["complex"]);
    }

    #[test]
    fn test_parse_council_with_multiple_filters() {
        let inst = ModelRoutingInstruction::parse("council/openai/anthropic");
        assert_eq!(inst.mode, RoutingMode::Council);
        assert_eq!(inst.filters, vec!["openai", "anthropic"]);
    }

    #[test]
    fn test_parse_single_segment_is_direct() {
        let inst = ModelRoutingInstruction::parse("anthropic");
        assert_eq!(inst.mode, RoutingMode::Direct);
        assert_eq!(inst.filters, vec!["anthropic"]);
    }

    #[test]
    fn test_parse_provider_model_is_direct() {
        let inst = ModelRoutingInstruction::parse("anthropic/opus");
        assert_eq!(inst.mode, RoutingMode::Direct);
        assert_eq!(inst.filters, vec!["anthropic", "opus"]);
    }

    #[test]
    fn test_parse_raw_preserved() {
        let input = "council/openai/vision";
        let inst = ModelRoutingInstruction::parse(input);
        assert_eq!(inst.raw, input);
    }

    #[test]
    fn test_parse_trailing_slash_ignored() {
        // Trailing slashes produce empty segments that are filtered out
        let inst = ModelRoutingInstruction::parse("auto/anthropic/");
        assert_eq!(inst.mode, RoutingMode::Auto);
        assert_eq!(inst.filters, vec!["anthropic"]);
    }

    #[test]
    fn test_parse_consecutive_slashes_ignored() {
        // Consecutive slashes produce empty segments that are filtered out
        let inst = ModelRoutingInstruction::parse("council//openai");
        assert_eq!(inst.mode, RoutingMode::Council);
        assert_eq!(inst.filters, vec!["openai"]);
    }

    #[test]
    fn test_parse_whitespace_trimmed() {
        let inst = ModelRoutingInstruction::parse("  auto  ");
        assert_eq!(inst.mode, RoutingMode::Auto);
        assert!(inst.filters.is_empty());
    }

    // ── Filter resolution tests ───────────────────────────────────────────

    fn make_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                routing_name: "anthropic/opus".to_string(),
                provider_name: "anthropic".to_string(),
                capabilities: vec!["vision".to_string()],
            },
            ModelInfo {
                routing_name: "anthropic/haiku".to_string(),
                provider_name: "anthropic".to_string(),
                capabilities: vec!["fast".to_string()],
            },
            ModelInfo {
                routing_name: "openai/gpt-4o".to_string(),
                provider_name: "openai".to_string(),
                capabilities: vec!["vision".to_string(), "audio_transcriptions".to_string()],
            },
            ModelInfo {
                routing_name: "openai/gpt-4o-mini".to_string(),
                provider_name: "openai".to_string(),
                capabilities: vec!["fast".to_string()],
            },
        ]
    }

    #[test]
    fn test_resolve_filters_empty_returns_all() {
        let models = make_models();
        let result = resolve_filters(&[], &models);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_resolve_filters_by_model_name() {
        let models = make_models();
        let filters = vec!["anthropic/opus".to_string()];
        let result = resolve_filters(&filters, &models);
        assert_eq!(result, vec!["anthropic/opus"]);
    }

    #[test]
    fn test_resolve_filters_by_provider_name() {
        let models = make_models();
        let filters = vec!["anthropic".to_string()];
        let result = resolve_filters(&filters, &models);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"anthropic/opus".to_string()));
        assert!(result.contains(&"anthropic/haiku".to_string()));
    }

    #[test]
    fn test_resolve_filters_by_capability() {
        let models = make_models();
        let filters = vec!["vision".to_string()];
        let result = resolve_filters(&filters, &models);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"anthropic/opus".to_string()));
        assert!(result.contains(&"openai/gpt-4o".to_string()));
    }

    #[test]
    fn test_resolve_filters_multiple_narrow() {
        let models = make_models();
        // First filter to openai, then filter to vision — only gpt-4o has both
        let filters = vec!["openai".to_string(), "vision".to_string()];
        let result = resolve_filters(&filters, &models);
        assert_eq!(result, vec!["openai/gpt-4o"]);
    }

    #[test]
    fn test_resolve_filters_no_match_returns_empty() {
        let models = make_models();
        let filters = vec!["nonexistent_provider".to_string()];
        let result = resolve_filters(&filters, &models);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_filters_empty_models() {
        let result = resolve_filters(&["anthropic".to_string()], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_filters_capability_audio() {
        let models = make_models();
        let filters = vec!["audio_transcriptions".to_string()];
        let result = resolve_filters(&filters, &models);
        assert_eq!(result, vec!["openai/gpt-4o"]);
    }

    // ── Council validation tests ───────────────────────────────────────────

    #[test]
    fn test_validate_routing_empty_survivors_fails() {
        let inst = ModelRoutingInstruction::parse("anthropic");
        let err = validate_routing(&inst, &[]).unwrap_err();
        assert!(matches!(err, WeftError::Routing(_)));
        assert!(err.to_string().contains("no models match"));
    }

    #[test]
    fn test_validate_routing_council_one_survivor_fails() {
        let inst = ModelRoutingInstruction::parse("council");
        let survivors = vec!["anthropic/opus".to_string()];
        let err = validate_routing(&inst, &survivors).unwrap_err();
        assert!(matches!(err, WeftError::Routing(_)));
        assert!(err.to_string().contains("at least 2 models"));
    }

    #[test]
    fn test_validate_routing_council_two_survivors_ok() {
        let inst = ModelRoutingInstruction::parse("council");
        let survivors = vec!["anthropic/opus".to_string(), "openai/gpt-4o".to_string()];
        assert!(validate_routing(&inst, &survivors).is_ok());
    }

    #[test]
    fn test_validate_routing_council_three_survivors_ok() {
        let inst = ModelRoutingInstruction::parse("council");
        let survivors = vec![
            "anthropic/opus".to_string(),
            "openai/gpt-4o".to_string(),
            "gemini/pro".to_string(),
        ];
        assert!(validate_routing(&inst, &survivors).is_ok());
    }

    #[test]
    fn test_validate_routing_direct_one_survivor_ok() {
        let inst = ModelRoutingInstruction::parse("anthropic/opus");
        let survivors = vec!["anthropic/opus".to_string()];
        assert!(validate_routing(&inst, &survivors).is_ok());
    }

    #[test]
    fn test_validate_routing_auto_one_survivor_ok() {
        let inst = ModelRoutingInstruction::parse("auto");
        let survivors = vec!["anthropic/opus".to_string()];
        assert!(validate_routing(&inst, &survivors).is_ok());
    }

    #[test]
    fn test_validate_routing_error_message_includes_raw() {
        let inst = ModelRoutingInstruction::parse("council/vision");
        let survivors = vec!["anthropic/opus".to_string()];
        let err = validate_routing(&inst, &survivors).unwrap_err();
        assert!(err.to_string().contains("council/vision"));
    }
}
