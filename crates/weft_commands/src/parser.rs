//! Slash command parser: extracts `CommandInvocation` instances from LLM output text.
//!
//! The parser is initialized with the set of registered command names to avoid false positives.
//! Only `/name` patterns where `name` is a registered command are matched.
//!
//! TOON arguments from the LLM output are converted to `serde_json::Value::Object` at the
//! parse boundary (in `weft_core::toon::parse_toon_args`).

use std::collections::HashSet;

use weft_core::{CommandAction, CommandInvocation, CommandResult, toon::parse_toon_args};

/// Result of parsing an LLM response.
#[derive(Debug, Clone)]
pub struct ParsedResponse {
    /// The clean text (everything outside command invocations).
    pub text: String,
    /// All parsed command invocations, in order of appearance.
    pub invocations: Vec<CommandInvocation>,
    /// Commands whose TOON argument parsing failed. Each has `success: false`.
    /// These are not included in `invocations`.
    pub parse_errors: Vec<CommandResult>,
}

/// Parse LLM output into clean text and command invocations.
///
/// `known_commands` must be the set of registered command names. Only `/name` where `name`
/// appears in this set will be matched as commands.
///
/// Any command whose TOON argument parsing fails produces a `CommandResult` with
/// `success: false` in `ParsedResponse::parse_errors`, and is NOT added to `invocations`.
/// The rest of parsing continues normally.
pub fn parse_response(text: &str, known_commands: &HashSet<String>) -> ParsedResponse {
    let mut invocations = Vec::new();
    let mut parse_errors: Vec<CommandResult> = Vec::new();
    let mut clean_parts: Vec<String> = Vec::new();

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Try to match a command on this line.
        if let Some((cmd_name, rest_of_line)) = try_match_command(line, known_commands) {
            // Collect any indented continuation lines (2+ leading spaces).
            let mut arg_parts: Vec<String> = Vec::new();

            // rest_of_line is the text after the command name on the same line (trimmed).
            // It may be empty, a flag, or inline TOON args.
            let inline = rest_of_line.trim().to_string();
            if !inline.is_empty() {
                arg_parts.push(inline.clone());
            }

            i += 1;
            while i < lines.len() {
                let next = lines[i];
                if is_continuation_line(next) {
                    arg_parts.push(next.trim().to_string());
                    i += 1;
                } else {
                    break;
                }
            }

            // Determine the action: detect flags from the inline portion only.
            // A flag like `--describe` or `--verbose` is "standalone" when the inline
            // contains only the flag and nothing else (no TOON key-value pairs). Continuation
            // lines are never flag-only lines, so they don't affect flag detection.
            let action = detect_flag(inline.trim());

            match action {
                FlagAction::Describe => {
                    invocations.push(CommandInvocation {
                        name: cmd_name.clone(),
                        action: CommandAction::Describe,
                        arguments: serde_json::Value::Null,
                    });
                }
                FlagAction::UnknownFlag(_) => {
                    // Unknown flags: do NOT match as command (treat entire invocation as prose).
                    // Re-emit the command line and all consumed continuation lines as clean text.
                    clean_parts.push(line.to_string());
                    // arg_parts holds the trimmed content of continuation lines (and inline).
                    // Re-emit each consumed continuation line. We use the original indented form
                    // (two leading spaces + trimmed content) to faithfully reconstruct the line.
                    let continuation_start = if !inline.is_empty() { 1 } else { 0 };
                    for part in arg_parts.iter().skip(continuation_start) {
                        clean_parts.push(format!("  {part}"));
                    }
                }
                FlagAction::Execute => {
                    // Parse TOON args from all collected arg parts (inline + continuation).
                    let combined = arg_parts.join(", ");
                    let toon_input = combined.trim();
                    match parse_toon_args(toon_input) {
                        Ok(arguments) => {
                            invocations.push(CommandInvocation {
                                name: cmd_name.clone(),
                                action: CommandAction::Execute,
                                arguments,
                            });
                        }
                        Err(e) => {
                            parse_errors.push(CommandResult {
                                command_name: cmd_name.clone(),
                                success: false,
                                output: String::new(),
                                error: Some(format!(
                                    "TOON argument parse error for /{}: {}",
                                    cmd_name, e
                                )),
                            });
                        }
                    }
                }
            }
        } else {
            clean_parts.push(line.to_string());
            i += 1;
        }
    }

    // Reconstruct clean text, collapsing consecutive blank lines.
    let clean_text = clean_parts.join("\n").trim().to_string();

    ParsedResponse {
        text: clean_text,
        invocations,
        parse_errors,
    }
}

/// Describes what flag (if any) was detected in the argument text.
enum FlagAction {
    /// `--describe` or `--help` flag.
    Describe,
    /// Execute with TOON args (no flag, or no recognized flag).
    Execute,
    /// An unknown flag (anything other than --describe and --help).
    /// The flag name is stored for future diagnostics.
    UnknownFlag(#[allow(dead_code)] String),
}

/// Detect the flag in the argument text after the command name.
fn detect_flag(args: &str) -> FlagAction {
    let trimmed = args.trim();

    if trimmed.is_empty() {
        return FlagAction::Execute;
    }

    // Check for --describe or --help (these must be the only content, or first token)
    if trimmed == "--describe" || trimmed == "--help" {
        return FlagAction::Describe;
    }

    // Check for unknown flags: if it starts with `--` and is a single word
    // (no subsequent TOON key-value pairs), treat as unknown flag.
    if trimmed.starts_with("--") {
        // Extract the flag name (up to whitespace or end)
        let flag = trimmed
            .split_whitespace()
            .next()
            .unwrap_or(trimmed)
            .to_string();
        // If there's nothing after the flag, it's a standalone unknown flag
        let rest = trimmed[flag.len()..].trim();
        if rest.is_empty() {
            return FlagAction::UnknownFlag(flag);
        }
        // If there IS content after the flag, proceed as execute (mixed content)
        // This handles hypothetical `--schema key: value` cases in future.
    }

    FlagAction::Execute
}

/// Try to match a command name at the start of a line.
///
/// Returns `(command_name, text_after_command_name)` if a registered command is found.
/// The `/` must be at the start of the line or preceded only by whitespace.
///
/// After the command name, the next character must be whitespace, `--`, or end-of-line.
/// A single `-` does NOT terminate the name — only `--` or whitespace does. This prevents
/// `/web_search-related topic` from falsely matching `web_search` as a command.
fn try_match_command<'a>(
    line: &'a str,
    known_commands: &HashSet<String>,
) -> Option<(String, &'a str)> {
    let trimmed = line.trim_start();

    if !trimmed.starts_with('/') {
        return None;
    }

    // Strip the leading '/'
    let after_slash = &trimmed[1..];

    // The command name ends only at whitespace or end-of-string. Stopping at a single '-'
    // would cause `/web_search-related` to falsely extract `web_search`. The `--` flag
    // prefix is detected later by `detect_flag`, after the full name is extracted.
    let name_end = after_slash
        .find(|c: char| c.is_whitespace())
        .unwrap_or(after_slash.len());

    let candidate_name = &after_slash[..name_end];

    if candidate_name.is_empty() {
        return None;
    }

    // After the name, only whitespace or end-of-string is a valid boundary.
    // This rejects `/web_search--flag` (no space before --) as a false positive.
    let after_name = &after_slash[name_end..];
    let valid_boundary =
        after_name.is_empty() || after_name.starts_with(|c: char| c.is_whitespace());
    if !valid_boundary {
        return None;
    }

    if !known_commands.contains(candidate_name) {
        return None;
    }

    let rest = after_name.trim_start();
    Some((candidate_name.to_string(), rest))
}

/// Returns true if this line is a continuation line (2+ leading spaces).
fn is_continuation_line(line: &str) -> bool {
    line.len() >= 2 && line.starts_with("  ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn commands(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn parse(text: &str, cmds: &[&str]) -> ParsedResponse {
        parse_response(text, &commands(cmds))
    }

    // ── Basic matching ────────────────────────────────────────────────────

    #[test]
    fn test_no_commands_returns_clean_text() {
        let resp = parse("Hello, world!", &["web_search"]);
        assert_eq!(resp.text, "Hello, world!");
        assert!(resp.invocations.is_empty());
        assert!(resp.parse_errors.is_empty());
    }

    #[test]
    fn test_simple_command_no_args() {
        let resp = parse("/web_search", &["web_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(resp.invocations[0].name, "web_search");
        assert_eq!(resp.invocations[0].action, CommandAction::Execute);
        assert_eq!(resp.invocations[0].arguments, json!({}));
        assert!(resp.text.is_empty());
    }

    #[test]
    fn test_command_with_toon_args() {
        let resp = parse(
            r#"/web_search query: "Rust async patterns""#,
            &["web_search"],
        );
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(
            resp.invocations[0].arguments,
            json!({"query": "Rust async patterns"})
        );
    }

    #[test]
    fn test_command_with_multiple_toon_args() {
        let resp = parse(
            r#"/web_search query: "Rust async patterns 2026", max_results: 10"#,
            &["web_search"],
        );
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(
            resp.invocations[0].arguments,
            json!({"query": "Rust async patterns 2026", "max_results": 10})
        );
    }

    #[test]
    fn test_describe_flag() {
        let resp = parse("/docs_search --describe", &["docs_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(resp.invocations[0].name, "docs_search");
        assert_eq!(resp.invocations[0].action, CommandAction::Describe);
        assert_eq!(resp.invocations[0].arguments, serde_json::Value::Null);
    }

    #[test]
    fn test_help_flag_alias_for_describe() {
        let resp = parse("/docs_search --help", &["docs_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(resp.invocations[0].action, CommandAction::Describe);
    }

    // ── Multiple commands ─────────────────────────────────────────────────

    #[test]
    fn test_multiple_commands_in_response() {
        let text = "I'll search for that.\n\n/web_search query: \"Rust\"\n\nLet me check docs.\n\n/docs_search --describe";
        let resp = parse(text, &["web_search", "docs_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 2);
        assert_eq!(resp.invocations[0].name, "web_search");
        assert_eq!(resp.invocations[1].name, "docs_search");
        assert_eq!(resp.invocations[1].action, CommandAction::Describe);
        assert!(resp.text.contains("I'll search for that."));
        assert!(resp.text.contains("Let me check docs."));
    }

    // ── Unregistered commands treated as text ─────────────────────────────

    #[test]
    fn test_unregistered_command_is_text() {
        let resp = parse("/unknown_cmd foo: bar", &["web_search"]);
        assert!(resp.parse_errors.is_empty());
        assert!(resp.invocations.is_empty());
        assert!(resp.text.contains("/unknown_cmd foo: bar"));
    }

    #[test]
    fn test_path_not_matched() {
        let resp = parse(
            "See /usr/local/bin for details",
            &["web_search", "docs_search"],
        );
        assert!(resp.parse_errors.is_empty());
        assert!(resp.invocations.is_empty());
        // usr is not a registered command
        assert!(resp.text.contains("/usr/local/bin"));
    }

    #[test]
    fn test_api_path_not_matched() {
        let resp = parse("POST /v1/chat/completions", &["web_search", "docs_search"]);
        assert!(resp.parse_errors.is_empty());
        assert!(resp.invocations.is_empty());
    }

    // ── Command name boundary ─────────────────────────────────────────────

    #[test]
    fn test_hyphen_in_text_after_command_name_not_matched() {
        // `/web_search-related topic` must NOT match `web_search` — the name runs up to
        // the hyphen with no whitespace boundary, so it is not a registered command name.
        let resp = parse("/web_search-related topic", &["web_search"]);
        assert!(resp.invocations.is_empty());
        assert!(resp.parse_errors.is_empty());
        assert!(resp.text.contains("/web_search-related topic"));
    }

    // ── Multi-line arguments ──────────────────────────────────────────────

    #[test]
    fn test_multiline_arguments() {
        let text = "/create_document\n  title: \"Architecture Decision Record\"\n  draft: true";
        let resp = parse(text, &["create_document"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(resp.invocations[0].name, "create_document");
        assert_eq!(resp.invocations[0].action, CommandAction::Execute);
        assert_eq!(
            resp.invocations[0].arguments,
            json!({"title": "Architecture Decision Record", "draft": true})
        );
    }

    #[test]
    fn test_multiline_with_inline_and_continuation() {
        let text = "/web_search query: \"Rust\"\n  max_results: 10";
        let resp = parse(text, &["web_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(
            resp.invocations[0].arguments,
            json!({"query": "Rust", "max_results": 10})
        );
    }

    #[test]
    fn test_multiline_terminates_at_non_indented_line() {
        let text = "/create_document\n  title: \"Test\"\nThis is prose after.";
        let resp = parse(text, &["create_document"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert!(resp.text.contains("This is prose after."));
    }

    // ── Array arguments ───────────────────────────────────────────────────

    #[test]
    fn test_command_with_array_argument() {
        let resp = parse(
            r#"/create_note tags: [ml, ai, "deep learning"]"#,
            &["create_note"],
        );
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(
            resp.invocations[0].arguments,
            json!({"tags": ["ml", "ai", "deep learning"]})
        );
    }

    // ── Malformed TOON ────────────────────────────────────────────────────

    #[test]
    fn test_malformed_toon_produces_error_result() {
        let resp = parse("/web_search query:", &["web_search"]);
        // Parse error should be in parse_errors, not invocations
        assert_eq!(resp.invocations.len(), 0);
        assert_eq!(resp.parse_errors.len(), 1);
        assert_eq!(resp.parse_errors[0].command_name, "web_search");
        assert!(!resp.parse_errors[0].success);
        assert!(resp.parse_errors[0].error.is_some());
    }

    #[test]
    fn test_malformed_toon_does_not_abort_rest() {
        let text = "/web_search query:\n/docs_search query: \"hello\"";
        let resp = parse(text, &["web_search", "docs_search"]);
        // web_search fails, but docs_search succeeds
        assert_eq!(resp.invocations.len(), 1);
        assert_eq!(resp.invocations[0].name, "docs_search");
        assert_eq!(resp.parse_errors.len(), 1);
        assert_eq!(resp.parse_errors[0].command_name, "web_search");
    }

    // ── Unknown flags treated as prose ────────────────────────────────────

    #[test]
    fn test_unknown_flag_treated_as_prose() {
        // /web_search --verbose has an unknown flag — the entire invocation is prose
        let resp = parse("/web_search --verbose", &["web_search"]);
        assert_eq!(resp.invocations.len(), 0);
        assert!(resp.text.contains("/web_search --verbose"));
    }

    #[test]
    fn test_unknown_flag_with_continuation_lines_re_emitted() {
        // Unknown flag: the command and its continuation lines must all appear in clean text
        let text = "/web_search --verbose\n  query: \"rust\"\n  limit: 5";
        let resp = parse(text, &["web_search"]);
        assert_eq!(resp.invocations.len(), 0);
        assert!(resp.parse_errors.is_empty());
        // The command line and its continuation lines must all be in clean text
        assert!(resp.text.contains("/web_search --verbose"));
        assert!(resp.text.contains("query: \"rust\""));
        assert!(resp.text.contains("limit: 5"));
    }

    // ── Clean text extraction ─────────────────────────────────────────────

    #[test]
    fn test_clean_text_excludes_command_lines() {
        let text = "I'll help you.\n\n/web_search query: \"test\"\n\nDone.";
        let resp = parse(text, &["web_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 1);
        assert!(resp.text.contains("I'll help you."));
        assert!(resp.text.contains("Done."));
        assert!(!resp.text.contains("/web_search"));
    }

    #[test]
    fn test_only_commands_produces_empty_text() {
        let resp = parse("/web_search query: \"test\"", &["web_search"]);
        assert!(resp.parse_errors.is_empty());
        assert!(resp.text.is_empty());
    }

    // ── Spec example ─────────────────────────────────────────────────────

    #[test]
    fn test_spec_example_from_section_4_2() {
        let text = "I'll search for that information now.\n\n/web_search query: \"Rust async patterns 2026\"\n\nLet me also check the documentation.\n\n/docs_search --describe";
        let resp = parse(text, &["web_search", "docs_search"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 2);
        assert_eq!(resp.invocations[0].name, "web_search");
        assert_eq!(
            resp.invocations[0].arguments,
            json!({"query": "Rust async patterns 2026"})
        );
        assert_eq!(resp.invocations[1].name, "docs_search");
        assert_eq!(resp.invocations[1].action, CommandAction::Describe);
        assert!(resp.text.contains("I'll search for that information now."));
    }

    #[test]
    fn test_spec_multiline_example() {
        let text = "/web_search query: \"Rust async patterns 2026\", max_results: 10\n\n/create_note\n  title: \"Research findings\"\n  content: \"Found several relevant articles on async Rust patterns.\"\n  tags: [rust, async, research]";
        let resp = parse(text, &["web_search", "create_note"]);
        assert!(resp.parse_errors.is_empty());
        assert_eq!(resp.invocations.len(), 2);
        assert_eq!(resp.invocations[0].name, "web_search");
        assert_eq!(resp.invocations[1].name, "create_note");
        assert_eq!(
            resp.invocations[1].arguments,
            json!({
                "title": "Research findings",
                "content": "Found several relevant articles on async Rust patterns.",
                "tags": ["rust", "async", "research"]
            })
        );
    }
}
