//! Context window assembly for the gateway engine.
//!
//! Assembles the system prompt from three parts (in order):
//! 1. Weft foundational prompt (slash command format tutorial)
//! 2. Command stubs section (dash-list of commands with "Use when..." trigger conditions)
//! 3. Agent system prompt (from gateway config)

use weft_core::{
    CommandStub,
    toon::{fenced_toon, serialize_table},
};

/// The foundational system prompt that teaches the LLM the slash command format.
///
/// `{command_stubs}` is a placeholder replaced at runtime with a dash-list of
/// available commands, each formatted as `- /name — Use when <description>`.
const FOUNDATIONAL_PROMPT_TEMPLATE: &str = r#"You have commands available that provide specialized capabilities and actions.

When users ask you to perform tasks, check if any of the available commands match. Commands provide capabilities beyond your training data.

When a command matches the user request, this is a BLOCKING REQUIREMENT: call the relevant command BEFORE generating any other response about the task. NEVER mention a command without actually calling it.

How to call a command:
- /command_name key: value, key2: "quoted value"
- Examples:
  - /some_command target: "example value"
  - /another_command id: 42, verbose: true
  - /third_command query: "multi word search"

How to learn about a command before using it:
- /command_name --describe

Available commands:

{command_stubs}

If no command matches the request, respond directly. Do not explain or apologize about commands."#;

/// Assemble the full system prompt with command stubs injected as a dash-list.
///
/// Order: foundational prompt (with dash-list command stubs) + agent system prompt.
///
/// # Arguments
/// - `selected_commands`: The semantically-filtered list of command stubs to inject.
/// - `agent_system_prompt`: The operator-configured system prompt from `gateway.system_prompt`.
/// - `memory_stubs`: Optional built-in memory command stubs `(name, description)`.
///   When `Some`, `/recall` and `/remember` stubs are appended after the tool registry commands.
///   These are always available when memory stores are configured — not gated by semantic routing.
pub fn assemble_system_prompt(
    selected_commands: &[CommandStub],
    agent_system_prompt: &str,
    memory_stubs: Option<&[(&str, &str)]>,
) -> String {
    // Build dash-list from tool registry commands: `- /name — Use when <description>`
    let mut stub_lines: Vec<String> = selected_commands
        .iter()
        .map(|cmd| format!("- /{} \u{2014} Use when {}", cmd.name, cmd.description))
        .collect();

    // Append built-in memory command stubs after tool registry commands.
    if let Some(mem_stubs) = memory_stubs {
        for (name, description) in mem_stubs {
            stub_lines.push(format!("- /{name} \u{2014} Use when {description}"));
        }
    }

    let command_stubs = stub_lines.join("\n");

    // Replace the placeholder in the foundational prompt
    let foundational = FOUNDATIONAL_PROMPT_TEMPLATE.replace("{command_stubs}", &command_stubs);

    // Concatenate foundational + agent prompt
    if agent_system_prompt.trim().is_empty() {
        foundational
    } else {
        format!("{foundational}\n\n{agent_system_prompt}")
    }
}

/// Assemble a system prompt without command injection.
///
/// Used when the semantic router determines tools are not needed for this turn
/// (`tools_needed = Some(false)` and `skip_tools_when_unnecessary = true`).
/// The system prompt is just the agent system prompt — no foundational command
/// instructions, no TOON block.
///
/// If the LLM emits slash commands anyway, the parser runs with an empty known-
/// commands set and treats them as prose.
pub fn assemble_system_prompt_no_tools(agent_system_prompt: &str) -> String {
    agent_system_prompt.to_string()
}

/// Format command results as TOON for injection back into the conversation.
///
/// Produces a fenced TOON block containing the results of all executed commands.
pub fn format_command_results_toon(results: &[weft_core::CommandResult]) -> String {
    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            let status = if r.success {
                "success".to_string()
            } else {
                "error".to_string()
            };
            // Use output if available, fall back to error message
            let output = if r.output.is_empty() {
                r.error.clone().unwrap_or_default()
            } else {
                r.output.clone()
            };
            vec![r.command_name.clone(), status, output]
        })
        .collect();

    let table = serialize_table("results", &["command", "status", "output"], &rows);
    fenced_toon(&table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use weft_core::{CommandResult, CommandStub};

    fn make_stub(name: &str, desc: &str) -> CommandStub {
        CommandStub {
            name: name.to_string(),
            description: desc.to_string(),
        }
    }

    // ── assemble_system_prompt ─────────────────────────────────────────────

    #[test]
    fn test_system_prompt_contains_foundational_and_agent() {
        let stubs = vec![make_stub(
            "web_search",
            "search the web for current information",
        )];
        let agent_prompt = "You are a helpful assistant.";
        let prompt = assemble_system_prompt(&stubs, agent_prompt, None);

        // Contains foundational preamble
        assert!(prompt.contains("You have commands available"));
        assert!(prompt.contains("BLOCKING REQUIREMENT"));
        assert!(prompt.contains("--describe"));

        // Contains command in dash-list format
        assert!(prompt.contains("web_search"));

        // Contains agent prompt after foundational
        assert!(prompt.contains("You are a helpful assistant."));

        // Order: foundational before agent
        let foundational_pos = prompt.find("You have commands available").unwrap();
        let agent_pos = prompt.find("You are a helpful assistant.").unwrap();
        assert!(
            foundational_pos < agent_pos,
            "foundational must come before agent prompt"
        );
    }

    #[test]
    fn test_system_prompt_dash_list_format_for_command_stubs() {
        let stubs = vec![
            make_stub("web_search", "search the web for current information"),
            make_stub("code_review", "review code for issues and improvements"),
        ];
        let prompt = assemble_system_prompt(&stubs, "", None);

        // Dash-list format: `- /name — Use when <description>`
        assert!(
            prompt
                .contains("- /web_search \u{2014} Use when search the web for current information")
        );
        assert!(
            prompt.contains(
                "- /code_review \u{2014} Use when review code for issues and improvements"
            )
        );
        // No TOON fenced blocks for command stubs
        assert!(!prompt.contains("commands[2]{name, description}:"));
    }

    #[test]
    fn test_system_prompt_no_commands_shows_empty_stubs() {
        let prompt = assemble_system_prompt(&[], "Do something.", None);

        // No TOON table — placeholder replaced with empty string
        assert!(!prompt.contains("commands[0]{name, description}:"));
        // Still contains the "Available commands:" header
        assert!(prompt.contains("Available commands:"));
    }

    #[test]
    fn test_system_prompt_empty_agent_prompt() {
        let stubs = vec![make_stub("recall", "retrieve memory")];
        let prompt = assemble_system_prompt(&stubs, "", None);

        // Should still produce valid prompt with foundational content
        assert!(prompt.contains("You have commands available"));
        assert!(!prompt.ends_with("\n\n")); // No double-newline at end
    }

    #[test]
    fn test_system_prompt_contains_call_syntax_examples() {
        let prompt = assemble_system_prompt(&[], "", None);

        // The foundational prompt must show call syntax with key: value pairs
        assert!(prompt.contains("key: value, key2: \"quoted value\""));
        assert!(prompt.contains("NEVER mention a command without actually calling it"));
    }

    #[test]
    fn test_system_prompt_contains_blocking_requirement() {
        let prompt = assemble_system_prompt(&[], "", None);

        // Must have BLOCKING REQUIREMENT language
        assert!(prompt.contains("BLOCKING REQUIREMENT"));
        assert!(prompt.contains("NEVER mention a command without actually calling it"));
    }

    #[test]
    fn test_system_prompt_no_toon_jargon_in_stubs() {
        let stubs = vec![make_stub("web_search", "search the web")];
        let prompt = assemble_system_prompt(&stubs, "", None);
        // Command stubs section must not use TOON fenced blocks
        assert!(!prompt.contains("```toon"));
        // But command stubs show in dash-list format
        assert!(prompt.contains("- /web_search \u{2014} Use when search the web"));
    }

    #[test]
    fn test_system_prompt_use_when_trigger_conditions() {
        let stubs = vec![make_stub("web_search", "search the web")];
        let prompt = assemble_system_prompt(&stubs, "", None);
        // "Use when" trigger condition must appear before each command description
        assert!(prompt.contains("Use when search the web"));
    }

    // ── assemble_system_prompt_no_tools ───────────────────────────────────

    #[test]
    fn test_no_tools_prompt_contains_only_agent_prompt() {
        let prompt = assemble_system_prompt_no_tools("You are a helpful assistant.");
        assert_eq!(prompt, "You are a helpful assistant.");
    }

    #[test]
    fn test_no_tools_prompt_has_no_toon_blocks() {
        let prompt = assemble_system_prompt_no_tools("Answer without tools.");
        assert!(!prompt.contains("```toon"), "must not contain TOON blocks");
        assert!(
            !prompt.contains("commands["),
            "must not contain command stubs"
        );
        assert!(
            !prompt.contains("You have access to commands"),
            "must not contain foundational prompt"
        );
    }

    #[test]
    fn test_no_tools_prompt_empty_agent() {
        let prompt = assemble_system_prompt_no_tools("");
        assert_eq!(prompt, "");
    }

    // ── format_command_results_toon ────────────────────────────────────────

    #[test]
    fn test_format_results_basic() {
        let results = vec![CommandResult {
            command_name: "web_search".to_string(),
            success: true,
            output: "Found 3 results".to_string(),
            error: None,
        }];
        let toon = format_command_results_toon(&results);

        // Wrapped in fenced block
        assert!(toon.contains("```toon"));
        // TOON array format
        assert!(toon.contains("results[1]{command, status, output}:"));
        assert!(toon.contains("web_search"));
        assert!(toon.contains("success"));
        assert!(toon.contains("Found 3 results"));
    }

    #[test]
    fn test_format_results_error_status() {
        let results = vec![CommandResult {
            command_name: "bad_cmd".to_string(),
            success: false,
            output: String::new(),
            error: Some("not found".to_string()),
        }];
        let toon = format_command_results_toon(&results);

        assert!(toon.contains("bad_cmd"));
        assert!(toon.contains("error"));
        assert!(toon.contains("not found"));
    }

    #[test]
    fn test_format_results_multiple() {
        let results = vec![
            CommandResult {
                command_name: "web_search".to_string(),
                success: true,
                output: "Result A".to_string(),
                error: None,
            },
            CommandResult {
                command_name: "docs_search".to_string(),
                success: false,
                output: String::new(),
                error: Some("docs_search is not a registered command".to_string()),
            },
        ];
        let toon = format_command_results_toon(&results);

        assert!(toon.contains("web_search"));
        assert!(toon.contains("docs_search"));
        assert!(toon.contains("success"));
        assert!(toon.contains("error"));
        // Two results
        assert!(toon.contains("results[2]{command, status, output}:"));
    }

    #[test]
    fn test_format_results_empty() {
        let toon = format_command_results_toon(&[]);
        // Still produces a fenced block with the array header
        assert!(toon.contains("```toon"));
        assert!(toon.contains("results[0]{command, status, output}:"));
    }

    #[test]
    fn test_format_results_fenced_block() {
        let results = vec![CommandResult {
            command_name: "cmd".to_string(),
            success: true,
            output: "ok".to_string(),
            error: None,
        }];
        let toon = format_command_results_toon(&results);
        assert!(toon.starts_with("```toon\n"));
        assert!(toon.ends_with("```"));
    }

    // ── memory stubs in system prompt ────────────────────────────────────

    #[test]
    fn test_memory_stubs_included_when_provided() {
        let stubs = vec![make_stub("web_search", "search the web")];
        let memory_stubs: &[(&str, &str)] = &[
            (
                "recall",
                "you need to retrieve relevant context from memory",
            ),
            (
                "remember",
                "the user shares important information worth preserving",
            ),
        ];
        let prompt = assemble_system_prompt(&stubs, "Agent prompt.", Some(memory_stubs));

        // Both memory stubs appear in dash-list format
        assert!(
            prompt.contains(
                "- /recall \u{2014} Use when you need to retrieve relevant context from memory"
            ),
            "recall stub must appear in prompt"
        );
        assert!(
            prompt.contains("- /remember \u{2014} Use when the user shares important information worth preserving"),
            "remember stub must appear in prompt"
        );
        // Tool stubs still present
        assert!(prompt.contains("- /web_search \u{2014} Use when search the web"));
    }

    #[test]
    fn test_memory_stubs_excluded_when_none() {
        let stubs = vec![make_stub("web_search", "search the web")];
        let prompt = assemble_system_prompt(&stubs, "Agent prompt.", None);

        // No memory stubs injected
        assert!(
            !prompt.contains("/recall"),
            "recall must not appear when no memory config"
        );
        assert!(
            !prompt.contains("/remember"),
            "remember must not appear when no memory config"
        );
        // Tool stubs still present
        assert!(prompt.contains("- /web_search \u{2014} Use when search the web"));
    }

    #[test]
    fn test_memory_stubs_appended_after_tool_stubs() {
        let stubs = vec![make_stub("web_search", "search the web")];
        let memory_stubs: &[(&str, &str)] = &[("recall", "you need to retrieve relevant context")];
        let prompt = assemble_system_prompt(&stubs, "", Some(memory_stubs));

        // Memory stubs appear after tool stubs
        let tool_pos = prompt.find("web_search").expect("tool stub must appear");
        let memory_pos = prompt.find("recall").expect("memory stub must appear");
        assert!(
            tool_pos < memory_pos,
            "tool stubs must appear before memory stubs"
        );
    }

    #[test]
    fn test_no_memory_stubs_when_empty_slice() {
        let prompt = assemble_system_prompt(&[], "", Some(&[]));
        // Empty memory stubs slice should produce the same output as None
        assert!(!prompt.contains("/recall"));
        assert!(!prompt.contains("/remember"));
    }
}
