//! Context window assembly for the gateway engine.
//!
//! Assembles the system prompt from three parts (in order):
//! 1. Weft foundational prompt (slash command format tutorial)
//! 2. Command stubs section (filtered commands in TOON fenced block)
//! 3. Agent system prompt (from gateway config)

use weft_core::{
    CommandStub,
    toon::{fenced_toon, serialize_table},
};

/// The foundational system prompt that teaches the LLM the slash command format.
///
/// `{command_stubs_toon}` is a placeholder replaced at runtime with a fenced TOON
/// block containing the semantically-selected command stubs.
const FOUNDATIONAL_PROMPT_TEMPLATE: &str = r#"You have access to commands. To use a command, write it as a slash command with TOON arguments:

/command_name key: value, key2: "value with spaces"

Arguments are key-value pairs. Quote values containing spaces or commas. Numbers and booleans are unquoted. Arrays use brackets: [a, b, c].

For many arguments, use multiple lines (indent with spaces):

/command_name
  key: value
  key2: "long value here"

To learn more about a command before using it:

/command_name --describe

{command_stubs_toon}

Rules:
- Only use commands listed above.
- Arguments use TOON format (key: value pairs), not JSON.
- You may use multiple commands in a single response.
- Text outside of slash commands is your response to the user.
- Wait for command results before drawing conclusions from them."#;

/// Assemble the full system prompt with command stubs injected as a TOON fenced block.
///
/// Order: foundational prompt (with fenced TOON command stubs) + agent system prompt.
///
/// # Arguments
/// - `selected_commands`: The semantically-filtered list of command stubs to inject.
/// - `agent_system_prompt`: The operator-configured system prompt from `gateway.system_prompt`.
pub fn assemble_system_prompt(
    selected_commands: &[CommandStub],
    agent_system_prompt: &str,
) -> String {
    // Build TOON command stubs section
    let rows: Vec<Vec<String>> = selected_commands
        .iter()
        .map(|cmd| vec![cmd.name.clone(), cmd.description.clone()])
        .collect();

    let table = serialize_table("commands", &["name", "description"], &rows);
    let command_stubs_toon = fenced_toon(&table);

    // Replace the placeholder in the foundational prompt
    let foundational =
        FOUNDATIONAL_PROMPT_TEMPLATE.replace("{command_stubs_toon}", &command_stubs_toon);

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
/// commands set and treats them as prose (spec Section 6.4).
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
            "Search the web for current information",
        )];
        let agent_prompt = "You are a helpful assistant.";
        let prompt = assemble_system_prompt(&stubs, agent_prompt);

        // Contains foundational preamble
        assert!(prompt.contains("You have access to commands"));
        assert!(prompt.contains("/command_name key: value"));
        assert!(prompt.contains("--describe"));

        // Contains fenced TOON block
        assert!(prompt.contains("```toon"));
        assert!(prompt.contains("web_search"));

        // Contains agent prompt after foundational
        assert!(prompt.contains("You are a helpful assistant."));

        // Order: foundational before agent
        let foundational_pos = prompt.find("You have access to commands").unwrap();
        let agent_pos = prompt.find("You are a helpful assistant.").unwrap();
        assert!(
            foundational_pos < agent_pos,
            "foundational must come before agent prompt"
        );
    }

    #[test]
    fn test_system_prompt_toon_format_for_command_stubs() {
        let stubs = vec![
            make_stub("web_search", "Search the web for current information"),
            make_stub("code_review", "Review code for issues and improvements"),
        ];
        let prompt = assemble_system_prompt(&stubs, "");

        // TOON array format: label[N]{fields}:
        assert!(prompt.contains("commands[2]{name, description}:"));
        assert!(prompt.contains("web_search"));
        assert!(prompt.contains("code_review"));
        // Wrapped in fenced block
        assert!(prompt.contains("```toon"));
    }

    #[test]
    fn test_system_prompt_no_commands_shows_empty_toon() {
        let prompt = assemble_system_prompt(&[], "Do something.");

        // Empty table still has the array header
        assert!(prompt.contains("commands[0]{name, description}:"));
        // Still in a fenced block
        assert!(prompt.contains("```toon"));
    }

    #[test]
    fn test_system_prompt_empty_agent_prompt() {
        let stubs = vec![make_stub("recall", "Retrieve memory")];
        let prompt = assemble_system_prompt(&stubs, "");

        // Should still produce valid prompt with foundational content
        assert!(prompt.contains("You have access to commands"));
        assert!(!prompt.ends_with("\n\n")); // No double-newline at end
    }

    #[test]
    fn test_system_prompt_contains_toon_argument_instructions() {
        let prompt = assemble_system_prompt(&[], "");

        // The foundational prompt must teach TOON argument syntax
        assert!(prompt.contains("key: value, key2:"));
        assert!(prompt.contains("Numbers and booleans are unquoted"));
        assert!(prompt.contains("Arrays use brackets"));
    }

    #[test]
    fn test_system_prompt_contains_rules_section() {
        let prompt = assemble_system_prompt(&[], "");

        // Rules section must be present
        assert!(prompt.contains("Rules:"));
        assert!(prompt.contains("Only use commands listed above"));
        assert!(prompt.contains("Arguments use TOON format"));
        assert!(prompt.contains("Text outside of slash commands is your response to the user"));
    }

    #[test]
    fn test_system_prompt_fenced_block_format() {
        let stubs = vec![make_stub("web_search", "Search the web")];
        let prompt = assemble_system_prompt(&stubs, "");
        // Fenced block must open and close correctly
        assert!(prompt.contains("```toon\n"));
        assert!(prompt.contains("\n```"));
    }

    #[test]
    fn test_system_prompt_command_rows_indented() {
        let stubs = vec![make_stub("web_search", "Search the web")];
        let prompt = assemble_system_prompt(&stubs, "");
        // Rows in TOON array syntax are 2-space indented
        assert!(prompt.contains("  web_search, Search the web"));
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
}
