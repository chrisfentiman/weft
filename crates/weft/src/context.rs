//! Context window assembly for the gateway engine.
//!
//! Assembles the system prompt from three parts (in order):
//! 1. Weft foundational prompt (slash command format tutorial)
//! 2. Command stubs section (filtered commands in TOON format)
//! 3. Agent system prompt (from gateway config)

use weft_core::{CommandStub, toon::serialize_table};

/// The foundational system prompt that teaches the LLM the slash command format.
///
/// `{command_stubs_toon}` is a placeholder replaced at runtime with the TOON
/// table of semantically-selected command stubs.
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

/// Assemble the full system prompt.
///
/// Order: foundational prompt (with command stubs TOON) + agent system prompt from config.
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

    let command_stubs_toon =
        serialize_table("Available commands:", &["name", "description"], &rows);

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

/// Format command results as TOON for injection back into the conversation.
///
/// Produces a user-role message containing the results of all executed commands.
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

    serialize_table("[Command Results]", &["command", "status", "output"], &rows)
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

        // Contains TOON command stubs
        assert!(prompt.contains("Available commands:"));
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

        // TOON table format: label, headers, rows
        assert!(prompt.contains("Available commands:"));
        assert!(prompt.contains("name, description"));
        assert!(prompt.contains("web_search"));
        assert!(prompt.contains("code_review"));
    }

    #[test]
    fn test_system_prompt_no_commands_shows_empty_toon() {
        let prompt = assemble_system_prompt(&[], "Do something.");

        // Even with no commands, the table headers should appear (TOON format)
        assert!(prompt.contains("Available commands:"));
        assert!(prompt.contains("name, description"));
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

        assert!(toon.contains("[Command Results]"));
        assert!(toon.contains("command, status, output"));
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
    }

    #[test]
    fn test_format_results_empty() {
        let toon = format_command_results_toon(&[]);
        // Should still produce the TOON headers
        assert!(toon.contains("[Command Results]"));
        assert!(toon.contains("command, status, output"));
    }
}
