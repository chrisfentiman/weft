//! Command types: stubs, descriptions, invocations, and results.

/// Lightweight command representation injected into context.
/// Name + one-liner. Not full schema.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandStub {
    /// Command name, e.g. "web_search"
    pub name: String,
    /// One-line description, e.g. "Search the web for information"
    pub description: String,
}

/// Full command description returned by progressive disclosure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandDescription {
    pub name: String,
    pub description: String,
    /// Detailed usage instructions and parameter descriptions.
    /// Usage examples show TOON argument syntax (e.g., `/cmd key: value, key2: 10`).
    /// The gateway formats this from the tool registry's raw usage string,
    /// converting any JSON examples to TOON syntax before injection.
    pub usage: String,
    /// JSON schema for the command's arguments, if structured.
    /// None for commands that take free-form text input.
    /// Note: this schema is used internally for validation. It is NOT
    /// injected into the context window directly (the usage field provides
    /// human-readable parameter descriptions instead).
    pub parameters_schema: Option<serde_json::Value>,
}

/// What the LLM wants to do with a command.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CommandAction {
    /// Execute the command with arguments.
    Execute,
    /// Request the full description (progressive disclosure).
    /// Triggered by `--describe` or `--help` flag.
    Describe,
}

/// A parsed command invocation from LLM output.
/// Produced by the parser when it finds `/command_name` in the response.
///
/// The LLM writes arguments in TOON format (`key: value, key2: value2`).
/// The parser converts TOON arguments to `serde_json::Value` at parse time
/// so the rest of the system (command registry, gRPC client) works with JSON.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandInvocation {
    pub name: String,
    /// What action to take: execute or describe.
    pub action: CommandAction,
    /// Parsed arguments as JSON Value (converted from TOON at parse time).
    /// For Execute: a JSON object (or empty object for no-arg commands).
    /// For Describe: Value::Null (arguments are ignored).
    pub arguments: serde_json::Value,
}

/// Result of executing a command.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommandResult {
    pub command_name: String,
    /// Whether the command succeeded.
    pub success: bool,
    /// The result content to inject back into the conversation.
    pub output: String,
    /// Optional error message if success is false.
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_stub_construction() {
        let stub = CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        };
        assert_eq!(stub.name, "web_search");
        assert_eq!(stub.description, "Search the web");
    }

    #[test]
    fn test_command_stub_serde_round_trip() {
        let stub = CommandStub {
            name: "recall".to_string(),
            description: "Retrieve from memory".to_string(),
        };
        let json = serde_json::to_string(&stub).unwrap();
        let back: CommandStub = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, stub.name);
        assert_eq!(back.description, stub.description);
    }

    #[test]
    fn test_command_description_with_schema() {
        let schema =
            serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}});
        let desc = CommandDescription {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            usage: "/web_search query: \"search terms\"".to_string(),
            parameters_schema: Some(schema.clone()),
        };
        assert_eq!(desc.parameters_schema, Some(schema));
    }

    #[test]
    fn test_command_description_without_schema() {
        let desc = CommandDescription {
            name: "recall".to_string(),
            description: "Retrieve from memory".to_string(),
            usage: "/recall".to_string(),
            parameters_schema: None,
        };
        assert!(desc.parameters_schema.is_none());
    }

    #[test]
    fn test_command_action_equality() {
        assert_eq!(CommandAction::Execute, CommandAction::Execute);
        assert_eq!(CommandAction::Describe, CommandAction::Describe);
        assert_ne!(CommandAction::Execute, CommandAction::Describe);
    }

    #[test]
    fn test_command_invocation_execute() {
        let inv = CommandInvocation {
            name: "web_search".to_string(),
            action: CommandAction::Execute,
            arguments: serde_json::json!({"query": "Rust async"}),
        };
        assert_eq!(inv.action, CommandAction::Execute);
        assert!(inv.arguments.is_object());
    }

    #[test]
    fn test_command_invocation_describe() {
        let inv = CommandInvocation {
            name: "web_search".to_string(),
            action: CommandAction::Describe,
            arguments: serde_json::Value::Null,
        };
        assert_eq!(inv.action, CommandAction::Describe);
        assert!(inv.arguments.is_null());
    }

    #[test]
    fn test_command_result_success() {
        let result = CommandResult {
            command_name: "web_search".to_string(),
            success: true,
            output: "Found results".to_string(),
            error: None,
        };
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_command_result_failure() {
        let result = CommandResult {
            command_name: "web_search".to_string(),
            success: false,
            output: String::new(),
            error: Some("Connection timeout".to_string()),
        };
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("Connection timeout"));
    }

    #[test]
    fn test_command_result_serde_round_trip_success() {
        let result = CommandResult {
            command_name: "web_search".to_string(),
            success: true,
            output: "Found 10 results".to_string(),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CommandResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.command_name, result.command_name);
        assert_eq!(back.success, result.success);
        assert_eq!(back.output, result.output);
        assert_eq!(back.error, result.error);
    }

    #[test]
    fn test_command_result_serde_round_trip_failure() {
        let result = CommandResult {
            command_name: "web_search".to_string(),
            success: false,
            output: String::new(),
            error: Some("rate limit exceeded".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CommandResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.command_name, result.command_name);
        assert!(!back.success);
        assert_eq!(back.error.as_deref(), Some("rate limit exceeded"));
    }

    #[test]
    fn test_command_invocation_serde_round_trip_execute() {
        let inv = CommandInvocation {
            name: "web_search".to_string(),
            action: CommandAction::Execute,
            arguments: serde_json::json!({"query": "Rust async", "limit": 10}),
        };
        let json = serde_json::to_string(&inv).unwrap();
        let back: CommandInvocation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, inv.name);
        assert_eq!(back.action, CommandAction::Execute);
        assert_eq!(back.arguments["query"], "Rust async");
        assert_eq!(back.arguments["limit"], 10);
    }

    #[test]
    fn test_command_invocation_serde_round_trip_describe() {
        let inv = CommandInvocation {
            name: "recall".to_string(),
            action: CommandAction::Describe,
            arguments: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&inv).unwrap();
        let back: CommandInvocation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "recall");
        assert_eq!(back.action, CommandAction::Describe);
        assert!(back.arguments.is_null());
    }
}
