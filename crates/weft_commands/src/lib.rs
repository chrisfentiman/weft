//! `weft_commands` — Command registry trait, parser, and adapter.
//!
//! Contains:
//! - `CommandRegistry` trait for managing available commands
//! - `CommandError` error type
//! - `parser`: slash command parser for LLM output
//! - `adapter`: `ToolRegistryCommandAdapter` bridging `ToolRegistryClient` to `CommandRegistry`
//!
//! Tool client types have moved to `weft_tools`.
//! Memory store types and clients have moved to `weft_memory`.

pub mod adapter;
pub mod parser;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use adapter::ToolRegistryCommandAdapter;

// Re-export trait, error, and parser from weft_commands_trait so existing import paths work.
pub use weft_commands_trait::{CommandError, CommandRegistry, ParsedResponse, parse_response};

// Re-export tool types from weft_tools for consumers that import via weft_commands.
pub use weft_tools::{
    GrpcToolRegistryClient, ToolDescription, ToolExecutionResult, ToolInfo, ToolRegistryClient,
    ToolRegistryError,
};

// Re-export memory types from weft_memory for backward compatibility.
pub use weft_memory::{
    GrpcMemoryStoreClient, MemoryEntry, MemoryQueryResult, MemoryStoreClient, MemoryStoreError,
    MemoryStoreMux, MemoryStoreResult,
};
