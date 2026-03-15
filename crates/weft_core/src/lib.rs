//! `weft_core` — Shared primitives for the Weft gateway.
//!
//! This crate contains types that cross domain boundaries: message types,
//! command types, configuration, errors, and the TOON serializer/parser.
//! All domain crates depend on this crate. It has no dependencies on domain crates.

pub mod command;
pub mod config;
pub mod error;
pub mod message;
pub mod toon;

// Re-export everything at the crate root for convenience.
pub use command::{
    CommandAction, CommandDescription, CommandInvocation, CommandResult, CommandStub,
};
pub use config::{
    ClassifierConfig, GatewayConfig, LlmConfig, LlmProviderKind, ServerConfig, ToolRegistryConfig,
    WeftConfig,
};
pub use error::WeftError;
pub use message::{ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Role, Usage};
