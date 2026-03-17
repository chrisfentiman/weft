//! Concrete engine and service type aliases for the weft binary.
//!
//! Defines the concrete generic instantiation of `GatewayEngine` and `WeftService`
//! used by the binary. Shared between `grpc.rs`, `server.rs`, and `main.rs` so each
//! module uses the same concrete type without repeating the type params.

use async_trait::async_trait;
use weft_commands::ToolRegistryCommandAdapter;
use weft_core::{CommandDescription, CommandInvocation, CommandResult, CommandStub};
use weft_engine::GatewayEngine;
use weft_hooks::HookRegistry;
use weft_llm::ProviderRegistry;
use weft_memory::DefaultMemoryService;
use weft_router::ModernBertRouter;

/// Unified command registry for the weft binary.
///
/// Dispatches to the live gRPC-backed tool registry or an empty no-op registry
/// when no `tool_registry` section is present in the TOML configuration.
pub enum BinaryCommandRegistry {
    /// gRPC-backed tool registry adapter.
    Tool(ToolRegistryCommandAdapter),
    /// No-op registry: returns empty command lists and `NotFound` errors.
    Empty,
}

#[async_trait]
impl weft_commands::CommandRegistry for BinaryCommandRegistry {
    async fn list_commands(&self) -> Result<Vec<CommandStub>, weft_commands::CommandError> {
        match self {
            Self::Tool(r) => r.list_commands().await,
            Self::Empty => Ok(vec![]),
        }
    }

    async fn describe_command(
        &self,
        name: &str,
    ) -> Result<CommandDescription, weft_commands::CommandError> {
        match self {
            Self::Tool(r) => r.describe_command(name).await,
            Self::Empty => Err(weft_commands::CommandError::NotFound(name.to_string())),
        }
    }

    async fn execute_command(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandResult, weft_commands::CommandError> {
        match self {
            Self::Tool(r) => r.execute_command(invocation).await,
            Self::Empty => Err(weft_commands::CommandError::NotFound(
                invocation.name.clone(),
            )),
        }
    }
}

/// Concrete engine type used by the weft binary.
///
/// - `H = HookRegistry`: HTTP hook runner
/// - `R = ModernBertRouter`: ModernBERT bi-encoder semantic classifier
/// - `M = DefaultMemoryService`: memory service backed by MemoryStoreMux
/// - `P = ProviderRegistry`: provider dispatch table
/// - `C = BinaryCommandRegistry`: tool/command dispatch (live gRPC or no-op)
pub type WeftEngine = GatewayEngine<
    HookRegistry,
    ModernBertRouter,
    DefaultMemoryService,
    ProviderRegistry,
    BinaryCommandRegistry,
>;
