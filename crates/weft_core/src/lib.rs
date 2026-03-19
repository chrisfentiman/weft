//! `weft_core` — Shared primitives for the Weft gateway.
//!
//! This crate contains types that cross domain boundaries: message types,
//! command types, configuration, errors, and the TOON serializer/parser.
//! All domain crates depend on this crate. It has no dependencies on domain crates.

pub mod command;
pub mod config;
pub mod error;
pub mod message;
pub mod routing;
pub mod toon;
pub mod wire;

// Re-export everything at the crate root for convenience.
pub use command::{
    CommandAction, CommandDescription, CommandInvocation, CommandResult, CommandStub,
};
pub use config::{
    ClassifierConfig, DomainConfig, DomainsConfig, EventLogConfig, GatewayConfig, HookConfig,
    HookEvent, HookRoutingDomain, HookType, MemoryConfig, MemoryStoreConfig, ModelEntry,
    ProviderConfig, ResolvedModel, RouterConfig, RoutingTrigger, ServerConfig, StoreCapability,
    ToolRegistryConfig, WeftConfig, WireFormat,
};
pub use error::WeftError;
pub use message::{
    CommandCallContent, CommandResultContent, ContentPart, CouncilStartActivity, DocumentContent,
    HookActivity, MediaContent, MediaSource, MemoryResultContent, MemoryResultEntry,
    MemoryStoredContent, Role, RoutingActivity, Source, WeftMessage,
};
pub use routing::{ModelInfo, ModelRoutingInstruction, RoutingMode};
pub use wire::{
    ConversionError, ResponseFormat, SamplingOptions, SourceValidationError, WeftRequest,
    WeftResponse, WeftTiming, WeftUsage, validate_message_source, validate_request,
};
