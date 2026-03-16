//! Gateway engine orchestration layer.

pub mod context;
pub mod engine;

pub use engine::{GatewayEngine, tool_necessity_candidates};
