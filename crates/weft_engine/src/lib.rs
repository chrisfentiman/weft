//! Gateway engine orchestration layer.

pub mod context;
pub mod engine;

pub use engine::GatewayEngine;
// `tool_necessity_candidates` has moved to `weft_router` — re-export from there.
pub use weft_router::tool_necessity_candidates;
