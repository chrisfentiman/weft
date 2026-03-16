//! Re-export the gateway engine from the weft_engine crate.
//!
//! All engine logic now lives in `weft_engine`. This module exists so that
//! `crate::engine::GatewayEngine` (and related items) continue to resolve
//! correctly within the `weft` binary crate without changing every import site.

// GatewayEngine is used in test modules via crate::engine::GatewayEngine.
#[allow(unused_imports)]
pub use weft_engine::{GatewayEngine, tool_necessity_candidates};
