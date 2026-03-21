//! Slash command parser: re-exported from `weft_commands_trait`.
//!
//! The implementation lives in `weft_commands_trait` so that consumers
//! like `weft_activities` can use it without depending on `weft_commands`.
//! This module re-exports for backward compatibility.

pub use weft_commands_trait::{ParsedResponse, parse_response};
