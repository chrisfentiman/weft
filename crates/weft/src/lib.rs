//! Weft gateway library root.
//!
//! Re-exports the public API needed by integration tests and external consumers.
//! The binary entry point (`main.rs`) imports from here.

pub mod event_log;
pub mod grpc;
pub mod server;
pub mod types;

pub use grpc::WeftService;
pub use server::{build_router, serve};
