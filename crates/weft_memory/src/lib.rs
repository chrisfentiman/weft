//! `weft_memory` — Memory store client, multiplexer, and domain types.
//!
//! Contains:
//! - `MemoryStoreClient` trait and `GrpcMemoryStoreClient` gRPC implementation
//! - `MemoryStoreError` error type
//! - `MemoryStoreMux` — routes memory operations to N named stores
//! - Domain types: `MemoryEntry`, `MemoryQueryResult`, `MemoryStoreResult`

pub mod client;
pub mod mux;
pub mod types;

pub use client::{GrpcMemoryStoreClient, MemoryStoreClient, MemoryStoreError};
pub use mux::MemoryStoreMux;
pub use types::{MemoryEntry, MemoryQueryResult, MemoryStoreResult};
