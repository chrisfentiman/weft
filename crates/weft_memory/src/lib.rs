//! `weft_memory` — Memory store client, multiplexer, domain types, and service trait.
//!
//! Contains:
//! - `MemoryService` trait — pure domain abstraction over memory operations
//! - `DefaultMemoryService` — implementation backed by `MemoryStoreMux`
//! - `NullMemoryService` — no-op implementation for testing
//! - `MemoryStoreClient` trait and `GrpcMemoryStoreClient` gRPC implementation
//! - `MemoryStoreError` error type
//! - `MemoryStoreMux` — routes memory operations to N named stores
//! - Domain types: `MemoryEntry`, `MemoryQueryResult`, `MemoryStoreResult`, `StoreInfo`
//! - TOON formatting helpers: `format_memory_query_results`, `format_memory_store_results`

pub mod client;
pub mod default_service;
pub mod mux;
pub mod null_service;
pub mod service;
pub mod types;

pub use client::{GrpcMemoryStoreClient, MemoryStoreClient, MemoryStoreError};
pub use default_service::{
    DefaultMemoryService, format_memory_query_results, format_memory_store_results,
};
pub use mux::MemoryStoreMux;
pub use null_service::NullMemoryService;
pub use service::{MemoryService, StoreInfo};
pub use types::{MemoryEntry, MemoryQueryResult, MemoryStoreResult};
