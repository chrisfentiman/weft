//! Routing domain types — re-exported from `weft_router_trait`.
//!
//! The canonical definitions live in `weft_router_trait`. This module re-exports
//! them so the existing `use weft_router::domain::*` paths keep working.

pub use weft_router_trait::{
    RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
};
