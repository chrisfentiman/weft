//! Gateway engine orchestration layer.
//!
//! The engine routes requests semantically, assembles context, calls the LLM,
//! parses commands from the response, executes them, and loops until no more
//! commands are emitted or the iteration cap or timeout is reached.
//!
//! ## Request flow (with hooks)
//!
//! 1. [HOOK: RequestStart] — hard block (403), can modify request
//! 2. Validate request
//! 3. list_commands()
//! 4. extract_latest_user_message()
//! 5. For each routing domain (model, commands, tool_necessity):
//!    a. [HOOK: PreRoute(domain, trigger=request_start)]  — hard block (403)
//!    b. Router scores this domain
//!    c. [HOOK: PostRoute(domain, trigger=request_start)] — hard block (403)
//! 6. Assemble system prompt
//! 7. LOOP:
//!    a. Call LLM
//!    b. Parse response
//!    c. [HOOK: PreResponse] — feedback block (re-run LLM with reason), can modify;
//!    block with retries left injects reason and retries; exhausted retries returns 422.
//!    d. For each command invocation:
//!       - [HOOK: PreToolUse] — feedback block
//!       - For /recall and /remember: [HOOK: PreRoute/PostRoute(memory)] — feedback block
//!       - Execute command
//!       - [HOOK: PostToolUse] — can modify result
//! 8. Build HTTP response, return
//! 9. [HOOK: RequestEnd] — fire-and-forget (semaphore-gated)
//!
//! ## Activity events
//!
//! When `options.activity = true`, the engine collects routing decisions and hook
//! events as `ActivityEvent` values during processing and includes them as
//! `source: gateway` system messages in the response. See `ActivityEvent` and
//! `assemble_response`.

use std::sync::Arc;

use weft_commands::CommandRegistry;
use weft_core::WeftConfig;
use weft_hooks::HookRunner;
use weft_llm::ProviderService;
use weft_memory::MemoryService;
use weft_router::{MemoryStoreRef, RoutingCandidate, SemanticRouter, build_memory_candidates};

pub mod context;

mod activity;
mod hooks;
mod memory;
mod provider;
mod request;
mod routing;
pub(crate) mod util;

#[cfg(test)]
mod test_support;

/// The gateway engine: holds shared components and drives the request loop.
///
/// Generic parameters:
/// - `H`: hook runner (e.g. `weft_hooks::HookRegistry` or `NullHookRunner`)
/// - `R`: semantic router (e.g. `weft_router::ModernBertClassifier`)
/// - `M`: memory service (e.g. `weft_memory::DefaultMemoryService`)
/// - `P`: provider service (e.g. `weft_llm::ProviderRegistry`)
/// - `C`: command registry (e.g. `weft_commands::ToolRegistryCommandAdapter`)
///
/// All fields are `Arc` so `GatewayEngine` is cheaply cloneable — axum clones
/// it into each request handler. A manual `Clone` impl avoids requiring the
/// type params themselves to be `Clone` (they are behind `Arc`).
pub struct GatewayEngine<H, R, M, P, C> {
    config: Arc<WeftConfig>,
    providers: Arc<P>,
    router: Arc<R>,
    commands: Arc<C>,
    /// Optional memory service. `None` when no memory stores are configured.
    memory: Option<Arc<M>>,
    /// All memory store routing candidates (from `memory.stores()`). Used for the Memory
    /// domain in `route_all_domains()` and for per-invocation routing by both `/recall`
    /// and `/remember`. Empty if no memory stores are configured.
    memory_candidates: Vec<RoutingCandidate>,
    /// Memory candidates filtered to read-capable stores only. Used by `/recall`
    /// for per-invocation routing via `score_memory_candidates()`.
    read_memory_candidates: Vec<RoutingCandidate>,
    /// Memory candidates filtered to write-capable stores only. Used by `/remember`
    /// for per-invocation routing via `score_memory_candidates()`.
    write_memory_candidates: Vec<RoutingCandidate>,
    /// Hook runner. Shared immutably across all request handlers.
    hooks: Arc<H>,
    /// Semaphore limiting concurrent RequestEnd hook tasks.
    /// Prevents unbounded task accumulation under burst load.
    request_end_semaphore: Arc<tokio::sync::Semaphore>,
}

impl<H, R, M, P, C> Clone for GatewayEngine<H, R, M, P, C> {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            providers: Arc::clone(&self.providers),
            router: Arc::clone(&self.router),
            commands: Arc::clone(&self.commands),
            memory: self.memory.clone(),
            memory_candidates: self.memory_candidates.clone(),
            read_memory_candidates: self.read_memory_candidates.clone(),
            write_memory_candidates: self.write_memory_candidates.clone(),
            hooks: Arc::clone(&self.hooks),
            request_end_semaphore: Arc::clone(&self.request_end_semaphore),
        }
    }
}

impl<H, R, M, P, C> GatewayEngine<H, R, M, P, C>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    /// Expose the config for use by the health handler and other modules.
    pub fn config(&self) -> &WeftConfig {
        &self.config
    }

    /// Construct a new gateway engine.
    ///
    /// `memory`: Optional memory service. `None` when no memory stores configured.
    pub fn new(
        config: Arc<WeftConfig>,
        providers: Arc<P>,
        router: Arc<R>,
        commands: Arc<C>,
        memory: Option<Arc<M>>,
        hooks: Arc<H>,
    ) -> Self {
        // Build per-capability candidate sets from the memory service's store metadata.
        // Convert `StoreInfo` to `MemoryStoreRef` so `weft_router` doesn't depend on `weft_memory`.
        let mem_candidates = if let Some(mem) = &memory {
            let refs: Vec<MemoryStoreRef> = mem
                .stores()
                .into_iter()
                .map(|s| MemoryStoreRef {
                    name: s.name,
                    capabilities: s.capabilities,
                    examples: s.examples,
                })
                .collect();
            build_memory_candidates(&refs)
        } else {
            weft_router::MemoryCandidates::default()
        };
        let (memory_candidates, read_memory_candidates, write_memory_candidates) = (
            mem_candidates.all,
            mem_candidates.read,
            mem_candidates.write,
        );

        let request_end_concurrency = config.request_end_concurrency;
        let request_end_semaphore = Arc::new(tokio::sync::Semaphore::new(request_end_concurrency));

        Self {
            config,
            providers,
            router,
            commands,
            memory,
            memory_candidates,
            read_memory_candidates,
            write_memory_candidates,
            hooks,
            request_end_semaphore,
        }
    }
}

// `tool_necessity_candidates` has moved to `weft_router` — re-export from there.
pub use weft_router::tool_necessity_candidates;
