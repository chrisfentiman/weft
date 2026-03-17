//! Weft gateway binary entry point.
//!
//! Loads configuration, constructs concrete implementations of all components,
//! wires them into the GatewayEngine, and starts the axum HTTP server.

mod engine;
mod grpc;
mod server;
mod types;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn};
use weft_commands::ToolRegistryCommandAdapter;
use weft_tools::GrpcToolRegistryClient;
use weft_core::{WeftConfig, WireFormat};
use weft_llm::{AnthropicProvider, Capability, OpenAIProvider, ProviderRegistry, RhaiProvider};
use weft_memory::{DefaultMemoryService, GrpcMemoryStoreClient, MemoryStoreMux, StoreInfo};
use weft_router::{ModernBertRouter, RoutingCandidate, RoutingDomainKind};

use crate::engine::tool_necessity_candidates;
use crate::grpc::WeftService;
use crate::server::build_router;
use crate::types::{BinaryCommandRegistry, WeftEngine};

/// Weft — AI orchestration gateway
#[derive(Debug, Parser)]
#[command(name = "weft", version, about = "weft - AI orchestration gateway")]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(
        short = 'c',
        long,
        default_value = "config/weft.toml",
        value_name = "PATH"
    )]
    config: PathBuf,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize tracing from RUST_LOG environment variable.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Load and resolve configuration.
    let config_str = std::fs::read_to_string(&cli.config).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to read config file '{}': {e}",
            cli.config.display()
        );
        std::process::exit(1);
    });

    let mut config: WeftConfig = toml::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to parse config file '{}': {e}",
            cli.config.display()
        );
        std::process::exit(1);
    });

    config.resolve().unwrap_or_else(|e| {
        eprintln!("error: configuration error: {e}");
        std::process::exit(1);
    });

    config.validate().unwrap_or_else(|e| {
        eprintln!("error: configuration validation error: {e}");
        std::process::exit(1);
    });

    info!(path = %cli.config.display(), "configuration loaded");

    // ── Startup log: providers and models (spec Section 12) ───────────────

    let provider_count = config.router.providers.len();
    let model_count: usize = config.router.providers.iter().map(|p| p.models.len()).sum();
    let default_model = config.router.effective_default_model().to_string();

    // Find the provider name for the default model.
    let default_provider_name = config
        .router
        .providers
        .iter()
        .find(|p| p.models.iter().any(|m| m.name == default_model))
        .map(|p| p.name.as_str())
        .unwrap_or("unknown");

    info!(
        providers = provider_count,
        models = model_count,
        default_model = %default_model,
        default_provider = %default_provider_name,
        "router configured"
    );

    // Log which routing domains are enabled.
    let tool_skipping = config.router.skip_tools_when_unnecessary;
    let model_domain_threshold = config
        .router
        .domains
        .model
        .as_ref()
        .and_then(|d| d.threshold);
    let tool_domain_enabled = config
        .router
        .domains
        .tool_necessity
        .as_ref()
        .map(|d| d.enabled)
        .unwrap_or(true);

    let memory_domain_active = config
        .memory
        .as_ref()
        .map(|m| !m.stores.is_empty())
        .unwrap_or(false);

    info!(
        commands_domain = true,
        model_domain = (model_count > 1),
        tool_necessity_domain = (tool_skipping && tool_domain_enabled),
        memory_domain = memory_domain_active,
        model_domain_threshold = ?model_domain_threshold,
        tool_skipping_active = tool_skipping,
        "routing domains configured"
    );

    // Warn about models with fewer than 3 examples (spec Section 11.9).
    for provider in &config.router.providers {
        for model in &provider.models {
            if model.examples.len() < 3 {
                warn!(
                    model = %model.name,
                    provider = %provider.name,
                    examples = model.examples.len(),
                    "model has fewer than 3 examples — centroid quality may be poor (minimum 3 recommended)"
                );
            }
            info!(
                model = %model.name,
                provider = %provider.name,
                examples = model.examples.len(),
                "model registered"
            );
        }
    }

    let config = Arc::new(config);

    // ── Build ProviderRegistry ─────────────────────────────────────────────
    //
    // One `Provider` instance per `ProviderConfig`. Multiple model entries
    // under the same provider share the same `Arc<dyn Provider>`.
    //
    // Build a map from provider name -> Arc<dyn Provider> first, then
    // iterate resolved models to build the registry maps.

    let resolved_models = config.router.resolve_models();

    // One provider instance per unique provider name.
    let mut provider_instances: HashMap<String, Arc<dyn weft_llm::Provider>> = HashMap::new();
    for provider_config in &config.router.providers {
        let instance: Arc<dyn weft_llm::Provider> = match &provider_config.wire_format {
            WireFormat::Anthropic => Arc::new(AnthropicProvider::new(
                provider_config.api_key.clone(),
                provider_config.base_url.clone(),
            )),
            WireFormat::OpenAI => Arc::new(OpenAIProvider::new(
                provider_config.api_key.clone(),
                provider_config.base_url.clone(),
            )),
            WireFormat::Custom => {
                // wire_script presence validated by config.validate() above.
                // base_url is required for custom providers (no sensible default).
                let script_path = provider_config.wire_script.as_deref().unwrap_or("");
                let base_url = provider_config.base_url.clone().unwrap_or_default();
                match RhaiProvider::new(
                    script_path,
                    provider_config.api_key.clone(),
                    base_url,
                    provider_config.name.clone(),
                ) {
                    Ok(rhai_provider) => {
                        info!(
                            provider = %provider_config.name,
                            script = %script_path,
                            "custom wire format provider initialized"
                        );
                        Arc::new(rhai_provider)
                    }
                    Err(e) => {
                        eprintln!(
                            "error: provider '{}': failed to initialize custom wire format provider: {e}",
                            provider_config.name
                        );
                        std::process::exit(1);
                    }
                }
            }
        };
        provider_instances.insert(provider_config.name.clone(), instance);
    }

    // Build registry maps: model routing name -> provider/model_id/max_tokens/capabilities.
    let mut registry_providers: HashMap<String, Arc<dyn weft_llm::Provider>> = HashMap::new();
    let mut registry_model_ids: HashMap<String, String> = HashMap::new();
    let mut registry_max_tokens: HashMap<String, u32> = HashMap::new();
    let mut registry_capabilities: HashMap<String, HashSet<Capability>> = HashMap::new();

    for resolved in &resolved_models {
        let provider_arc = provider_instances
            .get(&resolved.provider_name)
            .expect("provider instance must exist for resolved model")
            .clone();
        registry_providers.insert(resolved.name.clone(), provider_arc);
        registry_model_ids.insert(resolved.name.clone(), resolved.model.clone());
        registry_max_tokens.insert(resolved.name.clone(), resolved.max_tokens);
        // Convert Vec<String> capabilities to HashSet<Capability> newtypes.
        let caps: HashSet<Capability> = resolved
            .capabilities
            .iter()
            .map(|s| Capability::new(s.clone()))
            .collect();
        registry_capabilities.insert(resolved.name.clone(), caps);
    }

    let provider_registry = Arc::new(ProviderRegistry::new(
        registry_providers,
        registry_model_ids,
        registry_max_tokens,
        registry_capabilities,
        default_model.clone(),
    ));

    // ── Build pre-embed candidates for the router ──────────────────────────
    //
    // Pre-embed model, tool-necessity, and memory candidates at startup.
    // Command candidates are embedded lazily (they come from a remote registry).

    let model_candidates: Vec<RoutingCandidate> = resolved_models
        .iter()
        .map(|m| RoutingCandidate {
            id: m.name.clone(),
            examples: m.examples.clone(),
        })
        .collect();

    let tool_candidates = tool_necessity_candidates();

    // Build memory candidates from config for pre-embedding and per-invocation routing.
    let memory_candidates: Vec<RoutingCandidate> = config
        .memory
        .as_ref()
        .map(|mc| {
            mc.stores
                .iter()
                .map(|store| RoutingCandidate {
                    id: store.name.clone(),
                    examples: store.examples.clone(),
                })
                .collect()
        })
        .unwrap_or_default();

    let mut pre_embed: Vec<(RoutingDomainKind, Vec<RoutingCandidate>)> = Vec::new();

    // Only pre-embed model candidates if there are multiple models to route between.
    if model_candidates.len() > 1 {
        pre_embed.push((RoutingDomainKind::Model, model_candidates));
    }

    // Pre-embed tool-necessity candidates if tool-skipping is enabled.
    if config.router.skip_tools_when_unnecessary {
        pre_embed.push((RoutingDomainKind::ToolNecessity, tool_candidates));
    }

    // Pre-embed memory candidates if any stores are configured.
    if !memory_candidates.is_empty() {
        pre_embed.push((RoutingDomainKind::Memory, memory_candidates));
    }

    // ── Build domain thresholds from config ───────────────────────────────

    let mut domain_thresholds: HashMap<RoutingDomainKind, f32> = HashMap::new();

    if let Some(model_domain) = &config.router.domains.model
        && let Some(t) = model_domain.threshold
    {
        domain_thresholds.insert(RoutingDomainKind::Model, t);
    }
    if let Some(tool_domain) = &config.router.domains.tool_necessity
        && let Some(t) = tool_domain.threshold
    {
        domain_thresholds.insert(RoutingDomainKind::ToolNecessity, t);
    }
    if let Some(mem_domain) = &config.router.domains.memory
        && let Some(t) = mem_domain.threshold
    {
        domain_thresholds.insert(RoutingDomainKind::Memory, t);
    }

    // ── Construct semantic router ──────────────────────────────────────────
    //
    // `ModernBertRouter::new` is infallible — falls back to passthrough mode
    // if the model or tokenizer can't be loaded, logging a warning internally.

    let router = {
        let r = ModernBertRouter::new(
            &config.router.classifier.model_path,
            &config.router.classifier.tokenizer_path,
            &pre_embed,
            domain_thresholds,
        )
        .await;
        info!(
            model_path = %config.router.classifier.model_path,
            threshold = config.router.classifier.threshold,
            max_commands = config.router.classifier.max_commands,
            "semantic router initialized"
        );
        Arc::new(r)
    };

    // ── Construct command registry ─────────────────────────────────────────

    let command_registry: Arc<BinaryCommandRegistry> =
        if let Some(tr_config) = &config.tool_registry {
            info!(
                endpoint = %tr_config.endpoint,
                "tool registry configured"
            );
            let grpc_client = GrpcToolRegistryClient::new(
                tr_config.endpoint.clone(),
                tr_config.connect_timeout_ms,
                tr_config.request_timeout_ms,
            );
            Arc::new(BinaryCommandRegistry::Tool(
                ToolRegistryCommandAdapter::new(Arc::new(grpc_client)),
            ))
        } else {
            info!("no tool registry configured, using empty registry");
            Arc::new(BinaryCommandRegistry::Empty)
        };

    // ── Build memory service (optional) ────────────────────────────────────
    //
    // Builds one GrpcMemoryStoreClient per configured store, wraps them in a
    // MemoryStoreMux, then constructs a DefaultMemoryService with the mux and
    // store metadata (name, capabilities, examples) for routing candidate construction.
    // Uses lazy gRPC connections — failures appear on first /recall or /remember.

    let memory_service: Option<Arc<DefaultMemoryService>> = if let Some(mem_config) = &config.memory
    {
        if mem_config.stores.is_empty() {
            None
        } else {
            let mut stores: HashMap<String, Arc<dyn weft_memory::MemoryStoreClient>> =
                HashMap::new();
            let mut max_results_map: HashMap<String, u32> = HashMap::new();
            let mut readable: HashSet<String> = HashSet::new();
            let mut writable: HashSet<String> = HashSet::new();
            let mut store_infos: Vec<StoreInfo> = Vec::new();

            for store_cfg in &mem_config.stores {
                let client = GrpcMemoryStoreClient::new(
                    store_cfg.endpoint.clone(),
                    store_cfg.connect_timeout_ms,
                    store_cfg.request_timeout_ms,
                );
                stores.insert(store_cfg.name.clone(), Arc::new(client));
                max_results_map.insert(store_cfg.name.clone(), store_cfg.max_results);

                let mut caps: Vec<String> = Vec::new();
                if store_cfg.can_read() {
                    readable.insert(store_cfg.name.clone());
                    caps.push("read".to_string());
                }
                if store_cfg.can_write() {
                    writable.insert(store_cfg.name.clone());
                    caps.push("write".to_string());
                }

                store_infos.push(StoreInfo {
                    name: store_cfg.name.clone(),
                    capabilities: caps.clone(),
                    examples: store_cfg.examples.clone(),
                });

                info!(
                    store = %store_cfg.name,
                    endpoint = %store_cfg.endpoint,
                    capabilities = %caps.join(","),
                    examples = store_cfg.examples.len(),
                    "memory store registered"
                );
            }

            let mux =
                MemoryStoreMux::new(stores, max_results_map, readable.clone(), writable.clone());

            info!(
                memory_stores = mem_config.stores.len(),
                readable = readable.len(),
                writable = writable.len(),
                "memory configured"
            );

            Some(Arc::new(DefaultMemoryService::new(
                Arc::new(mux),
                store_infos,
            )))
        }
    } else {
        None
    };

    // ── Construct hook registry ────────────────────────────────────────────
    //
    // One shared reqwest::Client for all HTTP hooks (connection pooling).
    // HookRegistry::from_config validates all hook configs and compiles matchers.
    // Startup fails fast if any hook is misconfigured.

    let http_client = std::sync::Arc::new(reqwest::Client::new());

    let hook_registry = {
        let registry = weft_hooks::HookRegistry::from_config(&config.hooks, http_client)
            .unwrap_or_else(|e| {
                eprintln!("error: hook configuration error: {e}");
                std::process::exit(1);
            });
        info!(hooks = config.hooks.len(), "hook registry initialized");
        std::sync::Arc::new(registry)
    };

    // ── Wire the gateway engine ────────────────────────────────────────────

    let engine: WeftEngine = WeftEngine::new(
        Arc::clone(&config),
        provider_registry,
        router,
        command_registry,
        memory_service,
        hook_registry,
    );

    // ── Wire WeftService (shared between gRPC and HTTP) ────────────────────
    //
    // WeftService is the single code path to the engine. Both the tonic gRPC
    // server and the axum HTTP router hold an Arc<WeftService> and call
    // handle_weft_request() — no separate engine references in the HTTP handler.

    let weft_service = Arc::new(WeftService::new(engine));

    // ── Start the combined gRPC + HTTP server ──────────────────────────────

    let router = build_router(Arc::clone(&weft_service));
    let bind_address = &config.server.bind_address;

    if let Err(e) = server::serve(router, bind_address).await {
        eprintln!("error: server failed: {e}");
        std::process::exit(1);
    }
}
