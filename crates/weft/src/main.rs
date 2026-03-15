//! Weft gateway binary entry point.
//!
//! Loads configuration, constructs concrete implementations of all components,
//! wires them into the GatewayEngine, and starts the axum HTTP server.

mod context;
mod engine;
mod server;

use std::sync::Arc;

use clap::Parser;
use std::path::PathBuf;
use tracing::info;
use weft_classifier::ModernBertClassifier;
use weft_commands::{GrpcToolRegistryClient, ToolRegistryCommandAdapter};
use weft_core::{LlmProviderKind, WeftConfig};
use weft_llm::{AnthropicProvider, OpenAIProvider};

use crate::engine::GatewayEngine;
use crate::server::build_router;

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

    info!(path = %cli.config.display(), "configuration loaded");
    info!(
        provider = ?config.llm.provider,
        model = %config.llm.model,
        "LLM provider configured"
    );

    let config = Arc::new(config);

    // ── Construct LLM provider ─────────────────────────────────────────────

    let llm_provider: Arc<dyn weft_llm::LlmProvider> = match &config.llm.provider {
        LlmProviderKind::Anthropic => Arc::new(AnthropicProvider::new(&config.llm)),
        LlmProviderKind::OpenAI => Arc::new(OpenAIProvider::new(&config.llm)),
    };

    // ── Construct semantic classifier ──────────────────────────────────────
    //
    // `ModernBertClassifier::new` is infallible — it falls back to passthrough mode
    // if the model or tokenizer can't be loaded, logging a warning internally.
    // We use FallbackClassifier only when the model path doesn't exist at all,
    // to ensure the gateway starts cleanly even without model files.

    let classifier: Arc<dyn weft_classifier::SemanticClassifier> = {
        let c = ModernBertClassifier::new(
            &config.classifier.model_path,
            &config.classifier.tokenizer_path,
            &[], // No commands at startup — commands are fetched lazily per-request
        );
        info!(
            model_path = %config.classifier.model_path,
            "semantic classifier initialized"
        );
        Arc::new(c)
    };

    // ── Construct command registry ─────────────────────────────────────────

    let command_registry: Arc<dyn weft_commands::CommandRegistry> =
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
            Arc::new(ToolRegistryCommandAdapter::new(Arc::new(grpc_client)))
        } else {
            info!("no tool registry configured, using empty registry");
            Arc::new(EmptyCommandRegistry)
        };

    // ── Wire the gateway engine ────────────────────────────────────────────

    let engine = GatewayEngine::new(
        Arc::clone(&config),
        llm_provider,
        classifier,
        command_registry,
    );

    // ── Start the HTTP server ──────────────────────────────────────────────

    let router = build_router(engine);
    let bind_address = &config.server.bind_address;

    if let Err(e) = server::serve(router, bind_address).await {
        eprintln!("error: server failed: {e}");
        std::process::exit(1);
    }
}

// ── Empty command registry ─────────────────────────────────────────────────

/// A command registry with no commands. Used when no tool registry is configured.
struct EmptyCommandRegistry;

#[async_trait::async_trait]
impl weft_commands::CommandRegistry for EmptyCommandRegistry {
    async fn list_commands(
        &self,
    ) -> Result<Vec<weft_core::CommandStub>, weft_commands::CommandError> {
        Ok(vec![])
    }

    async fn describe_command(
        &self,
        name: &str,
    ) -> Result<weft_core::CommandDescription, weft_commands::CommandError> {
        Err(weft_commands::CommandError::NotFound(name.to_string()))
    }

    async fn execute_command(
        &self,
        invocation: &weft_core::CommandInvocation,
    ) -> Result<weft_core::CommandResult, weft_commands::CommandError> {
        Err(weft_commands::CommandError::NotFound(
            invocation.name.clone(),
        ))
    }
}
