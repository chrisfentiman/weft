//! Weft gateway binary entry point.
//!
//! Phase 1 stub: loads config via clap, resolves env vars, validates config.
//! Full gateway engine, HTTP server, and component wiring are implemented in Phase 4.

use clap::Parser;
use std::path::PathBuf;

/// Weft — a semantic command gateway for LLMs.
#[derive(Debug, Parser)]
#[command(name = "weft", version, about)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(short, long, default_value = "config/weft.toml")]
    config: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Load and resolve configuration
    let config_str = std::fs::read_to_string(&cli.config).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to read config file '{}': {e}",
            cli.config.display()
        );
        std::process::exit(1);
    });

    let mut config: weft_core::WeftConfig = toml::from_str(&config_str).unwrap_or_else(|e| {
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

    tracing::info!("configuration loaded from {}", cli.config.display());
    tracing::info!("gateway not yet implemented (Phase 4)");
}
