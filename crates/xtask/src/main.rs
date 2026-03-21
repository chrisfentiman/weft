mod build;
mod ci;
mod grpc;
mod lint;
mod run;
mod setup;
mod test;
mod util;

use clap::{Parser, Subcommand};

/// Weft workspace task runner.
///
/// Common development workflows for the Weft workspace.
/// Run `cargo xtask <command> --help` for details on each command.
#[derive(Debug, Parser)]
#[command(
    name = "xtask",
    about = "Weft workspace task runner",
    long_about = "Common development workflows for the Weft workspace.\nRun `cargo xtask <command> --help` for details on each command."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build the workspace.
    Build(BuildArgs),

    /// Run tests.
    ///
    /// Uses cargo-nextest if installed, falls back to cargo test.
    Test(TestArgs),

    /// Run clippy and check formatting.
    Lint,

    /// Fix formatting (cargo fmt).
    Fmt,

    /// Build and start weft.
    Run(RunArgs),

    /// Start mockd with the default config.
    Mockd,

    /// Full CI check: fmt + clippy + test + build. What a PR must pass.
    Ci,

    /// Send gRPC requests to a running weft instance.
    Grpc(GrpcArgs),

    /// Install development tools (nextest, mutants).
    Setup,
}

/// Arguments for `cargo xtask build`.
#[derive(Debug, clap::Args)]
pub(crate) struct BuildArgs {
    /// Build in release mode.
    #[arg(long)]
    pub(crate) release: bool,

    /// Enable the Postgres event log feature.
    #[arg(long)]
    pub(crate) postgres: bool,
}

/// Arguments for `cargo xtask test`.
#[derive(Debug, clap::Args)]
pub(crate) struct TestArgs {
    /// Test a specific crate instead of the whole workspace.
    #[arg(long = "crate", value_name = "CRATE")]
    pub(crate) crate_name: Option<String>,

    /// Run only integration tests (tests/ directory).
    ///
    /// When used without --crate, defaults to -p weft (integration tests live there).
    #[arg(long, conflicts_with = "unit")]
    pub(crate) integration: bool,

    /// Run only unit tests (lib tests).
    #[arg(long, conflicts_with = "integration")]
    pub(crate) unit: bool,

    /// Enable the Postgres event log feature.
    #[arg(long)]
    pub(crate) postgres: bool,
}

/// Arguments for `cargo xtask run`.
#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    /// Path to the config file.
    #[arg(
        short = 'c',
        long,
        default_value = "config/weft.toml",
        value_name = "PATH"
    )]
    pub(crate) config: String,

    /// Run in release mode.
    #[arg(long)]
    pub(crate) release: bool,

    /// Enable the Postgres event log feature.
    #[arg(long)]
    pub(crate) postgres: bool,
}

/// Arguments for `cargo xtask grpc`.
#[derive(Debug, clap::Args)]
pub(crate) struct GrpcArgs {
    #[command(subcommand)]
    pub(crate) command: GrpcCommand,
}

#[derive(Debug, Subcommand)]
pub(crate) enum GrpcCommand {
    /// Send a Chat request to a running weft instance.
    Chat(GrpcChatArgs),

    /// Check weft health via HTTP.
    Health(GrpcHealthArgs),
}

/// Arguments for `cargo xtask grpc chat`.
#[derive(Debug, clap::Args)]
pub(crate) struct GrpcChatArgs {
    /// The message to send.
    pub(crate) message: String,

    /// Host and port of the weft instance.
    #[arg(long, default_value = "localhost:8080")]
    pub(crate) addr: String,
}

/// Arguments for `cargo xtask grpc health`.
#[derive(Debug, clap::Args)]
pub(crate) struct GrpcHealthArgs {
    /// Host and port of the weft instance.
    #[arg(long, default_value = "localhost:8080")]
    pub(crate) addr: String,
}

fn main() -> util::Result<()> {
    let cli = Cli::parse();

    let sh = xshell::Shell::new()?;
    let workspace_root = util::workspace_root()?;
    sh.change_dir(&workspace_root);

    match cli.command {
        Command::Build(args) => build::run(&sh, args),
        Command::Test(args) => test::run(&sh, args),
        Command::Lint => lint::run_lint(&sh),
        Command::Fmt => lint::run_fmt(&sh),
        Command::Run(args) => run::run_weft(&sh, args),
        Command::Mockd => run::run_mockd(&sh),
        Command::Ci => ci::run(&sh),
        Command::Grpc(args) => grpc::run(&sh, args),
        Command::Setup => setup::run(&sh),
    }
}
