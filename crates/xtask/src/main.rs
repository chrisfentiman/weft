mod build;
mod ci;
mod completions;
mod grpc;
mod lint;
mod run;
mod setup;
mod test;
mod util;
mod workspace;

use std::ffi::OsString;

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

    /// Generate shell completions.
    ///
    /// Examples:
    ///   cargo xtask completions bash >> ~/.bashrc
    ///   cargo xtask completions zsh > ~/.zfunc/_cargo-xtask
    ///   cargo xtask completions fish > ~/.config/fish/completions/cargo-xtask.fish
    ///
    /// After generating, restart your shell or source the file.
    Completions(CompletionsArgs),
}

/// Arguments for `cargo xtask build`.
#[derive(Debug, clap::Args)]
pub(crate) struct BuildArgs {
    #[command(subcommand)]
    pub(crate) command: Option<BuildCommand>,

    /// Enable the Postgres event log feature.
    #[arg(long, global = true)]
    pub(crate) postgres: bool,
}

#[derive(Debug, Subcommand)]
pub(crate) enum BuildCommand {
    /// Build in release mode.
    Release,

    /// List workspace crates.
    Ls,
}

/// Arguments for `cargo xtask test`.
#[derive(Debug, clap::Args)]
#[command(
    after_help = "Run `cargo xtask test <crate-name>` to test a specific crate.\nReserved names: unit, integration, ls (cannot be used as bare crate names)."
)]
pub(crate) struct TestArgs {
    #[command(subcommand)]
    pub(crate) command: Option<TestCommand>,

    /// Enable the Postgres event log feature.
    #[arg(long, global = true)]
    pub(crate) postgres: bool,
}

#[derive(Debug, Subcommand)]
pub(crate) enum TestCommand {
    /// Run unit tests only (lib tests).
    Unit(TestScope),

    /// Run integration tests only (tests/ directory).
    Integration(TestScope),

    /// List crates with tests, or test targets for a specific crate.
    Ls(LsScope),

    /// Run all tests for a specific crate.
    ///
    /// Catches bare crate names: `cargo xtask test weft_reactor`.
    #[command(external_subcommand)]
    Crate(Vec<OsString>),
}

/// Optional crate scope for unit/integration subcommands.
#[derive(Debug, clap::Args)]
pub(crate) struct TestScope {
    /// Crate to test. If omitted, tests the whole workspace.
    pub(crate) crate_name: Option<String>,
}

/// Optional crate scope for the ls subcommand.
#[derive(Debug, clap::Args)]
pub(crate) struct LsScope {
    /// Crate to inspect. If omitted, lists all crates with tests.
    pub(crate) crate_name: Option<String>,
}

/// Arguments for `cargo xtask run`.
///
/// The run command keeps flags because `--release` and `--postgres` are genuine
/// modifiers, not distinct modes. `cargo xtask run --release --postgres` is
/// natural; `cargo xtask run release postgres` would imply they are independent
/// modes, which they are not.
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

    /// List available RPC methods from the proto definition.
    Ls,
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

/// Arguments for `cargo xtask completions`.
#[derive(Debug, clap::Args)]
pub(crate) struct CompletionsArgs {
    /// Shell to generate completions for.
    pub(crate) shell: clap_complete::Shell,
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
        Command::Completions(args) => completions::run(args),
    }
}
