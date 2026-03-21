use xshell::{Shell, cmd};

use crate::RunArgs;
use crate::util::Result;

/// Run `cargo xtask run`: build then start the `weft` binary.
///
/// Sets `RUST_LOG=info` unless already present in the environment.
/// In debug mode, uses `cargo run` directly (faster iteration).
/// In release mode, builds first then executes the compiled binary.
pub(crate) fn run_weft(sh: &Shell, args: RunArgs) -> Result<()> {
    // Set RUST_LOG=info only if the user has not already configured it.
    if sh.var("RUST_LOG").is_err() {
        sh.set_var("RUST_LOG", "info");
    }

    let config = args.config.as_str();

    if args.release {
        // Build the release binary first.
        if args.postgres {
            eprintln!("[xtask] building weft (release, postgres)...");
            cmd!(
                sh,
                "cargo build -p weft --release --features weft_eventlog_postgres"
            )
            .run()?;
        } else {
            eprintln!("[xtask] building weft (release)...");
            cmd!(sh, "cargo build -p weft --release").run()?;
        }

        // Verify the binary exists before attempting to execute it.
        let binary = sh.current_dir().join("target/release/weft");
        if !binary.exists() {
            return Err(format!(
                "expected binary at {} but it does not exist after build",
                binary.display()
            )
            .into());
        }

        eprintln!("[xtask] starting weft...");
        cmd!(sh, "./target/release/weft -c {config}").run()?;
    } else {
        // Debug mode: use cargo run for faster iteration.
        eprintln!("[xtask] starting weft (debug)...");
        if args.postgres {
            cmd!(
                sh,
                "cargo run -p weft --features weft_eventlog_postgres -- -c {config}"
            )
            .run()?;
        } else {
            cmd!(sh, "cargo run -p weft -- -c {config}").run()?;
        }
    }

    Ok(())
}

/// Run `cargo xtask mockd`: start the mock server.
///
/// Requires `mockd` to be on PATH. Prints a clear error if not found.
pub(crate) fn run_mockd(sh: &Shell) -> Result<()> {
    eprintln!("[xtask] starting mockd...");

    // Check for mockd on PATH before attempting to run it.
    if cmd!(sh, "mockd --version")
        .quiet()
        .ignore_status()
        .run()
        .is_err()
    {
        eprintln!("[xtask] error: mockd not found. See README for installation.");
        return Err("mockd not found on PATH".into());
    }

    cmd!(sh, "mockd serve --config config/mockd.yaml").run()?;
    Ok(())
}
