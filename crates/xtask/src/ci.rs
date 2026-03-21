use xshell::{Shell, cmd};

use crate::util::Result;

/// Run `cargo xtask ci`: the full CI gate in order.
///
/// Steps (fail-fast):
/// 1. `cargo fmt --check`      — formatting is cheapest; catch it first
/// 2. `cargo clippy ...`       — static analysis before spending time on tests
/// 3. `cargo nextest run` / `cargo test` — workspace tests
/// 4. `cargo build --workspace` — confirm everything compiles cleanly
///
/// The CI subcommand calls cargo commands directly rather than delegating to
/// the lint/test module functions. This keeps CI output formatting independent
/// so each step can print its own `[xtask] FAILED: <step>` banner.
pub(crate) fn run(sh: &Shell) -> Result<()> {
    eprintln!("[xtask] running CI checks...");

    // Step 1: formatting check.
    eprintln!("[xtask] checking formatting...");
    if let Err(e) = cmd!(sh, "cargo fmt --check").run() {
        eprintln!("[xtask] FAILED: formatting");
        return Err(e.into());
    }

    // Step 2: clippy.
    eprintln!("[xtask] running clippy...");
    if let Err(e) =
        cmd!(sh, "cargo clippy --workspace --exclude xtask --all-targets -- -D warnings").run()
    {
        eprintln!("[xtask] FAILED: clippy");
        return Err(e.into());
    }

    // Step 3: tests (use nextest if available, fall back to cargo test).
    eprintln!("[xtask] running tests...");
    let test_result = if has_nextest(sh) {
        cmd!(sh, "cargo nextest run --workspace --exclude xtask").run()
    } else {
        eprintln!(
            "[xtask] note: cargo-nextest not found, using cargo test. \
             Run `cargo xtask setup` to install."
        );
        cmd!(sh, "cargo test --workspace --exclude xtask").run()
    };
    if let Err(e) = test_result {
        eprintln!("[xtask] FAILED: tests");
        return Err(e.into());
    }

    // Step 4: build.
    eprintln!("[xtask] building workspace...");
    if let Err(e) = cmd!(sh, "cargo build --workspace --exclude xtask").run() {
        eprintln!("[xtask] FAILED: build");
        return Err(e.into());
    }

    eprintln!("[xtask] CI checks passed");
    Ok(())
}

/// Check whether `cargo-nextest` is available on PATH.
fn has_nextest(sh: &Shell) -> bool {
    cmd!(sh, "cargo nextest --version")
        .quiet()
        .ignore_status()
        .run()
        .is_ok()
}
