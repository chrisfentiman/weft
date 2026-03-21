use xshell::{Shell, cmd};

use crate::util::Result;

/// Run `cargo xtask lint`: clippy followed by formatting check.
///
/// Fails fast on clippy errors — if the codebase does not compile or has
/// clippy errors, there is no point checking formatting.
pub(crate) fn run_lint(sh: &Shell) -> Result<()> {
    eprintln!("[xtask] running clippy...");
    cmd!(sh, "cargo clippy --workspace --all-targets -- -D warnings").run()?;

    eprintln!("[xtask] checking formatting...");
    cmd!(sh, "cargo fmt --check").run()?;

    Ok(())
}

/// Run `cargo xtask fmt`: apply `cargo fmt` to the entire workspace.
pub(crate) fn run_fmt(sh: &Shell) -> Result<()> {
    eprintln!("[xtask] formatting...");
    cmd!(sh, "cargo fmt").run()?;
    Ok(())
}
