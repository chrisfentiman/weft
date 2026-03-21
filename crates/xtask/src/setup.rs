use xshell::{Shell, cmd};

use crate::util::Result;

/// A development tool that can be installed via cargo.
struct Tool {
    /// The crate name used by `cargo install` / `cargo binstall`.
    crate_name: &'static str,
    /// The cargo subcommand used to detect whether the tool is installed.
    /// E.g. `"nextest"` → `cargo nextest --version`.
    version_cmd: &'static str,
}

const TOOLS: &[Tool] = &[
    Tool {
        crate_name: "cargo-nextest",
        version_cmd: "nextest",
    },
    Tool {
        crate_name: "cargo-mutants",
        version_cmd: "mutants",
    },
];

/// Run `cargo xtask setup`: install development tools if not already present.
///
/// Uses `cargo-binstall` for fast binary installation if available,
/// falling back to `cargo install` (compiles from source).
pub(crate) fn run(sh: &Shell) -> Result<()> {
    let use_binstall = has_binstall(sh);

    if !use_binstall {
        eprintln!("[xtask] note: install cargo-binstall for faster tool installation");
    }

    for tool in TOOLS {
        install_tool(sh, tool, use_binstall)?;
    }

    eprintln!("[xtask] setup complete");
    Ok(())
}

/// Check whether `cargo-binstall` is available on PATH.
fn has_binstall(sh: &Shell) -> bool {
    cmd!(sh, "cargo binstall --version")
        .quiet()
        .ignore_status()
        .run()
        .is_ok()
}

/// Install a single tool if not already present.
fn install_tool(sh: &Shell, tool: &Tool, use_binstall: bool) -> Result<()> {
    let version_subcmd = tool.version_cmd;
    let already_installed = cmd!(sh, "cargo {version_subcmd} --version")
        .quiet()
        .ignore_status()
        .run()
        .is_ok();

    if already_installed {
        eprintln!("[xtask] {} already installed", tool.crate_name);
        return Ok(());
    }

    eprintln!("[xtask] installing {}...", tool.crate_name);
    let crate_name = tool.crate_name;

    if use_binstall {
        cmd!(sh, "cargo binstall --no-confirm {crate_name}").run()?;
    } else {
        cmd!(sh, "cargo install {crate_name}").run()?;
    }

    Ok(())
}
