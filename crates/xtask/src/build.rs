use xshell::{Shell, cmd};

use crate::util::{Result, feature_args};
use crate::workspace;
use crate::{BuildArgs, BuildCommand};

/// Run `cargo xtask build`.
///
/// Dispatch table:
/// - `build ls`:           list workspace crates.
/// - `build release`:      `cargo build --workspace --release` (or weft-scoped with --postgres).
/// - `build` (no command): `cargo build --workspace` (or weft-scoped with --postgres).
pub(crate) fn run(sh: &Shell, args: BuildArgs) -> Result<()> {
    if let Some(BuildCommand::Ls) = &args.command {
        return run_ls(sh);
    }

    let release = matches!(&args.command, Some(BuildCommand::Release));

    if args.postgres {
        eprintln!("[xtask] building weft with postgres...");
        let features = feature_args(true);
        if release {
            cmd!(sh, "cargo build -p weft --release {features...}").run()?;
        } else {
            cmd!(sh, "cargo build -p weft {features...}").run()?;
        }
    } else {
        eprintln!("[xtask] building workspace...");
        if release {
            cmd!(sh, "cargo build --workspace --release").run()?;
        } else {
            cmd!(sh, "cargo build --workspace").run()?;
        }
    }

    Ok(())
}

/// Print a table of workspace crates for `build ls`.
fn run_ls(sh: &Shell) -> Result<()> {
    // workspace_root() is based on CARGO_MANIFEST_DIR; use the shell's working
    // directory as the resolved workspace root since main() changes into it.
    let root = sh.current_dir();
    let crates = workspace::discover_crates(&root)?;

    if crates.is_empty() {
        eprintln!("[xtask] no workspace crates found");
        return Ok(());
    }

    println!("Workspace crates:");
    for info in &crates {
        let kind = match (info.has_bin, info.has_lib) {
            (true, true) => "(bin + lib)",
            (true, false) => "(bin)",
            (false, true) => "(lib)",
            (false, false) => "",
        };
        println!("  {:<30} {kind}", info.name);
    }

    Ok(())
}
