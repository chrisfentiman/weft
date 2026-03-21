use xshell::{Shell, cmd};

use crate::BuildArgs;
use crate::util::{Result, feature_args};

/// Run `cargo xtask build`.
///
/// Without `--postgres`: builds the entire workspace.
/// With `--postgres`: builds only the `weft` crate with the postgres feature flag,
/// because `--features` applied to `--workspace` does not work as expected for
/// package-scoped features.
pub(crate) fn run(sh: &Shell, args: BuildArgs) -> Result<()> {
    if args.postgres {
        eprintln!("[xtask] building weft with postgres...");
        let features = feature_args(true);
        if args.release {
            cmd!(sh, "cargo build -p weft --release {features...}").run()?;
        } else {
            cmd!(sh, "cargo build -p weft {features...}").run()?;
        }
    } else {
        eprintln!("[xtask] building workspace...");
        if args.release {
            cmd!(sh, "cargo build --workspace --release").run()?;
        } else {
            cmd!(sh, "cargo build --workspace").run()?;
        }
    }
    Ok(())
}
