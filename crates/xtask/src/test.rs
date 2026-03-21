use xshell::{Shell, cmd};

use crate::TestArgs;
use crate::util::Result;

/// Check whether `cargo-nextest` is available on PATH.
pub(crate) fn has_nextest(sh: &Shell) -> bool {
    cmd!(sh, "cargo nextest --version")
        .quiet()
        .ignore_status()
        .run()
        .is_ok()
}

/// Run `cargo xtask test`.
///
/// Uses `cargo-nextest` if installed, falls back to `cargo test`.
/// Scope is controlled by `--crate`, `--integration`, and `--unit` flags.
/// When `--postgres` is set without `--crate`, runs two commands:
/// the workspace test suite plus a separate `weft`-scoped run with the
/// postgres feature flag.
pub(crate) fn run(sh: &Shell, args: TestArgs) -> Result<()> {
    let nextest = has_nextest(sh);
    if nextest {
        eprintln!("[xtask] using cargo-nextest");
    } else {
        eprintln!(
            "[xtask] note: cargo-nextest not found, using cargo test. \
             Run `cargo xtask setup` to install."
        );
    }

    run_tests(sh, &args, nextest, args.postgres)?;

    // When --postgres is set without a specific --crate, also run the weft
    // crate tests with the postgres feature flag (workspace --features does
    // not work correctly for package-scoped features).
    if args.postgres && args.crate_name.is_none() {
        eprintln!("[xtask] running weft tests with postgres feature...");
        run_postgres_weft_tests(sh, nextest)?;
    }

    Ok(())
}

/// Build and run the core test command based on args.
fn run_tests(sh: &Shell, args: &TestArgs, nextest: bool, postgres: bool) -> Result<()> {
    // Determine scope flags.
    //
    // Build into a Vec<String> first, then convert to &str slices for xshell.
    // This avoids lifetime issues with Option<String> temporaries.
    let package_scope_owned: Vec<String> = if let Some(name) = &args.crate_name {
        vec!["-p".to_owned(), name.clone()]
    } else if args.integration {
        // Integration tests live in crates/weft/tests/; scope to -p weft.
        vec!["-p".to_owned(), "weft".to_owned()]
    } else {
        vec![
            "--workspace".to_owned(),
            "--exclude".to_owned(),
            "xtask".to_owned(),
        ]
    };
    let package_scope: Vec<&str> = package_scope_owned.iter().map(String::as_str).collect();

    // Determine test filter flags.
    let filter_flags: Vec<&str> = if args.integration {
        vec!["--test", "*"]
    } else if args.unit {
        vec!["--lib"]
    } else {
        vec![]
    };

    // Feature flags only apply when --postgres and there IS a --crate specified
    // (workspace-wide postgres tests are handled separately in run()).
    let feature_flags: Vec<&str> = if postgres && args.crate_name.is_some() {
        vec!["--features", "weft_eventlog_postgres"]
    } else {
        vec![]
    };

    if nextest {
        cmd!(
            sh,
            "cargo nextest run {package_scope...} {filter_flags...} {feature_flags...}"
        )
        .run()?;
    } else {
        cmd!(
            sh,
            "cargo test {package_scope...} {filter_flags...} {feature_flags...}"
        )
        .run()?;
    }

    Ok(())
}

/// Run weft crate tests with the postgres feature flag (used by --postgres without --crate).
fn run_postgres_weft_tests(sh: &Shell, nextest: bool) -> Result<()> {
    if nextest {
        cmd!(
            sh,
            "cargo nextest run -p weft --features weft_eventlog_postgres"
        )
        .run()?;
    } else {
        cmd!(sh, "cargo test -p weft --features weft_eventlog_postgres").run()?;
    }
    Ok(())
}
