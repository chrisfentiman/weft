use xshell::{Shell, cmd};

use crate::util::Result;
use crate::workspace;
use crate::{LsScope, TestArgs, TestCommand, TestScope};

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
/// Dispatch:
/// - No subcommand:            run all workspace tests.
/// - `unit [crate]`:           run unit tests (--lib filter).
/// - `integration [crate]`:    run integration tests (--test '*' filter).
/// - `ls [crate]`:             list crates with tests or targets for one crate.
/// - `<crate-name>`:           run all tests for one crate (external_subcommand).
pub(crate) fn run(sh: &Shell, args: TestArgs) -> Result<()> {
    match args.command {
        None => run_all(sh, args.postgres),
        Some(TestCommand::Unit(scope)) => run_unit(sh, scope, args.postgres),
        Some(TestCommand::Integration(scope)) => run_integration(sh, scope, args.postgres),
        Some(TestCommand::Ls(scope)) => run_ls(sh, scope),
        Some(TestCommand::Crate(args_os)) => run_crate(sh, args_os, args.postgres),
    }
}

/// Run all workspace tests.
///
/// When `--postgres` is set, also runs the weft crate with the postgres feature flag.
fn run_all(sh: &Shell, postgres: bool) -> Result<()> {
    let nextest = has_nextest(sh);
    announce_nextest(nextest);

    if nextest {
        cmd!(sh, "cargo nextest run --workspace").run()?;
    } else {
        cmd!(sh, "cargo test --workspace").run()?;
    }

    if postgres {
        eprintln!("[xtask] running weft tests with postgres feature...");
        run_postgres_weft_tests(sh, nextest)?;
    }

    Ok(())
}

/// Run unit tests only (--lib filter).
fn run_unit(sh: &Shell, scope: TestScope, postgres: bool) -> Result<()> {
    let nextest = has_nextest(sh);
    announce_nextest(nextest);

    if let Some(crate_name) = &scope.crate_name {
        // Single crate unit tests.
        let feature_flags = postgres_flags_for_crate(crate_name, postgres);
        if nextest {
            cmd!(
                sh,
                "cargo nextest run -p {crate_name} --lib {feature_flags...}"
            )
            .run()?;
        } else {
            cmd!(sh, "cargo test -p {crate_name} --lib {feature_flags...}").run()?;
        }
    } else {
        // Workspace-wide unit tests.
        if nextest {
            cmd!(sh, "cargo nextest run --workspace --lib").run()?;
        } else {
            cmd!(sh, "cargo test --workspace --lib").run()?;
        }
        if postgres {
            eprintln!("[xtask] running weft unit tests with postgres feature...");
            run_postgres_weft_unit_tests(sh, nextest)?;
        }
    }

    Ok(())
}

/// Run integration tests only (--test '*' filter).
fn run_integration(sh: &Shell, scope: TestScope, postgres: bool) -> Result<()> {
    let nextest = has_nextest(sh);
    announce_nextest(nextest);

    let using_default = scope.crate_name.is_none();
    let crate_name = scope.crate_name.unwrap_or_else(|| "weft".to_owned());

    // Integration tests typically live in the `weft` crate; print a note when
    // falling back to the default.
    if using_default {
        eprintln!(
            "[xtask] note: integration tests default to -p weft. \
             Use `cargo xtask test integration <crate>` to target another crate."
        );
    }

    let feature_flags = postgres_flags_for_crate(&crate_name, postgres);

    if nextest {
        cmd!(
            sh,
            "cargo nextest run -p {crate_name} --test * {feature_flags...}"
        )
        .run()?;
    } else {
        cmd!(sh, "cargo test -p {crate_name} --test * {feature_flags...}").run()?;
    }

    Ok(())
}

/// Run all tests for a specific crate (from external_subcommand).
fn run_crate(sh: &Shell, args_os: Vec<std::ffi::OsString>, postgres: bool) -> Result<()> {
    // external_subcommand guarantees at least one element.
    if args_os.len() > 1 {
        return Err(
            "unexpected arguments after crate name. \
             Use 'cargo xtask test unit <crate>' or 'cargo xtask test integration <crate>' to filter."
                .into(),
        );
    }

    let crate_name = args_os
        .into_iter()
        .next()
        .expect("external_subcommand always has at least one element")
        .into_string()
        .map_err(|_| "crate name contains non-UTF-8 characters")?;

    let nextest = has_nextest(sh);
    announce_nextest(nextest);

    let feature_flags = postgres_flags_for_crate(&crate_name, postgres);

    if nextest {
        cmd!(sh, "cargo nextest run -p {crate_name} {feature_flags...}").run()?;
    } else {
        cmd!(sh, "cargo test -p {crate_name} {feature_flags...}").run()?;
    }

    Ok(())
}

/// List crates with tests, or detailed test targets for a single crate.
fn run_ls(sh: &Shell, scope: LsScope) -> Result<()> {
    let root = sh.current_dir();

    if let Some(crate_name) = &scope.crate_name {
        // Detailed view for one crate.
        match workspace::find_crate(&root, crate_name)? {
            None => {
                eprintln!("[xtask] crate '{crate_name}' not found in workspace");
                return Err(format!("crate '{crate_name}' not found").into());
            }
            Some(info) => {
                println!("{}:", info.name);
                if info.has_lib {
                    println!("  Unit tests:       src/lib.rs");
                } else {
                    println!("  Unit tests:       none");
                }
                if info.integration_tests.is_empty() {
                    println!("  Integration tests: none");
                } else {
                    println!("  Integration tests:");
                    for target in &info.integration_tests {
                        println!("    tests/{target}.rs");
                    }
                }
                if info.features.is_empty() {
                    println!("  Features:         none");
                } else {
                    println!("  Features:         {}", info.features.join(", "));
                }
            }
        }
    } else {
        // Summary view: all crates with tests.
        let crates = workspace::discover_crates(&root)?;
        let crates_with_tests: Vec<_> = crates
            .iter()
            .filter(|c| c.has_lib || !c.integration_tests.is_empty())
            .collect();

        if crates_with_tests.is_empty() {
            println!("No crates with tests found.");
            return Ok(());
        }

        println!("Crates with tests:");
        for info in &crates_with_tests {
            let mut tags = Vec::new();
            if info.has_lib {
                tags.push("unit".to_owned());
            }
            if !info.integration_tests.is_empty() {
                let targets = info.integration_tests.join(", ");
                tags.push(format!("integration ({targets})"));
            }
            println!("  {:<30} {}", info.name, tags.join("  "));
        }
    }

    Ok(())
}

/// Build the --features flag slice for postgres when the target crate supports it.
///
/// The `weft_eventlog_postgres` feature only exists on the `weft` crate.
/// Do not pass it to other crates — cargo would error.
fn postgres_flags_for_crate(crate_name: &str, postgres: bool) -> Vec<&'static str> {
    if postgres && crate_name == "weft" {
        vec!["--features", "weft_eventlog_postgres"]
    } else {
        vec![]
    }
}

/// Run weft crate tests with the postgres feature flag.
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

/// Run weft unit tests with the postgres feature flag.
fn run_postgres_weft_unit_tests(sh: &Shell, nextest: bool) -> Result<()> {
    if nextest {
        cmd!(
            sh,
            "cargo nextest run -p weft --lib --features weft_eventlog_postgres"
        )
        .run()?;
    } else {
        cmd!(
            sh,
            "cargo test -p weft --lib --features weft_eventlog_postgres"
        )
        .run()?;
    }
    Ok(())
}

/// Print a note about which test runner is in use.
fn announce_nextest(nextest: bool) {
    if nextest {
        eprintln!("[xtask] using cargo-nextest");
    } else {
        eprintln!(
            "[xtask] note: cargo-nextest not found, using cargo test. \
             Run `cargo xtask setup` to install."
        );
    }
}
