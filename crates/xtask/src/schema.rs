//! `cargo xtask schema` — generate and verify the JSON Schema for weft configuration.

use std::path::{Path, PathBuf};

use weft_core::config::WeftConfig;

use crate::util::Result;

/// Generate the JSON Schema for `WeftConfig` and write it to `output`.
pub(crate) fn cmd_generate(output: &Path) -> Result<()> {
    let schema = schemars::schema_for!(WeftConfig);
    let json = serde_json::to_string_pretty(&schema)?;
    // Ensure parent directory exists.
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, &json)?;
    println!("schema written to {}", output.display());
    Ok(())
}

/// Verify that the committed schema at `schema_path` matches the current config types.
///
/// Called by `cargo xtask ci` to ensure the committed schema is not stale.
/// Fails with an actionable error message if the schema needs regeneration.
pub(crate) fn cmd_verify(schema_path: &Path) -> Result<()> {
    let current = schemars::schema_for!(WeftConfig);
    let current_json = serde_json::to_string_pretty(&current)?;
    let committed =
        std::fs::read_to_string(schema_path).map_err(|e| -> Box<dyn std::error::Error> {
            format!(
                "cannot read schema at '{}': {} — run `cargo xtask schema` to generate it",
                schema_path.display(),
                e
            )
            .into()
        })?;
    if current_json != committed {
        return Err(format!(
            "schema at '{}' is out of date — run `cargo xtask schema` to regenerate",
            schema_path.display()
        )
        .into());
    }
    println!("[xtask] schema is up to date");
    Ok(())
}

/// Default schema output path relative to the workspace root.
pub(crate) fn default_schema_path() -> PathBuf {
    PathBuf::from("config/weft.schema.json")
}
