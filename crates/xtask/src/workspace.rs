use std::path::{Path, PathBuf};

use crate::util::Result;

/// A workspace crate discovered from the root Cargo.toml.
pub(crate) struct CrateInfo {
    /// Crate name from [package] name in its Cargo.toml.
    pub(crate) name: String,

    /// Absolute path to the crate directory.
    #[allow(dead_code)]
    pub(crate) path: PathBuf,

    /// Whether this crate has a lib.rs (can have unit tests).
    pub(crate) has_lib: bool,

    /// Whether this crate has a main.rs (binary crate).
    pub(crate) has_bin: bool,

    /// Integration test files (files in tests/ directory).
    ///
    /// Each entry is the file stem (e.g., "api" for tests/api.rs).
    pub(crate) integration_tests: Vec<String>,

    /// Feature flags defined in the crate's Cargo.toml.
    pub(crate) features: Vec<String>,
}

/// Discover all workspace member crates.
///
/// Reads the root Cargo.toml, resolves the `members` glob pattern,
/// excludes entries in the `exclude` list, and scans each crate
/// directory for test targets and features.
pub(crate) fn discover_crates(workspace_root: &Path) -> Result<Vec<CrateInfo>> {
    let cargo_toml_path = workspace_root.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml_path)
        .map_err(|e| format!("failed to read {}: {e}", cargo_toml_path.display()))?;

    let members = extract_toml_string_array(&content, "members");
    let excludes = extract_toml_string_array(&content, "exclude");

    let mut crates = Vec::new();

    for pattern in &members {
        // Handle the common pattern "crates/*" — list directories under a prefix.
        // We do not use the glob crate; we handle the specific patterns used in this workspace.
        if let Some(prefix) = pattern.strip_suffix("/*") {
            let dir = workspace_root.join(prefix);
            if !dir.is_dir() {
                continue;
            }

            let mut entries: Vec<_> = std::fs::read_dir(&dir)
                .map_err(|e| format!("failed to read {}: {e}", dir.display()))?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();

            // Sort for deterministic output.
            entries.sort_by_key(|e| e.file_name());

            for entry in entries {
                let crate_path = entry.path();
                let relative = crate_path
                    .strip_prefix(workspace_root)
                    .unwrap_or(&crate_path);
                let relative_str = relative.to_string_lossy();

                // Skip excluded paths.
                if excludes.iter().any(|ex| ex == relative_str.as_ref()) {
                    continue;
                }

                if let Some(info) = scan_crate(&crate_path)? {
                    crates.push(info);
                }
            }
        } else {
            // Exact path entry (no glob).
            let crate_path = workspace_root.join(pattern);
            let relative = crate_path
                .strip_prefix(workspace_root)
                .unwrap_or(&crate_path);
            let relative_str = relative.to_string_lossy();

            if excludes.iter().any(|ex| ex == relative_str.as_ref()) {
                continue;
            }

            if crate_path.is_dir()
                && let Some(info) = scan_crate(&crate_path)?
            {
                crates.push(info);
            }
        }
    }

    Ok(crates)
}

/// Discover a single crate by name.
///
/// Returns None if the crate is not a workspace member.
pub(crate) fn find_crate(workspace_root: &Path, name: &str) -> Result<Option<CrateInfo>> {
    let crates = discover_crates(workspace_root)?;
    Ok(crates.into_iter().find(|c| c.name == name))
}

/// Scan a crate directory and return a CrateInfo, or None if the directory
/// does not contain a valid Cargo.toml with a [package] name.
fn scan_crate(path: &Path) -> Result<Option<CrateInfo>> {
    let cargo_toml_path = path.join("Cargo.toml");
    if !cargo_toml_path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&cargo_toml_path)
        .map_err(|e| format!("failed to read {}: {e}", cargo_toml_path.display()))?;

    let name = match extract_package_name(&content) {
        Some(n) => n,
        None => return Ok(None),
    };

    let has_lib = path.join("src/lib.rs").exists();
    let has_bin = path.join("src/main.rs").exists() || has_bin_target(&content);
    let integration_tests = scan_integration_tests(path);
    let features = extract_features(&content);

    Ok(Some(CrateInfo {
        name,
        path: path.to_path_buf(),
        has_lib,
        has_bin,
        integration_tests,
        features,
    }))
}

/// Scan a crate's tests/ directory for integration test targets.
///
/// Only lists *.rs files directly in tests/ (not in subdirectories, which are
/// modules/helpers, not test targets).
fn scan_integration_tests(crate_path: &Path) -> Vec<String> {
    let tests_dir = crate_path.join("tests");
    if !tests_dir.is_dir() {
        return Vec::new();
    }

    let mut targets = Vec::new();

    let entries = match std::fs::read_dir(&tests_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        // Only files directly in tests/, not subdirectories.
        if !p.is_file() {
            continue;
        }
        if p.extension().map(|e| e == "rs").unwrap_or(false)
            && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
        {
            targets.push(stem.to_owned());
        }
    }

    targets.sort();
    targets
}

/// Check if the Cargo.toml defines [[bin]] targets.
fn has_bin_target(content: &str) -> bool {
    content.contains("[[bin]]")
}

/// Extract a string array value from a TOML-formatted string.
///
/// Searches for `key = [...]` and returns the quoted strings within.
/// Handles both single-line and multi-line arrays.
///
/// Returns an empty Vec if the key is not found.
pub(crate) fn extract_toml_string_array(content: &str, key: &str) -> Vec<String> {
    // Find the start of the array value for this key.
    // We look for `key = [` potentially with spaces around `=`.
    let search = format!("{key} = [");
    let start = match content.find(&search) {
        Some(pos) => pos + search.len(),
        None => {
            // Also try without spaces: `key=[`
            let search2 = format!("{key}=[");
            match content.find(&search2) {
                Some(pos) => pos + search2.len(),
                None => return Vec::new(),
            }
        }
    };

    // Collect everything from `[` until the matching `]`.
    let rest = &content[start..];
    let end = match rest.find(']') {
        Some(pos) => pos,
        None => return Vec::new(),
    };

    let array_content = &rest[..end];

    // Extract all double-quoted strings.
    let mut results = Vec::new();
    let mut remaining = array_content;
    while let Some(open) = remaining.find('"') {
        let after_open = &remaining[open + 1..];
        match after_open.find('"') {
            Some(close) => {
                let value = &after_open[..close];
                results.push(value.to_owned());
                remaining = &after_open[close + 1..];
            }
            None => break,
        }
    }

    results
}

/// Extract the package name from a crate Cargo.toml.
///
/// Looks for `name = "..."` under `[package]`.
pub(crate) fn extract_package_name(content: &str) -> Option<String> {
    // Find the [package] section first, then look for `name = "..."` within it.
    let package_section = content.find("[package]")?;
    let after_package = &content[package_section..];

    // Find the end of the [package] section: the next `[` that starts a new section.
    // The `[package]` marker itself is 9 chars. We look after that.
    let section_body = &after_package["[package]".len()..];
    let section_end = section_body.find("\n[").unwrap_or(section_body.len());
    let section = &section_body[..section_end];

    // Find `name = "..."` in this section.
    for line in section.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let rest = rest.trim();
                if let Some(inner) = rest.strip_prefix('"')
                    && let Some(end_quote) = inner.find('"')
                {
                    return Some(inner[..end_quote].to_owned());
                }
            }
        }
    }

    None
}

/// Extract feature names from a crate Cargo.toml.
///
/// Looks for keys under `[features]`.
pub(crate) fn extract_features(content: &str) -> Vec<String> {
    let features_section = match content.find("[features]") {
        Some(pos) => pos,
        None => return Vec::new(),
    };

    let after_features = &content[features_section + "[features]".len()..];

    // Section ends at the next top-level section header.
    let section_end = after_features.find("\n[").unwrap_or(after_features.len());
    let section = &after_features[..section_end];

    let mut features = Vec::new();
    for line in section.lines() {
        let trimmed = line.trim();
        // Skip empty lines, comments, and lines that don't define a feature key.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Feature lines look like: `feature-name = [...]` or `feature-name = "..."`.
        if let Some(eq_pos) = trimmed.find('=') {
            let name = trimmed[..eq_pos].trim();
            // Filter out non-identifier-like lines (e.g., `default = [...]` is valid).
            if !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
            {
                features.push(name.to_owned());
            }
        }
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_single_line_array() {
        let content = r#"
[workspace]
members = ["crates/*"]
"#;
        assert_eq!(
            extract_toml_string_array(content, "members"),
            vec!["crates/*"]
        );
    }

    #[test]
    fn test_extract_multi_line_array() {
        let content = r#"
[workspace]
members = [
    "crates/*",
    "tools/*",
]
"#;
        assert_eq!(
            extract_toml_string_array(content, "members"),
            vec!["crates/*", "tools/*"]
        );
    }

    #[test]
    fn test_extract_missing_key() {
        let content = "[workspace]\nresolver = \"3\"\n";
        assert!(extract_toml_string_array(content, "members").is_empty());
    }

    #[test]
    fn test_extract_package_name() {
        let content = r#"
[package]
name = "weft_reactor"
version.workspace = true
"#;
        assert_eq!(
            extract_package_name(content),
            Some("weft_reactor".to_string())
        );
    }

    #[test]
    fn test_extract_package_name_missing() {
        let content = "[workspace]\nresolver = \"3\"\n";
        assert_eq!(extract_package_name(content), None);
    }

    #[test]
    fn test_extract_features() {
        let content = r#"
[features]
test-support = []
weft_eventlog_postgres = ["dep:weft_eventlog_postgres", "dep:sqlx"]
"#;
        let features = extract_features(content);
        assert!(features.contains(&"test-support".to_string()));
        assert!(features.contains(&"weft_eventlog_postgres".to_string()));
    }

    #[test]
    fn test_extract_features_none() {
        let content = "[package]\nname = \"weft_core\"\n";
        assert!(extract_features(content).is_empty());
    }

    #[test]
    fn test_extract_exclude_array() {
        let content = r#"
[workspace]
members = ["crates/*"]
exclude = ["crates/xtask"]
"#;
        assert_eq!(
            extract_toml_string_array(content, "exclude"),
            vec!["crates/xtask"]
        );
    }

    #[test]
    fn test_extract_package_name_with_workspace_key() {
        // Workspace Cargo.toml has [workspace] before package sections in crates;
        // verify we only look at [package].
        let content = r#"
[package]
name = "weft_tools"
edition = "2024"
"#;
        assert_eq!(
            extract_package_name(content),
            Some("weft_tools".to_string())
        );
    }
}
