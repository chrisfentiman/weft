use std::path::PathBuf;

/// Type alias used throughout the xtask crate.
///
/// xshell errors (command failures) are automatically boxed via the `?` operator.
/// Workspace root detection and other file I/O errors are also boxed through this alias.
pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Find the workspace root by walking up from `CARGO_MANIFEST_DIR` until a
/// `Cargo.toml` containing `[workspace]` is found.
///
/// When invoked via `cargo xtask`, `CARGO_MANIFEST_DIR` is set to the xtask
/// crate directory (`crates/xtask/`). The walk reaches the repo root, which
/// contains the workspace manifest.
///
/// Falls back to the current working directory if `CARGO_MANIFEST_DIR` is
/// not set (e.g., direct binary invocation outside of cargo).
pub(crate) fn workspace_root() -> Result<PathBuf> {
    let start = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().expect("cannot determine current directory"));

    let mut dir: &std::path::Path = start.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let contents = std::fs::read_to_string(&cargo_toml)
                .map_err(|e| format!("failed to read {}: {e}", cargo_toml.display()))?;
            if contents.contains("[workspace]") {
                return Ok(dir.to_path_buf());
            }
        }
        dir = dir
            .parent()
            .ok_or("cannot find workspace root: no Cargo.toml with [workspace] found")?;
    }
}

/// Build the feature flags argument slice for cargo commands.
///
/// Returns an empty vec when no features are requested, or
/// `["--features", "weft_eventlog_postgres"]` when `postgres` is true.
///
/// The returned vec contains `&'static str` slices so they can be spliced
/// directly into `cmd!` macro calls.
pub(crate) fn feature_args(postgres: bool) -> Vec<&'static str> {
    if postgres {
        vec!["--features", "weft_eventlog_postgres"]
    } else {
        vec![]
    }
}

/// Escape a string for embedding as a JSON string value.
///
/// Returns the value wrapped in double quotes with internal characters
/// escaped per the JSON spec (RFC 8259 §7):
/// - `"` → `\"`
/// - `\` → `\\`
/// - `\n`, `\r`, `\t` → their two-character escape sequences
/// - Other control characters (U+0000–U+001F) → `\uXXXX`
/// - All other characters pass through unchanged.
pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if c.is_control() => {
                // Emit \uXXXX for remaining control characters (U+0000–U+001F,
                // excluding the named escapes handled above).
                // Use encode_utf16 to correctly handle surrogate pairs for
                // code points above U+FFFF (though those are not control chars,
                // the loop handles them uniformly).
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!(r"\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::json_escape;

    #[test]
    fn test_simple_string() {
        assert_eq!(json_escape("hello"), r#""hello""#);
    }

    #[test]
    fn test_quotes_escaped() {
        assert_eq!(json_escape(r#"say "hello""#), r#""say \"hello\"""#);
    }

    #[test]
    fn test_backslash_escaped() {
        assert_eq!(json_escape(r"path\to\file"), r#""path\\to\\file""#);
    }

    #[test]
    fn test_newline_escaped() {
        assert_eq!(json_escape("line1\nline2"), r#""line1\nline2""#);
    }

    #[test]
    fn test_control_char_escaped() {
        assert_eq!(json_escape("\x00"), r#""\u0000""#);
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(json_escape(""), r#""""#);
    }

    #[test]
    fn test_unicode_passthrough() {
        // Non-control unicode characters pass through unescaped.
        assert_eq!(json_escape("hello world"), r#""hello world""#);
    }

    #[test]
    fn test_tab_escaped() {
        assert_eq!(json_escape("col1\tcol2"), r#""col1\tcol2""#);
    }

    #[test]
    fn test_carriage_return_escaped() {
        assert_eq!(json_escape("line\r\n"), r#""line\r\n""#);
    }

    #[test]
    fn test_multiple_special_chars() {
        assert_eq!(json_escape("a\"b\\c\nd"), r#""a\"b\\c\nd""#);
    }
}
