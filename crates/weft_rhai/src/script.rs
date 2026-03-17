//! Compiled Rhai script type.
//!
//! `CompiledScript` bundles a compiled Rhai AST with the source path for
//! diagnostics. Scripts are loaded once at startup and held for the process
//! lifetime. Immutable after construction — safe to share via `Arc`.

use std::sync::Arc;

use rhai::Engine;

use crate::error::ScriptError;

/// A compiled Rhai script with metadata.
///
/// Bundles the compiled AST with the source path for diagnostics.
/// Immutable after construction — safe to share via Arc across threads.
///
/// `CompiledScript` does NOT own an engine. The same script can be executed
/// on different engines (though typically each script has one engine).
pub struct CompiledScript {
    /// Path to the source file (for error messages and logging).
    path: String,
    /// Compiled AST. Immutable. `Send + Sync` with `rhai/sync` feature.
    ast: Arc<rhai::AST>,
}

impl CompiledScript {
    /// Load and compile a Rhai script from disk.
    ///
    /// Reads the file, compiles it with the provided engine, and returns
    /// the compiled script. Fails fast on file I/O errors or syntax errors.
    ///
    /// # Arguments
    ///
    /// - `path`: Path to the `.rhai` file.
    /// - `engine`: Engine to compile with. The engine's registered functions
    ///   affect compilation (unknown functions are not compile-time errors in
    ///   Rhai, but registered types are needed for custom type syntax).
    ///
    /// # Errors
    ///
    /// - `ScriptError::FileNotFound` if the file cannot be read.
    /// - `ScriptError::CompilationFailed` if the script has syntax errors.
    pub fn load(path: &str, engine: &Engine) -> Result<Self, ScriptError> {
        let source = std::fs::read_to_string(path).map_err(|e| ScriptError::FileNotFound {
            path: path.to_string(),
            source: e,
        })?;

        let ast = engine
            .compile(&source)
            .map_err(|e| ScriptError::CompilationFailed {
                path: path.to_string(),
                message: e.to_string(),
            })?;

        Ok(Self {
            path: path.to_string(),
            ast: Arc::new(ast),
        })
    }

    /// Validate that specific function names are defined in the script.
    ///
    /// Uses `ast.iter_functions()` to check function metadata without
    /// executing the script.
    ///
    /// # Errors
    ///
    /// - `ScriptError::MissingFunction` for the first missing function.
    pub fn validate_functions(&self, required: &[&str]) -> Result<(), ScriptError> {
        let defined: Vec<&str> = self.ast.iter_functions().map(|f| f.name).collect();

        for &name in required {
            if !defined.contains(&name) {
                return Err(ScriptError::MissingFunction {
                    path: self.path.clone(),
                    function: name.to_string(),
                });
            }
        }

        Ok(())
    }

    /// The path to the source file.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The compiled AST, wrapped in Arc for sharing into spawn_blocking closures.
    pub fn ast(&self) -> &Arc<rhai::AST> {
        &self.ast
    }
}

impl std::fmt::Debug for CompiledScript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledScript")
            .field("path", &self.path)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineBuilder, SandboxLimits};
    use std::io::Write;

    fn default_engine() -> Engine {
        EngineBuilder::new(SandboxLimits::default()).build()
    }

    fn write_temp_script(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // ── Load ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_load_valid_script_succeeds() {
        let script = write_temp_script(r#"fn hook(e) { #{ decision: "allow" } }"#);
        let engine = default_engine();
        let result = CompiledScript::load(script.path().to_str().unwrap(), &engine);
        assert!(result.is_ok(), "valid script should compile: {result:?}");
    }

    #[test]
    fn test_load_nonexistent_path_returns_file_not_found() {
        let engine = default_engine();
        let result = CompiledScript::load("/nonexistent/path/to/script.rhai", &engine);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ScriptError::FileNotFound { .. }),
            "expected FileNotFound"
        );
    }

    #[test]
    fn test_load_syntax_error_returns_compilation_failed() {
        let script = write_temp_script("fn hook(e) { let x = ; }"); // syntax error
        let engine = default_engine();
        let result = CompiledScript::load(script.path().to_str().unwrap(), &engine);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ScriptError::CompilationFailed { .. }),
            "expected CompilationFailed"
        );
    }

    #[test]
    fn test_load_empty_script_succeeds() {
        // Empty scripts are valid Rhai (empty AST).
        let script = write_temp_script("");
        let engine = default_engine();
        let result = CompiledScript::load(script.path().to_str().unwrap(), &engine);
        assert!(result.is_ok(), "empty script is valid Rhai");
    }

    // ── Validate functions ───────────────────────────────────────────────────

    #[test]
    fn test_validate_functions_all_present_succeeds() {
        let script = write_temp_script(
            r#"
            fn format_request(req) { req }
            fn parse_response(resp) { resp }
        "#,
        );
        let engine = default_engine();
        let compiled = CompiledScript::load(script.path().to_str().unwrap(), &engine).unwrap();
        let result = compiled.validate_functions(&["format_request", "parse_response"]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_functions_missing_returns_error() {
        let script = write_temp_script(r#"fn format_request(req) { req }"#);
        let engine = default_engine();
        let compiled = CompiledScript::load(script.path().to_str().unwrap(), &engine).unwrap();
        let result = compiled.validate_functions(&["format_request", "parse_response"]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, ScriptError::MissingFunction { function, .. } if function == "parse_response"),
            "expected MissingFunction for parse_response, got: {err:?}"
        );
    }

    #[test]
    fn test_validate_functions_empty_required_succeeds() {
        let script = write_temp_script("fn hook(e) { e }");
        let engine = default_engine();
        let compiled = CompiledScript::load(script.path().to_str().unwrap(), &engine).unwrap();
        // No required functions — always passes.
        let result = compiled.validate_functions(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_functions_empty_script_returns_missing() {
        let script = write_temp_script("");
        let engine = default_engine();
        let compiled = CompiledScript::load(script.path().to_str().unwrap(), &engine).unwrap();
        let result = compiled.validate_functions(&["hook"]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ScriptError::MissingFunction { .. }
        ));
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    #[test]
    fn test_path_accessor() {
        let script = write_temp_script("fn hook(e) { e }");
        let path = script.path().to_str().unwrap().to_string();
        let engine = default_engine();
        let compiled = CompiledScript::load(&path, &engine).unwrap();
        assert_eq!(compiled.path(), path);
    }

    #[test]
    fn test_ast_accessor_returns_arc() {
        let script = write_temp_script("fn hook(e) { e }");
        let engine = default_engine();
        let compiled = CompiledScript::load(script.path().to_str().unwrap(), &engine).unwrap();
        let ast = compiled.ast();
        // Can clone the Arc cheaply.
        let _cloned = Arc::clone(ast);
    }

    #[test]
    fn test_debug_format_shows_path() {
        let script = write_temp_script("fn hook(e) { e }");
        let path = script.path().to_str().unwrap().to_string();
        let engine = default_engine();
        let compiled = CompiledScript::load(&path, &engine).unwrap();
        let debug_str = format!("{compiled:?}");
        assert!(debug_str.contains("CompiledScript"));
        assert!(debug_str.contains(&path));
    }
}
