//! Error types for Rhai script operations.
//!
//! `ScriptError` is the unified error type for all script lifecycle operations:
//! file I/O, compilation, execution, and type conversion. Consumer crates
//! convert this into their own domain error types via `From<ScriptError>`.

/// Error from Rhai script operations.
///
/// This is the shared error type for script loading, compilation, and execution.
/// Consumer crates wrap this in their own error types (HookError::RhaiError,
/// ProviderError::WireScriptError) to add domain context.
#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    /// Script file could not be read.
    #[error("script not found or unreadable: {path}: {source}")]
    FileNotFound {
        path: String,
        source: std::io::Error,
    },

    /// Script failed to compile (syntax error).
    #[error("script compilation failed: {path}: {message}")]
    CompilationFailed { path: String, message: String },

    /// Script execution failed at runtime (Rhai error, operation limit, etc.).
    #[error("script execution error: {path}: {message}")]
    ExecutionError { path: String, message: String },

    /// Script panicked during execution (caught by catch_unwind).
    #[error("script panicked: {path}: {message}")]
    Panic { path: String, message: String },

    /// spawn_blocking task failed (JoinError).
    #[error("script task join error: {path}: {message}")]
    TaskJoinError { path: String, message: String },

    /// A required function is missing from the script.
    #[error("script missing required function '{function}': {path}")]
    MissingFunction { path: String, function: String },

    /// Dynamic-to-JSON or JSON-to-Dynamic conversion failed.
    #[error("conversion error: {message}")]
    ConversionError { message: String },
}

impl ScriptError {
    /// The script path associated with this error, if any.
    pub fn script_path(&self) -> Option<&str> {
        match self {
            Self::FileNotFound { path, .. }
            | Self::CompilationFailed { path, .. }
            | Self::ExecutionError { path, .. }
            | Self::Panic { path, .. }
            | Self::TaskJoinError { path, .. }
            | Self::MissingFunction { path, .. } => Some(path),
            Self::ConversionError { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_not_found_display() {
        let err = ScriptError::FileNotFound {
            path: "/path/to/script.rhai".to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        };
        let msg = err.to_string();
        assert!(msg.contains("script not found or unreadable"));
        assert!(msg.contains("/path/to/script.rhai"));
    }

    #[test]
    fn test_compilation_failed_display() {
        let err = ScriptError::CompilationFailed {
            path: "hook.rhai".to_string(),
            message: "unexpected token at line 3".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("script compilation failed"));
        assert!(msg.contains("hook.rhai"));
        assert!(msg.contains("unexpected token"));
    }

    #[test]
    fn test_execution_error_display() {
        let err = ScriptError::ExecutionError {
            path: "wire.rhai".to_string(),
            message: "runtime error".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("script execution error"));
        assert!(msg.contains("wire.rhai"));
    }

    #[test]
    fn test_panic_display() {
        let err = ScriptError::Panic {
            path: "script.rhai".to_string(),
            message: "index out of bounds".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("script panicked"));
        assert!(msg.contains("index out of bounds"));
    }

    #[test]
    fn test_task_join_error_display() {
        let err = ScriptError::TaskJoinError {
            path: "script.rhai".to_string(),
            message: "task was cancelled".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("script task join error"));
    }

    #[test]
    fn test_missing_function_display() {
        let err = ScriptError::MissingFunction {
            path: "script.rhai".to_string(),
            function: "format_request".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("missing required function 'format_request'"));
        assert!(msg.contains("script.rhai"));
    }

    #[test]
    fn test_conversion_error_display() {
        let err = ScriptError::ConversionError {
            message: "cannot represent as JSON".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("conversion error"));
        assert!(msg.contains("cannot represent as JSON"));
    }

    #[test]
    fn test_script_path_returns_some_for_path_variants() {
        let variants: Vec<ScriptError> = vec![
            ScriptError::FileNotFound {
                path: "a.rhai".to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, ""),
            },
            ScriptError::CompilationFailed {
                path: "b.rhai".to_string(),
                message: String::new(),
            },
            ScriptError::ExecutionError {
                path: "c.rhai".to_string(),
                message: String::new(),
            },
            ScriptError::Panic {
                path: "d.rhai".to_string(),
                message: String::new(),
            },
            ScriptError::TaskJoinError {
                path: "e.rhai".to_string(),
                message: String::new(),
            },
            ScriptError::MissingFunction {
                path: "f.rhai".to_string(),
                function: "fn_name".to_string(),
            },
        ];

        for err in &variants {
            assert!(
                err.script_path().is_some(),
                "expected Some for variant: {err:?}"
            );
        }
    }

    #[test]
    fn test_script_path_returns_none_for_conversion_error() {
        let err = ScriptError::ConversionError {
            message: "test".to_string(),
        };
        assert!(err.script_path().is_none());
    }
}
