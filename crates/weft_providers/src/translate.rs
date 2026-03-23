//! Shared translation types used by all vendor wire format modules.

/// Errors that can occur during wire format translation.
///
/// These are distinct from `ProviderError` (which covers HTTP/transport failures)
/// and `WeftError` (which covers domain-level errors). Translation errors indicate
/// that the wire format data could not be mapped to/from domain types.
#[derive(Debug, thiserror::Error)]
pub enum TranslationError {
    #[error("unrecognized role: {0}")]
    UnrecognizedRole(String),
    #[error("missing required field: {0}")]
    MissingField(String),
}
