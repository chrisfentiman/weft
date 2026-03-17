//! Dynamic <-> JSON conversion utilities.
//!
//! Provides zero-panic conversion from `serde_json::Value` to Rhai `Dynamic`,
//! and fallible conversion from `Dynamic` back to `serde_json::Value`.

use rhai::Dynamic;
use serde_json::Value;

use crate::error::ScriptError;

/// Convert a `serde_json::Value` to a Rhai `Dynamic`.
///
/// Returns `Dynamic::UNIT` on conversion failure. This should not happen
/// for well-formed JSON but we never panic on conversion.
pub fn json_to_dynamic(value: &Value) -> Dynamic {
    rhai::serde::to_dynamic(value).unwrap_or(Dynamic::UNIT)
}

/// Convert a Rhai `Dynamic` to a `serde_json::Value`.
///
/// Returns `Err(ScriptError::ConversionError)` if the Dynamic cannot
/// be represented as JSON.
pub fn dynamic_to_json(dynamic: &Dynamic) -> Result<Value, ScriptError> {
    rhai::serde::from_dynamic(dynamic).map_err(|e| ScriptError::ConversionError {
        message: format!("Dynamic to JSON conversion failed: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_to_dynamic_object_is_not_unit() {
        let json = json!({"key": "value", "num": 42});
        let dynamic = json_to_dynamic(&json);
        assert!(!dynamic.is_unit(), "object should not convert to UNIT");
    }

    #[test]
    fn test_json_to_dynamic_string() {
        let json = json!("hello");
        let dynamic = json_to_dynamic(&json);
        assert!(!dynamic.is_unit());
    }

    #[test]
    fn test_json_to_dynamic_number() {
        let json = json!(42);
        let dynamic = json_to_dynamic(&json);
        assert!(!dynamic.is_unit());
    }

    #[test]
    fn test_json_to_dynamic_null_does_not_panic() {
        let json = Value::Null;
        // Should not panic — just returns something (UNIT is acceptable).
        let _dynamic = json_to_dynamic(&json);
    }

    #[test]
    fn test_json_to_dynamic_array_is_not_unit() {
        let json = json!([1, 2, 3]);
        let dynamic = json_to_dynamic(&json);
        assert!(!dynamic.is_unit());
    }

    #[test]
    fn test_json_to_dynamic_bool() {
        let json = json!(true);
        let dynamic = json_to_dynamic(&json);
        assert!(!dynamic.is_unit());
    }

    #[test]
    fn test_dynamic_to_json_round_trip_object() {
        let original = json!({"key": "value", "nested": {"a": 1}});
        let dynamic = json_to_dynamic(&original);
        let result = dynamic_to_json(&dynamic).expect("round-trip should succeed");
        assert_eq!(result, original);
    }

    #[test]
    fn test_dynamic_to_json_round_trip_string() {
        let original = json!("hello world");
        let dynamic = json_to_dynamic(&original);
        let result = dynamic_to_json(&dynamic).expect("round-trip should succeed");
        assert_eq!(result, original);
    }

    #[test]
    fn test_dynamic_to_json_round_trip_number() {
        let original = json!(123);
        let dynamic = json_to_dynamic(&original);
        let result = dynamic_to_json(&dynamic).expect("round-trip should succeed");
        assert_eq!(result, original);
    }

    #[test]
    fn test_dynamic_to_json_round_trip_array() {
        let original = json!([1, "two", true]);
        let dynamic = json_to_dynamic(&original);
        let result = dynamic_to_json(&dynamic).expect("round-trip should succeed");
        assert_eq!(result, original);
    }

    #[test]
    fn test_dynamic_to_json_map_produces_object() {
        let original = json!({"a": 1, "b": "two"});
        let dynamic = json_to_dynamic(&original);
        let result = dynamic_to_json(&dynamic).expect("should succeed");
        assert!(result.is_object());
    }

    #[test]
    fn test_dynamic_to_json_unit_returns_null() {
        // Dynamic::UNIT maps to JSON null
        let result = dynamic_to_json(&Dynamic::UNIT).expect("UNIT should convert to null");
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_dynamic_to_json_integer() {
        let dynamic = Dynamic::from(42_i64);
        let result = dynamic_to_json(&dynamic).expect("integer should convert");
        assert_eq!(result, json!(42));
    }
}
