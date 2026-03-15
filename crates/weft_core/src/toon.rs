//! TOON (Token-Oriented Object Notation) serializer and argument parser.
//!
//! This module provides two functions:
//! - `serialize_table`: produces TOON table output from labeled tabular data
//! - `parse_toon_args`: parses TOON inline key-value pairs into `serde_json::Value`
//!
//! TOON is used for everything the LLM reads or produces in the context window.
//! It saves 30-60% tokens over JSON for flat/repetitive data.
//!
//! JSON remains on the wire (gRPC, HTTP) and in config files (TOML).

/// Errors from TOON argument parsing.
#[derive(Debug, thiserror::Error)]
pub enum ToonParseError {
    #[error("missing value for key: {key}")]
    MissingValue { key: String },
    #[error("unterminated quoted string")]
    UnterminatedString,
    #[error("unterminated array")]
    UnterminatedArray,
    #[error("empty key")]
    EmptyKey,
}

/// Serialize a labeled table of rows into TOON format.
///
/// `label` is the section header (e.g., "Available commands:" or "[Command Results]").
/// `headers` are the column names.
/// `rows` are the data rows; each row is a Vec of string values.
///
/// Values containing commas, quotes, or newlines are automatically quoted.
/// Values that parse as numbers or booleans are left unquoted.
///
/// # Example
///
/// ```
/// use weft_core::toon::serialize_table;
///
/// let output = serialize_table(
///     "Available commands:",
///     &["name", "description"],
///     &[
///         vec!["web_search".into(), "Search, find, retrieve information".into()],
///     ],
/// );
/// assert!(output.contains("web_search"));
/// // Descriptions with commas are quoted
/// assert!(output.contains("\"Search, find, retrieve information\""));
/// ```
pub fn serialize_table(label: &str, headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();

    out.push_str(label);
    out.push('\n');

    // Header row
    let header_line: Vec<String> = headers.iter().map(|h| quote_value(h)).collect();
    out.push_str(&header_line.join(", "));
    out.push('\n');

    // Data rows
    for row in rows {
        let cells: Vec<String> = row.iter().map(|v| quote_value(v)).collect();
        out.push_str(&cells.join(", "));
        out.push('\n');
    }

    out
}

/// Quote a TOON value if it contains special characters.
///
/// Values containing commas, double quotes, or newlines are wrapped in double quotes.
/// Internal double quotes are escaped as `""` (CSV-style escaping).
/// Newlines within quoted values are preserved as literal `\n` sequences.
/// Values that are purely numeric or boolean (`true`/`false`) are left unquoted.
/// Empty strings are represented as `""`.
pub(crate) fn quote_value(value: &str) -> String {
    // Empty string must be quoted
    if value.is_empty() {
        return "\"\"".to_string();
    }

    // Numbers and booleans stay unquoted
    if is_unquoted_literal(value) {
        return value.to_string();
    }

    // Check if quoting is needed
    let needs_quoting = value.contains(',') || value.contains('"') || value.contains('\n');
    if !needs_quoting {
        return value.to_string();
    }

    // Wrap in double quotes, escape internal double quotes, replace newlines with \n sequences
    let escaped = value.replace('"', "\"\"").replace('\n', r"\n");
    format!("\"{}\"", escaped)
}

/// Returns true if the value should remain unquoted (numbers and booleans).
fn is_unquoted_literal(value: &str) -> bool {
    value == "true"
        || value == "false"
        || value.parse::<i64>().is_ok()
        || value.parse::<f64>().is_ok()
}

/// Parse TOON inline key-value pairs into a JSON Value object.
///
/// Input: `query: "Rust async patterns", max_results: 10`
/// Output: `{"query": "Rust async patterns", "max_results": 10}`
///
/// Returns `Value::Object` on success.
/// Empty input returns an empty object (valid for no-arg commands).
///
/// # Example
///
/// ```
/// use weft_core::toon::parse_toon_args;
/// use serde_json::json;
///
/// let result = parse_toon_args(r#"query: "Rust async", max_results: 10"#).unwrap();
/// assert_eq!(result, json!({"query": "Rust async", "max_results": 10}));
///
/// let empty = parse_toon_args("").unwrap();
/// assert_eq!(empty, json!({}));
/// ```
pub fn parse_toon_args(input: &str) -> Result<serde_json::Value, ToonParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }

    let mut map = serde_json::Map::new();

    // Split on top-level commas (not inside quotes or brackets)
    let pairs = split_top_level(trimmed)?;

    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        // Split on first colon only
        let colon_pos = pair.find(':').ok_or_else(|| ToonParseError::MissingValue {
            key: pair.to_string(),
        })?;
        let key = pair[..colon_pos].trim();
        let raw_value = pair[colon_pos + 1..].trim();

        if key.is_empty() {
            return Err(ToonParseError::EmptyKey);
        }

        if raw_value.is_empty() {
            return Err(ToonParseError::MissingValue {
                key: key.to_string(),
            });
        }

        let value = parse_toon_value(raw_value)?;
        map.insert(key.to_string(), value);
    }

    Ok(serde_json::Value::Object(map))
}

/// Split a TOON argument string on top-level commas (not inside quotes or brackets).
fn split_top_level(input: &str) -> Result<Vec<String>, ToonParseError> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut depth = 0usize; // bracket depth
    let mut in_quote = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_quote => {
                in_quote = true;
                current.push(ch);
            }
            '"' if in_quote => {
                // Check for escaped quote: ""
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                    current.push('"');
                } else {
                    in_quote = false;
                    current.push(ch);
                }
            }
            '[' if !in_quote => {
                depth += 1;
                current.push(ch);
            }
            ']' if !in_quote => {
                if depth == 0 {
                    return Err(ToonParseError::UnterminatedArray);
                }
                depth -= 1;
                current.push(ch);
            }
            ',' if !in_quote && depth == 0 => {
                parts.push(current.clone());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if in_quote {
        return Err(ToonParseError::UnterminatedString);
    }
    if depth > 0 {
        return Err(ToonParseError::UnterminatedArray);
    }

    // Push the last segment
    parts.push(current);

    Ok(parts)
}

/// Parse a single TOON value (the part after the colon in a key-value pair).
fn parse_toon_value(raw: &str) -> Result<serde_json::Value, ToonParseError> {
    let raw = raw.trim();

    // Quoted string
    if raw.starts_with('"') {
        return parse_quoted_string(raw);
    }

    // Array
    if raw.starts_with('[') {
        return parse_toon_array(raw);
    }

    // Null
    if raw == "null" {
        return Ok(serde_json::Value::Null);
    }

    // Boolean
    if raw == "true" {
        return Ok(serde_json::Value::Bool(true));
    }
    if raw == "false" {
        return Ok(serde_json::Value::Bool(false));
    }

    // Integer (must check before float to avoid treating integers as floats)
    if let Ok(n) = raw.parse::<i64>() {
        return Ok(serde_json::Value::Number(n.into()));
    }

    // Float
    if let Ok(f) = raw.parse::<f64>()
        && let Some(num) = serde_json::Number::from_f64(f)
    {
        return Ok(serde_json::Value::Number(num));
    }

    // Unquoted string (everything else, including paths, URLs, etc.)
    Ok(serde_json::Value::String(raw.to_string()))
}

/// Parse a quoted TOON string value, handling CSV-style `""` escaping.
fn parse_quoted_string(raw: &str) -> Result<serde_json::Value, ToonParseError> {
    // Must start and end with '"'
    if !raw.starts_with('"') {
        return Err(ToonParseError::UnterminatedString);
    }

    let inner = &raw[1..]; // strip leading quote
    let mut result = String::new();
    let mut chars = inner.chars().peekable();
    let mut closed = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if chars.peek() == Some(&'"') {
                    // Escaped quote: ""  → "
                    chars.next();
                    result.push('"');
                } else {
                    // Closing quote
                    closed = true;
                    break;
                }
            }
            _ => result.push(ch),
        }
    }

    if !closed {
        return Err(ToonParseError::UnterminatedString);
    }

    Ok(serde_json::Value::String(result))
}

/// Parse a TOON array value: `[val1, val2, "val with spaces"]`.
fn parse_toon_array(raw: &str) -> Result<serde_json::Value, ToonParseError> {
    let raw = raw.trim();
    if !raw.starts_with('[') || !raw.ends_with(']') {
        return Err(ToonParseError::UnterminatedArray);
    }

    let inner = &raw[1..raw.len() - 1];
    if inner.trim().is_empty() {
        return Ok(serde_json::Value::Array(vec![]));
    }

    // Split array elements on top-level commas
    let elements = split_top_level(inner)?;
    let mut values = Vec::new();
    for elem in elements {
        let elem = elem.trim();
        if elem.is_empty() {
            continue;
        }
        values.push(parse_toon_value(elem)?);
    }

    Ok(serde_json::Value::Array(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── serialize_table tests ─────────────────────────────────────────────

    #[test]
    fn test_serialize_table_basic() {
        let output = serialize_table(
            "Available commands:",
            &["name", "description"],
            &[
                vec![
                    "web_search".into(),
                    "Search the web for current information".into(),
                ],
                vec!["code_review".into(), "Review code for issues".into()],
            ],
        );
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "Available commands:");
        assert_eq!(lines[1], "name, description");
        assert!(lines[2].contains("web_search"));
        assert!(lines[3].contains("code_review"));
    }

    #[test]
    fn test_serialize_table_quotes_descriptions_with_commas() {
        let output = serialize_table(
            "Commands:",
            &["name", "description"],
            &[vec!["cmd".into(), "Do X, Y, and Z".into()]],
        );
        // The description contains commas, so it must be quoted
        assert!(output.contains("\"Do X, Y, and Z\""));
    }

    #[test]
    fn test_serialize_table_quotes_values_with_double_quotes() {
        let output = serialize_table("Test:", &["col"], &[vec!["value with \"quotes\"".into()]]);
        // Internal quotes should be escaped as ""
        assert!(output.contains("\"\"quotes\"\""));
    }

    #[test]
    fn test_serialize_table_newlines_in_values() {
        let output = serialize_table("Test:", &["col"], &[vec!["line one\nline two".into()]]);
        // Newlines become literal \n sequences inside quoted value
        assert!(output.contains(r#""line one\nline two""#));
        // The table itself should not contain actual newlines inside cells
        let lines: Vec<&str> = output.lines().collect();
        // Should be 3 lines: label, header, data row
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_serialize_table_numeric_and_bool_unquoted() {
        let output = serialize_table(
            "Test:",
            &["num", "flag"],
            &[vec!["42".into(), "true".into()]],
        );
        // Numbers and booleans should not be quoted
        assert!(output.contains("42"));
        assert!(output.contains("true"));
        assert!(!output.contains("\"42\""));
        assert!(!output.contains("\"true\""));
    }

    #[test]
    fn test_serialize_table_empty_string_quoted() {
        let output = serialize_table("Test:", &["col"], &[vec![String::new()]]);
        assert!(output.contains("\"\""));
    }

    #[test]
    fn test_serialize_table_headers_only_no_rows() {
        let output = serialize_table("Commands:", &["name", "description"], &[]);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2); // label + header row only
        assert_eq!(lines[0], "Commands:");
        assert_eq!(lines[1], "name, description");
    }

    #[test]
    fn test_serialize_table_command_results_format() {
        let output = serialize_table(
            "[Command Results]",
            &["command", "status", "output"],
            &[vec![
                "web_search".into(),
                "success".into(),
                "Found 3 results".into(),
            ]],
        );
        assert!(output.starts_with("[Command Results]"));
        assert!(output.contains("command, status, output"));
        assert!(output.contains("web_search"));
        assert!(output.contains("success"));
    }

    // ── quote_value tests ─────────────────────────────────────────────────

    #[test]
    fn test_quote_value_simple_string() {
        assert_eq!(quote_value("hello"), "hello");
        assert_eq!(quote_value("web_search"), "web_search");
        assert_eq!(quote_value("src/main.rs"), "src/main.rs");
    }

    #[test]
    fn test_quote_value_number() {
        assert_eq!(quote_value("42"), "42");
        assert_eq!(quote_value("3.14"), "3.14");
        assert_eq!(quote_value("-1"), "-1");
    }

    #[test]
    fn test_quote_value_boolean() {
        assert_eq!(quote_value("true"), "true");
        assert_eq!(quote_value("false"), "false");
    }

    #[test]
    fn test_quote_value_with_comma() {
        assert_eq!(quote_value("a, b"), "\"a, b\"");
    }

    #[test]
    fn test_quote_value_with_double_quote() {
        assert_eq!(quote_value("say \"hello\""), "\"say \"\"hello\"\"\"");
    }

    #[test]
    fn test_quote_value_empty() {
        assert_eq!(quote_value(""), "\"\"");
    }

    // ── parse_toon_args tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_empty() {
        let result = parse_toon_args("").unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = parse_toon_args("   ").unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn test_parse_single_string() {
        let result = parse_toon_args(r#"query: "Rust async patterns""#).unwrap();
        assert_eq!(result, json!({"query": "Rust async patterns"}));
    }

    #[test]
    fn test_parse_integer() {
        let result = parse_toon_args("max_results: 10").unwrap();
        assert_eq!(result, json!({"max_results": 10}));
    }

    #[test]
    fn test_parse_negative_integer() {
        let result = parse_toon_args("offset: -5").unwrap();
        assert_eq!(result, json!({"offset": -5}));
    }

    #[test]
    fn test_parse_float() {
        let result = parse_toon_args("threshold: 0.5").unwrap();
        // Compare as f64 due to floating point representation
        let val = result.get("threshold").unwrap().as_f64().unwrap();
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_boolean_true() {
        let result = parse_toon_args("verbose: true").unwrap();
        assert_eq!(result, json!({"verbose": true}));
    }

    #[test]
    fn test_parse_boolean_false() {
        let result = parse_toon_args("verbose: false").unwrap();
        assert_eq!(result, json!({"verbose": false}));
    }

    #[test]
    fn test_parse_null() {
        let result = parse_toon_args("key: null").unwrap();
        assert_eq!(result, json!({"key": null}));
    }

    #[test]
    fn test_parse_multiple_pairs() {
        let result =
            parse_toon_args(r#"query: "Rust async patterns", max_results: 10, verbose: true"#)
                .unwrap();
        assert_eq!(
            result,
            json!({"query": "Rust async patterns", "max_results": 10, "verbose": true})
        );
    }

    #[test]
    fn test_parse_array_simple() {
        let result = parse_toon_args("tags: [ml, ai, rust]").unwrap();
        assert_eq!(result, json!({"tags": ["ml", "ai", "rust"]}));
    }

    #[test]
    fn test_parse_array_with_quoted_elements() {
        let result = parse_toon_args(r#"tags: [ml, ai, "deep learning"]"#).unwrap();
        assert_eq!(result, json!({"tags": ["ml", "ai", "deep learning"]}));
    }

    #[test]
    fn test_parse_array_mixed_types() {
        let result = parse_toon_args(r#"data: [42, "hello", true]"#).unwrap();
        assert_eq!(result, json!({"data": [42, "hello", true]}));
    }

    #[test]
    fn test_parse_array_with_limit() {
        let result = parse_toon_args(r#"tags: [ml, ai, "deep learning"], limit: 5"#).unwrap();
        assert_eq!(
            result,
            json!({"tags": ["ml", "ai", "deep learning"], "limit": 5})
        );
    }

    #[test]
    fn test_parse_trailing_comma_tolerated() {
        let result = parse_toon_args("key: value,").unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_parse_escaped_quotes_in_string() {
        // "she said ""hello"""  -> she said "hello"
        let result = parse_toon_args(r#"message: "she said ""hello""""#).unwrap();
        assert_eq!(result, json!({"message": r#"she said "hello""#}));
    }

    #[test]
    fn test_parse_unquoted_simple_string() {
        let result = parse_toon_args("format: json").unwrap();
        assert_eq!(result, json!({"format": "json"}));
    }

    #[test]
    fn test_parse_unquoted_path() {
        let result = parse_toon_args("path: src/main.rs").unwrap();
        assert_eq!(result, json!({"path": "src/main.rs"}));
    }

    #[test]
    fn test_parse_url_unquoted_colon_in_value() {
        // url: https://example.com — parser splits on first colon only
        // so key=url, value=https (rest after second colon treated as part of value? No.)
        // Actually: split on first ':' gives key="url", value="https://example.com"
        // Wait — find(':') finds the first colon. So key="url", value="https://example.com"
        // But the raw value is "https", then we have "//example.com"... let's check.
        // pair = "url: https://example.com"
        // colon_pos = 3 (the colon after "url")
        // key = "url"
        // raw_value = " https://example.com"  -> trim -> "https://example.com"
        // parse_toon_value("https://example.com") -> unquoted string
        let result = parse_toon_args("url: https://example.com").unwrap();
        assert_eq!(result, json!({"url": "https://example.com"}));
    }

    #[test]
    fn test_parse_error_missing_value() {
        let result = parse_toon_args("key:");
        assert!(matches!(result, Err(ToonParseError::MissingValue { .. })));
    }

    #[test]
    fn test_parse_error_unterminated_string() {
        let result = parse_toon_args(r#"key: "unterminated"#);
        assert!(matches!(result, Err(ToonParseError::UnterminatedString)));
    }

    #[test]
    fn test_parse_error_unterminated_array() {
        let result = parse_toon_args("key: [a, b");
        assert!(matches!(result, Err(ToonParseError::UnterminatedArray)));
    }

    #[test]
    fn test_parse_empty_array() {
        let result = parse_toon_args("items: []").unwrap();
        assert_eq!(result, json!({"items": []}));
    }

    #[test]
    fn test_spec_example_round_trip() {
        // From spec Section 4.7
        let result =
            parse_toon_args(r#"query: "Rust async patterns", max_results: 10, verbose: true"#)
                .unwrap();
        assert_eq!(
            result,
            json!({"query": "Rust async patterns", "max_results": 10, "verbose": true})
        );
    }

    #[test]
    fn test_spec_example_tags_and_limit() {
        // From spec Section 4.7
        let result = parse_toon_args(r#"tags: [ml, ai, "deep learning"], limit: 5"#).unwrap();
        assert_eq!(
            result,
            json!({"tags": ["ml", "ai", "deep learning"], "limit": 5})
        );
    }
}
