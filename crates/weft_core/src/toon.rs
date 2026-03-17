//! TOON (Token-Oriented Object Notation) parser and serializer.
//!
//! This module provides a full TOON implementation following the toonformat.dev spec:
//! - Types: `ToonDocument`, `ToonNode`, `ToonValue`
//! - Serializer: `serialize_document`, `serialize_node`, `serialize_value`, `serialize_table`, `fenced_toon`
//! - Parser: `parse_document`, `parse_value`, `extract_fenced_toon`
//! - Inline args: `parse_toon_args` (preserved for slash command argument parsing)
//! - JSON interop: `From<ToonValue> for serde_json::Value`, `From<ToonDocument> for serde_json::Value`
//!
//! TOON is used for everything the LLM reads or produces in the context window.
//! It saves 30-60% tokens over JSON for flat/repetitive data.
//!
//! JSON remains on the wire (gRPC, HTTP) and in config files (TOML).

// ── Types ─────────────────────────────────────────────────────────────────────

/// A TOON document is a sequence of nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct ToonDocument {
    pub nodes: Vec<ToonNode>,
}

/// A single node in a TOON document.
#[derive(Debug, Clone, PartialEq)]
pub enum ToonNode {
    /// A section with a label and nested content.
    /// `label:` followed by indented child nodes.
    Section {
        label: String,
        children: Vec<ToonNode>,
    },
    /// A typed array with column headers and rows.
    /// `label[N]{field1, field2}:` followed by indented rows.
    TypedArray {
        label: String,
        fields: Vec<String>,
        rows: Vec<Vec<ToonValue>>,
    },
    /// A key-value pair: `key: value`
    KeyValue { key: String, value: ToonValue },
    /// A text block -- free-form text content within a section.
    Text(String),
}

/// A TOON value.
#[derive(Debug, Clone, PartialEq)]
pub enum ToonValue {
    /// A string value. May be quoted or unquoted in serialized form.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A null value.
    Null,
    /// An inline array value.
    Array(Vec<ToonValue>),
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors from TOON parsing.
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
    #[error("invalid typed array header: {0}")]
    InvalidArrayHeader(String),
    #[error("row field count mismatch: expected {expected}, got {got}")]
    FieldCountMismatch { expected: usize, got: usize },
    #[error("invalid indentation at line {line}")]
    InvalidIndentation { line: usize },
}

// ── JSON Interop ──────────────────────────────────────────────────────────────

/// Convert a `ToonValue` to a `serde_json::Value`.
///
/// Used by `parse_toon_args` to maintain backward compatibility with the
/// command execution pipeline.
impl From<ToonValue> for serde_json::Value {
    fn from(tv: ToonValue) -> Self {
        match tv {
            ToonValue::String(s) => serde_json::Value::String(s),
            ToonValue::Integer(n) => serde_json::Value::Number(n.into()),
            ToonValue::Float(f) => serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            ToonValue::Bool(b) => serde_json::Value::Bool(b),
            ToonValue::Null => serde_json::Value::Null,
            ToonValue::Array(arr) => {
                serde_json::Value::Array(arr.into_iter().map(Into::into).collect())
            }
        }
    }
}

/// Convert a `ToonDocument` to `serde_json::Value` for interop.
///
/// Mapping rules:
///
/// 1. `ToonDocument` -> `Value::Object`. Each top-level node is a key.
/// 2. `Section` -> nested `Value::Object` under the section label.
/// 3. `TypedArray` -> `Value::Array` of `Value::Object` (one per row, fields as keys).
/// 4. `KeyValue` -> field in parent object.
/// 5. `Text` -> `Value::String` under key `"_text"`.
/// 6. Duplicate keys: last value wins.
impl From<ToonDocument> for serde_json::Value {
    fn from(doc: ToonDocument) -> Self {
        let mut map = serde_json::Map::new();
        for node in doc.nodes {
            match node {
                ToonNode::Section { label, children } => {
                    let child_doc = ToonDocument { nodes: children };
                    map.insert(label, child_doc.into());
                }
                ToonNode::TypedArray {
                    label,
                    fields,
                    rows,
                } => {
                    let arr: Vec<serde_json::Value> = rows
                        .into_iter()
                        .map(|row| {
                            let mut obj = serde_json::Map::new();
                            for (field, val) in fields.iter().zip(row) {
                                obj.insert(field.clone(), val.into());
                            }
                            serde_json::Value::Object(obj)
                        })
                        .collect();
                    map.insert(label, serde_json::Value::Array(arr));
                }
                ToonNode::KeyValue { key, value } => {
                    map.insert(key, value.into());
                }
                ToonNode::Text(text) => {
                    map.insert("_text".to_string(), serde_json::Value::String(text));
                }
            }
        }
        serde_json::Value::Object(map)
    }
}

// ── Serializer ────────────────────────────────────────────────────────────────

/// Serialize a TOON document to string.
pub fn serialize_document(doc: &ToonDocument) -> String {
    let mut out = String::new();
    for node in &doc.nodes {
        out.push_str(&serialize_node(node, 0));
    }
    out
}

/// Serialize a single node to string with the given indentation level.
pub fn serialize_node(node: &ToonNode, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    let mut out = String::new();
    match node {
        ToonNode::Section { label, children } => {
            out.push_str(&format!("{prefix}{label}:\n"));
            for child in children {
                out.push_str(&serialize_node(child, indent + 1));
            }
        }
        ToonNode::TypedArray {
            label,
            fields,
            rows,
        } => {
            let fields_str = fields.join(", ");
            out.push_str(&format!(
                "{prefix}{}[{}]{{{}}}:\n",
                label,
                rows.len(),
                fields_str
            ));
            for row in rows {
                let cells: Vec<String> = row.iter().map(serialize_value).collect();
                out.push_str(&format!("{}  {}\n", prefix, cells.join(", ")));
            }
        }
        ToonNode::KeyValue { key, value } => {
            out.push_str(&format!("{prefix}{key}: {}\n", serialize_value(value)));
        }
        ToonNode::Text(text) => {
            // Each line of the text block gets the prefix
            for line in text.lines() {
                out.push_str(&format!("{prefix}{line}\n"));
            }
        }
    }
    out
}

/// Serialize a TOON value to its string representation.
pub fn serialize_value(value: &ToonValue) -> String {
    match value {
        ToonValue::String(s) => quote_value(s),
        ToonValue::Integer(n) => n.to_string(),
        ToonValue::Float(f) => f.to_string(),
        ToonValue::Bool(b) => b.to_string(),
        ToonValue::Null => "null".to_string(),
        ToonValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(serialize_value).collect();
            format!("[{}]", items.join(", "))
        }
    }
}

/// Serialize a labeled table of rows into TOON typed array format.
///
/// `label` is a bare noun (e.g., `"commands"`, `"results"`).
/// `headers` are the column field names.
/// `rows` are the data rows; each row is a `Vec` of string values.
///
/// Produces TOON array syntax: `label[N]{field1, field2}:` followed by
/// 2-space indented rows.
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
///     "commands",
///     &["name", "description"],
///     &[
///         vec!["web_search".into(), "Search the web".into()],
///         vec!["recall".into(), "Retrieve memory".into()],
///     ],
/// );
/// assert_eq!(
///     output,
///     "commands[2]{name, description}:\n  web_search, Search the web\n  recall, Retrieve memory\n"
/// );
/// ```
pub fn serialize_table(label: &str, headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();

    // TOON array header: label[N]{field1, field2}:
    let header_str = headers.join(", ");
    out.push_str(&format!("{label}[{}]{{{header_str}}}:\n", rows.len()));

    // 2-space indented data rows
    for row in rows {
        let cells: Vec<String> = row.iter().map(|v| quote_value(v)).collect();
        out.push_str("  ");
        out.push_str(&cells.join(", "));
        out.push('\n');
    }

    out
}

/// Wrap TOON content in a fenced code block for embedding in other formats.
///
/// Produces:
/// ````text
/// ```toon
/// {content}
/// ```
/// ````
///
/// The content should already be valid TOON (e.g., output from `serialize_table`).
pub fn fenced_toon(content: &str) -> String {
    if content.ends_with('\n') {
        format!("```toon\n{content}```")
    } else {
        format!("```toon\n{content}\n```")
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse a TOON document from a string.
///
/// Handles sections, key-value pairs, typed arrays, and text blocks.
/// Returns a `ToonDocument` containing the parsed nodes.
pub fn parse_document(input: &str) -> Result<ToonDocument, ToonParseError> {
    let nodes = parse_nodes(input, 0)?;
    Ok(ToonDocument { nodes })
}

/// Parse TOON inline key-value pairs into a JSON `Value` object.
///
/// This is the existing function, preserved for slash command argument
/// parsing. The JSON output format is required by the command execution
/// pipeline (commands receive `serde_json::Value` arguments).
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

        let value: serde_json::Value = parse_value(raw_value)?.into();
        map.insert(key.to_string(), value);
    }

    Ok(serde_json::Value::Object(map))
}

/// Parse a single TOON value from a string.
///
/// Handles quoted strings, inline arrays, booleans, null, integers, floats,
/// and bare (unquoted) strings.
pub fn parse_value(input: &str) -> Result<ToonValue, ToonParseError> {
    let raw = input.trim();

    // Quoted string
    if raw.starts_with('"') {
        return parse_quoted_toon_string(raw);
    }

    // Array
    if raw.starts_with('[') {
        return parse_toon_array_value(raw);
    }

    // Null
    if raw == "null" {
        return Ok(ToonValue::Null);
    }

    // Boolean
    if raw == "true" {
        return Ok(ToonValue::Bool(true));
    }
    if raw == "false" {
        return Ok(ToonValue::Bool(false));
    }

    // Integer (must check before float)
    if let Ok(n) = raw.parse::<i64>() {
        return Ok(ToonValue::Integer(n));
    }

    // Float
    if let Ok(f) = raw.parse::<f64>() {
        return Ok(ToonValue::Float(f));
    }

    // Bare string (paths, URLs, identifiers, etc.)
    Ok(ToonValue::String(raw.to_string()))
}

/// Extract TOON content from fenced blocks in a larger text.
///
/// Finds all ` ```toon ` ... ` ``` ` blocks and parses each as a `ToonDocument`.
/// Returns an error if any fenced block has no closing fence or is unparseable.
pub fn extract_fenced_toon(input: &str) -> Result<Vec<ToonDocument>, ToonParseError> {
    let mut docs = Vec::new();
    let mut rest = input;

    loop {
        // Find opening fence
        let Some(open_pos) = rest.find("```toon") else {
            break;
        };
        let after_open = &rest[open_pos + "```toon".len()..];

        // Skip the rest of the opening fence line (there may be trailing spaces)
        let content_start = after_open
            .find('\n')
            .ok_or(ToonParseError::UnterminatedString)?;
        let content = &after_open[content_start + 1..];

        // Find closing fence (must be ``` on its own, possibly preceded by whitespace)
        let close_pos = find_closing_fence(content).ok_or(ToonParseError::UnterminatedString)?;

        let block = &content[..close_pos];
        docs.push(parse_document(block)?);

        // Advance past the closing fence
        let after_close_line_end = content[close_pos + 3..]
            .find('\n')
            .map(|p| close_pos + 3 + p + 1)
            .unwrap_or(close_pos + 3);
        let consumed = open_pos + "```toon".len() + content_start + 1 + after_close_line_end;
        rest = &rest[consumed..];
    }

    Ok(docs)
}

// ── Internal Parser Helpers ───────────────────────────────────────────────────

/// Find the position of the ` ``` ` closing fence in `content`.
/// The fence must be three backticks at the start of a line (after optional whitespace).
fn find_closing_fence(content: &str) -> Option<usize> {
    let mut pos = 0;
    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            return Some(pos);
        }
        pos += line.len() + 1; // +1 for '\n'
    }
    None
}

/// Determine the indentation level (in units of 2 spaces) of a line.
/// Returns `None` if the line is empty or whitespace-only.
fn indent_level(line: &str) -> Option<usize> {
    if line.trim().is_empty() {
        return None;
    }
    let spaces = line.len() - line.trim_start_matches(' ').len();
    Some(spaces / 2)
}

/// Try to match a typed array header: `label[N]{field1, field2}:`
/// Returns `(label, fields)` on success.
fn try_parse_array_header(line: &str) -> Option<(String, Vec<String>)> {
    // Pattern: word chars + '[' + digits + ']{' + fields + '}:'
    let trimmed = line.trim();
    let bracket_pos = trimmed.find('[')?;
    let label = &trimmed[..bracket_pos];
    if label.is_empty() {
        return None;
    }

    let after_bracket = &trimmed[bracket_pos + 1..];
    let close_bracket = after_bracket.find(']')?;
    // The count is informational; we don't enforce it at parse time
    let _count_str = &after_bracket[..close_bracket];

    let after_count = &after_bracket[close_bracket + 1..];
    if !after_count.starts_with('{') {
        return None;
    }
    let close_brace = after_count.find('}')?;
    let fields_str = &after_count[1..close_brace];
    let after_brace = &after_count[close_brace + 1..];
    if after_brace != ":" {
        return None;
    }

    let fields: Vec<String> = fields_str
        .split(',')
        .map(|f| f.trim().to_string())
        .filter(|f| !f.is_empty())
        .collect();

    if fields.is_empty() {
        return None;
    }

    Some((label.to_string(), fields))
}

/// Check whether a line (already stripped of its indent) looks like a key-value pair.
/// A line is kv if it contains ':' with a non-empty key before it and a non-empty value after.
fn is_kv_line(line: &str) -> bool {
    let Some(colon) = line.find(':') else {
        return false;
    };
    let key = line[..colon].trim();
    let value = line[colon + 1..].trim();
    !key.is_empty() && !value.is_empty()
}

/// Check whether a line (already stripped of its indent) is a section header.
///
/// A section header ends with `:` and has a non-empty label before the colon with
/// no value after it. Labels may contain spaces (e.g., `multi word label:`).
fn is_section_header(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.ends_with(':') {
        return false;
    }
    let label = trimmed[..trimmed.len() - 1].trim();
    !label.is_empty()
}

/// Parse nodes at the given indent level from the lines of `input`.
/// Lines that belong to a deeper indent level are consumed as children.
fn parse_nodes(input: &str, base_indent: usize) -> Result<Vec<ToonNode>, ToonParseError> {
    let lines: Vec<&str> = input.lines().collect();
    parse_nodes_from_lines(&lines, base_indent, &mut 0)
}

/// Core recursive parser over a line slice with a mutable cursor.
fn parse_nodes_from_lines(
    lines: &[&str],
    base_indent: usize,
    cursor: &mut usize,
) -> Result<Vec<ToonNode>, ToonParseError> {
    let mut nodes = Vec::new();

    while *cursor < lines.len() {
        let line = lines[*cursor];

        // Skip blank lines at this level
        if line.trim().is_empty() {
            *cursor += 1;
            continue;
        }

        let level = indent_level(line).unwrap_or(0);

        // If this line is less indented than our base, we're done with this section
        if level < base_indent {
            break;
        }

        // If it's MORE indented than base, it's orphaned content — treat as text
        if level > base_indent {
            // Should be consumed by a parent section; skip
            *cursor += 1;
            continue;
        }

        // Strip the base indentation
        let stripped = line.get(base_indent * 2..).unwrap_or(line.trim_start());

        // Try typed array header
        if let Some((label, fields)) = try_parse_array_header(stripped) {
            *cursor += 1;
            // Collect rows: lines indented one more level
            let child_indent = base_indent + 1;
            let mut rows: Vec<Vec<ToonValue>> = Vec::new();
            while *cursor < lines.len() {
                let row_line = lines[*cursor];
                if row_line.trim().is_empty() {
                    *cursor += 1;
                    continue;
                }
                let row_level = indent_level(row_line).unwrap_or(0);
                if row_level < child_indent {
                    break;
                }
                let row_stripped = row_line
                    .get(child_indent * 2..)
                    .unwrap_or(row_line.trim_start());
                let row_values = parse_csv_row(row_stripped, fields.len())?;
                rows.push(row_values);
                *cursor += 1;
            }
            nodes.push(ToonNode::TypedArray {
                label,
                fields,
                rows,
            });
            continue;
        }

        // Try section header: line ends with ':' and the key has no value after colon
        // A section header is `word:` (colon at end, nothing after it)
        if let Some(colon_pos) = stripped.rfind(':')
            && colon_pos == stripped.len() - 1
        {
            // Everything before the colon is the label
            let label = stripped[..colon_pos].trim().to_string();
            if !label.is_empty() {
                *cursor += 1;
                // Collect children: lines at base_indent + 1
                let child_indent = base_indent + 1;
                let children = collect_section_children(lines, child_indent, cursor)?;
                nodes.push(ToonNode::Section { label, children });
                continue;
            }
        }

        // Try key-value pair: `key: value`
        if is_kv_line(stripped)
            && let Some(colon_pos) = stripped.find(':')
        {
            let key = stripped[..colon_pos].trim().to_string();
            let raw_val = stripped[colon_pos + 1..].trim();
            if !key.is_empty() && !raw_val.is_empty() {
                let value = parse_value(raw_val)?;
                nodes.push(ToonNode::KeyValue { key, value });
                *cursor += 1;
                continue;
            }
        }

        // Fallback: treat as a bare text line (shouldn't normally occur at top level)
        nodes.push(ToonNode::Text(stripped.to_string()));
        *cursor += 1;
    }

    Ok(nodes)
}

/// Collect all child lines at `child_indent` for a section, applying the
/// all-or-nothing text block heuristic.
fn collect_section_children(
    lines: &[&str],
    child_indent: usize,
    cursor: &mut usize,
) -> Result<Vec<ToonNode>, ToonParseError> {
    // Gather all lines that belong to this section
    let mut child_lines: Vec<&str> = Vec::new();
    while *cursor < lines.len() {
        let line = lines[*cursor];
        if line.trim().is_empty() {
            // Blank lines inside a section are included
            child_lines.push(line);
            *cursor += 1;
            continue;
        }
        let level = indent_level(line).unwrap_or(0);
        if level < child_indent {
            break;
        }
        child_lines.push(line);
        *cursor += 1;
    }

    // Remove trailing blank lines
    while child_lines
        .last()
        .map(|l: &&str| l.trim().is_empty())
        .unwrap_or(false)
    {
        child_lines.pop();
    }

    if child_lines.is_empty() {
        return Ok(vec![]);
    }

    // Apply all-or-nothing heuristic at child_indent level:
    // If ANY non-blank line at exactly child_indent fails the kv pattern AND
    // is not a typed array header, ALL lines are a text block.
    let mut has_typed_array = false;
    let mut all_kv_or_array = true;

    for line in &child_lines {
        if line.trim().is_empty() {
            continue;
        }
        let level = indent_level(line).unwrap_or(0);
        if level != child_indent {
            // Deeper lines belong to children -- don't count them for the heuristic
            continue;
        }
        let stripped = line.get(child_indent * 2..).unwrap_or(line.trim_start());

        if try_parse_array_header(stripped).is_some() {
            has_typed_array = true;
            continue;
        }

        // Section headers are structured content -- a nested `label:` should
        // not trigger the text-block fallback (Spec 11.12).
        if is_section_header(stripped) {
            continue;
        }

        if !is_kv_line(stripped) {
            all_kv_or_array = false;
            break;
        }
    }

    // If it's not all kv/array, treat everything as a text block
    // (typed arrays are exempt -- they can coexist with kv pairs)
    if !all_kv_or_array && !has_typed_array {
        // All lines become a single text block: strip indent, join with newlines
        let text_lines: Vec<String> = child_lines
            .iter()
            .map(|l| {
                if l.trim().is_empty() {
                    String::new()
                } else {
                    l.get(child_indent * 2..)
                        .unwrap_or(l.trim_start())
                        .to_string()
                }
            })
            .collect();
        let text = text_lines.join("\n");
        return Ok(vec![ToonNode::Text(text)]);
    }

    // Parse as structured content
    let mut sub_cursor = 0;
    parse_nodes_from_lines(&child_lines, child_indent, &mut sub_cursor)
}

/// Parse a comma-separated row of values for a typed array.
/// Expects exactly `expected_fields` values.
fn parse_csv_row(row: &str, expected_fields: usize) -> Result<Vec<ToonValue>, ToonParseError> {
    let parts = split_top_level(row)?;
    let values: Vec<ToonValue> = parts
        .iter()
        .map(|p| parse_value(p.trim()))
        .collect::<Result<Vec<_>, _>>()?;

    if values.len() != expected_fields {
        return Err(ToonParseError::FieldCountMismatch {
            expected: expected_fields,
            got: values.len(),
        });
    }

    Ok(values)
}

// ── Internal Value Parsers ────────────────────────────────────────────────────

/// Parse a quoted TOON string value, handling CSV-style `""` escaping.
fn parse_quoted_toon_string(raw: &str) -> Result<ToonValue, ToonParseError> {
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
                    // Escaped quote: "" -> "
                    chars.next();
                    result.push('"');
                } else {
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

    Ok(ToonValue::String(result))
}

/// Parse a TOON array value: `[val1, val2, "val with spaces"]`.
fn parse_toon_array_value(raw: &str) -> Result<ToonValue, ToonParseError> {
    let raw = raw.trim();
    if !raw.starts_with('[') || !raw.ends_with(']') {
        return Err(ToonParseError::UnterminatedArray);
    }

    let inner = &raw[1..raw.len() - 1];
    if inner.trim().is_empty() {
        return Ok(ToonValue::Array(vec![]));
    }

    let elements = split_top_level(inner)?;
    let mut values = Vec::new();
    for elem in elements {
        let elem = elem.trim();
        if elem.is_empty() {
            continue;
        }
        values.push(parse_value(elem)?);
    }

    Ok(ToonValue::Array(values))
}

// ── Quoting ───────────────────────────────────────────────────────────────────

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

    // Wrap in double quotes, escape internal double quotes, replace newlines
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

// ── Shared Low-Level Helpers ──────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── serialize_table tests (new TOON array format) ─────────────────────

    #[test]
    fn test_serialize_table_array_header_format() {
        let output = serialize_table(
            "commands",
            &["name", "description"],
            &[
                vec!["web_search".into(), "Search the web".into()],
                vec!["recall".into(), "Retrieve memory".into()],
            ],
        );
        // Header: label[N]{fields}:
        assert!(output.starts_with("commands[2]{name, description}:"));
        // Rows are 2-space indented
        assert!(output.contains("\n  web_search, Search the web\n"));
        assert!(output.contains("\n  recall, Retrieve memory\n"));
    }

    #[test]
    fn test_serialize_table_exact_output() {
        let output = serialize_table(
            "commands",
            &["name", "description"],
            &[
                vec!["web_search".into(), "Search the web".into()],
                vec!["recall".into(), "Retrieve memory".into()],
            ],
        );
        assert_eq!(
            output,
            "commands[2]{name, description}:\n  web_search, Search the web\n  recall, Retrieve memory\n"
        );
    }

    #[test]
    fn test_serialize_table_row_count_in_header() {
        let output = serialize_table("results", &["a", "b"], &[]);
        assert!(output.starts_with("results[0]{a, b}:"));
    }

    #[test]
    fn test_serialize_table_three_rows() {
        let rows = vec![
            vec!["a".to_string(), "1".to_string()],
            vec!["b".to_string(), "2".to_string()],
            vec!["c".to_string(), "3".to_string()],
        ];
        let output = serialize_table("items", &["key", "val"], &rows);
        assert!(output.starts_with("items[3]{key, val}:"));
    }

    #[test]
    fn test_serialize_table_quotes_descriptions_with_commas() {
        let output = serialize_table(
            "commands",
            &["name", "description"],
            &[vec!["cmd".into(), "Do X, Y, and Z".into()]],
        );
        assert!(output.contains("\"Do X, Y, and Z\""));
    }

    #[test]
    fn test_serialize_table_quotes_values_with_double_quotes() {
        let output = serialize_table("test", &["col"], &[vec!["value with \"quotes\"".into()]]);
        assert!(output.contains("\"\"quotes\"\""));
    }

    #[test]
    fn test_serialize_table_newlines_in_values() {
        let output = serialize_table("test", &["col"], &[vec!["line one\nline two".into()]]);
        assert!(output.contains(r#""line one\nline two""#));
        // Newlines in cell should not break line structure
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2); // header + one data row
    }

    #[test]
    fn test_serialize_table_numeric_and_bool_unquoted() {
        let output = serialize_table(
            "test",
            &["num", "flag"],
            &[vec!["42".into(), "true".into()]],
        );
        assert!(output.contains("42"));
        assert!(output.contains("true"));
        assert!(!output.contains("\"42\""));
        assert!(!output.contains("\"true\""));
    }

    #[test]
    fn test_serialize_table_empty_string_quoted() {
        let output = serialize_table("test", &["col"], &[vec![String::new()]]);
        assert!(output.contains("\"\""));
    }

    #[test]
    fn test_serialize_table_empty_rows() {
        let output = serialize_table("commands", &["name", "description"], &[]);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1); // only header line
        assert_eq!(lines[0], "commands[0]{name, description}:");
    }

    #[test]
    fn test_serialize_table_results_format() {
        let output = serialize_table(
            "results",
            &["command", "status", "output"],
            &[vec![
                "web_search".into(),
                "success".into(),
                "Found 3 results".into(),
            ]],
        );
        assert!(output.starts_with("results[1]{command, status, output}:"));
        assert!(output.contains("  web_search, success, Found 3 results"));
    }

    // ── fenced_toon tests ─────────────────────────────────────────────────

    #[test]
    fn test_fenced_toon_with_newline() {
        let content = "commands[2]{name, desc}:\n  web_search, Search\n  recall, Memory\n";
        let result = fenced_toon(content);
        assert_eq!(
            result,
            "```toon\ncommands[2]{name, desc}:\n  web_search, Search\n  recall, Memory\n```"
        );
    }

    #[test]
    fn test_fenced_toon_without_trailing_newline() {
        let result = fenced_toon("key: value");
        assert_eq!(result, "```toon\nkey: value\n```");
    }

    #[test]
    fn test_fenced_toon_starts_and_ends_correctly() {
        let result = fenced_toon("content\n");
        assert!(result.starts_with("```toon\n"));
        assert!(result.ends_with("```"));
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
        assert_eq!(quote_value("1.5"), "1.5");
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

    // ── serialize_value tests ─────────────────────────────────────────────

    #[test]
    fn test_serialize_value_string() {
        assert_eq!(serialize_value(&ToonValue::String("hello".into())), "hello");
        assert_eq!(
            serialize_value(&ToonValue::String("a, b".into())),
            "\"a, b\""
        );
    }

    #[test]
    fn test_serialize_value_integer() {
        assert_eq!(serialize_value(&ToonValue::Integer(42)), "42");
        assert_eq!(serialize_value(&ToonValue::Integer(-5)), "-5");
    }

    #[test]
    fn test_serialize_value_float() {
        let s = serialize_value(&ToonValue::Float(1.5));
        assert!(s.starts_with("1.5"));
    }

    #[test]
    fn test_serialize_value_bool() {
        assert_eq!(serialize_value(&ToonValue::Bool(true)), "true");
        assert_eq!(serialize_value(&ToonValue::Bool(false)), "false");
    }

    #[test]
    fn test_serialize_value_null() {
        assert_eq!(serialize_value(&ToonValue::Null), "null");
    }

    #[test]
    fn test_serialize_value_array() {
        let arr = ToonValue::Array(vec![
            ToonValue::String("a".into()),
            ToonValue::Integer(42),
            ToonValue::Bool(true),
        ]);
        assert_eq!(serialize_value(&arr), "[a, 42, true]");
    }

    // ── serialize_document / serialize_node tests ─────────────────────────

    #[test]
    fn test_serialize_document_empty() {
        let doc = ToonDocument { nodes: vec![] };
        assert_eq!(serialize_document(&doc), "");
    }

    #[test]
    fn test_serialize_document_kv() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::KeyValue {
                key: "role".into(),
                value: ToonValue::String("assistant".into()),
            }],
        };
        assert_eq!(serialize_document(&doc), "role: assistant\n");
    }

    #[test]
    fn test_serialize_document_section() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::Section {
                label: "system".into(),
                children: vec![ToonNode::KeyValue {
                    key: "role".into(),
                    value: ToonValue::String("assistant".into()),
                }],
            }],
        };
        let out = serialize_document(&doc);
        assert_eq!(out, "system:\n  role: assistant\n");
    }

    #[test]
    fn test_serialize_document_typed_array() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::TypedArray {
                label: "commands".into(),
                fields: vec!["name".into(), "desc".into()],
                rows: vec![vec![
                    ToonValue::String("web_search".into()),
                    ToonValue::String("Search".into()),
                ]],
            }],
        };
        let out = serialize_document(&doc);
        assert_eq!(out, "commands[1]{name, desc}:\n  web_search, Search\n");
    }

    // ── parse_document tests ──────────────────────────────────────────────

    #[test]
    fn test_parse_document_empty() {
        let doc = parse_document("").unwrap();
        assert_eq!(doc.nodes, vec![]);
    }

    #[test]
    fn test_parse_document_kv() {
        let doc = parse_document("role: assistant\n").unwrap();
        assert_eq!(
            doc.nodes,
            vec![ToonNode::KeyValue {
                key: "role".into(),
                value: ToonValue::String("assistant".into()),
            }]
        );
    }

    #[test]
    fn test_parse_document_section_with_kv() {
        let input = "system:\n  role: assistant\n  name: weft\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(
            doc.nodes,
            vec![ToonNode::Section {
                label: "system".into(),
                children: vec![
                    ToonNode::KeyValue {
                        key: "role".into(),
                        value: ToonValue::String("assistant".into()),
                    },
                    ToonNode::KeyValue {
                        key: "name".into(),
                        value: ToonValue::String("weft".into()),
                    },
                ],
            }]
        );
    }

    #[test]
    fn test_parse_document_typed_array() {
        let input = "commands[2]{name, desc}:\n  web_search, Search\n  recall, Memory\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(
            doc.nodes,
            vec![ToonNode::TypedArray {
                label: "commands".into(),
                fields: vec!["name".into(), "desc".into()],
                rows: vec![
                    vec![
                        ToonValue::String("web_search".into()),
                        ToonValue::String("Search".into()),
                    ],
                    vec![
                        ToonValue::String("recall".into()),
                        ToonValue::String("Memory".into()),
                    ],
                ],
            }]
        );
    }

    #[test]
    fn test_parse_document_text_block() {
        let input =
            "prompt:\n  You are a helpful assistant that specializes in\n  Rust development.\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(doc.nodes.len(), 1);
        if let ToonNode::Section { label, children } = &doc.nodes[0] {
            assert_eq!(label, "prompt");
            assert_eq!(children.len(), 1);
            if let ToonNode::Text(text) = &children[0] {
                assert!(text.contains("You are a helpful assistant"));
                assert!(text.contains("Rust development."));
            } else {
                panic!("Expected ToonNode::Text");
            }
        } else {
            panic!("Expected ToonNode::Section");
        }
    }

    #[test]
    fn test_parse_document_text_block_single_line() {
        // Single line with no colon -- becomes a text block
        let input = "prompt:\n  Free form text\n";
        let doc = parse_document(input).unwrap();
        if let ToonNode::Section { children, .. } = &doc.nodes[0] {
            assert_eq!(children.len(), 1);
            assert!(matches!(&children[0], ToonNode::Text(_)));
        } else {
            panic!("Expected section");
        }
    }

    #[test]
    fn test_parse_document_empty_section() {
        let doc = parse_document("system:\n").unwrap();
        assert_eq!(
            doc.nodes,
            vec![ToonNode::Section {
                label: "system".into(),
                children: vec![],
            }]
        );
    }

    #[test]
    fn test_parse_document_typed_array_zero_rows() {
        let input = "commands[0]{name, desc}:\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(
            doc.nodes,
            vec![ToonNode::TypedArray {
                label: "commands".into(),
                fields: vec!["name".into(), "desc".into()],
                rows: vec![],
            }]
        );
    }

    #[test]
    fn test_parse_document_typed_array_row_count_mismatch() {
        // Row has wrong number of fields
        let input = "commands[1]{name, desc}:\n  web_search\n";
        let result = parse_document(input);
        assert!(matches!(
            result,
            Err(ToonParseError::FieldCountMismatch { .. })
        ));
    }

    #[test]
    fn test_parse_document_multiple_kv() {
        let input = "name: web_search\ncount: 42\nenabled: true\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(doc.nodes.len(), 3);
    }

    #[test]
    fn test_parse_document_multiple_top_level_nodes() {
        let input = "system:\n  role: assistant\nprompt:\n  You are helpful.\n";
        let doc = parse_document(input).unwrap();
        assert_eq!(doc.nodes.len(), 2);
        assert!(matches!(&doc.nodes[0], ToonNode::Section { label, .. } if label == "system"));
        assert!(matches!(&doc.nodes[1], ToonNode::Section { label, .. } if label == "prompt"));
    }

    // ── extract_fenced_toon tests ─────────────────────────────────────────

    #[test]
    fn test_extract_fenced_toon_single_block() {
        let input = "Some text\n```toon\nrole: assistant\n```\nMore text";
        let docs = extract_fenced_toon(input).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].nodes.len(), 1);
        assert!(matches!(
            &docs[0].nodes[0],
            ToonNode::KeyValue { key, .. } if key == "role"
        ));
    }

    #[test]
    fn test_extract_fenced_toon_multiple_blocks() {
        let input = "Text\n```toon\nrole: assistant\n```\nMiddle\n```toon\nname: weft\n```\nEnd";
        let docs = extract_fenced_toon(input).unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_extract_fenced_toon_no_blocks() {
        let docs = extract_fenced_toon("No fenced blocks here").unwrap();
        assert_eq!(docs.len(), 0);
    }

    #[test]
    fn test_extract_fenced_toon_unclosed_block() {
        let input = "```toon\nrole: assistant\n";
        let result = extract_fenced_toon(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_fenced_toon_typed_array() {
        let input =
            "```toon\ncommands[2]{name, desc}:\n  web_search, Search\n  recall, Memory\n```";
        let docs = extract_fenced_toon(input).unwrap();
        assert_eq!(docs.len(), 1);
        assert!(matches!(&docs[0].nodes[0], ToonNode::TypedArray { .. }));
    }

    // ── parse_value tests ─────────────────────────────────────────────────

    #[test]
    fn test_parse_value_string() {
        let v = parse_value("hello").unwrap();
        assert_eq!(v, ToonValue::String("hello".into()));
    }

    #[test]
    fn test_parse_value_quoted_string() {
        let v = parse_value("\"hello world\"").unwrap();
        assert_eq!(v, ToonValue::String("hello world".into()));
    }

    #[test]
    fn test_parse_value_integer() {
        let v = parse_value("42").unwrap();
        assert_eq!(v, ToonValue::Integer(42));
    }

    #[test]
    fn test_parse_value_negative_integer() {
        let v = parse_value("-5").unwrap();
        assert_eq!(v, ToonValue::Integer(-5));
    }

    #[test]
    fn test_parse_value_float() {
        let v = parse_value("1.5").unwrap();
        assert!(matches!(v, ToonValue::Float(f) if (f - 1.5).abs() < 1e-6));
    }

    #[test]
    fn test_parse_value_bool_true() {
        let v = parse_value("true").unwrap();
        assert_eq!(v, ToonValue::Bool(true));
    }

    #[test]
    fn test_parse_value_bool_false() {
        let v = parse_value("false").unwrap();
        assert_eq!(v, ToonValue::Bool(false));
    }

    #[test]
    fn test_parse_value_null() {
        let v = parse_value("null").unwrap();
        assert_eq!(v, ToonValue::Null);
    }

    #[test]
    fn test_parse_value_array() {
        let v = parse_value("[a, b, c]").unwrap();
        assert_eq!(
            v,
            ToonValue::Array(vec![
                ToonValue::String("a".into()),
                ToonValue::String("b".into()),
                ToonValue::String("c".into()),
            ])
        );
    }

    // ── JSON interop tests ────────────────────────────────────────────────

    #[test]
    fn test_toon_value_string_to_json() {
        let v: serde_json::Value = ToonValue::String("hello".into()).into();
        assert_eq!(v, json!("hello"));
    }

    #[test]
    fn test_toon_value_integer_to_json() {
        let v: serde_json::Value = ToonValue::Integer(42).into();
        assert_eq!(v, json!(42));
    }

    #[test]
    fn test_toon_value_float_to_json() {
        let v: serde_json::Value = ToonValue::Float(1.5).into();
        let f = v.as_f64().unwrap();
        assert!((f - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_toon_value_bool_to_json() {
        let v: serde_json::Value = ToonValue::Bool(true).into();
        assert_eq!(v, json!(true));
    }

    #[test]
    fn test_toon_value_null_to_json() {
        let v: serde_json::Value = ToonValue::Null.into();
        assert_eq!(v, json!(null));
    }

    #[test]
    fn test_toon_value_array_to_json() {
        let arr = ToonValue::Array(vec![ToonValue::Integer(1), ToonValue::Integer(2)]);
        let v: serde_json::Value = arr.into();
        assert_eq!(v, json!([1, 2]));
    }

    #[test]
    fn test_toon_document_kv_to_json() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::KeyValue {
                key: "role".into(),
                value: ToonValue::String("assistant".into()),
            }],
        };
        let v: serde_json::Value = doc.into();
        assert_eq!(v, json!({"role": "assistant"}));
    }

    #[test]
    fn test_toon_document_section_to_json() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::Section {
                label: "system".into(),
                children: vec![ToonNode::KeyValue {
                    key: "role".into(),
                    value: ToonValue::String("assistant".into()),
                }],
            }],
        };
        let v: serde_json::Value = doc.into();
        assert_eq!(v, json!({"system": {"role": "assistant"}}));
    }

    #[test]
    fn test_toon_document_typed_array_to_json() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::TypedArray {
                label: "commands".into(),
                fields: vec!["name".into(), "desc".into()],
                rows: vec![
                    vec![
                        ToonValue::String("web_search".into()),
                        ToonValue::String("Search".into()),
                    ],
                    vec![
                        ToonValue::String("recall".into()),
                        ToonValue::String("Memory".into()),
                    ],
                ],
            }],
        };
        let v: serde_json::Value = doc.into();
        assert_eq!(
            v,
            json!({"commands": [
                {"name": "web_search", "desc": "Search"},
                {"name": "recall", "desc": "Memory"}
            ]})
        );
    }

    #[test]
    fn test_toon_document_text_to_json() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::Section {
                label: "prompt".into(),
                children: vec![ToonNode::Text("You are helpful".into())],
            }],
        };
        let v: serde_json::Value = doc.into();
        assert_eq!(v, json!({"prompt": {"_text": "You are helpful"}}));
    }

    #[test]
    fn test_toon_document_duplicate_keys_last_wins() {
        let doc = ToonDocument {
            nodes: vec![
                ToonNode::KeyValue {
                    key: "role".into(),
                    value: ToonValue::String("first".into()),
                },
                ToonNode::KeyValue {
                    key: "role".into(),
                    value: ToonValue::String("last".into()),
                },
            ],
        };
        let v: serde_json::Value = doc.into();
        assert_eq!(v["role"], json!("last"));
    }

    // ── round-trip tests ──────────────────────────────────────────────────

    #[test]
    fn test_round_trip_kv() {
        let doc = ToonDocument {
            nodes: vec![
                ToonNode::KeyValue {
                    key: "name".into(),
                    value: ToonValue::String("weft".into()),
                },
                ToonNode::KeyValue {
                    key: "count".into(),
                    value: ToonValue::Integer(42),
                },
                ToonNode::KeyValue {
                    key: "active".into(),
                    value: ToonValue::Bool(true),
                },
            ],
        };
        let serialized = serialize_document(&doc);
        let parsed = parse_document(&serialized).unwrap();
        assert_eq!(doc, parsed);
    }

    #[test]
    fn test_round_trip_section() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::Section {
                label: "system".into(),
                children: vec![ToonNode::KeyValue {
                    key: "role".into(),
                    value: ToonValue::String("assistant".into()),
                }],
            }],
        };
        let serialized = serialize_document(&doc);
        let parsed = parse_document(&serialized).unwrap();
        assert_eq!(doc, parsed);
    }

    #[test]
    fn test_round_trip_typed_array() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::TypedArray {
                label: "commands".into(),
                fields: vec!["name".into(), "desc".into()],
                rows: vec![
                    vec![
                        ToonValue::String("web_search".into()),
                        ToonValue::String("Search the web".into()),
                    ],
                    vec![
                        ToonValue::String("recall".into()),
                        ToonValue::String("Memory".into()),
                    ],
                ],
            }],
        };
        let serialized = serialize_document(&doc);
        let parsed = parse_document(&serialized).unwrap();
        assert_eq!(doc, parsed);
    }

    #[test]
    fn test_parse_document_nested_sections() {
        let input = "outer:\n  inner:\n    key: value\n";
        let doc = parse_document(input).unwrap();
        let expected = ToonDocument {
            nodes: vec![ToonNode::Section {
                label: "outer".into(),
                children: vec![ToonNode::Section {
                    label: "inner".into(),
                    children: vec![ToonNode::KeyValue {
                        key: "key".into(),
                        value: ToonValue::String("value".into()),
                    }],
                }],
            }],
        };
        assert_eq!(doc, expected);
    }

    #[test]
    fn test_round_trip_nested_sections() {
        let doc = ToonDocument {
            nodes: vec![ToonNode::Section {
                label: "outer".into(),
                children: vec![ToonNode::Section {
                    label: "inner".into(),
                    children: vec![ToonNode::KeyValue {
                        key: "key".into(),
                        value: ToonValue::String("value".into()),
                    }],
                }],
            }],
        };
        let serialized = serialize_document(&doc);
        let parsed = parse_document(&serialized).unwrap();
        assert_eq!(doc, parsed);
    }

    // ── parse_toon_args tests (preserved, must all pass unchanged) ────────

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
        let result = parse_toon_args(r#"tags: [ml, ai, "deep learning"], limit: 5"#).unwrap();
        assert_eq!(
            result,
            json!({"tags": ["ml", "ai", "deep learning"], "limit": 5})
        );
    }
}
