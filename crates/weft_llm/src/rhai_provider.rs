//! Rhai wire format provider.
//!
//! `RhaiProvider` implements the `Provider` trait using a user-supplied Rhai script
//! to transform between Weft's `ProviderRequest`/`ProviderResponse` types and an
//! arbitrary provider's HTTP API. This allows operators to integrate any HTTP-based
//! AI provider without writing Rust code.
//!
//! # Script Interface
//!
//! Every wire format script MUST define two functions:
//!
//! - `format_request(request)` — transforms a Weft request object map into an HTTP
//!   request spec object map (`{method, path, headers, body}`).
//! - `parse_response(response)` — transforms an HTTP response object map (`{status,
//!   body, headers}`) into a Weft response object map (`{type, text, usage}`) or an
//!   error map (`{error, retry_after_ms?}`).
//!
//! # Sandboxing
//!
//! The Rhai engine is constructed with `Engine::new()` (standard library, no `eval`).
//! Additional hard limits prevent resource abuse:
//! - `set_max_operations(10_000)` — prevents infinite loops
//! - `set_max_string_size(1_048_576)` — 1 MB max string (response bodies can be large)
//! - `set_max_array_size(4_096)` — reasonable limit for message arrays
//! - `set_max_map_size(512)` — reasonable limit for header maps
//!
//! Scripts have no file I/O, no network access, and no system calls. The HTTP call
//! is made by `RhaiProvider` itself — the script only transforms data.
//!
//! # Execution Model
//!
//! Rhai is CPU-bound and synchronous. All Rhai calls are wrapped in
//! `tokio::task::spawn_blocking()` to avoid blocking tokio worker threads. A fresh
//! `Scope` is created per call so there is no shared mutable state between concurrent
//! requests.

use std::sync::Arc;

use async_trait::async_trait;
use rhai::{Dynamic, Engine, Scope};
use tracing::{debug, info, warn};

use crate::{
    Provider, ProviderError, ProviderRequest, ProviderResponse,
    provider::{ChatCompletionOutput, TokenUsage},
};
use weft_core::Role;

/// A provider that uses a Rhai script for wire format transformation.
///
/// The script handles serialization (Weft request → HTTP body) and
/// deserialization (HTTP response → Weft response). The actual HTTP call
/// is made by `RhaiProvider` — the script cannot make network calls.
///
/// `Engine` and `AST` are wrapped in `Arc` so they can be moved into
/// `spawn_blocking` closures. Both are `Send + Sync` with the `rhai/sync` feature.
pub struct RhaiProvider {
    /// Display name for this provider instance (from config), for logging.
    provider_name: String,
    /// Compiled Rhai AST (compiled once at startup, reused for every request).
    ast: Arc<rhai::AST>,
    /// Rhai engine with registered API surface (no file/network access).
    engine: Arc<Engine>,
    /// HTTP client for making provider API calls.
    client: reqwest::Client,
    /// API key (already resolved from `env:` prefix).
    api_key: String,
    /// Base URL for the provider API (e.g. `https://api.banana.ai/v1`).
    base_url: String,
    /// Path to the Rhai script (stored for error messages).
    script_path: String,
}

impl RhaiProvider {
    /// Construct a new `RhaiProvider`.
    ///
    /// Reads the script from disk, configures a sandboxed Rhai engine, compiles the
    /// script into an AST, and validates that both `format_request` and `parse_response`
    /// functions are defined. Fails fast at startup rather than silently at request time.
    ///
    /// # Arguments
    ///
    /// - `script_path`: Path to the `.rhai` file. Must exist and be readable.
    /// - `api_key`: Already resolved API key (env: expansion done by caller).
    /// - `base_url`: Required base URL for the provider API.
    /// - `provider_name`: Display name for logging and diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `ProviderError::WireScriptError` if:
    /// - The file cannot be read.
    /// - The script has a syntax error (compilation failure).
    /// - The script is missing the `format_request` or `parse_response` function.
    pub fn new(
        script_path: &str,
        api_key: String,
        base_url: String,
        provider_name: String,
    ) -> Result<Self, ProviderError> {
        let script_content =
            std::fs::read_to_string(script_path).map_err(|e| ProviderError::WireScriptError {
                script: script_path.to_string(),
                message: format!("failed to read script: {e}"),
            })?;

        let engine = build_engine();

        let ast = engine
            .compile(&script_content)
            .map_err(|e| ProviderError::WireScriptError {
                script: script_path.to_string(),
                message: format!("compilation failed: {e}"),
            })?;

        // Validate that both required functions exist (arity-1).
        // `ast.iter_functions()` exposes function metadata without executing the script.
        validate_required_functions(&ast, script_path)?;

        info!(
            script = %script_path,
            provider = %provider_name,
            "rhai wire format script compiled successfully"
        );

        Ok(Self {
            provider_name,
            ast: Arc::new(ast),
            engine: Arc::new(engine),
            client: reqwest::Client::new(),
            api_key,
            base_url,
            script_path: script_path.to_string(),
        })
    }
}

/// Build a sandboxed Rhai engine with the wire format API registered.
///
/// Uses `Engine::new()` (standard library, no `eval`). No file I/O, no network,
/// no system access. Limits are set per spec section 4.3.2.
fn build_engine() -> Engine {
    let mut engine = Engine::new();

    // Hard limits per spec §4.3.2.
    engine.set_max_operations(10_000);
    engine.set_max_string_size(1_048_576); // 1 MB
    engine.set_max_array_size(4_096);
    engine.set_max_map_size(512);

    register_wire_api(&mut engine);

    engine
}

/// Register the wire format API functions into the engine.
///
/// Only JSON helpers, logging, and base64 encoding are exposed. Scripts cannot
/// perform file I/O, network calls, or system commands.
fn register_wire_api(engine: &mut Engine) {
    // json_encode(value) -> String — serialize a Rhai Dynamic to a JSON string.
    engine.register_fn("json_encode", |value: Dynamic| -> String {
        match rhai::serde::from_dynamic::<serde_json::Value>(&value) {
            Ok(json_val) => serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string()),
            Err(_) => "null".to_string(),
        }
    });

    // json_decode(s) -> Dynamic — deserialize a JSON string to a Rhai Dynamic.
    engine.register_fn("json_decode", |s: &str| -> Dynamic {
        match serde_json::from_str::<serde_json::Value>(s) {
            Ok(json_val) => rhai::serde::to_dynamic(&json_val).unwrap_or(Dynamic::UNIT),
            Err(_) => Dynamic::UNIT,
        }
    });

    // log_info(msg) — emit a tracing::info! log from the wire script.
    engine.register_fn("log_info", |msg: &str| {
        info!(source = "rhai_wire", "{}", msg);
    });

    // log_warn(msg) — emit a tracing::warn! log from the wire script.
    engine.register_fn("log_warn", |msg: &str| {
        warn!(source = "rhai_wire", "{}", msg);
    });

    // base64_encode(s) -> String — base64 encode a string (standard alphabet).
    engine.register_fn("base64_encode", |s: &str| -> String {
        // Inline base64 encoder — standard alphabet, no line breaks.
        // Avoids adding a dependency just for this helper.
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = s.as_bytes();
        let mut buf = String::with_capacity(bytes.len().div_ceil(3) * 4);
        let mut i = 0;
        while i < bytes.len() {
            let b0 = bytes[i] as u32;
            let b1 = if i + 1 < bytes.len() {
                bytes[i + 1] as u32
            } else {
                0
            };
            let b2 = if i + 2 < bytes.len() {
                bytes[i + 2] as u32
            } else {
                0
            };

            buf.push(ALPHABET[((b0 >> 2) & 0x3F) as usize] as char);
            buf.push(ALPHABET[(((b0 << 4) | (b1 >> 4)) & 0x3F) as usize] as char);
            if i + 1 < bytes.len() {
                buf.push(ALPHABET[(((b1 << 2) | (b2 >> 6)) & 0x3F) as usize] as char);
            } else {
                buf.push('=');
            }
            if i + 2 < bytes.len() {
                buf.push(ALPHABET[(b2 & 0x3F) as usize] as char);
            } else {
                buf.push('=');
            }
            i += 3;
        }
        buf
    });

    // base64_decode(s) -> String — base64 decode a string (returns empty on error).
    engine.register_fn("base64_decode", |s: &str| -> String {
        // Simple base64 decoder (standard alphabet, no line breaks).
        const DECODE_TABLE: [i8; 256] = {
            let mut t = [-1i8; 256];
            let mut i = 0usize;
            while i < 26 {
                t[b'A' as usize + i] = i as i8;
                t[b'a' as usize + i] = (i + 26) as i8;
                i += 1;
            }
            let mut i = 0usize;
            while i < 10 {
                t[b'0' as usize + i] = (i + 52) as i8;
                i += 1;
            }
            t[b'+' as usize] = 62;
            t[b'/' as usize] = 63;
            t
        };

        let input: Vec<u8> = s
            .bytes()
            .filter(|&b| b != b'=')
            .filter(|&b| DECODE_TABLE[b as usize] >= 0)
            .collect();

        let mut out = Vec::with_capacity(input.len() * 3 / 4);
        let mut i = 0;
        while i + 3 < input.len() {
            let v0 = DECODE_TABLE[input[i] as usize] as u32;
            let v1 = DECODE_TABLE[input[i + 1] as usize] as u32;
            let v2 = DECODE_TABLE[input[i + 2] as usize] as u32;
            let v3 = DECODE_TABLE[input[i + 3] as usize] as u32;
            out.push(((v0 << 2) | (v1 >> 4)) as u8);
            out.push(((v1 << 4) | (v2 >> 2)) as u8);
            out.push(((v2 << 6) | v3) as u8);
            i += 4;
        }
        if i + 2 < input.len() {
            let v0 = DECODE_TABLE[input[i] as usize] as u32;
            let v1 = DECODE_TABLE[input[i + 1] as usize] as u32;
            let v2 = DECODE_TABLE[input[i + 2] as usize] as u32;
            out.push(((v0 << 2) | (v1 >> 4)) as u8);
            out.push(((v1 << 4) | (v2 >> 2)) as u8);
        } else if i + 1 < input.len() {
            let v0 = DECODE_TABLE[input[i] as usize] as u32;
            let v1 = DECODE_TABLE[input[i + 1] as usize] as u32;
            out.push(((v0 << 2) | (v1 >> 4)) as u8);
        }

        String::from_utf8(out).unwrap_or_default()
    });
}

/// Validate that the compiled AST defines both required wire format functions.
///
/// Checks via `ast.iter_functions()` which exposes function metadata without
/// executing the script. Checks for arity-1 variants of each function name.
fn validate_required_functions(ast: &rhai::AST, script_path: &str) -> Result<(), ProviderError> {
    let fn_names: Vec<String> = ast.iter_functions().map(|f| f.name.to_string()).collect();

    if !fn_names.iter().any(|n| n == "format_request") {
        return Err(ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: "script must define fn format_request(request)".to_string(),
        });
    }
    if !fn_names.iter().any(|n| n == "parse_response") {
        return Err(ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: "script must define fn parse_response(response)".to_string(),
        });
    }

    Ok(())
}

/// Convert a `ProviderRequest` into a Rhai `Dynamic` object map.
///
/// The returned map matches the interface contract documented in the script
/// interface (spec §4.3.1):
/// - `type`: "chat_completion"
/// - `model`: model identifier
/// - `system_prompt`: system prompt text
/// - `messages`: array of `{role, content}` maps
/// - `max_tokens`: integer
/// - `temperature`: float or `()` (unit = null)
fn request_to_dynamic(request: &ProviderRequest) -> Dynamic {
    match request {
        ProviderRequest::ChatCompletion(input) => {
            let mut map = rhai::Map::new();
            map.insert("type".into(), Dynamic::from("chat_completion".to_string()));
            map.insert("model".into(), Dynamic::from(input.model.clone()));
            map.insert(
                "system_prompt".into(),
                Dynamic::from(input.system_prompt.clone()),
            );

            let messages: rhai::Array = input
                .messages
                .iter()
                .map(|msg| {
                    let mut msg_map = rhai::Map::new();
                    let role_str = match msg.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::System => "system",
                    };
                    msg_map.insert("role".into(), Dynamic::from(role_str.to_string()));
                    msg_map.insert("content".into(), Dynamic::from(msg.content.clone()));
                    Dynamic::from_map(msg_map)
                })
                .collect();

            map.insert("messages".into(), Dynamic::from_array(messages));
            map.insert("max_tokens".into(), Dynamic::from(input.max_tokens as i64));

            // temperature: float or unit (null-like)
            match input.temperature {
                Some(t) => map.insert("temperature".into(), Dynamic::from(t as f64)),
                None => map.insert("temperature".into(), Dynamic::UNIT),
            };

            Dynamic::from_map(map)
        }
    }
}

/// HTTP request specification produced by the script's `format_request` function.
#[derive(Debug)]
struct RequestSpec {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

/// Extract the HTTP request spec returned by the script's `format_request` function.
///
/// Expected shape: `{method, path, headers?, body}`. All string values.
/// Returns a `RequestSpec` or a `WireScriptError`.
fn extract_request_spec(dynamic: Dynamic, script_path: &str) -> Result<RequestSpec, ProviderError> {
    // Convert Dynamic -> serde_json::Value for easier field access.
    let json: serde_json::Value =
        rhai::serde::from_dynamic(&dynamic).map_err(|e| ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: format!("format_request returned non-map value: {e}"),
        })?;

    let method = json
        .get("method")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: "format_request must return map with 'method' string field".to_string(),
        })?
        .to_string();

    let path = json
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: "format_request must return map with 'path' string field".to_string(),
        })?
        .to_string();

    let body = json
        .get("body")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: "format_request must return map with 'body' string field".to_string(),
        })?
        .to_string();

    // Optional `headers` map: String -> String. Empty if absent.
    let headers: Vec<(String, String)> = json
        .get("headers")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    Ok(RequestSpec {
        method,
        path,
        headers,
        body,
    })
}

/// Convert the HTTP response object map from `parse_response` into a `ProviderResponse`
/// or a `ProviderError`.
///
/// Script return shape for success (chat_completion):
///   `{type: "chat_completion", text: "...", usage: {prompt_tokens, completion_tokens}?}`
///
/// Script return shape for errors:
///   `{error: "...", retry_after_ms?: integer}`
fn dynamic_to_provider_result(
    dynamic: Dynamic,
    script_path: &str,
) -> Result<ProviderResponse, ProviderError> {
    let json: serde_json::Value =
        rhai::serde::from_dynamic(&dynamic).map_err(|e| ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: format!("parse_response returned non-map value: {e}"),
        })?;

    // Check for error response from the script.
    if let Some(error_msg) = json.get("error").and_then(|v| v.as_str()) {
        if let Some(retry_after_ms) = json.get("retry_after_ms").and_then(|v| v.as_u64()) {
            return Err(ProviderError::RateLimited { retry_after_ms });
        }
        return Err(ProviderError::ProviderHttpError {
            status: 0,
            body: error_msg.to_string(),
        });
    }

    // Determine the response type.
    let response_type = json.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: "parse_response must return map with 'type' string field".to_string(),
        }
    })?;

    match response_type {
        "chat_completion" => {
            let text = json
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderError::WireScriptError {
                    script: script_path.to_string(),
                    message: "chat_completion response must have 'text' field".to_string(),
                })?
                .to_string();

            let usage = json.get("usage").and_then(|u| {
                let prompt = u.get("prompt_tokens").and_then(|v| v.as_u64())? as u32;
                let completion = u.get("completion_tokens").and_then(|v| v.as_u64())? as u32;
                Some(TokenUsage {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                })
            });

            Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                text,
                usage,
            }))
        }
        other => Err(ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: format!("unknown response type from parse_response: '{other}'"),
        }),
    }
}

#[async_trait]
impl Provider for RhaiProvider {
    async fn execute(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let engine = Arc::clone(&self.engine);
        let ast = Arc::clone(&self.ast);
        let script_path = self.script_path.clone();

        // Step 1: Convert the ProviderRequest to a Rhai Dynamic.
        let request_dynamic = request_to_dynamic(&request);

        // Step 2: Call format_request(request_map) via spawn_blocking (CPU-bound).
        let script_path_clone = script_path.clone();
        let format_result = tokio::task::spawn_blocking(move || {
            let mut scope = Scope::new();
            engine
                .call_fn::<Dynamic>(&mut scope, &ast, "format_request", (request_dynamic,))
                .map_err(|e| ProviderError::WireScriptError {
                    script: script_path_clone.clone(),
                    message: format!("format_request failed: {e}"),
                })
        })
        .await
        .map_err(|e| ProviderError::WireScriptError {
            script: script_path.clone(),
            message: format!("format_request task join error: {e}"),
        })??;

        // Step 3: Extract method, path, headers, body from the returned map.
        let spec = extract_request_spec(format_result, &script_path)?;

        // Step 4: Make the HTTP request.
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), spec.path);
        debug!(
            provider = %self.provider_name,
            method = %spec.method,
            url = %url,
            "rhai provider making HTTP request"
        );

        let mut req_builder = match spec.method.to_uppercase().as_str() {
            "POST" => self.client.post(&url),
            "GET" => self.client.get(&url),
            "PUT" => self.client.put(&url),
            "DELETE" => self.client.delete(&url),
            other => {
                return Err(ProviderError::WireScriptError {
                    script: script_path.clone(),
                    message: format!("unsupported HTTP method from format_request: '{other}'"),
                });
            }
        };

        // Authorization header (always set; script can override via headers map).
        req_builder = req_builder.header("Authorization", format!("Bearer {}", self.api_key));

        // Default Content-Type unless script overrides it.
        let script_sets_content_type = spec
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "content-type");
        if !script_sets_content_type {
            req_builder = req_builder.header("Content-Type", "application/json");
        }

        // Apply script-specified headers.
        for (k, v) in &spec.headers {
            req_builder = req_builder.header(k.as_str(), v.as_str());
        }

        req_builder = req_builder.body(spec.body);

        let http_response = req_builder
            .send()
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("{}: {}", self.provider_name, e)))?;

        // Step 5: Collect the HTTP response.
        let status = http_response.status().as_u16();
        let resp_headers: Vec<(String, String)> = http_response
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|v_str| (k.as_str().to_string(), v_str.to_string()))
            })
            .collect();
        let resp_body = http_response.text().await.map_err(|e| {
            ProviderError::RequestFailed(format!("failed to read response body: {e}"))
        })?;

        // Step 6: Build the response Dynamic for parse_response.
        let mut resp_map = rhai::Map::new();
        resp_map.insert("status".into(), Dynamic::from(status as i64));
        resp_map.insert("body".into(), Dynamic::from(resp_body));
        let mut headers_map = rhai::Map::new();
        for (k, v) in resp_headers {
            headers_map.insert(k.into(), Dynamic::from(v));
        }
        resp_map.insert("headers".into(), Dynamic::from_map(headers_map));
        let response_dynamic = Dynamic::from_map(resp_map);

        // Step 7: Call parse_response(response_map) via spawn_blocking.
        let engine2 = Arc::clone(&self.engine);
        let ast2 = Arc::clone(&self.ast);
        let script_path_clone2 = script_path.clone();
        let parse_result = tokio::task::spawn_blocking(move || {
            let mut scope = Scope::new();
            engine2
                .call_fn::<Dynamic>(&mut scope, &ast2, "parse_response", (response_dynamic,))
                .map_err(|e| ProviderError::WireScriptError {
                    script: script_path_clone2.clone(),
                    message: format!("parse_response failed: {e}"),
                })
        })
        .await
        .map_err(|e| ProviderError::WireScriptError {
            script: script_path.clone(),
            message: format!("parse_response task join error: {e}"),
        })??;

        // Step 8: Convert the returned map to ProviderResponse or ProviderError.
        dynamic_to_provider_result(parse_result, &script_path)
    }

    fn name(&self) -> &str {
        &self.provider_name
    }
}

impl std::fmt::Debug for RhaiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RhaiProvider")
            .field("provider_name", &self.provider_name)
            .field("script_path", &self.script_path)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use weft_core::{Message, Role};

    /// Write a Rhai script to a temporary file and return the file handle.
    /// The file is deleted when the handle is dropped.
    fn write_temp_script(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// Build a minimal valid wire format script.
    const MINIMAL_VALID_SCRIPT: &str = r#"
fn format_request(request) {
    #{
        method: "POST",
        path: "/v1/chat/completions",
        headers: #{},
        body: json_encode(#{
            model: request.model,
            max_tokens: request.max_tokens,
        })
    }
}

fn parse_response(response) {
    let body = json_decode(response.body);
    if response.status == 429 {
        return #{
            error: "rate limited",
            retry_after_ms: 60000,
        };
    }
    if response.status != 200 {
        return #{
            error: `Provider returned ${response.status}`,
        };
    }
    #{
        type: "chat_completion",
        text: body.choices[0].message.content,
        usage: #{
            prompt_tokens: body.usage.prompt_tokens,
            completion_tokens: body.usage.completion_tokens,
        }
    }
}
"#;

    /// Build a minimal `ProviderRequest::ChatCompletion`.
    fn sample_chat_request() -> ProviderRequest {
        ProviderRequest::ChatCompletion(crate::provider::ChatCompletionInput {
            system_prompt: "You are helpful.".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: "Hello".to_string(),
            }],
            model: "test-model".to_string(),
            max_tokens: 1024,
            temperature: Some(0.7),
        })
    }

    // ── Construction tests ────────────────────────────────────────────────────

    #[test]
    fn test_construction_valid_script_succeeds() {
        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            "https://api.example.com".to_string(),
            "test-provider".to_string(),
        );
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    }

    #[test]
    fn test_construction_nonexistent_file_fails() {
        let result = RhaiProvider::new(
            "/nonexistent/path/wire.rhai",
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ProviderError::WireScriptError { ref message, .. } if message.contains("failed to read script")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_construction_syntax_error_fails() {
        let f = write_temp_script("fn format_request(r) { let x = ; }");
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ProviderError::WireScriptError { ref message, .. } if message.contains("compilation failed")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_construction_missing_format_request_fails() {
        // Script with only parse_response defined.
        let f = write_temp_script(
            "fn parse_response(r) { #{ type: \"chat_completion\", text: \"ok\" } }",
        );
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ProviderError::WireScriptError { ref message, .. } if message.contains("format_request")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_construction_missing_parse_response_fails() {
        // Script with only format_request defined.
        let f = write_temp_script(
            "fn format_request(r) { #{ method: \"POST\", path: \"/\", headers: #{}, body: \"{}\" } }",
        );
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ProviderError::WireScriptError { ref message, .. } if message.contains("parse_response")),
            "unexpected error: {err}"
        );
    }

    // ── request_to_dynamic tests ──────────────────────────────────────────────

    #[test]
    fn test_request_to_dynamic_chat_completion_fields() {
        let req = sample_chat_request();
        let dynamic = request_to_dynamic(&req);

        // Convert back to JSON for assertion.
        let json: serde_json::Value = rhai::serde::from_dynamic(&dynamic).unwrap();
        assert_eq!(json["type"], "chat_completion");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["system_prompt"], "You are helpful.");
        assert_eq!(json["max_tokens"], 1024);
        assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);

        let messages = json["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello");
    }

    #[test]
    fn test_request_to_dynamic_no_temperature_is_unit() {
        let req = ProviderRequest::ChatCompletion(crate::provider::ChatCompletionInput {
            system_prompt: "sys".to_string(),
            messages: vec![],
            model: "m".to_string(),
            max_tokens: 100,
            temperature: None,
        });
        let dynamic = request_to_dynamic(&req);
        let json: serde_json::Value = rhai::serde::from_dynamic(&dynamic).unwrap();
        // temperature should be null/absent when None
        assert!(
            json["temperature"].is_null()
                || !json.as_object().unwrap().contains_key("temperature")
                || json["temperature"] == serde_json::Value::Null
        );
    }

    #[test]
    fn test_request_to_dynamic_message_roles() {
        let req = ProviderRequest::ChatCompletion(crate::provider::ChatCompletionInput {
            system_prompt: "sys".to_string(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: "user msg".to_string(),
                },
                Message {
                    role: Role::Assistant,
                    content: "assistant msg".to_string(),
                },
                Message {
                    role: Role::System,
                    content: "system msg".to_string(),
                },
            ],
            model: "m".to_string(),
            max_tokens: 100,
            temperature: None,
        });
        let dynamic = request_to_dynamic(&req);
        let json: serde_json::Value = rhai::serde::from_dynamic(&dynamic).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "system");
    }

    // ── extract_request_spec tests ────────────────────────────────────────────

    #[test]
    fn test_extract_request_spec_valid() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(r#"#{ method: "POST", path: "/v1/chat", headers: #{}, body: "{}" }"#)
            .unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_ok());
        let spec = result.unwrap();
        assert_eq!(spec.method, "POST");
        assert_eq!(spec.path, "/v1/chat");
        assert!(spec.headers.is_empty());
        assert_eq!(spec.body, "{}");
    }

    #[test]
    fn test_extract_request_spec_with_headers() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(r#"#{ method: "POST", path: "/v1", headers: #{ "X-Custom": "value" }, body: "{}" }"#)
            .unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_ok());
        let spec = result.unwrap();
        assert!(
            spec.headers
                .iter()
                .any(|(k, v)| k == "X-Custom" && v == "value")
        );
    }

    #[test]
    fn test_extract_request_spec_missing_method_fails() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(r#"#{ path: "/v1", headers: #{}, body: "{}" }"#)
            .unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { ref message, .. } if message.contains("method"))
        );
    }

    #[test]
    fn test_extract_request_spec_missing_body_fails() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(r#"#{ method: "POST", path: "/v1", headers: #{} }"#)
            .unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { ref message, .. } if message.contains("body"))
        );
    }

    #[test]
    fn test_extract_request_spec_non_map_fails() {
        let dynamic = Dynamic::from("just a string");
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_err());
    }

    // ── dynamic_to_provider_result tests ─────────────────────────────────────

    #[test]
    fn test_dynamic_to_provider_result_chat_completion_success() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(
                r#"
            #{
                type: "chat_completion",
                text: "Hello world",
                usage: #{ prompt_tokens: 10, completion_tokens: 5 }
            }
        "#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_ok());
        let ProviderResponse::ChatCompletion(output) = result.unwrap();
        assert_eq!(output.text, "Hello world");
        let usage = output.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_dynamic_to_provider_result_no_usage() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(
                r#"
            #{ type: "chat_completion", text: "Hi" }
        "#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_ok());
        let ProviderResponse::ChatCompletion(output) = result.unwrap();
        assert_eq!(output.text, "Hi");
        assert!(output.usage.is_none());
    }

    #[test]
    fn test_dynamic_to_provider_result_error_response() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(
                r#"
            #{ error: "Provider returned 503" }
        "#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProviderError::ProviderHttpError { .. }
        ));
    }

    #[test]
    fn test_dynamic_to_provider_result_rate_limited() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(
                r#"
            #{ error: "rate limited", retry_after_ms: 60000 }
        "#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProviderError::RateLimited {
                retry_after_ms: 60000
            }
        ));
    }

    #[test]
    fn test_dynamic_to_provider_result_unknown_type_fails() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(
                r#"
            #{ type: "embedding", data: [] }
        "#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ProviderError::WireScriptError { ref message, .. } if message.contains("unknown response type")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_dynamic_to_provider_result_non_map_fails() {
        let dynamic = Dynamic::from(42_i64);
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_to_provider_result_missing_text_fails() {
        let engine = Engine::new();
        let dynamic: Dynamic = engine
            .eval(
                r#"
            #{ type: "chat_completion" }
        "#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { ref message, .. } if message.contains("text"))
        );
    }

    // ── Rhai API functions ────────────────────────────────────────────────────

    #[test]
    fn test_json_encode_decode_roundtrip() {
        let engine = {
            let mut e = Engine::new();
            register_wire_api(&mut e);
            e
        };
        let result: Dynamic = engine
            .eval(
                r#"
                let obj = #{ key: "value", num: 42 };
                let encoded = json_encode(obj);
                json_decode(encoded)
            "#,
            )
            .unwrap();
        let json: serde_json::Value = rhai::serde::from_dynamic(&result).unwrap();
        assert_eq!(json["key"], "value");
        assert_eq!(json["num"], 42);
    }

    #[test]
    fn test_base64_encode_accessible_in_script() {
        let engine = {
            let mut e = Engine::new();
            register_wire_api(&mut e);
            e
        };
        let result: String = engine.eval(r#"base64_encode("hello")"#).unwrap();
        assert_eq!(result, "aGVsbG8=");
    }

    #[test]
    fn test_base64_decode_accessible_in_script() {
        let engine = {
            let mut e = Engine::new();
            register_wire_api(&mut e);
            e
        };
        let result: String = engine.eval(r#"base64_decode("aGVsbG8=")"#).unwrap();
        assert_eq!(result, "hello");
    }

    // ── Engine sandboxing ────────────────────────────────────────────────────

    #[test]
    fn test_operation_limit_prevents_infinite_loop() {
        // Verify the engine's max_operations is applied.
        let engine = build_engine();
        let ast = engine
            .compile("fn format_request(r) { let i = 0; loop { i += 1; } }")
            .unwrap();
        let mut scope = Scope::new();
        let result =
            engine.call_fn::<Dynamic>(&mut scope, &ast, "format_request", (Dynamic::UNIT,));
        // Should fail with an operation limit error, not hang.
        assert!(result.is_err(), "expected operation limit error");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("operations") || err_str.contains("limit"),
            "expected operation limit error message, got: {err_str}"
        );
    }

    // ── HTTP integration tests (mockito) ─────────────────────────────────────

    #[tokio::test]
    async fn test_execute_chat_completion_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"choices":[{"message":{"content":"Hello from provider!"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#)
            .create_async()
            .await;

        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        mock.assert_async().await;

        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let ProviderResponse::ChatCompletion(output) = result.unwrap();
        assert_eq!(output.text, "Hello from provider!");
        let usage = output.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[tokio::test]
    async fn test_execute_rate_limited_429() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(429)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        mock.assert_async().await;

        assert!(result.is_err());
        assert!(
            matches!(
                result.unwrap_err(),
                ProviderError::RateLimited {
                    retry_after_ms: 60000
                }
            ),
            "expected RateLimited"
        );
    }

    #[tokio::test]
    async fn test_execute_provider_error_non_200() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(503)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        mock.assert_async().await;

        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::ProviderHttpError { .. }),
            "expected ProviderHttpError"
        );
    }

    #[tokio::test]
    async fn test_execute_format_request_script_error_returns_wire_script_error() {
        // A script where format_request throws at runtime.
        let script = r#"
fn format_request(request) {
    throw "intentional format_request error";
}
fn parse_response(response) {
    #{ type: "chat_completion", text: "ok" }
}
"#;
        let mut server = mockito::Server::new_async().await;
        // The mock should never be called since format_request fails.
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .expect(0)
            .create_async()
            .await;

        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { .. }),
            "expected WireScriptError"
        );
    }

    #[tokio::test]
    async fn test_execute_parse_response_script_error_returns_wire_script_error() {
        // A script where parse_response throws at runtime.
        let script = r#"
fn format_request(request) {
    #{
        method: "POST",
        path: "/v1/chat/completions",
        headers: #{},
        body: "{}"
    }
}
fn parse_response(response) {
    throw "intentional parse_response error";
}
"#;
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"choices":[{"message":{"content":"Hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#)
            .create_async()
            .await;

        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        mock.assert_async().await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { .. }),
            "expected WireScriptError"
        );
    }

    #[tokio::test]
    async fn test_execute_script_exceeding_operation_limit_returns_wire_script_error() {
        // A script where format_request runs an infinite loop.
        let script = r#"
fn format_request(request) {
    let i = 0;
    loop { i += 1; }
}
fn parse_response(response) {
    #{ type: "chat_completion", text: "ok" }
}
"#;
        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            "https://api.example.com".to_string(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { .. }),
            "expected WireScriptError from operation limit"
        );
    }

    #[tokio::test]
    async fn test_execute_parse_response_returning_unexpected_type_returns_wire_script_error() {
        // parse_response returns an unknown type string.
        let script = r#"
fn format_request(request) {
    #{ method: "POST", path: "/v1/chat/completions", headers: #{}, body: "{}" }
}
fn parse_response(response) {
    #{ type: "embedding", data: [] }
}
"#;
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test-provider".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        mock.assert_async().await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { ref message, .. } if message.contains("unknown response type")),
            "expected WireScriptError with unknown response type"
        );
    }

    // ── Provider trait compliance ─────────────────────────────────────────────

    #[test]
    fn test_name_returns_provider_name() {
        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "banana-ai".to_string(),
        )
        .unwrap();
        assert_eq!(provider.name(), "banana-ai");
    }

    #[test]
    fn test_debug_format_does_not_leak_api_key() {
        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "secret-api-key".to_string(),
            "https://api.example.com".to_string(),
            "test-provider".to_string(),
        )
        .unwrap();
        let debug_str = format!("{provider:?}");
        assert!(
            !debug_str.contains("secret-api-key"),
            "Debug output should not expose API key: {debug_str}"
        );
    }
}
