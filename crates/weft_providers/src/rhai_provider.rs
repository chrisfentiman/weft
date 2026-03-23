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
//! # Script Request Map Shape
//!
//! The request map passed to `format_request` has the following shape:
//!
//! ```text
//! {
//!   type: "chat_completion",
//!   model: "model-name",
//!   messages: [
//!     {
//!       role: "user"|"assistant"|"system",
//!       source: "client"|"gateway"|"provider"|...,
//!       model: "model-name" or (),
//!       content: "concatenated text" (backward compat),
//!       content_parts: [
//!         {type: "text", text: "..."} |
//!         {type: "command_call", command: "...", arguments_json: "..."} |
//!         {type: "command_result", command: "...", success: true, output: "..."} |
//!         {type: "memory_result"} |
//!         ...
//!       ]
//!     }
//!   ],
//!   max_tokens: integer or (),  // backward compat top-level field
//!   temperature: float or (),   // backward compat top-level field
//!   options: {
//!     max_tokens: integer or (),
//!     temperature: float or (),
//!     top_p: float or (),
//!     top_k: integer or (),
//!     frequency_penalty: float or (),
//!     presence_penalty: float or (),
//!     seed: integer or (),
//!     stop: ["..."]  // only present if non-empty
//!   }
//! }
//! ```
//!
//! Scripts accessing `request.messages[n].content` (string) continue to work for
//! backward compatibility. New scripts can access `content_parts` for structured
//! content. The system prompt, if present, is `messages[0]` with `role == "system"`.
//! There is no `system_prompt` top-level field.
//!
//! # Sandboxing
//!
//! The Rhai engine is constructed via `weft_rhai::EngineBuilder` with
//! `SandboxLimits::relaxed()`. Relaxed limits are appropriate because wire format
//! scripts process full request/response bodies:
//! - `max_operations: 10_000` — prevents infinite loops
//! - `max_string_size: 1_048_576` — 1 MB max string (response bodies can be large)
//! - `max_array_size: 4_096` — reasonable limit for message arrays
//! - `max_map_size: 512` — reasonable limit for header maps
//!
//! Scripts have no file I/O, no network access, and no system calls. The HTTP call
//! is made by `RhaiProvider` itself — the script only transforms data.
//!
//! # Execution Model
//!
//! Rhai is CPU-bound and synchronous. All Rhai calls go through
//! `weft_rhai::safe_call_fn`, which wraps `tokio::task::spawn_blocking` and
//! `std::panic::catch_unwind` to prevent blocking tokio workers and to provide
//! structured error messages on Rhai-internal panics. A fresh `Scope` is created
//! per call so there is no shared mutable state between concurrent requests.

use std::sync::Arc;

use async_trait::async_trait;
use rhai::Dynamic;
use tracing::{Instrument, debug, info, info_span, warn};
use weft_core::{ContentPart, Role, Source, WeftMessage};
use weft_rhai::{CompiledScript, EngineBuilder, SandboxLimits, ScriptError, safe_call_fn};

use crate::{Provider, ProviderError, ProviderRequest, ProviderResponse, provider::TokenUsage};

/// A provider that uses a Rhai script for wire format transformation.
///
/// The script handles serialization (Weft request → HTTP body) and
/// deserialization (HTTP response → Weft response). The actual HTTP call
/// is made by `RhaiProvider` — the script cannot make network calls.
///
/// `Engine` is wrapped in `Arc` so it can be shared into `spawn_blocking`
/// closures. `CompiledScript` owns an `Arc<AST>` internally.
pub struct RhaiProvider {
    /// Display name for this provider instance (from config), for logging.
    provider_name: String,
    /// Compiled Rhai script with metadata (compiled once at startup).
    script: CompiledScript,
    /// Rhai engine with registered API surface (no file/network access).
    engine: Arc<weft_rhai::Engine>,
    /// HTTP client for making provider API calls.
    client: reqwest::Client,
    /// API key (already resolved from `env:` prefix).
    api_key: String,
    /// Base URL for the provider API (e.g. `https://api.banana.ai/v1`).
    base_url: String,
}

impl RhaiProvider {
    /// Construct a new `RhaiProvider`.
    ///
    /// Reads the script from disk, configures a sandboxed Rhai engine via
    /// `weft_rhai::EngineBuilder`, compiles the script, and validates that both
    /// `format_request` and `parse_response` functions are defined. Fails fast at
    /// startup rather than silently at request time.
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
        let engine = EngineBuilder::new(SandboxLimits::relaxed())
            .log_source("rhai_wire")
            .with_base64_helpers(true)
            .build();
        let engine = Arc::new(engine);

        let script = CompiledScript::load(script_path, &engine)
            .map_err(|e| script_error_to_provider(e, script_path))?;

        script
            .validate_functions(&["format_request", "parse_response"])
            .map_err(|e| script_error_to_provider(e, script_path))?;

        info!(
            script = %script_path,
            provider = %provider_name,
            "rhai wire format script compiled successfully"
        );

        Ok(Self {
            provider_name,
            script,
            engine,
            client: reqwest::Client::new(),
            api_key,
            base_url,
        })
    }
}

/// Convert a `ScriptError` from `weft_rhai` into a `ProviderError::WireScriptError`.
///
/// This maps the shared infrastructure error into the domain error type for
/// `weft_providers`, preserving the script path and message for diagnostics.
fn script_error_to_provider(e: ScriptError, script_path: &str) -> ProviderError {
    match &e {
        ScriptError::FileNotFound { path, source } => ProviderError::WireScriptError {
            script: path.clone(),
            message: format!("failed to read script: {source}"),
        },
        ScriptError::CompilationFailed { path, message } => ProviderError::WireScriptError {
            script: path.clone(),
            message: format!("compilation failed: {message}"),
        },
        ScriptError::ExecutionError { path, message } => ProviderError::WireScriptError {
            script: path.clone(),
            message: format!("execution error: {message}"),
        },
        ScriptError::Panic { path, message } => ProviderError::WireScriptError {
            script: path.clone(),
            message: format!("script panicked: {message}"),
        },
        ScriptError::TaskJoinError { path, message } => ProviderError::WireScriptError {
            script: path.clone(),
            message: format!("task join error: {message}"),
        },
        ScriptError::MissingFunction { path, function } => ProviderError::WireScriptError {
            script: path.clone(),
            message: format!("script must define fn {function}(request)"),
        },
        ScriptError::ConversionError { message } => ProviderError::WireScriptError {
            script: script_path.to_string(),
            message: format!("conversion error: {message}"),
        },
    }
}

/// Convert a `ProviderRequest` into a Rhai `Dynamic` object map.
///
/// The returned map has:
/// - `type`: "chat_completion"
/// - `model`: model identifier
/// - `messages`: array of full WeftMessage maps (role, source, model, content, content_parts)
/// - `max_tokens`: integer or `()` (top-level, backward compat)
/// - `temperature`: float or `()` (top-level, backward compat)
/// - `options`: sub-map with all sampling fields
///
/// Unlike OpenAI/Anthropic providers, the Rhai provider does NOT filter gateway
/// activity messages. The script receives ALL messages and decides what to include.
/// This gives scripts maximum flexibility.
fn request_to_dynamic(request: &ProviderRequest) -> Dynamic {
    match request {
        ProviderRequest::ChatCompletion {
            messages,
            model,
            options,
        } => {
            let mut map = rhai::Map::new();
            map.insert("type".into(), Dynamic::from("chat_completion".to_string()));
            map.insert("model".into(), Dynamic::from(model.clone()));

            // Convert WeftMessages to Rhai maps.
            // Each message is {role, source, model, content_parts, content}
            // where `content` is the concatenated text (backward compat)
            // and `content_parts` is the full typed array.
            //
            // Unlike OpenAI/Anthropic, the Rhai provider passes ALL messages
            // (including gateway activity) -- the script decides what to include.
            let messages_array: rhai::Array =
                messages.iter().map(weft_message_to_dynamic).collect();
            map.insert("messages".into(), Dynamic::from_array(messages_array));

            // Backward-compat top-level sampling fields
            map.insert(
                "max_tokens".into(),
                option_to_dynamic_i64(options.max_tokens),
            );
            map.insert(
                "temperature".into(),
                option_to_dynamic_f64(options.temperature),
            );

            // Full sampling options as sub-map for new scripts
            let mut opts_map = rhai::Map::new();
            opts_map.insert(
                "max_tokens".into(),
                option_to_dynamic_i64(options.max_tokens),
            );
            opts_map.insert(
                "temperature".into(),
                option_to_dynamic_f64(options.temperature),
            );
            opts_map.insert("top_p".into(), option_to_dynamic_f64(options.top_p));
            opts_map.insert("top_k".into(), option_to_dynamic_i64(options.top_k));
            opts_map.insert(
                "frequency_penalty".into(),
                option_to_dynamic_f64(options.frequency_penalty),
            );
            opts_map.insert(
                "presence_penalty".into(),
                option_to_dynamic_f64(options.presence_penalty),
            );
            opts_map.insert(
                "seed".into(),
                match options.seed {
                    Some(s) => Dynamic::from(s),
                    None => Dynamic::UNIT,
                },
            );
            if !options.stop.is_empty() {
                let stop_arr: rhai::Array = options
                    .stop
                    .iter()
                    .map(|s| Dynamic::from(s.clone()))
                    .collect();
                opts_map.insert("stop".into(), Dynamic::from_array(stop_arr));
            }
            map.insert("options".into(), Dynamic::from_map(opts_map));

            Dynamic::from_map(map)
        }
    }
}

/// Convert a WeftMessage to a Rhai Dynamic map.
///
/// Structure:
/// ```text
/// {
///   role: "user"|"assistant"|"system",
///   source: "client"|"gateway"|"provider"|...,
///   model: "model-name" or (),
///   content: "concatenated text" (backward compat),
///   content_parts: [
///     {type: "text", text: "..."} |
///     {type: "command_call", command: "...", arguments_json: "..."} |
///     {type: "command_result", command: "...", success: true, output: "..."} |
///     {type: "memory_result"} |
///     {type: "memory_stored"} |
///     {type: "council_start"} |
///     {type: "image"} | {type: "audio"} | {type: "video"} |
///     {type: "document"} | {type: "routing"} | {type: "hook"}
///   ]
/// }
/// ```
fn weft_message_to_dynamic(msg: &WeftMessage) -> Dynamic {
    let mut map = rhai::Map::new();

    let role_str = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    };
    map.insert("role".into(), Dynamic::from(role_str.to_string()));

    let source_str = match msg.source {
        Source::Client => "client",
        Source::Gateway => "gateway",
        Source::Provider => "provider",
        Source::Member => "member",
        Source::Judge => "judge",
        Source::Tool => "tool",
        Source::Memory => "memory",
    };
    map.insert("source".into(), Dynamic::from(source_str.to_string()));

    match &msg.model {
        Some(m) => map.insert("model".into(), Dynamic::from(m.clone())),
        None => map.insert("model".into(), Dynamic::UNIT),
    };

    // Backward-compat: concatenated text content
    let text: String = msg
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    map.insert("content".into(), Dynamic::from(text));

    // Full content parts array
    let parts: rhai::Array = msg.content.iter().map(content_part_to_dynamic).collect();
    map.insert("content_parts".into(), Dynamic::from_array(parts));

    Dynamic::from_map(map)
}

/// Convert a ContentPart to a Rhai Dynamic map.
///
/// All type names use consistent snake_case for Rhai script consumers.
/// Explicit arms for all ContentPart variants.
fn content_part_to_dynamic(part: &ContentPart) -> Dynamic {
    let mut map = rhai::Map::new();
    match part {
        ContentPart::Text(t) => {
            map.insert("type".into(), Dynamic::from("text".to_string()));
            map.insert("text".into(), Dynamic::from(t.clone()));
        }
        ContentPart::CommandCall(cc) => {
            map.insert("type".into(), Dynamic::from("command_call".to_string()));
            map.insert("command".into(), Dynamic::from(cc.command.clone()));
            map.insert(
                "arguments_json".into(),
                Dynamic::from(cc.arguments_json.clone()),
            );
        }
        ContentPart::CommandResult(cr) => {
            map.insert("type".into(), Dynamic::from("command_result".to_string()));
            map.insert("command".into(), Dynamic::from(cr.command.clone()));
            map.insert("success".into(), Dynamic::from(cr.success));
            map.insert("output".into(), Dynamic::from(cr.output.clone()));
        }
        ContentPart::MemoryResult(_) => {
            map.insert("type".into(), Dynamic::from("memory_result".to_string()));
        }
        ContentPart::MemoryStored(_) => {
            map.insert("type".into(), Dynamic::from("memory_stored".to_string()));
        }
        ContentPart::CouncilStart(_) => {
            map.insert("type".into(), Dynamic::from("council_start".to_string()));
        }
        ContentPart::Image(_) => {
            map.insert("type".into(), Dynamic::from("image".to_string()));
        }
        ContentPart::Audio(_) => {
            map.insert("type".into(), Dynamic::from("audio".to_string()));
        }
        ContentPart::Video(_) => {
            map.insert("type".into(), Dynamic::from("video".to_string()));
        }
        ContentPart::Document(_) => {
            map.insert("type".into(), Dynamic::from("document".to_string()));
        }
        ContentPart::Routing(_) => {
            map.insert("type".into(), Dynamic::from("routing".to_string()));
        }
        ContentPart::Hook(_) => {
            map.insert("type".into(), Dynamic::from("hook".to_string()));
        }
    }
    Dynamic::from_map(map)
}

/// Helper: `Option<f32>` -> Dynamic (f64 or UNIT)
fn option_to_dynamic_f64(opt: Option<f32>) -> Dynamic {
    match opt {
        Some(v) => Dynamic::from(v as f64),
        None => Dynamic::UNIT,
    }
}

/// Helper: `Option<u32>` -> Dynamic (i64 or UNIT)
fn option_to_dynamic_i64(opt: Option<u32>) -> Dynamic {
    match opt {
        Some(v) => Dynamic::from(v as i64),
        None => Dynamic::UNIT,
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
    let json =
        weft_rhai::dynamic_to_json(&dynamic).map_err(|e| ProviderError::WireScriptError {
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
///
/// The `model` parameter is used to attribute the response `WeftMessage` to its source model.
fn dynamic_to_provider_result(
    dynamic: Dynamic,
    script_path: &str,
    model: &str,
) -> Result<ProviderResponse, ProviderError> {
    let json =
        weft_rhai::dynamic_to_json(&dynamic).map_err(|e| ProviderError::WireScriptError {
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

            let message = WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some(model.to_string()),
                content: vec![ContentPart::Text(text)],
                delta: false,
                message_index: 0,
            };

            Ok(ProviderResponse::ChatCompletion { message, usage })
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
        let script_path = self.script.path().to_string();

        // Extract model for response attribution before converting to dynamic.
        let model = match &request {
            ProviderRequest::ChatCompletion { model, .. } => model.clone(),
        };

        // Step 1: Convert the ProviderRequest to a Rhai Dynamic.
        let request_dynamic = request_to_dynamic(&request);

        // Step 2: Call format_request(request_map) via safe_call_fn (CPU-bound,
        // wrapped in spawn_blocking + catch_unwind).
        let format_result = safe_call_fn(
            Arc::clone(&self.engine),
            &self.script,
            "format_request",
            (request_dynamic,),
        )
        .await
        .map_err(|e| {
            warn!(
                script = %script_path,
                provider = %self.provider_name,
                error = %e,
                "format_request failed"
            );
            script_error_to_provider(e, &script_path)
        })?;

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

        // Step 4b: Wrap the HTTP round-trip in a `provider_call` span.
        let provider_call_span = info_span!(
            "provider_call",
            http.request.method = %spec.method,
            url.full = %url,
            otel.kind = "client",
            http.response.status_code = tracing::field::Empty,
        );

        let http_response = async {
            req_builder
                .send()
                .await
                .map_err(|e| ProviderError::RequestFailed(format!("{}: {}", self.provider_name, e)))
        }
        .instrument(provider_call_span.clone())
        .await?;

        // Step 5: Collect the HTTP response.
        let status = http_response.status().as_u16();
        provider_call_span.record("http.response.status_code", status as i64);
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

        // Step 7: Call parse_response(response_map) via safe_call_fn.
        let parse_result = safe_call_fn(
            Arc::clone(&self.engine),
            &self.script,
            "parse_response",
            (response_dynamic,),
        )
        .await
        .map_err(|e| {
            warn!(
                script = %script_path,
                provider = %self.provider_name,
                error = %e,
                "parse_response failed"
            );
            script_error_to_provider(e, &script_path)
        })?;

        // Step 8: Convert the returned map to ProviderResponse or ProviderError.
        dynamic_to_provider_result(parse_result, &script_path, &model)
    }

    fn name(&self) -> &str {
        &self.provider_name
    }
}

impl std::fmt::Debug for RhaiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RhaiProvider")
            .field("provider_name", &self.provider_name)
            .field("script_path", &self.script.path())
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use weft_core::{ContentPart, Role, SamplingOptions, Source, WeftMessage};
    use weft_rhai::{EngineBuilder, SandboxLimits};

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

    /// Build a minimal `ProviderRequest::ChatCompletion` using new inline struct fields.
    fn sample_chat_request() -> ProviderRequest {
        ProviderRequest::ChatCompletion {
            messages: vec![
                WeftMessage {
                    role: Role::System,
                    source: Source::Gateway,
                    model: None,
                    content: vec![ContentPart::Text("You are helpful.".to_string())],
                    delta: false,
                    message_index: 0,
                },
                WeftMessage {
                    role: Role::User,
                    source: Source::Client,
                    model: None,
                    content: vec![ContentPart::Text("Hello".to_string())],
                    delta: false,
                    message_index: 0,
                },
            ],
            model: "test-model".to_string(),
            options: SamplingOptions {
                max_tokens: Some(1024),
                temperature: Some(0.7),
                ..SamplingOptions::default()
            },
        }
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

        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        assert_eq!(json["type"], "chat_completion");
        assert_eq!(json["model"], "test-model");
        // No system_prompt top-level field -- system prompt is messages[0]
        assert!(
            !json.as_object().unwrap().contains_key("system_prompt"),
            "request map must not have system_prompt field"
        );
        assert_eq!(json["max_tokens"], 1024);
        assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);

        let messages = json["messages"].as_array().unwrap();
        // Both messages are present (Rhai passes all messages, no filtering)
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["source"], "gateway");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["source"], "client");
    }

    #[test]
    fn test_request_to_dynamic_messages_have_content_and_content_parts() {
        let req = ProviderRequest::ChatCompletion {
            messages: vec![WeftMessage {
                role: Role::User,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("Hello".to_string())],
                delta: false,
                message_index: 0,
            }],
            model: "m".to_string(),
            options: SamplingOptions::default(),
        };
        let dynamic = request_to_dynamic(&req);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        let msg = &json["messages"][0];
        // Backward-compat content field
        assert_eq!(msg["content"], "Hello");
        // Full content_parts array
        let parts = msg["content_parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Hello");
    }

    #[test]
    fn test_request_to_dynamic_no_temperature_is_unit() {
        let req = ProviderRequest::ChatCompletion {
            messages: vec![],
            model: "m".to_string(),
            options: SamplingOptions::default(),
        };
        let dynamic = request_to_dynamic(&req);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        // temperature should be null/absent when None
        assert!(
            json["temperature"].is_null()
                || !json.as_object().unwrap().contains_key("temperature")
                || json["temperature"] == serde_json::Value::Null
        );
    }

    #[test]
    fn test_request_to_dynamic_options_submap_present() {
        let req = ProviderRequest::ChatCompletion {
            messages: vec![],
            model: "m".to_string(),
            options: SamplingOptions {
                max_tokens: Some(512),
                temperature: Some(0.5),
                top_p: Some(0.9),
                top_k: Some(20),
                frequency_penalty: Some(0.1),
                presence_penalty: Some(0.2),
                seed: Some(99),
                stop: vec!["STOP".to_string()],
                ..SamplingOptions::default()
            },
        };
        let dynamic = request_to_dynamic(&req);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        let opts = &json["options"];
        assert_eq!(opts["max_tokens"], 512);
        assert!((opts["temperature"].as_f64().unwrap() - 0.5).abs() < 0.001);
        assert!((opts["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
        assert_eq!(opts["top_k"], 20);
        assert!((opts["frequency_penalty"].as_f64().unwrap() - 0.1).abs() < 0.001);
        assert!((opts["presence_penalty"].as_f64().unwrap() - 0.2).abs() < 0.001);
        assert_eq!(opts["seed"], 99);
        let stop = opts["stop"].as_array().unwrap();
        assert_eq!(stop.len(), 1);
        assert_eq!(stop[0], "STOP");
    }

    #[test]
    fn test_request_to_dynamic_message_roles() {
        let req = ProviderRequest::ChatCompletion {
            messages: vec![
                WeftMessage {
                    role: Role::User,
                    source: Source::Client,
                    model: None,
                    content: vec![ContentPart::Text("user msg".to_string())],
                    delta: false,
                    message_index: 0,
                },
                WeftMessage {
                    role: Role::Assistant,
                    source: Source::Provider,
                    model: None,
                    content: vec![ContentPart::Text("assistant msg".to_string())],
                    delta: false,
                    message_index: 0,
                },
                WeftMessage {
                    role: Role::System,
                    source: Source::Client,
                    model: None,
                    content: vec![ContentPart::Text("system msg".to_string())],
                    delta: false,
                    message_index: 0,
                },
            ],
            model: "m".to_string(),
            options: SamplingOptions::default(),
        };
        let dynamic = request_to_dynamic(&req);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "system");
    }

    #[test]
    fn test_request_to_dynamic_gateway_messages_not_filtered() {
        // Unlike OpenAI/Anthropic, Rhai provider passes ALL messages including
        // gateway activity -- the script decides what to include.
        let req = ProviderRequest::ChatCompletion {
            messages: vec![
                WeftMessage {
                    role: Role::System,
                    source: Source::Gateway,
                    model: None,
                    content: vec![ContentPart::Text("system prompt".to_string())],
                    delta: false,
                    message_index: 0,
                },
                WeftMessage {
                    role: Role::System,
                    source: Source::Gateway,
                    model: None,
                    content: vec![ContentPart::Text("routing activity".to_string())],
                    delta: false,
                    message_index: 0,
                },
                WeftMessage {
                    role: Role::User,
                    source: Source::Client,
                    model: None,
                    content: vec![ContentPart::Text("user".to_string())],
                    delta: false,
                    message_index: 0,
                },
            ],
            model: "m".to_string(),
            options: SamplingOptions::default(),
        };
        let dynamic = request_to_dynamic(&req);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        // All 3 messages present -- Rhai doesn't filter
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn test_content_part_to_dynamic_text() {
        let part = ContentPart::Text("hello".to_string());
        let dynamic = content_part_to_dynamic(&part);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
    }

    #[test]
    fn test_content_part_to_dynamic_command_call() {
        use weft_core::CommandCallContent;
        let part = ContentPart::CommandCall(CommandCallContent {
            command: "search".to_string(),
            arguments_json: r#"{"query":"test"}"#.to_string(),
        });
        let dynamic = content_part_to_dynamic(&part);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        assert_eq!(json["type"], "command_call");
        assert_eq!(json["command"], "search");
        assert_eq!(json["arguments_json"], r#"{"query":"test"}"#);
    }

    #[test]
    fn test_content_part_to_dynamic_command_result() {
        use weft_core::CommandResultContent;
        let part = ContentPart::CommandResult(CommandResultContent {
            command: "search".to_string(),
            success: true,
            output: "Found results".to_string(),
            error: None,
        });
        let dynamic = content_part_to_dynamic(&part);
        let json = weft_rhai::dynamic_to_json(&dynamic).unwrap();
        assert_eq!(json["type"], "command_result");
        assert_eq!(json["command"], "search");
        assert_eq!(json["success"], true);
        assert_eq!(json["output"], "Found results");
    }

    #[test]
    fn test_content_part_to_dynamic_type_names_are_snake_case() {
        use weft_core::{
            CouncilStartActivity, MemoryResultContent, MemoryStoredContent, RoutingActivity,
        };
        // Verify all type names use snake_case (no camelCase, no PascalCase)
        let memory_result =
            content_part_to_dynamic(&ContentPart::MemoryResult(MemoryResultContent {
                store: "s".to_string(),
                entries: vec![],
            }));
        let json = weft_rhai::dynamic_to_json(&memory_result).unwrap();
        assert_eq!(json["type"], "memory_result");

        let memory_stored =
            content_part_to_dynamic(&ContentPart::MemoryStored(MemoryStoredContent {
                store: "s".to_string(),
                id: "id1".to_string(),
            }));
        let json = weft_rhai::dynamic_to_json(&memory_stored).unwrap();
        assert_eq!(json["type"], "memory_stored");

        let council_start =
            content_part_to_dynamic(&ContentPart::CouncilStart(CouncilStartActivity {
                models: vec![],
                judge: "j".to_string(),
            }));
        let json = weft_rhai::dynamic_to_json(&council_start).unwrap();
        assert_eq!(json["type"], "council_start");

        let routing = content_part_to_dynamic(&ContentPart::Routing(RoutingActivity {
            model: "m".to_string(),
            score: 0.9,
            filters: vec![],
        }));
        let json = weft_rhai::dynamic_to_json(&routing).unwrap();
        assert_eq!(json["type"], "routing");
    }

    // ── extract_request_spec tests ────────────────────────────────────────────

    #[test]
    fn test_extract_request_spec_valid() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
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
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
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
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ path: "/v1", headers: #{}, body: "{}" }"#)
            .unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { ref message, .. } if message.contains("method")),
        );
    }

    #[test]
    fn test_extract_request_spec_missing_body_fails() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ method: "POST", path: "/v1", headers: #{} }"#)
            .unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), ProviderError::WireScriptError { ref message, .. } if message.contains("body")),
        );
    }

    #[test]
    fn test_extract_request_spec_non_map_fails() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine.eval(r#""just a string""#).unwrap();
        let result = extract_request_spec(dynamic, "test.rhai");
        assert!(result.is_err());
    }

    // ── dynamic_to_provider_result tests ─────────────────────────────────────

    #[test]
    fn test_dynamic_to_provider_result_chat_completion_success() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(
                r#"#{
                type: "chat_completion",
                text: "Hello!",
                usage: #{ prompt_tokens: 10, completion_tokens: 5 }
            }"#,
            )
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "gpt-4");
        assert!(result.is_ok());
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, usage } = result.unwrap() else {
            panic!("expected ChatCompletion");
        };
        // Verify the WeftMessage has correct fields
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.source, Source::Provider);
        assert_eq!(message.model, Some("gpt-4".to_string()));
        let text = message
            .content
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "Hello!");
        let u = usage.expect("usage should be present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
    }

    #[test]
    fn test_dynamic_to_provider_result_no_usage() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ type: "chat_completion", text: "ok" }"#)
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "model-x");
        assert!(result.is_ok());
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { usage, .. } = result.unwrap() else {
            panic!("expected ChatCompletion");
        };
        assert!(usage.is_none());
    }

    #[test]
    fn test_dynamic_to_provider_result_error_response() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ error: "provider unavailable" }"#)
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "m");
        assert!(matches!(
            result,
            Err(ProviderError::ProviderHttpError { .. })
        ));
    }

    #[test]
    fn test_dynamic_to_provider_result_rate_limited() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ error: "rate limited", retry_after_ms: 30000 }"#)
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "m");
        assert!(matches!(
            result,
            Err(ProviderError::RateLimited {
                retry_after_ms: 30000
            })
        ));
    }

    #[test]
    fn test_dynamic_to_provider_result_missing_text_fails() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine.eval(r#"#{ type: "chat_completion" }"#).unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "m");
        assert!(matches!(result, Err(ProviderError::WireScriptError { .. })));
    }

    #[test]
    fn test_dynamic_to_provider_result_unknown_type_fails() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ type: "embeddings", vectors: [] }"#)
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "m");
        assert!(matches!(result, Err(ProviderError::WireScriptError { .. })));
    }

    #[test]
    fn test_dynamic_to_provider_result_non_map_fails() {
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine.eval(r#""just a string""#).unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "m");
        assert!(matches!(result, Err(ProviderError::WireScriptError { .. })));
    }

    #[test]
    fn test_dynamic_to_provider_result_response_has_model_attribution() {
        // The response WeftMessage must carry the model identifier for attribution.
        let engine = EngineBuilder::new(SandboxLimits::default()).build();
        let dynamic: Dynamic = engine
            .eval(r#"#{ type: "chat_completion", text: "hi" }"#)
            .unwrap();
        let result = dynamic_to_provider_result(dynamic, "test.rhai", "claude-3-5-sonnet");
        assert!(result.is_ok());
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, .. } = result.unwrap() else {
            panic!("expected ChatCompletion");
        };
        assert_eq!(message.model, Some("claude-3-5-sonnet".to_string()));
    }

    // ── Integration tests (full execute path with mockito) ────────────────────

    #[tokio::test]
    async fn test_execute_chat_completion_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "test-key".to_string(),
            server.url(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(result.is_ok(), "expected ok, got: {:?}", result.err());
        #[allow(irrefutable_let_patterns)]
        let ProviderResponse::ChatCompletion { message, usage } = result.unwrap() else {
            panic!("expected ChatCompletion");
        };
        let text = message
            .content
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "Hi!");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.source, Source::Provider);
        assert_eq!(message.model, Some("test-model".to_string()));
        let u = usage.expect("usage should be present");
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 3);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_execute_provider_error_non_200() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(400)
            .with_body(r#"{"error": "bad request"}"#)
            .create_async()
            .await;

        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            server.url(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(matches!(
            result,
            Err(ProviderError::ProviderHttpError { .. })
        ));
    }

    #[tokio::test]
    async fn test_execute_rate_limited_429() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(429)
            .with_body("{}")
            .create_async()
            .await;

        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            server.url(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(matches!(result, Err(ProviderError::RateLimited { .. })));
    }

    #[tokio::test]
    async fn test_execute_format_request_script_error_returns_wire_script_error() {
        let script = r#"
fn format_request(request) {
    throw "deliberate error";
}
fn parse_response(r) { #{ type: "chat_completion", text: "ok" } }
"#;
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .create_async()
            .await;

        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            server.url(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(matches!(result, Err(ProviderError::WireScriptError { .. })));
    }

    #[tokio::test]
    async fn test_execute_parse_response_script_error_returns_wire_script_error() {
        let script = r#"
fn format_request(r) {
    #{ method: "POST", path: "/v1", headers: #{}, body: "{}" }
}
fn parse_response(response) {
    throw "parse error";
}
"#;
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1")
            .with_status(200)
            .with_body("{}")
            .create_async()
            .await;

        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            server.url(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(matches!(result, Err(ProviderError::WireScriptError { .. })));
    }

    #[tokio::test]
    async fn test_execute_parse_response_returning_unexpected_type_returns_wire_script_error() {
        let script = r#"
fn format_request(r) {
    #{ method: "POST", path: "/v1", headers: #{}, body: "{}" }
}
fn parse_response(response) {
    #{ type: "unknown_type", data: "whatever" }
}
"#;
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1")
            .with_status(200)
            .with_body("{}")
            .create_async()
            .await;

        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            server.url(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(matches!(result, Err(ProviderError::WireScriptError { .. })));
    }

    #[tokio::test]
    async fn test_execute_script_exceeding_operation_limit_returns_wire_script_error() {
        let script = r#"
fn format_request(r) {
    // Infinite loop -- should be killed by operation limit
    let x = 0;
    loop { x += 1; }
}
fn parse_response(r) { #{ type: "chat_completion", text: "ok" } }
"#;
        let f = write_temp_script(script);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        )
        .unwrap();

        let result = provider.execute(sample_chat_request()).await;
        assert!(
            matches!(result, Err(ProviderError::WireScriptError { .. })),
            "expected WireScriptError from operation limit, got: {:?}",
            result
        );
    }

    #[test]
    fn test_operation_limit_prevents_infinite_loop() {
        // Verify that the operation limit blocks infinite loops in Rhai.
        let engine = EngineBuilder::new(SandboxLimits::relaxed()).build();
        let result: Result<(), _> = engine.eval(
            r#"
            let x = 0;
            loop { x += 1; }
        "#,
        );
        assert!(
            result.is_err(),
            "infinite loop should be blocked by operation limit"
        );
    }

    #[test]
    fn test_debug_format_does_not_leak_api_key() {
        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "super-secret-key".to_string(),
            "https://api.example.com".to_string(),
            "test-provider".to_string(),
        )
        .unwrap();
        let debug_str = format!("{provider:?}");
        assert!(
            !debug_str.contains("super-secret-key"),
            "debug output must not contain api_key"
        );
    }

    #[test]
    fn test_name_returns_provider_name() {
        let f = write_temp_script(MINIMAL_VALID_SCRIPT);
        let provider = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "my-custom-provider".to_string(),
        )
        .unwrap();
        assert_eq!(provider.name(), "my-custom-provider");
    }

    #[test]
    fn test_json_encode_decode_roundtrip() {
        let script = r#"
fn format_request(r) {
    let encoded = json_encode(#{ key: "value", num: 42 });
    let decoded = json_decode(encoded);
    #{
        method: "POST",
        path: "/v1",
        headers: #{},
        body: encoded,
    }
}
fn parse_response(r) { #{ type: "chat_completion", text: "ok" } }
"#;
        let f = write_temp_script(script);
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_base64_encode_accessible_in_script() {
        let script = r#"
fn format_request(r) {
    let encoded = base64_encode("hello");
    #{ method: "POST", path: "/v1", headers: #{}, body: encoded }
}
fn parse_response(r) { #{ type: "chat_completion", text: "ok" } }
"#;
        let f = write_temp_script(script);
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_base64_decode_accessible_in_script() {
        let script = r#"
fn format_request(r) {
    let decoded = base64_decode("aGVsbG8=");
    #{ method: "POST", path: "/v1", headers: #{}, body: decoded }
}
fn parse_response(r) { #{ type: "chat_completion", text: "ok" } }
"#;
        let f = write_temp_script(script);
        let result = RhaiProvider::new(
            f.path().to_str().unwrap(),
            "key".to_string(),
            "https://api.example.com".to_string(),
            "test".to_string(),
        );
        assert!(result.is_ok());
    }
}
