// providers/example.rhai
//
// Template for a custom wire format script. Copy this file and adapt it to
// your provider's API. Place the script anywhere on the filesystem and set
// `wire_script = "path/to/your/script.rhai"` in your provider config block.
//
// Config example:
//
//   [[router.providers]]
//   name = "my-provider"
//   wire_format = "custom"
//   wire_script = "providers/my-provider.rhai"
//   api_key = "env:MY_PROVIDER_KEY"
//   base_url = "https://api.my-provider.com"
//
//   [[router.providers.models]]
//   name = "my-model"
//   model = "my-model-v1"
//   max_tokens = 4096
//   capabilities = ["chat_completions"]
//   examples = ["Tell me a story", "Help me debug this code"]
//
// ============================================================================
// REQUIRED FUNCTIONS
// ============================================================================
//
// Every wire script MUST define both functions below with exactly these names.
// Missing either function causes a startup error.

// ─── format_request(request) ─────────────────────────────────────────────────
//
// Transform a Weft ProviderRequest into an HTTP request specification.
//
// Input — `request` is an object map with:
//   - type         String     "chat_completion" (other types may be added later)
//   - model        String     the model identifier from your model config
//   - system_prompt String    the assembled system prompt
//   - messages     Array      list of #{role, content} maps
//                             role is one of "user", "assistant", "system"
//   - max_tokens   Integer    maximum tokens in the response
//   - temperature  Float|()   sampling temperature; () means use provider default
//
// Output — must return an object map with:
//   - method   String         HTTP method: "POST", "GET", "PUT", or "DELETE"
//   - path     String         URL path appended to base_url (e.g. "/v1/chat/completions")
//   - headers  Map            optional extra HTTP headers (String -> String)
//                             Authorization is always set from api_key; do not repeat it
//                             Content-Type defaults to "application/json" if not set here
//   - body     String         the serialized request body (usually JSON via json_encode)
//
fn format_request(request) {
    // Build the message list for the provider's API.
    // This example follows OpenAI-compatible format; adapt as needed.
    let messages = [];
    if request.system_prompt != "" {
        messages.push(#{
            role: "system",
            content: request.system_prompt,
        });
    }
    for msg in request.messages {
        messages.push(#{
            role: msg.role,
            content: msg.content,
        });
    }

    // Build the request body.
    let body_map = #{
        model: request.model,
        messages: messages,
        max_tokens: request.max_tokens,
    };

    // Include temperature only when explicitly set (not unit/null).
    if request.temperature != () {
        body_map.temperature = request.temperature;
    }

    #{
        method: "POST",
        path: "/v1/chat/completions",
        headers: #{},
        body: json_encode(body_map),
    }
}

// ─── parse_response(response) ────────────────────────────────────────────────
//
// Transform a provider HTTP response into a Weft ProviderResponse.
//
// Input — `response` is an object map with:
//   - status   Integer    HTTP status code (200, 429, 503, etc.)
//   - body     String     raw response body (usually JSON)
//   - headers  Map        response headers (String -> String)
//
// Output — must return an object map. Two shapes are supported:
//
//   Success shape:
//     - type   String         "chat_completion" (must match the request type)
//     - text   String         the assistant's response text
//     - usage  Map|()         optional #{prompt_tokens, completion_tokens}
//                             return () to omit usage from the response
//
//   Error shape (causes an error to propagate to the gateway):
//     - error          String     human-readable error message
//     - retry_after_ms Integer    optional; present for rate limiting (HTTP 429)
//                                 causes RateLimited error with the given delay
//
fn parse_response(response) {
    // Rate limiting — return a retry hint.
    if response.status == 429 {
        let retry_ms = 60000; // default 60s

        // Try to extract Retry-After from headers (value in seconds).
        if response.headers["retry-after"] != () {
            let after_secs = parse_int(response.headers["retry-after"]);
            if after_secs > 0 {
                retry_ms = after_secs * 1000;
            }
        }

        return #{
            error: "rate limited",
            retry_after_ms: retry_ms,
        };
    }

    // Non-success status — return an error.
    if response.status != 200 {
        return #{
            error: `Provider returned ${response.status}: ${response.body}`,
        };
    }

    // Parse the JSON response body.
    let body = json_decode(response.body);

    // Extract the response text.
    // Adapt this path to match your provider's response structure.
    let text = body.choices[0].message.content;

    // Extract token usage if available.
    let usage = ();
    if body.usage != () {
        usage = #{
            prompt_tokens: body.usage.prompt_tokens,
            completion_tokens: body.usage.completion_tokens,
        };
    }

    #{
        type: "chat_completion",
        text: text,
        usage: usage,
    }
}

// ============================================================================
// AVAILABLE FUNCTIONS
// ============================================================================
//
// The following helper functions are registered by Weft and available to your
// script. No other external calls are possible — scripts are sandboxed.
//
// json_encode(value) -> String
//   Serialize a Rhai value (object map, array, string, number, etc.) to a JSON
//   string. Returns "null" if the value cannot be serialized.
//
// json_decode(s) -> Dynamic
//   Deserialize a JSON string to a Rhai value. Returns () (unit/null) on error.
//   Access fields like: let body = json_decode(response.body); body.choices[0]
//
// log_info(msg)
//   Emit an INFO-level log line. Visible in gateway logs.
//
// log_warn(msg)
//   Emit a WARN-level log line. Use for unexpected-but-recoverable situations.
//
// base64_encode(s) -> String
//   Base64-encode a string (standard alphabet, no line breaks).
//
// base64_decode(s) -> String
//   Base64-decode a string. Returns empty string on invalid input.
//
// ============================================================================
// SANDBOX LIMITS
// ============================================================================
//
// Scripts are subject to the following hard limits to prevent runaway execution:
//   - max_operations: 10,000  (prevents infinite loops)
//   - max_string_size: 1 MB   (prevents large string allocation)
//   - max_array_size: 4,096   (prevents large array allocation)
//   - max_map_size: 512       (prevents large map allocation)
//
// Scripts that exceed any limit return a WireScriptError, which the gateway
// treats as a provider failure and applies the fallback logic.
