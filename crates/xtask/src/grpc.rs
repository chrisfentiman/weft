use xshell::{Shell, cmd};

use crate::util::{Result, json_escape};
use crate::{GrpcArgs, GrpcCommand};

/// Dispatch `cargo xtask grpc <subcommand>`.
pub(crate) fn run(sh: &Shell, args: GrpcArgs) -> Result<()> {
    match args.command {
        GrpcCommand::Chat(chat_args) => run_chat(sh, chat_args),
        GrpcCommand::Health(health_args) => run_health(sh, health_args),
        GrpcCommand::Ls => run_ls(sh),
    }
}

/// Run `cargo xtask grpc chat`: send a Chat gRPC request via grpcurl.
///
/// Requires `grpcurl` on PATH. If not found, prints a clear installation hint.
fn run_chat(sh: &Shell, args: crate::GrpcChatArgs) -> Result<()> {
    // Verify grpcurl is available before doing any work.
    if cmd!(sh, "grpcurl --version")
        .quiet()
        .ignore_status()
        .run()
        .is_err()
    {
        eprintln!("[xtask] error: grpcurl not found. Install with: brew install grpcurl");
        return Err("grpcurl not found on PATH".into());
    }

    // Construct the JSON payload programmatically to handle quotes/special chars.
    let content = json_escape(&args.message);
    let payload = format!(r#"{{"messages":[{{"role":"USER","content":{content}}}]}}"#);

    let addr = args.addr.as_str();

    cmd!(
        sh,
        "grpcurl -plaintext
            -import-path crates/weft_proto/proto
            -proto weft.proto
            -d {payload}
            {addr}
            weft.v1.Weft/Chat"
    )
    .run()?;

    Ok(())
}

/// Run `cargo xtask grpc health`: check weft health via HTTP.
///
/// Uses `curl -s` (universally available on macOS/Linux).
/// Does NOT require grpcurl — the health endpoint is plain HTTP, not gRPC.
fn run_health(sh: &Shell, args: crate::GrpcHealthArgs) -> Result<()> {
    let addr = args.addr.as_str();
    cmd!(sh, "curl -s http://{addr}/health").run()?;
    Ok(())
}

/// Run `cargo xtask grpc ls`: list available RPC methods from the proto definition.
///
/// Reads `crates/weft_proto/proto/weft.proto` relative to the workspace root,
/// parses service and RPC declarations, and prints a formatted table showing
/// the method name, streaming type, and grpcurl-compatible method path.
fn run_ls(sh: &Shell) -> Result<()> {
    let root = sh.current_dir();
    let proto_path = root.join("crates/weft_proto/proto/weft.proto");

    if !proto_path.exists() {
        eprintln!("[xtask] error: proto file not found at crates/weft_proto/proto/weft.proto");
        return Err("proto file not found".into());
    }

    let content = std::fs::read_to_string(&proto_path)
        .map_err(|e| format!("failed to read {}: {e}", proto_path.display()))?;

    // Extract the package name (e.g., "weft.v1") from the proto file.
    let package = extract_proto_package(&content).unwrap_or_default();

    let services = parse_proto_services(&content, &package);

    if services.is_empty() {
        eprintln!("[xtask] no RPC services found in proto file");
        return Ok(());
    }

    for (service_name, methods) in &services {
        println!("Service {service_name}:");

        // Calculate column width for method names for aligned output.
        let max_method_len = methods.iter().map(|m| m.method.len()).max().unwrap_or(0);

        for m in methods {
            let streaming_label = match m.streaming {
                StreamingType::Unary => "(unary)         ",
                StreamingType::ServerStreaming => "(server stream) ",
                StreamingType::ClientStreaming => "(client stream) ",
                StreamingType::Bidirectional => "(bidirectional) ",
            };
            let grpcurl_path = format!("{service_name}/{}", m.method);
            println!(
                "  {:<width$}  {streaming_label} — grpcurl method: {grpcurl_path}",
                m.method,
                width = max_method_len
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Proto parsing
// ---------------------------------------------------------------------------

/// An RPC method discovered from the proto file.
pub(crate) struct RpcMethod {
    /// Method name (e.g., "Chat").
    pub(crate) method: String,
    /// Streaming type determined from request/response stream keywords.
    pub(crate) streaming: StreamingType,
}

/// Streaming type of an RPC method.
pub(crate) enum StreamingType {
    Unary,
    ServerStreaming,
    ClientStreaming,
    Bidirectional,
}

/// Extract the `package` declaration from a proto file.
///
/// Returns `None` if no `package` line is found.
fn extract_proto_package(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("package ") {
            // Strip trailing semicolon and whitespace.
            let pkg = rest.trim_end_matches(';').trim();
            if !pkg.is_empty() {
                return Some(pkg.to_owned());
            }
        }
    }
    None
}

/// Parse all services and their RPC methods from a proto file.
///
/// Returns a vec of `(fully_qualified_service_name, methods)` pairs.
/// The service name is qualified with the package prefix if one is present
/// (e.g., `"weft.v1.Weft"` when package is `"weft.v1"` and service is `"Weft"`).
///
/// This parser handles proto3 syntax as used in `weft.proto`. It is NOT a
/// general protobuf parser — it relies on the specific formatting conventions
/// of the controlled proto file.
pub(crate) fn parse_proto_services(content: &str, package: &str) -> Vec<(String, Vec<RpcMethod>)> {
    let mut services: Vec<(String, Vec<RpcMethod>)> = Vec::new();
    let mut current_service: Option<String> = None;
    let mut current_methods: Vec<RpcMethod> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Detect `service <name> {`
        if let Some(rest) = trimmed.strip_prefix("service ") {
            // Close any previously open service first (shouldn't happen in valid proto, but safe).
            if let Some(svc) = current_service.take() {
                services.push((svc, std::mem::take(&mut current_methods)));
            }
            // Extract service name: everything before the first `{` or whitespace.
            let name = rest
                .split(|c: char| c == '{' || c.is_whitespace())
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                let fq_name = if package.is_empty() {
                    name.to_owned()
                } else {
                    format!("{package}.{name}")
                };
                current_service = Some(fq_name);
            }
            continue;
        }

        // Detect closing `}` — ends the current service block.
        if trimmed == "}" {
            if let Some(svc) = current_service.take() {
                services.push((svc, std::mem::take(&mut current_methods)));
            }
            continue;
        }

        // Detect `rpc <name>(<input>) returns (<output>);`
        if current_service.is_some()
            && let Some(method) = parse_rpc_line(trimmed)
        {
            current_methods.push(method);
        }
    }

    // Handle unclosed service block (malformed proto, but don't panic).
    if let Some(svc) = current_service {
        services.push((svc, current_methods));
    }

    services
}

/// Parse a single `rpc ...` line and return an `RpcMethod` if successful.
///
/// Handles the proto3 RPC syntax:
/// ```text
/// rpc <Name>(<maybe "stream "><InputType>) returns (<maybe "stream "><OutputType>);
/// ```
///
/// Returns `None` for lines that do not match.
fn parse_rpc_line(line: &str) -> Option<RpcMethod> {
    let rest = line.strip_prefix("rpc ")?;

    // Extract method name: up to the first `(`.
    let paren_pos = rest.find('(')?;
    let method = rest[..paren_pos].trim().to_owned();
    if method.is_empty() {
        return None;
    }

    // Extract the input type region: between first `(` and first `)`.
    let after_open = &rest[paren_pos + 1..];
    let close_pos = after_open.find(')')?;
    let input_region = &after_open[..close_pos];

    // Extract the output type region: after `returns (`, before `)`.
    let after_input_close = &after_open[close_pos + 1..];
    let returns_pos = after_input_close.find("returns")?;
    let after_returns = &after_input_close[returns_pos + "returns".len()..];
    let out_open = after_returns.find('(')?;
    let after_out_open = &after_returns[out_open + 1..];
    let out_close = after_out_open.find(')')?;
    let output_region = &after_out_open[..out_close];

    // Determine streaming from `stream` keyword presence.
    let input_stream = input_region.trim_start().starts_with("stream ");
    let output_stream = output_region.trim_start().starts_with("stream ");

    let streaming = match (input_stream, output_stream) {
        (false, false) => StreamingType::Unary,
        (false, true) => StreamingType::ServerStreaming,
        (true, false) => StreamingType::ClientStreaming,
        (true, true) => StreamingType::Bidirectional,
    };

    Some(RpcMethod { method, streaming })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_unary_rpc() {
        let proto = r#"
service Weft {
  rpc Chat(ChatRequest) returns (ChatResponse);
}
"#;
        let services = parse_proto_services(proto, "weft.v1");
        assert_eq!(services.len(), 1);
        let (name, methods) = &services[0];
        assert_eq!(name, "weft.v1.Weft");
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].method, "Chat");
        assert!(matches!(methods[0].streaming, StreamingType::Unary));
    }

    #[test]
    fn test_parse_server_streaming_rpc() {
        let proto = r#"
service Weft {
  rpc ChatStream(ChatRequest) returns (stream ChatEvent);
}
"#;
        let services = parse_proto_services(proto, "weft.v1");
        assert_eq!(services[0].1[0].method, "ChatStream");
        assert!(matches!(
            services[0].1[0].streaming,
            StreamingType::ServerStreaming
        ));
    }

    #[test]
    fn test_parse_bidirectional_rpc() {
        let proto = r#"
service Weft {
  rpc Live(stream LiveInput) returns (stream LiveOutput);
}
"#;
        let services = parse_proto_services(proto, "weft.v1");
        assert_eq!(services[0].1[0].method, "Live");
        assert!(matches!(
            services[0].1[0].streaming,
            StreamingType::Bidirectional
        ));
    }

    #[test]
    fn test_parse_client_streaming_rpc() {
        let proto = r#"
service Upload {
  rpc Send(stream Chunk) returns (UploadResponse);
}
"#;
        let services = parse_proto_services(proto, "test.v1");
        assert_eq!(services[0].1[0].method, "Send");
        assert!(matches!(
            services[0].1[0].streaming,
            StreamingType::ClientStreaming
        ));
    }

    #[test]
    fn test_parse_multiple_rpcs() {
        let proto = r#"
service Weft {
  rpc Chat(ChatRequest) returns (ChatResponse);
  rpc ChatStream(ChatRequest) returns (stream ChatEvent);
  rpc Live(stream LiveInput) returns (stream LiveOutput);
}
"#;
        let services = parse_proto_services(proto, "weft.v1");
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].1.len(), 3);
    }

    #[test]
    fn test_parse_multiple_services() {
        let proto = r#"
service ServiceA {
  rpc Foo(FooReq) returns (FooResp);
}
service ServiceB {
  rpc Bar(BarReq) returns (stream BarResp);
}
"#;
        let services = parse_proto_services(proto, "pkg.v1");
        assert_eq!(services.len(), 2);
        assert_eq!(services[0].0, "pkg.v1.ServiceA");
        assert_eq!(services[1].0, "pkg.v1.ServiceB");
        assert_eq!(services[0].1.len(), 1);
        assert_eq!(services[1].1.len(), 1);
    }

    #[test]
    fn test_extract_proto_package() {
        let content = "syntax = \"proto3\";\n\npackage weft.v1;\n\nservice Weft {}";
        assert_eq!(extract_proto_package(content), Some("weft.v1".to_string()));
    }

    #[test]
    fn test_extract_proto_package_missing() {
        let content = "syntax = \"proto3\";\n\nservice Weft {}";
        assert_eq!(extract_proto_package(content), None);
    }

    #[test]
    fn test_no_package_prefix() {
        let proto = r#"
service Simple {
  rpc Ping(PingReq) returns (PingResp);
}
"#;
        // Empty package string means no prefix added.
        let services = parse_proto_services(proto, "");
        assert_eq!(services[0].0, "Simple");
    }

    // Verify that the actual weft.proto parses correctly.
    #[test]
    fn test_parse_weft_proto_methods() {
        let proto = r#"
syntax = "proto3";
package weft.v1;
service Weft {
  rpc Chat(ChatRequest) returns (ChatResponse);
  rpc ChatStream(ChatRequest) returns (stream ChatEvent);
  rpc Live(stream LiveInput) returns (stream LiveOutput);
}
"#;
        let package = extract_proto_package(proto).unwrap_or_default();
        let services = parse_proto_services(proto, &package);
        assert_eq!(services.len(), 1);
        let (svc_name, methods) = &services[0];
        assert_eq!(svc_name, "weft.v1.Weft");
        assert_eq!(methods.len(), 3);

        let chat = methods.iter().find(|m| m.method == "Chat").unwrap();
        assert!(matches!(chat.streaming, StreamingType::Unary));

        let chat_stream = methods.iter().find(|m| m.method == "ChatStream").unwrap();
        assert!(matches!(
            chat_stream.streaming,
            StreamingType::ServerStreaming
        ));

        let live = methods.iter().find(|m| m.method == "Live").unwrap();
        assert!(matches!(live.streaming, StreamingType::Bidirectional));
    }
}
