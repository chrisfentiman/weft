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
/// Full implementation is in Phase 2. This stub notifies the user.
fn run_ls(_sh: &Shell) -> Result<()> {
    eprintln!(
        "[xtask] grpc ls not yet implemented (Phase 2). \
         Proto method discovery is coming in the next phase."
    );
    Ok(())
}
