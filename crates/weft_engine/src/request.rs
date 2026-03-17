//! Request handling and the main gateway loop.
//!
//! Provides `handle_request` (timeout wrapper, hook lifecycle, RequestEnd) and
//! `run_loop` (the inner routing-LLM-command loop).

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};
use weft_commands::{CommandRegistry, parse_response};
use weft_core::{
    CommandAction, CommandResult, ContentPart, HookEvent, Role, SamplingOptions, Source, WeftError,
    WeftMessage, WeftRequest, WeftResponse,
};
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::{Capability, ProviderRequest, ProviderService};
use weft_memory::MemoryService;
use weft_router::SemanticRouter;

use crate::GatewayEngine;
use crate::activity::assemble_response;
use crate::context::{
    assemble_system_prompt, assemble_system_prompt_no_tools, format_command_results_toon,
};
use crate::util::{ActivityEvent, BUILTIN_COMMANDS, extract_latest_user_text};

impl<H, R, M, P, C> GatewayEngine<H, R, M, P, C>
where
    H: HookRunner + Send + Sync + 'static,
    R: SemanticRouter + Send + Sync + 'static,
    M: MemoryService,
    P: ProviderService + Send + Sync + 'static,
    C: CommandRegistry + Send + Sync + 'static,
{
    /// Handle a single chat completion request.
    ///
    /// This is the main gateway loop: route → assemble → LLM → parse → execute → loop.
    pub async fn handle_request(&self, request: WeftRequest) -> Result<WeftResponse, WeftError> {
        // ── [HOOK: RequestStart] ─────────────────────────────────────────────
        // Hard block — fires before any routing or LLM involvement.
        // WeftRequest does not derive Serialize, so the payload is built manually.
        // Hook modifications to the payload are logged but not applied back to the
        // WeftRequest (the hook may inspect/block the request, but full deserialization
        // back to WeftRequest is deferred to a future phase when serde support is added).
        let request_payload = serde_json::json!({
            "model": request.routing.raw,
            "message_count": request.messages.len(),
            "temperature": request.options.temperature,
            "max_tokens": request.options.max_tokens,
        });
        match self
            .hooks
            .run_chain(HookEvent::RequestStart, request_payload, None)
            .await
        {
            HookChainResult::Blocked { reason, hook_name } => {
                info!(
                    hook = %hook_name,
                    reason = %reason,
                    "RequestStart hook blocked request"
                );
                return Err(WeftError::HookBlocked {
                    event: "RequestStart".to_string(),
                    reason,
                    hook_name,
                });
            }
            HookChainResult::Allowed {
                payload: _,
                context: _,
            } => {
                // Context from RequestStart is accumulated along with routing context.
                // Full WeftRequest modification via hook payload deferred to future phase.
            }
        }

        let timeout_secs = self.config.gateway.request_timeout_secs;
        let timeout = Duration::from_secs(timeout_secs);

        // Clone the routing raw string for RequestEnd telemetry.
        let request_model = request.routing.raw.clone();

        // Record start time for duration_ms in RequestEnd payload.
        let request_start = Instant::now();

        // Wrap the entire gateway loop in a timeout.
        let result = tokio::time::timeout(timeout, self.run_loop(request))
            .await
            .map_err(|_| WeftError::RequestTimeout { timeout_secs })?;

        let duration_ms = request_start.elapsed().as_millis() as u64;

        // ── [HOOK: RequestEnd] ───────────────────────────────────────────────
        // Fire-and-forget after the response is returned. Gated by semaphore.
        // We fire it regardless of success/failure.
        let (status, commands_executed) = match &result {
            Ok((_, count)) => (200u16, *count),
            Err(WeftError::HookBlocked { .. }) => (403u16, 0u32),
            Err(WeftError::HookBlockedAfterRetries { .. }) => (422u16, 0u32),
            Err(WeftError::RateLimited { .. }) => (429u16, 0u32),
            Err(WeftError::RequestTimeout { .. }) => (504u16, 0u32),
            Err(_) => (500u16, 0u32),
        };

        let end_payload = serde_json::json!({
            "request_id": uuid::Uuid::new_v4().to_string(),
            "duration_ms": duration_ms,
            "model": request_model,
            "status": status,
            "commands_executed": commands_executed,
        });

        let hooks_clone = Arc::clone(&self.hooks);
        let semaphore = Arc::clone(&self.request_end_semaphore);

        match semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                tokio::spawn(async move {
                    let _permit = permit; // Dropped when task completes.
                    hooks_clone
                        .run_chain(HookEvent::RequestEnd, end_payload, None)
                        .await;
                });
            }
            Err(_) => {
                warn!("RequestEnd semaphore exhausted — dropping RequestEnd hook task");
            }
        }

        // Strip the commands_executed count from the result before returning.
        result.map(|(response, _)| response)
    }

    /// Inner gateway loop (no timeout wrapping — caller handles that).
    ///
    /// Returns the response and the total number of commands executed during this request.
    pub(crate) async fn run_loop(
        &self,
        request: WeftRequest,
    ) -> Result<(WeftResponse, u32), WeftError> {
        // Verify that at least one model supports the required capability before routing.
        // For chat completions (the only implemented endpoint), the required capability is
        // always chat_completions. If no model supports this, fail fast with 400.
        let required_capability = Capability::new(Capability::CHAT_COMPLETIONS);
        let eligible_models = self.providers.models_with_capability(&required_capability);
        if eligible_models.is_empty() {
            return Err(WeftError::NoEligibleModels {
                capability: Capability::CHAT_COMPLETIONS.to_string(),
            });
        }

        // Get all available commands from the registry.
        let all_commands = self
            .commands
            .list_commands()
            .await
            .map_err(|e| WeftError::Command(e.to_string()))?;

        // Extract the latest user message text for semantic routing.
        let user_text = extract_latest_user_text(&request.messages).ok_or_else(|| {
            WeftError::Config(
                "request must contain at least one user message with text content".to_string(),
            )
        })?;

        // Build all routing domains and call the router once — with per-domain PreRoute/PostRoute hooks.
        let (routing_result, hook_context) =
            self.route_with_hooks(&user_text, &all_commands).await?;

        let selected_commands = routing_result.selected_commands;
        let inject_tools = routing_result.inject_tools;
        let selected_model_name = routing_result.selected_model;

        // Collect activity events for the response.
        // Routing events are always gathered here — they are only included in the response
        // message list when `request.options.activity = true` (handled by assemble_response).
        let activity_events: Vec<ActivityEvent> = routing_result
            .activity_events
            .into_iter()
            .map(ActivityEvent::Routing)
            .collect();

        debug!(
            model = %selected_model_name,
            inject_tools = inject_tools,
            selected_commands = selected_commands.len(),
            "routing decision applied"
        );

        // Look up the provider and model_id for the selected model.
        let provider = self.providers.get(&selected_model_name);
        let model_id = self
            .providers
            .model_id(&selected_model_name)
            .map(String::from);
        let max_tokens = self
            .providers
            .max_tokens_for(&selected_model_name)
            .or(request.options.max_tokens);

        // Build memory stubs to append when memory stores are configured.
        // These appear regardless of semantic routing — memory commands are always available.
        let memory_is_configured = self.memory.as_ref().is_some_and(|m| m.is_configured());
        let memory_stubs: Option<&[(&str, &str)]> = if memory_is_configured {
            Some(&[
                (
                    "recall",
                    "you need to retrieve relevant context, past conversations, or knowledge from memory",
                ),
                (
                    "remember",
                    "the user shares important information, preferences, or decisions worth preserving",
                ),
            ])
        } else {
            None
        };

        // Assemble the system prompt (with or without commands).
        // Append any accumulated hook context from routing hooks.
        let base_system_prompt = if inject_tools {
            assemble_system_prompt(
                &selected_commands,
                &self.config.gateway.system_prompt,
                memory_stubs,
            )
        } else {
            assemble_system_prompt_no_tools(&self.config.gateway.system_prompt)
        };

        let system_prompt = if let Some(ctx) = hook_context {
            format!("{base_system_prompt}\n{ctx}")
        } else {
            base_system_prompt
        };

        // Build the initial message list (clone from request).
        let mut messages = request.messages.clone();

        // Build the set of known command names for the parser.
        // Always include built-in commands (recall/remember) when memory is configured —
        // they are intercepted before the registry, so the parser must recognise them.
        // If tool skipping is active, use only built-in names so stray external slash
        // commands in the LLM response are treated as prose (spec Section 6.4).
        let mut known_commands: HashSet<String> = if inject_tools {
            all_commands.iter().map(|c| c.name.clone()).collect()
        } else {
            HashSet::new()
        };
        if memory_is_configured {
            for name in BUILTIN_COMMANDS {
                known_commands.insert((*name).to_string());
            }
        }

        let max_iterations = self.config.gateway.max_command_iterations;
        let max_retries = self.config.effective_max_pre_response_retries();
        let mut iterations = 0u32;
        let mut pre_response_retries = 0u32;
        // Count every command execution (builtin + external) for RequestEnd telemetry.
        let mut commands_executed: u32 = 0;

        loop {
            if iterations >= max_iterations {
                return Err(WeftError::CommandLoopExceeded {
                    max: max_iterations,
                });
            }

            // Prepend the system prompt as messages[0] with Role::System, Source::Gateway.
            // This is the Weft Wire convention: the system prompt is not a separate field —
            // it is the first message, and providers extract it per their wire format requirements.
            let mut provider_messages = Vec::with_capacity(messages.len() + 1);
            provider_messages.push(WeftMessage {
                role: Role::System,
                source: Source::Gateway,
                model: None,
                content: vec![ContentPart::Text(system_prompt.clone())],
                delta: false,
                message_index: 0,
            });
            provider_messages.extend(messages.iter().cloned());

            // Build the provider request for this model.
            let provider_request = ProviderRequest::ChatCompletion {
                messages: provider_messages,
                model: model_id.clone().unwrap_or_default(),
                options: SamplingOptions {
                    max_tokens: Some(max_tokens.unwrap_or(4096)),
                    temperature: request.options.temperature,
                    top_p: request.options.top_p,
                    top_k: request.options.top_k,
                    stop: request.options.stop.clone(),
                    frequency_penalty: request.options.frequency_penalty,
                    presence_penalty: request.options.presence_penalty,
                    seed: request.options.seed,
                    response_format: request.options.response_format.clone(),
                    activity: false, // Never passed to provider
                },
            };

            // Call the selected provider, with fallback to default on non-rate-limit error.
            let (completion_message, completion_usage) = self
                .call_with_fallback(
                    provider.clone(),
                    &selected_model_name,
                    provider_request,
                    request.options.temperature,
                )
                .await?;

            // Extract text from the response WeftMessage.
            let completion_text: String = completion_message
                .content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            // Parse the response for slash commands.
            let parsed = parse_response(&completion_text, &known_commands);

            if parsed.invocations.is_empty() && parsed.parse_errors.is_empty() {
                // No commands — this is a final response candidate. Fire PreResponse hook.
                let pre_response_payload = serde_json::json!({
                    "text": parsed.text,
                    "model": selected_model_name,
                });

                match self
                    .hooks
                    .run_chain(HookEvent::PreResponse, pre_response_payload, None)
                    .await
                {
                    HookChainResult::Blocked { reason, hook_name } => {
                        // Feedback block: inject reason and re-run LLM if retries remain.
                        if pre_response_retries < max_retries {
                            pre_response_retries += 1;
                            info!(
                                hook = %hook_name,
                                reason = %reason,
                                retry = pre_response_retries,
                                max_retries,
                                "PreResponse hook blocked response — injecting reason and regenerating"
                            );
                            // Inject the blocked assistant response, then a User-role
                            // directive so the LLM receives the feedback as an external
                            // instruction rather than its own prior output.
                            let injection = format!(
                                "[Hook {hook_name} blocked your response: {reason}. Please reconsider and generate a new response.]"
                            );
                            messages.push(WeftMessage {
                                role: Role::Assistant,
                                source: Source::Provider,
                                model: Some(selected_model_name.clone()),
                                content: vec![ContentPart::Text(completion_text.clone())],
                                delta: false,
                                message_index: 0,
                            });
                            messages.push(WeftMessage {
                                role: Role::User,
                                source: Source::Client,
                                model: None,
                                content: vec![ContentPart::Text(injection)],
                                delta: false,
                                message_index: 0,
                            });
                            iterations += 1;
                            continue;
                        } else {
                            return Err(WeftError::HookBlockedAfterRetries {
                                event: "PreResponse".to_string(),
                                reason,
                                hook_name,
                                retries: pre_response_retries,
                            });
                        }
                    }
                    HookChainResult::Allowed { payload, .. } => {
                        // Extract (possibly modified) text from the payload.
                        let final_text = payload
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&parsed.text)
                            .to_string();
                        return Ok((
                            assemble_response(
                                &request.routing.raw,
                                final_text,
                                &selected_model_name,
                                completion_usage,
                                &activity_events,
                                request.options.activity,
                            ),
                            commands_executed,
                        ));
                    }
                }
            }

            // Partition invocations: built-in memory commands vs external tool commands.
            // Built-in commands are intercepted before the command registry.
            let (builtin_invocations, external_invocations): (Vec<_>, Vec<_>) = parsed
                .invocations
                .into_iter()
                .partition(|inv| BUILTIN_COMMANDS.contains(&inv.name.as_str()));

            // Include parse errors as failed results so the LLM can see them.
            let mut results: Vec<CommandResult> = Vec::new();
            results.extend(parsed.parse_errors);

            // Execute built-in memory commands first (results available in same turn).
            for invocation in &builtin_invocations {
                let result = self.handle_builtin_with_hooks(invocation, &user_text).await;
                commands_executed += 1;
                results.push(result);
            }

            // Execute external commands via the command registry.
            for invocation in &external_invocations {
                // ── [HOOK: PreToolUse] ───────────────────────────────────────
                let tool_payload = serde_json::json!({
                    "command": invocation.name,
                    "arguments": invocation.arguments,
                    "action": match &invocation.action {
                        CommandAction::Execute => "execute",
                        CommandAction::Describe => "describe",
                    },
                });

                let (effective_invocation, pre_tool_blocked) = match self
                    .hooks
                    .run_chain(HookEvent::PreToolUse, tool_payload, Some(&invocation.name))
                    .await
                {
                    HookChainResult::Blocked { reason, hook_name } => {
                        info!(
                            hook = %hook_name,
                            command = %invocation.name,
                            reason = %reason,
                            "PreToolUse hook blocked command"
                        );
                        results.push(CommandResult {
                            command_name: invocation.name.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(reason),
                        });
                        (None, true)
                    }
                    HookChainResult::Allowed { payload, .. } => {
                        // If modified, extract potentially updated arguments.
                        let modified_invocation = if payload
                            != serde_json::json!({
                                "command": invocation.name,
                                "arguments": invocation.arguments,
                                "action": match &invocation.action {
                                    CommandAction::Execute => "execute",
                                    CommandAction::Describe => "describe",
                                },
                            }) {
                            // Reconstruct invocation from payload.
                            let mut inv = invocation.clone();
                            if let Some(args) = payload.get("arguments")
                                && let Ok(updated) = serde_json::from_value(args.clone())
                            {
                                inv.arguments = updated;
                            }
                            inv
                        } else {
                            invocation.clone()
                        };
                        (Some(modified_invocation), false)
                    }
                };

                if pre_tool_blocked {
                    continue;
                }

                let effective_invocation = effective_invocation.unwrap();

                // Execute the command.
                let mut cmd_result = self
                    .commands
                    .execute_command(&effective_invocation)
                    .await
                    .unwrap_or_else(|e| CommandResult {
                        command_name: effective_invocation.name.clone(),
                        success: false,
                        output: String::new(),
                        error: Some(e.to_string()),
                    });
                commands_executed += 1;

                // ── [HOOK: PostToolUse] ──────────────────────────────────────
                let post_tool_payload = serde_json::json!({
                    "command": effective_invocation.name,
                    "action": match &effective_invocation.action {
                        CommandAction::Execute => "execute",
                        CommandAction::Describe => "describe",
                    },
                    "success": cmd_result.success,
                    "output": cmd_result.output,
                    "error": cmd_result.error,
                });

                match self
                    .hooks
                    .run_chain(
                        HookEvent::PostToolUse,
                        post_tool_payload,
                        Some(&effective_invocation.name),
                    )
                    .await
                {
                    HookChainResult::Allowed { payload, .. } => {
                        // Apply any modifications to the result.
                        if let Some(output) = payload.get("output").and_then(|v| v.as_str()) {
                            cmd_result.output = output.to_string();
                        }
                        if let Some(success) = payload.get("success").and_then(|v| v.as_bool()) {
                            cmd_result.success = success;
                        }
                    }
                    HookChainResult::Blocked { hook_name, reason } => {
                        // Block on PostToolUse is non-blocking per spec — log and continue.
                        warn!(
                            hook = %hook_name,
                            reason = %reason,
                            "PostToolUse hook returned Block (non-blocking event) — ignoring"
                        );
                    }
                }

                results.push(cmd_result);
            }

            // Append the full assistant response (with command lines) to message history.
            messages.push(WeftMessage {
                role: Role::Assistant,
                source: Source::Provider,
                model: Some(selected_model_name.clone()),
                content: vec![ContentPart::Text(completion_text.clone())],
                delta: false,
                message_index: 0,
            });

            // Inject command results as an assistant message in TOON format.
            // The assistant called the commands; results are part of its workflow.
            messages.push(WeftMessage {
                role: Role::Assistant,
                source: Source::Tool,
                model: None,
                content: vec![ContentPart::Text(format_command_results_toon(&results))],
                delta: false,
                message_index: 0,
            });

            iterations += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use std::sync::Arc;
    use std::time::Duration;
    use weft_core::{
        ContentPart, GatewayConfig, Role, Source, WeftError, WeftMessage, WeftRequest,
    };
    use weft_llm::{Provider, ProviderError, ProviderRequest, ProviderResponse};

    // ── Test: no-command response (single pass) ────────────────────────────

    #[tokio::test]
    async fn test_no_command_response_single_pass() {
        let registry = single_model_registry(
            MockLlmProvider::single("Hello, I can help you with that!"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Hello, I can help you with that!");
        assert!(resp.id.starts_with("chatcmpl-"));
    }

    // ── Test: single command loop ──────────────────────────────────────────

    #[tokio::test]
    async fn test_single_command_executes_and_loops() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust async\"",
                "Here are the results I found.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "Found 5 results about Rust async".to_string(),
                    error: None,
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust async"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Here are the results I found.");
    }

    // ── Test: multiple commands ────────────────────────────────────────────

    #[tokio::test]
    async fn test_multiple_commands_executed_sequentially() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"\n/code_review target: src",
                "Done with both commands.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![
                ("web_search", "Search the web"),
                ("code_review", "Review code"),
            ])
            .with_execute_result(CommandResult {
                command_name: "any".to_string(),
                success: true,
                output: "OK".to_string(),
                error: None,
            }),
        );

        let resp = engine
            .handle_request(make_user_request("Do stuff"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Done with both commands.");
    }

    // ── Test: max iterations exceeded ─────────────────────────────────────

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        let config = Arc::new(weft_core::WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 2,
                request_timeout_secs: 30,
            },
            ..(*test_config()).clone()
        });

        let registry = single_model_registry(
            MockLlmProvider::single("/web_search query: \"loop\""),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "still looping".to_string(),
                    error: None,
                },
            ),
        );

        let result = engine
            .handle_request(make_user_request("Search forever"))
            .await;

        assert!(
            matches!(result, Err(WeftError::CommandLoopExceeded { max: 2 })),
            "expected CommandLoopExceeded, got: {:?}",
            result
        );
    }

    // ── Test: command failure recovery ────────────────────────────────────

    #[tokio::test]
    async fn test_command_failure_injected_as_error_result() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "The search failed, but I can still answer.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("registry unavailable".to_string()),
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed despite command failure");

        assert_eq!(
            resp_text(&resp),
            "The search failed, but I can still answer."
        );
    }

    // ── Test: router fallback ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_router_failure_falls_back_to_all_commands() {
        let registry = single_model_registry(
            MockLlmProvider::single("No commands needed."),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::failing(),
            MockCommandRegistry::new(vec![
                ("web_search", "Search the web"),
                ("code_review", "Review code"),
            ]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("router fallback must not fail the request");

        assert!(!resp_text(&resp).is_empty());
    }

    // ── Test: --describe flag ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_describe_action_handled_by_registry() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search --describe",
                "Now I understand the command.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_describe_result(
                weft_core::CommandDescription {
                    name: "web_search".to_string(),
                    description: "Search the web for current information".to_string(),
                    usage: "/web_search query: \"search terms\"".to_string(),
                    parameters_schema: None,
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("What is web_search?"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Now I understand the command.");
    }

    // ── Test: request timeout ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_request_timeout() {
        let config = Arc::new(weft_core::WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 0,
            },
            ..(*test_config()).clone()
        });

        /// A provider that sleeps forever before responding.
        struct SlowLlmProvider;

        #[async_trait::async_trait]
        impl Provider for SlowLlmProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(ProviderResponse::ChatCompletion {
                    message: WeftMessage {
                        role: Role::Assistant,
                        source: Source::Provider,
                        model: None,
                        content: vec![ContentPart::Text("never".to_string())],
                        delta: false,
                        message_index: 0,
                    },
                    usage: None,
                })
            }

            fn name(&self) -> &str {
                "slow"
            }
        }

        let registry = single_model_registry(SlowLlmProvider, "test-model", "claude-test");
        let engine = make_engine_with_config(
            config,
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::RequestTimeout { .. })),
            "expected RequestTimeout, got: {:?}",
            result
        );
    }

    // ── Test: response format matches OpenAI schema ────────────────────────

    #[tokio::test]
    async fn test_response_format_matches_openai_schema() {
        let registry = single_model_registry(
            MockLlmProvider::single("Test response"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hi"))
            .await
            .expect("should succeed");

        assert!(
            resp.id.starts_with("chatcmpl-"),
            "id must start with chatcmpl-"
        );
        assert_eq!(
            resp.model, "test-model",
            "model must be preserved from request"
        );
        let assistant_msg = resp
            .messages
            .iter()
            .find(|m| m.role == Role::Assistant && m.source == Source::Provider)
            .expect("response must contain an assistant Provider message");
        assert_eq!(assistant_msg.role, Role::Assistant);
        assert_eq!(
            resp.usage.total_tokens,
            resp.usage.prompt_tokens + resp.usage.completion_tokens
        );
    }

    // ── Test: no user message ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_no_user_message_returns_error() {
        let registry = single_model_registry(
            MockLlmProvider::single("irrelevant"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let req = WeftRequest {
            messages: vec![WeftMessage {
                role: Role::System,
                source: Source::Client,
                model: None,
                content: vec![ContentPart::Text("system only".to_string())],
                delta: false,
                message_index: 0,
            }],
            routing: weft_core::ModelRoutingInstruction::parse("test-model"),
            options: weft_core::SamplingOptions::default(),
        };

        let result = engine.handle_request(req).await;
        assert!(result.is_err(), "expected error for missing user message");
    }

    // ── Test: rate limit error propagated ─────────────────────────────────

    #[tokio::test]
    async fn test_rate_limit_error_propagated() {
        let registry = single_model_registry(RateLimitedLlmProvider, "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::RateLimited {
                    retry_after_ms: 1000
                })
            ),
            "expected RateLimited, got: {:?}",
            result
        );
    }

    // ── Test: LLM error propagated ─────────────────────────────────────────

    #[tokio::test]
    async fn test_llm_error_propagated() {
        let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "expected Llm error, got: {:?}",
            result
        );
    }

    // ── Test: list commands failure ────────────────────────────────────────

    #[tokio::test]
    async fn test_list_commands_failure_propagated() {
        let registry = single_model_registry(
            MockLlmProvider::single("irrelevant"),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::with_list_failure(),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Command(_))),
            "expected Command error from list failure, got: {:?}",
            result
        );
    }

    // ── Test: empty LLM response is valid ─────────────────────────────────

    #[tokio::test]
    async fn test_empty_llm_response_is_valid() {
        let registry =
            single_model_registry(MockLlmProvider::single(""), "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("empty LLM response must be valid");

        assert_eq!(resp_text(&resp), "");
    }

    // ── Test: usage propagated from last LLM call ──────────────────────────

    #[tokio::test]
    async fn test_usage_extracted_from_provider() {
        let registry =
            single_model_registry(MockLlmProvider::single("Done"), "test-model", "claude-test");
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn test_end_to_end_uses_provider_execute_path() {
        let provider = MockLlmProvider::single("Hello from provider");
        let registry = single_model_registry(provider, "test-model", "claude-test");

        let engine = make_engine(
            registry,
            MockRouter::with_model("test-model"),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hi there"))
            .await
            .expect("end-to-end should succeed");

        assert_eq!(
            resp_text(&resp),
            "Hello from provider",
            "response text must come from provider.execute()"
        );
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
    }

    #[tokio::test]
    async fn test_no_eligible_models_error_message() {
        let err = WeftError::NoEligibleModels {
            capability: "chat_completions".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "no models configured with capability 'chat_completions'"
        );
    }
}

#[cfg(test)]
mod hook_tests {
    use super::*;
    use crate::test_support::*;
    use std::sync::Arc;
    use weft_core::{HookEvent, WeftError};
    use weft_hooks::types::HookMatcher;
    use weft_hooks::{HookRegistry, RegisteredHook};
    use weft_memory::DefaultMemoryService;

    // ── Phase 4: RequestStart hook integration ─────────────────────────────

    #[tokio::test]
    async fn test_request_start_hook_blocks_returns_hook_blocked_error() {
        let registry = single_model_registry(
            MockLlmProvider::single("should not be called"),
            "test-model",
            "claude-test",
        );
        let hook_reg = hook_registry_with(
            HookEvent::RequestStart,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "auth failed",
            ))),
            None,
            100,
        );
        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlocked { ref event, ref reason, .. })
                    if event == "RequestStart" && reason == "auth failed"
            ),
            "expected HookBlocked for RequestStart, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_request_start_hook_allows_request_proceeds() {
        let registry = single_model_registry(
            MockLlmProvider::single("Response text"),
            "test-model",
            "claude-test",
        );
        let hook_reg = hook_registry_with(
            HookEvent::RequestStart,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::allow())),
            None,
            100,
        );
        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed when RequestStart hook allows");
        assert_eq!(resp_text(&resp), "Response text");
    }

    // ── Phase 4: PreRoute hook integration ─────────────────────────────────

    #[tokio::test]
    async fn test_pre_route_hook_blocks_model_domain_returns_hook_blocked() {
        let hook_reg = hook_registry_with(
            HookEvent::PreRoute,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "model blocked",
            ))),
            Some("model"),
            100,
        );

        let engine = crate::GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("should not be called"),
                MockLlmProvider::single("should not be called"),
            ),
            Arc::new(MockRouter::with_model("complex-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlocked { ref event, .. })
                    if event == "PreRoute"
            ),
            "expected HookBlocked for PreRoute on model domain, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_pre_route_hook_no_matcher_fires_on_all_domains() {
        let recorder = Arc::new(RecordingHookExecutor::new());
        let recorder_clone = Arc::clone(&recorder);

        struct ArcRecording(Arc<RecordingHookExecutor>);
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for ArcRecording {
            async fn execute(
                &self,
                payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.recorded.lock().unwrap().push(payload.clone());
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::PreRoute,
            Box::new(ArcRecording(recorder_clone)),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            single_model_registry(MockLlmProvider::single("Done"), "test-model", "claude-test"),
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search")]),
            hook_reg,
        );

        engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        let payloads = recorder.recorded_payloads();
        assert!(
            !payloads.is_empty(),
            "expected PreRoute hook to fire at least once"
        );
        let domains: Vec<&str> = payloads
            .iter()
            .filter_map(|p| p.get("domain").and_then(|d| d.as_str()))
            .collect();
        assert!(
            domains.contains(&"commands"),
            "expected PreRoute to fire for commands domain, got: {domains:?}"
        );
    }

    // ── Phase 4: PostRoute hook integration ────────────────────────────────

    #[tokio::test]
    async fn test_post_route_hook_overrides_model_selection() {
        let hook_reg = hook_registry_with(
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "selected": ["default-model"]
            }))),
            Some("model"),
            100,
        );

        let engine = crate::GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("final response"),
                MockLlmProvider::single("WRONG: complex-model was used"),
            ),
            Arc::new(MockRouter::with_model("complex-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed with PostRoute model override to default-model");
        assert_eq!(resp_text(&resp), "final response");
    }

    #[tokio::test]
    async fn test_post_route_hook_overrides_model_to_invalid_falls_back_to_default() {
        let hook_reg = hook_registry_with(
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "selected": ["nonexistent-model"]
            }))),
            Some("model"),
            100,
        );

        let engine = crate::GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("fallback response"),
                MockLlmProvider::single("should not be called"),
            ),
            Arc::new(MockRouter::with_model("complex-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should fall back to default model on invalid PostRoute override");
        assert_eq!(resp_text(&resp), "fallback response");
    }

    // ── Phase 4: PreToolUse hook integration ───────────────────────────────

    #[tokio::test]
    async fn test_pre_tool_use_blocks_command_returns_failed_result() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "Search blocked, answering from memory.",
            ]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreToolUse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "permission denied",
            ))),
            Some("web_search"),
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed — block returns error result not 403");
        assert_eq!(resp_text(&resp), "Search blocked, answering from memory.");
    }

    #[tokio::test]
    async fn test_pre_tool_use_blocks_one_command_other_executes() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"\n/code_review target: src",
                "Done.",
            ]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreToolUse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "blocked",
            ))),
            Some("web_search"),
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search"), ("code_review", "Review")])
                .with_execute_result(CommandResult {
                    command_name: "code_review".to_string(),
                    success: true,
                    output: "review done".to_string(),
                    error: None,
                }),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Do both"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "Done.");
    }

    // ── Phase 4: PostToolUse hook integration ──────────────────────────────

    #[tokio::test]
    async fn test_post_tool_use_hook_modifies_output() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "Search result was: hooked output",
            ]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PostToolUse,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "output": "hooked output"
            }))),
            Some("web_search"),
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "original output".to_string(),
                    error: None,
                },
            ),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "Search result was: hooked output");
    }

    // ── Phase 4: PreResponse hook integration ──────────────────────────────

    #[tokio::test]
    async fn test_pre_response_hook_blocks_triggers_regeneration() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec!["first response", "second response"]),
            "test-model",
            "claude-test",
        );

        struct BlockOnceExecutor {
            count: std::sync::Mutex<u32>,
        }
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for BlockOnceExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                let mut count = self.count.lock().unwrap();
                *count += 1;
                if *count == 1 {
                    weft_hooks::types::HookResponse::block("content policy")
                } else {
                    weft_hooks::types::HookResponse::allow()
                }
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::PreResponse,
            Box::new(BlockOnceExecutor {
                count: std::sync::Mutex::new(0),
            }),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed after regeneration");
        assert_eq!(resp_text(&resp), "second response");
    }

    #[tokio::test]
    async fn test_pre_response_hook_blocks_after_max_retries_returns_422() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec!["bad response"; 10]),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreResponse,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "always blocked",
            ))),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlockedAfterRetries { ref event, retries, .. })
                    if event == "PreResponse" && retries == 2
            ),
            "expected HookBlockedAfterRetries after max retries, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_pre_response_hook_modifies_response_text() {
        let registry = single_model_registry(
            MockLlmProvider::single("original text"),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreResponse,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "text": "modified text"
            }))),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "modified text");
    }

    // ── Phase 4: RequestEnd hook integration ───────────────────────────────

    #[tokio::test]
    async fn test_request_end_hook_fires_after_response() {
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        struct FireFlagExecutor(Arc<std::sync::atomic::AtomicBool>);
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for FireFlagExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.store(true, std::sync::atomic::Ordering::Release);
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::RequestEnd,
            Box::new(FireFlagExecutor(fired_clone)),
            None,
            100,
        );

        let engine = make_engine_with_hooks(
            single_model_registry(
                MockLlmProvider::single("Hello"),
                "test-model",
                "claude-test",
            ),
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");
        assert_eq!(resp_text(&resp), "Hello");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            fired.load(std::sync::atomic::Ordering::Acquire),
            "RequestEnd hook should have fired after response"
        );
    }

    #[tokio::test]
    async fn test_request_end_semaphore_exhausted_drops_task_with_warning() {
        let config = Arc::new(weft_core::WeftConfig {
            request_end_concurrency: 0,
            ..(*test_config()).clone()
        });

        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        struct FireFlagExecutor(Arc<std::sync::atomic::AtomicBool>);
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for FireFlagExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.store(true, std::sync::atomic::Ordering::Release);
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_with(
            HookEvent::RequestEnd,
            Box::new(FireFlagExecutor(fired_clone)),
            None,
            100,
        );

        let engine = crate::GatewayEngine::new(
            config,
            single_model_registry(
                MockLlmProvider::single("Hello"),
                "test-model",
                "claude-test",
            ),
            Arc::new(MockRouter::with_score(0.9)),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed even with exhausted semaphore");
        assert_eq!(resp_text(&resp), "Hello");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            !fired.load(std::sync::atomic::Ordering::Acquire),
            "RequestEnd hook should have been dropped due to semaphore exhaustion"
        );
    }

    // ── Phase 4: no-hooks unchanged behavior ───────────────────────────────

    #[tokio::test]
    async fn test_no_hooks_configured_behavior_unchanged() {
        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Results found."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "Rust results".to_string(),
                    error: None,
                },
            ),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed with no hooks");
        assert_eq!(resp_text(&resp), "Results found.");
    }

    // ── Phase 4: hook priority ordering ────────────────────────────────────

    #[tokio::test]
    async fn test_hooks_execute_in_priority_order_lower_first() {
        let registry = single_model_registry(
            MockLlmProvider::single("should not be called"),
            "test-model",
            "claude-test",
        );

        let hook_reg = hook_registry_multi(vec![
            (
                HookEvent::RequestStart,
                Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                    "low priority block",
                ))),
                None,
                50,
            ),
            (
                HookEvent::RequestStart,
                Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::allow())),
                None,
                200,
            ),
        ]);

        let engine = make_engine_with_hooks(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(
                result,
                Err(WeftError::HookBlocked { ref reason, .. }) if reason == "low priority block"
            ),
            "lower priority hook should have fired first and blocked, got: {result:?}"
        );
    }

    // ── Phase 4: memory command hook integration ───────────────────────────

    /// Build a `GatewayEngine` with a custom hook registry AND a memory mux.
    fn make_engine_with_hooks_and_mux<R, C>(
        registry: Arc<weft_llm::ProviderRegistry>,
        router: R,
        commands: C,
        hook_registry: weft_hooks::HookRegistry,
        mux: Option<Arc<weft_memory::MemoryStoreMux>>,
    ) -> crate::GatewayEngine<
        weft_hooks::HookRegistry,
        R,
        DefaultMemoryService,
        weft_llm::ProviderRegistry,
        C,
    >
    where
        R: weft_router::SemanticRouter + Send + Sync + 'static,
        C: weft_commands::CommandRegistry + Send + Sync + 'static,
    {
        let memory = mux.map(wrap_mux);
        crate::GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            memory,
            Arc::new(hook_registry),
        )
    }

    /// Build a minimal single-store memory mux with both read and write capability.
    fn make_single_rw_mux(
        name: &str,
        client: Arc<dyn weft_memory::MemoryStoreClient>,
    ) -> Arc<weft_memory::MemoryStoreMux> {
        make_mux_with_stores(vec![(name, true, true, client)])
    }

    #[tokio::test]
    async fn test_pre_tool_use_blocks_recall_routing_hooks_not_fired() {
        let mux = make_single_rw_mux(
            "conv",
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry("m1", "data")])),
        );

        struct PanicExecutor;
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for PanicExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                panic!("PreRoute should not fire after PreToolUse block");
            }
        }

        let hook_reg = hook_registry_multi(vec![
            (
                HookEvent::PreToolUse,
                Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                    "recall blocked by policy",
                ))),
                Some("recall"),
                100,
            ),
            (
                HookEvent::PreRoute,
                Box::new(PanicExecutor),
                Some("memory"),
                100,
            ),
        ]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"test\"", "No memory needed."]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What do you know about me?"))
            .await
            .expect("engine should not error — blocked command is a failed CommandResult");

        assert_eq!(resp_text(&resp), "No memory needed.");
    }

    #[tokio::test]
    async fn test_pre_tool_use_modifies_recall_arguments_used_in_routing() {
        use weft_memory::{MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreResult};
        let query_seen = Arc::new(std::sync::Mutex::new(String::new()));
        let query_seen_clone = Arc::clone(&query_seen);

        struct RecordingMemStoreClient {
            query_seen: Arc<std::sync::Mutex<String>>,
        }

        #[async_trait::async_trait]
        impl MemoryStoreClient for RecordingMemStoreClient {
            async fn query(
                &self,
                query: &str,
                _max_results: u32,
                _min_score: f32,
            ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
                *self.query_seen.lock().unwrap() = query.to_string();
                Ok(vec![])
            }

            async fn store(
                &self,
                _content: &str,
                _metadata: Option<&serde_json::Value>,
            ) -> Result<MemoryStoreResult, MemoryStoreError> {
                Ok(MemoryStoreResult {
                    id: "mock".to_string(),
                })
            }
        }

        let mux = make_single_rw_mux(
            "conv",
            Arc::new(RecordingMemStoreClient {
                query_seen: query_seen_clone,
            }),
        );

        let hook_reg = hook_registry_with(
            HookEvent::PreToolUse,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "arguments": {"query": "modified query text"}
            }))),
            Some("recall"),
            100,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"original query\"",
                "Got the recall results.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9).with_memory_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        engine
            .handle_request(make_user_request("recall test"))
            .await
            .expect("should succeed");

        let seen = query_seen.lock().unwrap().clone();
        assert_eq!(
            seen, "modified query text",
            "routing should use the modified argument, not the original"
        );
    }

    #[tokio::test]
    async fn test_pre_route_memory_blocks_recall_returns_failed_command_result() {
        let mux = make_single_rw_mux("conv", Arc::new(MockMemStoreClient::succeeds(vec![])));

        let hook_reg = hook_registry_with(
            HookEvent::PreRoute,
            Box::new(FixedHookExecutor(weft_hooks::types::HookResponse::block(
                "memory access denied",
            ))),
            Some("memory"),
            100,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"preferences\"",
                "Memory was blocked.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What do you know about me?"))
            .await
            .expect("engine should not error — blocked recall is a failed CommandResult");

        assert_eq!(resp_text(&resp), "Memory was blocked.");
    }

    #[tokio::test]
    async fn test_post_route_memory_overrides_selected_stores_for_remember() {
        use weft_memory::{MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreResult};
        let store_a_stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let store_b_stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let store_a_clone = Arc::clone(&store_a_stored);
        let store_b_clone = Arc::clone(&store_b_stored);

        struct TrackingMemStoreClient(Arc<std::sync::atomic::AtomicBool>);

        #[async_trait::async_trait]
        impl MemoryStoreClient for TrackingMemStoreClient {
            async fn query(
                &self,
                _query: &str,
                _max_results: u32,
                _min_score: f32,
            ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
                Ok(vec![])
            }

            async fn store(
                &self,
                _content: &str,
                _metadata: Option<&serde_json::Value>,
            ) -> Result<MemoryStoreResult, MemoryStoreError> {
                self.0.store(true, std::sync::atomic::Ordering::Release);
                Ok(MemoryStoreResult {
                    id: "tracked".to_string(),
                })
            }
        }

        let mux = make_mux_with_stores(vec![
            (
                "store-a",
                true,
                true,
                Arc::new(TrackingMemStoreClient(store_a_clone)),
            ),
            (
                "store-b",
                true,
                true,
                Arc::new(TrackingMemStoreClient(store_b_clone)),
            ),
        ]);

        let hook_reg = hook_registry_with(
            HookEvent::PostRoute,
            Box::new(ModifyHookExecutor(serde_json::json!({
                "selected": ["store-b"]
            }))),
            Some("memory"),
            100,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Stored the preference.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9).with_memory_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember that I prefer dark mode"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Stored the preference.");

        assert!(
            store_b_stored.load(std::sync::atomic::Ordering::Acquire),
            "store-b should have been written to after PostRoute override"
        );
        assert!(
            !store_a_stored.load(std::sync::atomic::Ordering::Acquire),
            "store-a should NOT have been written to (PostRoute overrode selection)"
        );
    }

    #[tokio::test]
    async fn test_multiple_recall_invocations_each_fires_independent_hook_lifecycle() {
        let pre_tool_call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count_clone = Arc::clone(&pre_tool_call_count);

        struct CountingExecutor(Arc<std::sync::atomic::AtomicU32>);
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for CountingExecutor {
            async fn execute(
                &self,
                _payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                weft_hooks::types::HookResponse::allow()
            }
        }

        let mux = make_single_rw_mux(
            "conv",
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry("m1", "data")])),
        );

        let matcher = HookMatcher::new(Some("recall"), 0).expect("valid matcher");
        let hook = RegisteredHook {
            event: HookEvent::PreToolUse,
            matcher,
            executor: Box::new(CountingExecutor(count_clone)),
            name: "counting-hook".to_string(),
            priority: 100,
        };
        let mut map = std::collections::HashMap::new();
        map.insert(HookEvent::PreToolUse, vec![hook]);
        let hook_reg = HookRegistry::from_registered(map);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"query one\"\n/recall query: \"query two\"",
                "Done with both recalls.",
            ]),
            "test-model",
            "claude-test",
        );

        let engine = make_engine_with_hooks_and_mux(
            registry,
            MockRouter::with_score(0.9).with_memory_score(0.9),
            MockCommandRegistry::new(vec![]),
            hook_reg,
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Recall twice"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Done with both recalls.");

        assert_eq!(
            pre_tool_call_count.load(std::sync::atomic::Ordering::Acquire),
            2,
            "PreToolUse should fire independently for each /recall invocation"
        );
    }

    #[tokio::test]
    async fn test_post_route_modifies_scores_visible_to_subsequent_hooks() {
        let second_hook_payload = Arc::new(std::sync::Mutex::new(None::<serde_json::Value>));
        let capture_clone = Arc::clone(&second_hook_payload);

        struct CapturePayloadExecutor(Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        #[async_trait::async_trait]
        impl weft_hooks::executor::HookExecutor for CapturePayloadExecutor {
            async fn execute(
                &self,
                payload: &serde_json::Value,
            ) -> weft_hooks::types::HookResponse {
                *self.0.lock().unwrap() = Some(payload.clone());
                weft_hooks::types::HookResponse::allow()
            }
        }

        let hook_reg = hook_registry_multi(vec![
            (
                HookEvent::PostRoute,
                Box::new(ModifyHookExecutor(serde_json::json!({
                    "scores": [{"id": "test-model", "score": 0.5}]
                }))),
                Some("model"),
                50,
            ),
            (
                HookEvent::PostRoute,
                Box::new(CapturePayloadExecutor(capture_clone)),
                Some("model"),
                100,
            ),
        ]);

        let engine = crate::GatewayEngine::new(
            multi_model_config(),
            two_model_registry(
                MockLlmProvider::single("Response"),
                MockLlmProvider::single("Response"),
            ),
            Arc::new(MockRouter::with_model("default-model")),
            Arc::new(MockCommandRegistry::new(vec![])),
            None::<Arc<weft_memory::NullMemoryService>>,
            Arc::new(hook_reg),
        );

        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("should succeed");

        assert_eq!(resp_text(&resp), "Response");

        let captured = second_hook_payload.lock().unwrap().clone();
        let captured = captured.expect("second hook should have been called");
        let scores = captured.get("scores").expect("scores field must exist");
        assert_eq!(
            scores,
            &serde_json::json!([{"id": "test-model", "score": 0.5}]),
            "second PostRoute hook should see scores modified by first hook"
        );
    }
}
