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
    CommandAction, CommandResult, ContentPart, HookEvent, Role, SamplingOptions, Source,
    WeftError, WeftMessage, WeftRequest, WeftResponse,
};
use weft_hooks::{HookChainResult, HookRunner};
use weft_llm::{Capability, ProviderRequest, ProviderService};
use weft_memory::MemoryService;
use weft_router::SemanticRouter;

use super::GatewayEngine;
use super::activity::assemble_response;
use super::util::{ActivityEvent, BUILTIN_COMMANDS, extract_latest_user_text};
use crate::context::{
    assemble_system_prompt, assemble_system_prompt_no_tools, format_command_results_toon,
};

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
    pub(super) async fn run_loop(
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
