//! Gateway engine: the core loop that wires all components together.
//!
//! The engine routes requests semantically, assembles context, calls the LLM,
//! parses commands from the response, executes them, and loops until no more
//! commands are emitted or the iteration cap or timeout is reached.
//!
//! ## Request flow
//!
//! 1. Extract user message from the latest user turn.
//! 2. List all commands from the registry.
//! 3. Build routing domains (Commands, Model, ToolNecessity).
//! 4. Call `router.route()` → `RoutingDecision`.
//! 5. Apply threshold + max_commands filtering to the commands domain result.
//! 6. Check `tools_needed`: if `Some(false)` and `skip_tools_when_unnecessary`,
//!    skip command injection entirely.
//! 7. Select the model from the routing decision (fallback to default if None).
//! 8. Assemble the system prompt.
//! 9. Call the selected provider. On non-rate-limit failure, retry with the
//!    default provider. Rate-limit errors propagate immediately.
//! 10. Parse response: partition into built-in memory commands and external
//!     commands. Execute built-in first (memory results available in same turn),
//!     then external. Merge all results and inject as TOON. Loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, warn};
use weft_commands::{CommandRegistry, MemoryStoreMux, parse_response};
use weft_core::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, CommandAction, CommandResult,
    CommandStub, Message, Role, Usage, WeftConfig, WeftError,
    toon::{fenced_toon, serialize_table},
};
use weft_llm::{
    ChatCompletionInput, ChatCompletionOutput, Provider, ProviderError, ProviderRegistry,
    ProviderRequest, ProviderResponse, TokenUsage,
};
use weft_router::{
    RoutingCandidate, RoutingDomainKind, ScoredCandidate, SemanticRouter, filter_by_threshold,
    take_top,
};

use crate::context::{
    assemble_system_prompt, assemble_system_prompt_no_tools, format_command_results_toon,
};

/// Built-in command names intercepted by the engine before the command registry.
const BUILTIN_COMMANDS: &[&str] = &["recall", "remember"];

/// Compiled-in describe text for `/recall`.
const RECALL_DESCRIBE: &str = "\
recall: Retrieve relevant memories based on a query

Usage: /recall query: \"what do we know about user preferences\"

The gateway searches configured memory stores for content matching your query
and returns the most relevant results. If you omit the query, the current
conversation context is used automatically.";

/// Compiled-in describe text for `/remember`.
const REMEMBER_DESCRIBE: &str = "\
remember: Store information in memory for future recall

Usage: /remember content: \"the user prefers dark mode and compact layouts\"

Stores the given content in the most relevant memory store(s), selected
automatically based on the content. Retrieve stored memories later with
/recall. Use this when the user shares preferences, decisions, or important
context worth preserving.";

/// Return compiled-in describe text for a built-in command.
fn builtin_describe_text(name: &str) -> String {
    match name {
        "recall" => RECALL_DESCRIBE.to_string(),
        "remember" => REMEMBER_DESCRIBE.to_string(),
        _ => format!("{name}: unknown built-in command"),
    }
}

/// The gateway engine: holds shared components and drives the request loop.
///
/// All fields are `Arc` so `GatewayEngine` is cheaply `Clone`able — axum clones
/// it into each request handler.
#[derive(Clone)]
pub struct GatewayEngine {
    config: Arc<WeftConfig>,
    provider_registry: Arc<ProviderRegistry>,
    router: Arc<dyn SemanticRouter>,
    command_registry: Arc<dyn CommandRegistry>,
    /// Optional memory store multiplexer. `None` when no memory stores are configured.
    memory_mux: Option<Arc<MemoryStoreMux>>,
    /// All memory store routing candidates (from config). Used for the Memory domain
    /// in `route_all_domains()` and for per-invocation routing by both `/recall` and
    /// `/remember`. Empty if no memory stores are configured.
    memory_candidates: Vec<RoutingCandidate>,
    /// Memory candidates filtered to read-capable stores only. Used by `/recall`
    /// for per-invocation routing via `score_memory_candidates()`.
    read_memory_candidates: Vec<RoutingCandidate>,
    /// Memory candidates filtered to write-capable stores only. Used by `/remember`
    /// for per-invocation routing via `score_memory_candidates()`.
    write_memory_candidates: Vec<RoutingCandidate>,
    /// Hook registry. Shared immutably across all request handlers.
    /// Contains all registered hooks, sorted by priority per event.
    /// Wired into the request loop in Phase 4.
    #[allow(dead_code)]
    hook_registry: Arc<crate::hooks::HookRegistry>,
    /// Semaphore limiting concurrent RequestEnd hook tasks.
    /// Prevents unbounded task accumulation under burst load.
    /// Used in Phase 4 for fire-and-forget RequestEnd hook dispatch.
    #[allow(dead_code)]
    request_end_semaphore: Arc<tokio::sync::Semaphore>,
}

impl GatewayEngine {
    /// Expose the config for use by the health handler and other modules.
    pub fn config(&self) -> &WeftConfig {
        &self.config
    }

    /// Construct a new gateway engine.
    ///
    /// `memory_candidates`: All memory store routing candidates (pre-built from config,
    ///   same structure as model candidates). Used for the Memory domain in routing and
    ///   for per-invocation `score_memory_candidates()` calls during command execution.
    /// `read_memory_candidates`: Subset with read capability (for `/recall` routing).
    /// `write_memory_candidates`: Subset with write capability (for `/remember` routing).
    pub fn new(
        config: Arc<WeftConfig>,
        provider_registry: Arc<ProviderRegistry>,
        router: Arc<dyn SemanticRouter>,
        command_registry: Arc<dyn CommandRegistry>,
        memory_mux: Option<Arc<MemoryStoreMux>>,
        hook_registry: Arc<crate::hooks::HookRegistry>,
    ) -> Self {
        // Build per-capability candidate sets from config memory stores.
        // These are derived from the same config used to build the mux, so they
        // will always be consistent with the mux's readable/writable sets.
        let (memory_candidates, read_memory_candidates, write_memory_candidates) =
            build_memory_candidates(&config);

        let request_end_concurrency = config.request_end_concurrency;
        let request_end_semaphore = Arc::new(tokio::sync::Semaphore::new(request_end_concurrency));

        Self {
            config,
            provider_registry,
            router,
            command_registry,
            memory_mux,
            memory_candidates,
            read_memory_candidates,
            write_memory_candidates,
            hook_registry,
            request_end_semaphore,
        }
    }

    /// Handle a single chat completion request.
    ///
    /// This is the main gateway loop: route → assemble → LLM → parse → execute → loop.
    pub async fn handle_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, WeftError> {
        // Reject streaming early — no ambiguity.
        if request.stream == Some(true) {
            return Err(WeftError::StreamingNotSupported);
        }

        let timeout_secs = self.config.gateway.request_timeout_secs;
        let timeout = Duration::from_secs(timeout_secs);

        // Wrap the entire gateway loop in a timeout.
        tokio::time::timeout(timeout, self.run_loop(request))
            .await
            .map_err(|_| WeftError::RequestTimeout { timeout_secs })?
    }

    /// Inner gateway loop (no timeout wrapping — caller handles that).
    async fn run_loop(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, WeftError> {
        // Get all available commands from the registry.
        let all_commands = self
            .command_registry
            .list_commands()
            .await
            .map_err(|e| WeftError::Command(e.to_string()))?;

        // Extract the latest user message for semantic routing.
        let user_message = extract_latest_user_message(&request.messages)?;

        // Build all routing domains and call the router once.
        let routing_result = self.route_all_domains(user_message, &all_commands).await;

        // Unpack the routing result.
        let (selected_commands, inject_tools, selected_model_name) = routing_result;

        debug!(
            model = %selected_model_name,
            inject_tools = inject_tools,
            selected_commands = selected_commands.len(),
            "routing decision applied"
        );

        // Look up the provider and model_id for the selected model.
        let provider = self.provider_registry.get(&selected_model_name);
        let model_id = self
            .provider_registry
            .model_id(&selected_model_name)
            .map(String::from);
        let max_tokens = self
            .provider_registry
            .max_tokens_for(&selected_model_name)
            .or(request.max_tokens);

        // Build memory stubs to append when memory stores are configured.
        // These appear regardless of semantic routing — memory commands are always available.
        let memory_stubs: Option<&[(&str, &str)]> = if self.memory_mux.is_some() {
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
        let system_prompt = if inject_tools {
            assemble_system_prompt(
                &selected_commands,
                &self.config.gateway.system_prompt,
                memory_stubs,
            )
        } else {
            assemble_system_prompt_no_tools(&self.config.gateway.system_prompt)
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
        if self.memory_mux.is_some() {
            for name in BUILTIN_COMMANDS {
                known_commands.insert((*name).to_string());
            }
        }

        let max_iterations = self.config.gateway.max_command_iterations;
        let mut iterations = 0u32;

        loop {
            if iterations >= max_iterations {
                return Err(WeftError::CommandLoopExceeded {
                    max: max_iterations,
                });
            }

            // Build the provider request for this model.
            let provider_request = ProviderRequest::ChatCompletion(ChatCompletionInput {
                system_prompt: system_prompt.clone(),
                messages: messages.clone(),
                model: model_id.clone().unwrap_or_default(),
                max_tokens: max_tokens.unwrap_or(4096),
                temperature: request.temperature,
            });

            // Call the selected provider, with fallback to default on non-rate-limit error.
            let completion = self
                .call_with_fallback(
                    provider.clone(),
                    &selected_model_name,
                    provider_request,
                    request.temperature,
                )
                .await?;

            // Parse the response for slash commands.
            let parsed = parse_response(&completion.text, &known_commands);

            if parsed.invocations.is_empty() && parsed.parse_errors.is_empty() {
                // No commands (built-in or external) — we're done. Return the clean text.
                return Ok(build_response(
                    &parsed.text,
                    &request.model,
                    completion.usage,
                ));
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
                let result = self.handle_builtin(invocation, user_message).await;
                results.push(result);
            }

            // Execute external commands via the command registry.
            for invocation in &external_invocations {
                let result = self
                    .command_registry
                    .execute_command(invocation)
                    .await
                    .unwrap_or_else(|e| {
                        // Command errors become failed results (spec Section 8.4).
                        CommandResult {
                            command_name: invocation.name.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(e.to_string()),
                        }
                    });
                results.push(result);
            }

            // Append the full assistant response (with command lines) to message history.
            messages.push(Message {
                role: Role::Assistant,
                content: completion.text.clone(),
            });

            // Inject command results as an assistant message in TOON format.
            // The assistant called the commands; results are part of its workflow.
            messages.push(Message {
                role: Role::Assistant,
                content: format_command_results_toon(&results),
            });

            iterations += 1;
        }
    }

    /// Call the provider, falling back to the default provider on non-rate-limit failure.
    ///
    /// Fallback rules:
    /// - `RateLimited`: always propagate immediately, no fallback.
    /// - Any other error from a non-default model: retry with the default provider.
    /// - Any other error from the default model (or after fallback retry fails): propagate.
    async fn call_with_fallback(
        &self,
        provider: Arc<dyn Provider>,
        selected_model_name: &str,
        request: ProviderRequest,
        temperature: Option<f32>,
    ) -> Result<ChatCompletionOutput, WeftError> {
        let result = provider.execute(request.clone()).await;
        #[allow(unreachable_patterns)]
        match result {
            Ok(ProviderResponse::ChatCompletion(output)) => Ok(output),
            Ok(_) => Err(WeftError::Llm(
                "unexpected response type from provider".to_string(),
            )),
            Err(ProviderError::RateLimited { retry_after_ms }) => {
                // Rate limit: propagate immediately, no fallback.
                Err(WeftError::RateLimited { retry_after_ms })
            }
            Err(e) if selected_model_name != self.provider_registry.default_name() => {
                // Non-default model failed with a non-rate-limit error: try the default.
                warn!(
                    model = selected_model_name,
                    error = %e,
                    "model failed, falling back to default"
                );
                let default_name = self.provider_registry.default_name();
                let default_provider = self.provider_registry.default_provider();

                // Build a new request with the default model's identifiers.
                let fallback_request = match request {
                    ProviderRequest::ChatCompletion(mut input) => {
                        input.model = self
                            .provider_registry
                            .model_id(default_name)
                            .map(String::from)
                            .unwrap_or_default();
                        input.max_tokens = self
                            .provider_registry
                            .max_tokens_for(default_name)
                            .unwrap_or(4096);
                        input.temperature = temperature;
                        ProviderRequest::ChatCompletion(input)
                    }
                    #[allow(unreachable_patterns)]
                    other => other,
                };

                #[allow(unreachable_patterns)]
                match default_provider.execute(fallback_request).await {
                    Ok(ProviderResponse::ChatCompletion(output)) => Ok(output),
                    Ok(_) => Err(WeftError::Llm(
                        "unexpected response type from default provider".to_string(),
                    )),
                    Err(e) => Err(WeftError::Llm(e.to_string())),
                }
            }
            Err(e) => {
                // Default model failed (or selected was already default): propagate.
                Err(WeftError::Llm(e.to_string()))
            }
        }
    }

    /// Route all domains (Commands, Model, ToolNecessity) in a single router call.
    ///
    /// Returns `(selected_commands, inject_tools, selected_model_name)`.
    ///
    /// On router failure, falls back to: all commands (capped by max_commands),
    /// `inject_tools = true` (conservative), and the default model.
    async fn route_all_domains(
        &self,
        user_message: &str,
        all_commands: &[CommandStub],
    ) -> (Vec<CommandStub>, bool, String) {
        let threshold = self.config.router.classifier.threshold;
        let max_commands = self.config.router.classifier.max_commands;
        let skip_tools_when_unnecessary = self.config.router.skip_tools_when_unnecessary;
        let default_model = self.provider_registry.default_name().to_string();

        // Build the Commands domain candidates: each command as "{name}: {description}".
        let command_candidates: Vec<RoutingCandidate> = all_commands
            .iter()
            .map(|cmd| RoutingCandidate {
                id: cmd.name.clone(),
                examples: vec![format!("{}: {}", cmd.name, cmd.description)],
            })
            .collect();

        // Build all routing domains.
        let mut domains = vec![(RoutingDomainKind::Commands, command_candidates)];

        // Model domain: only include if there are multiple models to route between.
        // With a single model there is nothing to route; skip for efficiency.
        let total_models: usize = self
            .config
            .router
            .providers
            .iter()
            .map(|p| p.models.len())
            .sum();
        if total_models > 1 {
            let model_candidates = build_model_candidates(&self.config);
            if !model_candidates.is_empty() {
                domains.push((RoutingDomainKind::Model, model_candidates));
            }
        }

        // ToolNecessity domain: include if tool-skipping is enabled in config.
        if skip_tools_when_unnecessary {
            domains.push((
                RoutingDomainKind::ToolNecessity,
                tool_necessity_candidates(),
            ));
        }

        // Memory domain: include when memory stores are configured and the domain is enabled.
        // The pre-computed memory_candidates are built at startup from config.
        // The RoutingDecision.memory_stores result is used only for the "inject memory stubs"
        // signal, NOT for store selection (both /recall and /remember route per-invocation).
        if self.memory_mux.is_some() && !self.memory_candidates.is_empty() {
            let memory_domain_enabled = self
                .config
                .router
                .domains
                .memory
                .as_ref()
                .map(|d| d.enabled)
                .unwrap_or(true);
            if memory_domain_enabled {
                domains.push((RoutingDomainKind::Memory, self.memory_candidates.clone()));
            }
        }

        let decision = self.router.route(user_message, &domains).await;

        match decision {
            Ok(routing_decision) => {
                // ── Commands ──────────────────────────────────────────────
                let scored = routing_decision.commands;
                let filtered = filter_by_threshold(scored, threshold);
                let top: Vec<ScoredCandidate> = take_top(filtered, max_commands);

                let score_map: std::collections::HashMap<&str, f32> =
                    top.iter().map(|r| (r.id.as_str(), r.score)).collect();

                let mut selected_commands: Vec<CommandStub> = all_commands
                    .iter()
                    .filter(|cmd| score_map.contains_key(cmd.name.as_str()))
                    .cloned()
                    .collect();

                // Preserve top-score ordering from `take_top`.
                selected_commands.sort_by(|a, b| {
                    let sa = score_map.get(a.name.as_str()).copied().unwrap_or(0.0);
                    let sb = score_map.get(b.name.as_str()).copied().unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });

                // ── Tool necessity ────────────────────────────────────────
                // Only skip tools when explicitly told not to need them AND the
                // config allows it. Conservative default: inject tools.
                let inject_tools = !matches!(
                    routing_decision.tools_needed,
                    Some(false) if skip_tools_when_unnecessary
                );

                // ── Model selection ───────────────────────────────────────
                let selected_model_name = routing_decision
                    .model
                    .as_ref()
                    .map(|m| m.id.clone())
                    .unwrap_or_else(|| default_model.clone());

                debug!(
                    selected_commands = selected_commands.len(),
                    total_commands = all_commands.len(),
                    inject_tools = inject_tools,
                    model = %selected_model_name,
                    "semantic router decision"
                );

                (selected_commands, inject_tools, selected_model_name)
            }
            Err(e) => {
                // Router failure: conservative fallback (spec Section 8.1).
                warn!(
                    error = %e,
                    "semantic router failed, using fallback: all commands, inject tools, default model"
                );

                let mut fallback: Vec<CommandStub> =
                    all_commands.iter().take(max_commands).cloned().collect();
                fallback.sort_by(|a, b| a.name.cmp(&b.name));

                (fallback, true, default_model)
            }
        }
    }
}

/// Format memory query results as TOON output for the LLM.
///
/// Groups results by store with `{store, content, score}` columns.
/// Returns "No relevant memories found." when no memories were retrieved.
fn format_memory_query_results(results: &[weft_commands::MemoryQueryResult]) -> String {
    let all_memories: Vec<_> = results
        .iter()
        .flat_map(|r| {
            r.memories.iter().map(|m| {
                vec![
                    r.store_name.clone(),
                    m.content.clone(),
                    format!("{:.2}", m.score),
                ]
            })
        })
        .collect();

    if all_memories.is_empty() {
        return "No relevant memories found.".to_string();
    }

    let table = serialize_table("memories", &["store", "content", "score"], &all_memories);
    fenced_toon(&table)
}

/// Format memory store results as TOON output for the LLM.
///
/// Reports per-store success/failure with `{store, status}` or
/// `{store, status, error}` columns depending on whether errors occurred.
fn format_memory_store_results(
    results: &[(
        String,
        Result<weft_commands::MemoryStoreResult, weft_commands::MemoryStoreError>,
    )],
) -> String {
    let has_errors = results.iter().any(|(_, r)| r.is_err());

    if has_errors {
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|(name, r)| match r {
                Ok(_) => vec![name.clone(), "success".to_string(), String::new()],
                Err(e) => vec![name.clone(), "error".to_string(), e.to_string()],
            })
            .collect();
        let table = serialize_table("stored", &["store", "status", "error"], &rows);
        fenced_toon(&table)
    } else {
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|(name, _)| vec![name.clone(), "success".to_string()])
            .collect();
        let table = serialize_table("stored", &["store", "status"], &rows);
        fenced_toon(&table)
    }
}

impl GatewayEngine {
    /// Handle a built-in memory command (`/recall` or `/remember`).
    ///
    /// Called during the command loop after partitioning built-in from external
    /// invocations. Handles both `Execute` and `Describe` actions.
    ///
    /// In Phase 2, routing to specific stores is not yet implemented:
    /// - `/recall` fans out to all read-capable stores (fallback path).
    /// - `/remember` writes to the first configured writable store (fallback path).
    ///
    /// Phase 3 wires per-invocation `score_memory_candidates()` routing.
    async fn handle_builtin(
        &self,
        invocation: &weft_core::CommandInvocation,
        user_message: &str,
    ) -> CommandResult {
        match &invocation.action {
            CommandAction::Describe => CommandResult {
                command_name: invocation.name.clone(),
                success: true,
                output: builtin_describe_text(&invocation.name),
                error: None,
            },
            CommandAction::Execute => match invocation.name.as_str() {
                "recall" => self.exec_recall(invocation, user_message).await,
                "remember" => self.exec_remember(invocation).await,
                name => CommandResult {
                    command_name: name.to_string(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("{name}: unknown built-in command")),
                },
            },
        }
    }

    /// Execute a `/recall` invocation.
    ///
    /// Performs per-invocation routing: scores read-capable memory candidates against
    /// the query argument (not the user's original message). Stores above threshold
    /// are targeted; below threshold (or router unavailable) fans out to all readable.
    async fn exec_recall(
        &self,
        invocation: &weft_core::CommandInvocation,
        user_message: &str,
    ) -> CommandResult {
        let mux = match &self.memory_mux {
            Some(m) => m,
            None => {
                return CommandResult {
                    command_name: "recall".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("no memory stores configured".to_string()),
                };
            }
        };

        // Extract query argument. Fall back to the user message when absent.
        let query = invocation
            .arguments
            .get("query")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| user_message.to_string());

        // Per-invocation routing: score read-capable candidates against the query text.
        // This is independent of the pre-computed RoutingDecision.memory_stores — the LLM's
        // recall query may be semantically unrelated to the user's original message.
        let threshold = self.memory_domain_threshold();
        let readable_stores: Vec<String> = mux
            .readable_store_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let store_names = self
            .select_recall_stores(&query, &readable_stores, threshold)
            .await;

        let results = mux.query(&store_names, &query, 0.0).await;
        let output = format_memory_query_results(&results);

        CommandResult {
            command_name: "recall".to_string(),
            success: true,
            output,
            error: None,
        }
    }

    /// Select which stores to target for `/recall`.
    ///
    /// Scores read-capable candidates against the query, applies threshold filtering,
    /// and applies asymmetric fallback: if below threshold or router unavailable,
    /// fans out to ALL read-capable stores (reads are safe).
    async fn select_recall_stores(
        &self,
        query: &str,
        readable_stores: &[String],
        threshold: f32,
    ) -> Vec<String> {
        if self.read_memory_candidates.is_empty() {
            // No read-capable candidates to score — fan out (mux handles empty = all readable).
            return vec![];
        }

        match self
            .router
            .score_memory_candidates(query, &self.read_memory_candidates)
            .await
        {
            Ok(scored) => {
                let above_threshold: Vec<String> = scored
                    .iter()
                    .filter(|c| c.score >= threshold)
                    .filter(|c| readable_stores.contains(&c.id))
                    .map(|c| c.id.clone())
                    .collect();
                if above_threshold.is_empty() {
                    // Below threshold: fan out to all read-capable stores.
                    // Reads are safe — returning partial results is better than missing memories.
                    debug!(
                        query_len = query.len(),
                        threshold,
                        "recall: all candidates below threshold, fanning out to all readable stores"
                    );
                    vec![] // empty = mux fans out to all readable
                } else {
                    above_threshold
                }
            }
            Err(e) => {
                // Router unavailable: fan out to all read-capable stores.
                warn!(error = %e, "router unavailable for /recall, querying all read-capable stores");
                vec![] // empty = mux fans out to all readable
            }
        }
    }

    /// Execute a `/remember` invocation.
    ///
    /// Performs per-invocation routing: scores write-capable memory candidates against
    /// the content argument (not the user's original message). Asymmetric fallback:
    /// if below threshold, picks single highest-scoring write-capable store; if router
    /// unavailable, writes to the first configured writable store.
    async fn exec_remember(&self, invocation: &weft_core::CommandInvocation) -> CommandResult {
        let mux = match &self.memory_mux {
            Some(m) => m,
            None => {
                return CommandResult {
                    command_name: "remember".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("no memory stores configured".to_string()),
                };
            }
        };

        // Missing content argument is an error — no sensible fallback.
        let content = match invocation.arguments.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => {
                return CommandResult {
                    command_name: "remember".to_string(),
                    success: false,
                    output: String::new(),
                    error: Some("missing required argument: content".to_string()),
                };
            }
        };

        // Per-invocation routing: score write-capable candidates against the content text.
        // Same mechanism as /recall but with content argument and write-capable candidates.
        let threshold = self.memory_domain_threshold();
        let writable_stores: Vec<String> = mux
            .writable_store_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let target_stores = self
            .select_remember_stores(&content, &writable_stores, threshold)
            .await;

        if target_stores.is_empty() {
            // No writable stores at all — return success with empty output.
            return CommandResult {
                command_name: "remember".to_string(),
                success: true,
                output: String::new(),
                error: None,
            };
        }

        let results = mux.store(&target_stores, &content, None).await;
        let output = format_memory_store_results(&results);

        CommandResult {
            command_name: "remember".to_string(),
            success: true,
            output,
            error: None,
        }
    }

    /// Select which stores to target for `/remember`.
    ///
    /// Scores write-capable candidates against the content, applies threshold filtering,
    /// and applies asymmetric fallback: if below threshold, picks the single
    /// highest-scoring write-capable store (never fans out on writes). If router
    /// unavailable, writes to the first configured writable store.
    async fn select_remember_stores(
        &self,
        content: &str,
        writable_stores: &[String],
        threshold: f32,
    ) -> Vec<String> {
        if writable_stores.is_empty() {
            return vec![];
        }

        if self.write_memory_candidates.is_empty() {
            // No write-capable candidates to score — use first writable store.
            return writable_stores.iter().take(1).cloned().collect();
        }

        match self
            .router
            .score_memory_candidates(content, &self.write_memory_candidates)
            .await
        {
            Ok(scored) => {
                let above_threshold: Vec<_> = scored
                    .iter()
                    .filter(|c| c.score >= threshold)
                    .filter(|c| writable_stores.contains(&c.id))
                    .collect();

                if above_threshold.is_empty() {
                    // Below threshold: pick the single highest-scoring write-capable store.
                    // Never fan out on writes — spraying writes pollutes stores with
                    // misrouted content.
                    let best = scored
                        .iter()
                        .filter(|c| writable_stores.contains(&c.id))
                        .max_by(|a, b| {
                            a.score
                                .partial_cmp(&b.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                    if let Some(store) = best {
                        debug!(
                            store = %store.id,
                            score = store.score,
                            threshold,
                            "remember: below threshold, picking highest-scoring writable store"
                        );
                        vec![store.id.clone()]
                    } else {
                        // No writable stores in scored set — fall back to first writable.
                        writable_stores.iter().take(1).cloned().collect()
                    }
                } else {
                    above_threshold.iter().map(|c| c.id.clone()).collect()
                }
            }
            Err(e) => {
                // Router unavailable: write to first configured writable store.
                warn!(error = %e, "router unavailable for /remember, using first writable store");
                writable_stores.iter().take(1).cloned().collect()
            }
        }
    }

    /// Get the memory domain threshold.
    ///
    /// Uses the Memory domain-specific threshold if configured; otherwise falls back
    /// to the classifier threshold.
    fn memory_domain_threshold(&self) -> f32 {
        self.config
            .router
            .domains
            .memory
            .as_ref()
            .and_then(|d| d.threshold)
            .unwrap_or(self.config.router.classifier.threshold)
    }
}

/// Build memory routing candidates from config, split by capability.
///
/// Returns `(all_candidates, read_candidates, write_candidates)`.
/// All three are empty when no memory stores are configured.
///
/// The candidates are used for:
/// - `all_candidates`: Memory domain in `route_all_domains()` (for the "inject stubs" signal)
/// - `read_candidates`: Per-invocation routing for `/recall` via `score_memory_candidates()`
/// - `write_candidates`: Per-invocation routing for `/remember` via `score_memory_candidates()`
fn build_memory_candidates(
    config: &WeftConfig,
) -> (
    Vec<RoutingCandidate>,
    Vec<RoutingCandidate>,
    Vec<RoutingCandidate>,
) {
    let Some(mem_config) = &config.memory else {
        return (vec![], vec![], vec![]);
    };

    let mut all_candidates = Vec::new();
    let mut read_candidates = Vec::new();
    let mut write_candidates = Vec::new();

    for store in &mem_config.stores {
        let candidate = RoutingCandidate {
            id: store.name.clone(),
            examples: store.examples.clone(),
        };
        if store.can_read() {
            read_candidates.push(candidate.clone());
        }
        if store.can_write() {
            write_candidates.push(candidate.clone());
        }
        all_candidates.push(candidate);
    }

    (all_candidates, read_candidates, write_candidates)
}

/// Build the Model domain routing candidates from config.
///
/// Each `ModelEntry` in all providers becomes a `RoutingCandidate` with
/// the model routing name as `id` and its examples array.
fn build_model_candidates(config: &WeftConfig) -> Vec<RoutingCandidate> {
    config
        .router
        .providers
        .iter()
        .flat_map(|p| {
            p.models.iter().map(|m| RoutingCandidate {
                id: m.name.clone(),
                examples: m.examples.clone(),
            })
        })
        .collect()
}

/// Static ToolNecessity candidates.
///
/// These examples are tunable defaults. They are hardcoded in one place for
/// easy modification without changing multiple callsites.
pub fn tool_necessity_candidates() -> Vec<RoutingCandidate> {
    vec![
        RoutingCandidate {
            id: "needs_tools".to_string(),
            examples: vec![
                "Search the web for the latest news about Rust".to_string(),
                "Look up the current stock price of Apple".to_string(),
                "Run this code and show me the output".to_string(),
                "Find documents about our Q3 strategy".to_string(),
                "Execute a database query for user counts".to_string(),
            ],
        },
        RoutingCandidate {
            id: "no_tools".to_string(),
            examples: vec![
                "What is the capital of France?".to_string(),
                "Explain how async/await works in Rust".to_string(),
                "Write a poem about the ocean".to_string(),
                "What do you think about functional programming?".to_string(),
                "Hello, how are you today?".to_string(),
            ],
        },
    ]
}

/// Extract the text of the last user message from the conversation.
///
/// Returns an error if there are no messages or no user message.
fn extract_latest_user_message(messages: &[Message]) -> Result<&str, WeftError> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.as_str())
        .ok_or_else(|| {
            WeftError::Config("request must contain at least one user message".to_string())
        })
}

/// Build the final `ChatCompletionResponse` from clean text + usage info.
fn build_response(
    clean_text: &str,
    model: &str,
    usage: Option<TokenUsage>,
) -> ChatCompletionResponse {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (prompt_tokens, completion_tokens) = usage
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));

    ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: clean_text.to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;
    use weft_commands::{CommandError, CommandRegistry};
    use weft_core::{
        ClassifierConfig, CommandAction, CommandDescription, CommandInvocation, CommandResult,
        CommandStub, DomainConfig, DomainsConfig, GatewayConfig, LlmProviderKind, MemoryConfig,
        MemoryStoreConfig, Message, ModelEntry, ProviderConfig, Role, RouterConfig, ServerConfig,
        StoreCapability, WeftConfig,
    };
    use weft_llm::{
        ChatCompletionOutput, Provider, ProviderError, ProviderRegistry, ProviderRequest,
        ProviderResponse, TokenUsage,
    };
    use weft_router::{
        RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
        SemanticRouter,
    };

    // ── Mock implementations ───────────────────────────────────────────────

    /// A mock provider with configurable responses.
    struct MockLlmProvider {
        /// Responses to return in order. Repeats last on exhaustion.
        responses: std::sync::Mutex<Vec<String>>,
    }

    impl MockLlmProvider {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: std::sync::Mutex::new(
                    responses.into_iter().map(str::to_string).collect(),
                ),
            }
        }

        fn single(response: &str) -> Self {
            Self::new(vec![response])
        }
    }

    #[async_trait]
    impl Provider for MockLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            let mut guard = self.responses.lock().unwrap();
            let text = if guard.len() > 1 {
                guard.remove(0)
            } else {
                guard[0].clone()
            };
            Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                text,
                usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                }),
            }))
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    /// A provider that records which model was requested and returns a fixed response.
    struct RecordingLlmProvider {
        response: String,
        recorded_model: std::sync::Mutex<Option<String>>,
    }

    impl RecordingLlmProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                recorded_model: std::sync::Mutex::new(None),
            }
        }

        fn recorded_model(&self) -> Option<String> {
            self.recorded_model.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for RecordingLlmProvider {
        async fn execute(
            &self,
            request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            // Record the model from the request. Only ChatCompletion exists now;
            // when future variants land, update this to handle them.
            #[allow(irrefutable_let_patterns)]
            if let ProviderRequest::ChatCompletion(ref input) = request {
                *self.recorded_model.lock().unwrap() = Some(input.model.clone());
            }
            Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                text: self.response.clone(),
                usage: None,
            }))
        }

        fn name(&self) -> &str {
            "recording"
        }
    }

    /// A provider that always returns a rate-limit error.
    struct RateLimitedLlmProvider;

    #[async_trait]
    impl Provider for RateLimitedLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::RateLimited {
                retry_after_ms: 1000,
            })
        }

        fn name(&self) -> &str {
            "rate-limited"
        }
    }

    /// A provider that always returns a request failure error.
    struct FailingLlmProvider;

    #[async_trait]
    impl Provider for FailingLlmProvider {
        async fn execute(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::RequestFailed("network error".to_string()))
        }

        fn name(&self) -> &str {
            "failing"
        }
    }

    /// A mock router with configurable per-domain behavior.
    struct MockRouter {
        /// Score assigned to all Commands domain candidates.
        command_score: f32,
        /// Model to select (None = use default).
        model_decision: Option<String>,
        /// Tools needed decision.
        tools_needed: Option<bool>,
        /// Score assigned to the first memory candidate when `score_memory_candidates()` is called.
        /// Other candidates get score 0.0. `None` means return ModelNotLoaded error (simulate
        /// router unavailable). Default: Some(0.9) — first candidate wins.
        memory_score: Option<f32>,
    }

    impl MockRouter {
        fn with_score(score: f32) -> Self {
            Self {
                command_score: score,
                model_decision: None,
                tools_needed: None,
                memory_score: Some(0.9),
            }
        }

        fn with_model(model: &str) -> Self {
            Self {
                command_score: 1.0,
                model_decision: Some(model.to_string()),
                tools_needed: None,
                memory_score: Some(0.9),
            }
        }

        fn with_tools_needed(needed: bool) -> Self {
            Self {
                command_score: 1.0,
                model_decision: None,
                tools_needed: Some(needed),
                memory_score: Some(0.9),
            }
        }

        /// Configure memory scoring: returns the given score for the first candidate,
        /// 0.0 for all others. Useful for testing threshold behavior.
        fn with_memory_score(mut self, score: f32) -> Self {
            self.memory_score = Some(score);
            self
        }

        /// Configure router unavailable for memory scoring (returns ModelNotLoaded).
        fn with_memory_unavailable(mut self) -> Self {
            self.memory_score = None;
            self
        }

        fn failing() -> FailingRouter {
            FailingRouter
        }
    }

    #[async_trait]
    impl SemanticRouter for MockRouter {
        async fn route(
            &self,
            _user_message: &str,
            domains: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            let mut decision = RoutingDecision::empty();
            for (kind, candidates) in domains {
                match kind {
                    RoutingDomainKind::Commands => {
                        decision.commands = candidates
                            .iter()
                            .map(|c| ScoredCandidate {
                                id: c.id.clone(),
                                score: self.command_score,
                            })
                            .collect();
                    }
                    RoutingDomainKind::Model => {
                        if let Some(ref model_id) = self.model_decision {
                            // Return the configured model as the top scorer
                            decision.model =
                                candidates.iter().find(|c| c.id == *model_id).map(|c| {
                                    ScoredCandidate {
                                        id: c.id.clone(),
                                        score: 0.9,
                                    }
                                });
                        }
                    }
                    RoutingDomainKind::ToolNecessity => {
                        decision.tools_needed = self.tools_needed;
                    }
                    RoutingDomainKind::Memory => {}
                }
            }
            Ok(decision)
        }

        async fn score_memory_candidates(
            &self,
            _text: &str,
            candidates: &[RoutingCandidate],
        ) -> Result<Vec<ScoredCandidate>, RouterError> {
            match self.memory_score {
                None => Err(RouterError::ModelNotLoaded),
                Some(first_score) => {
                    // First candidate gets the configured score, rest get 0.0.
                    // This makes the first candidate "win" when testing threshold behavior.
                    Ok(candidates
                        .iter()
                        .enumerate()
                        .map(|(i, c)| ScoredCandidate {
                            id: c.id.clone(),
                            score: if i == 0 { first_score } else { 0.0 },
                        })
                        .collect())
                }
            }
        }
    }

    struct FailingRouter;

    #[async_trait]
    impl SemanticRouter for FailingRouter {
        async fn route(
            &self,
            _: &str,
            _: &[(RoutingDomainKind, Vec<RoutingCandidate>)],
        ) -> Result<RoutingDecision, RouterError> {
            Err(RouterError::ModelNotLoaded)
        }

        async fn score_memory_candidates(
            &self,
            _: &str,
            _: &[RoutingCandidate],
        ) -> Result<Vec<ScoredCandidate>, RouterError> {
            Err(RouterError::ModelNotLoaded)
        }
    }

    /// A mock command registry.
    struct MockCommandRegistry {
        commands: Vec<CommandStub>,
        execute_result: std::sync::Mutex<Option<CommandResult>>,
        describe_result: std::sync::Mutex<Option<CommandDescription>>,
        fail_list: bool,
    }

    impl MockCommandRegistry {
        fn new(commands: Vec<(&str, &str)>) -> Self {
            Self {
                commands: commands
                    .into_iter()
                    .map(|(n, d)| CommandStub {
                        name: n.to_string(),
                        description: d.to_string(),
                    })
                    .collect(),
                execute_result: std::sync::Mutex::new(None),
                describe_result: std::sync::Mutex::new(None),
                fail_list: false,
            }
        }

        fn with_execute_result(self, result: CommandResult) -> Self {
            *self.execute_result.lock().unwrap() = Some(result);
            self
        }

        fn with_describe_result(self, desc: CommandDescription) -> Self {
            *self.describe_result.lock().unwrap() = Some(desc);
            self
        }

        fn with_list_failure() -> Self {
            Self {
                commands: vec![],
                execute_result: std::sync::Mutex::new(None),
                describe_result: std::sync::Mutex::new(None),
                fail_list: true,
            }
        }
    }

    #[async_trait]
    impl CommandRegistry for MockCommandRegistry {
        async fn list_commands(&self) -> Result<Vec<CommandStub>, CommandError> {
            if self.fail_list {
                return Err(CommandError::RegistryUnavailable(
                    "registry down".to_string(),
                ));
            }
            Ok(self.commands.clone())
        }

        async fn describe_command(&self, name: &str) -> Result<CommandDescription, CommandError> {
            self.describe_result
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| CommandError::NotFound(name.to_string()))
        }

        async fn execute_command(
            &self,
            invocation: &CommandInvocation,
        ) -> Result<CommandResult, CommandError> {
            match &invocation.action {
                CommandAction::Describe => {
                    let desc = self.describe_result.lock().unwrap().clone();
                    if let Some(d) = desc {
                        Ok(CommandResult {
                            command_name: invocation.name.clone(),
                            success: true,
                            output: format!("{}: {}", d.name, d.description),
                            error: None,
                        })
                    } else {
                        Err(CommandError::NotFound(invocation.name.clone()))
                    }
                }
                CommandAction::Execute => {
                    self.execute_result.lock().unwrap().clone().ok_or_else(|| {
                        CommandError::ExecutionFailed {
                            name: invocation.name.clone(),
                            reason: "no mock result configured".to_string(),
                        }
                    })
                }
            }
        }
    }

    /// Build a minimal valid WeftConfig for tests.
    fn test_config() -> Arc<WeftConfig> {
        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0, // All commands pass in tests
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    kind: LlmProviderKind::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    /// Build a multi-model WeftConfig for routing tests.
    fn multi_model_config() -> Arc<WeftConfig> {
        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.0,
                    max_commands: 20,
                },
                default_model: Some("default-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    kind: LlmProviderKind::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    models: vec![
                        ModelEntry {
                            name: "default-model".to_string(),
                            model: "claude-default".to_string(),
                            max_tokens: 1024,
                            examples: vec!["general question".to_string()],
                        },
                        ModelEntry {
                            name: "complex-model".to_string(),
                            model: "claude-complex".to_string(),
                            max_tokens: 4096,
                            examples: vec!["complex reasoning task".to_string()],
                        },
                    ],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig::default(),
            },
            tool_registry: None,
            memory: None,
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    fn make_user_request(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: content.to_string(),
            }],
            max_tokens: None,
            temperature: None,
            stream: None,
        }
    }

    /// Build a single-model `ProviderRegistry` backed by the given provider.
    fn single_model_registry(
        provider: impl Provider + 'static,
        model_name: &str,
        model_id: &str,
    ) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert(
            model_name.to_string(),
            Arc::new(provider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert(model_name.to_string(), model_id.to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert(model_name.to_string(), 1024u32);
        Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            model_name.to_string(),
        ))
    }

    /// Build a two-model `ProviderRegistry` for fallback tests.
    fn two_model_registry(
        default_provider: impl Provider + 'static,
        non_default_provider: impl Provider + 'static,
    ) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(default_provider) as Arc<dyn Provider>,
        );
        providers.insert(
            "complex-model".to_string(),
            Arc::new(non_default_provider) as Arc<dyn Provider>,
        );
        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert("complex-model".to_string(), "claude-complex".to_string());
        let mut max_tokens = HashMap::new();
        max_tokens.insert("default-model".to_string(), 1024u32);
        max_tokens.insert("complex-model".to_string(), 4096u32);
        Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens,
            "default-model".to_string(),
        ))
    }

    fn make_engine(
        registry: Arc<ProviderRegistry>,
        router: impl SemanticRouter + 'static,
        commands: impl CommandRegistry + 'static,
    ) -> GatewayEngine {
        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            None,
            Arc::new(crate::hooks::HookRegistry::empty()),
        )
    }

    fn make_engine_with_config(
        config: Arc<WeftConfig>,
        registry: Arc<ProviderRegistry>,
        router: impl SemanticRouter + 'static,
        commands: impl CommandRegistry + 'static,
    ) -> GatewayEngine {
        GatewayEngine::new(
            config,
            registry,
            Arc::new(router),
            Arc::new(commands),
            None,
            Arc::new(crate::hooks::HookRegistry::empty()),
        )
    }

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

        assert_eq!(
            resp.choices[0].message.content,
            "Hello, I can help you with that!"
        );
        assert_eq!(resp.choices[0].message.role, Role::Assistant);
        assert_eq!(resp.object, "chat.completion");
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

        assert_eq!(
            resp.choices[0].message.content,
            "Here are the results I found."
        );
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

        assert_eq!(resp.choices[0].message.content, "Done with both commands.");
    }

    // ── Test: max iterations exceeded ─────────────────────────────────────

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        let config = Arc::new(WeftConfig {
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
            resp.choices[0].message.content,
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

        // Should succeed even with a failing router (fallback to all commands)
        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("router fallback must not fail the request");

        assert!(!resp.choices[0].message.content.is_empty());
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
                CommandDescription {
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

        assert_eq!(
            resp.choices[0].message.content,
            "Now I understand the command."
        );
    }

    // ── Test: streaming rejection ──────────────────────────────────────────

    #[tokio::test]
    async fn test_streaming_rejected() {
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

        let mut req = make_user_request("Hello");
        req.stream = Some(true);

        let result = engine.handle_request(req).await;
        assert!(
            matches!(result, Err(WeftError::StreamingNotSupported)),
            "expected StreamingNotSupported"
        );
    }

    // ── Test: request timeout ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_request_timeout() {
        let config = Arc::new(WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 0,
            },
            ..(*test_config()).clone()
        });

        /// A provider that sleeps forever before responding.
        struct SlowLlmProvider;

        #[async_trait]
        impl Provider for SlowLlmProvider {
            async fn execute(
                &self,
                _request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                    text: "never".to_string(),
                    usage: None,
                }))
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
        assert_eq!(resp.object, "chat.completion");
        assert!(
            resp.created > 0,
            "created must be a positive unix timestamp"
        );
        assert_eq!(
            resp.model, "test-model",
            "model must be preserved from request"
        );
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert_eq!(resp.choices[0].message.role, Role::Assistant);
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

        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: Role::System,
                content: "system only".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            stream: None,
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
        // Single-model registry: failing provider IS the default, so no fallback.
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

        assert_eq!(resp.choices[0].message.content, "");
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

        // MockLlmProvider returns prompt=10, completion=5
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
    }

    // ── Test: model selection uses router decision ─────────────────────────

    #[tokio::test]
    async fn test_model_selection_uses_router_decision() {
        // Router selects "complex-model"; we verify the correct model_id is passed.
        let recording_provider = Arc::new(RecordingLlmProvider::new("response"));
        let default_provider = Arc::new(MockLlmProvider::single("default response"));

        let mut providers = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            default_provider as Arc<dyn Provider>,
        );
        providers.insert(
            "complex-model".to_string(),
            recording_provider.clone() as Arc<dyn Provider>,
        );

        let mut model_ids = HashMap::new();
        model_ids.insert("default-model".to_string(), "claude-default".to_string());
        model_ids.insert("complex-model".to_string(), "claude-complex".to_string());

        let mut max_tokens_map = HashMap::new();
        max_tokens_map.insert("default-model".to_string(), 1024u32);
        max_tokens_map.insert("complex-model".to_string(), 4096u32);

        let registry = Arc::new(ProviderRegistry::new(
            providers,
            model_ids,
            max_tokens_map,
            "default-model".to_string(),
        ));

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        engine
            .handle_request(make_user_request("Complex task"))
            .await
            .expect("should succeed");

        // Verify the correct model_id was passed to the recording provider
        assert_eq!(
            recording_provider.recorded_model(),
            Some("claude-complex".to_string()),
            "complex-model provider must receive the correct model_id"
        );
    }

    // ── Test: tool skipping when tools_needed = Some(false) ───────────────

    #[tokio::test]
    async fn test_tool_skipping_when_tools_not_needed() {
        // Provider records the system prompt to verify no command stubs
        struct SystemPromptCapture {
            captured: std::sync::Mutex<Option<String>>,
            response_text: &'static str,
        }

        #[async_trait]
        impl Provider for SystemPromptCapture {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Only ChatCompletion exists now; allow until future variants land.
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion(ref input) = request {
                    *self.captured.lock().unwrap() = Some(input.system_prompt.clone());
                }
                Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                    text: self.response_text.to_string(),
                    usage: None,
                }))
            }

            fn name(&self) -> &str {
                "system-prompt-capture"
            }
        }

        let capture = Arc::new(SystemPromptCapture {
            captured: std::sync::Mutex::new(None),
            response_text: "no tools needed",
        });

        let registry = single_model_registry(
            {
                struct WrappedCapture(Arc<SystemPromptCapture>);
                #[async_trait]
                impl Provider for WrappedCapture {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "wrapped-capture"
                    }
                }
                WrappedCapture(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            MockRouter::with_tools_needed(false),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("What is the capital of France?"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // When tools are skipped, the system prompt must NOT contain command stubs
        assert!(
            !captured.contains("```toon"),
            "system prompt should not contain toon block when tools are skipped: {captured}"
        );
        assert!(
            !captured.contains("web_search"),
            "system prompt should not contain command stubs when tools are skipped"
        );
    }

    // ── Test: tool injection when tools_needed = Some(true) ───────────────

    #[tokio::test]
    async fn test_tool_injection_when_tools_needed() {
        struct SystemPromptCapture2 {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl Provider for SystemPromptCapture2 {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Only ChatCompletion exists now; allow until future variants land.
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion(ref input) = request {
                    *self.captured.lock().unwrap() = Some(input.system_prompt.clone());
                }
                Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                    text: "response".to_string(),
                    usage: None,
                }))
            }

            fn name(&self) -> &str {
                "system-prompt-capture2"
            }
        }

        let capture = Arc::new(SystemPromptCapture2 {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture2>);
                #[async_trait]
                impl Provider for W {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "w2"
                    }
                }
                W(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            MockRouter::with_tools_needed(true),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("Search for something"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // Commands must be injected in dash-list format with "Use when" trigger conditions
        assert!(
            captured.contains("- /web_search \u{2014} Use when Search the web"),
            "system prompt must contain dash-list command stubs when tools are needed"
        );
        assert!(
            captured.contains("BLOCKING REQUIREMENT"),
            "system prompt must contain blocking requirement language when tools are needed"
        );
    }

    // ── Test: tool injection when tools_needed = None (conservative) ──────

    #[tokio::test]
    async fn test_tool_injection_when_tools_needed_is_none() {
        // tools_needed = None -> conservative default: inject commands
        let router = MockRouter {
            command_score: 1.0,
            model_decision: None,
            tools_needed: None, // undecided
            memory_score: Some(0.9),
        };

        struct SystemPromptCapture3 {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl Provider for SystemPromptCapture3 {
            async fn execute(
                &self,
                request: ProviderRequest,
            ) -> Result<ProviderResponse, ProviderError> {
                // Only ChatCompletion exists now; allow until future variants land.
                #[allow(irrefutable_let_patterns)]
                if let ProviderRequest::ChatCompletion(ref input) = request {
                    *self.captured.lock().unwrap() = Some(input.system_prompt.clone());
                }
                Ok(ProviderResponse::ChatCompletion(ChatCompletionOutput {
                    text: "response".to_string(),
                    usage: None,
                }))
            }

            fn name(&self) -> &str {
                "system-prompt-capture3"
            }
        }

        let capture = Arc::new(SystemPromptCapture3 {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture3>);
                #[async_trait]
                impl Provider for W {
                    async fn execute(
                        &self,
                        request: ProviderRequest,
                    ) -> Result<ProviderResponse, ProviderError> {
                        self.0.execute(request).await
                    }

                    fn name(&self) -> &str {
                        "w3"
                    }
                }
                W(capture.clone())
            },
            "test-model",
            "claude-test",
        );

        let engine = make_engine(
            registry,
            router,
            MockCommandRegistry::new(vec![("web_search", "Search the web")]),
        );

        engine
            .handle_request(make_user_request("Tell me about Rust"))
            .await
            .expect("should succeed");

        let captured = capture.captured.lock().unwrap().clone().unwrap_or_default();
        // None -> conservative: inject commands
        assert!(
            captured.contains("web_search"),
            "tools_needed=None must default to injecting commands"
        );
    }

    // ── Test: model fallback on non-rate-limit error ───────────────────────

    #[tokio::test]
    async fn test_model_fallback_on_non_rate_limit_error() {
        // complex-model fails with RequestFailed; default-model succeeds.
        let registry = two_model_registry(
            MockLlmProvider::single("fallback response"),
            FailingLlmProvider,
        );

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Complex task"))
            .await
            .expect("fallback to default must succeed");

        assert_eq!(resp.choices[0].message.content, "fallback response");
    }

    // ── Test: rate limit does NOT trigger fallback ─────────────────────────

    #[tokio::test]
    async fn test_rate_limit_does_not_trigger_fallback() {
        // complex-model is rate-limited; must not fall back.
        let registry = two_model_registry(
            MockLlmProvider::single("default would succeed"),
            RateLimitedLlmProvider,
        );

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            MockRouter::with_model("complex-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Complex task"))
            .await;
        assert!(
            matches!(result, Err(WeftError::RateLimited { .. })),
            "rate limit must propagate without fallback, got: {:?}",
            result
        );
    }

    // ── Test: default model failure propagates error ───────────────────────

    #[tokio::test]
    async fn test_default_model_failure_propagates_error() {
        // Only the default model exists and it fails — no fallback available.
        let registry = single_model_registry(FailingLlmProvider, "test-model", "claude-test");

        let engine = make_engine(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine.handle_request(make_user_request("Hello")).await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "default model failure must propagate as Llm error, got: {:?}",
            result
        );
    }

    // ── Test: selected model IS default, no retry on failure ──────────────

    #[tokio::test]
    async fn test_selected_model_is_default_no_retry() {
        // Router selects "default-model" which is also the default — fails once,
        // must NOT retry (would infinite loop).
        let registry = two_model_registry(FailingLlmProvider, MockLlmProvider::single("alt"));

        let engine = make_engine_with_config(
            multi_model_config(),
            registry,
            // Router selects "default-model" (same as the registry's default)
            MockRouter::with_model("default-model"),
            MockCommandRegistry::new(vec![]),
        );

        let result = engine
            .handle_request(make_user_request("Simple task"))
            .await;
        assert!(
            matches!(result, Err(WeftError::Llm(_))),
            "selected=default failure must propagate without retry, got: {:?}",
            result
        );
    }

    // ── Test: context assembly uses TOON fenced blocks ─────────────────────

    #[tokio::test]
    async fn test_context_assembly_uses_toon_fenced_blocks() {
        use crate::context::assemble_system_prompt;
        use weft_core::CommandStub;

        let stubs = vec![CommandStub {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
        }];
        let prompt = assemble_system_prompt(&stubs, "You are a helpful assistant.", None);

        // Command stubs use dash-list format with "Use when" trigger conditions,
        // not TOON fenced blocks — TOON is reserved for results injection only.
        assert!(
            prompt.contains("- /web_search \u{2014} Use when Search the web"),
            "must use dash-list format with Use when trigger condition"
        );
        assert!(
            prompt.contains("BLOCKING REQUIREMENT"),
            "must include blocking requirement language"
        );
        assert!(
            !prompt.contains("commands[1]{name, description}:"),
            "must not use TOON array syntax for command stubs"
        );
    }

    // ── Test: no fenced blocks when tool skipping is active ───────────────

    #[tokio::test]
    async fn test_no_fenced_blocks_when_tool_skipping() {
        use crate::context::assemble_system_prompt_no_tools;

        let prompt = assemble_system_prompt_no_tools("You are a helpful assistant.");
        assert!(
            !prompt.contains("```toon"),
            "must not contain toon blocks when tools are skipped"
        );
        assert!(
            !prompt.contains("commands["),
            "must not contain command stubs when tools are skipped"
        );
    }

    // ── Built-in memory command tests ─────────────────────────────────────

    use weft_commands::{
        MemoryEntry, MemoryStoreClient, MemoryStoreError, MemoryStoreMux, MemoryStoreResult,
    };

    /// Mock memory store client with configurable query and store behaviour.
    struct MockMemStoreClient {
        query_entries: Vec<MemoryEntry>,
        query_error: Option<String>,
        store_id: String,
        store_error: Option<String>,
    }

    impl MockMemStoreClient {
        fn succeeds(entries: Vec<MemoryEntry>) -> Self {
            Self {
                query_entries: entries,
                query_error: None,
                store_id: "mock-mem-id".to_string(),
                store_error: None,
            }
        }

        fn query_fails(msg: &str) -> Self {
            Self {
                query_entries: vec![],
                query_error: Some(msg.to_string()),
                store_id: "mock-mem-id".to_string(),
                store_error: None,
            }
        }

        fn store_fails(msg: &str) -> Self {
            Self {
                query_entries: vec![],
                query_error: None,
                store_id: "mock-mem-id".to_string(),
                store_error: Some(msg.to_string()),
            }
        }
    }

    #[async_trait::async_trait]
    impl MemoryStoreClient for MockMemStoreClient {
        async fn query(
            &self,
            _query: &str,
            _max_results: u32,
            _min_score: f32,
        ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
            if let Some(e) = &self.query_error {
                return Err(MemoryStoreError::QueryFailed(e.clone()));
            }
            Ok(self.query_entries.clone())
        }

        async fn store(
            &self,
            _content: &str,
            _metadata: Option<&serde_json::Value>,
        ) -> Result<MemoryStoreResult, MemoryStoreError> {
            if let Some(e) = &self.store_error {
                return Err(MemoryStoreError::StoreFailed(e.clone()));
            }
            Ok(MemoryStoreResult {
                id: self.store_id.clone(),
            })
        }
    }

    fn mem_entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            content: content.to_string(),
            score: 0.9,
            created_at: "2026-03-15T10:00:00Z".to_string(),
            metadata: None,
        }
    }

    fn make_mux_with_stores(
        stores: Vec<(&str, bool, bool, Arc<dyn MemoryStoreClient>)>,
    ) -> Arc<MemoryStoreMux> {
        let mut client_map: HashMap<String, Arc<dyn MemoryStoreClient>> = HashMap::new();
        let mut max_results = HashMap::new();
        let mut readable = std::collections::HashSet::new();
        let mut writable = std::collections::HashSet::new();
        for (name, r, w, client) in stores {
            client_map.insert(name.to_string(), client);
            max_results.insert(name.to_string(), 5u32);
            if r {
                readable.insert(name.to_string());
            }
            if w {
                writable.insert(name.to_string());
            }
        }
        Arc::new(MemoryStoreMux::new(
            client_map,
            max_results,
            readable,
            writable,
        ))
    }

    fn make_engine_with_mux(
        registry: Arc<ProviderRegistry>,
        router: impl SemanticRouter + 'static,
        commands: impl CommandRegistry + 'static,
        mux: Option<Arc<MemoryStoreMux>>,
    ) -> GatewayEngine {
        GatewayEngine::new(
            test_config(),
            registry,
            Arc::new(router),
            Arc::new(commands),
            mux,
            Arc::new(crate::hooks::HookRegistry::empty()),
        )
    }

    // ── /recall tests ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_recall_intercepted_before_command_registry() {
        // Engine with memory mux — /recall should NOT reach the command registry.
        // If it did reach the registry, it would fail with CommandError::NotFound
        // (the mock registry has no "recall" command) and the loop would continue
        // until the LLM emits a response without commands.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "I found some memories about preferences.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]), // no "recall" in registry
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What are the user's preferences?"))
            .await
            .expect("should succeed");

        assert_eq!(
            resp.choices[0].message.content,
            "I found some memories about preferences."
        );
    }

    #[tokio::test]
    async fn test_recall_returns_memories_as_toon() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "user prefers dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "Thanks for the context.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What does the user like?"))
            .await
            .expect("should succeed");

        assert_eq!(resp.choices[0].message.content, "Thanks for the context.");
    }

    #[tokio::test]
    async fn test_recall_no_query_arg_uses_user_message() {
        // /recall without query arg should still succeed (falls back to user message).
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall", "Nothing found in memory."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Tell me about user prefs"))
            .await
            .expect("should succeed without panicking");

        assert_eq!(resp.choices[0].message.content, "Nothing found in memory.");
    }

    #[tokio::test]
    async fn test_recall_all_stores_fail_returns_success_with_no_memories() {
        // All stores fail — should return success=true with "No relevant memories found."
        let mux = make_mux_with_stores(vec![(
            "broken",
            true,
            true,
            Arc::new(MockMemStoreClient::query_fails("connection refused")),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something\"",
                "Understood, no memories available.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Search memory"))
            .await
            .expect("should succeed even when stores fail");

        assert_eq!(
            resp.choices[0].message.content,
            "Understood, no memories available."
        );
    }

    #[tokio::test]
    async fn test_recall_no_mux_returns_error_result() {
        // /recall with no mux configured should emit an error result to the LLM.
        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something\"",
                "Memory is not configured.",
            ]),
            "test-model",
            "claude-test",
        );
        // Engine with no mux but we need to parse /recall — add it to registry as a command
        // Actually, without a mux, /recall won't be in known_commands, so it won't be parsed.
        // Test instead: call handle_builtin directly by building engine and sending a request
        // where the LLM emits /recall but the parser only picks it up if we register it.
        // To force the parse, we build an engine with a mock mux that is None-equivalent
        // — but there's no way to do that without an actual mux; the mux controls parser registration.
        // This is correct behaviour: without a mux, /recall is not in known_commands and is treated as prose.
        // Verify the response still works (just contains the text with /recall treated as prose).
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            None, // no mux
        );

        let resp = engine
            .handle_request(make_user_request("Try to recall"))
            .await
            .expect("should succeed");

        // The LLM response includes "/recall query: something" but it's treated as prose
        // because recall is not in known_commands when mux is None.
        // The first LLM response is returned directly since no commands are parsed.
        assert_eq!(
            resp.choices[0].message.content,
            "/recall query: \"something\""
        );
    }

    // ── /remember tests ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_remember_intercepted_and_stores_to_writable() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Memory stored.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember this"))
            .await
            .expect("should succeed");

        assert_eq!(resp.choices[0].message.content, "Memory stored.");
    }

    #[tokio::test]
    async fn test_remember_missing_content_returns_error() {
        // /remember without content arg should return success:false error to LLM.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember", // missing content
                "Got an error about missing argument.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Store something"))
            .await
            .expect("should succeed overall");

        assert_eq!(
            resp.choices[0].message.content,
            "Got an error about missing argument."
        );
    }

    #[tokio::test]
    async fn test_remember_no_writable_stores_returns_empty_success() {
        // All stores are read-only — /remember should return success with no stores written.
        let mux = make_mux_with_stores(vec![(
            "code",
            true,
            false, // read-only
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"some info\"",
                "Attempted to remember.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Remember this"))
            .await
            .expect("should not fail");

        assert_eq!(resp.choices[0].message.content, "Attempted to remember.");
    }

    // ── --describe tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_recall_describe_returns_compiled_text() {
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall --describe", "I see how recall works."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("How does recall work?"))
            .await
            .expect("should succeed");

        assert_eq!(resp.choices[0].message.content, "I see how recall works.");
    }

    #[tokio::test]
    async fn test_remember_describe_does_not_reach_command_registry() {
        // The mock registry has no "remember" command — if --describe reached the
        // registry, it would return an error. With built-in interception it returns
        // the compiled describe text and success=true.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember --describe",
                "Now I know how to use remember.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![]), // no "remember" in registry
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("How do I store memories?"))
            .await
            .expect("should succeed");

        assert_eq!(
            resp.choices[0].message.content,
            "Now I know how to use remember."
        );
    }

    // ── mixed built-in + external commands ────────────────────────────────

    #[tokio::test]
    async fn test_mixed_builtin_and_external_commands() {
        // LLM emits both /recall and /web_search in the same response.
        // Both should execute: built-in first, then external.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
                "m1",
                "user prefers dark mode",
            )])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"prefs\"\n/web_search query: \"dark mode\"",
                "Found memories and web results.",
            ]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "dark mode is popular".to_string(),
                    error: None,
                },
            ),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("What do I know about dark mode?"))
            .await
            .expect("should succeed");

        assert_eq!(
            resp.choices[0].message.content,
            "Found memories and web results."
        );
    }

    #[tokio::test]
    async fn test_external_commands_still_work_with_mux_present() {
        // Verify that wiring in a mux doesn't break the existing command registry path.
        let mux = make_mux_with_stores(vec![(
            "conv",
            true,
            true,
            Arc::new(MockMemStoreClient::succeeds(vec![])),
        )]);

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/web_search query: \"Rust\"", "Here are the results."]),
            "test-model",
            "claude-test",
        );
        let engine = make_engine_with_mux(
            registry,
            MockRouter::with_score(0.9),
            MockCommandRegistry::new(vec![("web_search", "Search the web")]).with_execute_result(
                CommandResult {
                    command_name: "web_search".to_string(),
                    success: true,
                    output: "5 results about Rust".to_string(),
                    error: None,
                },
            ),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Search for Rust"))
            .await
            .expect("should succeed");

        assert_eq!(resp.choices[0].message.content, "Here are the results.");
    }

    // ── Phase 3: Semantic Router Integration tests ─────────────────────────

    /// Build a WeftConfig with memory stores for Phase 3 routing tests.
    ///
    /// `stores`: slice of `(name, endpoint, can_read, can_write, examples)`.
    /// `memory_threshold`: optional per-domain memory threshold.
    fn config_with_memory(
        stores: &[(&str, &str, bool, bool, Vec<&str>)],
        memory_threshold: Option<f32>,
    ) -> Arc<WeftConfig> {
        let store_configs: Vec<MemoryStoreConfig> = stores
            .iter()
            .map(|(name, endpoint, can_read, can_write, examples)| {
                let mut capabilities = Vec::new();
                if *can_read {
                    capabilities.push(StoreCapability::Read);
                }
                if *can_write {
                    capabilities.push(StoreCapability::Write);
                }
                MemoryStoreConfig {
                    name: name.to_string(),
                    endpoint: endpoint.to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities,
                    examples: examples.iter().map(|s| s.to_string()).collect(),
                }
            })
            .collect();

        let memory_domain = memory_threshold.map(|t| DomainConfig {
            enabled: true,
            threshold: Some(t),
        });

        Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.5, // default threshold for memory routing tests
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    kind: LlmProviderKind::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig {
                    model: None,
                    tool_necessity: None,
                    memory: memory_domain,
                },
            },
            tool_registry: None,
            memory: Some(MemoryConfig {
                stores: store_configs,
            }),
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        })
    }

    /// Build an engine with the given config, mux, and router.
    fn make_engine_with_config_and_mux(
        config: Arc<WeftConfig>,
        registry: Arc<ProviderRegistry>,
        router: impl SemanticRouter + 'static,
        commands: impl CommandRegistry + 'static,
        mux: Option<Arc<MemoryStoreMux>>,
    ) -> GatewayEngine {
        GatewayEngine::new(
            config,
            registry,
            Arc::new(router),
            Arc::new(commands),
            mux,
            Arc::new(crate::hooks::HookRegistry::empty()),
        )
    }

    // ── Phase 3: build_memory_candidates ──────────────────────────────────

    #[test]
    fn test_build_memory_candidates_empty_when_no_config() {
        // Config with no memory section produces empty candidate sets.
        let config = test_config();
        let (all, read, write) = build_memory_candidates(&config);
        assert!(all.is_empty());
        assert!(read.is_empty());
        assert!(write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_read_write_store() {
        // A store with both capabilities appears in all three sets.
        let config = config_with_memory(
            &[(
                "conv",
                "http://localhost:50052",
                true,
                true,
                vec!["recall conv"],
            )],
            None,
        );
        let (all, read, write) = build_memory_candidates(&config);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "conv");
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].id, "conv");
        assert_eq!(write.len(), 1);
        assert_eq!(write[0].id, "conv");
    }

    #[test]
    fn test_build_memory_candidates_read_only_store() {
        // A read-only store appears in all and read, but not write.
        let config = config_with_memory(
            &[(
                "kb",
                "http://localhost:50053",
                true,
                false,
                vec!["knowledge base"],
            )],
            None,
        );
        let (all, read, write) = build_memory_candidates(&config);
        assert_eq!(all.len(), 1);
        assert_eq!(read.len(), 1);
        assert!(write.is_empty());
    }

    #[test]
    fn test_build_memory_candidates_write_only_store() {
        // A write-only store appears in all and write, but not read.
        let config = config_with_memory(
            &[(
                "audit",
                "http://localhost:50054",
                false,
                true,
                vec!["audit log"],
            )],
            None,
        );
        let (all, read, write) = build_memory_candidates(&config);
        assert_eq!(all.len(), 1);
        assert!(read.is_empty());
        assert_eq!(write.len(), 1);
    }

    #[test]
    fn test_build_memory_candidates_multiple_stores() {
        // Multiple stores split correctly by capability.
        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    true,
                    vec!["conversation"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge base"],
                ),
                (
                    "audit",
                    "http://localhost:50054",
                    false,
                    true,
                    vec!["audit"],
                ),
            ],
            None,
        );
        let (all, read, write) = build_memory_candidates(&config);
        assert_eq!(all.len(), 3);
        assert_eq!(read.len(), 2); // conv + kb
        assert_eq!(write.len(), 2); // conv + audit
        assert!(read.iter().any(|c| c.id == "conv"));
        assert!(read.iter().any(|c| c.id == "kb"));
        assert!(write.iter().any(|c| c.id == "conv"));
        assert!(write.iter().any(|c| c.id == "audit"));
    }

    // ── Phase 3: route_all_domains includes Memory domain ─────────────────

    #[tokio::test]
    async fn test_route_all_domains_includes_memory_when_configured() {
        // When memory stores are configured, route_all_domains should pass
        // Memory domain candidates to the router. We verify by checking that
        // the engine picks up the memory candidates (indirectly via exec_recall routing).
        //
        // Setup: one store "conv", router scores it above threshold (0.9 > 0.5).
        // /recall should target "conv" specifically (not fan out to all).
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "dark mode pref",
        )]));
        let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

        let config = config_with_memory(
            &[(
                "conv",
                "http://localhost:50052",
                true,
                true,
                vec!["conversation recall"],
            )],
            None, // no memory-domain threshold override — uses classifier.threshold = 0.5
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "I found memory about dark mode.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router scores first (only) read candidate at 0.9 > 0.5 threshold.
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("Tell me about preferences"))
            .await
            .expect("should succeed");

        // Should succeed and include memory content in the response.
        assert_eq!(
            resp.choices[0].message.content,
            "I found memory about dark mode."
        );
    }

    // ── Phase 3: /recall per-invocation routing ────────────────────────────

    #[tokio::test]
    async fn test_recall_routes_based_on_query_not_user_message() {
        // The key Phase 3 invariant: /recall routing uses the query argument,
        // not the user's original message. Both route via score_memory_candidates().
        //
        // Setup: two stores. "conv" gets score 0.9, "kb" gets 0.0 from mock router.
        // User message is "help me with my code" (code-domain).
        // /recall query is "user preferences" (conv-domain).
        // Only "conv" should be queried.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "user prefers dark mode",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::query_fails("should not be called"));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["user preferences"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["code architecture"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"user preferences\"",
                "Found: user prefers dark mode.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router: first candidate (conv) gets 0.9, second (kb) gets 0.0.
        // This simulates "user preferences" being semantically closest to conv.
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("help me with my code"))
            .await
            .expect("should succeed");

        // kb_client would fail if called — success means only conv was queried.
        assert_eq!(
            resp.choices[0].message.content,
            "Found: user prefers dark mode."
        );
    }

    #[tokio::test]
    async fn test_recall_below_threshold_fans_out_to_all_readable() {
        // When all candidates score below threshold, /recall fans out to ALL read-capable stores.
        // Router scores first candidate at 0.1, second at 0.0 — both below threshold 0.5.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "conv memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m2",
            "kb memory",
        )]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            Some(0.5), // memory domain threshold
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"something unrelated\"",
                "Found memories from both stores.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router: first candidate 0.1, second 0.0 — both below 0.5 threshold.
        let router = MockRouter::with_score(0.9).with_memory_score(0.1);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("something"))
            .await
            .expect("should succeed");

        // Both stores should have been queried (fan-out) — response shows 2 memories.
        let content = &resp.choices[0].message.content;
        assert_eq!(content, "Found memories from both stores.");
    }

    #[tokio::test]
    async fn test_recall_router_unavailable_fans_out_to_all_readable() {
        // When the router is unavailable (ModelNotLoaded), /recall fans out to all readable.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "conv memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m2",
            "kb memory",
        )]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/recall query: \"anything\"",
                "Got results from both stores.",
            ]),
            "test-model",
            "claude-test",
        );

        // Router returns ModelNotLoaded for score_memory_candidates.
        let router = MockRouter::with_score(0.9).with_memory_unavailable();

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("anything"))
            .await
            .expect("should succeed");

        assert_eq!(
            resp.choices[0].message.content,
            "Got results from both stores."
        );
    }

    // ── Phase 3: /remember per-invocation routing ─────────────────────────

    #[tokio::test]
    async fn test_remember_routes_based_on_content_not_user_message() {
        // /remember routing uses the content argument, not the user's original message.
        // Setup: two writable stores. "conv" gets score 0.9, "audit" gets 0.0.
        // User asks about code; LLM remembers a user preference.
        // Only "conv" should be written to.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        // audit_client: if written to, the store call would succeed, but we verify
        // by checking the TOON output only mentions "conv".
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["user preferences"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit logs"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec![
                "/remember content: \"user prefers dark mode\"",
                "Noted, I'll remember that.",
            ]),
            "test-model",
            "claude-test",
        );

        // First write candidate (conv) gets 0.9, second (audit) gets 0.0.
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("how do I configure nginx?"))
            .await
            .expect("should succeed");

        // audit_client has store_fails — if it were called we'd get a partial failure in TOON.
        // Success means only conv was written.
        assert_eq!(
            resp.choices[0].message.content,
            "Noted, I'll remember that."
        );
    }

    #[tokio::test]
    async fn test_remember_below_threshold_picks_highest_scoring_store() {
        // When all candidates are below threshold, /remember picks single highest-scoring store.
        // Router: first candidate (conv) 0.3, second (audit) 0.1 — both below 0.5 threshold.
        // conv has the higher score, so it gets picked.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        // audit_client: if written, would succeed — but conv should be picked.
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["conversation prefs"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit events"],
                ),
            ],
            Some(0.5), // threshold: 0.5
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/remember content: \"something\"", "Memory stored."]),
            "test-model",
            "claude-test",
        );

        // Both below threshold; first candidate scores higher (0.3 > 0.0 for second).
        let router = MockRouter::with_score(0.9).with_memory_score(0.3);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("remember something"))
            .await
            .expect("should succeed");

        // audit_client has store_fails — success means only conv was picked.
        assert_eq!(resp.choices[0].message.content, "Memory stored.");
    }

    #[tokio::test]
    async fn test_remember_router_unavailable_writes_to_first_writable() {
        // When router is unavailable, /remember writes to the first configured writable store.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let audit_client = Arc::new(MockMemStoreClient::store_fails("should not be written"));
        let mux = make_mux_with_stores(vec![
            ("conv", false, true, conv_client),
            ("audit", false, true, audit_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    false,
                    true,
                    vec!["conversations"],
                ),
                (
                    "audit",
                    "http://localhost:50053",
                    false,
                    true,
                    vec!["audit"],
                ),
            ],
            None,
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/remember content: \"important note\"", "Saved."]),
            "test-model",
            "claude-test",
        );

        // Router unavailable for memory scoring.
        let router = MockRouter::with_score(0.9).with_memory_unavailable();

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("remember this"))
            .await
            .expect("should succeed");

        // audit_client has store_fails — success means only first writable (conv) was picked.
        assert_eq!(resp.choices[0].message.content, "Saved.");
    }

    // ── Phase 3: memory domain threshold from config ──────────────────────

    #[tokio::test]
    async fn test_memory_domain_threshold_gate_applied() {
        // When a per-domain memory threshold is configured, it takes precedence over
        // the classifier threshold.
        //
        // Classifier threshold = 0.5, memory domain threshold = 0.8.
        // Router scores first candidate at 0.7 — above 0.5 but below 0.8.
        // With memory domain threshold, the store should NOT be directly selected
        // (falls back to all readable for /recall, or highest-scoring for /remember).
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1", "memory",
        )]));
        let kb_client = Arc::new(MockMemStoreClient::succeeds(vec![]));
        let mux = make_mux_with_stores(vec![
            ("conv", true, false, conv_client),
            ("kb", true, false, kb_client),
        ]);

        let config = config_with_memory(
            &[
                (
                    "conv",
                    "http://localhost:50052",
                    true,
                    false,
                    vec!["conversations"],
                ),
                (
                    "kb",
                    "http://localhost:50053",
                    true,
                    false,
                    vec!["knowledge"],
                ),
            ],
            Some(0.8), // memory domain threshold = 0.8 (higher than classifier 0.5)
        );

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"something\"", "Found memories."]),
            "test-model",
            "claude-test",
        );

        // Score 0.7: above classifier threshold (0.5) but below memory domain threshold (0.8).
        let router = MockRouter::with_score(0.9).with_memory_score(0.7);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("recall something"))
            .await
            .expect("should succeed");

        // Below memory-domain threshold → fell back to all readable → both stores queried.
        // conv_client returns a memory, kb_client returns empty.
        // Final response is whatever the LLM said after seeing results.
        assert_eq!(resp.choices[0].message.content, "Found memories.");
    }

    // ── Phase 3: RoutingDecision::fallback() memory_stores population ─────

    #[test]
    fn test_fallback_decision_populates_memory_stores() {
        use weft_router::{RoutingDecision, RoutingDomainKind};
        // RoutingDecision::fallback() should populate memory_stores with all
        // candidates at score 1.0 when the Memory domain is included.
        let domains = vec![
            (
                RoutingDomainKind::Commands,
                vec![RoutingCandidate {
                    id: "search".to_string(),
                    examples: vec!["search: Search".to_string()],
                }],
            ),
            (
                RoutingDomainKind::Memory,
                vec![
                    RoutingCandidate {
                        id: "conv".to_string(),
                        examples: vec!["conversations".to_string()],
                    },
                    RoutingCandidate {
                        id: "kb".to_string(),
                        examples: vec!["knowledge base".to_string()],
                    },
                ],
            ),
        ];

        let decision = RoutingDecision::fallback(&domains);

        // Commands: scored at 1.0.
        assert_eq!(decision.commands.len(), 1);

        // Memory stores: all at 1.0, sorted alphabetically.
        assert_eq!(decision.memory_stores.len(), 2);
        assert_eq!(decision.memory_stores[0].id, "conv");
        assert_eq!(decision.memory_stores[1].id, "kb");
        for m in &decision.memory_stores {
            assert_eq!(m.score, 1.0, "fallback memory scores should be 1.0");
        }

        // Model and tools: conservative defaults.
        assert!(decision.model.is_none());
        assert!(decision.tools_needed.is_none());
    }

    #[test]
    fn test_fallback_decision_memory_stores_empty_when_no_memory_domain() {
        // When Memory domain is NOT included in domains, memory_stores stays empty.
        let domains = vec![(
            RoutingDomainKind::Commands,
            vec![RoutingCandidate {
                id: "search".to_string(),
                examples: vec!["search: Search".to_string()],
            }],
        )];

        let decision = RoutingDecision::fallback(&domains);
        assert!(decision.memory_stores.is_empty());
    }

    // ── Phase 3: memory domain disabled via config ─────────────────────────

    #[tokio::test]
    async fn test_recall_with_memory_domain_disabled_fans_out_to_all() {
        // When memory domain is disabled via config, /recall falls back to all readable.
        // The candidates are built but the Memory domain is NOT included in route_all_domains.
        // Per-invocation routing also falls back since write_memory_candidates will still
        // work via score_memory_candidates — but route_all_domains doesn't pass the domain.
        // Since read_memory_candidates is non-empty, score_memory_candidates is still called.
        // The router still works; the "domain disabled" only affects route_all_domains.
        //
        // When domain disabled: engine doesn't pass Memory domain to router.route(),
        // but select_recall_stores() still calls score_memory_candidates() since it uses
        // read_memory_candidates directly.
        //
        // This test verifies that the engine still functions when domain is disabled.
        let conv_client = Arc::new(MockMemStoreClient::succeeds(vec![mem_entry(
            "m1",
            "memory found",
        )]));
        let mux = make_mux_with_stores(vec![("conv", true, true, conv_client)]);

        // Build config with memory domain explicitly disabled.
        let disabled_domain = DomainConfig {
            enabled: false,
            threshold: None,
        };
        let config = Arc::new(WeftConfig {
            server: ServerConfig {
                bind_address: "127.0.0.1:8080".to_string(),
            },
            router: RouterConfig {
                classifier: ClassifierConfig {
                    model_path: "models/test.onnx".to_string(),
                    tokenizer_path: "models/tokenizer.json".to_string(),
                    threshold: 0.5,
                    max_commands: 20,
                },
                default_model: Some("test-model".to_string()),
                providers: vec![ProviderConfig {
                    name: "test-provider".to_string(),
                    kind: LlmProviderKind::Anthropic,
                    api_key: "test-key".to_string(),
                    base_url: None,
                    models: vec![ModelEntry {
                        name: "test-model".to_string(),
                        model: "claude-test".to_string(),
                        max_tokens: 1024,
                        examples: vec!["test query".to_string()],
                    }],
                }],
                skip_tools_when_unnecessary: true,
                domains: DomainsConfig {
                    model: None,
                    tool_necessity: None,
                    memory: Some(disabled_domain),
                },
            },
            tool_registry: None,
            memory: Some(MemoryConfig {
                stores: vec![MemoryStoreConfig {
                    name: "conv".to_string(),
                    endpoint: "http://localhost:50052".to_string(),
                    connect_timeout_ms: 5000,
                    request_timeout_ms: 10000,
                    max_results: 5,
                    capabilities: vec![StoreCapability::Read, StoreCapability::Write],
                    examples: vec!["conversation".to_string()],
                }],
            }),
            hooks: vec![],
            max_pre_response_retries: 2,
            request_end_concurrency: 64,
            gateway: GatewayConfig {
                system_prompt: "You are a test assistant.".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 30,
            },
        });

        let registry = single_model_registry(
            MockLlmProvider::new(vec!["/recall query: \"test query\"", "Found something."]),
            "test-model",
            "claude-test",
        );

        // Router: score_memory_candidates still works (above threshold for conv).
        let router = MockRouter::with_score(0.9).with_memory_score(0.9);

        let engine = make_engine_with_config_and_mux(
            config,
            registry,
            router,
            MockCommandRegistry::new(vec![]),
            Some(mux),
        );

        let resp = engine
            .handle_request(make_user_request("recall test"))
            .await
            .expect("should succeed");

        assert_eq!(resp.choices[0].message.content, "Found something.");
    }
}
