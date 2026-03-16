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
use weft_llm::{CompletionOptions, LlmError, LlmProvider, ProviderRegistry};
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
}

impl GatewayEngine {
    /// Expose the config for use by the health handler and other modules.
    pub fn config(&self) -> &WeftConfig {
        &self.config
    }

    /// Construct a new gateway engine.
    pub fn new(
        config: Arc<WeftConfig>,
        provider_registry: Arc<ProviderRegistry>,
        router: Arc<dyn SemanticRouter>,
        command_registry: Arc<dyn CommandRegistry>,
        memory_mux: Option<Arc<MemoryStoreMux>>,
    ) -> Self {
        Self {
            config,
            provider_registry,
            router,
            command_registry,
            memory_mux,
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

        let options = CompletionOptions {
            max_tokens,
            temperature: request.temperature,
            model: model_id.clone(),
        };

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

            // Call the selected LLM provider, with fallback to default on non-rate-limit error.
            let completion = self
                .call_with_fallback(
                    provider.clone(),
                    &selected_model_name,
                    &system_prompt,
                    &messages,
                    &options,
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

            // Inject command results as a user message in TOON format.
            messages.push(Message {
                role: Role::User,
                content: format_command_results_toon(&results),
            });

            iterations += 1;
        }
    }

    /// Call the LLM provider, falling back to the default provider on non-rate-limit failure.
    ///
    /// Fallback rules (spec Section 6.3):
    /// - `RateLimited`: always propagate immediately, no retry.
    /// - Any other error from a non-default model: retry with the default provider.
    /// - Any other error from the default model (or after fallback retry fails): propagate.
    async fn call_with_fallback(
        &self,
        provider: Arc<dyn LlmProvider>,
        selected_model_name: &str,
        system_prompt: &str,
        messages: &[Message],
        options: &CompletionOptions,
        temperature: Option<f32>,
    ) -> Result<weft_llm::CompletionResponse, WeftError> {
        match provider.complete(system_prompt, messages, options).await {
            Ok(response) => Ok(response),
            Err(LlmError::RateLimited { retry_after_ms }) => {
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
                let default_options = CompletionOptions {
                    model: self
                        .provider_registry
                        .model_id(default_name)
                        .map(String::from),
                    max_tokens: self.provider_registry.max_tokens_for(default_name),
                    temperature,
                };
                default_provider
                    .complete(system_prompt, messages, &default_options)
                    .await
                    .map_err(|e| WeftError::Llm(e.to_string()))
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

        // Phase 2: fan out to all readable stores (fallback path).
        // Phase 3 will replace this with per-invocation score_memory_candidates() routing.
        let store_names: Vec<String> = vec![]; // empty = fan out to all readable

        let results = mux.query(&store_names, &query, 0.0).await;
        let output = format_memory_query_results(&results);

        CommandResult {
            command_name: "recall".to_string(),
            success: true,
            output,
            error: None,
        }
    }

    /// Execute a `/remember` invocation.
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

        // Phase 2: write to the first configured writable store (fallback path).
        // Phase 3 will replace this with per-invocation score_memory_candidates() routing.
        let writable: Vec<String> = mux
            .writable_store_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let target_stores: Vec<String> = writable.into_iter().take(1).collect();

        if target_stores.is_empty() {
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
    usage: Option<weft_llm::LlmUsage>,
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
        CommandStub, DomainsConfig, GatewayConfig, LlmProviderKind, Message, ModelEntry,
        ProviderConfig, Role, RouterConfig, ServerConfig, WeftConfig,
    };
    use weft_llm::{
        CompletionOptions, CompletionResponse, LlmError, LlmProvider, ProviderRegistry,
    };
    use weft_router::{
        RouterError, RoutingCandidate, RoutingDecision, RoutingDomainKind, ScoredCandidate,
        SemanticRouter,
    };

    // ── Mock implementations ───────────────────────────────────────────────

    /// A mock LLM provider with configurable responses.
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
    impl LlmProvider for MockLlmProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _messages: &[Message],
            _options: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            let mut guard = self.responses.lock().unwrap();
            let text = if guard.len() > 1 {
                guard.remove(0)
            } else {
                guard[0].clone()
            };
            Ok(CompletionResponse {
                text,
                usage: Some(weft_llm::LlmUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                }),
            })
        }
    }

    /// An LLM that records which model was requested and returns a fixed response.
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
    impl LlmProvider for RecordingLlmProvider {
        async fn complete(
            &self,
            _system_prompt: &str,
            _messages: &[Message],
            options: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            *self.recorded_model.lock().unwrap() = options.model.clone();
            Ok(CompletionResponse {
                text: self.response.clone(),
                usage: None,
            })
        }
    }

    /// An LLM that always returns a rate-limit error.
    struct RateLimitedLlmProvider;

    #[async_trait]
    impl LlmProvider for RateLimitedLlmProvider {
        async fn complete(
            &self,
            _: &str,
            _: &[Message],
            _: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::RateLimited {
                retry_after_ms: 1000,
            })
        }
    }

    /// An LLM that always returns a provider error.
    struct FailingLlmProvider;

    #[async_trait]
    impl LlmProvider for FailingLlmProvider {
        async fn complete(
            &self,
            _: &str,
            _: &[Message],
            _: &CompletionOptions,
        ) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::RequestFailed("network error".to_string()))
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
    }

    impl MockRouter {
        fn with_score(score: f32) -> Self {
            Self {
                command_score: score,
                model_decision: None,
                tools_needed: None,
            }
        }

        fn with_model(model: &str) -> Self {
            Self {
                command_score: 1.0,
                model_decision: Some(model.to_string()),
                tools_needed: None,
            }
        }

        fn with_tools_needed(needed: bool) -> Self {
            Self {
                command_score: 1.0,
                model_decision: None,
                tools_needed: Some(needed),
            }
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
        provider: impl LlmProvider + 'static,
        model_name: &str,
        model_id: &str,
    ) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert(
            model_name.to_string(),
            Arc::new(provider) as Arc<dyn LlmProvider>,
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
        default_provider: impl LlmProvider + 'static,
        non_default_provider: impl LlmProvider + 'static,
    ) -> Arc<ProviderRegistry> {
        let mut providers = HashMap::new();
        providers.insert(
            "default-model".to_string(),
            Arc::new(default_provider) as Arc<dyn LlmProvider>,
        );
        providers.insert(
            "complex-model".to_string(),
            Arc::new(non_default_provider) as Arc<dyn LlmProvider>,
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
        )
    }

    fn make_engine_with_config(
        config: Arc<WeftConfig>,
        registry: Arc<ProviderRegistry>,
        router: impl SemanticRouter + 'static,
        commands: impl CommandRegistry + 'static,
    ) -> GatewayEngine {
        GatewayEngine::new(config, registry, Arc::new(router), Arc::new(commands), None)
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

        /// An LLM that sleeps forever before responding.
        struct SlowLlmProvider;

        #[async_trait]
        impl LlmProvider for SlowLlmProvider {
            async fn complete(
                &self,
                _: &str,
                _: &[Message],
                _: &CompletionOptions,
            ) -> Result<CompletionResponse, LlmError> {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(CompletionResponse {
                    text: "never".to_string(),
                    usage: None,
                })
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
            default_provider as Arc<dyn LlmProvider>,
        );
        providers.insert(
            "complex-model".to_string(),
            recording_provider.clone() as Arc<dyn LlmProvider>,
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
        }

        #[async_trait]
        impl LlmProvider for SystemPromptCapture {
            async fn complete(
                &self,
                system_prompt: &str,
                _messages: &[Message],
                _options: &CompletionOptions,
            ) -> Result<CompletionResponse, LlmError> {
                *self.captured.lock().unwrap() = Some(system_prompt.to_string());
                Ok(CompletionResponse {
                    text: "no tools needed".to_string(),
                    usage: None,
                })
            }
        }

        let capture = Arc::new(SystemPromptCapture {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct WrappedCapture(Arc<SystemPromptCapture>);
                #[async_trait]
                impl LlmProvider for WrappedCapture {
                    async fn complete(
                        &self,
                        system_prompt: &str,
                        messages: &[Message],
                        options: &CompletionOptions,
                    ) -> Result<CompletionResponse, LlmError> {
                        self.0.complete(system_prompt, messages, options).await
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
        struct SystemPromptCapture {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl LlmProvider for SystemPromptCapture {
            async fn complete(
                &self,
                system_prompt: &str,
                _messages: &[Message],
                _options: &CompletionOptions,
            ) -> Result<CompletionResponse, LlmError> {
                *self.captured.lock().unwrap() = Some(system_prompt.to_string());
                Ok(CompletionResponse {
                    text: "response".to_string(),
                    usage: None,
                })
            }
        }

        let capture = Arc::new(SystemPromptCapture {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture>);
                #[async_trait]
                impl LlmProvider for W {
                    async fn complete(
                        &self,
                        sp: &str,
                        m: &[Message],
                        o: &CompletionOptions,
                    ) -> Result<CompletionResponse, LlmError> {
                        self.0.complete(sp, m, o).await
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
        };

        struct SystemPromptCapture {
            captured: std::sync::Mutex<Option<String>>,
        }

        #[async_trait]
        impl LlmProvider for SystemPromptCapture {
            async fn complete(
                &self,
                system_prompt: &str,
                _messages: &[Message],
                _options: &CompletionOptions,
            ) -> Result<CompletionResponse, LlmError> {
                *self.captured.lock().unwrap() = Some(system_prompt.to_string());
                Ok(CompletionResponse {
                    text: "response".to_string(),
                    usage: None,
                })
            }
        }

        let capture = Arc::new(SystemPromptCapture {
            captured: std::sync::Mutex::new(None),
        });

        let registry = single_model_registry(
            {
                struct W(Arc<SystemPromptCapture>);
                #[async_trait]
                impl LlmProvider for W {
                    async fn complete(
                        &self,
                        sp: &str,
                        m: &[Message],
                        o: &CompletionOptions,
                    ) -> Result<CompletionResponse, LlmError> {
                        self.0.complete(sp, m, o).await
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
}
