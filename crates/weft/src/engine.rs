//! Gateway engine: the core loop that wires all components together.
//!
//! The engine classifies commands, assembles context, calls the LLM, parses
//! commands from the response, executes them, and loops until no more commands
//! are emitted or the iteration cap or timeout is reached.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, warn};
use weft_classifier::{ClassificationResult, SemanticClassifier, filter_by_threshold, take_top};
use weft_commands::{CommandRegistry, parse_response};
use weft_core::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, CommandResult, CommandStub, Message,
    Role, Usage, WeftConfig, WeftError,
};
use weft_llm::{CompletionOptions, LlmProvider};

use crate::context::{assemble_system_prompt, format_command_results_toon};

/// The gateway engine: holds shared components and drives the request loop.
///
/// All fields are `Arc` so `GatewayEngine` is cheaply `Clone`able — axum clones
/// it into each request handler.
#[derive(Clone)]
pub struct GatewayEngine {
    config: Arc<WeftConfig>,
    llm_provider: Arc<dyn LlmProvider>,
    classifier: Arc<dyn SemanticClassifier>,
    command_registry: Arc<dyn CommandRegistry>,
}

impl GatewayEngine {
    /// Expose the config for use by the health handler and other modules.
    pub fn config(&self) -> &WeftConfig {
        &self.config
    }

    /// Construct a new gateway engine.
    pub fn new(
        config: Arc<WeftConfig>,
        llm_provider: Arc<dyn LlmProvider>,
        classifier: Arc<dyn SemanticClassifier>,
        command_registry: Arc<dyn CommandRegistry>,
    ) -> Self {
        Self {
            config,
            llm_provider,
            classifier,
            command_registry,
        }
    }

    /// Handle a single chat completion request.
    ///
    /// This is the main gateway loop: classify → assemble → LLM → parse → execute → loop.
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

        // Extract the latest user message for semantic classification.
        let user_message = extract_latest_user_message(&request.messages)?;

        // Semantic classification: filter commands relevant to this request.
        let selected_commands = self.classify_and_filter(user_message, &all_commands).await;

        // Assemble the system prompt with foundational + TOON stubs + agent prompt.
        let system_prompt =
            assemble_system_prompt(&selected_commands, &self.config.gateway.system_prompt);

        // Build the initial message list (clone from request).
        let mut messages = request.messages.clone();

        let options = CompletionOptions {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        // Build the set of known command names for the parser.
        let known_commands: HashSet<String> = all_commands.iter().map(|c| c.name.clone()).collect();

        let max_iterations = self.config.gateway.max_command_iterations;
        let mut iterations = 0u32;

        loop {
            if iterations >= max_iterations {
                return Err(WeftError::CommandLoopExceeded {
                    max: max_iterations,
                });
            }

            // Call the LLM.
            let completion = self
                .llm_provider
                .complete(&system_prompt, &messages, &options)
                .await
                .map_err(|e| {
                    use weft_llm::LlmError;
                    match &e {
                        // Map rate-limit to a typed WeftError variant.
                        LlmError::RateLimited { retry_after_ms } => WeftError::RateLimited {
                            retry_after_ms: *retry_after_ms,
                        },
                        _ => WeftError::Llm(e.to_string()),
                    }
                })?;

            // Parse the response for slash commands.
            let parsed = parse_response(&completion.text, &known_commands);

            // Collect any parse-error results alongside invocation results.
            if parsed.invocations.is_empty() && parsed.parse_errors.is_empty() {
                // No commands — we're done. Return the clean text.
                return Ok(build_response(
                    &parsed.text,
                    &request.model,
                    completion.usage,
                ));
            }

            // Execute commands sequentially (spec Section 8.4).
            let mut results: Vec<CommandResult> = Vec::new();

            // Include parse errors as failed results so the LLM can see them.
            results.extend(parsed.parse_errors);

            // Execute each valid invocation.
            for invocation in &parsed.invocations {
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

            // If there were only parse errors (no valid invocations) and they were all
            // from `parse_errors`, the LLM still gets the error results. Continue the loop.

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

    /// Classify commands relevant to the user message and apply threshold + max filtering.
    ///
    /// On classifier failure, falls back to all commands (spec Section 5.5).
    async fn classify_and_filter(
        &self,
        user_message: &str,
        all_commands: &[CommandStub],
    ) -> Vec<CommandStub> {
        let threshold = self.config.classifier.threshold;
        let max_commands = self.config.classifier.max_commands;

        let scores = self.classifier.classify(user_message, all_commands).await;

        match scores {
            Ok(results) => {
                // Filter by threshold, then take top N.
                let filtered = filter_by_threshold(results, threshold);
                let top: Vec<ClassificationResult> = take_top(filtered, max_commands);

                // Convert back to CommandStub by matching names.
                let score_map: std::collections::HashMap<&str, f32> = top
                    .iter()
                    .map(|r| (r.command_name.as_str(), r.score))
                    .collect();

                let mut selected: Vec<CommandStub> = all_commands
                    .iter()
                    .filter(|cmd| score_map.contains_key(cmd.name.as_str()))
                    .cloned()
                    .collect();

                // Preserve the top-score ordering from `take_top`.
                selected.sort_by(|a, b| {
                    let sa = score_map.get(a.name.as_str()).copied().unwrap_or(0.0);
                    let sb = score_map.get(b.name.as_str()).copied().unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });

                debug!(
                    selected_count = selected.len(),
                    total_count = all_commands.len(),
                    "semantic classifier selected commands"
                );

                selected
            }
            Err(e) => {
                // Classifier failure: fall back to all commands, sorted alphabetically.
                warn!(
                    error = %e,
                    "semantic classifier failed, falling back to all commands"
                );

                let mut fallback: Vec<CommandStub> =
                    all_commands.iter().take(max_commands).cloned().collect();
                fallback.sort_by(|a, b| a.name.cmp(&b.name));
                fallback
            }
        }
    }
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
    use std::sync::Arc;
    use weft_classifier::{ClassificationResult, ClassifierError, SemanticClassifier};
    use weft_commands::{CommandError, CommandRegistry};
    use weft_core::{
        CommandAction, CommandDescription, CommandInvocation, CommandResult, CommandStub,
        GatewayConfig, LlmConfig, LlmProviderKind, Message, Role, ServerConfig, WeftConfig,
    };
    use weft_llm::{CompletionOptions, CompletionResponse, LlmError, LlmProvider};

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

    /// A mock classifier that always succeeds with configurable scores.
    struct MockClassifier {
        score: f32,
    }

    impl MockClassifier {
        fn with_score(score: f32) -> Self {
            Self { score }
        }

        fn failing() -> FailingClassifier {
            FailingClassifier
        }
    }

    #[async_trait]
    impl SemanticClassifier for MockClassifier {
        async fn classify(
            &self,
            _user_message: &str,
            commands: &[CommandStub],
        ) -> Result<Vec<ClassificationResult>, ClassifierError> {
            Ok(commands
                .iter()
                .map(|c| ClassificationResult {
                    command_name: c.name.clone(),
                    score: self.score,
                })
                .collect())
        }
    }

    struct FailingClassifier;

    #[async_trait]
    impl SemanticClassifier for FailingClassifier {
        async fn classify(
            &self,
            _: &str,
            _: &[CommandStub],
        ) -> Result<Vec<ClassificationResult>, ClassifierError> {
            Err(ClassifierError::ModelNotLoaded)
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
            llm: LlmConfig {
                provider: LlmProviderKind::Anthropic,
                api_key: "test-key".to_string(),
                model: "claude-test".to_string(),
                max_tokens: 1024,
                base_url: None,
            },
            classifier: weft_core::ClassifierConfig {
                model_path: "models/test.onnx".to_string(),
                tokenizer_path: "models/tokenizer.json".to_string(),
                threshold: 0.0, // All commands pass in tests
                max_commands: 20,
            },
            tool_registry: None,
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

    fn make_engine(
        llm: impl LlmProvider + 'static,
        classifier: impl SemanticClassifier + 'static,
        registry: impl CommandRegistry + 'static,
    ) -> GatewayEngine {
        GatewayEngine::new(
            test_config(),
            Arc::new(llm),
            Arc::new(classifier),
            Arc::new(registry),
        )
    }

    // ── Test: no-command response (single pass) ────────────────────────────

    #[tokio::test]
    async fn test_no_command_response_single_pass() {
        let engine = make_engine(
            MockLlmProvider::single("Hello, I can help you with that!"),
            MockClassifier::with_score(0.9),
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
        let engine = make_engine(
            MockLlmProvider::new(vec![
                // First response: issue a command
                "/web_search query: \"Rust async\"",
                // Second response (after command result): clean answer
                "Here are the results I found.",
            ]),
            MockClassifier::with_score(0.9),
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
        // LLM emits two commands in one response, then finishes
        let engine = make_engine(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"\n/code_review target: src",
                "Done with both commands.",
            ]),
            MockClassifier::with_score(0.9),
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
        // Config with max 2 iterations
        let config = Arc::new(WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 2,
                request_timeout_secs: 30,
            },
            ..(*test_config()).clone()
        });

        // LLM always emits a command — infinite loop
        let engine = GatewayEngine::new(
            config,
            Arc::new(MockLlmProvider::single("/web_search query: \"loop\"")),
            Arc::new(MockClassifier::with_score(0.9)),
            Arc::new(
                MockCommandRegistry::new(vec![("web_search", "Search")]).with_execute_result(
                    CommandResult {
                        command_name: "web_search".to_string(),
                        success: true,
                        output: "still looping".to_string(),
                        error: None,
                    },
                ),
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
        let engine = make_engine(
            MockLlmProvider::new(vec![
                "/web_search query: \"Rust\"",
                "The search failed, but I can still answer.",
            ]),
            MockClassifier::with_score(0.9),
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

    // ── Test: classifier fallback ──────────────────────────────────────────

    #[tokio::test]
    async fn test_classifier_failure_falls_back_to_all_commands() {
        let engine = make_engine(
            MockLlmProvider::single("No commands needed."),
            MockClassifier::failing(),
            MockCommandRegistry::new(vec![
                ("web_search", "Search the web"),
                ("code_review", "Review code"),
            ]),
        );

        // Should succeed even with a failing classifier (fallback to all commands)
        let resp = engine
            .handle_request(make_user_request("Hello"))
            .await
            .expect("classifier fallback must not fail the request");

        assert!(!resp.choices[0].message.content.is_empty());
    }

    // ── Test: --describe flag ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_describe_action_handled_by_registry() {
        let engine = make_engine(
            MockLlmProvider::new(vec![
                "/web_search --describe",
                "Now I understand the command.",
            ]),
            MockClassifier::with_score(0.9),
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
        let engine = make_engine(
            MockLlmProvider::single("irrelevant"),
            MockClassifier::with_score(0.9),
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
        // Very short timeout: 1 millisecond
        let config = Arc::new(WeftConfig {
            gateway: GatewayConfig {
                system_prompt: "test".to_string(),
                max_command_iterations: 10,
                request_timeout_secs: 0, // 0 seconds = immediately timeout
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
                // Sleep for a very long time
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(CompletionResponse {
                    text: "never".to_string(),
                    usage: None,
                })
            }
        }

        let engine = GatewayEngine::new(
            config,
            Arc::new(SlowLlmProvider),
            Arc::new(MockClassifier::with_score(0.9)),
            Arc::new(MockCommandRegistry::new(vec![])),
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
        let engine = make_engine(
            MockLlmProvider::single("Test response"),
            MockClassifier::with_score(0.9),
            MockCommandRegistry::new(vec![]),
        );

        let resp = engine
            .handle_request(make_user_request("Hi"))
            .await
            .expect("should succeed");

        // Validate required OpenAI response fields
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
        let engine = make_engine(
            MockLlmProvider::single("irrelevant"),
            MockClassifier::with_score(0.9),
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
        // Should fail because there is no user message
        assert!(result.is_err(), "expected error for missing user message");
    }

    // ── Test: rate limit error propagated ─────────────────────────────────

    #[tokio::test]
    async fn test_rate_limit_error_propagated() {
        let engine = make_engine(
            RateLimitedLlmProvider,
            MockClassifier::with_score(0.9),
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
        let engine = make_engine(
            FailingLlmProvider,
            MockClassifier::with_score(0.9),
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
        let engine = make_engine(
            MockLlmProvider::single("irrelevant"),
            MockClassifier::with_score(0.9),
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
        let engine = make_engine(
            MockLlmProvider::single(""),
            MockClassifier::with_score(0.9),
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
        let engine = make_engine(
            MockLlmProvider::single("Done"),
            MockClassifier::with_score(0.9),
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
}
