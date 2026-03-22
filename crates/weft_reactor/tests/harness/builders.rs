//! Builder types and registry/config convenience helpers for `TestActivity`.
//!
//! This module is `pub(super)` — consumers use the entry-point methods on
//! `TestActivity` in `mod.rs`, not these structs directly.
#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::AtomicU32,
};

use weft_core::{
    ContentPart, ModelRoutingInstruction, Role, SamplingOptions, Source, WeftMessage, WeftRequest,
};
use weft_reactor::activity::Activity;
use weft_reactor::config::{ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig};
use weft_reactor::event::{
    CommandEvent, CommandFormat, ContextEvent, ExecutionEvent, FailureDetail, MessageInjectionSource,
    PipelineEvent, SelectionEvent,
};
use weft_reactor::registry::ActivityRegistry;
use weft_reactor::test_support::TestEventLog;
use weft_reactor_trait::Criticality;

use super::{Behavior, CallAction, PreStallBehavior, TestActivity};

// ── GenerateBuilder ───────────────────────────────────────────────────────────

pub struct GenerateBuilder {
    pub(super) name: String,
    pub(super) criticality: Criticality,
    pub(super) response_text: Option<String>,
    pub(super) input_tokens: Option<u32>,
    pub(super) output_tokens: Option<u32>,
}

impl GenerateBuilder {
    /// Set response text (default: None — no content chunk emitted).
    pub fn with_text(mut self, text: &str) -> Self {
        self.response_text = Some(text.to_string());
        self
    }

    /// Set token counts.
    pub fn with_tokens(mut self, input: u32, output: u32) -> Self {
        self.input_tokens = Some(input);
        self.output_tokens = Some(output);
        self
    }

    /// Switch to fail-then-succeed behavior. Returns Arc directly.
    pub fn fails_then_succeeds(self, fail_count: u32) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::FailThenSucceed {
                fail_count: Arc::new(AtomicU32::new(fail_count)),
            },
        })
    }

    /// Switch to per-call behavior: first call invokes commands, then generates Done.
    pub fn invokes_commands(self, commands: Vec<&str>) -> PerCallBuilder {
        PerCallBuilder {
            name: self.name,
            criticality: self.criticality,
            call_count: Arc::new(AtomicU32::new(0)),
            first_call_commands: commands.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Set criticality (default: Critical).
    pub fn with_criticality(mut self, c: Criticality) -> Self {
        self.criticality = c;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::Generate {
                response_text: self.response_text,
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
            },
        })
    }
}

// ── PerCallBuilder ────────────────────────────────────────────────────────────

pub struct PerCallBuilder {
    pub(super) name: String,
    pub(super) criticality: Criticality,
    pub(super) call_count: Arc<AtomicU32>,
    pub(super) first_call_commands: Vec<String>,
}

impl PerCallBuilder {
    /// Inject a shared call counter so tests can observe call counts.
    pub fn with_call_count(mut self, counter: Arc<AtomicU32>) -> Self {
        self.call_count = counter;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::PerCall {
                call_count: self.call_count,
                actions: vec![CallAction::WithCommands {
                    commands: self.first_call_commands,
                }],
                default_action: CallAction::Done,
            },
        })
    }
}

// ── EmittingBuilder ───────────────────────────────────────────────────────────

pub struct EmittingBuilder {
    pub(super) name: String,
    pub(super) criticality: Criticality,
    pub(super) events: Vec<PipelineEvent>,
}

impl EmittingBuilder {
    /// Add a PipelineEvent to emit between Started and Completed.
    pub fn event(mut self, event: PipelineEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Set criticality.
    pub fn with_criticality(mut self, c: Criticality) -> Self {
        self.criticality = c;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::EmitEvents {
                events: self.events,
            },
        })
    }
}

// ── FailBuilder ───────────────────────────────────────────────────────────────

pub struct FailBuilder {
    pub(super) name: String,
    pub(super) criticality: Criticality,
    pub(super) error: String,
    pub(super) retryable: bool,
    pub(super) detail: FailureDetail,
}

impl FailBuilder {
    /// Set error message.
    pub fn with_error(mut self, msg: &str) -> Self {
        self.error = msg.to_string();
        self
    }

    /// Set retryable flag.
    pub fn retryable(mut self, r: bool) -> Self {
        self.retryable = r;
        self
    }

    /// Set criticality.
    pub fn with_criticality(mut self, c: Criticality) -> Self {
        self.criticality = c;
        self
    }

    /// Set FailureDetail fields.
    pub fn with_detail(mut self, detail: FailureDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Build into Arc<dyn Activity>.
    pub fn build(self) -> Arc<dyn Activity> {
        Arc::new(TestActivity {
            activity_name: self.name,
            criticality: self.criticality,
            behavior: Behavior::Fail {
                error: self.error,
                retryable: self.retryable,
                detail: self.detail,
            },
        })
    }
}

// ── Pre-loop activity stubs ───────────────────────────────────────────────────

/// Stub for validate: emits ValidationPassed + CommandsAvailable.
pub fn validate_stub() -> Arc<dyn Activity> {
    TestActivity::emitting("validate")
        .event(PipelineEvent::Execution(ExecutionEvent::ValidationPassed))
        .event(PipelineEvent::Command(CommandEvent::Available {
            commands: vec![],
        }))
        .build()
}

/// Stub for model_selection: emits ModelSelected with stub-model.
pub fn model_selection_stub() -> Arc<dyn Activity> {
    TestActivity::emitting("model_selection")
        .event(PipelineEvent::Selection(SelectionEvent::ModelSelected {
            model_name: "stub-model".to_string(),
            score: 0.9,
        }))
        .build()
}

/// Stub for command_selection: emits CommandsSelected (empty).
pub fn command_selection_stub() -> Arc<dyn Activity> {
    TestActivity::emitting("command_selection")
        .event(PipelineEvent::Selection(SelectionEvent::CommandsSelected {
            selected: vec![],
        }))
        .build()
}

/// Stub for provider_resolution: emits ProviderResolved with stub values.
pub fn provider_resolution_stub() -> Arc<dyn Activity> {
    TestActivity::emitting("provider_resolution")
        .event(PipelineEvent::Selection(SelectionEvent::ProviderResolved {
            model_name: "stub-model".to_string(),
            model_id: "stub-model-v1".to_string(),
            provider_name: "anthropic".to_string(),
            capabilities: vec![],
            max_tokens: 4096,
        }))
        .build()
}

/// Stub for system_prompt_assembly: emits MessageInjected + SystemPromptAssembled.
pub fn system_prompt_assembly_stub() -> Arc<dyn Activity> {
    let system_message = WeftMessage {
        role: Role::System,
        source: Source::Provider,
        model: None,
        content: vec![ContentPart::Text("You are helpful.".to_string())],
        delta: false,
        message_index: 0,
    };
    TestActivity::emitting("system_prompt_assembly")
        .event(PipelineEvent::Context(ContextEvent::MessageInjected {
            message: system_message,
            source: MessageInjectionSource::SystemPromptAssembly,
        }))
        .event(PipelineEvent::Context(
            ContextEvent::SystemPromptAssembled { message_count: 1 },
        ))
        .build()
}

/// Stub for command_formatting: emits CommandsFormatted (NoCommands).
pub fn command_formatting_stub() -> Arc<dyn Activity> {
    TestActivity::emitting("command_formatting")
        .event(PipelineEvent::Context(ContextEvent::CommandsFormatted {
            format: CommandFormat::NoCommands,
            command_count: 0,
        }))
        .build()
}

/// Stub for sampling_adjustment: emits SamplingUpdated (max_tokens 4096).
pub fn sampling_adjustment_stub() -> Arc<dyn Activity> {
    TestActivity::emitting("sampling_adjustment")
        .event(PipelineEvent::Context(ContextEvent::SamplingUpdated {
            max_tokens: 4096,
            temperature: None,
            top_p: None,
        }))
        .build()
}

// ── Failing / hanging execute_command helpers ─────────────────────────────────

/// Build a failing execute_command activity (NonCritical, emits Failed).
pub fn failing_execute_command(error: &str, detail: FailureDetail) -> Arc<dyn Activity> {
    Arc::new(TestActivity {
        activity_name: "execute_command".to_string(),
        criticality: Criticality::NonCritical,
        behavior: Behavior::Fail {
            error: error.to_string(),
            retryable: false,
            detail,
        },
    })
}

/// Build a hanging execute_command activity (NonCritical, waits for cancel).
pub fn hanging_execute_command() -> Arc<dyn Activity> {
    Arc::new(TestActivity {
        activity_name: "execute_command".to_string(),
        criticality: Criticality::NonCritical,
        behavior: Behavior::WaitForCancel {
            pre_stall: PreStallBehavior::None,
        },
    })
}

// ── Registry convenience helpers ──────────────────────────────────────────────

/// Registry with generate + assemble_response + execute_command.
pub fn simple_registry(generate: Arc<dyn Activity>) -> Arc<ActivityRegistry> {
    build_registry(vec![
        generate,
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ])
}

/// Registry with full pre-loop pipeline + generate + assemble_response + execute_command.
pub fn preloop_registry(generate: Arc<dyn Activity>) -> Arc<ActivityRegistry> {
    build_registry(vec![
        validate_stub(),
        model_selection_stub(),
        command_selection_stub(),
        provider_resolution_stub(),
        system_prompt_assembly_stub(),
        command_formatting_stub(),
        sampling_adjustment_stub(),
        generate,
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ])
}

// ── Reactor / pipeline config helpers ────────────────────────────────────────

/// Build a simple pipeline config with a single generate activity and no pre/post activities.
pub fn simple_pipeline_config(generate_name: &str) -> PipelineConfig {
    PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name(generate_name.to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    }
}

/// Build a pipeline config with validate in pre-loop.
pub fn pipeline_with_validate(generate_name: &str) -> PipelineConfig {
    PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![ActivityRef::Name("validate".to_string())],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name(generate_name.to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    }
}

/// Build a ReactorConfig with the given pipeline config.
pub fn reactor_config(pipeline: PipelineConfig) -> ReactorConfig {
    ReactorConfig {
        pipelines: vec![pipeline],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    }
}

/// Build a minimal WeftRequest for testing.
pub fn test_request() -> WeftRequest {
    WeftRequest {
        messages: vec![WeftMessage {
            role: Role::User,
            source: Source::Client,
            model: None,
            content: vec![ContentPart::Text("hello".to_string())],
            delta: false,
            message_index: 0,
        }],
        routing: ModelRoutingInstruction::parse("auto"),
        options: SamplingOptions::default(),
    }
}

/// Build a registry with all the given activities registered.
pub fn build_registry(activities: Vec<Arc<dyn Activity>>) -> Arc<ActivityRegistry> {
    let mut registry = ActivityRegistry::new();
    for activity in activities {
        registry
            .register(activity)
            .expect("duplicate activity name in test");
    }
    Arc::new(registry)
}

/// Build a TestEventLog for testing.
pub fn test_event_log() -> Arc<TestEventLog> {
    Arc::new(TestEventLog::new())
}

/// Build a registry with the full pre-loop activity set.
pub fn build_new_preloop_registry(generate_name: &str) -> Arc<ActivityRegistry> {
    preloop_registry(TestActivity::generate(generate_name).build())
}

/// Build a pipeline config with the full pre-loop activity set.
pub fn new_preloop_pipeline_config(generate_name: &str) -> PipelineConfig {
    PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![
            ActivityRef::Name("validate".to_string()),
            ActivityRef::Name("model_selection".to_string()),
            ActivityRef::Name("command_selection".to_string()),
            ActivityRef::Name("provider_resolution".to_string()),
            ActivityRef::Name("system_prompt_assembly".to_string()),
            ActivityRef::Name("command_formatting".to_string()),
            ActivityRef::Name("sampling_adjustment".to_string()),
        ],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name(generate_name.to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    }
}
