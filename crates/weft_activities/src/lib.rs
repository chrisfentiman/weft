//! `weft_activities` — Built-in activity implementations for the reactive pipeline.
//!
//! Each activity implements the [`weft_reactor_trait::Activity`] trait and is
//! registered in the `ActivityRegistry` at startup. Activities receive an
//! `mpsc::Sender<PipelineEvent>` and push events onto the channel as they work.
//!
//! # Built-in activities
//!
//! - [`ValidateActivity`] — validates the request and populates available commands
//! - [`ModelSelectionActivity`] — selects the model via semantic routing (pre-loop)
//! - [`CommandSelectionActivity`] — selects relevant commands via semantic routing (pre-loop)
//! - [`ProviderResolutionActivity`] — resolves provider and capabilities for the selected model (pre-loop)
//! - [`SystemPromptAssemblyActivity`] — layers the system prompt from gateway config and caller (pre-loop)
//! - [`CommandFormattingActivity`] — formats selected commands for the provider (pre-loop)
//! - [`SamplingAdjustmentActivity`] — clamps sampling parameters to model constraints (pre-loop)
//! - [`GenerateActivity`] — calls the generative source and streams tokens
//! - [`ExecuteCommandActivity`] — executes a single command invocation
//! - [`AssembleResponseActivity`] — constructs the final WeftResponse
//! - [`HookActivity`] — wraps the HookRunner at a specific lifecycle point
//!
//! # Dependencies
//!
//! This crate depends only on trait crates for the reactor contract, plus
//! concrete crates that activities directly call (weft_commands for
//! parse_response, weft_router for routing utilities, etc.).

pub mod assemble_response;
pub mod command_formatting;
pub mod command_selection;
pub mod execute_command;
pub mod generate;
pub mod hooks;
pub mod model_selection;
pub mod provider_resolution;
pub mod sampling_adjustment;
pub(crate) mod selection_util;
pub mod system_prompt_assembly;
pub mod validate;

pub use assemble_response::AssembleResponseActivity;
pub use command_formatting::CommandFormattingActivity;
pub use command_selection::CommandSelectionActivity;
pub use execute_command::ExecuteCommandActivity;
pub use generate::GenerateActivity;
pub use hooks::HookActivity;
pub use model_selection::ModelSelectionActivity;
pub use provider_resolution::ProviderResolutionActivity;
pub use sampling_adjustment::SamplingAdjustmentActivity;
pub use system_prompt_assembly::SystemPromptAssemblyActivity;
pub use validate::ValidateActivity;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
