//! Built-in activity implementations for the reactive pipeline.
//!
//! Each activity implements the [`crate::activity::Activity`] trait and is
//! registered in the [`crate::registry::ActivityRegistry`] at startup. Activities
//! receive an `mpsc::Sender<PipelineEvent>` and push events onto the channel as
//! they work — they do not return values.
//!
//! Built-in activities:
//! - [`ValidateActivity`] — validates the request and populates available commands
//! - [`ModelSelectionActivity`] — selects the model via semantic routing (pre-loop)
//! - [`CommandSelectionActivity`] — selects relevant commands via semantic routing (pre-loop)
//! - [`RouteActivity`] — performs semantic routing across all configured domains (legacy)
//! - [`AssemblePromptActivity`] — builds the system prompt and message list (legacy)
//! - [`GenerateActivity`] — calls the generative source and streams tokens
//! - [`ExecuteCommandActivity`] — executes a single command invocation
//! - [`AssembleResponseActivity`] — constructs the final WeftResponse
//! - [`HookActivity`] — wraps the HookRunner at a specific lifecycle point

pub mod assemble_prompt;
pub mod assemble_response;
pub mod command_selection;
pub mod execute_command;
pub mod generate;
pub mod hooks;
pub mod model_selection;
pub mod route;
pub mod validate;

pub use assemble_prompt::AssemblePromptActivity;
pub use assemble_response::AssembleResponseActivity;
pub use command_selection::CommandSelectionActivity;
pub use execute_command::ExecuteCommandActivity;
pub use generate::GenerateActivity;
pub use hooks::HookActivity;
pub use model_selection::ModelSelectionActivity;
pub use route::RouteActivity;
pub use validate::ValidateActivity;
