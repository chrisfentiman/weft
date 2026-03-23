//! OpenAI Chat Completions API provider implementation.

pub mod client;
pub mod translate;
pub mod wire;

pub use client::OpenAIProvider;
