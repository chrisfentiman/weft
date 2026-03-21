//! Reactor construction: `Reactor::new` and `compile_pipeline`.
//!
//! Resolves all activity name references from config to `Arc<dyn Activity>`,
//! returning a `ReactorError::ActivityNotFound` if any name is missing from
//! the registry.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{PipelineConfig, ReactorConfig};
use crate::error::ReactorError;
use crate::event_log::EventLog;
use crate::registry::ActivityRegistry;
use crate::services::Services;

use super::types::{CompiledLoopHooks, CompiledPipeline, ResolvedActivity};
use super::Reactor;

impl Reactor {
    /// Build a Reactor from config.
    ///
    /// Resolves all activity name references to Arc<dyn Activity>.
    /// Returns ReactorError::ActivityNotFound if any activity name
    /// in any pipeline config is not in the registry.
    pub fn new(
        services: Arc<Services>,
        event_log: Arc<dyn EventLog>,
        registry: Arc<ActivityRegistry>,
        config: &ReactorConfig,
    ) -> Result<Self, ReactorError> {
        let mut pipelines = HashMap::new();
        for pipeline_config in &config.pipelines {
            let compiled = Self::compile_pipeline(pipeline_config, &registry)?;
            pipelines.insert(pipeline_config.name.clone(), compiled);
        }

        if !pipelines.contains_key("default") {
            return Err(ReactorError::Config(
                "no 'default' pipeline defined".to_string(),
            ));
        }

        Ok(Self {
            services,
            event_log,
            registry,
            pipelines,
            budget_defaults: config.budget.clone(),
        })
    }

    pub(super) fn compile_pipeline(
        config: &PipelineConfig,
        registry: &ActivityRegistry,
    ) -> Result<CompiledPipeline, ReactorError> {
        let resolve =
            |activity_ref: &crate::config::ActivityRef| -> Result<ResolvedActivity, ReactorError> {
                let name = activity_ref.name();
                let activity = registry
                    .get(name)
                    .ok_or_else(|| ReactorError::ActivityNotFound(name.to_string()))?
                    .clone();
                Ok(ResolvedActivity {
                    activity,
                    retry_policy: activity_ref.retry_policy().cloned(),
                    timeout_secs: activity_ref.timeout_secs(),
                    heartbeat_interval_secs: activity_ref.heartbeat_interval_secs(),
                })
            };

        let pre_loop = config
            .pre_loop
            .iter()
            .map(resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let post_loop = config
            .post_loop
            .iter()
            .map(resolve)
            .collect::<Result<Vec<_>, _>>()?;
        let generate = resolve(&config.generate)?;
        let execute_command = resolve(&config.execute_command)?;

        let loop_hooks = CompiledLoopHooks {
            pre_generate: config
                .loop_hooks
                .pre_generate
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
            pre_response: config
                .loop_hooks
                .pre_response
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
            pre_tool_use: config
                .loop_hooks
                .pre_tool_use
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
            post_tool_use: config
                .loop_hooks
                .post_tool_use
                .iter()
                .map(resolve)
                .collect::<Result<Vec<_>, _>>()?,
        };

        Ok(CompiledPipeline {
            config: config.clone(),
            pre_loop,
            post_loop,
            generate,
            execute_command,
            loop_hooks,
        })
    }
}
