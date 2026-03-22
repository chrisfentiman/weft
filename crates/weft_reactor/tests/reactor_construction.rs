//! Reactor construction validation tests.
//!
//! Tests that Reactor::new validates its configuration and returns appropriate
//! errors for missing or invalid inputs.

mod harness;

use std::sync::Arc;

use weft_reactor::config::{
    ActivityRef, BudgetConfig, LoopHooks, PipelineConfig, ReactorConfig, RetryPolicy,
};
use weft_reactor::error::ReactorError;
use weft_reactor::reactor::Reactor;
use weft_reactor::test_support::make_test_services;

use harness::{TestActivity, build_registry, reactor_config, test_event_log};

#[allow(unused_imports)]
use pretty_assertions::{assert_eq, assert_ne};

#[test]
fn reactor_new_missing_default_pipeline_returns_error() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);
    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "other".to_string(), // not "default"
            pre_loop: vec![],
            post_loop: vec![],
            generate: ActivityRef::Name("generate".to_string()),
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let result = Reactor::new(services, event_log, registry, &config);
    assert!(
        matches!(result, Err(ReactorError::Config(_))),
        "expected Config error for missing default pipeline"
    );
}

#[test]
fn reactor_new_unknown_activity_returns_error() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    // Registry has no "nonexistent" activity.
    let registry = build_registry(vec![
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);
    let config = reactor_config(PipelineConfig {
        name: "default".to_string(),
        pre_loop: vec![],
        post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
        generate: ActivityRef::Name("nonexistent".to_string()),
        execute_command: ActivityRef::Name("execute_command".to_string()),
        loop_hooks: LoopHooks::default(),
    });

    let result = Reactor::new(services, event_log, registry, &config);
    assert!(
        matches!(result, Err(ReactorError::ActivityNotFound(_))),
        "expected ActivityNotFound error"
    );
}

#[test]
fn reactor_new_activity_with_retry_resolves_correctly() {
    let services = Arc::new(make_test_services());
    let event_log = test_event_log();
    let registry = build_registry(vec![
        TestActivity::generate("generate").build(),
        TestActivity::assemble_response().into(),
        TestActivity::execute_command().into(),
    ]);
    let config = ReactorConfig {
        pipelines: vec![PipelineConfig {
            name: "default".to_string(),
            pre_loop: vec![],
            post_loop: vec![ActivityRef::Name("assemble_response".to_string())],
            generate: ActivityRef::WithConfig {
                name: "generate".to_string(),
                config: serde_json::Value::Null,
                retry: Some(RetryPolicy {
                    max_retries: 3,
                    initial_backoff_ms: 100,
                    max_backoff_ms: 1000,
                    backoff_multiplier: 2.0,
                }),
                timeout_secs: Some(60),
                heartbeat_interval_secs: None,
            },
            execute_command: ActivityRef::Name("execute_command".to_string()),
            loop_hooks: LoopHooks::default(),
        }],
        budget: BudgetConfig {
            max_generation_calls: 5,
            max_iterations: 5,
            max_depth: 3,
            timeout_secs: 300,
            generation_timeout_secs: 60,
            command_timeout_secs: 10,
        },
    };

    let result = Reactor::new(services, event_log, registry, &config);
    assert!(
        result.is_ok(),
        "should construct with retry policy: {:?}",
        result
    );
}
