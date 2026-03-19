-- weft_eventlog_postgres/schema.sql
-- PostgresEventLog DDL. Applied externally before starting the service.
-- This file is NOT executed by Rust code. Apply via psql or a migration tool.

-- Each pipeline execution gets a record.
CREATE TABLE pipeline_executions (
    execution_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    parent_id       UUID REFERENCES pipeline_executions(execution_id),
    pipeline_name   TEXT NOT NULL DEFAULT 'default',
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    depth           INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    budget_json     JSONB NOT NULL
);

-- Find child executions of a parent.
CREATE INDEX idx_executions_parent ON pipeline_executions(parent_id)
    WHERE parent_id IS NOT NULL;

-- Find active executions by tenant.
CREATE INDEX idx_executions_active ON pipeline_executions(tenant_id, status)
    WHERE status = 'running';

-- The event log. Append-only. Every state change is an event.
CREATE TABLE pipeline_events (
    event_id        BIGSERIAL PRIMARY KEY,
    execution_id    UUID NOT NULL REFERENCES pipeline_executions(execution_id),
    sequence_num    INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    idempotency_key TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (execution_id, sequence_num)
);

-- Find events by idempotency key (deduplication on retry/replay).
CREATE UNIQUE INDEX idx_events_idempotency
    ON pipeline_events(execution_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Primary read pattern: events for an execution in order.
CREATE INDEX idx_events_execution_seq
    ON pipeline_events(execution_id, sequence_num);

-- Find events by type (analytics, debugging).
CREATE INDEX idx_events_type
    ON pipeline_events(execution_id, event_type);

-- Signal table. External events injected into running pipelines.
-- For distributed execution: external systems INSERT signals here.
-- A background task reads them and pushes onto the event channel.
CREATE TABLE pipeline_signals (
    signal_id       BIGSERIAL PRIMARY KEY,
    execution_id    UUID NOT NULL REFERENCES pipeline_executions(execution_id),
    signal_type     TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}',
    consumed        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Find unconsumed signals for an execution.
CREATE INDEX idx_signals_unconsumed
    ON pipeline_signals(execution_id, consumed)
    WHERE consumed = FALSE;

-- Task queue for distributed execution (future).
-- Workers claim tasks via SELECT FOR UPDATE SKIP LOCKED.
CREATE TABLE pipeline_tasks (
    task_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id    UUID NOT NULL REFERENCES pipeline_executions(execution_id),
    task_type       TEXT NOT NULL,
    payload         JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'claimed', 'completed', 'failed')),
    claimed_by      TEXT,
    claimed_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    result          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tasks_pending
    ON pipeline_tasks(task_type, status)
    WHERE status = 'pending';
