-- Auth user table migration for non-Docker setups.
-- Run with: clickhouse-client --database ic_agent -f db/auth/create_user_table.sql
-- Or: clickhouse-client --multiquery < db/auth/create_user_table.sql

CREATE DATABASE IF NOT EXISTS ic_agent;

CREATE TABLE IF NOT EXISTS ic_agent.ic_rag_agent_user (
    user_id UUID,
    user_name String,
    email String DEFAULT '',
    password_hash String,
    role LowCardinality(String) DEFAULT 'general',
    status LowCardinality(String) DEFAULT 'active',
    created_time DateTime64(3),
    updated_time DateTime64(3),
    last_login_time Nullable(DateTime64(3)),
    last_login_ip Nullable(String),
    metadata String DEFAULT '{}'
) ENGINE = ReplacingMergeTree(updated_time)
ORDER BY (user_id);

-- Indexes for user_name and email lookups
CREATE INDEX IF NOT EXISTS idx_user_name ON ic_agent.ic_rag_agent_user (user_name) TYPE bloom_filter(0.01) GRANULARITY 1;
CREATE INDEX IF NOT EXISTS idx_user_email ON ic_agent.ic_rag_agent_user (email) TYPE bloom_filter(0.01) GRANULARITY 1;
