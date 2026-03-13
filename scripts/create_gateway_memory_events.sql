CREATE TABLE IF NOT EXISTS rag_agent_message_event (
    ts DateTime64(3, 'UTC'),
    user_id String,
    session_id String,
    request_id String,
    event_type String,
    event_content String,
    status String,
    note String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (user_id, session_id, ts, request_id);
