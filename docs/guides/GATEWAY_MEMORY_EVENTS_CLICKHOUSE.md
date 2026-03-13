# Gateway Memory Events ClickHouse Setup

## Purpose

Create the `rag_agent_message_event` table used by the new short-term and long-term memory event envelope.

## SQL file

- `scripts/create_gateway_memory_events.sql`

## ECS execution example

Run inside the ClickHouse container or host where ClickHouse client is available:

```bash
clickhouse-client --host <CH_HOST> --port <CH_TCP_PORT> --user <CH_USER> --password '<CH_PASSWORD>' --database <CH_DB> --multiquery < scripts/create_gateway_memory_events.sql
```

## Verify table

```sql
DESCRIBE TABLE rag_agent_message_event;
```

Expected columns:

- `ts`
- `user_id`
- `session_id`
- `request_id`
- `event_type`
- `event_content`
- `status`
- `note`
