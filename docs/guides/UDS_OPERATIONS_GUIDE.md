# UDS Agent Operations Guide

Guide for operating, monitoring, and maintaining the UDS Agent in production.

---

## 1. Health Monitoring

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| GET /health | Overall health (API + database) |
| GET /metrics | Prometheus-style metrics |
| GET /api/v1/uds/status | Agent status (tools, queries) |

### Health Check Response

**Healthy:**

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2026-03-06T12:00:00"
}
```

**Unhealthy:**

```json
{
  "status": "unhealthy",
  "database": "disconnected",
  "error": "Connection refused",
  "timestamp": "2026-03-06T12:00:00"
}
```

### Monitoring Setup

**Uptime / Ping:**

- Poll `GET /health` every 30–60 seconds
- Alert if status is `unhealthy` for 2+ consecutive checks

**Prometheus (if integrated):**

- Scrape `GET /metrics`
- Example alert: `uds_agent_status != "running"`

**Custom Checks:**

```bash
# Simple health check script
curl -sf http://localhost:8000/health | jq -e '.status == "healthy"' || exit 1
```

### Dependencies to Monitor

- **ClickHouse** – Primary data store; health affects all queries
- **LLM (Ollama/API)** – Required for intent, planning, and tool use
- **Redis** (optional) – Cache; failure degrades performance but not correctness

---

## 2. Performance Tuning

### API-Level

| Setting | Recommendation | Notes |
|---------|----------------|-------|
| Workers | 2–4 (uvicorn) | Match CPU cores; avoid too many for LLM contention |
| Timeout | 30–60s default | Use streaming for long queries |
| Caching | Enable UDSCache | Reduces repeated SQL/query execution |

### Database (ClickHouse)

- **RAM**: 4 GB minimum, 8 GB+ for heavy workloads
- **Disk**: Use SSD/ESSD; avoid HDD for analytics
- **Indexes**: Ensure date columns (e.g. start_date) are indexed
- **Query timeout**: Set `UDS_QUERY_TIMEOUT` (default 300s)

### LLM

- **Ollama**: Allocate 4–8 GB RAM per model
- **Remote API**: Use connection pooling, set timeouts
- **Model choice**: Smaller models (e.g. qwen3:1.7b) for faster response; larger for accuracy

### Streaming for Long Queries

Use `POST /api/v1/uds/query/stream` to avoid client timeouts on slow queries.

---

## 3. Log Analysis

### Log Locations

- **Application**: stdout/stderr (captured by Docker, systemd, or log aggregator)
- **Uvicorn**: Same as application
- **Docker**: `docker logs uds-agent`

### Log Levels

Set via `LOG_LEVEL` (e.g. DEBUG, INFO, WARNING, ERROR):

```bash
export LOG_LEVEL=INFO
```

### Key Log Patterns

| Pattern | Meaning |
|---------|---------|
| `Query failed` | Query execution error |
| `Database connection failed` | ClickHouse unreachable |
| `Circuit breaker is open` | Too many failures; retries blocked |
| `Intent: sales (confidence: 0.95)` | Intent classification result |
| `Plan: N subtasks` | Task plan created |

### Example Queries (grep/jq)

```bash
# Errors in last 1000 lines
docker logs uds-agent 2>&1 | tail -1000 | grep -i error

# Query failures
docker logs uds-agent 2>&1 | grep "Query failed"

# Circuit breaker events
docker logs uds-agent 2>&1 | grep -i "circuit breaker"
```

### Structured Logging (Future)

Consider JSON logging for easier parsing:

```python
import json
logger.info(json.dumps({"event": "query_complete", "query_id": qid, "duration": t}))
```

---

## 4. Troubleshooting Procedures

### Query Returns No Data

1. Check date range – data may be October 2025 only
2. Verify tables: `GET /api/v1/uds/tables`
3. Check schema: `GET /api/v1/uds/tables/amz_order`
4. Run a simple query: "What tables are available?"

### Query Timeout

1. Use streaming endpoint
2. Narrow date range or add filters
3. Increase `max_execution_time` in options
4. Check ClickHouse load and slow queries

### Database Connection Failed

1. Verify ClickHouse is running: `curl http://CH_HOST:8123/ping`
2. Check credentials (CH_USER, CH_PASSWORD)
3. Check network/firewall
4. Review ClickHouse logs

### LLM Errors

1. Verify Ollama: `curl http://localhost:11434/api/tags`
2. Ensure model is pulled: `ollama pull qwen3:1.7b`
3. For remote API: check URL, key, rate limits
4. Check LLM provider logs

### Circuit Breaker Open

- **Cause**: Repeated failures (e.g. DB or LLM)
- **Action**: Wait for half-open window; fix underlying issue
- **Prevention**: Resolve root cause before retrying

### Incorrect or Unexpected Results

1. Rephrase query with explicit dates and metrics
2. Check intent: review `metadata.intent` in response
3. Validate against raw SQL or sample data
4. Check for schema changes

---

## 5. Incident Response

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| P1 | Full outage (API down, DB down) | Immediate |
| P2 | Degraded (slow, partial failures) | Within 1 hour |
| P3 | Minor (single query type failing) | Within 4 hours |

### P1: Full Outage

1. **Verify**: Check `/health`, database, LLM
2. **Contain**: Restart API container if needed
3. **Communicate**: Notify users if SLA affected
4. **Fix**: Restore DB/LLM, restart services
5. **Post-mortem**: Document root cause and prevention

### P2: Degraded Performance

1. Check `/metrics` and `/api/v1/uds/status`
2. Review recent logs for errors
3. Check ClickHouse and LLM resource usage
4. Consider scaling or restarting overloaded services
5. Enable or verify caching

### P3: Partial Failure

1. Reproduce with specific query
2. Check intent classification and task plan
3. Validate tool execution and SQL
4. Fix or work around (e.g. query phrasing)
5. Add test case to prevent regression

### Rollback Procedure

```bash
# Docker
docker compose -f docker-compose.uds.yml down
docker compose -f docker-compose.uds.yml up -d --force-recreate

# Or revert to previous image
docker tag uds-agent:previous uds-agent:latest
docker compose up -d
```

---

## 6. Maintenance Procedures

### Regular Tasks

| Task | Frequency | Action |
|------|-----------|--------|
| Health check | Continuous | Automated monitoring |
| Log review | Daily | Scan for errors, anomalies |
| Dependency updates | Monthly | Update Python deps, base images |
| Backup verification | Weekly | Verify ClickHouse backups |
| Capacity review | Quarterly | Review CPU, memory, disk usage |

### Backup

- **ClickHouse**: Use `clickhouse-backup` or native backup tools
- **Configuration**: Version-control `.env` (without secrets)
- **Docker images**: Tag and push to registry before major changes

### Updates

1. Test in staging
2. Backup data and config
3. Pull new image or rebuild
4. Deploy with zero-downtime (rolling update if supported)
5. Verify health and run smoke tests

### Database Maintenance

- **ClickHouse**: OPTIMIZE TABLE, VACUUM as needed
- **Statistics**: Regenerate `uds_statistics.json` if schema or data changes

### Security Patches

- Apply OS and base image patches promptly
- Update Python dependencies for security fixes
- Rotate credentials after incidents or periodically

---

## 7. Capacity Planning

### Indicators to Scale

- CPU consistently > 70%
- Memory usage > 80%
- Query latency p95 > 10s
- Health check failures under load

### Scaling Actions

- **Vertical**: Increase ECS/VM size
- **Horizontal**: Add more UDS Agent instances behind load balancer
- **Database**: Scale ClickHouse (replicas, sharding) for read-heavy workloads

---

## 8. Contact and Escalation

- **On-call**: Define rotation for P1/P2
- **Documentation**: Keep runbooks in this guide or linked wiki
- **Post-mortems**: Store in shared drive with root cause and action items

---

## Quick Reference

| Need | Command / Endpoint |
|------|--------------------|
| Health | `curl http://localhost:8000/health` |
| Agent status | `curl http://localhost:8000/api/v1/uds/status` |
| Logs | `docker logs uds-agent` |
| Restart | `docker compose restart uds-agent` |
| ClickHouse ping | `curl http://CH_HOST:8123/ping` |
| Ollama models | `curl http://localhost:11434/api/tags` |
