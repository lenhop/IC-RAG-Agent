# UDS Agent Operations Manual

**Version:** 1.0.0  
**Last Updated:** 2026-03-06

---

## System Overview

UDS Agent is a Business Intelligence system that answers natural language questions about Amazon seller data using ClickHouse, Redis caching, and LLM-powered agents.

**Architecture:** User → API (FastAPI) → UDS Agent (ReAct) → Tools → ClickHouse

**Performance:** <5s (simple), <10s (medium), <15s (complex) | 78.5% cache hit rate

---

## Production Configuration

### Access

| Item | Value |
|------|-------|
| SSH | `ssh len` |
| Application directory | `/opt/uds-agent` |
| UDS Agent port | 8001 (host) - avoids conflict with ChromaDB on 8000 |
| Health endpoint | `http://<ECS_IP>:8001/health` |

### Infrastructure (ECS Services)

| Service | Port | Purpose |
|---------|------|---------|
| ClickHouse | 8123 (HTTP), 9000 (native) | Analytics database |
| Redis | 6379 | Cache |
| ChromaDB | 8000 | Vector store (RAG) |
| UDS Agent | 8001 | Business Intelligence API |

**Docker Network:** UDS Agent uses `ic-agent-services_default` (external) to connect to ClickHouse and Redis

**Compose File:** `docker/docker-compose.ecs.yml`

### Environment Variables

| Variable | Value |
|----------|-------|
| CH_HOST | clickhouse |
| CH_PORT | 8123 |
| CH_USER | ic_agent |
| CH_PASSWORD | ic_agent_2026 |
| CH_DATABASE | ic_agent |
| REDIS_URL | redis://redis:6379/0 |
| UDS_LLM_PROVIDER | ollama |
| UDS_LLM_MODEL | qwen3:1.7b |

### Service URLs

| Service | URL |
|---------|-----|
| UDS API | http://<ECS_IP>:8001 |
| Swagger | http://<ECS_IP>:8001/docs |
| Prometheus | http://<ECS_IP>:9090 |
| Grafana | http://<ECS_IP>:3000 |

**Notes:**
- Ollama not included in ic-agent-services - add Ollama container or configure remote LLM
- Port 8001 used to avoid conflict with ChromaDB on 8000
- UDS API health endpoint: `GET /health` (SP-API uses `GET /api/v1/health`)

### Gateway downstream backends

The unified gateway (port 8000) dispatches to three backend URLs. Set these for correct routing:

| Variable | Default (in code) | Purpose |
|----------|-------------------|---------|
| RAG_API_URL | http://127.0.0.1:8002 | RAG Pipeline (general + Amazon docs + IC docs when enabled) |
| UDS_API_URL | http://127.0.0.1:8001 | UDS Agent (BI/analytics) |
| SP_API_URL | http://127.0.0.1:8003 | SP-API Agent (seller operations) |

- **General knowledge** is answered by the RAG pipeline in general mode. To use DeepSeek for general, set RAG’s `RAG_LLM_PROVIDER=deepseek` or `RAG_GENERAL_LLM_PROVIDER=deepseek` (RAG env, not gateway).
- **IC_DOCS_ENABLED:** When `false` (default), the IC docs route does not call RAG; the gateway returns a friendly message (“IC document retrieval is not ready yet...”). Set to `true` once Chroma is populated with IC documents.

---

## Access & Credentials

| Resource | Access | Notes |
|----------|--------|-------|
| **ECS Server** | `ssh len` | Production environment |
| **Application** | `/opt/uds-agent` | Main directory |
| **API Port** | 8001 | UDS Agent (8000 is ChromaDB) |
| **Grafana** | `http://<ECS_IP>:3000` | 4 dashboards |
| **Prometheus** | `http://<ECS_IP>:9090` | Metrics |
| **SLS Logs** | Alibaba Cloud Console | 5 log stores, 30-day retention |
| **GitHub** | IC-RAG-Agent repo | CI/CD pipeline |

---

## Deployment

### Build and Deploy

```bash
# SSH to ECS
ssh len

# Navigate to application
cd /opt/uds-agent

# Build (first time or after code changes)
docker compose -f docker/docker-compose.ecs.yml build

# Start services
docker compose -f docker/docker-compose.ecs.yml up -d

# Check status
docker compose -f docker/docker-compose.ecs.yml ps

# View logs
docker logs uds-agent -f

# Health check
curl http://localhost:8001/health
```

### Test Execution

```bash
# Set ECS_HOST for smoke tests (from ECS or local)
export ECS_HOST=http://<ECS_IP>:8001

# Run all tests
python scripts/run_all_tests.py

# Run smoke tests only
pytest tests/test_smoke.py -v
```

---

## Daily Operations

### Health Monitoring
```bash
# Check health endpoint
curl http://localhost:8001/health

# Expected: {"status":"healthy","database":"connected"}
```

### Service Health Endpoints (Local)

Use the endpoint that matches each service implementation:

| Service | Port | Health Endpoint |
|---------|------|-----------------|
| Gateway | 8000 | `GET /health` |
| UDS API | 8001 | `GET /health` |
| RAG API | 8002 | `GET /health` |
| SP-API API | 8003 | `GET /api/v1/health` |

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/api/v1/health
```

### View Logs
```bash
# Application logs
docker logs uds-agent --tail 100
docker logs uds-agent -f  # follow

# Search for errors
docker logs uds-agent | grep -i error

# Check circuit breaker
docker logs uds-agent | grep -i "circuit breaker"
```

### Check Status
```bash
# Service status
docker compose -f docker/docker-compose.ecs.yml ps

# API status
curl http://localhost:8001/api/v1/uds/status

# Quick status script
./bin/uds_ops.sh status
```

---

## Common Tasks

### Deploy Updates
```bash
# Via CI/CD: Push to main branch (automated)

# Manual deployment:
ssh len
cd /opt/uds-agent
docker compose -f docker/docker-compose.ecs.yml pull
docker compose -f docker/docker-compose.ecs.yml up -d
```

### Rollback
```bash
./bin/uds_ops.sh rollback <previous-version>
```

### Restart Services
```bash
# Restart UDS Agent
docker compose -f docker/docker-compose.ecs.yml restart uds-agent

# Restart all services
docker compose -f docker/docker-compose.ecs.yml restart
```

### Scale Resources
- **Vertical:** Resize ECS instance in Alibaba Cloud Console
- **Horizontal:** Add more UDS Agent instances behind load balancer
- **Database:** Scale ClickHouse (replicas, sharding)

---

## Troubleshooting

### Query Returns No Data
1. Check date range (data may be October 2025 only)
2. Verify tables: `GET /api/v1/uds/tables`
3. Check schema: `GET /api/v1/uds/tables/amz_order`
4. Try simpler query

### Query Timeout
1. Use streaming endpoint: `POST /api/v1/uds/query/stream`
2. Narrow date range
3. Increase `max_execution_time` in options
4. Check ClickHouse load

### Database Connection Failed
1. Verify ClickHouse: `curl http://CH_HOST:8123/ping`
2. Check credentials (CH_USER, CH_PASSWORD in .env)
3. Check network/firewall
4. Review ClickHouse logs

### LLM Errors
1. Verify Ollama: `curl http://localhost:11434/api/tags`
2. Ensure model pulled: `ollama pull qwen3:1.7b`
3. For remote API: check URL, key, rate limits

### Circuit Breaker Open
- **Cause:** Repeated failures (DB or LLM)
- **Action:** Wait for half-open window; fix underlying issue
- **Prevention:** Resolve root cause before retrying

### High Error Rate
1. Check logs: `docker logs uds-agent | grep -i error`
2. Verify DB and LLM connectivity
3. Check resource limits: `docker stats`
4. Review recent deployments
5. Check Grafana error dashboard

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Action |
|-------|-------------|---------------|--------|
| **P1** | Full outage (API down, DB down) | Immediate | Restart services, check DB/LLM, notify stakeholders |
| **P2** | Degraded (slow, partial failures) | Within 1 hour | Investigate logs, check metrics, consider rollback |
| **P3** | Minor (single query type failing) | Within 4 hours | Document, reproduce, fix in next release |

### Escalation
1. Check health and logs
2. Attempt restart if safe
3. Notify on-call team
4. Document incident
5. Post-mortem after resolution

### Communication
- **DingTalk:** Automated alerts
- **Email:** Notifications
- **SMS:** Critical alerts only

---

## Monitoring

### Grafana Dashboards
1. **uds-agent-overview** - Key metrics, request rate, error rate
2. **uds-agent-performance** - Response times, throughput, cache hit rate
3. **uds-agent-errors** - Error rates, error logs, circuit breaker
4. **uds-agent-infrastructure** - CPU, memory, disk, containers

### Key Metrics
- Request rate, error rate, response times
- Cache hit rate (target: >70%)
- Database query times
- LLM response times
- Resource utilization (CPU, memory)

### Alert Rules (12)
- **Critical:** Error rate >5%, service down, DB unavailable
- **Warning:** Error rate >1%, response time >10s, cache hit <50%

---

## Maintenance

### Regular Tasks

| Task | Frequency |
|------|-----------|
| Health check | Continuous (automated) |
| Log review | Daily |
| Dependency updates | Monthly |
| Backup verification | Weekly |
| Capacity review | Quarterly |

### Database Maintenance
- ClickHouse: OPTIMIZE TABLE, VACUUM as needed
- Regenerate `uds_statistics.json` if schema/data changes

### Security
- Apply OS and base image patches
- Update Python dependencies for security fixes
- Rotate credentials after incidents or periodically
- Renew SSL/TLS certificates before expiry

---

## Backup & Recovery

### Backup
- **ClickHouse:** Automated or manual (daily/weekly)
- **Configuration:** Version-controlled in Git
- **Docker images:** Tagged in ACR

### Restore
```bash
# Rollback to previous version
./bin/uds_ops.sh rollback <version>

# ClickHouse restore (follow ClickHouse backup docs)
# Configuration restore: redeploy from Git
```

### RTO/RPO
- **RTO (Recovery Time):** 2-3 minutes (rollback)
- **RPO (Recovery Point):** Depends on backup frequency

---

## Production Launch Checklist

### Pre-Launch
- [ ] All 113 test frameworks passing on ECS
- [ ] Documentation complete (5 guides)
- [ ] Docker images built and tested
- [ ] Monitoring configured (Prometheus, Grafana, CMS, SLS)
- [ ] Security scan passed
- [ ] Backup procedures tested
- [ ] Rollback procedures tested

### Launch Day
- [ ] Deploy to production
- [ ] Verify health checks passing
- [ ] Run smoke tests
- [ ] Monitor error rates and response times
- [ ] Verify monitoring dashboards showing data
- [ ] Announce launch to stakeholders

### Post-Launch (Week 1)
- [ ] Monitor production metrics daily
- [ ] Review error logs
- [ ] Collect user feedback
- [ ] Address critical issues
- [ ] Optimize performance if needed
- [ ] Conduct retrospective

---

## Quick Reference

| Action | Command |
|--------|---------|
| SSH to ECS | `ssh len` |
| Health check | `curl http://localhost:8001/health` |
| View logs | `docker logs uds-agent --tail 100` |
| Check status | `./bin/uds_ops.sh status` |
| Restart | `docker compose -f docker/docker-compose.ecs.yml restart uds-agent` |
| Deploy | `docker compose -f docker/docker-compose.ecs.yml pull && docker compose -f docker/docker-compose.ecs.yml up -d` |
| Rollback | `./bin/uds_ops.sh rollback <version>` |
| Run tests | `python scripts/run_all_tests.py` |

---

## Route LLM Configuration and Rollout

### Overview

Route LLM is an LLM-based query classifier that routes incoming queries to one of five workflows: `general`, `amazon_docs`, `ic_docs`, `sp_api`, `uds`. It runs alongside the existing rule-based heuristic router and falls back to heuristics when disabled, failing, or low-confidence.

**Module:** `src/gateway/route_llm.py`  
**Integration:** `src/gateway/router.py` → `route_workflow()`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_ROUTE_LLM_ENABLED` | `false` | Master switch. Set `true` to activate LLM routing. |
| `GATEWAY_ROUTE_LLM_BACKEND` | `ollama` | Default backend: `"ollama"` (local) or `"deepseek"` (remote). Overridden by per-request `route_backend` field. |
| `GATEWAY_ROUTE_LLM_CONF_THRESHOLD` | `0.7` | Minimum confidence to accept LLM result. Below this, gateway falls back to rule-based routing. |
| `GATEWAY_ROUTE_LLM_TIMEOUT` | `5` | Timeout in seconds for the LLM HTTP call. On timeout, falls back to heuristics. |
| `GATEWAY_ROUTE_LLM_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL. Use ECS URL (e.g. `http://$CH_HOST:11434`) when gateway runs locally against ECS Ollama. |
| `GATEWAY_ROUTE_LLM_OLLAMA_MODEL` | `qwen3:1.7b` | Ollama model name for routing classification. |
| `GATEWAY_ROUTE_LLM_DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name. Uses existing `DEEPSEEK_API_KEY`. |

### Example `.env` Snippets

**Local-only mode (Ollama):**
```bash
GATEWAY_ROUTE_LLM_ENABLED=true
GATEWAY_ROUTE_LLM_BACKEND=ollama
GATEWAY_ROUTE_LLM_CONF_THRESHOLD=0.7
GATEWAY_ROUTE_LLM_TIMEOUT=5
GATEWAY_ROUTE_LLM_OLLAMA_URL=http://${CH_HOST}:11434
GATEWAY_ROUTE_LLM_OLLAMA_MODEL=qwen3:1.7b
```

**Remote-only mode (DeepSeek):**
```bash
GATEWAY_ROUTE_LLM_ENABLED=true
GATEWAY_ROUTE_LLM_BACKEND=deepseek
GATEWAY_ROUTE_LLM_CONF_THRESHOLD=0.7
GATEWAY_ROUTE_LLM_TIMEOUT=8
GATEWAY_ROUTE_LLM_DEEPSEEK_MODEL=deepseek-chat
# DEEPSEEK_API_KEY must also be set
```

**Disabled mode (heuristic-only, default):**
```bash
GATEWAY_ROUTE_LLM_ENABLED=false
# All other GATEWAY_ROUTE_LLM_* variables are ignored
```

### Rollout Stages

#### Stage 1: Shadow Mode (Recommended First Step)

Route LLM is called and its decision is **logged**, but the heuristic result is still used for the actual response. This lets you evaluate LLM accuracy without any user-facing impact.

**How to enable:**
```bash
GATEWAY_ROUTE_LLM_ENABLED=true
GATEWAY_ROUTE_LLM_CONF_THRESHOLD=999   # Forces fallback every time
```

Setting the threshold impossibly high means the LLM result is always below threshold, so heuristics are used. But the LLM call still runs and its output is logged.

**What to monitor:**
- `logs/gateway_queries.jsonl` — look for `route_llm_workflow`, `route_llm_confidence`, `route_llm_backend` fields
- Compare `route_llm_workflow` (LLM decision) vs `workflow` (heuristic decision) to measure agreement rate
- Check `route_llm_confidence` distribution — most values should be above 0.7
- Watch for timeout or parse errors in application logs

**Duration:** Run for 1–3 days with representative traffic before proceeding.

#### Stage 2: Hybrid Mode (Production Default)

Route LLM result is used when confidence ≥ threshold; otherwise falls back to heuristics. This is the intended production mode.

**How to enable:**
```bash
GATEWAY_ROUTE_LLM_ENABLED=true
GATEWAY_ROUTE_LLM_CONF_THRESHOLD=0.7
GATEWAY_ROUTE_LLM_TIMEOUT=5
```

**What to monitor:**
- LLM usage rate: percentage of queries routed by LLM vs heuristic
- Fallback rate: percentage of queries where LLM confidence < threshold
- Error rate: percentage of LLM call failures (timeout, parse error, network)
- Routing distribution: breakdown of queries per workflow
- User-reported misroutes or unexpected answers

**Tuning:**
- If too many queries fall back → lower threshold (e.g., 0.6)
- If misroutes increase → raise threshold (e.g., 0.8)
- If timeouts are frequent → increase `GATEWAY_ROUTE_LLM_TIMEOUT` or switch backend

#### Stage 3: LLM-Primary Mode (Optional Future)

Rely mostly on LLM routing; heuristics only for LLM failures. Consider this after extended hybrid mode validation.

**How to enable:**
```bash
GATEWAY_ROUTE_LLM_ENABLED=true
GATEWAY_ROUTE_LLM_CONF_THRESHOLD=0.3   # Accept most LLM decisions
```

**Prerequisites before entering this stage:**
- Shadow mode data shows >90% agreement between LLM and heuristic
- Hybrid mode running stable for >1 week with no misroute incidents
- LLM error rate consistently <5%

### Operational Runbook

#### Quick Disable (Emergency)

If Route LLM is misbehaving (misroutes, high latency, errors), disable immediately:

```bash
# Option 1: Set env var and restart gateway
export GATEWAY_ROUTE_LLM_ENABLED=false
# Then restart the gateway process

# Option 2: If using .env file
# Edit .env: GATEWAY_ROUTE_LLM_ENABLED=false
# Restart gateway
docker compose -f docker/docker-compose.ecs.yml restart gateway
```

**Impact of disabling:** Gateway reverts to rule-based heuristic routing. No user-facing disruption. All queries continue to be served.

#### Verify Route LLM Is Working

**Step 1: Check gateway health**
```bash
curl http://localhost:8000/health
# Confirm gateway is healthy
```

**Step 2: Send test queries for each workflow**
```bash
# general
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "workflow": "auto"}'

# amazon_docs
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the FBA fee structure?", "workflow": "auto"}'

# ic_docs
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our internal return policy?", "workflow": "auto"}'

# sp_api
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Create a shipment for order 123", "workflow": "auto"}'

# uds
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Total sales in October 2025", "workflow": "auto"}'
```

**Step 3: Inspect logs**
```bash
# Check latest log entries
tail -5 logs/gateway_queries.jsonl | python -m json.tool

# Fields to inspect:
#   "workflow"              — final workflow used
#   "routing_confidence"    — confidence score
#   "route_llm_backend"    — which LLM backend was used (ollama/deepseek)
#   "route_source"         — "llm", "heuristic", or "manual"
```

**Step 4: Verify Ollama is reachable (if using local backend)**
```bash
curl http://localhost:11434/api/tags
# Should list available models including qwen3:1.7b

# Pull model if missing
ollama pull qwen3:1.7b
```

#### Known Failure Modes

| Failure | Behavior | Log Indicator |
|---------|----------|---------------|
| Ollama not running | Falls back to heuristic routing | `route_source: "heuristic"`, error in app logs |
| DeepSeek API key invalid | Falls back to heuristic routing | `error_code` in log, HTTP 401 in app logs |
| LLM returns malformed JSON | Falls back to heuristic routing | Parse error in app logs |
| LLM timeout (>5s) | Falls back to heuristic routing | `route_source: "heuristic"`, timeout in app logs |
| LLM confidence below threshold | Falls back to heuristic routing | `route_llm_confidence < 0.7`, `route_source: "heuristic"` |
| All backends fail | Heuristic routing used, gateway continues | No LLM fields in log entry |

**In all failure cases:** The gateway continues to serve requests using rule-based heuristics. No user-facing errors occur due to Route LLM failures.

#### Switching Backends

```bash
# Switch from Ollama to DeepSeek
export GATEWAY_ROUTE_LLM_BACKEND=deepseek
# Restart gateway

# Switch from DeepSeek to Ollama
export GATEWAY_ROUTE_LLM_BACKEND=ollama
# Restart gateway

# Per-request override (no restart needed)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FBA?", "workflow": "auto", "route_backend": "deepseek"}'
```

#### Threshold Tuning

```bash
# More conservative (fewer LLM-routed queries, more fallbacks)
export GATEWAY_ROUTE_LLM_CONF_THRESHOLD=0.85

# More aggressive (more LLM-routed queries, fewer fallbacks)
export GATEWAY_ROUTE_LLM_CONF_THRESHOLD=0.5

# Restart gateway after changing
```

**Guideline:** Start at 0.7. If fallback rate >30%, consider lowering. If misroute rate >5%, consider raising.

---

## Documentation Links

- [Query Rewriting Guide](guides/QUERY_REWRITING.md) - Query rewriting env vars, UI, and backend switching
- [User Guide](guides/UDS_USER_GUIDE.md) - How to use UDS Agent
- [Developer Guide](guides/UDS_DEVELOPER_GUIDE.md) - Architecture, development
- [API Reference](guides/UDS_API_REFERENCE.md) - API endpoints
- [Deployment Guide](guides/UDS_DEPLOYMENT_GUIDE.md) - Deployment instructions
- [Project Documentation](PROJECT.md) - Project summary, metrics, ADRs

---

**Operations Manual Version:** 1.1.0  
**Last Updated:** 2026-03-06  
**Status:** Production Ready
