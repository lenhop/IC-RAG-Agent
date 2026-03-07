# Architecture Decision Records (ADRs)

**Project:** UDS Agent  
**Version:** 1.0.0  
**Last Updated:** 2026-03-06

---

## ADR-001: Multi-Agent Architecture

### Decision

Use a pipeline of specialized components: Intent Classifier -> Task Planner -> UDS Agent (ReAct) -> Result Formatter.

### Rationale

- **Separation of concerns:** Each component has a single responsibility
- **Modularity:** Components can be tested and replaced independently
- **Testability:** Easier to mock and unit test each stage
- **Extensibility:** New intents or tools can be added without changing the core agent

### Alternatives Considered

- **Single monolithic agent:** Simpler but harder to maintain and extend
- **Chain-of-Thought only:** Less structured, no explicit task decomposition

### Consequences

- Better maintainability and easier testing
- More complex orchestration and more components to deploy
- Slightly higher latency due to multiple LLM calls (intent, plan, ReAct)

---

## ADR-002: ReAct Loop for Agent

### Decision

Use the ReAct (Reasoning + Acting) pattern for the UDS Agent.

### Rationale

- **Proven pattern:** Widely used for LLM-based agents
- **Balance:** Good trade-off between reasoning and action
- **Explainability:** Thought-Action-Observation loop is interpretable
- **Tool use:** Natural fit for tool selection and execution

### Alternatives Considered

- **Chain-of-Thought:** Less structured, no explicit tool invocation
- **Plan-and-Execute:** More upfront planning, less adaptive
- **ReWOO:** More planning overhead, less flexible

### Consequences

- Better explainability and debuggability
- More LLM calls per query (higher latency and cost)
- Requires careful prompt engineering for tool selection

---

## ADR-003: ClickHouse for Analytics

### Decision

Use ClickHouse as the primary database for UDS analytics data.

### Rationale

- **OLAP optimized:** Columnar storage, fast aggregations
- **Scale:** Handles 40M+ rows efficiently
- **Query performance:** Sub-second for typical analytical queries
- **Ecosystem:** Good Python client (clickhouse-connect), SQL support

### Alternatives Considered

- **PostgreSQL:** General-purpose, less optimized for analytics
- **MySQL:** Similar limitations for OLAP
- **MongoDB:** Document model less suited for analytical queries

### Consequences

- Excellent query performance for aggregations and time-series
- Limited transaction support (acceptable for analytics)
- Requires ClickHouse-specific SQL in some cases

---

## ADR-004: Redis for Caching

### Decision

Use Redis for query result, intent, and schema caching.

### Rationale

- **Fast:** In-memory, sub-millisecond retrieval
- **Simple:** Key-value store, easy to integrate
- **TTL support:** Built-in expiration for cache invalidation
- **Production-ready:** Widely used, well-supported

### Alternatives Considered

- **Memcached:** Simpler but fewer features (no persistence options)
- **In-process cache:** No cross-process sharing, lost on restart
- **No cache:** Simpler but 68% slower for repeated queries

### Consequences

- 78.5% cache hit rate achieved
- 68% performance improvement for cached operations
- Additional dependency (Redis server)
- Requires cache invalidation strategy for schema changes

---

## ADR-005: Docker Containerization

### Decision

Use Docker for containerization of the UDS Agent.

### Rationale

- **Industry standard:** Consistent environments, easy deployment
- **Portability:** Runs anywhere (local, ECS, cloud)
- **Reproducibility:** Same image across dev/staging/prod
- **Isolation:** Dependencies and runtime isolated

### Alternatives Considered

- **Native deployment:** Simpler but environment drift risk
- **VMs:** Heavier, slower startup
- **Serverless:** Less control, cold start latency

### Consequences

- Consistent deployments (~480MB image)
- Easier scaling (container orchestration)
- Container overhead (minimal for this workload)
- Requires Docker expertise for operations

---

## ADR-006: Alibaba Cloud ECS Deployment

### Decision

Deploy to Alibaba Cloud ECS (Elastic Compute Service).

### Rationale

- **User's existing infrastructure:** Aligns with current cloud usage
- **Cost-effective:** Competitive pricing for China region
- **China performance:** Low latency for China-based users
- **Integration:** Easier integration with Alibaba Cloud CMS, SLS

### Alternatives Considered

- **AWS ECS:** Strong ecosystem but higher latency in China
- **Azure:** Similar considerations
- **GCP:** Less common in China

### Consequences

- Good China performance and cost
- Alibaba Cloud service integration (CMS, SLS)
- Platform-specific tooling (e.g., Aliyun CLI)

---

## ADR-007: Prometheus + Grafana Monitoring

### Decision

Use Prometheus for metrics collection and Grafana for dashboards.

### Rationale

- **Industry standard:** Widely adopted, rich ecosystem
- **Open source:** No vendor lock-in
- **Flexibility:** Custom metrics, alert rules, dashboards
- **Integration:** Works with Alibaba Cloud CMS for alerts

### Alternatives Considered

- **Alibaba Cloud CMS only:** Simpler but less flexible
- **Datadog:** More features but cost and vendor lock-in
- **New Relic:** Similar trade-offs

### Consequences

- Full control over metrics and dashboards
- Self-hosted overhead (Prometheus, Grafana)
- 4 dashboards, 12 alert rules, 50+ metrics
- Can be extended with Alibaba Cloud SLS for logs

---

## ADR-008: FastAPI for REST API

### Decision

Use FastAPI for the REST API layer.

### Rationale

- **Modern Python framework:** Async support, type hints
- **Performance:** High throughput, low latency
- **Developer experience:** Auto-generated OpenAPI/Swagger docs
- **Validation:** Pydantic integration for request/response schemas

### Alternatives Considered

- **Flask:** Simpler but less async support, no built-in validation
- **Django REST Framework:** Heavier, more opinionated
- **Starlette:** Lower-level, more manual work

### Consequences

- High performance and good async support
- Excellent developer experience (Swagger, ReDoc)
- Async complexity for some operations
- 11 endpoints implemented with clean schemas

---

## ADR Index

| ADR | Title | Status |
|-----|-------|--------|
| ADR-001 | Multi-Agent Architecture | Accepted |
| ADR-002 | ReAct Loop for Agent | Accepted |
| ADR-003 | ClickHouse for Analytics | Accepted |
| ADR-004 | Redis for Caching | Accepted |
| ADR-005 | Docker Containerization | Accepted |
| ADR-006 | Alibaba Cloud ECS Deployment | Accepted |
| ADR-007 | Prometheus + Grafana Monitoring | Accepted |
| ADR-008 | FastAPI for REST API | Accepted |
