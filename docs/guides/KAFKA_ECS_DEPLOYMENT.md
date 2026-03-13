# Kafka ECS Docker Deployment

Deploy Kafka on ECS using Docker Compose for the memory pipeline (Gateway → Kafka → Redis + ClickHouse).

---

## 1. Prerequisites

- Docker and Docker Compose installed on ECS instance
- For ECS: existing `ic-agent-services_default` network (create with `docker network create ic-agent-services_default` if needed)
- ~2 GB RAM and ~15 GB disk for Kafka

---

## 2. Deployment

### 2.1 Local (standalone)

```bash
cd IC-RAG-Agent
docker compose -f docker/docker-compose.kafka.yml up -d
```

Kafka listens on `localhost:9092`. Topic `rag_agent_message_event` is created automatically by `kafka-init`.

### 2.2 ECS (join existing network)

```bash
cd IC-RAG-Agent
docker compose -f docker/docker-compose.kafka.yml -f docker/docker-compose.kafka.ecs.yml up -d
```

Kafka joins `ic-agent-services_default`; other services (Gateway, UDS Agent) reach it at `kafka:9092`.

---

## 3. Verification

### 3.1 Broker health

```bash
docker exec ic-rag-agent-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

### 3.2 List topics

```bash
docker exec ic-rag-agent-kafka kafka-topics --list --bootstrap-server localhost:9092
```

Expected: `rag_agent_message_event` (and internal topics).

### 3.3 Manual topic creation (if needed)

```bash
docker exec ic-rag-agent-kafka kafka-topics --create \
  --topic rag_agent_message_event \
  --bootstrap-server localhost:9092 \
  --partitions 6 \
  --replication-factor 1 \
  --config retention.ms=604800000
```

---

## 4. Gateway configuration

When the Kafka producer is implemented, set:

| Variable | Value |
|----------|-------|
| `GATEWAY_MEMORY_KAFKA_ENABLED` | `true` |
| `GATEWAY_MEMORY_KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` (when in same Docker network) |
| `GATEWAY_MEMORY_KAFKA_TOPIC` | `rag_agent_message_event` |

---

## 5. Optional: Kafka UI

For debugging:

```bash
docker compose -f docker/docker-compose.kafka.yml --profile with-ui up -d
```

UI at `http://localhost:8080`.

---

## 6. Troubleshooting

| Issue | Action |
|-------|--------|
| Kafka not ready | Wait 45–60s for `start_period`; check `docker logs ic-rag-agent-kafka` |
| Topic missing | Run manual create command above; or `docker compose up kafka-init` |
| Network not found (ECS) | `docker network create ic-agent-services_default` |
| Disk full | Set `KAFKA_LOG_RETENTION_HOURS` (default 168); monitor disk |
| Port 9092 in use | Stop conflicting service or change port mapping |
| **403 Forbidden** from registry mirror | Docker Hub mirror (e.g. docker.1panel.live) may block pulls. Pre-pull image on a machine with Docker Hub access, then `docker save` and `docker load` on ECS. Or temporarily remove `registry-mirrors` from `/etc/docker/daemon.json`, restart Docker, pull, then restore mirrors. |

---

## 7. Stop

```bash
docker compose -f docker/docker-compose.kafka.yml down
# With ECS override:
docker compose -f docker/docker-compose.kafka.yml -f docker/docker-compose.kafka.ecs.yml down
```

Data persists in `kafka_data` volume. Add `-v` to remove volume.
