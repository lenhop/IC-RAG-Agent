# IC-RAG-Agent Docker Services

Redis, ClickHouse, ChromaDB for local dev and Alibaba Cloud ECS deployment.

## Quick Start

```bash
# From project root
docker compose -f docker/docker-compose.yml up -d

# Verify
redis-cli -h 127.0.0.1 -p 6379 ping
curl -s http://127.0.0.1:8123/ping
curl -s http://127.0.0.1:8001/api/v1/heartbeat
```

## Firewall Rules

| Port | Service   | Access              |
|------|-----------|---------------------|
| 6379 | Redis     | localhost (127.0.0.1) |
| 8123 | ClickHouse HTTP | localhost |
| 9000 | ClickHouse native | localhost |
| 8001 | ChromaDB  | localhost |

All services bind to 127.0.0.1 by default. For remote access (e.g. from app server in same VPC), change port bindings in docker-compose.yml from `127.0.0.1:PORT` to `PORT`.

## China: Docker Hub Mirror

If pulls from Docker Hub are slow, apply `daemon-china.json`:

```bash
sudo cp docker/daemon-china.json /etc/docker/daemon.json
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## ClickHouse Schema

After ClickHouse is up:

```bash
docker exec -i ic-rag-agent-clickhouse clickhouse-client --multiquery < scripts/uds/create_tables.sql
```

## .env

```
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8001
UDS_DB_HOST=127.0.0.1
UDS_DB_PORT=8123
```
