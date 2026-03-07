# UDS Agent Deployment Guide

Guide for deploying the UDS Agent in local, Docker, and cloud environments.

---

## 1. Prerequisites

- Python 3.10+ (3.11 recommended)
- ClickHouse database (for UDS data)
- LLM service (Ollama or remote API)
- Optional: Redis (for caching)

---

## 2. Local Development Setup

### Step 1: Clone and Install

```bash
cd IC-RAG-Agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create `.env` in project root:

```bash
# ClickHouse (UDS data)
UDS_CH_HOST=localhost
UDS_CH_PORT=8123
UDS_CH_USER=ic_agent
UDS_CH_PASSWORD=ic_agent_2026
UDS_CH_DATABASE=ic_agent

# LLM (Ollama or remote)
UDS_LLM_PROVIDER=ollama
UDS_LLM_MODEL=qwen3:1.7b
```

### Step 3: Start ClickHouse (if local)

Using Docker:

```bash
docker run -d --name clickhouse \
  -p 8123:8123 -p 9000:9000 \
  clickhouse/clickhouse-server:24.3
```

Or use an existing ClickHouse instance and update `.env`.

### Step 4: Start Ollama (if using local LLM)

```bash
ollama serve
ollama pull qwen3:1.7b
```

### Step 5: Run the API

```bash
uvicorn src.uds.api:app --host 0.0.0.0 --port 8000 --reload
```

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/uds/tables
```

---

## 3. Docker Deployment

### Build Image

```bash
cd IC-RAG-Agent
docker build -f docker/Dockerfile -t uds-agent:latest .
```

### Run Standalone Container

```bash
docker run -d --name uds-agent \
  -p 8000:8000 \
  -e CH_HOST=host.docker.internal \
  -e CH_PORT=8123 \
  -e CH_USER=ic_agent \
  -e CH_PASSWORD=ic_agent_2026 \
  -e CH_DATABASE=ic_agent \
  -e UDS_LLM_PROVIDER=ollama \
  -e UDS_LLM_MODEL=qwen3:1.7b \
  uds-agent:latest
```

For Linux, use the host IP instead of `host.docker.internal` for ClickHouse.

### Docker Compose (Full Stack)

```bash
cd IC-RAG-Agent/docker
docker compose -f docker-compose.uds.yml up -d
```

This starts:

- UDS Agent (port 8000)
- ClickHouse (8123, 9000)
- Redis (6379)
- Optional: Ollama (profile `with-llm`), Nginx (profile `with-proxy`)

With Ollama:

```bash
docker compose -f docker-compose.uds.yml --profile with-llm up -d
```

### Health Check

The Dockerfile and compose use health checks. Ensure each service uses the correct endpoint:

- UDS API exposes: `GET /health`
- RAG API exposes: `GET /health`
- Gateway exposes: `GET /health`
- SP-API API exposes: `GET /api/v1/health`

---

## 4. Alibaba Cloud ECS Deployment

### Recommended Specs

| Tier | Spec | Use Case |
|------|------|----------|
| Starter | 4 vCPU / 8 GB / 100 GB ESSD | Dev, 1–5 users |
| Production | 4 vCPU / 16 GB / 200 GB ESSD | 10–20 users |
| Heavy | 8 vCPU / 16 GB / 200 GB ESSD | Heavy ClickHouse queries |

### Step 1: Create ECS Instance

1. Log in to Alibaba Cloud Console
2. Create ECS instance: Ubuntu 22.04 LTS, 4 vCPU / 16 GB
3. Attach ESSD disk (200 GB)
4. Configure security group: allow 22 (SSH), 80 (HTTP), 443 (HTTPS), 8000 (API)

### Step 2: Install Docker

```bash
ssh root@<ECS_IP>
apt update && apt install -y docker.io docker-compose
systemctl enable docker && systemctl start docker
```

### Step 3: Deploy Services

Option A: ClickHouse + Redis on same ECS, UDS Agent in Docker

```bash
# Create project directory
mkdir -p /opt/uds && cd /opt/uds

# Copy docker-compose.uds.yml and Dockerfile
# Update CH_HOST to localhost or 127.0.0.1 for same-host ClickHouse

docker compose -f docker-compose.uds.yml up -d
```

Option B: Use external ClickHouse (e.g. Alibaba Cloud ApsaraDB for ClickHouse)

```bash
# Set in .env or docker-compose environment:
CH_HOST=<clickhouse-host>
CH_PORT=8123
CH_USER=...
CH_PASSWORD=...
CH_DATABASE=ic_agent
```

### Step 4: Nginx Reverse Proxy (Optional)

Enable the `with-proxy` profile for Nginx:

```bash
docker compose -f docker-compose.uds.yml --profile with-proxy up -d
```

Configure Nginx to proxy `/api` to `http://uds-agent:8000`.

### Step 5: SSL (HTTPS)

Use Alibaba Cloud SSL certificates or Let's Encrypt. Configure Nginx with:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    location / {
        proxy_pass http://uds-agent:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Step 6: Auto-Start on Boot

```bash
# Add to crontab or systemd
@reboot cd /opt/uds && docker compose -f docker-compose.uds.yml up -d
```

---

## 5. AWS ECS Deployment (Alternative)

### Overview

- Use ECS Fargate or EC2-backed ECS
- Store images in ECR
- Use RDS or self-hosted ClickHouse
- Use Application Load Balancer for HTTPS

### Steps

1. Build and push image to ECR
2. Create ECS task definition (CPU: 1024, Memory: 2048)
3. Create ECS service with ALB
4. Configure environment variables (CH_HOST, CH_PORT, etc.)
5. Use Secrets Manager for passwords

---

## 6. Security Best Practices

### Network

- Restrict ClickHouse to internal network only
- Use firewall rules to limit API access
- Prefer VPN or private subnet for admin access

### Credentials

- Never commit `.env` or secrets
- Use environment variables or secrets manager
- Rotate passwords periodically

### API

- Add authentication (API key, OAuth2, JWT)
- Enable HTTPS in production
- Restrict CORS origins (avoid `*` in production)
- Add rate limiting (e.g. slowapi, nginx limit_req)

### Container

- Run as non-root user (Dockerfile already does this)
- Use read-only filesystem where possible
- Scan images for vulnerabilities

---

## 7. Scaling Guidelines

### Vertical Scaling

- Increase ECS/VM CPU and memory for more concurrent queries
- ClickHouse: 4 GB RAM minimum, 8 GB+ for heavy workloads
- Ollama: 4–8 GB RAM per model

### Horizontal Scaling

- Run multiple UDS Agent instances behind a load balancer
- Use shared Redis for cache (if implemented)
- ClickHouse can handle many concurrent read queries

### Performance Tuning

- Set `UDS_QUERY_TIMEOUT` appropriately (default 300s)
- Use streaming endpoint for long queries
- Enable query/SQL caching (UDSCache)
- Consider connection pooling for ClickHouse

---

## 8. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| UDS_CH_HOST / CH_HOST | 8.163.3.40 | ClickHouse host |
| UDS_CH_PORT / CH_PORT | 8123 | ClickHouse HTTP port |
| UDS_CH_USER / CH_USER | ic_agent | ClickHouse user |
| UDS_CH_PASSWORD / CH_PASSWORD | ic_agent_2026 | ClickHouse password |
| UDS_CH_DATABASE / CH_DATABASE | ic_agent | Database name |
| UDS_LLM_PROVIDER | ollama | LLM provider |
| UDS_LLM_MODEL | qwen3:1.7b | Model name |
| UDS_QUERY_TIMEOUT | 300 | Query timeout (seconds) |
| UDS_HOST | 0.0.0.0 | API bind host |
| UDS_PORT | 8000 | API bind port |

---

## 9. Troubleshooting Deployment

| Issue | Cause | Action |
|-------|-------|--------|
| Health check fails | Wrong health URL | Use `GET /health` |
| Database connection refused | Wrong host/port/firewall | Verify CH_* vars, network |
| LLM timeout | Ollama not running or slow | Start Ollama, check model |
| Container OOM | Insufficient memory | Increase memory limit |
| Slow queries | ClickHouse under-resourced | Add RAM, use ESSD |

---

## Related Documentation

- [OPERATIONS.md](../OPERATIONS.md) – Monitoring, maintenance, and incident response
- [UDS_DEVELOPER_GUIDE.md](UDS_DEVELOPER_GUIDE.md) – Architecture and configuration
- [ALIBABA_CLOUD_SERVER_SCHEME.md](../archive/ALIBABA_CLOUD_SERVER_SCHEME.md) – Alibaba Cloud resource planning
