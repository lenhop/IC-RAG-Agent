# Alibaba Cloud Server Plan — Database & Memory Services

**Date:** 2026-03-05  
**Purpose:** Single ECS instance for ClickHouse + Redis + ChromaDB (Docker deployment)

---

## Services to Deploy

| Service | Role | Deployment |
|---------|------|------------|
| ClickHouse | Analytics/reporting DB (SP-API data, UDS) | Docker |
| Redis | Short-term memory (session cache, 24h TTL) | Docker |
| ChromaDB | Long-term memory (vector store) + document retrieval | Docker |

---

## Resource Requirements

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| ClickHouse | 2 cores | 4 GB | 50–100 GB SSD |
| Redis | 0.5 core | 1 GB | minimal |
| ChromaDB | 1 core | 2 GB | 20–50 GB SSD |
| OS + Docker overhead | 0.5 core | 1 GB | 20 GB |
| **Total** | **4 cores** | **8 GB** | **~150 GB SSD** |

---

## Alibaba Cloud ECS Recommendations

| Tier | Spec | Monthly Cost (approx) | Use Case |
|------|------|-----------------------|----------|
| Starter | 4 vCPU / 8 GB / 100 GB ESSD | ¥200–300/mo (~$28–42) | Dev/testing, 1–5 users |
| **Recommended** | **4 vCPU / 16 GB / 200 GB ESSD** | **¥400–600/mo (~$55–83)** | Small production, 10–20 users |
| Comfortable | 8 vCPU / 16 GB / 200 GB ESSD | ¥600–900/mo (~$83–125) | Heavy ClickHouse queries |

---

## Key Notes

- **ClickHouse** is the memory hog — loves RAM for query caching. 4 GB minimum, 8 GB comfortable.
- **Redis** barely uses anything for session data with 24h TTL.
- **ChromaDB** with 1536-dim embeddings — 2 GB handles tens of thousands of documents.
- Use **ESSD** (not regular cloud disk) — ClickHouse performance depends heavily on disk I/O.
- Pick a region close to you (e.g., cn-hangzhou or cn-shanghai) for low latency.
- Start with 4 vCPU / 16 GB, scale up later without reinstalling.

---

## Memory Architecture Context

| Memory Layer | Storage | TTL | Scope |
|-------------|---------|-----|-------|
| Short-term | Redis | 24h | Session-scoped |
| Long-term | ChromaDB | Persistent | Cross-session, semantic search |
| Documents | ChromaDB | Persistent | RAG document retrieval |
| Analytics | ClickHouse | Persistent | SP-API data, reporting |

---

## OS Choice: Ubuntu 22.04 LTS (Recommended)

**Decision:** Ubuntu 22.04 LTS over Alibaba Cloud Linux

| Factor | Alibaba Cloud Linux | Ubuntu 22.04 LTS |
|--------|---------------------|-------------------|
| Docker support | Good | Excellent — first-class, official docs |
| ClickHouse official support | Works but less tested | Officially supported & tested |
| Community / Stack Overflow | Small, mostly Chinese | Massive, global |
| Package ecosystem | yum/dnf (CentOS-based) | apt (Debian-based) — more familiar |
| Portability | Locked to Alibaba Cloud | Runs anywhere — easy to migrate |
| Local dev (macOS) compatibility | Different package manager | Closer to macOS toolchain |
| LTS support | Tied to Alibaba's schedule | Until April 2027 (security until 2032) |

**Reasons:**

1. ClickHouse, Redis, ChromaDB all have official Docker images tested on Ubuntu — more troubleshooting resources available.
2. Ubuntu works everywhere (Tencent Cloud, AWS, bare metal) — no vendor lock-in.
3. Team uses macOS locally — Ubuntu's `apt` ecosystem and file paths are closer to what we're used to.
4. Alibaba Cloud Linux's kernel-level optimization advantage is negligible for Docker containers running databases.

---

## Next Steps

- [ ] Purchase Alibaba Cloud ECS (recommended tier)
- [ ] Create `docker-compose.yml` for all three services
- [ ] Configure firewall rules (only allow internal access)
- [ ] Update `.env` with remote Redis/ChromaDB/ClickHouse endpoints
- [ ] Test connectivity from local dev machine
