# UDS Agent Docker Deployment

This directory contains the Docker configuration and scripts for deploying the UDS Agent in a containerized environment.

## Architecture

The UDS Agent deployment consists of the following services:

- **uds-agent**: Main UDS Agent application (FastAPI)
- **clickhouse**: ClickHouse database for analytics data
- **redis**: Redis cache for performance optimization
- **ollama**: LLM service (optional, for local AI models)
- **nginx**: Reverse proxy (optional, for production deployment)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- At least 10GB free disk space

### Basic Deployment

1. **Build the containers:**
   ```bash
   ./docker/build.sh
   ```

2. **Start the services:**
   ```bash
   ./docker/run.sh
   ```

3. **Check the status:**
   ```bash
   ./docker/run.sh --status
   ```

4. **Test the deployment:**
   ```python
   python tests/test_docker.py
   ```

5. **Stop the services:**
   ```bash
   ./docker/stop.sh
   ```

### Advanced Deployment

#### With LLM Support

```bash
./docker/run.sh --profile with-llm
```

This starts the Ollama service for local AI model inference.

#### With Reverse Proxy

```bash
./docker/run.sh --profile with-proxy
```

This starts the Nginx reverse proxy for production deployment.

#### Combined Setup

```bash
./docker/run.sh --profile with-llm --profile with-proxy --wait
```

## Configuration

### Environment Variables

The UDS Agent can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `UDS_HOST` | `0.0.0.0` | API server host |
| `UDS_PORT` | `8000` | API server port |
| `CH_HOST` | `clickhouse` | ClickHouse host |
| `CH_PORT` | `9000` | ClickHouse port |
| `CH_USER` | `default` | ClickHouse username |
| `CH_PASSWORD` | `` | ClickHouse password |
| `CH_DATABASE` | `uds` | ClickHouse database |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `OLLAMA_HOST` | `0.0.0.0` | Ollama host |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Default LLM model |

### Docker Compose Profiles

- `with-llm`: Includes Ollama service for local AI inference
- `with-proxy`: Includes Nginx reverse proxy for production

## API Endpoints

Once deployed, the UDS Agent API is available at:

- **API Base**: `http://localhost:8000`
- **Health Check**: `http://localhost:8000/api/v1/health`
- **API Info**: `http://localhost:8000/api/v1/info`
- **Query**: `http://localhost:8000/api/v1/query` (POST)

With reverse proxy:
- **API Base**: `http://localhost`
- **Health Check**: `http://localhost/health`

## Database Setup

The ClickHouse database is automatically initialized with the schema defined in `init-clickhouse.sql`. This includes:

- Tables for Amazon data (inventory, orders, fees, etc.)
- Performance indexes for common query patterns
- Default user permissions

## Monitoring and Troubleshooting

### Health Checks

All services include health checks that can be monitored with:

```bash
docker-compose -f docker/docker-compose.uds.yml ps
```

### Logs

View service logs:

```bash
# All services
docker-compose -f docker/docker-compose.uds.yml logs

# Specific service
docker-compose -f docker/docker-compose.uds.yml logs uds-agent

# Follow logs
docker-compose -f docker/docker-compose.uds.yml logs -f
```

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 8123, 9000, 6379, 11434, 80 are available
2. **Memory issues**: Ensure at least 8GB RAM is available
3. **Database connection**: Wait for ClickHouse to fully initialize (can take 30-60 seconds)
4. **LLM models**: First run with Ollama may take time to download models

### Performance Tuning

- **Memory limits**: Adjust in `docker-compose.uds.yml` based on your system
- **CPU limits**: Modify CPU reservations for optimal performance
- **Database optimization**: Indexes are pre-configured for common queries
- **Cache size**: Redis memory can be limited in the compose file

## Development

### Local Development with Docker

For development, you can mount the source code:

```yaml
# In docker-compose.uds.yml, modify uds-agent service:
volumes:
  - ../src:/app/src:ro
  - ../requirements.txt:/app/requirements.txt:ro
```

### Testing

Run the Docker integration tests:

```bash
# With services running
python tests/test_docker.py

# Wait for services to be ready
python tests/test_docker.py --wait 120

# Test different URL
python tests/test_docker.py --url http://localhost:8000
```

### Building Custom Images

```bash
# Build specific image
./docker/build.sh --image

# Build with custom tag
docker build -f docker/Dockerfile -t uds-agent:custom .

# Build with no cache
docker build --no-cache -f docker/Dockerfile -t uds-agent:latest .
```

## Security Considerations

### Production Deployment

1. **Change default passwords**: Update ClickHouse and Redis passwords
2. **Use secrets**: Store sensitive data in Docker secrets or environment files
3. **Network isolation**: Use internal networks for service communication
4. **Resource limits**: Set appropriate CPU and memory limits
5. **SSL/TLS**: Configure SSL certificates for Nginx
6. **Firewall**: Restrict access to necessary ports only

### Security Headers

The Nginx configuration includes security headers:
- `X-Frame-Options`: Prevents clickjacking
- `X-Content-Type-Options`: Prevents MIME type sniffing
- `X-XSS-Protection`: Enables XSS filtering
- `Content-Security-Policy`: Restricts resource loading

## Backup and Recovery

### Database Backup

```bash
# Backup ClickHouse data
docker exec uds-clickhouse clickhouse-client --query="BACKUP DATABASE uds TO Disk('backups', 'uds_backup')"

# Copy backup to host
docker cp uds-clickhouse:/var/lib/clickhouse/data/backups/ ./backups/
```

### Cache Backup

Redis data is persisted in the `redis_data` volume. To backup:

```bash
docker run --rm -v uds-redis_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz -C /data .
```

## File Structure

```
docker/
├── Dockerfile              # Multi-stage build for UDS Agent
├── docker-compose.uds.yml  # Service orchestration
├── .dockerignore          # Files to exclude from build context
├── init-clickhouse.sql    # Database initialization
├── nginx.conf            # Reverse proxy configuration
├── build.sh              # Build script
├── run.sh                # Run script
└── stop.sh               # Stop script
```

## Support

For issues and questions:

1. Check the logs: `docker-compose -f docker/docker-compose.uds.yml logs`
2. Run diagnostics: `python tests/test_docker.py`
3. Review configuration files
4. Check system resources (memory, disk space)

## License

This Docker deployment configuration is part of the IC-RAG-Agent project.
