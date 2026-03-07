# Multi-stage Dockerfile for UDS Agent
# Target image size: <500MB

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r uds && useradd -r -g uds uds

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    UDS_HOST="0.0.0.0" \
    UDS_PORT="8000"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory and set permissions
RUN mkdir -p /app && chown -R uds:uds /app
WORKDIR /app

# Copy application code
COPY --chown=uds:uds src/ ./src/
COPY --chown=uds:uds scripts/ ./scripts/
COPY --chown=uds:uds docs/ ./docs/

# Switch to non-root user
USER uds

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "src.uds.api:app", "--host", "0.0.0.0", "--port", "8000"]