FROM python:3.13-slim

LABEL org.opencontainers.image.title="Membrane"
LABEL org.opencontainers.image.description="Global Contextual Memory Fabric for LLM inference"
LABEL org.opencontainers.image.source="https://github.com/your-org/membrane"

WORKDIR /app

# Install tini for proper signal handling and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r membrane && useradd -r -g membrane membrane

# Copy package source
COPY membrane/ ./membrane/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml ./
COPY setup.sh cleanup.sh ./
COPY README.md ./
COPY docs/DEPLOY.md ./docs/

# Install in production mode (includes typer + rich + fastapi + uvicorn + grpc)
RUN pip install --no-cache-dir -e ".[server]"

# Switch to non-root user
USER membrane

# Expose HTTP and gRPC ports
EXPOSE 8080
EXPOSE 50051

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/heartbeat || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: run the CLI serve command
CMD ["membrane", "serve", "--node-id", "docker-0", "--port", "8080", "--transport", "http", "--compute", "cpu", "--host", "0.0.0.0"]
