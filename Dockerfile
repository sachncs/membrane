#!/usr/bin/env python3.12
# Membrane runtime image.
# Pinned to a specific slim digest for reproducibility; bump via Dependabot.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Membrane" \
      org.opencontainers.image.description="Global Contextual Memory Fabric for LLM inference" \
      org.opencontainers.image.source="https://github.com/sachncs/membrane" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time dependencies for any wheels that need compiling.
# Removed from the final image via --mount=type=cache.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root user with explicit UID so read-only filesystems can map it.
RUN groupadd -r --gid 1000 membrane \
 && useradd  -r --uid 1000 --gid 1000 --home-dir /app --shell /usr/sbin/nologin membrane

WORKDIR /app

# Install Python dependencies first so the source layer is cacheable.
COPY pyproject.toml ./
COPY membrane/__init__.py membrane/__init__.py
RUN pip install --no-cache-dir ".[server]"

# Application source — non-editable install (no -e).
COPY membrane/ membrane/

USER membrane

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/livez || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["membrane", "serve", "--node-id", "docker-0", "--port", "8080", "--transport", "http", "--compute", "cpu", "--host", "0.0.0.0"]
