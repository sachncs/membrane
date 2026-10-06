# syntax=docker/dockerfile:1
# Membrane runtime image (Python 3.14).
#
# The builder installs the exact locked dependency set (uv.lock) and the
# project itself, non-editable and byte-compiled, into /opt/venv. The
# runtime stage copies only that virtualenv onto a slim base without
# compilers, pip, or curl.
#
# Both base images are pinned by digest; Dependabot (docker ecosystem)
# proposes bumps.
ARG PYTHON_IMAGE=python:3.14-slim@sha256:c3e521df8b2b498a7a682e7e18676771cb80c6b75b8699af886b2d554ce40151
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711

FROM ${UV_IMAGE} AS uv


FROM ${PYTHON_IMAGE} AS builder

# Optional extras on top of `server`, space-separated; e.g.
# --build-arg EXTRAS=disagg adds grpcio for --grpc-port.
ARG EXTRAS=""
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.14 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /src
# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done) \
        --no-install-project
COPY README.md LICENSE ./
COPY membrane/ membrane/
# uv caches a local project's wheel until pyproject.toml changes, so
# rebuild the project itself from the copied source every time.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done) \
        --no-editable --reinstall-package membrane


FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="Membrane" \
      org.opencontainers.image.description="Global Contextual Memory Fabric for LLM inference" \
      org.opencontainers.image.source="https://github.com/sachncs/membrane" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    MEMBRANE_HOST=0.0.0.0 \
    MEMBRANE_PORT=8080 \
    MEMBRANE_DAEMON=true \
    MEMBRANE_LOG_FORMAT=json

# tini reaps zombies and forwards SIGTERM so `membrane serve` drains
# gracefully. Apply Debian security updates, and drop the base image's
# pip: the runtime never installs packages, and pip's vendored libraries
# are what scanners flag.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /usr/local/lib/python3*/site-packages/pip /usr/local/lib/python3*/site-packages/pip-* /usr/local/bin/pip*

# Non-root user with explicit UID so read-only filesystems can map it.
RUN groupadd -r --gid 1000 membrane \
 && useradd  -r --uid 1000 --gid 1000 --home-dir /app --shell /usr/sbin/nologin membrane \
 && install -d -o membrane -g membrane -m 0750 /var/lib/membrane

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER membrane

EXPOSE 8080
# Mount a volume here and set MEMBRANE_DATA_DIR=/var/lib/membrane to keep
# KV bytes across restarts (pair with MEMBRANE_REDIS_URL for metadata).
VOLUME ["/var/lib/membrane"]

# Uses the same MEMBRANE_* settings as the server, including mTLS.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-m", "membrane.healthcheck"]

# SIGTERM drains the node (MEMBRANE_DRAIN_TIMEOUT, default 30 s): give the
# container a longer stop timeout, e.g. `docker stop -t 40`.
STOPSIGNAL SIGTERM

# Configuration comes from MEMBRANE_* environment variables (see
# `membrane serve --help`). The server refuses to start on 0.0.0.0
# without authentication: mount a keyfile (mode 0600 or 0640) and set
# MEMBRANE_API_KEY_FILE, configure mTLS, or (development only) set
# MEMBRANE_ALLOW_UNAUTHENTICATED=true.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["membrane", "serve"]
