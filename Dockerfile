# syntax=docker/dockerfile:1
# Membrane runtime image.
#
# Two stages: the builder compiles a wheel from the full source tree and
# installs it with its dependencies into a virtualenv; the runtime stage
# copies only that virtualenv. Installing a wheel (not the source tree)
# is what makes `membrane` importable from any working directory.
#
# The base image is pinned by digest for reproducibility; Dependabot
# (docker ecosystem) proposes digest bumps.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY membrane/ membrane/
RUN pip install --upgrade pip build \
 && python -m build --wheel --outdir /dist \
 && pip install "$(ls /dist/membrane-*.whl)[server]" \
 && pip uninstall -y build


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
    MEMBRANE_DAEMON=true

# tini reaps zombies and forwards SIGTERM so `membrane serve` can shut
# down gracefully; curl backs the HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS "http://localhost:${MEMBRANE_PORT}/livez" || exit 1

# Configuration comes from MEMBRANE_* environment variables (see
# `membrane serve --help`). The server refuses to start on 0.0.0.0
# without authentication: mount a keyfile and set
# MEMBRANE_API_KEY_FILE, configure mTLS, or (development only) set
# MEMBRANE_ALLOW_UNAUTHENTICATED=true.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["membrane", "serve"]
