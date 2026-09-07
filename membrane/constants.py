"""Centralized constants for Membrane.

Magic numbers (default ports, default memory budgets, default cluster timing,
default cache TTLs) live here so that all defaults are auditable in one place
and CLI / server constructors cannot drift apart.
"""

from __future__ import annotations

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8080
DEFAULT_GRPC_PORT: int = 50051

DEFAULT_MAX_MEMORY: int = 1 << 30
DEFAULT_MAX_WORKERS: int = 4

DEFAULT_HEARTBEAT_INTERVAL: float = 2.0
DEFAULT_HEARTBEAT_TIMEOUT: float = 5.0
DEFAULT_GOSSIP_INTERVAL: float = 5.0
DEFAULT_FAILURE_SUSPECT_THRESHOLD: int = 2
DEFAULT_FAILURE_REMOVE_THRESHOLD: int = 4
DEFAULT_REPLICA_COUNT: int = 2
DEFAULT_GOSSIP_FANOUT: int = 2
DEFAULT_GOSSIP_MAX_FRAGMENT_ENTRIES: int = 50

DEFAULT_MAX_BODY_BYTES: int = 100 << 20
DEFAULT_TLS_PORT: int = 443

DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

DEFAULT_TTL: float = 3600.0
DEFAULT_TTL_SWEEP_INTERVAL: float = 60.0

DEFAULT_REDIS_URL: str = "redis://localhost:6379/0"
DEFAULT_OLLAMA_URL: str = "http://localhost:11434"

CAPACITY_PRESSURE_THRESHOLD: float = 0.90
EVICTION_REUSE_EPSILON: float = 0.01
