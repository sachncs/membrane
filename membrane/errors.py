"""Custom exception hierarchy for Membrane.

This module defines the typed exception classes used across Membrane in place
of bare ``Exception`` catches. Each subclass maps to a specific failure domain
so callers can discriminate transient errors from configuration or programming
errors and respond appropriately.

Hierarchy::

    Error                         # base
    ├── ConfigError               # invalid configuration / setup
    ├── BackendError              # persistence or compute backend failure
    │   └── PersistenceError
    │       ├── ConnectionError   # cannot reach backend
    │       └── SchemaError       # incompatible serialization version
    ├── NetworkError              # peer / gossip / sync failure
    │   └── AuthError             # authentication or authorization failure
    ├── CapacityError             # store / cache full
    └── MigrationError            # shard migration failure
"""

from __future__ import annotations


class Error(Exception):
    """Base class for all Membrane-raised exceptions."""


class ConfigError(Error):
    """Raised when configuration is invalid or required values are missing."""


class BackendError(Error):
    """Raised when a backend (persistence or compute) fails to satisfy a request."""


class PersistenceError(BackendError):
    """Raised by a ``PersistenceBackend`` for non-recoverable failures."""


class ConnectionError(PersistenceError):
    """Raised when a backend cannot be reached after retries are exhausted."""


class SchemaError(BackendError):
    """Raised when serialized data cannot be deserialized because the version does not match."""


class CorruptPayloadError(BackendError):
    """Raised when payload bytes or a canonical frame fail integrity verification.

    Distinguishes storage corruption (truncated SHA-256 mismatch, byte
    flip, length past EOF) from config-level rejection
    (:class:`SchemaError`). Callers typically do **not** retry on
    corruption; the bytes are gone. Logged as a server-side error and
    surfaced through :class:`membrane.metrics`.
    """


class NetworkError(Error):
    """Raised when a network operation (peer call, gossip, sync) fails."""


class AuthError(NetworkError):
    """Raised when authentication or authorization fails."""


class CapacityError(Error):
    """Raised when a store or cache is full and cannot accept new entries."""


class TimeoutError(Error):
    """Raised when an operation exceeds its configured wall-clock budget."""


class MigrationError(Error):
    """Raised when shard migration between nodes fails irrecoverably."""


class TenantScopeError(AuthError):
    """Raised when an op touches a fragment in another tenant.

    The v3.0.0 release enforces per-tenant access on every
    store / retrieve / replicate op: a non-admin caller cannot
    write to or read from a fragment whose ``tenant_id`` does
    not match the caller's tenant. This is the typed
    authorization failure the :func:`op_store` /
    :func:`op_retrieve` paths raise on a cross-tenant access.
    """
