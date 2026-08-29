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
