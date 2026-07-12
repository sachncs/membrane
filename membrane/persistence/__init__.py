"""Persistence backends for Membrane production serving.

This package groups the storage backends that can be plugged in
behind the :class:`~membrane.persistence.memory_backend.InMemoryBackend`
and :class:`~membrane.persistence.redis_backend.RedisBackend`
interfaces.

The two backends expose the same conceptual surface (set / get /
delete keyed by content hash) but differ in their durability
guarantees:

* ``InMemoryBackend`` — process-local, fast, and ephemeral. The
  default for tests and development.
* ``RedisBackend`` — durable (subject to Redis's own persistence
  configuration), shared across processes, and useful for
  multi-node deployments.

Adding a new backend is a matter of providing a class with the
same key/value interface and re-exporting it here.
"""

from membrane.persistence.memory_backend import InMemoryBackend
from membrane.persistence.redis_backend import RedisBackend

__all__ = ["InMemoryBackend", "RedisBackend"]
