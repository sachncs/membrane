"""Membrane — Global Contextual Memory Fabric for LLM inference.

The public API of Membrane. This module intentionally re-exports only
the durable domain concepts; implementation details live in their
own submodules and should be imported via the deep path.

Durable concepts (in import order, by domain):

* Memory objects — :class:`Fragment` and its family
  :class:`Prefix`, :class:`Segment`, :class:`Artifact`,
  :class:`Trace`, plus the discriminator :class:`FragmentKind` and
  the structural metadata :class:`PayloadIdentity`.
* Serving plane — :class:`Node` with its :class:`Origin` and
  :class:`Replica` variants.
* Index — :class:`Index` aggregate (sub-indexes are deep-imported
  from ``membrane.exacts`` etc.).
* Placement — :class:`Ring` and :class:`Shard`.
* Reconstruction — :class:`Reconstructor`.
* Transfer — :class:`TransferService` (the unified in-process +
  remote-aware transfer plane).
* Persistence — :class:`PersistenceBackend` protocol with the
  :class:`Memory`, :class:`Redis`, and :class:`CachingPersistence`
  implementations.
* Compute — :class:`Backend` ABC plus the concrete CPU, GPU,
  Transformers, OpenAI, Anthropic, and Ollama backends.
* Transports — :class:`FastAPIServer`.
* Composition — :class:`Server` (the runnable entry point).
* Auth — :class:`Authenticator` protocol.
* Errors — :class:`Error` and its typed hierarchy.
* Logging — :func:`configure_logging`.

Implementation details that used to be re-exported here have been
moved to deep imports:

* Sub-indexes (``Exacts``, ``Semantics``, ``Tree``, ``Coaccess``,
  ``Weighted``, ``Graph``, ``Canonical``, ``KVCache``, ``Registry``,
  ``Chunk``) — import from ``membrane.exacts`` etc.
* Cluster plumbing (``Cluster``, ``Membership``, ``Peer``,
  ``Transfer`` legacy alias, ``GossipState``) — import from
  ``membrane.network.cluster`` etc.
* Decision / policy classes (``Economic``, ``Latency``, ``Joint``,
  ``Promotion``, ``Offload``, ``Isolation``, ``Tenant``,
  ``Selector``, ``Roles``, ``Predict``, ``Workload``) — import
  from ``membrane.analytical``.

v0.3.0 removed the following previously-exported research-only
names; deep-import paths continue to work but the names are
no longer at the package root:

* ``Adapter`` — import from ``membrane.adapter``.
* ``Prefiller`` — import from ``membrane.prefiller``.

These classes are not wired into the production serving plane;
they live in their own modules for research and are exercised
by tests/demos.
* Resilience policies, metrics primitives, observability helpers,
  model analytical code, and CLI commands — import from their own
  modules.
"""

import importlib
from typing import Any

#: Public name -> defining module. Exports load on first access (PEP 562),
#: so ``import membrane.model`` does not pull in the HTTP server stack.
LAZY_EXPORTS: dict[str, str] = {
    "Anthropic": "membrane.compute.anthropic",
    "Artifact": "membrane.artifact",
    "Authenticator": "membrane.auth",
    "Backend": "membrane.compute.base",
    "BackendError": "membrane.errors",
    "CPU": "membrane.compute.cpu",
    "CachingPersistence": "membrane.persistence.cache",
    "CapacityError": "membrane.errors",
    "ConfigError": "membrane.errors",
    "Error": "membrane.errors",
    "FastAPIServer": "membrane.transport.fastapi",
    "Fragment": "membrane.fragment",
    "FragmentKind": "membrane.fragment_kind",
    "GPU": "membrane.compute.gpu",
    "Index": "membrane.index",
    "Memory": "membrane.persistence.memory",
    "MigrationError": "membrane.errors",
    "NetworkError": "membrane.errors",
    "Node": "membrane.node",
    "Ollama": "membrane.compute.ollama",
    "OpenAI": "membrane.compute.openai",
    "Origin": "membrane.origin",
    "PayloadIdentity": "membrane.identity",
    "PersistenceBackend": "membrane.persistence.base",
    "PersistenceError": "membrane.errors",
    "Prefix": "membrane.prefix",
    "Reconstructor": "membrane.reconstructor",
    "Redis": "membrane.persistence.redis",
    "Replica": "membrane.replica",
    "Ring": "membrane.ring",
    "SchemaError": "membrane.errors",
    "Segment": "membrane.segment",
    "Server": "membrane.server",
    "Shard": "membrane.shard",
    "TimeoutError": "membrane.errors",
    "Trace": "membrane.trace",
    "TransferService": "membrane.transfer",
    "Transformers": "membrane.compute.transformers",
    "configure_logging": "membrane.logging",
}


def __getattr__(name: str) -> Any:
    """Load a public export on first access.

    Args:
        name: Attribute requested from the package.

    Returns:
        Any: The exported object.

    Raises:
        AttributeError: When ``name`` is not a public export.
    """
    module_name = LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'membrane' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's public names, including not-yet-loaded exports.

    Returns:
        list[str]: Sorted attribute names.
    """
    return sorted({*globals(), *LAZY_EXPORTS})


try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("membrane")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "CPU",
    "GPU",
    "Anthropic",
    "Artifact",
    # Auth
    "Authenticator",
    # Compute
    "Backend",
    "BackendError",
    "CachingPersistence",
    "CapacityError",
    "ConfigError",
    # Errors
    "Error",
    # Transports
    "FastAPIServer",
    # Memory objects
    "Fragment",
    "FragmentKind",
    # Index
    "Index",
    "Memory",
    "MigrationError",
    "NetworkError",
    # Serving plane
    "Node",
    "Ollama",
    "OpenAI",
    "Origin",
    # Persistence
    "PayloadIdentity",
    "PersistenceBackend",
    "PersistenceError",
    "Prefix",
    # Reconstruction
    "Reconstructor",
    "Redis",
    "Replica",
    # Placement
    "Ring",
    "SchemaError",
    "Segment",
    # Composition
    "Server",
    "Shard",
    "TimeoutError",
    "Trace",
    # Transfer
    "TransferService",
    "Transformers",
    "__version__",
    # Logging
    "configure_logging",
]
