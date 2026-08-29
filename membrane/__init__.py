"""Membrane — Global Contextual Memory Fabric for LLM inference.

The public API of Membrane. This module intentionally re-exports only
the durable domain concepts; implementation details live in their
own submodules and should be imported via the deep path.

Durable concepts (in import order, by domain):

* Memory objects — :class:`Fragment` and its family
  :class:`Prefix`, :class:`Segment`, :class:`Artifact`,
  :class:`Trace`, plus the discriminator :class:`FragmentKind` and
  the structural metadata :class:`Signature`.
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
* Transports — :class:`FastAPIServer`, :class:`GrpcServer`,
  :class:`HTTPServer`.
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
  ``Promotion``, ``Offload``, ``Isolation``, ``Selector``, ``Roles``,
  ``Predict``, ``Workload``) — import from their own modules.
* Resilience policies, metrics primitives, observability helpers,
  model analytical code, and CLI commands — import from their own
  modules.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.artifact import Artifact
from membrane.auth import Authenticator
from membrane.compute.anthropic import Anthropic
from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.compute.gpu import GPU
from membrane.compute.ollama import Ollama
from membrane.compute.openai import OpenAI
from membrane.compute.transformers import Transformers
from membrane.errors import (
    BackendError,
    CapacityError,
    ConfigError,
    Error,
    MigrationError,
    NetworkError,
    PersistenceError,
    SchemaError,
    TimeoutError,
)
from membrane.fragment import Fragment
from membrane.fragment_kind import FragmentKind
from membrane.index import Index
from membrane.logging import configure_logging
from membrane.node import Node
from membrane.origin import Origin
from membrane.persistence.base import PersistenceBackend
from membrane.persistence.cache import CachingPersistence
from membrane.persistence.memory import Memory
from membrane.persistence.redis import Redis
from membrane.prefiller import Prefiller
from membrane.prefix import Prefix
from membrane.reconstructor import Reconstructor
from membrane.replica import Replica
from membrane.ring import Ring
from membrane.segment import Segment
from membrane.server import Server
from membrane.shard import Shard
from membrane.signature import Signature
from membrane.trace import Trace
from membrane.transfer import TransferService
from membrane.transport.fastapi import FastAPIServer
from membrane.transport.grpc import GrpcServer
from membrane.transport.http import HTTPServer

__all__ = [
    # Memory objects
    "Fragment",
    "FragmentKind",
    "Signature",
    "Prefix",
    "Segment",
    "Artifact",
    "Trace",
    # Serving plane
    "Node",
    "Origin",
    "Replica",
    # Index
    "Index",
    # Placement
    "Ring",
    "Shard",
    # Reconstruction
    "Reconstructor",
    # Transfer
    "TransferService",
    # Persistence
    "PersistenceBackend",
    "Memory",
    "Redis",
    "CachingPersistence",
    # Compute
    "Backend",
    "CPU",
    "GPU",
    "Transformers",
    "OpenAI",
    "Anthropic",
    "Ollama",
    # Transports
    "FastAPIServer",
    "GrpcServer",
    "HTTPServer",
    # Composition
    "Server",
    # Auth
    "Authenticator",
    # Errors
    "Error",
    "BackendError",
    "CapacityError",
    "ConfigError",
    "MigrationError",
    "NetworkError",
    "PersistenceError",
    "SchemaError",
    "TimeoutError",
    # Logging
    "configure_logging",
    # Prefill
    "Prefiller",
]
