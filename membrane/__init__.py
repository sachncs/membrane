"""Membrane — Global Contextual Memory Fabric for LLM inference.

This package provides the foundational data model, indexing, graph layer,
caching, routing, and multi-tenant deduplication for Membrane.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.adapter import Adapter
from membrane.artifact import Artifact
from membrane.auth import (
    SCOPES,
    AuthBackendError,
    AuthContext,
    Authenticator,
    AuthRequest,
    require_scope,
)
from membrane.auth.apikey import APIKey, APIKeyAuthenticator, NoopAuthenticator
from membrane.auth.tls import TLSConfig
from membrane.canonical import Canonical, CanonicalRef
from membrane.chunks import Chunk
from membrane.clusters import SemanticCluster
from membrane.coaccess import Coaccess
from membrane.compute.anthropic import Anthropic
from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.compute.gpu import GPU
from membrane.compute.ollama import Ollama
from membrane.compute.openai import OpenAI
from membrane.compute.transformers import Transformers
from membrane.cost import CostModel
from membrane.delta import Delta
from membrane.density import density
from membrane.directory import Directory
from membrane.economic import Economic
from membrane.errors import (
    AuthError,
    BackendError,
    CapacityError,
    ConfigError,
    Error,
    MigrationError,
    NetworkError,
    PersistenceError,
    SchemaError,
)
from membrane.errors import (
    ConnectionError as PersistenceConnectionError,
)
from membrane.exacts import Exacts
from membrane.fragment import Fragment
from membrane.fragmenter import Fragmenter, FragmenterConfig
from membrane.graph import Graph
from membrane.index import Index
from membrane.isolation import Isolation, Tenant
from membrane.joint import Joint, PlacementDecision
from membrane.kv import KVCache
from membrane.latency import Latency
from membrane.logging import configure_logging, get_logger
from membrane.metrics import (
    ClusterMetrics,
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    NodeMetrics,
    PersistenceMetrics,
    TransportMetrics,
    metrics_summary,
)
from membrane.network.cluster import Cluster, PeerInfo
from membrane.network.config import ClusterConfig
from membrane.network.gossip import GossipState, PeerEndpoint
from membrane.network.peer import Peer as PeerClient
from membrane.network.strategy import (
    EagerMigrator,
    FailureDetector,
    Migrator,
    QuorumDetector,
    RateLimitedMigrator,
    ThresholdDetector,
)
from membrane.network.transfer import Transfer as RemoteTransfer
from membrane.node import Node, Stats
from membrane.offload import Offload, OffloadConfig, OffloadResult
from membrane.origin import Origin
from membrane.persistence.base import PersistenceBackend
from membrane.persistence.cache import CachingPersistence
from membrane.persistence.memory import Memory
from membrane.persistence.redis import Redis
from membrane.policy import Promotion, PromotionConfig, PromotionResult
from membrane.predict import Predict
from membrane.prefill_async import PrefillAsync
from membrane.prefill_remote import PrefillRemote
from membrane.prefix import Prefix
from membrane.reconstructor import Reconstructor, ReconstructorResult
from membrane.registry import Registry
from membrane.replica import Replica
from membrane.resilience import (
    BulkheadPolicy,
    CircuitBreakerPolicy,
    ResiliencePolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from membrane.ring import Ring
from membrane.roles import NodeRole, Roles, SystemState
from membrane.segment import Segment
from membrane.selector import Selector, SelectorConfig
from membrane.semantics import Semantics
from membrane.semhash import compute_semantic_hash, semantic_distance
from membrane.server import Server, ServerDiagnostics, ServerEvent
from membrane.sessions import Session, Sessions
from membrane.shard import Shard
from membrane.signature import Signature
from membrane.store import Store, StoreMetrics
from membrane.supernode import Supernode
from membrane.sync import DeltaSync, SyncPlan, SyncResult
from membrane.telemetry import Telemetry, telemetry
from membrane.trace import Trace
from membrane.tracker import LRUTracker
from membrane.transfer import TransferService
from membrane.transport.fastapi import FastAPIServer
from membrane.transport.grpc import GrpcServer
from membrane.transport.http import HTTPServer
from membrane.transport.http import StdlibServer as StdlibServerTransport
from membrane.tree import Tree
from membrane.versions import VersionEntry, Versions
from membrane.weighted import Weighted
from membrane.workload import Workload

__all__ = [
    "configure_logging",
    "get_logger",
    # Errors
    "AuthError",
    "BackendError",
    "CapacityError",
    "ConfigError",
    "Error",
    "MigrationError",
    "NetworkError",
    "PersistenceConnectionError",
    "PersistenceError",
    "SchemaError",
    # Authentication
    "APIKey",
    "APIKeyAuthenticator",
    "AuthBackendError",
    "AuthContext",
    "AuthRequest",
    "Authenticator",
    "NoopAuthenticator",
    "SCOPES",
    "TLSConfig",
    "require_scope",
    # Resilience
    "BulkheadPolicy",
    "CircuitBreakerPolicy",
    "ResiliencePolicy",
    "RetryPolicy",
    "TimeoutPolicy",
    # Metrics
    "ClusterMetrics",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsCollector",
    "NodeMetrics",
    "PersistenceMetrics",
    "TransportMetrics",
    "metrics_summary",
    # Core data
    "Artifact",
    "Chunk",
    "Delta",
    "Fragment",
    "Prefix",
    "Session",
    "Signature",
    "StoreMetrics",
    "Trace",
    "VersionEntry",
    # Storage / indexes
    "Canonical",
    "CanonicalRef",
    "Coaccess",
    "CostModel",
    "Exacts",
    "Graph",
    "Index",
    "Semantics",
    "Store",
    "Tree",
    # Cluster / network
    "Cluster",
    "ClusterConfig",
    "Directory",
    "GossipState",
    "Memory",
    "Node",
    "PersistenceBackend",
    "Origin",
    "PeerClient",
    "PeerEndpoint",
    "PeerInfo",
    "Redis",
    "Registry",
    "RemoteTransfer",
    "Replica",
    "Ring",
    "Shard",
    "Stats",
    "Supernode",
    "Sync",
    "SyncPlan",
    "SyncResult",
    "DeltaSync",
    "Telemetry",
    "Transfer",
    "TransferService",
    "Weighted",
    # Compute
    "Backend",
    "CachingPersistence",
    "CPU",
    "GPU",
    "Ollama",
    "OpenAI",
    # Cluster strategies
    "EagerMigrator",
    "FailureDetector",
    "Migrator",
    "QuorumDetector",
    "RateLimitedMigrator",
    "ThresholdDetector",
    "Anthropic",
    "Transformers",
    # Decision classes
    "Economic",
    "Isolation",
    "Joint",
    "Latency",
    "Offload",
    "OffloadConfig",
    "OffloadResult",
    "PlacementDecision",
    "Predict",
    "Promotion",
    "PromotionConfig",
    "PromotionResult",
    "Roles",
    "Selector",
    "SelectorConfig",
    "SystemState",
    "Tenant",
    "Workload",
    # Engines
    "Adapter",
    "Chunk",
    "Fragmenter",
    "FragmenterConfig",
    "KVCache",
    "LRUTracker",
    "PrefillAsync",
    "PrefillRemote",
    "Reconstructor",
    "ReconstructorResult",
    "Replicator",
    "Segment",
    "Sessions",
    "Versions",
    # Decisions helpers
    "density",
    "telemetry",
    "replicate",
    "compute_semantic_hash",
    "semantic_distance",
    "NodeRole",
    # Server
    "Server",
    "ServerDiagnostics",
    "ServerEvent",
    # Transports
    "FastAPIServer",
    "GrpcServer",
    "HTTPServer",
    "StdlibServerTransport",
    # Other
    "Clusters",
    "SemanticCluster",
]
