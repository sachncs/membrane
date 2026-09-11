"""Shared HTTP operation logic.

This module holds the *business* logic that backs each Membrane HTTP
endpoint. Both the stdlib :mod:`membrane.transport.routes` module
and the FastAPI binding :mod:`membrane.transport.routes_fastapi`
delegate to these functions so the actual store / retrieve / sync
logic lives in exactly one place.

Each function takes plain domain objects (``Node``,
``TransferService``, ``Cluster``, ``Backend``) and returns either
a JSON-ready dict (success) or a tuple ``(status_code, body)``
that the transport layer maps onto its native response type.

Thread safety:
    The operations are stateless and forward to the domain objects,
    which own their own concurrency.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from membrane.auth import AuthContext
from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.errors import TenantScopeError
from membrane.metrics import ClusterMetrics, MetricsCollector
from membrane.network.cluster import Cluster
from membrane.network.peer import JsonDict, Peer
from membrane.node import Node
from membrane.serialization import from_dict, to_dict

logger = logging.getLogger(__name__)


MAX_BODY_BYTES: int = 100 << 20
"""Maximum allowed request body size in bytes (100 MiB)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(status: int, message: str) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` error tuple."""
    return status, cast(JsonDict, {"error": message})


def _ok(body: Any) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` success tuple.

    Accepts any JSON-serializable mapping; the helper widens to
    ``JsonDict`` so deeply-typed nested dicts (``dict[str, int]``,
    ``list[dict[str, Any]]``, etc.) flow through without an
    explicit cast at every builder site.
    """
    return 200, cast(JsonDict, body)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def op_heartbeat(
    node: Node | None,
    cluster: Cluster | None = None,
    headers: dict[str, str] | None = None,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``GET /heartbeat`` — node health and load snapshot.

    Always returns 200 with the body indicating status, mirroring
    the existing contract that the heartbeat is informational, not a
    strict liveness check (``/livez`` is the dedicated liveness
    probe).

    When ``cluster`` is supplied, an inbound ``X-Local-Peer-CN``
    header (sent by the peer's :class:`~membrane.network.peer.Peer`
    heartbeat client) is captured into the
    :class:`~membrane.network.membership.PeerInfo` record so the
    cluster has a verified identity for every live peer. Missing
    headers on an mTLS-required cluster result in 401 (the
    FastAPI route is expected to have already enforced that).
    """
    if node is None:
        return _ok({"error": "no node"})
    stats = node.get_stats()
    if cluster is not None and headers is not None:
        cn = headers.get("x-local-peer-cn") or headers.get("X-Local-Peer-CN")
        if cn:
            cluster.membership.record_peer_cn(node.node_id, cn)
    return _ok(
        {
            "node_id": node.node_id,
            "load": node.heartbeat(),
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "healthy": True,
            "attributes": node.attributes.to_dict(),
        }
    )


def op_metrics(
    node: Node | None,
    metrics_registry: MetricsCollector | None = None,
) -> tuple[int, JsonDict | tuple[str, dict[str, str]]]:
    """``GET /metrics`` — Prometheus exposition or legacy JSON fallback.

    Returns ``(200, (text, headers))`` when a Prometheus
    registry is configured, ``(200, json_dict)`` when falling
    back to the node snapshot. The transport layer dispatches
    on the body type.
    """
    if metrics_registry is not None:
        return 200, (
            metrics_registry.render(),
            {"media_type": "text/plain; version=0.0.4"},
        )
    if node is None:
        return _ok({"error": "no node"})
    stats = node.get_stats()
    return _ok(
        {
            "node_id": node.node_id,
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "load": node.heartbeat(),
        }
    )


def op_inventory(node: Node | None, auth_context: AuthContext | None = None) -> tuple[int, JsonDict]:
    """``GET /inventory`` — node's inventory digest."""
    if node is None:
        return _ok({"node_id": "", "digest": {}})
    digest = {h: frag.version_id for h, frag in node.fragments.items()}
    return _ok({"node_id": node.node_id, "digest": digest})


def op_peers(cluster: Cluster | None, auth_context: AuthContext | None = None) -> tuple[int, JsonDict]:
    """``GET /peers`` — cluster membership view."""
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    return _ok({"peers": cluster.membership.to_json()})


def op_retrieve(
    node: Node | None,
    content_hash: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``GET /retrieve?content_hash=...``.

    Returns:

    * ``{"found": True, "fragment": ...}`` on success;
    * ``{"found": False, "fragment": None}`` on a benign miss;
    * ``{"found": False, "fragment": None, "corrupt": True,
      "payload_hash": ...}`` when the fragment metadata is
      present but the on-disk bytes fail decryption (so a
      client can distinguish tampering / key-rotation loss
      from a simple miss).
    """
    if node is None:
        return _ok({"found": False, "fragment": None})
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    frag = node.retrieve(
        content_hash,
        caller_tenant=caller_tenant,
        caller_scopes=caller_scopes,
    )
    if not frag:
        return _ok({"found": False, "fragment": None})

    # Probe the bytes: if the fragment's payload_ref points
    # at a blob that fails decryption, surface that as
    # ``corrupt: True`` rather than a successful 200 with
    # metadata that the client cannot use.
    payload_ref = getattr(frag, "payload_ref", None)
    if payload_ref:
        try:
            blob = node.content_store.get(payload_ref)
        except Exception as exc:
            from membrane.security.encryption import DecryptError

            if isinstance(exc, DecryptError):
                logger.warning(
                    "op_retrieve: payload_ref=%s is corrupt: %s",
                    payload_ref,
                    exc,
                )
                return _ok(
                    {
                        "found": False,
                        "fragment": None,
                        "corrupt": True,
                        "payload_hash": content_hash,
                    }
                )
            raise
        if blob is None:
            return _ok({"found": False, "fragment": None})

    return _ok({"found": True, "fragment": to_dict(frag)})


def op_store(
    node: Node | None,
    fragment_payload: JsonDict,
    is_primary: bool = False,
    *,
    cluster: Cluster | None = None,
    quorum_attempt: object | None = None,
    draining: bool = False,
    auth_context: AuthContext | None = None,
    cluster_metrics: ClusterMetrics | None = None,
) -> tuple[int, JsonDict]:
    """``POST /store`` — store a fragment with a configured consistency level.

    Strong / quorum consistency: store locally first, then call
    ``quorum_attempt(fragment, quorum_count, timeout_sec)`` and
    return ``503`` + ``Retry-After: 1`` if it does not reach the
    write threshold. Failed writes are evicted locally so the
    cluster never holds a partial-write footprint.

    Eventual consistency: store locally and return immediately;
    the asynchronous replication thread propagates the fragment
    to replicas.

    Args:
        node: Local :class:`Node`.
        fragment_payload: Wire-format dict carrying the v3
            schema (consistency + hlc fields included).
        is_primary: Whether this node owns the primary shard.
        cluster: Optional cluster manager; consulted for the
            configured default consistency when the fragment
            ships with ``consistency='strong'`` (the v3 wire
            default).
        quorum_attempt: Optional callable matching
            :func:`membrane.quorum.attempt_quorum_acks`.
            ``None`` falls back to local-only writes for
            single-node deployments and tests.
        cluster_metrics: Optional :class:`ClusterMetrics` whose
            per-tenant operation counter is bumped on every
            successful store.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": True, ...})``
        on success, ``(503, {"error": "quorum timeout", ...})``
        on timeout, ``(200, {"error": ...})`` on user input
        failure.
    """
    if node is None:
        return _ok({"error": "no node"})
    if draining:
        return 503, {"error": "node draining", "Retry-After": 1}
    frag = from_dict(fragment_payload)

    # Honor the per-fragment consistency; fall back to the
    # cluster's default when the wire value matches "strong" and
    # the cluster has a different default configured.
    consistency = frag.consistency
    if cluster is not None:
        cfg_default = getattr(cluster.config, "default_consistency", "strong")
        if (
            consistency == "strong"
            and cfg_default in {"quorum", "eventual"}
        ):
            consistency = cfg_default
            # Fragment is frozen; rebuild a copy with the
            # downgraded level so the quorum attempt sees the
            # new value.
            frag = frag.with_consistency(consistency)

    # Local write always happens first so the cluster keeps a
    # single, source-of-truth copy at the primary even if quorum
    # is later achieved asynchronously.
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    try:
        ok = node.store(
            frag,
            is_primary=is_primary,
            caller_tenant=caller_tenant,
            caller_scopes=caller_scopes,
        )
    except TenantScopeError as exc:
        return 403, {"error": "tenant scope", "detail": str(exc)}
    if not ok:
        return _ok({"success": False, "content_hash": frag.identity.payload_hash})
    if cluster_metrics is not None and hasattr(cluster_metrics, "tenant"):
        cluster_metrics.tenant.bump_operation(frag.tenant_id, 1)

    if consistency == "eventual":
        return _ok({"success": True, "content_hash": frag.identity.payload_hash})

    # Strong / quorum paths block on a quorum fan-out. The
    # ``quorum_attempt`` callable is wired by Server from a
    # :class:`~membrane.quorum.attempt_quorum_acks` instance;
    # when it is absent we degrade to local-only success (the
    # production deployment path).
    if quorum_attempt is None or cluster is None:
        return _ok({"success": True, "content_hash": frag.identity.payload_hash})

    quorum_count = int(getattr(cluster.config, "quorum_count", 2))
    timeout_sec = float(getattr(cluster.config, "cluster_quorum_timeout_sec", 9.0))
    if quorum_count <= 1:
        return _ok({"success": True, "content_hash": frag.identity.payload_hash})

    replica_peers = list(_replica_peers(cluster, frag.identity.payload_hash, quorum_count))
    if not replica_peers:
        return _ok({"success": True, "content_hash": frag.identity.payload_hash})

    try:
        result = quorum_attempt(frag, replica_peers, quorum_count, timeout_sec)  # type: ignore[operator]
    except Exception as exc:
        logger.warning("op_store quorum_attempt failed: %s", exc)
        return 503, {"error": "quorum_attempt failed", "detail": str(exc)}

    ack_count = int(getattr(result, "ack_count", 0))
    timed_out = bool(getattr(result, "timed_out", True))
    success = bool(getattr(result, "success", False))
    if not success:
        # Roll back the local write so gossip does not propagate
        # a fragment that the cluster never acked. This is the
        # fail-closed contract.
        try:
            node.remove_fragment(frag.identity.payload_hash)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to roll back local fragment: %s", exc)
        return (
            503,
            {
                "error": "quorum timeout" if timed_out else "quorum not met",
                "ack_count": ack_count,
                "required": quorum_count,
                "Retry-After": 1,
            },
        )
    return _ok({"success": True, "content_hash": frag.identity.payload_hash})


def _replica_peers(cluster: Cluster, content_hash: str, count: int) -> list[Peer]:
    """Pick up to ``count`` replica peers from the cluster membership.

    Iteration order is the membership's natural snapshot order;
    the shard map owns the per-hash placement, but at write time
    op_store spreads the fan-out across the healthy peer set so
    a temporary primary shuffle still writes through. The
    cluster's :attr:`Membership.healthy` filters out the
    failing nodes for the duration of the fan-out.

    Args:
        cluster: Cluster manager whose Membership we read.
        content_hash: Fragment being replicated; unused at the
            moment because primary placement is determined by
            op_store, but kept for the future "place replicas on
            shard-peers" hook.
        count: Maximum number of peers to return.

    Returns:
        list[Peer]: Up to ``count`` peers.
    """
    healthy: list[str] = [p.node_id for p in cluster.membership.healthy()]
    healthy = [nid for nid in healthy if nid != cluster.node_id]
    peers: list[Peer] = []
    for nid in healthy[:count]:
        client = cluster.membership.get_client(nid)
        if client is not None:
            peers.append(client)
    return peers


def op_replicate(
    node: Node | None,
    fragment_payload: JsonDict,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /replicate`` — store a fragment as a non-primary replica."""
    if node is None:
        return _ok({"error": "no node"})
    frag = from_dict(fragment_payload)
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    try:
        ok = node.store(
            frag,
            is_primary=False,
            caller_tenant=caller_tenant,
            caller_scopes=caller_scopes,
        )
    except TenantScopeError as exc:
        return 403, {"error": "tenant scope", "detail": str(exc)}
    return _ok({"success": ok, "content_hash": frag.identity.payload_hash})


def op_prefill(
    node: Node | None,
    backend: Backend | None,
    prompt_tokens: list[int],
    model_id: str = "default",
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /prefill`` — run prefill and store fragments as primary."""
    if node is None:
        return _ok({"error": "no node"})
    backend = backend or CPU()
    fragments = backend.prefill(prompt_tokens, model_id)
    for frag in fragments:
        node.store(frag, is_primary=True)
    return _ok(
        {
            "success": True,
            "fragments": [to_dict(f) for f in fragments],
        }
    )


__all__ = [
    "MAX_BODY_BYTES",
    "op_heartbeat",
    "op_inventory",
    "op_metrics",
    "op_peers",
    "op_prefill",
    "op_replicate",
    "op_retrieve",
    "op_store",
]

# Re-export cluster-layer ops for backward compatibility.
# The cluster lifecycle lives in :mod:`membrane.transport.ops_cluster`;
# the original import paths (``from membrane.transport.ops import
# op_join`` etc.) continue to resolve.
from membrane.transport.ops_cluster import (  # noqa: F401
    op_delete,
    op_gossip,
    op_join,
    op_leave,
    op_purge,
    op_sync,
    op_tombstone,
    op_verify_received,
)
