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

import json
import logging
import time
from typing import Any, cast
from urllib.request import Request, urlopen

from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.gc import TombstoneTable
from membrane.metrics import MetricsCollector
from membrane.network.cluster import Cluster
from membrane.network.peer import JsonDict
from membrane.node import Node
from membrane.serialization import from_dict, to_dict
from membrane.transfer import TransferService

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


def op_inventory(node: Node | None) -> tuple[int, JsonDict]:
    """``GET /inventory`` — node's inventory digest."""
    if node is None:
        return _ok({"node_id": "", "digest": {}})
    digest = {h: frag.version_id for h, frag in node.fragments.items()}
    return _ok({"node_id": node.node_id, "digest": digest})


def op_peers(cluster: Cluster | None) -> tuple[int, JsonDict]:
    """``GET /peers`` — cluster membership view."""
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    return _ok({"peers": cluster.membership.to_json()})


def op_retrieve(node: Node | None, content_hash: str) -> tuple[int, JsonDict]:
    """``GET /retrieve?content_hash=...``."""
    if node is None:
        return _ok({"found": False, "fragment": None})
    frag = node.retrieve(content_hash)
    if frag:
        return _ok({"found": True, "fragment": to_dict(frag)})
    return _ok({"found": False, "fragment": None})


def op_store(
    node: Node | None,
    fragment_payload: JsonDict,
    is_primary: bool = False,
) -> tuple[int, JsonDict]:
    """``POST /store``."""
    if node is None:
        return _ok({"error": "no node"})
    frag = from_dict(fragment_payload)
    ok = node.store(frag, is_primary=is_primary)
    return _ok({"success": ok, "content_hash": frag.identity.payload_hash})


def op_replicate(
    node: Node | None,
    fragment_payload: JsonDict,
) -> tuple[int, JsonDict]:
    """``POST /replicate`` — store a fragment as a non-primary replica."""
    if node is None:
        return _ok({"error": "no node"})
    frag = from_dict(fragment_payload)
    ok = node.store(frag, is_primary=False)
    return _ok({"success": ok, "content_hash": frag.identity.payload_hash})


def op_prefill(
    node: Node | None,
    backend: Backend | None,
    prompt_tokens: list[int],
    model_id: str = "default",
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


def op_sync(
    node: Node | None,
    transfer_service: TransferService,
    source_url: str,
) -> tuple[int, JsonDict]:
    """``POST /sync`` — pull missing fragments from a source URL."""
    if not source_url:
        return _ok({"error": "missing source_url"})
    if node is None:
        return _ok({"error": "no node"})
    try:
        with urlopen(Request(f"{source_url}/inventory"), timeout=5) as resp:
            remote_data = json.loads(resp.read().decode())
        remote_digest = remote_data.get("digest", {})
        local_digest = transfer_service.inventory_digest(node) or {}
        missing = transfer_service.compare_inventories(local_digest, remote_digest)
        transferred: list[str] = []
        for h in missing:
            with urlopen(
                Request(f"{source_url}/retrieve?content_hash={h}"), timeout=5
            ) as resp:
                remote_frag_data = json.loads(resp.read().decode())
            if remote_frag_data.get("found"):
                frag = from_dict(remote_frag_data["fragment"])
                if node.store(frag, is_primary=False):
                    transferred.append(h)
        return _ok({"success": True, "transferred": transferred})
    except Exception as exc:
        logger.warning("sync failed: %s", exc)
        return _ok({"error": "internal"})


def op_join(
    cluster: Cluster | None,
    node_id: str,
    host: str,
    port: int,
    headers: dict[str, str] | None = None,
    authenticator: object | None = None,
) -> tuple[int, JsonDict]:
    """``POST /join`` — add a peer to the cluster.

    At 2.0+, when the cluster's :class:`~membrane.transport.tls.MTLSConfig`
    is set, ``op_join`` requires a verified peer cert. The
    ``authenticator`` (typically an
    :class:`~membrane.auth.mtls.MTLSAuthenticator`) is invoked
    against ``headers``; an authentication failure raises
    :class:`~membrane.auth.AuthBackendError` which the calling
    transport translates into a 401 response.

    Args:
        cluster: Local cluster manager.
        node_id: Identifier of the joining peer.
        host: Bind host reported by the peer.
        port: Bind port reported by the peer.
        headers: Inbound request headers (used to extract the
            verified mTLS peer CN).
        authenticator: Optional
            :class:`~membrane.auth.Authenticator` implementation
            that gates the join. ``None`` is supported for
            single-node deployments.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": True, "peers":
            [...]})`` on success, ``(401, ...)`` on auth failure,
            ``(200, {"error": ...})`` on user input failure.
    """
    if not node_id or not host or not port:
        return _ok({"error": "missing node_id, host, or port"})
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    peer_cn = ""
    if authenticator is not None and headers is not None:
        # Defer the import to avoid a top-level cycle between
        # transport.ops and auth.mtls.
        from membrane.auth import AuthBackendError, AuthRequest

        request = AuthRequest(
            method="POST",
            path="/join",
            headers={k.lower(): v for k, v in (headers or {}).items()},
            client="",
        )
        try:
            context = authenticator.authenticate(request)  # type: ignore[attr-defined]
        except AuthBackendError as exc:
            logger.warning("op_join rejected: %s", exc)
            return 401, {"error": str(exc)}
        peer_cn = context.subject
        # The CN prefix already grants the right scope; an
        # additional node_id-vs-CN check defends against
        # impersonation when a peer's CN is "admin-1" but its
        # node_id is "n2".
        expected_prefix = peer_cn.split("-", 1)[0]
        if not node_id.startswith(f"{expected_prefix}-") and peer_cn != node_id:
            logger.warning(
                "op_join rejected: cn=%s does not match node_id=%s",
                peer_cn,
                node_id,
            )
            return 401, {"error": "CN does not match node_id"}
    cluster.membership.add(node_id, host, port, peer_cn=peer_cn)
    return _ok({"success": True, "peers": cluster.membership.to_json()})


def op_leave(cluster: Cluster | None, node_id: str) -> tuple[int, JsonDict]:
    """``POST /leave``."""
    if not node_id:
        return _ok({"error": "missing node_id"})
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    cluster.membership.remove(node_id)
    leaving_hashes = {
        h for h, primary in cluster.shard_manager.primary_map.items() if primary == node_id
    }
    if cluster.migrator is not None and leaving_hashes:
        try:
            cluster.migrator.migrate(leaving_hashes, node_id)
        except Exception as exc:
            logger.warning("migrator.migrate(%d hashes, leaving=%s) failed: %s", len(leaving_hashes), node_id, exc)
    return _ok({"success": True})


def op_gossip(cluster: Cluster | None, data: JsonDict) -> tuple[int, JsonDict]:
    """``POST /gossip``."""
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    return _ok(cluster.gossip.handle(data))


def op_delete(
    node: Node | None,
    tombstones: TombstoneTable | None,
    content_hash: str,
    node_id: str,
    tombstone_until: float | None = None,
) -> tuple[int, JsonDict]:
    """``POST /delete`` — soft-delete a fragment on the local node.

    Writes a tombstone first so peer gossip has time to converge,
    then removes the fragment from the local :class:`Node`. If
    ``tombstones`` is ``None`` the operation degrades to an
    immediate hard delete (the producer is a single-process
    in-memory deployment with no replicas).

    Args:
        node: Local :class:`Node` or ``None`` to fail-fast.
        tombstones: Shared :class:`TombstoneTable` instance, or
            ``None`` to skip the soft-delete step.
        content_hash: Hash of the fragment to delete.
        node_id: Identifier of the node announcing the delete.
        tombstone_until: Optional explicit deadline (Unix time).
            ``None`` defaults to ``time.time() + 60.0``.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": bool, ...})``.
    """
    if node is None:
        return _ok({"error": "no node"})
    if content_hash not in node.fragments:
        return _ok({"success": True, "noop": True, "content_hash": content_hash})
    if tombstones is not None:
        deadline = tombstone_until if tombstone_until is not None else time.time() + 60.0
        tombstones.record(content_hash, until=deadline, node_ids={node_id})
    node.remove_fragment(content_hash)
    return _ok({"success": True, "content_hash": content_hash})


def op_tombstone(
    tombstones: TombstoneTable | None,
    content_hash: str,
    until: float,
    node_id: str,
) -> tuple[int, JsonDict]:
    """``POST /tombstone`` — record a soft-delete mark without removing.

    Used by gossip delivery to seed a tombstone when a peer
    announces a delete without forcing every replica to call
    :func:`op_delete` separately.

    Args:
        tombstones: Shared :class:`TombstoneTable`.
        content_hash: Hash being tombstoned.
        until: Wall-clock deadline after which the tombstone is
            considered expired.
        node_id: Identifier of the node announcing the delete.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": bool})``.
    """
    if tombstones is None:
        return _ok({"error": "no tombstone table configured"})
    tombstones.record(content_hash, until=until, node_ids={node_id})
    return _ok({"success": True, "content_hash": content_hash})


def op_purge(
    node: Node | None,
    tombstones: TombstoneTable | None,
    content_hash: str,
) -> tuple[int, JsonDict]:
    """``POST /purge`` — admin force-delete bypassing the soft-delete.

    Args:
        node: Local :class:`Node`.
        tombstones: Shared :class:`TombstoneTable`.
        content_hash: Hash to delete unconditionally.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": bool})``.
    """
    if node is None:
        return _ok({"error": "no node"})
    if tombstones is not None:
        tombstones.sweep_expired()  # opportunistic, not blocking
    removed = content_hash in node.fragments
    if removed:
        node.remove_fragment(content_hash)
    return _ok({"success": removed, "content_hash": content_hash})


__all__ = [
    "MAX_BODY_BYTES",
    "op_delete",
    "op_gossip",
    "op_heartbeat",
    "op_inventory",
    "op_join",
    "op_leave",
    "op_metrics",
    "op_peers",
    "op_prefill",
    "op_purge",
    "op_replicate",
    "op_retrieve",
    "op_store",
    "op_sync",
    "op_tombstone",
]
