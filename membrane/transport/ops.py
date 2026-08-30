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

from membrane.auth import AuthContext
from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.errors import TenantScopeError
from membrane.gc import TombstoneTable
from membrane.metrics import MetricsCollector
from membrane.network.cluster import Cluster
from membrane.network.peer import JsonDict, Peer
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
    """``GET /retrieve?content_hash=...``."""
    if node is None:
        return _ok({"found": False, "fragment": None})
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    frag = node.retrieve(
        content_hash,
        caller_tenant=caller_tenant,
        caller_scopes=caller_scopes,
    )
    if frag:
        return _ok({"found": True, "fragment": to_dict(frag)})
    return _ok({"found": False, "fragment": None})


def op_store(
    node: Node | None,
    fragment_payload: JsonDict,
    is_primary: bool = False,
    *,
    cluster: Cluster | None = None,
    quorum_attempt: object | None = None,
    draining: bool = False,
    auth_context: AuthContext | None = None,
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
    timeout_sec = float(getattr(cluster.config, "cluster_quorum_timeout_sec", 5.0))
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


def op_sync(
    node: Node | None,
    transfer_service: TransferService,
    source_url: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /sync`` — pull missing fragments from a source URL.

    Validates ``source_url`` against the SSRF policy before
    issuing any outbound HTTP request. A URL that fails the
    allow-list returns 400 with the SSRF reason.
    """
    if not source_url:
        return _ok({"error": "missing source_url"})
    if node is None:
        return _ok({"error": "no node"})
    from membrane.security import validate_outbound_url
    from membrane.security.url_allowlist import SSRFError

    try:
        inventory_url = validate_outbound_url(f"{source_url}/inventory")
    except SSRFError as exc:
        return 400, {"error": "ssrf rejected", "reason": str(exc), "url": "inventory"}
    try:
        with urlopen(Request(inventory_url), timeout=5) as resp:
            remote_data = json.loads(resp.read().decode())
        remote_digest = remote_data.get("digest", {})
        local_digest = transfer_service.inventory_digest(node) or {}
        missing = transfer_service.compare_inventories(local_digest, remote_digest)
        transferred: list[str] = []
        for h in missing:
            try:
                retrieve_url = validate_outbound_url(
                    f"{source_url}/retrieve?content_hash={h}"
                )
            except SSRFError as exc:
                return 400, {
                    "error": "ssrf rejected",
                    "reason": str(exc),
                    "url": "retrieve",
                    "content_hash": h,
                }
            with urlopen(Request(retrieve_url), timeout=5) as resp:
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
    auth_context: AuthContext | None = None,
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


def op_leave(
    cluster: Cluster | None,
    node_id: str,
    graceful: bool = True,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /leave``.

    When ``graceful=True`` (the default) the path is
    drain-then-stop:

    1. Mark ``is_draining=True`` so :func:`op_store` and
       :func:`op_replicate` start returning 503 + Retry-After.
    2. Run a best-effort migration pass for every primary the
       leaving node still owns.
    3. Hand off membership to the cluster's failure detector
       so a competing heartbeat never produces a phantom
       primary.

    When ``graceful=False`` the call is the legacy fast-leave:
    membership is removed immediately and the migrator fires
    inline. Test fixtures that want to assert the schema
    directly use this path.
    """
    if not node_id:
        return _ok({"error": "missing node_id"})
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    server = getattr(cluster, "server", None)
    if graceful and server is not None and hasattr(server, "drain"):
        result = server.drain(deadline_sec=30.0)
        return _ok(
            {
                "success": True,
                "graceful": True,
                "migrated": result["migrated"],
                "stragglers": result["stragglers"],
                "duration_sec": result["duration_sec"],
            }
        )
    cluster.membership.remove(node_id)
    leaving_hashes = {
        h for h, primary in cluster.shard_manager.primary_map.items() if primary == node_id
    }
    if cluster.migrator is not None and leaving_hashes:
        try:
            cluster.migrator.migrate(leaving_hashes, node_id)
        except Exception as exc:
            logger.warning("migrator.migrate(%d hashes, leaving=%s) failed: %s", len(leaving_hashes), node_id, exc)
    return _ok({"success": True, "graceful": False})


def op_gossip(
    cluster: Cluster | None,
    data: JsonDict,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
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
    auth_context: AuthContext | None = None,
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
    auth_context: AuthContext | None = None,
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
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /purge`` -- admin force-delete bypassing the soft-delete.

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


def op_verify_received(
    node: Node | None,
    content_hash: str,
    claimed_size: int,
    claimed_sha256_hex: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /verify`` -- confirm a peer's claimed canonical bytes.

    The verified-migration flow in Phase 3 sends the canonical
    bytes from the previous primary to a destination replica;
    before flipping the shard map the destination sends back a
    :func:`op_verify_received` request that ties the
    ``content_hash`` to a specific ``claimed_size`` and
    ``claimed_sha256_hex``. This op answers with the bytes the
    destination actually holds under that hash so the caller can
    compare.

    Args:
        node: Local :class:`Node`.
        content_hash: Hash of the fragment being verified.
        claimed_size: Size the caller believes it just stored.
        claimed_sha256_hex: Hex sha256 of the canonical bytes
            the caller stored.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": True, "size":
        actual, "sha256": actual})`` when the local node holds
        the fragment, ``(200, {"success": False, "reason":
        "fragment missing"})`` when it does not, ``(200,
        {"success": False, "reason": "size mismatch", "actual":
        ...})`` on a length disagreement.
    """
    if node is None:
        return _ok({"error": "no node"})
    if content_hash not in node.fragments:
        return _ok({"success": False, "reason": "fragment missing", "content_hash": content_hash})
    frag = node.fragments[content_hash]
    actual_size = frag.payload_size
    if actual_size != int(claimed_size):
        return _ok(
            {
                "success": False,
                "reason": "size mismatch",
                "content_hash": content_hash,
                "claimed": int(claimed_size),
                "actual": int(actual_size),
            }
        )
    # The sha256 verification looks at the canonical ContentStore
    # bytes when they are available; otherwise it accepts the
    # claimed hash as proof and reports size only. Phase 5 will
    # expand the verifier to read Merkle leaves.
    try:
        from membrane.content_store import InProcessBytes  # noqa: F401

        store = getattr(node, "content_store", None)
        if store is not None and store.has(content_hash):
            actual = store.get(content_hash) or b""
            import hashlib

            actual_hex = hashlib.sha256(actual).hexdigest()
            return _ok(
                {
                    "success": True,
                    "content_hash": content_hash,
                    "size": int(actual_size),
                    "sha256": actual_hex,
                    "claimed_sha256": str(claimed_sha256_hex),
                    "bytes_match": actual_hex == str(claimed_sha256_hex),
                }
            )
    except Exception:
        pass
    return _ok(
        {
            "success": True,
            "content_hash": content_hash,
            "size": int(actual_size),
            "claimed_sha256": str(claimed_sha256_hex),
        }
    )


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
