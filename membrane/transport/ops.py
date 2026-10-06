"""Shared HTTP operation logic.

This module holds the *business* logic that backs each Membrane HTTP
endpoint. The FastAPI binding :mod:`membrane.transport.routes_fastapi`
delegates to these functions so the actual store / retrieve / sync
logic lives in exactly one place.

Each function takes plain domain objects (``Node``,
``TransferService``, ``Cluster``, ``Backend``) and returns either
a JSON-ready dict (success) or a tuple ``(status_code, body)``
that the transport layer maps onto its native response type.

Thread safety:
    The operations are stateless and forward to the domain objects,
    which own their own concurrency.
"""

import logging
import re
from collections.abc import Callable
from typing import Any, cast

from membrane.auth import AuthContext
from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.errors import TenantScopeError
from membrane.fragment import Fragment
from membrane.metrics import ClusterMetrics, MetricsCollector
from membrane.network.cluster import Cluster
from membrane.network.peer import JsonDict, Peer
from membrane.node import Node
from membrane.serialization import from_dict, to_dict
from membrane.store.digest import BUCKETS, decode_cursor, encode_cursor
from membrane.wire.v3.chunks import sha256_hex

logger = logging.getLogger(__name__)

PAYLOAD_REF_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,256}")


MAX_BODY_BYTES: int = 100 << 20
"""Maximum allowed request body size in bytes (100 MiB)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def err(status: int, message: str) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` error tuple.

    Args:
        status: HTTP status code.
        message: Error message for the response body.

    Returns:
        tuple[int, JsonDict]: ``(status, {"error": message})``.
    """
    return status, cast(JsonDict, {"error": message})


def ok_response(body: Any) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` success tuple.

    Accepts any JSON-serializable mapping; the helper widens to
    ``JsonDict`` so deeply-typed nested dicts (``dict[str, int]``,
    ``list[dict[str, Any]]``, etc.) flow through without an
    explicit cast at every builder site.

    Args:
        body: Response body.

    Returns:
        tuple[int, JsonDict]: A uniform ``(status, body)`` success tuple.
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
    advertised: JsonDict | None = None,
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

    Args:
        node: Local :class:`Node`.
        cluster: Cluster manager; ``None`` on a single node.
        headers: Lowercased request headers.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.
        advertised: Further fields peers route by (role, GPU load).

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if node is None:
        return ok_response({"error": "no node"})
    stats = node.get_stats()
    if cluster is not None and headers is not None:
        cn = headers.get("x-local-peer-cn") or headers.get("X-Local-Peer-CN")
        if cn:
            cluster.membership.record_peer_cn(node.node_id, cn)
    return ok_response(
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
        | (advertised or {})
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

    Args:
        node: Local :class:`Node`.
        metrics_registry: Optional :class:`MetricsCollector` for the
            ``/metrics`` Prometheus endpoint. When ``None``, ``/metrics``
            falls back to a JSON snapshot of the node's stats.

    Returns:
        tuple[int, JsonDict | tuple[str, dict[str, str]]]: ``(200, (text,
        headers))`` for Prometheus, or ``(200, json)``.
    """
    if metrics_registry is not None:
        return 200, (
            metrics_registry.render(),
            {"media_type": "text/plain; version=0.0.4"},
        )
    if node is None:
        return ok_response({"error": "no node"})
    stats = node.get_stats()
    return ok_response(
        {
            "node_id": node.node_id,
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "load": node.heartbeat(),
        }
    )


def op_inventory(
    node: Node | None,
    auth_context: AuthContext | None = None,
    after: str = "",
    limit: int = 0,
    bucket: int | None = None,
) -> tuple[int, JsonDict]:
    """``GET /inventory`` — node's inventory digest, optionally one page at a time.

    Pages walk the inventory bucket by bucket
    (:mod:`membrane.store.digest`), so a page costs in proportion to one
    bucket, not to the whole inventory.

    Args:
        node: Local :class:`Node`.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.
        after: Cursor returned as ``next`` by the previous page.
        limit: Page size; ``0`` returns everything (or the whole bucket).
        bucket: Only this bucket's fragments.

    Returns:
        tuple[int, JsonDict]: ``(status, body)``; ``next`` is the cursor for
        the following page, empty on the last page.
    """
    if node is None:
        return ok_response({"node_id": "", "digest": {}, "next": ""})
    if bucket is not None:
        if not 0 <= bucket < BUCKETS:
            return 400, {"error": f"bucket must be in [0, {BUCKETS})"}
        digest, cursor = node.digest.page(bucket, after, limit)
        return ok_response({"node_id": node.node_id, "bucket": bucket, "digest": digest, "next": cursor})
    if limit <= 0:
        snapshot = node.fragment_snapshot()
        return ok_response(
            {"node_id": node.node_id, "digest": {h: f.version_id for h, f in snapshot.items()}, "next": ""}
        )
    current, last = decode_cursor(after)
    page: dict[str, int] = {}
    cursor = ""
    while current < BUCKETS:
        entries, more = node.digest.page(current, last, limit - len(page))
        page.update(entries)
        if more:
            cursor = encode_cursor(current, more)
            break
        current, last = current + 1, ""
        if len(page) >= limit:
            cursor = encode_cursor(current, "") if current < BUCKETS else ""
            break
    return ok_response({"node_id": node.node_id, "digest": page, "next": cursor})


def op_inventory_buckets(node: Node | None) -> tuple[int, JsonDict]:
    """``GET /inventory/buckets`` — every bucket's digest and the root.

    Args:
        node: Local :class:`Node`.

    Returns:
        tuple[int, JsonDict]: ``{"node_id", "buckets": [hex, ...], "root", "count"}``.
    """
    if node is None:
        return ok_response({"node_id": "", "buckets": [], "root": "", "count": 0})
    digest = node.digest
    return ok_response(
        {
            "node_id": node.node_id,
            "buckets": [f"{value:016x}" for value in digest.buckets()],
            "root": digest.root().hex(),
            "count": len(digest),
        }
    )


def op_peers(cluster: Cluster | None, auth_context: AuthContext | None = None) -> tuple[int, JsonDict]:
    """``GET /peers`` — cluster membership view.

    Args:
        cluster: Cluster manager; ``None`` on a single node.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if cluster is None:
        return ok_response({"error": "cluster manager not enabled"})
    return ok_response({"peers": cluster.membership.to_json()})


def op_retrieve(
    node: Node | None,
    content_hash: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``GET /retrieve?content_hash=...``.

    Args:
        node: Local :class:`Node`.
        content_hash: Content hash of the fragment.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

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
        return ok_response({"found": False, "fragment": None})
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    frag = node.retrieve(
        content_hash,
        caller_tenant=caller_tenant,
        caller_scopes=caller_scopes,
    )
    if not frag:
        return ok_response({"found": False, "fragment": None})

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
                return ok_response(
                    {
                        "found": False,
                        "fragment": None,
                        "corrupt": True,
                        "payload_hash": content_hash,
                    }
                )
            raise
        if blob is None:
            return ok_response({"found": False, "fragment": None})

    return ok_response({"found": True, "fragment": to_dict(frag)})


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
    store_guard: Callable[[Fragment], None] | None = None,
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
        draining: Whether the node is draining (writes are refused with
            503).
        auth_context: Authenticated caller; ``None`` when authentication is
            off.
        store_guard: Raises ``ValueError`` (with an optional ``status``) to
            refuse the fragment, e.g. for a compatibility mismatch.

    Returns:
        tuple[int, JsonDict]: ``(200, {"success": True, ...})``
        on success, ``(422, {"error": ...})`` when the fragment's
        ``payload_ref`` is not in the content store,
        ``(503, {"error": "quorum timeout", ...})``
        on timeout, ``(200, {"error": ...})`` on user input
        failure.
    """
    if node is None:
        return ok_response({"error": "no node"})
    if draining:
        return 503, {"error": "node draining", "Retry-After": 1}
    frag = from_dict(fragment_payload)

    # Honor the per-fragment consistency; fall back to the
    # cluster's default when the wire value matches "strong" and
    # the cluster has a different default configured.
    consistency = frag.consistency
    if cluster is not None:
        cfg_default = getattr(cluster.config, "default_consistency", "strong")
        if consistency == "strong" and cfg_default in {"quorum", "eventual"}:
            consistency = cfg_default
            # Fragment is frozen; rebuild a copy with the
            # downgraded level so the quorum attempt sees the
            # new value.
            frag = frag.with_consistency(consistency)

    # Payload bytes reach the content store out of band, before the
    # metadata is published. Accepting a fragment whose bytes are
    # absent would report success for a write that /retrieve then
    # reports as missing, so reject it up front.
    if frag.payload_ref is not None and not payload_present(node, frag.payload_ref):
        return 422, {
            "error": "payload_ref not found in content store",
            "payload_ref": frag.payload_ref,
            "content_hash": frag.identity.payload_hash,
        }
    if store_guard is not None:
        try:
            store_guard(frag)
        except ValueError as exc:
            return getattr(exc, "status", 409), {"error": "store refused", "detail": str(exc)}

    # Local write always happens first so the cluster keeps a
    # single, source-of-truth copy at the primary even if quorum
    # is later achieved asynchronously.
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    # A failed quorum only rolls back a copy this call created; an
    # idempotent re-store must not delete an already-acknowledged one.
    existed_before = frag.identity.payload_hash in node.fragments
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
        return ok_response({"success": False, "content_hash": frag.identity.payload_hash})
    if cluster_metrics is not None and hasattr(cluster_metrics, "tenant"):
        cluster_metrics.tenant.bump_operation(frag.tenant_id, 1)

    if consistency == "eventual":
        return ok_response({"success": True, "content_hash": frag.identity.payload_hash})

    # Strong / quorum paths block on a quorum fan-out. The
    # ``quorum_attempt`` callable is wired by Server from a
    # :class:`~membrane.quorum.attempt_quorum_acks` instance;
    # when it is absent we degrade to local-only success (the
    # production deployment path).
    if quorum_attempt is None or cluster is None:
        return ok_response({"success": True, "content_hash": frag.identity.payload_hash})

    # ``quorum_count`` is the number of copies that must exist before
    # the write is acknowledged, the local copy included, so the
    # fan-out waits for ``quorum_count - 1`` peer acks. It is sent to
    # every replica so one slow peer does not fail the write.
    quorum_count = int(getattr(cluster.config, "quorum_count", 2))
    timeout_sec = float(getattr(cluster.config, "cluster_quorum_timeout_sec", 9.0))
    required_peer_acks = quorum_count - 1
    if required_peer_acks <= 0:
        return ok_response({"success": True, "content_hash": frag.identity.payload_hash})

    fan_out = max(required_peer_acks, int(getattr(cluster.config, "replica_count", required_peer_acks)))
    replica_peers = list(select_replica_peers(cluster, frag.identity.payload_hash, fan_out))
    if len(replica_peers) < required_peer_acks:
        # Fail closed: an isolated node must not acknowledge a strong
        # write it cannot replicate.
        rollback_local_write(node, frag.identity.payload_hash, existed_before)
        return 503, {
            "error": "quorum not met",
            "detail": "not enough healthy peers",
            "ack_count": 0,
            "required": quorum_count,
            "Retry-After": 1,
        }

    # Replicas must get the KV bytes too, or they cannot serve the fragment.
    blob = node.content_store.get(frag.payload_ref) if frag.payload_ref is not None else None
    try:
        result = quorum_attempt(frag, replica_peers, required_peer_acks, timeout_sec, blob=blob)  # type: ignore[operator]
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
        rollback_local_write(node, frag.identity.payload_hash, existed_before)
        return (
            503,
            {
                "error": "quorum timeout" if timed_out else "quorum not met",
                "ack_count": ack_count,
                "required": quorum_count,
                "Retry-After": 1,
            },
        )
    return ok_response({"success": True, "content_hash": frag.identity.payload_hash})


def payload_present(node: Node, payload_ref: str) -> bool:
    """Whether ``payload_ref`` is in the node's content store.

    A ref the store cannot address at all (e.g. too short for the
    on-disk layout) counts as absent rather than a server error.

    Args:
        node: Local :class:`Node`.
        payload_ref: Content-store key of the fragment's payload bytes.

    Returns:
        bool: Whether ``payload_ref`` is in the node's content store.
    """
    try:
        return bool(node.content_store.has(payload_ref))
    except ValueError:
        return False


def rollback_local_write(node: Node, content_hash: str, existed_before: bool) -> None:
    """Undo a local write whose quorum failed, unless the copy pre-existed.

    Args:
        node: Local :class:`Node`.
        content_hash: Content hash of the fragment.
        existed_before: Whether the fragment was present before this write.
    """
    if existed_before:
        return
    try:
        node.remove_fragment(content_hash)
    except KeyError:
        pass  # Already evicted or removed concurrently.
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to roll back local fragment: %s", exc)


def select_replica_peers(cluster: Cluster, content_hash: str, count: int) -> list[Peer]:
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
    is_primary: bool = False,
) -> tuple[int, JsonDict]:
    """``POST /replicate`` — store a fragment sent by a peer.

    The payload bytes must already be here (``PUT /blobs`` first): a
    replica without its bytes cannot serve reads, so it is refused with
    ``422``.

    Args:
        node: Local :class:`Node`.
        fragment_payload: Wire-format dict carrying the v3 schema
            (consistency + hlc fields included).
        auth_context: Authenticated caller; ``None`` when authentication is
            off.
        is_primary: The sender is handing primary ownership to this node
            (drain or rebalancing).

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if node is None:
        return ok_response({"error": "no node"})
    frag = from_dict(fragment_payload)
    if frag.payload_ref is not None and not node.content_store.has(frag.payload_ref):
        return 422, {
            "error": "payload missing",
            "detail": f"PUT /blobs/{frag.payload_ref} before replicating its fragment",
            "content_hash": frag.identity.payload_hash,
        }
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
    return ok_response({"success": ok, "content_hash": frag.identity.payload_hash})


def valid_payload_ref(payload_ref: str) -> bool:
    """Whether ``payload_ref`` is safe to use as a content-store key from the wire.

    Args:
        payload_ref: Candidate key from a URL path.

    Returns:
        bool: True for 1-256 characters from ``[A-Za-z0-9._:-]`` that are
        not a path component like ``..``.
    """
    return bool(PAYLOAD_REF_PATTERN.fullmatch(payload_ref)) and payload_ref not in {".", ".."}


def op_put_blob(node: Node | None, payload_ref: str, data: bytes, claimed_sha256: str) -> tuple[int, JsonDict]:
    """``PUT /blobs/{payload_ref}`` — store KV bytes sent by a peer.

    Content-addressed and idempotent: bytes already held under the key are
    left as they are.

    Args:
        node: Local :class:`Node`.
        payload_ref: Content-store key.
        data: The request body.
        claimed_sha256: The ``X-Content-SHA256`` header.

    Returns:
        tuple[int, JsonDict]: ``200 {"stored": true}``; ``400`` for a bad
        key or a digest mismatch.
    """
    if node is None:
        return 503, {"error": "no node"}
    if not valid_payload_ref(payload_ref):
        return 400, {"error": "invalid payload_ref"}
    if claimed_sha256.lower() != sha256_hex(data):
        return 400, {"error": "digest mismatch", "detail": "X-Content-SHA256 does not match the body"}
    existing = node.content_store.has(payload_ref)
    if not existing:
        node.content_store.put(payload_ref, data)
    return 200, {"stored": True, "existing": existing, "bytes": len(data)}


def op_get_blob(node: Node | None, payload_ref: str) -> tuple[int, bytes | None]:
    """``GET /blobs/{payload_ref}`` — read KV bytes for a peer.

    Args:
        node: Local :class:`Node`.
        payload_ref: Content-store key.

    Returns:
        tuple[int, bytes | None]: ``(200, bytes)``, ``(404, None)`` when
        absent, or ``(400, None)`` for a bad key.
    """
    if node is None or not valid_payload_ref(payload_ref):
        return 400, None
    data = node.content_store.get(payload_ref)
    return (404, None) if data is None else (200, data)


def op_prefill(
    node: Node | None,
    backend: Backend | None,
    prompt_tokens: list[int],
    model_id: str = "default",
    auth_context: AuthContext | None = None,
    stamp: Callable[[Fragment], Fragment] | None = None,
) -> tuple[int, JsonDict]:
    """``POST /prefill`` — run prefill and store fragments as primary.

    Args:
        node: Local :class:`Node`.
        backend: Compute backend that runs the prefill; CPU by default.
        prompt_tokens: Prompt token IDs.
        model_id: Model identifier.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.
        stamp: Applied to each produced fragment (the node's compatibility
            fingerprint).

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if node is None:
        return ok_response({"error": "no node"})
    backend = backend or CPU()
    caller_tenant = auth_context.subject if auth_context is not None else ""
    caller_scopes = auth_context.scopes if auth_context is not None else frozenset()
    fragments = backend.prefill(prompt_tokens, model_id)
    # Fragments belong to the authenticated caller's tenant; without
    # this every tenant's prefill would land in the public tenant.
    if caller_tenant:
        fragments = [frag.with_tenant(caller_tenant) for frag in fragments]
    if stamp is not None:
        fragments = [stamp(frag) for frag in fragments]
    for frag in fragments:
        # Simulated / remote backends never write KV bytes; store
        # their placeholder payload so /retrieve can serve the
        # fragment. Backends with real frames already wrote them.
        if frag.payload_ref is not None and not payload_present(node, frag.payload_ref):
            payload = backend.simulated_payload(frag)
            if payload is not None:
                node.content_store.put(frag.payload_ref, payload)
        try:
            node.store(frag, is_primary=True, caller_tenant=caller_tenant, caller_scopes=caller_scopes)
        except TenantScopeError as exc:
            return 403, {"error": "tenant scope", "detail": str(exc)}
    return ok_response(
        {
            "success": True,
            "fragments": [to_dict(f) for f in fragments],
        }
    )


__all__ = [
    "MAX_BODY_BYTES",
    "PAYLOAD_REF_PATTERN",
    "op_get_blob",
    "op_heartbeat",
    "op_inventory",
    "op_inventory_buckets",
    "op_metrics",
    "op_peers",
    "op_prefill",
    "op_put_blob",
    "op_replicate",
    "op_retrieve",
    "op_store",
    "valid_payload_ref",
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
