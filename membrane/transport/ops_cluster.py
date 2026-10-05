"""Cluster-layer HTTP operations.

This module holds the per-op handlers that are cluster-state
machines rather than per-fragment CRUD:

* :func:`op_sync` -- pull missing fragments from a source URL.
* :func:`op_join` / :func:`op_leave` -- cluster membership
  transitions.
* :func:`op_gossip` -- gossip payload apply.
* :func:`op_delete` / :func:`op_tombstone` / :func:`op_purge`
  -- fragment deletion and tombstone bookkeeping.
* :func:`op_verify_received` -- verified-migration flow.

The split keeps the per-fragment CRUD (store / retrieve /
replicate / prefill) in :mod:`membrane.transport.ops` and
isolates the cluster-lifecycle surface here so the two
concerns can evolve independently. The original
:mod:`membrane.transport.ops` module re-exports every name
defined here for backward compatibility.
"""

import hashlib
import json
import logging
import time
from typing import Any, cast
from urllib.request import Request, urlopen

from membrane.auth import AuthContext
from membrane.errors import NetworkError
from membrane.gc import TombstoneTable
from membrane.network.cluster import Cluster
from membrane.network.peer import JsonDict
from membrane.node import Node
from membrane.serialization import from_dict
from membrane.transfer import TransferService

logger = logging.getLogger(__name__)


def ok(body: Any) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` success tuple.

    Args:
        body: Response body.

    Returns:
        tuple[int, JsonDict]: A uniform ``(status, body)`` success tuple.
    """
    return 200, cast(JsonDict, body)


def op_sync(
    node: Node | None,
    transfer_service: TransferService | None,
    source_url: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /sync`` — pull missing fragments from a source URL.

    Validates ``source_url`` against the SSRF policy before
    issuing any outbound HTTP request. A URL that fails the
    allow-list returns 400 with the SSRF reason.

    Args:
        node: Local node to fill.
        transfer_service: Service used to move fragments. A default
            :class:`TransferService` is created when ``None``.
        source_url: Base URL of the node to pull missing fragments from.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if not source_url:
        return ok({"error": "missing source_url"})
    if node is None:
        return ok({"error": "no node"})
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
        transfer_service = transfer_service or TransferService(local_node=node)
        local_digest = transfer_service.inventory_digest(node) or {}
        missing = transfer_service.compare_inventories(local_digest, remote_digest)
        transferred: list[str] = []
        for h in missing:
            try:
                retrieve_url = validate_outbound_url(f"{source_url}/retrieve?content_hash={h}")
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
        return ok({"success": True, "transferred": transferred})
    except NetworkError as exc:
        logger.warning("sync failed (network): %s", exc)
        return ok({"error": "network"})
    except Exception as exc:
        logger.warning("sync failed (unexpected): %s", exc)
        return ok({"error": "internal"})


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

    Args:
        cluster: Local cluster manager.
        node_id: Node identifier.
        host: Address of the joining node.
        port: Port of the joining node.
        headers: Lowercased request headers.
        authenticator: Authenticator used to verify the joining peer;
            ``None`` skips verification.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if not node_id or not host or not port:
        return ok({"error": "missing node_id, host, or port"})
    if cluster is None:
        return ok({"error": "cluster manager not enabled"})
    peer_cn = ""
    if authenticator is not None and headers is not None:
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
        if not cn_matches_node_id(authenticator, peer_cn, node_id):
            logger.warning(
                "op_join rejected: cn=%s does not match node_id=%s",
                peer_cn,
                node_id,
            )
            return 401, {"error": "CN does not match node_id"}
    cluster.membership.add(node_id, host, port, peer_cn=peer_cn)
    peers = cluster.membership.to_json()
    # Include the seed itself so the joiner can reach it; membership
    # entries only describe the seed's *other* peers.
    advertise_host = getattr(cluster, "advertise_host", None)
    if isinstance(advertise_host, str) and isinstance(cluster.node_id, str):
        peers = [*peers, {"node_id": cluster.node_id, "host": advertise_host, "port": cluster.port}]
    return ok({"success": True, "peers": peers})


def cn_matches_node_id(authenticator: object, peer_cn: str, node_id: str) -> bool:
    """Return whether an mTLS peer CN may register as ``node_id``.

    Under mTLS the certificate CN is the node's identity, so a CN may
    only join as itself: either ``node_id`` verbatim or
    ``<role>-<node_id>`` with a role prefix from
    :data:`~membrane.auth.mtls.CN_SCOPE_PREFIXES`. Other
    authenticators (API keys) identify services rather than nodes;
    for those, the route's ``write`` scope check is the gate.

    Args:
        authenticator: The authenticator that admitted the caller.
        peer_cn: Verified certificate CN of the caller.
        node_id: Node identifier.

    Returns:
        bool: Whether an mTLS peer CN may register as ``node_id``.
    """
    from membrane.auth.mtls import CN_SCOPE_PREFIXES, MTLSAuthenticator

    if not isinstance(authenticator, MTLSAuthenticator):
        return True
    if peer_cn == node_id:
        return True
    return any(peer_cn == f"{prefix}{node_id}" for prefix, _scope in CN_SCOPE_PREFIXES)


def op_leave(
    cluster: Cluster | None,
    node_id: str,
    graceful: bool = True,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /leave``.

    When ``graceful=True`` (the default) the path is
    drain-then-stop. ``graceful=False`` is the legacy fast-leave.

    Args:
        cluster: Local cluster manager.
        node_id: Node identifier.
        graceful: Drain before leaving instead of leaving immediately.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if not node_id:
        return ok({"error": "missing node_id"})
    if cluster is None:
        return ok({"error": "cluster manager not enabled"})
    server = getattr(cluster, "server", None)
    if graceful and server is not None and hasattr(server, "drain"):
        result = server.drain(deadline_sec=30.0)
        return ok(
            {
                "success": True,
                "graceful": True,
                "migrated": result["migrated"],
                "stragglers": result["stragglers"],
                "duration_sec": result["duration_sec"],
            }
        )
    cluster.membership.remove(node_id)
    return ok({"success": True, "graceful": False})


def op_gossip(
    cluster: Cluster | None,
    data: JsonDict,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /gossip``.

    Args:
        cluster: Local cluster manager.
        data: The sender's gossip state.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if cluster is None:
        return ok({"error": "cluster manager not enabled"})
    return ok(cluster.gossip.handle(data))


def op_delete(
    node: Node | None,
    tombstones: TombstoneTable | None,
    content_hash: str,
    node_id: str,
    tombstone_until: float | None = None,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /delete`` — soft-delete a fragment on the local node.

    Args:
        node: Local node.
        tombstones: Tombstone table to record the delete in.
        content_hash: Content hash of the fragment.
        node_id: Node identifier.
        tombstone_until: Unix time the tombstone expires; a default window
            when ``None``.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if node is None:
        return ok({"error": "no node"})
    if content_hash not in node.fragments:
        return ok({"success": True, "noop": True, "content_hash": content_hash})
    if tombstones is not None:
        deadline = tombstone_until if tombstone_until is not None else time.time() + 60.0
        tombstones.record(content_hash, until=deadline, node_ids={node_id})
    node.remove_fragment(content_hash)
    return ok({"success": True, "content_hash": content_hash})


def op_tombstone(
    tombstones: TombstoneTable | None,
    content_hash: str,
    until: float,
    node_id: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /tombstone`` — record a soft-delete mark without removing.

    Args:
        tombstones: Tombstone table to update.
        content_hash: Content hash of the fragment.
        until: Unix time the tombstone expires.
        node_id: Node identifier.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if tombstones is None:
        return ok({"error": "no tombstone table configured"})
    tombstones.record(content_hash, until=until, node_ids={node_id})
    return ok({"success": True, "content_hash": content_hash})


def op_purge(
    node: Node | None,
    tombstones: TombstoneTable | None,
    content_hash: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /purge`` -- admin force-delete bypassing the soft-delete.

    Args:
        node: Local node.
        tombstones: Tombstone table whose expired entries are purged.
        content_hash: Content hash of the fragment.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if node is None:
        return ok({"error": "no node"})
    if tombstones is not None:
        tombstones.sweep_expired()
    removed = content_hash in node.fragments
    if removed:
        node.remove_fragment(content_hash)
    return ok({"success": removed, "content_hash": content_hash})


def op_verify_received(
    node: Node | None,
    content_hash: str,
    claimed_size: int,
    claimed_sha256_hex: str,
    auth_context: AuthContext | None = None,
) -> tuple[int, JsonDict]:
    """``POST /verify`` -- confirm a peer's claimed canonical bytes.

    Args:
        node: Local node holding the fragment.
        content_hash: Content hash of the fragment.
        claimed_size: Payload size the sender claims.
        claimed_sha256_hex: SHA-256 of the payload the sender claims.
        auth_context: Authenticated caller; ``None`` when authentication is
            off.

    Returns:
        tuple[int, JsonDict]: ``(status, body)`` for the transport to send.
    """
    if node is None:
        return ok({"error": "no node"})
    if content_hash not in node.fragments:
        return ok({"success": False, "reason": "fragment missing", "content_hash": content_hash})
    frag = node.fragments[content_hash]
    actual_size = frag.payload_size
    if actual_size != int(claimed_size):
        return ok(
            {
                "success": False,
                "reason": "size mismatch",
                "content_hash": content_hash,
                "claimed": int(claimed_size),
                "actual": int(actual_size),
            }
        )
    try:
        store = getattr(node, "content_store", None)
        if store is not None and store.has(content_hash):
            actual = store.get(content_hash) or b""
            actual_hex = hashlib.sha256(actual).hexdigest()
            return ok(
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
    return ok(
        {
            "success": True,
            "content_hash": content_hash,
            "size": int(actual_size),
            "claimed_sha256": str(claimed_sha256_hex),
        }
    )


__all__ = [
    "op_delete",
    "op_gossip",
    "op_join",
    "op_leave",
    "op_purge",
    "op_sync",
    "op_tombstone",
    "op_verify_received",
]
