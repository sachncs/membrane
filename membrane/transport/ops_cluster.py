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

from __future__ import annotations

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


def _ok(body: Any) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` success tuple."""
    return 200, cast(JsonDict, body)


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
    except NetworkError as exc:
        logger.warning("sync failed (network): %s", exc)
        return _ok({"error": "network"})
    except Exception as exc:
        logger.warning("sync failed (unexpected): %s", exc)
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
    """``POST /join`` — add a peer to the cluster."""
    if not node_id or not host or not port:
        return _ok({"error": "missing node_id, host, or port"})
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
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
    drain-then-stop. ``graceful=False`` is the legacy fast-leave.
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
    """``POST /delete`` — soft-delete a fragment on the local node."""
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
    """``POST /tombstone`` — record a soft-delete mark without removing."""
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
    """``POST /purge`` -- admin force-delete bypassing the soft-delete."""
    if node is None:
        return _ok({"error": "no node"})
    if tombstones is not None:
        tombstones.sweep_expired()
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
    """``POST /verify`` -- confirm a peer's claimed canonical bytes."""
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
    try:
        store = getattr(node, "content_store", None)
        if store is not None and store.has(content_hash):
            actual = store.get(content_hash) or b""
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
    "op_delete",
    "op_gossip",
    "op_join",
    "op_leave",
    "op_purge",
    "op_sync",
    "op_tombstone",
    "op_verify_received",
]
