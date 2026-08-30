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
from urllib.request import Request, urlopen

from membrane.compute.base import Backend
from membrane.compute.cpu import CPU
from membrane.metrics import MetricsCollector
from membrane.network.cluster import Cluster
from membrane.node import Node
from membrane.serialization import JsonDict
from membrane.serialization import from_dict as deserialize_fragment
from membrane.serialization import to_dict as serialize_fragment
from membrane.transfer import TransferService

logger = logging.getLogger(__name__)


MAX_BODY_BYTES: int = 100 << 20
"""Maximum allowed request body size in bytes (100 MiB)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(status: int, message: str) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` error tuple."""
    return status, {"error": message}


def _ok(body: JsonDict) -> tuple[int, JsonDict]:
    """Build a uniform ``(status, body)`` success tuple."""
    return 200, body


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def op_heartbeat(node: Node | None) -> tuple[int, JsonDict]:
    """``GET /heartbeat`` — node health and load snapshot.

    Always returns 200 with the body indicating status, mirroring
    the existing contract that the heartbeat is informational, not a
    strict liveness check (``/livez`` is the dedicated liveness
    probe).
    """
    if node is None:
        return _ok({"error": "no node"})
    stats = node.get_stats()
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
        return _ok({"found": True, "fragment": serialize_fragment(frag)})
    return _ok({"found": False, "fragment": None})


def op_store(
    node: Node | None,
    fragment_payload: JsonDict,
    is_primary: bool = False,
) -> tuple[int, JsonDict]:
    """``POST /store``."""
    if node is None:
        return _ok({"error": "no node"})
    frag = deserialize_fragment(fragment_payload)
    ok = node.store(frag, is_primary=is_primary)
    return _ok({"success": ok, "content_hash": frag.content_hash})


def op_replicate(
    node: Node | None,
    fragment_payload: JsonDict,
) -> tuple[int, JsonDict]:
    """``POST /replicate`` — store a fragment as a non-primary replica."""
    if node is None:
        return _ok({"error": "no node"})
    frag = deserialize_fragment(fragment_payload)
    ok = node.store(frag, is_primary=False)
    return _ok({"success": ok, "content_hash": frag.content_hash})


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
            "fragments": [serialize_fragment(f) for f in fragments],
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
                frag = deserialize_fragment(remote_frag_data["fragment"])
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
) -> tuple[int, JsonDict]:
    """``POST /join``."""
    if not node_id or not host or not port:
        return _ok({"error": "missing node_id, host, or port"})
    if cluster is None:
        return _ok({"error": "cluster manager not enabled"})
    cluster.membership.add(node_id, host, port)
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


__all__ = [
    "MAX_BODY_BYTES",
    "op_gossip",
    "op_heartbeat",
    "op_inventory",
    "op_join",
    "op_leave",
    "op_metrics",
    "op_peers",
    "op_prefill",
    "op_replicate",
    "op_retrieve",
    "op_store",
    "op_sync",
]
