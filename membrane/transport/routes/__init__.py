"""HTTP route handlers, shared between stdlib HTTP and FastAPI transports.

Each route is a small module-level function that receives a
:class:`~membrane.transport.http.Handler` instance (which carries
``self.server.node``, ``self.server.compute_backend``,
``self.server.transfer_service``, ``self.server.cluster_manager``)
and uses ``self.send_json`` / ``self.read_json`` for I/O.

The :data:`ROUTES` mapping is ``(method, path) -> handler``. New
endpoints are added by writing a function below and registering it
in the table. Both transports (``stdlib`` HTTP and FastAPI) use the
same logic where possible; the FastAPI transport additionally
provides :mod:`fastapi` native async endpoints with the same URLs.

The 100 MiB request-body cap matches
:meth:`membrane.transport.fastapi.FastAPIServer`'s default
``client_max_body_size``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from membrane.compute.cpu import CPU
from membrane.serialization import from_dict as deserialize_fragment
from membrane.serialization import to_dict as serialize_fragment

logger = logging.getLogger(__name__)


MAX_BODY_BYTES: int = 100 << 20
"""Maximum allowed request body size in bytes (100 MiB)."""


# Type alias for the handler signature.
Handler = Callable[[Any], None]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def get_handler_node(handler: Any) -> dict[str, Any]:
    """Return ``handler.server.node`` or send a 500 and return an empty dict."""
    if not handler.server.node:
        handler.send_json(500, {"error": "no node"})
        return {}
    return handler.server.node  # type: ignore[return-value]


def get_handler_cluster(handler: Any) -> Any:
    """Return the cluster manager, or send a 503 if it is not enabled."""
    cm = handler.server.cluster_manager
    if not cm:
        handler.send_json(503, {"error": "cluster manager not enabled"})
        return None
    return cm


# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------


def store(handler: Any) -> None:
    """``POST /store``."""
    data = handler.read_json()
    node = get_handler_node(handler)
    if not node:
        return
    frag = deserialize_fragment(data["fragment"])
    is_primary = data.get("is_primary", False)
    ok = node.store(frag, is_primary=is_primary)
    handler.send_json(200 if ok else 500, {"success": ok, "content_hash": frag.content_hash})


def replicate(handler: Any) -> None:
    """``POST /replicate`` — store a fragment as a replica."""
    data = handler.read_json()
    node = get_handler_node(handler)
    if not node:
        return
    frag = deserialize_fragment(data["fragment"])
    ok = node.store(frag, is_primary=False)
    handler.send_json(200 if ok else 500, {"success": ok, "content_hash": frag.content_hash})


def retrieve(handler: Any) -> None:
    """``GET /retrieve?content_hash=...``."""
    qs = parse_qs(urlparse(handler.path).query)
    h = qs.get("content_hash", [None])[0]
    if not h:
        handler.send_json(400, {"error": "missing content_hash"})
        return
    node = get_handler_node(handler)
    if not node:
        return
    frag = node.retrieve(h)
    if frag:
        handler.send_json(200, {"found": True, "fragment": serialize_fragment(frag)})
    else:
        handler.send_json(404, {"found": False, "fragment": None})


def inventory(handler: Any) -> None:
    """``GET /inventory``."""
    node = get_handler_node(handler)
    if not node:
        return
    digest = {h: frag.version_id for h, frag in node.fragments.items()}
    handler.send_json(200, {"node_id": node.node_id, "digest": digest})


def heartbeat(handler: Any) -> None:
    """``GET /heartbeat``."""
    node = get_handler_node(handler)
    if not node:
        return
    stats = node.get_stats()
    handler.send_json(
        200,
        {
            "node_id": node.node_id,
            "load": node.heartbeat(),
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "healthy": True,
        },
    )


def metrics(handler: Any) -> None:
    """``GET /metrics`` — extended JSON metrics."""
    node = get_handler_node(handler)
    if not node:
        return
    stats = node.get_stats()
    handler.send_json(
        200,
        {
            "node_id": node.node_id,
            "memory_used_bytes": stats.memory_used_bytes,
            "memory_limit_bytes": stats.memory_limit_bytes,
            "fragment_count": stats.fragment_count,
            "primary_count": stats.primary_count,
            "load": node.heartbeat(),
        },
    )


def sync(handler: Any) -> None:
    """``POST /sync`` — pull missing fragments from a source URL."""
    data = handler.read_json()
    source_url = data.get("source_url", "")
    if not source_url:
        handler.send_json(400, {"error": "missing source_url"})
        return
    node = get_handler_node(handler)
    if not node:
        return
    try:
        with urlopen(Request(f"{source_url}/inventory"), timeout=5) as resp:  # noqa: S310
            remote_data = json.loads(resp.read().decode())
        remote_digest = remote_data.get("digest", {})
        local_digest = handler.server.transfer_service.inventory_digest(node)
        missing = handler.server.transfer_service.compare_inventories(local_digest, remote_digest)
        transferred: list[str] = []
        for h in missing:
            with urlopen(Request(f"{source_url}/retrieve?content_hash={h}"), timeout=5) as resp:  # noqa: S310
                remote_frag_data = json.loads(resp.read().decode())
            if remote_frag_data.get("found"):
                frag = deserialize_fragment(remote_frag_data["fragment"])
                if node.store(frag, is_primary=False):
                    transferred.append(h)
        handler.send_json(200, {"success": True, "transferred": transferred})
    except Exception as exc:
        logger.warning("sync failed: %s", exc)
        handler.send_json(500, {"error": "internal"})


def prefill(handler: Any) -> None:
    """``POST /prefill`` — run prefill and store fragments as primary."""
    data = handler.read_json()
    tokens = data.get("prompt_tokens", [])
    model_id = data.get("model_id", "default")
    backend = handler.server.compute_backend or CPU()
    node = get_handler_node(handler)
    if not node:
        return
    fragments = backend.prefill(tokens, model_id)
    for frag in fragments:
        node.store(frag, is_primary=True)
    handler.send_json(
        200,
        {
            "success": True,
            "fragments": [serialize_fragment(f) for f in fragments],
        },
    )


def join(handler: Any) -> None:
    """``POST /join``."""
    data = handler.read_json()
    node_id = data.get("node_id", "")
    host = data.get("host", "")
    port = data.get("port", 0)
    if not node_id or not host or not port:
        handler.send_json(400, {"error": "missing node_id, host, or port"})
        return
    cm = get_handler_cluster(handler)
    if cm is None:
        return
    result = cm.on_peer_join(node_id, host, port)
    handler.send_json(200, result)


def leave(handler: Any) -> None:
    """``POST /leave``."""
    data = handler.read_json()
    node_id = data.get("node_id", "")
    if not node_id:
        handler.send_json(400, {"error": "missing node_id"})
        return
    cm = get_handler_cluster(handler)
    if cm is None:
        return
    cm.on_peer_leave(node_id)
    handler.send_json(200, {"success": True})


def gossip(handler: Any) -> None:
    """``POST /gossip``."""
    data = handler.read_json()
    cm = get_handler_cluster(handler)
    if cm is None:
        return
    result = cm.on_gossip(data)
    handler.send_json(200, result)


def peers(handler: Any) -> None:
    """``GET /peers``."""
    cm = get_handler_cluster(handler)
    if cm is None:
        return
    peers = cm.get_peers()
    handler.send_json(200, {"peers": peers})


# ------------------------------------------------------------------
# Route table
# ------------------------------------------------------------------

ROUTES: dict[tuple[str, str], Handler] = {
    ("GET", "/retrieve"): retrieve,
    ("GET", "/inventory"): inventory,
    ("GET", "/heartbeat"): heartbeat,
    ("GET", "/metrics"): metrics,
    ("GET", "/peers"): peers,
    ("POST", "/store"): store,
    ("POST", "/replicate"): replicate,
    ("POST", "/sync"): sync,
    ("POST", "/prefill"): prefill,
    ("POST", "/join"): join,
    ("POST", "/leave"): leave,
    ("POST", "/gossip"): gossip,
}


__all__ = ["MAX_BODY_BYTES", "ROUTES"]
