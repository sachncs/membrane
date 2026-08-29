"""stdlib HTTP route handlers.

This module is a thin transport binding for the operations in
:mod:`membrane.transport._ops`. Each handler here adapts a
:class:`Handler` instance (which carries the server state and the
``send_json`` / ``read_json`` helpers) onto the underlying business
operations.

The :data:`ROUTES` mapping is ``(method, path) -> handler``. New
endpoints are added by writing a function below and registering it
in the table.

The 100 MiB request-body cap matches the
:class:`~membrane.transport.fastapi.FastAPIServer` default.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

from membrane.compute.cpu import CPU
from membrane.node import Node
from membrane.transport._ops import (
    MAX_BODY_BYTES,
    op_gossip,
    op_heartbeat,
    op_inventory,
    op_join,
    op_leave,
    op_metrics,
    op_peers,
    op_prefill,
    op_replicate,
    op_retrieve,
    op_store,
    op_sync,
)

logger = logging.getLogger(__name__)


# Type alias for the handler signature.
Handler = Callable[[Any], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _send(handler: Any, status_body: tuple[int, dict[str, Any]]) -> None:
    """Translate an operation's ``(status, body)`` to ``send_json``."""
    status, body = status_body
    handler.send_json(status, body)


def get_handler_node(handler: Any) -> Node | None:
    """Return ``handler.server.node`` or send a 500 and return ``None``."""
    if not handler.server.node:
        handler.send_json(500, {"error": "no node"})
    return handler.server.node  # type: ignore[return-value]


def get_handler_cluster(handler: Any) -> Any:
    """Return the cluster manager, or send a 503 if it is not enabled."""
    cm = handler.server.cluster_manager
    if not cm:
        handler.send_json(503, {"error": "cluster manager not enabled"})
        return None
    return cm


# ---------------------------------------------------------------------------
# Handlers — all delegate to membrane.transport._ops
# ---------------------------------------------------------------------------


def store(handler: Any) -> None:
    """``POST /store``."""
    data = handler.read_json()
    _send(handler, op_store(handler.server.node, data["fragment"], data.get("is_primary", False)))


def replicate(handler: Any) -> None:
    """``POST /replicate`` — store a fragment as a replica."""
    data = handler.read_json()
    _send(handler, op_replicate(handler.server.node, data["fragment"]))


def retrieve(handler: Any) -> None:
    """``GET /retrieve?content_hash=...``."""
    qs = parse_qs(urlparse(handler.path).query)
    h = qs.get("content_hash", [None])[0]
    if not h:
        handler.send_json(400, {"error": "missing content_hash"})
        return
    _send(handler, op_retrieve(handler.server.node, h))


def inventory(handler: Any) -> None:
    """``GET /inventory``."""
    _send(handler, op_inventory(handler.server.node))


def heartbeat(handler: Any) -> None:
    """``GET /heartbeat``."""
    _send(handler, op_heartbeat(handler.server.node))


def metrics(handler: Any) -> None:
    """``GET /metrics`` — extended JSON metrics."""
    _send(handler, op_metrics(handler.server.node))


def sync(handler: Any) -> None:
    """``POST /sync`` — pull missing fragments from a source URL."""
    data = handler.read_json()
    _send(handler, op_sync(handler.server.node, handler.server.transfer_service, data.get("source_url", "")))


def prefill(handler: Any) -> None:
    """``POST /prefill`` — run prefill and store fragments as primary."""
    data = handler.read_json()
    _send(
        handler,
        op_prefill(
            handler.server.node,
            handler.server.compute_backend or CPU(),
            data.get("prompt_tokens", []),
            data.get("model_id", "default"),
        ),
    )


def join(handler: Any) -> None:
    """``POST /join``."""
    data = handler.read_json()
    _send(
        handler,
        op_join(handler.server.cluster_manager, data.get("node_id", ""), data.get("host", ""), data.get("port", 0)),
    )


def leave(handler: Any) -> None:
    """``POST /leave``."""
    data = handler.read_json()
    _send(handler, op_leave(handler.server.cluster_manager, data.get("node_id", "")))


def gossip(handler: Any) -> None:
    """``POST /gossip``."""
    data = handler.read_json()
    _send(handler, op_gossip(handler.server.cluster_manager, data))


def peers(handler: Any) -> None:
    """``GET /peers``."""
    _send(handler, op_peers(handler.server.cluster_manager))


# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------

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
