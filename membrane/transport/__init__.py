"""Transport layer for Membrane inter-node communication.

This package groups the wire-protocol implementations that
expose a Membrane node's API over the network:

* :class:`~membrane.transport.http_server.HTTPServer` — minimal
  stdlib HTTP server.
* :class:`~membrane.transport.fastapi_server` (re-exported via
  the unified server entry point) — FastAPI-based HTTP server.
* :class:`~membrane.transport.grpc_server.GrpcServer` — gRPC
  server based on the generated ``membrane.proto``.

All transports speak the same logical surface (store, retrieve,
inventory, heartbeat, gossip, replicate) so clients can be
swapped without changing application code.
"""

from membrane.transport.grpc_server import GrpcServer
from membrane.transport.http_server import HTTPServer

__all__ = ["GrpcServer", "HTTPServer"]
