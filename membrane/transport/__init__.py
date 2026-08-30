"""Transport layer for Membrane inter-node communication.

This package groups the wire-protocol implementations that
expose a Membrane node's API over the network:

* :class:`~membrane.transport.http.HTTPServer` — minimal
  stdlib HTTP server.
* :class:`~membrane.transport.fastapi.FastAPIServer` — FastAPI-
  based HTTP server.
* :class:`~membrane.transport.grpc.GrpcServer` — gRPC server based
  on the generated ``membrane.proto``.

All transports speak the same logical surface (store, retrieve,
inventory, heartbeat, gossip, replicate) so clients can be
swapped without changing application code.
"""

from typing import Protocol, runtime_checkable

from membrane.transport.grpc import GrpcServer
from membrane.transport.http import HTTPServer

__all__ = ["GrpcServer", "HTTPServer", "Transport"]


@runtime_checkable
class Transport(Protocol):
    """Polymorphic surface every wire-protocol server implements.

    Used by ``Server`` to compose with whichever transport the
    caller picks via the ``transport`` constructor argument; the
    implementation can be swapped without touching the rest of
    the application.

    Attributes:
        host: Bind address (informational; ``Server.start`` reads
            its own ``host`` argument).
        port: Listen port.
    """

    host: str
    port: int

    def start(self) -> None:
        """Start the transport in the foreground or as a background task."""
        ...

    def stop(self) -> None:
        """Stop the transport cleanly."""
        ...
