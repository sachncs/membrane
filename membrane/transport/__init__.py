"""Transport layer for Membrane inter-node communication.

This package groups the wire-protocol implementations that
expose a Membrane node's API over the network:

* :class:`~membrane.transport.fastapi.FastAPIServer` — the
  production HTTP transport, mTLS-aware.
* :class:`~membrane.transport.grpc.GrpcServer` — production
  gRPC server backed by ``membrane.proto``.

The pre-2.0 stdlib ``HTTPServer`` was removed in 2.0: it was
unmaintained, had no TLS story, and the FastAPI transport is the
only HTTP path supported in production. Configure the cluster
through ``Server(transport='http' | 'grpc')``.

All transports speak the same logical surface (store, retrieve,
inventory, heartbeat, gossip, replicate) so clients can be
swapped without changing application code.
"""

from typing import Protocol, runtime_checkable

from membrane.transport.grpc import GrpcServer

__all__ = ["GrpcServer", "Transport"]


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
