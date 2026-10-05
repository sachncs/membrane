"""Wire v3: protobuf schema, chunk manifests, and resumable transfers.

Exports load on first access, so importing a light submodule such as
:mod:`membrane.wire.v3.chunks` does not pull in the generated gRPC stubs
(``grpcio`` is only in the ``disagg`` extra).
"""

import importlib
from typing import Any

LAZY_EXPORTS: dict[str, str] = {
    "AsyncWireClient": "membrane.wire.v3.aio_client",
    "CancellationToken": "membrane.wire.v3.aio_client",
    "ChunkManifest": "membrane.wire.v3.chunks",
    "CircuitBreaker": "membrane.resilience",
    "ResumableProducer": "membrane.wire.v3.resumable",
    "ResumableReceiver": "membrane.wire.v3.resumable",
    "ResumableTransfer": "membrane.wire.v3.resumable",
    "ResumeCursor": "membrane.wire.v3.resumable",
    "RetryPolicy": "membrane.resilience",
    "WireBulkhead": "membrane.wire.v3.aio_client",
    "compute_backoff": "membrane.resilience",
    "iter_chunks": "membrane.wire.v3.resumable",
    "sha256_hex": "membrane.wire.v3.chunks",
    "with_deadline": "membrane.wire.v3.aio_client",
}
LAZY_MODULES = frozenset({"wire_v3_pb2", "wire_v3_pb2_grpc"})


def __getattr__(name: str) -> Any:
    """Load a public export or generated module on first access.

    Args:
        name: Attribute requested from the package.

    Returns:
        Any: The exported object or module.

    Raises:
        AttributeError: When ``name`` is not a public export.
    """
    if name in LAZY_MODULES:
        value: Any = importlib.import_module(f"membrane.wire.v3.{name}")
    elif name in LAZY_EXPORTS:
        value = getattr(importlib.import_module(LAZY_EXPORTS[name]), name)
    else:
        raise AttributeError(f"module 'membrane.wire.v3' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's public names, including not-yet-loaded exports.

    Returns:
        list[str]: Sorted attribute names.
    """
    return sorted({*globals(), *LAZY_EXPORTS, *LAZY_MODULES})


__all__ = [*sorted(LAZY_EXPORTS), "wire_v3_pb2", "wire_v3_pb2_grpc"]
