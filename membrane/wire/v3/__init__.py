"""Wire v3 protobuf subpackage (Phase 3.3.1).

The :mod:`membrane.wire.v3.wire_v3.proto` definition and
generated stubs live under this package.
"""

from membrane.wire.v3 import wire_v3_pb2, wire_v3_pb2_grpc
from membrane.wire.v3.chunks import ChunkManifest, sha256_hex
from membrane.wire.v3.resumable import (
    ResumableProducer,
    ResumableReceiver,
    ResumableTransfer,
    ResumeCursor,
    iter_chunks,
)

__all__ = [
    "ChunkManifest",
    "ResumableProducer",
    "ResumableReceiver",
    "ResumableTransfer",
    "ResumeCursor",
    "iter_chunks",
    "sha256_hex",
    "wire_v3_pb2",
    "wire_v3_pb2_grpc",
]
