"""Data-integrity gauge (Phase 3.2.3).

The v3.0.0 release surfaces two new integrity signals that the
v2.0 release computed internally but never reported:

* ``membrane_merkle_drift_size`` (Gauge, labeled by node): the
  number of inventory leaves that differ between this node's
  :class:`~membrane.merkle.MerkleTree` and the peer's tree at
  the most recent gossip exchange. A non-zero value means the
  cluster has a drift to repair; the producer of the
  :func:`membrane.network.lag.record_replication_lag` snapshot
  also calls :func:`record_merkle_drift` so dashboards see
  both signals in lockstep.
* ``membrane_corrupt_payloads_total`` (Counter): every
  :class:`~membrane.errors.CorruptPayloadError` raised by
  :func:`membrane.canonical.parse_canonical` increments the
  counter; the
  :class:`~membrane.transport.metrics.record_corrupt_payload`
  helper is the single call site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from membrane.errors import CorruptPayloadError
from membrane.merkle import MerkleTree
from membrane.metrics import MetricsCollector as _Registry

logger = logging.getLogger(__name__)


MERKLE_DRIFT_GAUGE: str = "membrane_merkle_drift_size"
"""Per-node gauge for the number of divergent Merkle leaves."""

CORRUPT_PAYLOAD_COUNTER: str = "membrane_corrupt_payloads_total"
"""Counter incremented for every CorruptPayloadError raised."""


@dataclass(frozen=True)
class MerkleDrift:
    """Outcome of :func:`measure_merkle_drift`.

    Attributes:
        drift_size: Number of leaves that differ between the
            local and remote trees.
        local_root: Hex digest of the local Merkle root.
        remote_root: Hex digest of the remote Merkle root.
    """

    drift_size: int
    local_root: str
    remote_root: str


def measure_merkle_drift(
    local: MerkleTree, remote: MerkleTree
) -> MerkleDrift:
    """Compute the drift between two Merkle trees.

    Args:
        local: The local node's tree.
        remote: The remote node's tree.

    Returns:
        MerkleDrift: Size of the symmetric-difference set and
        the hex digests of both roots.
    """
    if local.root == remote.root:
        return MerkleDrift(
            drift_size=0, local_root=local.root.hex(), remote_root=remote.root.hex()
        )
    diff = set(local.diff(remote))
    return MerkleDrift(
        drift_size=len(diff),
        local_root=local.root.hex(),
        remote_root=remote.root.hex(),
    )


def record_merkle_drift(
    registry: _Registry,
    drift: MerkleDrift,
    peer_node_id: str,
) -> None:
    """Record the drift size on the named peer gauge.

    Args:
        registry: The Prometheus registry.
        drift: The drift measurement.
        peer_node_id: Peer identifier used as the label.
    """
    gauge_name = f"{MERKLE_DRIFT_GAUGE}:{peer_node_id}"
    registry.gauge(
        gauge_name,
        "Inventory leaves that diverge from this peer's tree at last gossip.",
    ).set(float(drift.drift_size))


def record_corrupt_payload(registry: _Registry | None) -> None:
    """Increment the global corrupt-payload counter.

    Args:
        registry: The Prometheus registry. ``None`` skips the
        increment so single-node tests without a registry
        continue to work.
    """
    if registry is None:
        return
    counter = registry.counter(
        CORRUPT_PAYLOAD_COUNTER,
        "Total CorruptPayloadError raised by parse_canonical.",
    )
    counter.inc()


def record_corrupt_from_exception(registry: _Registry | None, exc: BaseException) -> None:
    """Increment the corrupt-payload counter if ``exc`` is the right type.

    Args:
        registry: The Prometheus registry.
        exc: The exception to inspect.
    """
    if isinstance(exc, CorruptPayloadError):
        record_corrupt_payload(registry)


__all__ = [
    "CORRUPT_PAYLOAD_COUNTER",
    "MERKLE_DRIFT_GAUGE",
    "MerkleDrift",
    "measure_merkle_drift",
    "record_corrupt_from_exception",
    "record_corrupt_payload",
    "record_merkle_drift",
]
