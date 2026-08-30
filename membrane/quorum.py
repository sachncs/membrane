"""Quorum fan-out for strong / quorum-consistency writes.

When :func:`op_store` accepts a fragment with
``consistency = "strong"`` or ``consistency = "quorum"`` it must
block the caller until at least ``quorum_count`` replicas have
acknowledged the write or the cluster
``cluster_quorum_timeout_sec`` budget elapses. The actual fan-out
to peers is handled by :func:`attempt_quorum_acks`.

The implementation intentionally stays simple:

* Each peer call is a synchronous ``client.request_replicate``
  POST; the bytes ride the canonical wire path so the receiver
  accepts them via the same op_store machinery.
* The function uses a :class:`concurrent.futures.ThreadPoolExecutor`
  with a fixed-size worker pool of ``quorum_count`` threads so a
  cluster-wide surge does not consume unbounded memory.
* The :class:`QuorumResult` reports ``success`` (>= quorum_count
  acks), ``ack_count`` (the actual number), and ``timed_out``
  so the caller's failure message is accurate.

The caller (``op_store``) deletes the locally-stored fragment on
failure to keep the cluster from being left with a partial-write
footprint that gossip would otherwise propagate.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.network.peer import Peer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuorumResult:
    """Outcome of a quorum fan-out attempt.

    Attributes:
        success: ``True`` when ``ack_count >= quorum_count``.
        ack_count: Number of peers that acknowledged the write.
        timed_out: ``True`` when the fan-out budget elapsed
            before the ack count reached ``quorum_count``.
        replica_count: Number of peers contacted; useful for
            diagnostic logs.
    """

    success: bool
    ack_count: int
    timed_out: bool
    replica_count: int


def attempt_quorum_acks(
    fragment: Fragment,
    peers: Iterable[Peer],
    quorum_count: int,
    timeout_sec: float,
) -> QuorumResult:
    """Synchronously fan-out a fragment to peers and wait for `quorum_count` acks.

    Concurrent futures so a slow peer does not block the rest.

    Args:
        fragment: The fragment to replicate. The local node has
            already stored it; peers are additional replicas.
        peers: Iterable of :class:`~membrane.network.peer.Peer`
            clients. The function does not deduplicate; the
            caller is responsible for selecting which peers to
            contact based on the shard map.
        quorum_count: Number of acks required. ``quorum_count <=
            len(peers)`` (otherwise the function returns a
            QuorumResult with ``success=False`` immediately).
        timeout_sec: Total wall-clock budget.

    Returns:
        QuorumResult: Status + counters. ``success=True`` means
        the cluster reached the configured write threshold.
    """

    peer_list = list(peers)
    if quorum_count <= 0:
        # Even when the caller asks for zero acks the function
        # still records the configured replica set so that
        # error responses (telemetry, audit logs) can show which
        # peers would have been contacted.
        return QuorumResult(
            success=False,
            ack_count=0,
            timed_out=True,
            replica_count=len(peer_list),
        )
    if not peer_list:
        return QuorumResult(
            success=False, ack_count=0, timed_out=True, replica_count=0
        )

    payload = {"fragment": _wire_dict_for(fragment), "is_primary": False}
    submitted: list[concurrent.futures.Future[bool]] = []
    ack_count = 0
    timed_out = False

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(quorum_count, len(peer_list)))
    ) as pool:
        for peer in peer_list:
            submitted.append(pool.submit(_post_replicate, peer, payload))

        deadline = _now() + timeout_sec
        try:
            for future in concurrent.futures.as_completed(submitted, timeout=timeout_sec):
                remaining = max(0.0, deadline - _now())
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    if future.result(timeout=remaining):
                        ack_count += 1
                        if ack_count >= quorum_count:
                            # Done early; cancel outstanding requests
                            # so we don't pile writes on a fast
                            # deadline.
                            for outstanding in submitted:
                                if not outstanding.done():
                                    outstanding.cancel()
                            break
                except (concurrent.futures.TimeoutError, _PeerError) as exc:
                    timed_out = True
                    logger.debug("quorum peer ack errored: %s", exc)
        except concurrent.futures.TimeoutError:
            # The outer timeout fired while there were still
            # futures in flight.
            timed_out = True

        # If the deadline hit before quorum, mark timeout.
        if ack_count < quorum_count and _now() >= deadline:
            timed_out = True

    return QuorumResult(
        success=ack_count >= quorum_count,
        ack_count=ack_count,
        timed_out=timed_out,
        replica_count=len(peer_list),
    )


def _now() -> float:
    import time

    return time.monotonic()


def _post_replicate(peer: Peer, payload: dict) -> bool:
    try:
        return peer.request_replicate(_fragment_from(payload))
    except Exception as exc:  # pragma: no cover - propagation is the caller's job
        raise _PeerError(str(exc)) from exc


def _fragment_from(payload: dict) -> Fragment:
    """Reconstruct a Fragment from the wire dict carrying already-parsed bytes.

    The ``op_store`` route serializes a Fragment once and ships
    the same dict over the wire; the cluster's
    ``request_replicate`` handler accepts the dict via the
    same :func:`membrane.serialization.from_dict`. We import the
    fragment lazily to keep :mod:`membrane.quorum` independent
    of the serialization module's import cycle.
    """

    from membrane.serialization import from_dict

    return from_dict(payload["fragment"])


def _wire_dict_for(fragment: Fragment) -> dict:
    """Convert a Fragment to its v3 wire dict.

    The package-private default lives in
    :func:`membrane.serialization.to_dict`. We delegate so a
    schema-version bump here does not require a corresponding
    bump in :mod:`membrane.quorum`.
    """
    from membrane.serialization import to_dict

    return to_dict(fragment)


class _PeerError(Exception):
    """Out-of-band error raised by :func:`_post_replicate`."""


__all__ = ["QuorumResult", "attempt_quorum_acks"]
