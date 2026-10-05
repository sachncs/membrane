"""Quorum fan-out for strong / quorum-consistency writes.

When :func:`~membrane.transport.ops.op_store` accepts a ``strong`` or
``quorum`` write it blocks until enough replicas acknowledge it or the
``cluster_quorum_timeout_sec`` budget elapses.

:class:`QuorumReplicator` does the fan-out:

* every replica is contacted **in parallel** on one shared, bounded
  thread pool (no pool is created per write);
* it returns as soon as the quorum is reached or the deadline passes,
  and never waits for stragglers;
* the deadline is published to the peer client through
  :data:`peer_deadline`, so a slow replica stops retrying once the write
  has been decided.

The caller removes the local copy when the quorum is not met, so the
cluster never keeps a write it did not acknowledge.
"""

import concurrent.futures
import contextvars
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.network.peer import Peer, peer_deadline
from membrane.replication import replicate_fragment

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


class QuorumReplicator:
    """Parallel, deadline-bounded replica fan-out on a shared thread pool.

    Attributes:
        max_workers: Upper bound on concurrent replica calls across all
            writes on this node.
    """

    def __init__(self, max_workers: int = 32) -> None:
        """Create the replicator; the pool is created on first use.

        Args:
            max_workers: Upper bound on concurrent replica calls.
        """
        self.max_workers = max_workers
        self.__pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.__lock = threading.Lock()

    def pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the shared pool, creating it on first use.

        Returns:
            concurrent.futures.ThreadPoolExecutor: The pool.
        """
        with self.__lock:
            if self.__pool is None:
                self.__pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="membrane-quorum"
                )
            return self.__pool

    def __call__(
        self,
        fragment: Fragment,
        peers: Iterable[Peer],
        quorum_count: int,
        timeout_sec: float,
        blob: bytes | None = None,
    ) -> QuorumResult:
        """Send ``fragment`` to every peer and wait for ``quorum_count`` acks.

        A peer acknowledges only once it holds both the KV bytes and the
        metadata, so an acknowledged write survives the loss of this node.

        Args:
            fragment: The fragment to replicate (already stored locally).
            peers: Replica peers; all are contacted in parallel.
            quorum_count: Peer acknowledgements required.
            timeout_sec: Wall-clock budget for the whole fan-out.
            blob: The fragment's KV bytes; required when it has a
                ``payload_ref``.

        Returns:
            QuorumResult: Outcome and counters. Calls still in flight when
            this returns finish (or give up at the deadline) in the
            background; their results are ignored.
        """
        peer_list = list(peers)
        if quorum_count <= 0 or not peer_list:
            return QuorumResult(success=False, ack_count=0, timed_out=True, replica_count=len(peer_list))

        deadline = now() + timeout_sec
        pending: set[concurrent.futures.Future[bool]] = set()
        for peer in peer_list:
            # Each call runs in its own context carrying the deadline; a
            # Context cannot be entered by two threads at once.
            context = contextvars.copy_context()
            context.run(peer_deadline.set, deadline)
            pending.add(self.pool().submit(context.run, post_replicate, peer, fragment, blob))

        ack_count = 0
        while pending and ack_count < quorum_count:
            remaining = deadline - now()
            if remaining <= 0:
                break
            done, pending = concurrent.futures.wait(
                pending, timeout=remaining, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                try:
                    if future.result():
                        ack_count += 1
                except PeerError as exc:
                    logger.debug("quorum replica failed: %s", exc)
        for future in pending:
            future.cancel()  # only stops calls that have not started
        success = ack_count >= quorum_count
        return QuorumResult(
            success=success,
            ack_count=ack_count,
            timed_out=not success and now() >= deadline,
            replica_count=len(peer_list),
        )

    def shutdown(self) -> None:
        """Stop accepting work and release the pool's threads."""
        with self.__lock:
            if self.__pool is not None:
                self.__pool.shutdown(wait=False, cancel_futures=True)
                self.__pool = None


DEFAULT_REPLICATOR = QuorumReplicator()


def attempt_quorum_acks(
    fragment: Fragment,
    peers: Iterable[Peer],
    quorum_count: int,
    timeout_sec: float,
    blob: bytes | None = None,
) -> QuorumResult:
    """Fan out ``fragment`` on the process-wide :data:`DEFAULT_REPLICATOR`.

    Args:
        fragment: The fragment to replicate (already stored locally).
        peers: Replica peers; all are contacted in parallel.
        quorum_count: Peer acknowledgements required.
        timeout_sec: Wall-clock budget for the whole fan-out.
        blob: The fragment's KV bytes, when it has a ``payload_ref``.

    Returns:
        QuorumResult: Outcome and counters.
    """
    return DEFAULT_REPLICATOR(fragment, peers, quorum_count, timeout_sec, blob)


def now() -> float:
    """Return a monotonic timestamp in seconds.

    Returns:
        float: A monotonic timestamp in seconds.
    """
    return time.monotonic()


def post_replicate(peer: Peer, fragment: Fragment, blob: bytes | None) -> bool:
    """Send one replica write (bytes, then metadata) to ``peer``.

    Args:
        peer: Destination peer.
        fragment: The fragment.
        blob: Its KV bytes, when it has a ``payload_ref``.

    Returns:
        bool: True when the peer acknowledged.
    """
    try:
        return replicate_fragment(peer, fragment, blob)
    except Exception as exc:  # pragma: no cover - propagation is the caller's job
        raise PeerError(str(exc)) from exc


class PeerError(Exception):
    """Out-of-band error raised by :func:`post_replicate`."""


__all__ = ["DEFAULT_REPLICATOR", "PeerError", "QuorumReplicator", "QuorumResult", "attempt_quorum_acks"]
