"""Write-behind persistence: Redis writes leave the node's lock and hot path.

:class:`~membrane.node.Node` calls its persistence hooks while holding
its lock, so a synchronous Redis round trip there stalls every store,
retrieve, and eviction on the node. :class:`PersistenceWriter` makes
the hooks enqueue and return. A single background thread applies the
operations in order, which keeps a ``store`` and a later ``forget`` of
the same fragment correctly ordered.

Failure behaviour:

* A failed write is retried with capped exponential backoff for as long
  as the writer runs. The queue absorbs a Redis outage, and writes
  resume when Redis comes back.
* When the queue is full, the new operation is dropped, logged, and
  counted in ``membrane_persistence_dropped_total``. Redis is a recovery
  aid: the in-memory node stays authoritative.
* :meth:`PersistenceWriter.stop` flushes within a deadline. Operations
  still queued after it are counted as dropped.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from membrane.fragment import Fragment
from membrane.metrics import PersistenceMetrics

logger = logging.getLogger(__name__)

type OperationKind = Literal["store", "forget"]


@dataclass(frozen=True, slots=True)
class PersistenceOperation:
    """One queued write.

    Attributes:
        kind: ``"store"`` or ``"forget"``.
        content_hash: Content hash of the fragment.
        fragment: The fragment, for ``store``.
        is_primary: Whether this node owns the primary copy, for ``store``.
    """

    kind: OperationKind
    content_hash: str
    fragment: Fragment | None = None
    is_primary: bool = False


class PersistenceWriter:
    """Background, ordered, bounded writer in front of a persistence backend.

    Attributes:
        persistence: Backend exposing ``store_fragment`` and ``forget_on_node``.
        node_id: Node whose fragment set is written.
        capacity: Maximum queued operations before new ones are dropped.
        initial_backoff_sec: First retry delay after a failed write.
        max_backoff_sec: Cap on the retry delay.
    """

    def __init__(
        self,
        persistence: Any,
        node_id: str,
        metrics: PersistenceMetrics | None = None,
        capacity: int = 10_000,
        initial_backoff_sec: float = 0.05,
        max_backoff_sec: float = 2.0,
    ) -> None:
        """Create a stopped writer; it starts on the first enqueue or :meth:`start`.

        Args:
            persistence: Backend exposing ``store_fragment`` and
                ``forget_on_node`` (both return a truthy value on success).
            node_id: Node whose fragment set is written.
            metrics: Metrics updated with queue depth and drops.
            capacity: Maximum queued operations.
            initial_backoff_sec: First retry delay after a failed write.
            max_backoff_sec: Cap on the retry delay.
        """
        self.persistence = persistence
        self.node_id = node_id
        self.capacity = capacity
        self.initial_backoff_sec = initial_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.__metrics = metrics
        self.__queue: queue.Queue[PersistenceOperation] = queue.Queue(maxsize=capacity)
        self.__stopping = threading.Event()
        self.__thread: threading.Thread | None = None
        self.__start_lock = threading.Lock()

    @property
    def pending(self) -> int:
        """Operations queued or in flight."""
        return self.__queue.unfinished_tasks

    def start(self) -> None:
        """Start the background thread (idempotent)."""
        with self.__start_lock:
            if self.__thread is not None and self.__thread.is_alive():
                return
            self.__stopping.clear()
            self.__thread = threading.Thread(target=self.__run, daemon=True, name="membrane-persistence")
            self.__thread.start()

    def store(self, fragment: Fragment, is_primary: bool) -> None:
        """Queue a fragment write; never blocks.

        Args:
            fragment: The fragment.
            is_primary: Whether this node owns the fragment's primary copy.
        """
        self.__enqueue(PersistenceOperation("store", fragment.identity.payload_hash, fragment, is_primary))

    def forget(self, content_hash: str) -> None:
        """Queue removal of a fragment from this node's persisted set; never blocks.

        Args:
            content_hash: Content hash of the fragment.
        """
        self.__enqueue(PersistenceOperation("forget", content_hash))

    def flush(self, timeout_sec: float = 10.0) -> bool:
        """Wait until every queued operation has been applied.

        Args:
            timeout_sec: Maximum seconds to wait.

        Returns:
            bool: True when the queue drained within the timeout.
        """
        deadline = time.monotonic() + timeout_sec
        while self.__queue.unfinished_tasks:
            if time.monotonic() >= deadline or self.__thread is None or not self.__thread.is_alive():
                return False
            time.sleep(0.01)
        return True

    def stop(self, timeout_sec: float = 10.0) -> bool:
        """Flush within ``timeout_sec``, then stop the thread.

        Args:
            timeout_sec: Flush budget in seconds.

        Returns:
            bool: True when nothing was left unwritten.
        """
        flushed = self.flush(timeout_sec)
        self.__stopping.set()
        if self.__thread is not None:
            self.__thread.join(timeout=max(1.0, self.max_backoff_sec))
        left = 0
        while True:
            try:
                operation = self.__queue.get_nowait()
            except queue.Empty:
                break
            self.__queue.task_done()
            self.__count_drop(operation.kind)
            left += 1
        if left:
            logger.warning("persistence writer stopped with %s unwritten operations on %s", left, self.node_id)
        self.__set_depth()
        return flushed and left == 0

    def __enqueue(self, operation: PersistenceOperation) -> None:
        """Queue ``operation``, dropping it when the queue is full.

        Args:
            operation: The write to queue.
        """
        self.start()
        try:
            self.__queue.put_nowait(operation)
        except queue.Full:
            self.__count_drop(operation.kind)
            logger.warning(
                "persistence queue full (%s); dropped %s of %s on %s",
                self.capacity,
                operation.kind,
                operation.content_hash,
                self.node_id,
            )
        self.__set_depth()

    def __run(self) -> None:
        """Apply queued operations in order until stopped."""
        while not self.__stopping.is_set():
            try:
                operation = self.__queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.__apply_with_retry(operation)
            finally:
                self.__queue.task_done()
                self.__set_depth()

    def __apply_with_retry(self, operation: PersistenceOperation) -> None:
        """Apply ``operation``, retrying with backoff until it succeeds or the writer stops.

        Args:
            operation: The write to apply.
        """
        backoff = self.initial_backoff_sec
        attempt = 0
        while True:
            attempt += 1
            if self.__apply(operation):
                if attempt > 1:
                    logger.info("persistence write recovered after %s attempts on %s", attempt, self.node_id)
                return
            if attempt == 1:
                logger.warning(
                    "persistence %s of %s failed on %s; retrying", operation.kind, operation.content_hash, self.node_id
                )
            if self.__stopping.wait(backoff):
                self.__count_drop(operation.kind)
                return
            backoff = min(backoff * 2, self.max_backoff_sec)

    def __apply(self, operation: PersistenceOperation) -> bool:
        """Apply one operation against the backend.

        Args:
            operation: The write to apply.

        Returns:
            bool: True on success.
        """
        try:
            match operation:
                case PersistenceOperation(kind="store", fragment=Fragment() as fragment):
                    ok = bool(self.persistence.store_fragment(fragment, self.node_id, operation.is_primary))
                case PersistenceOperation(kind="forget"):
                    ok = self.persistence.forget_on_node(operation.content_hash, self.node_id) is not False
                case _:
                    ok = False
        except Exception as exc:
            logger.debug("persistence %s raised: %s", operation.kind, exc)
            ok = False
        if self.__metrics is not None:
            self.__metrics.operations.inc(kind=operation.kind, outcome="ok" if ok else "error")
        return ok

    def __count_drop(self, kind: OperationKind) -> None:
        """Count a dropped operation.

        Args:
            kind: Operation kind.
        """
        if self.__metrics is not None:
            self.__metrics.dropped.inc(kind=kind)

    def __set_depth(self) -> None:
        """Publish the current queue depth."""
        if self.__metrics is not None:
            self.__metrics.queue_depth.set(float(self.__queue.qsize()))


__all__ = ["OperationKind", "PersistenceOperation", "PersistenceWriter"]
