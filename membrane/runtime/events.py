"""Typed server events and the bus that delivers them to subscribers.

Extensions observe the server without patching it: they subscribe to
event types on the :class:`EventBus` (directly, or through a
``membrane.hooks`` entry point; see :mod:`membrane.runtime.plugins`).

Events are published from hot paths, some while the node lock is held,
so :meth:`EventBus.publish` only enqueues. A single dispatcher thread
delivers events in publication order. A subscriber that raises is
logged and skipped, and a full queue drops the event and counts it in
:attr:`EventBus.dropped`.
"""

import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FragmentStored:
    """A fragment became resident on this node.

    Attributes:
        content_hash: Storage key (:attr:`~membrane.fragment.Fragment.key`):
            the content hash, prefixed with the tenant outside the default one.
        tenant_id: Owning tenant.
        is_primary: Whether this node owns the primary copy.
    """

    content_hash: str
    tenant_id: str
    is_primary: bool


@dataclass(frozen=True, slots=True)
class FragmentRemoved:
    """A fragment left this node (eviction, expiry, delete, or rollback).

    Attributes:
        content_hash: Storage key, as in :class:`FragmentStored`.
    """

    content_hash: str


@dataclass(frozen=True, slots=True)
class PeerJoined:
    """A peer entered the membership table.

    Attributes:
        node_id: The peer.
    """

    node_id: str


@dataclass(frozen=True, slots=True)
class PeerLeft:
    """A peer left the membership table.

    Attributes:
        node_id: The peer.
    """

    node_id: str


@dataclass(frozen=True, slots=True)
class DrainStarted:
    """This node started draining.

    Attributes:
        deadline_sec: Drain budget in seconds.
    """

    deadline_sec: float


@dataclass(frozen=True, slots=True)
class DrainFinished:
    """This node finished draining.

    Attributes:
        migrated: Primaries handed off.
        stragglers: Primaries that could not be handed off.
    """

    migrated: int
    stragglers: int


type Event = FragmentStored | FragmentRemoved | PeerJoined | PeerLeft | DrainStarted | DrainFinished
type Handler = Callable[[Event], object]


class EventBus:
    """Ordered, non-blocking delivery of :data:`Event` objects to subscribers.

    Attributes:
        capacity: Maximum queued events before new ones are dropped.
        dropped: Events dropped because the queue was full.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        """Create a bus; the dispatcher starts on the first publish.

        Args:
            capacity: Maximum queued events.
        """
        self.capacity = capacity
        self.dropped = 0
        self.__handlers: dict[type, list[Handler]] = defaultdict(list)
        self.__queue: queue.Queue[Event | None] = queue.Queue(maxsize=capacity)
        self.__lock = threading.Lock()
        self.__thread: threading.Thread | None = None

    def subscribe(self, event_type: type, handler: Handler) -> None:
        """Call ``handler(event)`` for every published ``event_type``.

        Args:
            event_type: An event class, or :class:`object` for every event.
            handler: The subscriber.
        """
        with self.__lock:
            self.__handlers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        """Queue ``event`` for delivery; never blocks.

        Args:
            event: The event.
        """
        with self.__lock:
            if not self.__handlers:
                return
            if self.__thread is None or not self.__thread.is_alive():
                self.__thread = threading.Thread(target=self.__run, daemon=True, name="membrane-events")
                self.__thread.start()
        try:
            self.__queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def flush(self, timeout_sec: float = 5.0) -> bool:
        """Wait until every queued event has been delivered.

        Args:
            timeout_sec: Maximum seconds to wait.

        Returns:
            bool: True when the queue drained in time.
        """
        deadline = time.monotonic() + timeout_sec
        while self.__queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def close(self, timeout_sec: float = 5.0) -> None:
        """Deliver what is queued, then stop the dispatcher.

        Args:
            timeout_sec: Maximum seconds to wait for delivery.
        """
        if self.__thread is None:
            return
        self.flush(timeout_sec)
        self.__queue.put(None)
        self.__thread.join(timeout=timeout_sec)
        self.__thread = None

    def __run(self) -> None:
        """Deliver queued events until :meth:`close`."""
        while True:
            event = self.__queue.get()
            try:
                if event is None:
                    return
                with self.__lock:
                    handlers = [*self.__handlers.get(type(event), ()), *self.__handlers.get(object, ())]
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception:
                        logger.exception("event handler %r failed on %s", handler, type(event).__name__)
            finally:
                self.__queue.task_done()


__all__ = [
    "DrainFinished",
    "DrainStarted",
    "Event",
    "EventBus",
    "FragmentRemoved",
    "FragmentStored",
    "Handler",
    "PeerJoined",
    "PeerLeft",
]
