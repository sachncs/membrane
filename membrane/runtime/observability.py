"""Server events and the diagnostics snapshot shown by the dashboard."""

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerEvent:
    """A single server event for dashboard logging.

    Attributes:
        timestamp: Unix time at which the event was recorded.
        level: Log level (``"info"``, ``"warn"``, ``"error"``, etc.).
        message: Human-readable description.
        node_id: Optional node identifier associated with the event.
        bytes_affected: Optional size in bytes (e.g., a transfer size).
    """

    timestamp: float
    level: str
    message: str
    node_id: str = ""
    bytes_affected: int = 0


@dataclass(frozen=True, slots=True)
class ServerDiagnostics:
    """Snapshot of server health and performance.

    Attributes:
        node_id: Identifier of the local node.
        uptime_seconds: Seconds since the server was created.
        memory_used_bytes: Bytes currently held by the node.
        memory_limit_bytes: Configured node memory cap.
        fragment_count: Number of fragments stored locally.
        primary_count: Number of fragments owned as primary.
        hit_rate: External cache hit rate (currently always ``0.0``;
            tracked outside the server).
        miss_rate: External cache miss rate.
        request_count: Cumulative request count.
        error_count: Cumulative error count.
        connected_nodes: Number of distinct peers seen.
        backend_name: Compute backend descriptor.
        redis_connected: True when the Redis backend is reachable.
        load: Local node load ratio.
    """

    node_id: str
    uptime_seconds: float
    memory_used_bytes: int
    memory_limit_bytes: int
    fragment_count: int
    primary_count: int
    hit_rate: float
    miss_rate: float
    request_count: int
    error_count: int
    connected_nodes: int
    backend_name: str
    redis_connected: bool
    load: float


class EventLog:
    """Thread-safe ring buffer of the most recent :class:`ServerEvent` records.

    Attributes:
        capacity: Maximum number of events kept; older ones are dropped.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        """Create an empty log.

        Args:
            capacity: Maximum number of events kept.
        """
        self.capacity = capacity
        self.__events: deque[ServerEvent] = deque(maxlen=capacity)
        self.__lock = threading.Lock()

    def record(self, level: str, message: str, node_id: str = "", bytes_affected: int = 0) -> ServerEvent:
        """Append an event stamped with the current time.

        Args:
            level: Event level (``info``, ``warn``, ``error``).
            message: Human-readable description.
            node_id: Node identifier.
            bytes_affected: Optional size in bytes associated with the event.

        Returns:
            ServerEvent: The recorded event.
        """
        event = ServerEvent(time.time(), level, message, node_id, bytes_affected)
        with self.__lock:
            self.__events.append(event)
        return event

    def recent(self, n: int = 20) -> list[ServerEvent]:
        """Return the last ``n`` events, oldest first.

        Args:
            n: Number of events to return.

        Returns:
            list[ServerEvent]: The last ``n`` events.
        """
        with self.__lock:
            events = list(self.__events)
        return events[-n:] if n > 0 else []

    def __len__(self) -> int:
        """Number of events currently held.

        Returns:
            int: The event count.
        """
        with self.__lock:
            return len(self.__events)


__all__ = ["EventLog", "ServerDiagnostics", "ServerEvent"]
