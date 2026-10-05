"""Background tasks and process lifecycle for a running server.

* :class:`PeriodicTask`: a daemon thread that runs an action every
  interval and stops promptly.
* :func:`run_until_signalled`: blocks the main thread until the server
  exits or SIGTERM / SIGINT arrives, then drains the node gracefully.
"""

import logging
import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import Protocol

logger = logging.getLogger(__name__)


class PeriodicTask:
    """Run ``action`` every ``interval_sec`` on a daemon thread.

    Exceptions raised by ``action`` are logged and do not stop the task.

    Attributes:
        name: Thread name (also used in log messages).
        interval_sec: Seconds between runs.
    """

    def __init__(self, name: str, interval_sec: float, action: Callable[[], object]) -> None:
        """Create a stopped task.

        Args:
            name: Thread name.
            interval_sec: Seconds between runs; the first run happens one
                interval after :meth:`start`.
            action: Callable run each interval.
        """
        self.name = name
        self.interval_sec = interval_sec
        self.__action = action
        self.__stop = threading.Event()
        self.__thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the thread is alive."""
        return self.__thread is not None and self.__thread.is_alive()

    def start(self) -> None:
        """Start the thread (idempotent)."""
        if self.running:
            return
        self.__stop.clear()
        self.__thread = threading.Thread(target=self.__loop, daemon=True, name=self.name)
        self.__thread.start()

    def stop(self, timeout_sec: float = 10.0) -> bool:
        """Signal the thread and wait for it to exit.

        Args:
            timeout_sec: Maximum seconds to wait.

        Returns:
            bool: True when the thread exited (or never started).
        """
        self.__stop.set()
        if self.__thread is None:
            return True
        self.__thread.join(timeout=timeout_sec)
        if self.__thread.is_alive():
            logger.warning("%s did not exit within %.1fs", self.name, timeout_sec)
            return False
        return True

    def __loop(self) -> None:
        """Run the action every interval until stopped."""
        while not self.__stop.wait(self.interval_sec):
            try:
                self.__action()
            except Exception:
                logger.exception("%s failed", self.name)


class Drainable(Protocol):
    """What :func:`run_until_signalled` needs from a server."""

    def join(self, deadline_sec: float | None = None) -> bool:
        """Wait for the server thread.

        Args:
            deadline_sec: Maximum seconds to wait.

        Returns:
            bool: True when the server thread has exited.
        """
        ...

    def drain(self, deadline_sec: float = 30.0) -> dict[str, object]:
        """Stop accepting writes, hand off primaries, leave the cluster, and stop.

        Args:
            deadline_sec: Drain budget.

        Returns:
            dict[str, object]: Drain summary.
        """
        ...


def run_until_signalled(server: Drainable, drain_timeout_sec: float) -> None:
    """Block until the server exits or a stop signal arrives, then drain it.

    SIGTERM (sent by Kubernetes, Docker, and systemd) and SIGINT
    (Ctrl+C) both trigger :meth:`Drainable.drain`: readiness turns 503 so
    load balancers stop routing, writes get 503 + ``Retry-After``,
    primaries are handed off, and the node leaves the cluster before
    stopping. A second signal during the drain is logged and ignored;
    the drain is already bounded by ``drain_timeout_sec``.

    Args:
        server: The started server.
        drain_timeout_sec: Drain budget in seconds.
    """
    stop_requested = threading.Event()

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        name = signal.Signals(signum).name
        if stop_requested.is_set():
            logger.warning("%s received again; drain already in progress", name)
            return
        logger.info("%s received; draining (timeout %.0fs)", name, drain_timeout_sec)
        stop_requested.set()

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stop_requested.is_set():
            if server.join(deadline_sec=0.5):
                return
        summary = server.drain(deadline_sec=drain_timeout_sec)
        logger.info("drain complete: %s", summary)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


__all__ = ["Drainable", "PeriodicTask", "run_until_signalled"]
