"""Periodic background tasks and signal-driven drain."""

import logging
import os
import signal
import threading
import time

from membrane.runtime.lifecycle import PeriodicTask, run_until_signalled


def test_periodic_task_runs_survives_errors_and_stops(caplog) -> None:
    calls: list[int] = []

    def action() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("one bad pass")

    task = PeriodicTask("tick", 0.01, action)
    assert task.stop() is True  # never started
    task.start()
    task.start()  # idempotent
    deadline = time.monotonic() + 5
    while len(calls) < 4:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert task.running
    assert task.stop() is True and not task.running
    assert "tick failed" in caplog.text


def test_stop_reports_a_thread_that_will_not_exit(caplog) -> None:
    release = threading.Event()
    task = PeriodicTask("stuck", 0.001, lambda: release.wait(5))
    task.start()
    time.sleep(0.05)
    with caplog.at_level(logging.WARNING):
        assert task.stop(timeout_sec=0.05) is False
    assert "did not exit" in caplog.text
    release.set()
    assert task.stop() is True


class Server:
    def __init__(self, exits_after: float | None = None) -> None:
        self.started = time.monotonic()
        self.exits_after = exits_after
        self.drained: list[float] = []
        self.signal_during_drain = False

    def join(self, deadline_sec: float | None = None) -> bool:
        if self.exits_after is not None and time.monotonic() - self.started > self.exits_after:
            return True
        time.sleep(min(deadline_sec or 0.0, 0.05))
        return False

    def drain(self, deadline_sec: float = 30.0) -> dict[str, object]:
        self.drained.append(deadline_sec)
        if self.signal_during_drain:
            os.kill(os.getpid(), signal.SIGTERM)  # still handled: logged and ignored
            time.sleep(0.05)
        return {"ok": True}


def test_returns_when_the_server_exits_on_its_own() -> None:
    server = Server(exits_after=0.05)
    run_until_signalled(server, 5.0)
    assert server.drained == []


def test_sigterm_drains_once_and_restores_handlers(caplog) -> None:
    server = Server()
    server.signal_during_drain = True
    before = signal.getsignal(signal.SIGTERM)

    def send() -> None:
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=send).start()
    with caplog.at_level(logging.INFO):
        run_until_signalled(server, 7.0)
    assert server.drained == [7.0]
    assert "SIGTERM received; draining" in caplog.text
    assert "SIGTERM received again" in caplog.text
    assert signal.getsignal(signal.SIGTERM) is before
