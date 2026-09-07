"""Heartbeat loop: periodically ping every known peer.

The loop is started by :class:`~membrane.network.cluster.Cluster` as a
daemon thread; it reads the membership via :class:`Membership` and
records heartbeat results via the same.
"""

from __future__ import annotations

import logging
import threading
import time

from membrane.network.config import ClusterConfig
from membrane.network.membership import Membership

logger = logging.getLogger(__name__)


class Heartbeat:
    """Background heartbeat loop.

    Args:
        membership: Cluster membership table.
        config: Cluster configuration.
        stop_event: Stop signal shared across all cluster loops.
        running: Mutable bool flag; ``False`` exits the loop.
    """

    def __init__(
        self,
        membership: Membership,
        config: ClusterConfig,
        stop_event: threading.Event,
        running: list[bool],
    ) -> None:
        self.membership = membership
        self.config = config
        self.stop_event = stop_event
        self.running = running

    def loop(self) -> None:
        """Periodically ping every known peer.

        On success, the peer's heartbeat counters and lease
        deadline are refreshed. The lease is computed locally
        (``now() + lease_timeout_sec``) rather than on the
        receiver so a peer that does not advertise a lease
        deadline still benefits from the local clock-driven
        grace period. On failure, the missed-heartbeat counter
        is incremented.
        """
        lease_timeout = float(getattr(self.config, "lease_timeout_sec", 30.0))
        while self.running[0] and not self.stop_event.is_set():
            now = time.time()
            deadline = now + lease_timeout
            for p in self.membership.snapshot():
                if self.stop_event.is_set():
                    return
                client = self.membership.get_client(p.node_id)
                if client is None:
                    continue
                try:
                    resp = client.heartbeat()
                    if resp:
                        self.membership.record_heartbeat(
                            p.node_id,
                            lease_until=deadline,
                        )
                except Exception as exc:
                    self.membership.record_miss(p.node_id)
                    logger.debug("Heartbeat to %s failed: %s", p.node_id, exc)
            self.membership.evict_expired_leases(now=now)
            self.stop_event.wait(timeout=self.config.heartbeat_interval_sec)


__all__ = ["Heartbeat"]
