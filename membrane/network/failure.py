"""FailureDetector loop: mark suspect peers and remove failed peers.

Uses a pluggable :class:`~membrane.network.strategy.FailureDetector`
strategy to decide when a peer has failed. The default
:class:`ThresholdDetector` preserves the historical single-node
threshold behavior; :class:`QuorumDetector` requires majority votes
from healthy peers.
"""

from __future__ import annotations

import logging
import threading

from membrane.network.config import ClusterConfig
from membrane.network.membership import Membership
from membrane.network.strategy import FailureDetector, ThresholdDetector

logger = logging.getLogger(__name__)


class Failure:
    """Background failure-detection loop.

    Args:
        membership: Cluster membership table.
        config: Cluster configuration.
        stop_event: Stop signal shared across all cluster loops.
        running: Mutable bool flag; ``False`` exits the loop.
        detector: :class:`FailureDetector` strategy. Defaults to
            :class:`ThresholdDetector` for backward compatibility.
    """

    def __init__(
        self,
        membership: Membership,
        config: ClusterConfig,
        stop_event: threading.Event,
        running: list[bool],
        detector: FailureDetector | None = None,
    ) -> None:
        self.membership = membership
        self.config = config
        self.stop_event = stop_event
        self.running = running
        self.detector = detector or ThresholdDetector(
            failure_remove_threshold=config.failure_remove_threshold
        )

    def loop(self) -> None:
        """Iterate peers; mark suspect or remove per the strategy."""
        while self.running[0] and not self.stop_event.is_set():
            to_remove: list[str] = []
            suspect_threshold = getattr(self.config, "failure_suspect_threshold", 2)
            for p in self.membership.snapshot():
                if p.missed_heartbeats >= self.config.failure_remove_threshold:
                    if self.detector.should_remove(
                        peer_id=p.node_id,
                        peer_missed=p.missed_heartbeats,
                        suspect_votes=0,  # single-node observer; quorum impl wires votes separately
                        healthy_peer_count=len(self.membership.healthy()) + 1,
                    ):
                        to_remove.append(p.node_id)
                elif p.missed_heartbeats >= suspect_threshold:
                    if self.membership.mark_suspect(p.node_id):
                        logger.warning("Peer %s is now suspect", p.node_id)
            for node_id in to_remove:
                logger.warning("Removing failed peer %s", node_id)
                self.membership.remove(node_id)
            self.stop_event.wait(timeout=self.config.heartbeat_interval_sec)


__all__ = ["Failure"]
