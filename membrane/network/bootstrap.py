"""Bootstrap: contact seed peers and join the cluster.

A small one-shot helper. Tries each configured seed in order; the
first successful join terminates the loop. The other peers are then
added via the response payload.

Idempotent: running bootstrap twice with the same seeds produces the
same membership state.
"""

from __future__ import annotations

import logging

from membrane.network.config import ClusterConfig
from membrane.network.membership import Membership
from membrane.network.peer import Peer

logger = logging.getLogger(__name__)


def bootstrap(membership: Membership, config: ClusterConfig, node_id: str, host: str, port: int) -> bool:
    """Contact each configured seed and join the cluster.

    Args:
        membership: Cluster membership to populate.
        config: Cluster configuration (seeds live in ``config.peers``).
        node_id: Local node identifier.
        host: Local node bind host.
        port: Local node bind port.

    Returns:
        bool: ``True`` if at least one seed accepted the join.
    """
    for seed in config.peers:
        try:
            client = Peer(f"http://{seed}")
            result = client.join_cluster(node_id, host, port)
            if result and result.get("success"):
                for peer in result.get("peers", []):
                    membership.add(peer["node_id"], peer["host"], peer["port"])
                logger.info("Bootstrap successful via %s", seed)
                return True
        except Exception as exc:
            logger.warning("Bootstrap failed for seed %s: %s", seed, exc)
    return False


__all__ = ["bootstrap"]
