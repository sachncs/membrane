"""Network layer for Membrane peer-to-peer cluster management.

This package groups the network-facing components that connect
Membrane nodes together:

* :class:`~membrane.network.cluster_manager.Cluster` —
  high-level peer lifecycle (join, leave, discovery).
* :class:`~membrane.network.config.ClusterConfig` — declarative
  configuration for cluster endpoints.
* :class:`~membrane.network.gossip_state.GossipState` —
  eventually-consistent state propagation via gossip.
* :class:`~membrane.network.peer_client.Peer` —
  request/response transport to a specific peer.
* :class:`~membrane.network.remote_transfer.Transfer`
  — remote fragment transfer between nodes.

The public API of the package is the union of these classes; all
submodules are implementation details and should not be imported
directly by callers outside the package.
"""

from membrane.network.cluster import Cluster, PeerInfo
from membrane.network.config import ClusterConfig
from membrane.network.gossip import GossipState, PeerEndpoint
from membrane.network.peer import Peer
from membrane.network.transfer import Transfer

__all__ = [
    "ClusterConfig",
    "Cluster",
    "GossipState",
    "Peer",
    "PeerEndpoint",
    "PeerInfo",
    "Transfer",
]
