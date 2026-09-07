"""Network layer for Membrane peer-to-peer cluster management.

This package groups the network-facing components that connect
Membrane nodes together:

* :class:`~membrane.network.cluster.Cluster` —
  high-level peer lifecycle (join, leave, discovery).
* :class:`~membrane.network.config.ClusterConfig` — declarative
  configuration for cluster endpoints.
* :class:`~membrane.network.gossip.GossipState` —
  eventually-consistent state propagation via gossip.
* :class:`~membrane.network.peer.Peer` —
  request/response transport to a specific peer.

The public API of the package is the union of these classes; all
submodules are implementation details and should not be imported
directly by callers outside the package.
"""

from membrane.network.cluster import Cluster, PeerInfo
from membrane.network.config import ClusterConfig
from membrane.network.gossip import GossipState, PeerEndpoint
from membrane.network.peer import Peer

__all__ = [
    "Cluster",
    "ClusterConfig",
    "GossipState",
    "Peer",
    "PeerEndpoint",
    "PeerInfo",
]
