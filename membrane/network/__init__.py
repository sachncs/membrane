"""Network layer for Membrane peer-to-peer cluster management.

This package groups the network-facing components that connect
Membrane nodes together:

* :class:`~membrane.network.cluster_manager.ClusterManager` —
  high-level peer lifecycle (join, leave, discovery).
* :class:`~membrane.network.config.ClusterConfig` — declarative
  configuration for cluster endpoints.
* :class:`~membrane.network.gossip_state.GossipState` —
  eventually-consistent state propagation via gossip.
* :class:`~membrane.network.peer_client.PeerClient` —
  request/response transport to a specific peer.
* :class:`~membrane.network.remote_transfer.RemoteTransferService`
  — remote fragment transfer between nodes.

The public API of the package is the union of these classes; all
submodules are implementation details and should not be imported
directly by callers outside the package.
"""

from membrane.network.cluster import ClusterManager, PeerInfo
from membrane.network.config import ClusterConfig
from membrane.network.gossip import GossipState, PeerEndpoint
from membrane.network.peer import PeerClient
from membrane.network.transfer import RemoteTransferService

__all__ = [
    "ClusterConfig",
    "ClusterManager",
    "GossipState",
    "PeerClient",
    "PeerEndpoint",
    "PeerInfo",
    "RemoteTransferService",
]
