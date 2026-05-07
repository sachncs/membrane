"""ClusterReplicator: replicate entire graph clusters instead of isolated items."""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService


class ClusterReplicator:
    """Replicates connected components of fragments to target nodes."""

    def __init__(self, transfer_service: TransferService | None = None) -> None:
        """Initialize with optional transfer service.

        Args:
            transfer_service: Service used for fragment movement.
        """
        """Initialize with optional transfer service.

        Args:
            transfer_service: Service used for fragment movement.
        """
        self.transfer_service = transfer_service or TransferService()

    def replicate_cluster(
        self,
        component: set[str],
        source: MembraneNode,
        targets: list[MembraneNode],
    ) -> dict[str, list[str]]:
        """Replicate all fragments in a component to target nodes.

        Args:
            component: Set of content hashes to replicate.
            source: Node holding the fragments.
            targets: Nodes to receive replicas.

        Returns:
            Map of target node_id -> list of transferred hashes.
        """
        results: dict[str, list[str]] = {}
        for target in targets:
            transferred: list[str] = []
            for h in component:
                if self.transfer_service.transfer_fragment(source, target, h):
                    transferred.append(h)
            results[target.node_id] = transferred
        return results
