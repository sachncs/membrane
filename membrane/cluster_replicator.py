"""ClusterReplicator: replicate entire graph clusters instead of isolated items.

This module defines :class:`ClusterReplicator`, a helper that
replicates a *set* of fragments (typically the contents of a
connected component discovered via
:class:`~membrane.subgraph_retrieval.SubgraphRetrieval`) from a
source node to multiple target nodes.

The replicator is intentionally thin: it delegates every fragment
move to a :class:`~membrane.transfer_service.TransferService` and
records per-target success lists. Callers can compose it with the
graph layer to push hot subgraphs to multiple regions in parallel.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.membrane_node import MembraneNode
from membrane.transfer_service import TransferService


class ClusterReplicator:
    """Replicates connected components of fragments to target nodes.

    Attributes:
        transfer_service: Service used for fragment movement.
            Held by reference so callers can substitute a custom
            implementation (e.g., one that records transfers for
            testing).
    """

    def __init__(self, transfer_service: TransferService | None = None) -> None:
        """Initialize with an optional transfer service.

        Args:
            transfer_service: Service used for fragment movement.
                A default :class:`TransferService` is created
                when ``None``.
        """
        self.transfer_service = transfer_service or TransferService()

    def replicate_cluster(
        self,
        component: set[str],
        source: MembraneNode,
        targets: list[MembraneNode],
    ) -> dict[str, list[str]]:
        """Replicate all fragments in ``component`` to each target node.

        For every target, iterates over the component and attempts
        each transfer independently. Failures are silently
        skipped and do not propagate to other targets or other
        fragments.

        Args:
            component: Set of content hashes to replicate.
            source: Node holding the fragments.
            targets: Nodes to receive replicas.

        Returns:
            dict[str, list[str]]: Mapping from ``target.node_id``
            to the list of hashes that were successfully
            transferred to that target.
        """
        results: dict[str, list[str]] = {}
        for target in targets:
            transferred: list[str] = []
            for h in component:
                if self.transfer_service.transfer_fragment(source, target, h):
                    transferred.append(h)
            results[target.node_id] = transferred
        return results
