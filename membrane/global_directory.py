"""GlobalDirectory: resolves fragment locations and returns optimal node sets."""

import logging

logger = logging.getLogger(__name__)


from membrane.fragmentation_engine import compute_content_hash
from membrane.membrane_node import MembraneNode


class GlobalDirectory:
    """Routing plane registry that tracks nodes and fragment placements.

    Supports locating fragments and ranking nodes by reconstruction coverage.
    """

    def __init__(self) -> None:
        """Initialize an empty directory."""
        """Initialize an empty directory."""
        logger.info("Initialized %s", self.__class__.__name__)
        self.nodes: dict[str, MembraneNode] = {}
        self.fragment_locations: dict[str, set[str]] = {}

    def register_node(self, node: MembraneNode) -> None:
        """Register a node in the global directory.

        Args:
            node: MembraneNode to register.
        """
        self.nodes[node.node_id] = node

    def unregister_node(self, node_id: str) -> bool:
        """Remove a node from the directory.

        Args:
            node_id: Node identifier to remove.

        Returns:
            True if the node was known and removed, else False.
        """
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        for locs in self.fragment_locations.values():
            locs.discard(node_id)
        return True

    def record_fragment_location(self, content_hash: str, node_id: str) -> None:
        """Record that a fragment is stored on a node.

        Args:
            content_hash: Fragment content hash.
            node_id: Node holding the fragment.
        """
        self.fragment_locations.setdefault(content_hash, set()).add(node_id)

    def locate_fragment(self, content_hash: str) -> set[str]:
        """Return node IDs that hold the given fragment.

        Args:
            content_hash: Hash to look up.

        Returns:
            Set of node identifiers. Empty if unknown.
        """
        return set(self.fragment_locations.get(content_hash, set()))

    def optimal_nodes_for_reconstruction(
        self,
        prompt_tokens: list[int],
        model_id: str,
        k: int = 3,
        max_prefix_attempts: int = 128,
    ) -> list[str]:
        """Rank nodes by estimated fragment coverage for a prompt.

        Scans a bounded number of prefix lengths to avoid O(L²) hash
        computation cost for long prompts.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            k: Maximum number of nodes to return.
            max_prefix_attempts: Cap on prefix lengths scanned.

        Returns:
            Top-k node IDs ordered by descending coverage estimate.
        """
        if not self.nodes:
            return []

        length = len(prompt_tokens)
        step = max(1, length // max_prefix_attempts) if max_prefix_attempts > 0 else 1
        prefix_hashes = {
            compute_content_hash(tuple(prompt_tokens[:i]))
            for i in range(1, length + 1, step)
        }

        scores: dict[str, int] = {}
        for node_id, node in self.nodes.items():
            score = 0
            for h in prefix_hashes:
                if node.retrieve(h) is not None:
                    score += 1
            # Adjust by node health: lightly loaded nodes get a small boost
            load_penalty = int(node.heartbeat() * 10)
            scores[node_id] = max(0, score - load_penalty)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [node_id for node_id, score in ranked[:k] if score > 0]
