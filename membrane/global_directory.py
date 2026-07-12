"""GlobalDirectory: resolves fragment locations and returns optimal node sets.

This module defines :class:`GlobalDirectory`, the routing-plane
registry that tracks cluster membership and the set of nodes
holding each fragment. Unlike
:class:`~membrane.distributed_directory.DistributedDirectory`
(which delegates to supernodes), this implementation stores the
membership and placement tables directly in memory.

The directory also exposes a higher-level helper,
:meth:`optimal_nodes_for_reconstruction`, which scores each
registered node by an estimate of how many of the prompt's
prefix-length sub-hashes it already holds, adjusted by a small
load penalty derived from the node's heart beat.

Thread safety:
    The class is **not thread-safe**. Provide external locking
    when sharing across threads.

Complexity:
    * :meth:`register_node`, :meth:`unregister_node`,
      :meth:`record_fragment_location`, :meth:`locate_fragment` —
      O(1) amortized.
    * :meth:`optimal_nodes_for_reconstruction` — O(N · H) where
      ``N`` is the number of registered nodes and ``H`` is the
      number of prefix hashes sampled from the prompt. The
      ``max_prefix_attempts`` cap bounds ``H``.
"""

import logging

logger = logging.getLogger(__name__)


from membrane.fragmentation_engine import compute_content_hash
from membrane.membrane_node import MembraneNode


class GlobalDirectory:
    """Routing plane registry that tracks nodes and fragment placements.

    Supports locating fragments and ranking nodes by
    reconstruction coverage.

    Attributes:
        nodes: Mapping from ``node_id`` to the registered
            :class:`~membrane.membrane_node.MembraneNode`.
        fragment_locations: Mapping from ``content_hash`` to the
            set of node IDs holding a replica.
    """

    def __init__(self) -> None:
        """Initialize an empty directory."""
        logger.info("Initialized %s", self.__class__.__name__)
        self.nodes: dict[str, MembraneNode] = {}
        self.fragment_locations: dict[str, set[str]] = {}

    def register_node(self, node: MembraneNode) -> None:
        """Register a node in the global directory.

        Args:
            node: MembraneNode to register. Indexed by its
                ``node_id`` attribute.
        """
        self.nodes[node.node_id] = node

    def unregister_node(self, node_id: str) -> bool:
        """Remove a node from the directory.

        Removes the node from the membership table and discards
        it from every ``fragment_locations`` entry so the
        directory stays consistent.

        Args:
            node_id: Node identifier to remove.

        Returns:
            bool: True if the node was known and removed,
            False otherwise.
        """
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        # Garbage-collect the node from every fragment's holder
        # set so stale entries don't accumulate.
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
            set[str]: Defensive copy of the holder set. Empty if
            the directory has no record of the hash.
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

        Samples up to ``max_prefix_attempts`` prefix lengths
        (uniformly spaced along the prompt) and hashes each one
        with :func:`~membrane.fragmentation_engine.compute_content_hash`.
        Every node that already holds at least one of these
        prefixes contributes to its score; nodes that look heavily
        loaded (high heart beat) get a small penalty.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier. Currently unused by the
                scoring logic but accepted for forward
                compatibility with model-aware ranking.
            k: Maximum number of nodes to return.
            max_prefix_attempts: Cap on the number of prefix
                lengths sampled. Prevents O(L^2) hash computation
                for long prompts.

        Returns:
            list[str]: Up to ``k`` node IDs ordered by descending
            coverage estimate. Nodes with a final score of ``0``
            are excluded.
        """
        if not self.nodes:
            return []

        length = len(prompt_tokens)
        # Step size grows with prompt length so we keep the total
        # number of sampled prefixes bounded by max_prefix_attempts.
        step = max(1, length // max_prefix_attempts) if max_prefix_attempts > 0 else 1
        prefix_hashes = {compute_content_hash(tuple(prompt_tokens[:i])) for i in range(1, length + 1, step)}

        scores: dict[str, int] = {}
        for node_id, node in self.nodes.items():
            score = 0
            for h in prefix_hashes:
                if node.retrieve(h) is not None:
                    score += 1
            # Penalize nodes that report being heavily loaded via
            # heart beat. The factor of 10 was chosen empirically
            # so the penalty is small relative to a typical
            # coverage score but still biases selection toward
            # idle nodes.
            load_penalty = int(node.heartbeat() * 10)
            scores[node_id] = max(0, score - load_penalty)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [node_id for node_id, score in ranked[:k] if score > 0]
