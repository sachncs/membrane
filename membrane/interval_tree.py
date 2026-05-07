"""Interval tree: AVL-based augmented BST for interval overlap and adjacency queries.

Provides O(log n + m) overlap and adjacency lookups, plus O(log n) insertion
and deletion.  This is a reusable, self-contained data structure.
"""

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass
class IntervalNode:
    """Internal node for the interval tree."""

    fragment: Fragment
    start: int
    end: int
    max_end: int
    left: "IntervalNode | None" = None
    right: "IntervalNode | None" = None
    height: int = 1

    @property
    def content_hash(self) -> str:
        return self.fragment.content_hash


class IntervalTree:
    """Self-balancing interval tree (AVL-based) with max-end augmentation."""

    def __init__(self) -> None:
        self.root: IntervalNode | None = None
        self.node_map: dict[str, IntervalNode] = {}

    # ------------------------------------------------------------------
    # AVL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def height(node: IntervalNode | None) -> int:
        return node.height if node else 0

    @staticmethod
    def max_end(node: IntervalNode | None) -> int:
        return node.max_end if node else 0

    @classmethod
    def update(cls, node: IntervalNode) -> None:
        node.height = 1 + max(cls.height(node.left), cls.height(node.right))
        node.max_end = max(
            node.end,
            cls.max_end(node.left),
            cls.max_end(node.right),
        )

    @classmethod
    def rotate_right(cls, y: IntervalNode) -> IntervalNode:
        if y.left is None:
            raise ValueError("rotate_right called on node with no left child")
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        cls.update(y)
        cls.update(x)
        return x

    @classmethod
    def rotate_left(cls, x: IntervalNode) -> IntervalNode:
        if x.right is None:
            raise ValueError("rotate_left called on node with no right child")
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        cls.update(x)
        cls.update(y)
        return y

    @classmethod
    def balance_factor(cls, node: IntervalNode | None) -> int:
        if node is None:
            return 0
        return cls.height(node.left) - cls.height(node.right)

    @classmethod
    def rebalance(cls, node: IntervalNode) -> IntervalNode:
        bf = cls.balance_factor(node)
        if bf > 1:
            if node.left is None:
                raise ValueError("rebalance: left child missing with bf > 1")
            if cls.balance_factor(node.left) < 0:
                node.left = cls.rotate_left(node.left)
            return cls.rotate_right(node)
        if bf < -1:
            if node.right is None:
                raise ValueError("rebalance: right child missing with bf < -1")
            if cls.balance_factor(node.right) > 0:
                node.right = cls.rotate_right(node.right)
            return cls.rotate_left(node)
        return node

    # ------------------------------------------------------------------
    # Insert / remove
    # ------------------------------------------------------------------

    def insert(self, fragment: Fragment) -> None:
        self.root = self.insert_node(self.root, fragment)

    def insert_node(self, node: IntervalNode | None, fragment: Fragment) -> IntervalNode:
        start, end = fragment.structural_signature.token_span
        if node is None:
            n = IntervalNode(
                fragment=fragment, start=start, end=end, max_end=end
            )
            self.node_map[fragment.content_hash] = n
            return n
        if start < node.start:
            node.left = self.insert_node(node.left, fragment)
        elif start > node.start:
            node.right = self.insert_node(node.right, fragment)
        else:
            # Same start: decide by content_hash to keep tree deterministic
            if fragment.content_hash < node.content_hash:
                node.left = self.insert_node(node.left, fragment)
            else:
                node.right = self.insert_node(node.right, fragment)
        self.update(node)
        return self.rebalance(node)

    def remove(self, content_hash: str) -> bool:
        target = self.node_map.get(content_hash)
        if target is None:
            return False
        self.root = self.remove_node(self.root, target.start, content_hash)
        del self.node_map[content_hash]
        return True

    def remove_node(
        self, node: IntervalNode | None, start: int, content_hash: str
    ) -> IntervalNode | None:
        if node is None:
            return None
        if content_hash == node.content_hash:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # Two children: replace with in-order successor
            successor = self.min_value_node(node.right)
            node.fragment = successor.fragment
            node.start = successor.start
            node.end = successor.end
            # Update node_map entry for the successor's old hash to point to this node
            self.node_map[successor.content_hash] = node
            node.right = self.remove_node(node.right, successor.start, successor.content_hash)
        elif (start, content_hash) < (node.start, node.content_hash):
            node.left = self.remove_node(node.left, start, content_hash)
        else:
            node.right = self.remove_node(node.right, start, content_hash)
        self.update(node)
        return self.rebalance(node)

    @classmethod
    def min_value_node(cls, node: IntervalNode) -> IntervalNode:
        current = node
        while current.left is not None:
            current = current.left
        return current

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_overlapping(self, start: int, end: int) -> list[Fragment]:
        """Return all fragments whose token_span overlaps [start, end]."""
        results: list[Fragment] = []
        self.find_overlapping_recursive(self.root, start, end, results)
        return results

    @classmethod
    def find_overlapping_recursive(
        cls,
        node: IntervalNode | None,
        start: int,
        end: int,
        results: list[Fragment],
    ) -> None:
        if node is None:
            return
        # Prune left subtree if its max_end < start
        if node.left is not None and node.left.max_end >= start:
            cls.find_overlapping_recursive(node.left, start, end, results)
        # Check current node
        if node.start <= end and node.end >= start:
            results.append(node.fragment)
        # Always search right if current start <= end
        if node.start <= end:
            cls.find_overlapping_recursive(node.right, start, end, results)

    def find_adjacent(self, position: int, max_gap: int = 0) -> list[Fragment]:
        """Return fragments adjacent to position (within max_gap)."""
        results: list[Fragment] = []
        self.find_adjacent_recursive(self.root, position, max_gap, results)
        return results

    @classmethod
    def find_adjacent_recursive(
        cls,
        node: IntervalNode | None,
        position: int,
        max_gap: int,
        results: list[Fragment],
    ) -> None:
        if node is None:
            return
        # Check adjacency: fragment starts after position within gap
        gap_after = node.start - position
        if 0 <= gap_after <= max_gap:
            results.append(node.fragment)
        # Or fragment ends before position within gap
        gap_before = position - node.end
        if 0 <= gap_before <= max_gap:
            results.append(node.fragment)
        # Prune subtrees
        # Left subtree could have end >= position - max_gap
        if node.left is not None and node.left.max_end >= position - max_gap:
            cls.find_adjacent_recursive(node.left, position, max_gap, results)
        # Right subtree could have start <= position + max_gap
        if node.start <= position + max_gap:
            cls.find_adjacent_recursive(node.right, position, max_gap, results)
