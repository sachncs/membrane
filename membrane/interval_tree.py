"""Interval tree: AVL-based augmented BST for interval overlap and adjacency queries.

Provides O(log n + m) overlap and adjacency lookups, plus O(log n)
insertion and deletion.  This is a reusable, self-contained data
structure.

The tree is keyed by interval ``[start, end]`` (taken from each
fragment's :class:`~membrane.structural_signature.StructuralSignature
.token_span`). Each node is augmented with the maximum ``end`` value
in its subtree so that range queries can prune entire subtrees
without visiting them.

Algorithm:
    * The tree is balanced as an AVL BST using rotation operations
      after every insert/remove.
    * Ties on ``start`` are broken by ``content_hash`` to keep the
      tree deterministic across processes (important for consistent
      gossip behavior).
    * Removal of a two-child node uses the in-order successor
      replacement strategy, updating ``node_map`` so the successor's
      old key points to the same physical node.

Complexity:
    * :meth:`insert` — O(log n) expected.
    * :meth:`remove` — O(log n) expected.
    * :meth:`find_overlapping` — O(log n + m) where ``m`` is the
      number of overlaps.
    * :meth:`find_adjacent` — O(log n + m).

Thread safety:
    The class is **not thread-safe**. AVL rotations mutate parent
    pointers; concurrent access requires external synchronization.

References:
    * Cormen et al., *Introduction to Algorithms* (3rd ed.),
      Chapter 14 (Augmenting data structures).
"""

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass

from membrane.fragment import Fragment


@dataclass
class IntervalNode:
    """Internal node for the interval tree.

    Attributes:
        fragment: The fragment stored at this node.
        start: Inclusive start position from the fragment's
            ``token_span``. Used as the BST key.
        end: Inclusive end position. Stored explicitly to avoid
            re-reading the structural signature on every visit.
        max_end: Maximum ``end`` value in this node's subtree.
            Used as the augmentation that enables subtree pruning
            during overlap and adjacency queries.
        left: Left child, or ``None``.
        right: Right child, or ``None``.
        height: Height of the subtree rooted at this node.
            Updated by :meth:`IntervalTree.update` after every
            mutation. Height ``1`` corresponds to a leaf.
    """

    fragment: Fragment
    start: int
    end: int
    max_end: int
    left: "IntervalNode | None" = None
    right: "IntervalNode | None" = None
    height: int = 1

    @property
    def content_hash(self) -> str:
        """Return the fragment's ``content_hash``.

        Defined as a property for ergonomic comparisons within
        tree-traversal code.
        """
        return self.fragment.content_hash


class IntervalTree:
    """Self-balancing interval tree (AVL-based) with max-end augmentation.

    The tree provides both **overlap** queries (find all intervals
    that intersect a target range) and **adjacency** queries (find
    all intervals within a given gap of a point).

    Attributes:
        root: Root of the BST, or ``None`` when the tree is empty.
        node_map: Auxiliary mapping from ``content_hash`` to node,
            allowing O(1) lookup of an interval for deletion.
    """

    def __init__(self) -> None:
        """Initialize an empty interval tree."""
        self.root: IntervalNode | None = None
        self.node_map: dict[str, IntervalNode] = {}

    # ------------------------------------------------------------------
    # AVL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def height(node: IntervalNode | None) -> int:
        """Return the height of ``node`` (0 for a ``None`` child)."""
        return node.height if node else 0

    @staticmethod
    def max_end(node: IntervalNode | None) -> int:
        """Return the augmented max-end of ``node`` (0 for ``None``)."""
        return node.max_end if node else 0

    @classmethod
    def update(cls, node: IntervalNode) -> None:
        """Recompute ``height`` and ``max_end`` from children.

        Args:
            node: Node whose augmentation fields need refreshing
                after a structural change.
        """
        node.height = 1 + max(cls.height(node.left), cls.height(node.right))
        node.max_end = max(
            node.end,
            cls.max_end(node.left),
            cls.max_end(node.right),
        )

    @classmethod
    def rotate_right(cls, y: IntervalNode) -> IntervalNode:
        """Perform a right rotation around ``y``.

        Used to restore AVL balance after an insertion or deletion
        that left the left subtree too tall.

        Args:
            y: Pivot node; must have a non-None left child.

        Returns:
            IntervalNode: The new root of the rotated subtree
            (originally ``y.left``).

        Raises:
            ValueError: If ``y.left`` is ``None``.
        """
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
        """Perform a left rotation around ``x``.

        Used to restore AVL balance after an insertion or deletion
        that left the right subtree too tall.

        Args:
            x: Pivot node; must have a non-None right child.

        Returns:
            IntervalNode: The new root of the rotated subtree
            (originally ``x.right``).

        Raises:
            ValueError: If ``x.right`` is ``None``.
        """
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
        """Return the AVL balance factor of ``node``.

        The balance factor is ``height(left) - height(right)``. A
        balanced tree has a balance factor in ``{-1, 0, 1}`` at
        every node.

        Args:
            node: Node to inspect (may be ``None``).

        Returns:
            int: Balance factor, ``0`` for a ``None`` node.
        """
        if node is None:
            return 0
        return cls.height(node.left) - cls.height(node.right)

    @classmethod
    def rebalance(cls, node: IntervalNode) -> IntervalNode:
        """Restore AVL balance at ``node``.

        Performs single or double rotations as needed based on the
        balance factor. If the tree is already balanced the node is
        returned unchanged.

        Args:
            node: Potentially unbalanced node.

        Returns:
            IntervalNode: The (possibly new) root of the rotated
            subtree.

        Raises:
            ValueError: If internal invariants about child
                presence are violated.
        """
        bf = cls.balance_factor(node)
        if bf > 1:
            # Left-heavy: single or left-right rotation.
            if node.left is None:
                raise ValueError("rebalance: left child missing with bf > 1")
            if cls.balance_factor(node.left) < 0:
                node.left = cls.rotate_left(node.left)
            return cls.rotate_right(node)
        if bf < -1:
            # Right-heavy: single or right-left rotation.
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
        """Insert ``fragment`` into the tree.

        Args:
            fragment: Fragment whose ``token_span`` becomes the
                interval key. If a node with the same
                ``content_hash`` already exists it is overwritten
                via recursive descent.
        """
        self.root = self.insert_node(self.root, fragment)

    def insert_node(self, node: IntervalNode | None, fragment: Fragment) -> IntervalNode:
        """Recursive insertion helper.

        Args:
            node: Current subtree root (may be ``None``).
            fragment: Fragment to insert.

        Returns:
            IntervalNode: The (possibly new) root of the modified
            subtree.
        """
        start, end = fragment.structural_signature.token_span
        if node is None:
            # Base case: create the leaf node and register it in
            # the auxiliary map for O(1) deletion.
            n = IntervalNode(
                fragment=fragment, start=start, end=end, max_end=end
            )
            self.node_map[fragment.content_hash] = n
            return n
        # BST descent on (start, content_hash) for determinism.
        if start < node.start:
            node.left = self.insert_node(node.left, fragment)
        elif start > node.start:
            node.right = self.insert_node(node.right, fragment)
        else:
            # Same start: decide by content_hash to keep tree
            # deterministic across processes.
            if fragment.content_hash < node.content_hash:
                node.left = self.insert_node(node.left, fragment)
            else:
                node.right = self.insert_node(node.right, fragment)
        # Refresh augmentation and rebalance on the way back up.
        self.update(node)
        return self.rebalance(node)

    def remove(self, content_hash: str) -> bool:
        """Remove the fragment with ``content_hash`` from the tree.

        Uses :attr:`node_map` for an O(1) lookup of the target
        node's ``start`` position, then delegates the recursive
        removal.

        Args:
            content_hash: Identifier of the fragment to remove.

        Returns:
            bool: True if the fragment was present and removed,
            False otherwise.
        """
        target = self.node_map.get(content_hash)
        if target is None:
            return False
        self.root = self.remove_node(self.root, target.start, content_hash)
        del self.node_map[content_hash]
        return True

    def remove_node(
        self, node: IntervalNode | None, start: int, content_hash: str
    ) -> IntervalNode | None:
        """Recursive removal helper.

        Implements the standard BST-delete algorithm with in-order
        successor replacement for two-child nodes.

        Args:
            node: Current subtree root (may be ``None``).
            start: ``token_span`` start of the target fragment.
            content_hash: Identifier of the target fragment.

        Returns:
            IntervalNode | None: The (possibly new) root of the
            modified subtree.
        """
        if node is None:
            return None
        if content_hash == node.content_hash:
            # Found the node. Handle the three standard cases.
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # Two children: replace with in-order successor.
            successor = self.min_value_node(node.right)
            node.fragment = successor.fragment
            node.start = successor.start
            node.end = successor.end
            # Point the successor's old content_hash at this node
            # so subsequent remove() calls find the right target.
            self.node_map[successor.content_hash] = node
            node.right = self.remove_node(node.right, successor.start, successor.content_hash)
        elif (start, content_hash) < (node.start, node.content_hash):
            node.left = self.remove_node(node.left, start, content_hash)
        else:
            node.right = self.remove_node(node.right, start, content_hash)
        # Refresh augmentation and rebalance on the way back up.
        self.update(node)
        return self.rebalance(node)

    @classmethod
    def min_value_node(cls, node: IntervalNode) -> IntervalNode:
        """Return the leftmost node in the subtree rooted at ``node``.

        Used to find the in-order successor when removing a
        two-child node.

        Args:
            node: Root of the subtree to search.

        Returns:
            IntervalNode: The leftmost descendant.
        """
        current = node
        while current.left is not None:
            current = current.left
        return current

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_overlapping(self, start: int, end: int) -> list[Fragment]:
        """Return all fragments whose ``token_span`` overlaps ``[start, end]``.

        Args:
            start: Query start position (inclusive).
            end: Query end position (inclusive).

        Returns:
            list[Fragment]: Overlapping fragments in traversal
            order. Empty when no fragment overlaps.
        """
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
        """Recursive overlap search with subtree pruning.

        Pruning rules:
            * The left subtree can be skipped entirely when its
              ``max_end < start`` (no interval in it can overlap).
            * The right subtree can be skipped when the current
              node's ``start > end`` (no right descendant's start
              can be small enough to overlap).

        Args:
            node: Current subtree root.
            start: Query start (inclusive).
            end: Query end (inclusive).
            results: Output list, mutated in place.
        """
        if node is None:
            return
        # Prune left subtree if its max_end < start.
        if node.left is not None and node.left.max_end >= start:
            cls.find_overlapping_recursive(node.left, start, end, results)
        # Check current node.
        if node.start <= end and node.end >= start:
            results.append(node.fragment)
        # Always search right if current start <= end.
        if node.start <= end:
            cls.find_overlapping_recursive(node.right, start, end, results)

    def find_adjacent(self, position: int, max_gap: int = 0) -> list[Fragment]:
        """Return fragments adjacent to ``position`` (within ``max_gap``).

        A fragment is adjacent when either:

        * its ``start`` lies in ``(position, position + max_gap]``
          (it begins shortly after the position), or
        * its ``end`` lies in ``[position - max_gap, position)``
          (it ends shortly before the position).

        Args:
            position: Reference token position.
            max_gap: Maximum acceptable gap in tokens. ``0`` means
                only fragments that touch ``position`` are returned.

        Returns:
            list[Fragment]: Adjacent fragments in traversal order.
        """
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
        """Recursive adjacency search with subtree pruning.

        Args:
            node: Current subtree root.
            position: Reference token position.
            max_gap: Maximum gap in tokens.
            results: Output list, mutated in place.
        """
        if node is None:
            return
        # Check adjacency: fragment starts after position within gap.
        gap_after = node.start - position
        if 0 <= gap_after <= max_gap:
            results.append(node.fragment)
        # Or fragment ends before position within gap.
        gap_before = position - node.end
        if 0 <= gap_before <= max_gap:
            results.append(node.fragment)
        # Prune subtrees:
        # - left subtree may contain fragments ending near position
        # - right subtree may contain fragments starting near position
        if node.left is not None and node.left.max_end >= position - max_gap:
            cls.find_adjacent_recursive(node.left, position, max_gap, results)
        if node.start <= position + max_gap:
            cls.find_adjacent_recursive(node.right, position, max_gap, results)
