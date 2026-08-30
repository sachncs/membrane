"""Merkle tree over (hash, owner_node_id) inventory entries.

Phase 5 exchanges a Merkle root alongside the Bloom filter so
receivers can locate divergent subtrees in O(log n) without
shipping the full inventory. A peer that observes a different
root asks the sender for a path-walk; the sender returns the
level-by-level hash and the receiver descends until it locates
the divergent bucket, then pulls the bucket's contents.

The tree is a complete binary tree over a sorted list of
``(payload_hash, owner_node_id)`` pairs. Leaves are
``sha256(payload_hash | "|" | owner_node_id)``. Internal nodes
are ``sha256(left | right)``. The root is therefore deterministic
and stable across processes that have observed the same set
of pairs in the same order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def _leaf_hash(payload_hash: str, owner_node_id: str) -> bytes:
    """Compute the leaf hash for a single inventory entry.

    Args:
        payload_hash: Content hash (the wire-format key).
        owner_node_id: Node that holds the fragment.

    Returns:
        bytes: 32-byte SHA-256 of the canonical (payload_hash|owner_node_id) pair.
    """
    h = hashlib.sha256()
    h.update(payload_hash.encode("utf-8"))
    h.update(b"|")
    h.update(owner_node_id.encode("utf-8"))
    return h.digest()


def _internal_hash(left: bytes, right: bytes) -> bytes:
    """Internal node hash ``sha256(left || right)``.

    Args:
        left: Left child hash.
        right: Right child hash.

    Returns:
        bytes: 32-byte SHA-256.
    """
    h = hashlib.sha256()
    h.update(left)
    h.update(right)
    return h.digest()


@dataclass(frozen=True)
class MerkleTree:
    """A balanced Merkle tree over a sorted list of inventory entries.

    Attributes:
        items: Sorted list of ``(payload_hash, owner_node_id)`` pairs.
        root: 32-byte root hash.
    """

    items: tuple[tuple[str, str], ...]
    root: bytes

    @classmethod
    def from_inventory(cls, items: list[tuple[str, str]] | None) -> MerkleTree:
        """Build a tree from a list of ``(payload_hash, owner_node_id)`` pairs.

        Args:
            items: Inventory pairs. ``None`` and empty lists are
                treated as a degenerate single-leaf tree of an
                empty string to keep the root deterministic
                across nodes that have nothing in their inventory.

        Returns:
            MerkleTree: The built tree.
        """
        if not items:
            return cls(items=(), root=_internal_hash(b"", b""))
        ordered = tuple(sorted((str(h), str(n)) for h, n in items))
        levels: list[list[bytes]] = [
            [_leaf_hash(payload_hash, owner) for payload_hash, owner in ordered]
        ]
        while len(levels[-1]) > 1:
            current = levels[-1]
            nxt: list[bytes] = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else current[i]
                nxt.append(_internal_hash(left, right))
            levels.append(nxt)
        return cls(items=ordered, root=levels[-1][0])

    def diff(self, other: MerkleTree) -> list[int]:
        """Return the indices of leaves that differ between this tree and ``other``.

        The returned indices reference ``self.items`` (sorted
        order). Both trees must have been built from the same
        underlying sort order; a length mismatch returns the
        shorter of the two ranges.

        Args:
            other: Tree to compare against.

        Returns:
            list[int]: Indices into ``self.items`` whose pair
            differs.
        """
        if self.root == other.root:
            return []
        # Pair-level diff: a leaf is "differing" if it is
        # present in one tree and missing in the other, or if
        # both have it but the (owner_node_id) disagrees. Use
        # a dict of (payload_hash -> owner_node_id) for O(1)
        # lookup. The "remove hash-b" case: hash-b is missing
        # in c, so the diff for a is hash-b; the lookup walks
        # both trees.
        theirs = dict(other.items)
        out: list[int] = []
        for i, pair in enumerate(self.items):
            h, n = pair
            if h not in theirs:
                # Removed on the other side.
                out.append(i)
            elif theirs[h] != n:
                # Same key, different owner.
                out.append(i)
        # Items added on the other side: these are not in our
        # items list so we cannot return an index; the receiver
        # has to ask the sender to walk its own leaf list. The
        # gossip layer (Phase 5.3) handles that.
        return out


__all__ = ["MerkleTree"]
