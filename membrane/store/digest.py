"""An inventory digest kept up to date on every store and removal.

A node's fragments are spread over :data:`BUCKETS` buckets by a hash of
their content hash. Each bucket keeps its members (hash and version) and
the XOR of a 64-bit hash of every member, so adding or removing a fragment
costs O(1). The bucket digests are a one-level Merkle tree: the root
(:meth:`InventoryDigest.root`) changes whenever the inventory does, and
comparing bucket digests shows which buckets changed.

Peers fetch the bucket digests (``GET /inventory/buckets``) and page one
bucket at a time (``GET /inventory?bucket=N``). The replicator's repair
pass skips a bucket when neither the peer's digest nor the local
expectation changed since it last verified that bucket, so a pass costs
in proportion to what changed, not to the data held. Previously every
gossip round rebuilt a Merkle tree over the whole inventory.
"""

import hashlib
import random
import threading

#: Buckets the inventory is split into.
BUCKETS = 1024


def bucket_of(content_hash: str) -> int:
    """The bucket a content hash belongs to (uniform for any hash format).

    Args:
        content_hash: Content hash.

    Returns:
        int: Bucket index in ``[0, BUCKETS)``.
    """
    return int.from_bytes(hashlib.blake2b(content_hash.encode(), digest_size=4).digest(), "big") % BUCKETS


def leaf(content_hash: str, version: int) -> int:
    """The 64-bit contribution of one fragment to its bucket digest.

    Args:
        content_hash: Content hash.
        version: Fragment version.

    Returns:
        int: Unsigned 64-bit hash.
    """
    return int.from_bytes(hashlib.blake2b(f"{content_hash}\x00{version}".encode(), digest_size=8).digest(), "big")


def fold(entries: dict[str, int]) -> int:
    """XOR the leaves of ``entries`` (what a bucket holding exactly them digests to).

    Args:
        entries: ``content_hash -> version``.

    Returns:
        int: The digest.
    """
    value = 0
    for content_hash, version in entries.items():
        value ^= leaf(content_hash, version)
    return value


class InventoryDigest:
    """Bucketed set digest of one node's inventory.

    Attributes:
        changes: Updates applied so far (a cheap change detector).
    """

    def __init__(self) -> None:
        """Start empty."""
        self.__digests = [0] * BUCKETS
        self.__members: list[dict[str, int]] = [{} for _ in range(BUCKETS)]
        self.__lock = threading.Lock()
        self.__root: tuple[int, bytes] | None = None
        self.changes = 0

    def add(self, content_hash: str, version: int) -> None:
        """Record that ``content_hash`` at ``version`` is held.

        Args:
            content_hash: Content hash.
            version: Fragment version.
        """
        bucket = bucket_of(content_hash)
        with self.__lock:
            members = self.__members[bucket]
            previous = members.get(content_hash)
            if previous == version:
                return
            if previous is not None:
                self.__digests[bucket] ^= leaf(content_hash, previous)
            members[content_hash] = version
            self.__digests[bucket] ^= leaf(content_hash, version)
            self.changes += 1

    def remove(self, content_hash: str) -> None:
        """Record that ``content_hash`` is no longer held.

        Args:
            content_hash: Content hash.
        """
        bucket = bucket_of(content_hash)
        with self.__lock:
            previous = self.__members[bucket].pop(content_hash, None)
            if previous is None:
                return
            self.__digests[bucket] ^= leaf(content_hash, previous)
            self.changes += 1

    def buckets(self) -> list[int]:
        """Every bucket's digest.

        Returns:
            list[int]: :data:`BUCKETS` unsigned 64-bit values.
        """
        with self.__lock:
            return list(self.__digests)

    def __len__(self) -> int:
        """Fragments held.

        Returns:
            int: The count.
        """
        with self.__lock:
            return sum(len(m) for m in self.__members)

    def root(self) -> bytes:
        """SHA-256 over the bucket digests (cached until the next change).

        Returns:
            bytes: 32-byte root.
        """
        with self.__lock:
            if self.__root is not None and self.__root[0] == self.changes:
                return self.__root[1]
            packed = b"".join(d.to_bytes(8, "big") for d in self.__digests)
            root = hashlib.sha256(packed).digest()
            self.__root = (self.changes, root)
            return root

    def sample(self, count: int) -> list[str]:
        """Up to ``count`` held hashes from randomly chosen buckets.

        Costs O(count) plus one bucket, instead of copying the inventory.

        Args:
            count: Hashes wanted.

        Returns:
            list[str]: Distinct hashes (fewer when fewer are held).
        """
        picked: list[str] = []
        if count <= 0:
            return picked
        order = list(range(BUCKETS))
        random.shuffle(order)
        with self.__lock:
            for bucket in order:
                members = self.__members[bucket]
                if members:
                    picked.extend(list(members)[: count - len(picked)])
                    if len(picked) >= count:
                        break
        return picked

    def page(self, bucket: int, after: str = "", limit: int = 0) -> tuple[dict[str, int], str]:
        """Page through one bucket's members in hash order.

        Args:
            bucket: Bucket index.
            after: Return hashes sorting after this cursor.
            limit: Page size; ``0`` for the whole bucket.

        Returns:
            tuple[dict[str, int], str]: ``content_hash -> version`` and the
            cursor for the next page (``""`` on the last).
        """
        with self.__lock:
            members = dict(self.__members[bucket])
        ordered = sorted(h for h in members if h > after)
        if limit > 0 and len(ordered) > limit:
            ordered = ordered[:limit]
            return {h: members[h] for h in ordered}, ordered[-1]
        return {h: members[h] for h in ordered}, ""


def encode_cursor(bucket: int, content_hash: str) -> str:
    """Build a whole-inventory paging cursor.

    Args:
        bucket: Bucket of the last hash returned.
        content_hash: Last hash returned.

    Returns:
        str: ``"<bucket>:<hash>"``.
    """
    return f"{bucket}:{content_hash}"


def decode_cursor(cursor: str) -> tuple[int, str]:
    """Parse a whole-inventory paging cursor.

    Args:
        cursor: ``"<bucket>:<hash>"``, or ``""`` to start.

    Returns:
        tuple[int, str]: Bucket and hash to continue after.
    """
    bucket, sep, content_hash = cursor.partition(":")
    if not sep or not bucket.isdigit():
        return 0, ""
    return min(int(bucket), BUCKETS - 1), content_hash


__all__ = ["BUCKETS", "InventoryDigest", "bucket_of", "decode_cursor", "encode_cursor", "fold", "leaf"]
