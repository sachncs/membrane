"""Bloom filter for gossip inventory checks.

Phase 5 exchanges a compact Bloom filter alongside every gossip
state so receivers can answer ``contains?(hash)`` in O(k)
without the cost of a full inventory digest. The filter is sized
to a target false-positive rate at construction time and is
deterministic for a given (m_bits, k_hashes, salts) triple so two
peers that have both inserted the same hash produce byte-identical
filters.

A 64-bit SHA-256-based hash provides the per-bit and per-iter
seeds. The filter is fully deterministic across Python versions
and platforms because :class:`hashlib.sha256` is.
"""

from __future__ import annotations

import hashlib
import math

_BLOOM_MAGIC: bytes = b"MBF1"  # Marker; 2.x peers ignore filters without it.


class BloomFilter:
    """Fixed-size Bloom filter with k independent hash functions.

    Attributes:
        m_bits: Total number of bits in the filter.
        k_hashes: Number of hash functions per item.
        bits: Underlying byte string; ``len(bits) * 8 >= m_bits``.
    """

    def __init__(self, m_bits: int, k_hashes: int, bits: bytes) -> None:
        """Initialize the filter from raw bytes.

        Args:
            m_bits: Total number of bits.
            k_hashes: Hash-function count.
            bits: Raw byte string; ``len(bits) * 8`` must equal
                ``m_bits``.

        Raises:
            ValueError: On inconsistent sizes.
        """
        if m_bits <= 0:
            raise ValueError(f"m_bits must be > 0, got {m_bits}")
        if k_hashes <= 0:
            raise ValueError(f"k_hashes must be > 0, got {k_hashes}")
        if len(bits) * 8 != m_bits:
            raise ValueError(
                f"bits length {len(bits)} bytes (={len(bits) * 8} bits) "
                f"does not match m_bits={m_bits}"
            )
        self.m_bits = m_bits
        self.k_hashes = k_hashes
        self.bits = bits

    def __contains__(self, item: str | bytes) -> bool:
        """Return whether ``item`` was likely added.

        False positives are possible at the configured rate;
        false negatives never occur.
        """
        if not isinstance(item, bytes):
            item = str(item).encode("utf-8")
        return all(_get_bit(self.bits, idx) for idx in self._indices(item))

    def add(self, item: str | bytes) -> BloomFilter:
        """Return a new filter with ``item`` inserted.

        The original is left untouched. Membership is
        ``return a new filter`` to keep filters immutable +
        shareable across threads.
        """
        if not isinstance(item, bytes):
            item = str(item).encode("utf-8")
        bits = bytearray(self.bits)
        for idx in self._indices(item):
            _set_bit(bits, idx)
        return BloomFilter(self.m_bits, self.k_hashes, bytes(bits))

    def _indices(self, item: bytes) -> list[int]:
        """Compute the k bit positions for ``item``.

        Uses two independent SHA-256 streams so the k-th
        function is the high half of the digest for ``item``
        XORed with a counter-salted prefix, keeping the
        function deterministic.
        """
        primary = hashlib.sha256(_BLOOM_MAGIC + item).digest()
        secondary = hashlib.sha256(_BLOOM_MAGIC + b"-" + item).digest()
        out: list[int] = []
        for k in range(self.k_hashes):
            offset = (k * 8) % 64
            raw = int.from_bytes(
                (primary + secondary)[offset : offset + 8].ljust(8, b"\x00"),
                "big",
            )
            out.append(raw % self.m_bits)
        return out

    def serialize(self) -> bytes:
        """Encode as a self-describing byte string for the wire."""
        return (
            _BLOOM_MAGIC
            + self.m_bits.to_bytes(4, "big")
            + self.k_hashes.to_bytes(2, "big")
            + self.bits
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> "BloomFilter":
        """Decode the wire format produced by :meth:`serialize`.

        Returns:
            BloomFilter: Decoded filter.

        Raises:
            ValueError: On a malformed header.
        """
        if not payload.startswith(_BLOOM_MAGIC):
            raise ValueError("Bloom filter wire payload missing magic prefix")
        m_bits = int.from_bytes(payload[4:8], "big")
        k_hashes = int.from_bytes(payload[8:10], "big")
        bits = payload[10:]
        if len(bits) * 8 != m_bits:
            raise ValueError(
                f"Bloom bits length {len(bits) * 8} does not match m_bits={m_bits}"
            )
        return cls(m_bits=m_bits, k_hashes=k_hashes, bits=bits)

    @classmethod
    def tuned_for(cls, expected_items: int, fp_rate: float = 0.001) -> "BloomFilter":
        """Build an empty filter sized for the given capacity and FP rate.

        The optimal (m, k) pair comes from the closed-form
        derivation in Bloom's 1970 paper:

          m = ceil(-n * ln(p) / (ln(2) ** 2))
          k = round((m / n) * ln(2))

        with a minimum of 2 hash functions to keep small filters
        honest.

        Args:
            expected_items: Anticipated member count.
            fp_rate: Target false-positive rate (default ``0.001``,
                i.e. one in a thousand).

        Returns:
            BloomFilter: Empty filter ready to receive members.
        """
        if expected_items <= 0:
            raise ValueError(f"expected_items must be > 0, got {expected_items}")
        if not 0 < fp_rate < 1:
            raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")
        m = max(8, math.ceil(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        k = max(2, round((m / expected_items) * math.log(2)))
        # Round m up to a multiple of 8 so the byte array is clean.
        m = ((m + 7) // 8) * 8
        bits = bytes(m // 8)
        return cls(m_bits=m, k_hashes=k, bits=bits)


def _get_bit(buf: bytes, idx: int) -> bool:
    return bool(buf[idx // 8] & (1 << (idx % 8)))


def _set_bit(buf: bytearray, idx: int) -> None:
    buf[idx // 8] |= 1 << (idx % 8)


__all__ = ["BloomFilter"]
