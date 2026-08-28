"""SemanticHash: locality-sensitive hashing for approximate identity.

This module provides two functions used by
:class:`~membrane.semantic_index.SemanticIndex` to support efficient
*approximate* similarity lookups across fragments:

* :func:`compute_semantic_hash` — converts a dense embedding into a
  short hex string that is *similarity-preserving*: nearby vectors in
  cosine space share prefixes of the hash.
* :func:`semantic_distance` — approximates the distance between two
  hashes in ``[0, 1]`` without ever reconstructing the underlying
  embeddings.

Algorithm:
    :func:`compute_semantic_hash` performs uniform scalar quantization
    per dimension. Each component of the embedding (assumed to lie in
    ``[-1, 1]``) is mapped to a bin index in ``[0, 2^precision)`` and
    the bin indices are packed into bytes. Two vectors that are close
    in cosine similarity are very likely to share the same bin for
    most dimensions, so their hashes share long prefixes.

Limitations:
    * Out-of-range components (``|value| > 1``) are clamped to the
      first/last bin, which means they cannot be distinguished from
      values right at the boundary. Callers should normalize
      embeddings before hashing.
    * The hash is *not* cryptographically secure; it is intended only
      for locality-sensitive lookup, not for tamper detection.
    * Hamming-style distance over the hash is a proxy for true
      embedding distance; it is monotone but not strictly proportional.
"""

import logging

logger = logging.getLogger(__name__)


import struct


def compute_semantic_hash(embedding: tuple[float, ...], precision: int = 8) -> str:
    """Compute a locality-sensitive semantic hash from an embedding.

    Each component of the input vector is uniformly quantized into
    ``2 ** precision`` bins spanning ``[-1, 1]``. The resulting bin
    indices are concatenated into bytes and returned as a hex string.

    Args:
        embedding: Dense vector tuple. Each component should lie in
            ``[-1, 1]`` for the quantization to be meaningful;
            out-of-range values are clamped to the extreme bins.
        precision: Bits of quantization per dimension. Defaults to
            ``8`` (256 bins per dimension), which gives a good
            balance between hash size and discrimination. Values
            ``> 8`` use 16-bit packing per bin.

    Returns:
        str: Hexadecimal string representing the quantized embedding.
        Returns ``"0"`` for empty embeddings so callers can rely on a
        non-empty string representation.
    """
    if not embedding:
        return "0"
    bins_per_dim = 1 << precision
    quantized = []
    for value in embedding:
        # Map [-1, 1] to [0, bins_per_dim - 1] via an affine
        # transform; values outside the range are clamped below.
        normalized = (value + 1.0) / 2.0
        index = int(normalized * (bins_per_dim - 1))
        index = max(0, min(index, bins_per_dim - 1))
        quantized.append(index)

    # Pack into bytes: one byte per bin for precision <= 8, two
    # bytes (big-endian unsigned short) for higher precision.
    byte_array = bytearray()
    if precision <= 8:
        byte_array.extend(quantized)
    else:
        for index in quantized:
            byte_array.extend(struct.pack(">H", index))

    return byte_array.hex()


def semantic_distance(hash_a: str, hash_b: str) -> float:
    """Approximate Hamming-style distance between two semantic hashes.

    The distance is the fraction of byte positions that differ,
    treating length differences as differing positions. The result
    is in ``[0.0, 1.0]`` where ``0.0`` means the hashes are byte
    identical.

    Args:
        hash_a: First semantic hash (hex string), as produced by
            :func:`compute_semantic_hash`.
        hash_b: Second semantic hash (hex string).

    Returns:
        float: Distance ratio in ``[0.0, 1.0]``. Two empty hashes
        are considered identical (``0.0``).
    """
    if hash_a == hash_b:
        # Short-circuit: identical hex strings are always at zero
        # distance regardless of length.
        return 0.0
    bytes_a = bytes.fromhex(hash_a)
    bytes_b = bytes.fromhex(hash_b)
    max_len = max(len(bytes_a), len(bytes_b))
    if max_len == 0:
        # Both inputs decode to empty byte sequences — treat as
        # identical to avoid division by zero.
        return 0.0
    # Count differing positions within the overlapping region and
    # then add the length mismatch as additional differing
    # positions. This treats shorter hashes as if padded with
    # differing bytes.
    diff = sum(a != b for a, b in zip(bytes_a, bytes_b, strict=False))
    diff += abs(len(bytes_a) - len(bytes_b))
    return diff / max_len
