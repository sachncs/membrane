"""SemanticHash: locality-sensitive hashing for approximate identity."""

import logging

logger = logging.getLogger(__name__)


import struct


def compute_semantic_hash(embedding: tuple[float, ...], precision: int = 8) -> str:
    """Compute a locality-sensitive semantic hash from an embedding.

    The hash is derived by quantizing each dimension into bins and
    concatenating the bin indices. Nearby embeddings in cosine space
    will share the same semantic hash at coarse precision.

    Args:
        embedding: Dense vector tuple.
        precision: Number of bits per dimension quantization (default 8).

    Returns:
        Hexadecimal string representing the quantized embedding.
    """
    if not embedding:
        return "0"
    bins_per_dim = 1 << precision
    quantized = []
    for value in embedding:
        # Map [-1, 1] to [0, bins_per_dim - 1]
        normalized = (value + 1.0) / 2.0
        index = int(normalized * (bins_per_dim - 1))
        index = max(0, min(index, bins_per_dim - 1))
        quantized.append(index)

    # Pack into bytes
    byte_array = bytearray()
    if precision <= 8:
        byte_array.extend(quantized)
    else:
        for index in quantized:
            byte_array.extend(struct.pack(">H", index))

    return byte_array.hex()


def semantic_distance(hash_a: str, hash_b: str) -> float:
    """Approximate Hamming-style distance between two semantic hashes.

    Args:
        hash_a: First semantic hash hex string.
        hash_b: Second semantic hash hex string.

    Returns:
        Distance ratio in [0.0, 1.0] where 0.0 means identical.
    """
    if hash_a == hash_b:
        return 0.0
    bytes_a = bytes.fromhex(hash_a)
    bytes_b = bytes.fromhex(hash_b)
    max_len = max(len(bytes_a), len(bytes_b))
    if max_len == 0:
        return 0.0
    diff = sum(a != b for a, b in zip(bytes_a, bytes_b))
    diff += abs(len(bytes_a) - len(bytes_b))
    return diff / max_len
