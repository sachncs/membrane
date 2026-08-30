"""Compute backend helpers.

Provides a module-level :func:`token_hash` helper used by every
backend when simulated prefill needs a deterministic digest
over a token chunk. The earlier code had five copies of this
function (one per backend).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def token_hash(tokens: Sequence[int]) -> str:
    """Compute a deterministic MD5 digest over a token chunk.

    Args:
        tokens: Token IDs to hash.

    Returns:
        str: Hexadecimal MD5 digest.
    """
    payload = ",".join(str(t) for t in tokens)
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()
