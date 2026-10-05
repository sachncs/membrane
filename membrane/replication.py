"""Replicate fragments together with their KV bytes.

A fragment is metadata (:class:`~membrane.fragment.Fragment`) plus, when
``payload_ref`` is set, KV bytes in the node's content store. A replica
is only useful if it has both, so every replication path (quorum writes,
the background replicator, drain, rebalancing) goes through
:func:`replicate_fragment`:

1. ``PUT /blobs/{payload_ref}`` with the bytes and their SHA-256 (the
   peer verifies the digest before storing);
2. ``POST /replicate`` with the metadata (the peer refuses metadata
   whose bytes it does not hold).

:func:`hand_off_primary` additionally asks the peer to become the
primary owner and verifies the peer's copy before the caller gives up
ownership.
"""

import logging
from typing import Any, Protocol

from membrane.fragment import Fragment
from membrane.wire.v3.chunks import sha256_hex

logger = logging.getLogger(__name__)


class ReplicaTarget(Protocol):
    """The peer operations replication needs (implemented by :class:`~membrane.network.peer.Peer`)."""

    def put_blob(self, payload_ref: str, data: bytes) -> bool:
        """Upload KV bytes.

        Args:
            payload_ref: Content-store key.
            data: The bytes.

        Returns:
            bool: True when the peer holds the bytes.
        """
        ...

    def request_replicate(self, fragment: Fragment, is_primary: bool = False) -> bool:
        """Send fragment metadata.

        Args:
            fragment: The fragment.
            is_primary: Ask the peer to become the primary owner.

        Returns:
            bool: True when the peer stored the fragment.
        """
        ...

    def blob_digest(self, payload_ref: str) -> str | None:
        """Return the peer's SHA-256 of its copy of the bytes.

        Args:
            payload_ref: Content-store key.

        Returns:
            str | None: Hex digest, or ``None`` when absent.
        """
        ...

    def retrieve_fragment(self, content_hash: str) -> Fragment | None:
        """Fetch fragment metadata from the peer.

        Args:
            content_hash: Content hash of the fragment.

        Returns:
            Fragment | None: The fragment, or ``None`` when absent.
        """
        ...


def payload_for(fragment: Fragment, content_store: Any) -> bytes | None:
    """Read the local KV bytes of ``fragment``.

    Args:
        fragment: The fragment.
        content_store: The node's content store.

    Returns:
        bytes | None: The bytes, or ``None`` for a metadata-only fragment
        or when the local store does not hold them.
    """
    if fragment.payload_ref is None or content_store is None:
        return None
    try:
        data: bytes | None = content_store.get(fragment.payload_ref)
    except Exception as exc:
        logger.warning("cannot read payload %s for replication: %s", fragment.payload_ref, exc)
        return None
    return data


def replicate_fragment(
    target: ReplicaTarget,
    fragment: Fragment,
    payload: bytes | None,
    is_primary: bool = False,
    skip_if_present: bool = False,
) -> bool:
    """Copy ``fragment`` and its bytes to ``target``.

    Args:
        target: The receiving peer.
        fragment: The fragment.
        payload: Its KV bytes (from :func:`payload_for`); ignored for a
            metadata-only fragment.
        is_primary: Ask the peer to become the primary owner.
        skip_if_present: Ask for the peer's digest first and skip the
            upload when it already holds identical bytes (one extra round
            trip; worth it for hand-offs, where the target is usually
            already a replica).

    Returns:
        bool: True when the peer stored both the bytes and the metadata.
        False when the bytes are needed but missing locally, or either
        request fails.
    """
    if fragment.payload_ref is not None:
        if payload is None:
            logger.warning(
                "not replicating %s: its payload %s is not in the local store",
                fragment.identity.payload_hash,
                fragment.payload_ref,
            )
            return False
        present = skip_if_present and target.blob_digest(fragment.payload_ref) == sha256_hex(payload)
        if not present and not target.put_blob(fragment.payload_ref, payload):
            return False
    return bool(target.request_replicate(fragment, is_primary=is_primary))


def verify_replica(target: ReplicaTarget, fragment: Fragment, payload: bytes | None) -> bool:
    """Confirm ``target`` holds ``fragment`` and byte-identical KV bytes.

    Args:
        target: The peer to check.
        fragment: The fragment.
        payload: The local bytes; ignored for a metadata-only fragment.

    Returns:
        bool: True when the metadata is present and, if the fragment has
        a payload, the peer's digest matches the local bytes.
    """
    if target.retrieve_fragment(fragment.identity.payload_hash) is None:
        return False
    if fragment.payload_ref is None:
        return True
    return payload is not None and target.blob_digest(fragment.payload_ref) == sha256_hex(payload)


def hand_off_primary(target: ReplicaTarget, fragment: Fragment, payload: bytes | None) -> bool:
    """Make ``target`` the primary owner of ``fragment``, verified.

    Args:
        target: The new owner.
        fragment: The fragment.
        payload: Its local KV bytes.

    Returns:
        bool: True when the target accepted ownership and holds a verified
        copy; only then may the caller drop its own primary flag.
    """
    return replicate_fragment(target, fragment, payload, is_primary=True, skip_if_present=True) and verify_replica(
        target, fragment, payload
    )


__all__ = ["ReplicaTarget", "hand_off_primary", "payload_for", "replicate_fragment", "verify_replica"]
