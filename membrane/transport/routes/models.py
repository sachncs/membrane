"""Validated request bodies for the HTTP routes."""

from typing import Any

from pydantic import BaseModel, Field, conlist

from membrane.serialization import JsonDict
from membrane.transport.ops import (
    MAX_BODY_BYTES,
)


class FragmentPayload(BaseModel):
    """Wire format for a serialized Fragment.

    Mirrors the 3.0 canonical schema in
    :mod:`membrane.serialization`. The :class:`~membrane.identity.PayloadIdentity`
    is carried as a nested ``identity`` object; ranges are JSON
    arrays of two ints; ``shape`` is a list of ints; ``consistency``
    is one of strong / quorum / eventual; ``hlc`` is the wire
    integer from :class:`~membrane.hlc.HLC`.
    """

    schema_version: int = 5
    tenant_id: str = Field(default="public", max_length=128)
    identity: dict[str, Any] = Field(max_length=64)
    payload_ref: str | None = Field(default=None, max_length=512)
    payload_size: int = Field(ge=0, le=MAX_BODY_BYTES)
    ttl: float
    reuse_score: float
    version_id: int
    consistency: str = "strong"
    hlc: int = 0
    fingerprint_compat: str = Field(default="", max_length=128)

    def to_wire_dict(self) -> JsonDict:
        """Return the wire dict accepted by :func:`membrane.serialization.from_dict`.

        Returns:
            JsonDict: The wire-format fragment dict.
        """
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "identity": self.identity,
            "payload_ref": self.payload_ref,
            "payload_size": self.payload_size,
            "ttl": self.ttl,
            "reuse_score": self.reuse_score,
            "version_id": self.version_id,
            "consistency": self.consistency,
            "hlc": self.hlc,
            "fingerprint_compat": self.fingerprint_compat,
        }


class StoreRequest(BaseModel):
    """``POST /store`` body."""

    fragment: FragmentPayload
    is_primary: bool = False


class ReplicateRequest(BaseModel):
    """``POST /replicate`` body."""

    fragment: FragmentPayload
    is_primary: bool = False


class PrefillRequest(BaseModel):
    """``POST /prefill`` body.

    The ``prompt_tokens`` cap is generous (32768) so a long
    agentic prompt still fits; the per-token range is restricted
    to a valid int32 so a hostile payload cannot smuggle
    float / NaN values into the wire.
    """

    prompt_tokens: conlist(int, max_length=32768)  # type: ignore[valid-type]
    model_id: str = Field(default="default", max_length=256)


class SyncRequest(BaseModel):
    """``POST /sync`` body."""

    source_url: str = Field(max_length=2048)


class JoinRequest(BaseModel):
    """``POST /join`` body."""

    node_id: str = Field(min_length=1, max_length=128)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(ge=1, le=65535)


class LeaveRequest(BaseModel):
    """``POST /leave`` body."""

    node_id: str = Field(min_length=1, max_length=128)


class GossipRequest(BaseModel):
    """``POST /gossip`` body.

    The peers + fragment_locations lists are capped so a
    malicious peer cannot blow out the gossip budget with a
    million-element payload.

    The fields mirror :meth:`membrane.network.gossip.GossipState.to_json`;
    any field dropped here never reaches the receiver.
    """

    node_id: str = Field(min_length=1, max_length=128)
    timestamp: float
    peers: list[dict[str, Any]] = Field(default_factory=list, max_length=4096)
    fragment_locations: dict[str, list[str]] = Field(default_factory=dict, max_length=131072)
    inventory_bloom: str = Field(default="", max_length=4 * 1024 * 1024)
    inventory_merkle_root: str = Field(default="", max_length=128)
    inventory_size: int = Field(default=0, ge=0)
    fragment_tombstones: dict[str, float] = Field(default_factory=dict, max_length=131072)
    inventory_digest: dict[str, int] = Field(default_factory=dict, max_length=131072)


class DeleteRequest(BaseModel):
    """``POST /delete`` body."""

    content_hash: str = Field(min_length=1, max_length=256)
    node_id: str = Field(min_length=1, max_length=128)
    tombstone_until: float | None = None


class TombstoneRequest(BaseModel):
    """``POST /tombstone`` body."""

    content_hash: str = Field(min_length=1, max_length=256)
    until: float
    node_id: str = Field(min_length=1, max_length=128)


class PurgeRequest(BaseModel):
    """``POST /purge`` body."""

    content_hash: str = Field(min_length=1, max_length=256)


class VerifyRequest(BaseModel):
    """``POST /verify`` body."""

    content_hash: str = Field(min_length=1, max_length=256)
    claimed_size: int = Field(ge=0, le=MAX_BODY_BYTES)
    claimed_sha256_hex: str = Field(min_length=64, max_length=64)


class ReconstructRequest(BaseModel):
    """``POST /reconstruct`` body: assemble cached KV for a prompt.

    ``prefill`` also computes the uncovered spans with the node's compute
    backend (needs the ``write`` scope).
    """

    tokens: conlist(int, max_length=131072)  # type: ignore[valid-type]
    model_id: str = Field(default="default", max_length=256)
    prefill: bool = False


class PrefixLookupRequest(BaseModel):
    """``POST /prefix/lookup`` body (the ``GET`` form takes ``?tokens=1,2,3``)."""

    tokens: conlist(int, max_length=131072)  # type: ignore[valid-type]
    model_id: str = Field(default="default", max_length=256)


class RouteRequest(BaseModel):
    """``POST /route`` body: place a fragment (``content_hash``) or a prompt (``tokens``)."""

    content_hash: str = Field(default="", max_length=256)
    tokens: conlist(int, max_length=131072) | None = None  # type: ignore[valid-type]
    model_id: str = Field(default="default", max_length=256)
    local_cached_tokens: int = Field(default=0, ge=0)


__all__ = [
    "DeleteRequest",
    "FragmentPayload",
    "GossipRequest",
    "JoinRequest",
    "LeaveRequest",
    "PrefillRequest",
    "PrefixLookupRequest",
    "PurgeRequest",
    "ReconstructRequest",
    "ReplicateRequest",
    "RouteRequest",
    "StoreRequest",
    "SyncRequest",
    "TombstoneRequest",
    "VerifyRequest",
]
