"""Single source of truth for Fragment serialization.

Every transport, persistence backend, and peer client previously carried its
own copy of ``serialize_fragment`` / ``deserialize_fragment``. They were
identical but diverged over time, producing a class of subtle bugs that only
surfaced when one path was changed and another was not. This module is the
single authority.

The on-wire format is a plain ``dict[str, Any]``. Two fragments with the same
on-wire dict are considered byte-identical regardless of where they came from,
which is the property content-addressing relies on.

A ``schema_version`` discriminator is included so future format changes can
be detected and rejected loudly rather than silently mis-parsed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Union

from membrane.errors import SchemaError
from membrane.fragment import Fragment
from membrane.signature import Signature

SCHEMA_VERSION: int = 1

#: JSON-compatible value type used at every wire boundary.
JsonValue = Union[
    str,
    int,
    float,
    bool,
    None,
    list["JsonValue"],
    dict[str, "JsonValue"],
]

#: A JSON object (used for every Membrane wire payload).
JsonDict = dict[str, JsonValue]


def to_dict(fragment: Fragment) -> dict[str, Any]:
    """Serialize a Fragment to a wire-format ``dict``.

    Args:
        fragment: The fragment to serialize.

    Returns:
        A plain ``dict`` containing all fragment fields, with the
        ``embedding`` tuple encoded as a JSON string and ``schema_version``
        set to the current ``SCHEMA_VERSION``.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "content_hash": fragment.content_hash,
        "embedding": json.dumps(list(fragment.embedding)),
        "model_id": fragment.structural_signature.model_id,
        "layer_start": fragment.structural_signature.layer_range[0],
        "layer_end": fragment.structural_signature.layer_range[1],
        "token_start": fragment.structural_signature.token_span[0],
        "token_end": fragment.structural_signature.token_span[1],
        "size": fragment.size,
        "ttl": fragment.ttl,
        "reuse_score": fragment.reuse_score,
        "version_id": fragment.version_id,
    }


def from_dict(data: dict[str, Any]) -> Fragment:
    """Deserialize a Fragment from a wire-format ``dict``.

    Args:
        data: The dict produced by :func:`to_dict` (or a structurally
            compatible older version).

    Returns:
        The reconstructed ``Fragment``.

    Raises:
        SchemaError: If ``data["schema_version"]`` does not match
            ``SCHEMA_VERSION``, or if a required field is missing.
    """
    if "schema_version" not in data:
        raise SchemaError("serialized fragment missing schema_version")
    if data["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"incompatible schema_version={data['schema_version']}; expected {SCHEMA_VERSION}")
    try:
        signature = Signature(
            model_id=data["model_id"],
            layer_range=(int(data["layer_start"]), int(data["layer_end"])),
            token_span=(int(data["token_start"]), int(data["token_end"])),
        )
        return Fragment(
            content_hash=data["content_hash"],
            embedding=tuple(json.loads(data["embedding"])),
            structural_signature=signature,
            size=int(data["size"]),
            ttl=float(data["ttl"]),
            reuse_score=float(data["reuse_score"]),
            version_id=int(data["version_id"]),
        )
    except KeyError as exc:
        raise SchemaError(f"missing required field: {exc.args[0]}") from exc


def to_bytes(fragment: Fragment) -> bytes:
    """Serialize a Fragment to JSON bytes. Symmetric with :func:`from_bytes`."""
    return json.dumps(to_dict(fragment)).encode("utf-8")


def from_bytes(data: bytes) -> Fragment:
    """Deserialize a Fragment from JSON bytes produced by :func:`to_bytes`."""
    return from_dict(json.loads(data.decode("utf-8")))


def asdict_shallow(fragment: Fragment) -> dict[str, Any]:
    """Convenience wrapper that re-exports :func:`dataclasses.asdict` for callers that
    only need a structural copy (without the schema_version discriminator).
    """
    return asdict(fragment)


__all__: list[str] = [
    "SCHEMA_VERSION",
    "JsonDict",
    "JsonValue",
    "asdict_shallow",
    "from_bytes",
    "from_dict",
    "to_bytes",
    "to_dict",
]
