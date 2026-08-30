"""Single source of truth for Fragment serialization.

Every transport, persistence backend, and peer client previously
carried its own copy of ``serialize_fragment`` /
``deserialize_fragment``. They were identical but diverged over time,
producing a class of subtle bugs that only surfaced when one path was
changed and another was not. This module is the single authority.

The on-wire format is a plain ``dict[str, Any]``. Two fragments with
the same on-wire dict are considered byte-identical regardless of where
they came from, which is the property content-addressing relies on.

A ``schema_version`` discriminator is included so future format changes
can be detected and rejected loudly rather than silently mis-parsed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from membrane.errors import SchemaError
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity

SCHEMA_VERSION: int = 4

#: JSON-compatible value type used at every wire boundary.
#:
#: ``Any`` rather than a recursive union because the recursive
#: form breaks mypy's invariance for dict value types (a
#: ``dict[str, int]`` is not assignable to a
#: ``dict[str, JsonValue]`` even though every int is JSON).
#: Callers should re-validate the values they consume.
JsonValue = Any

#: A JSON object (used for every Membrane wire payload).
#: Same rationale as :data:`JsonValue`; the runtime shape is
#: ``dict[str, object]`` but the alias is named for
#: documentation purposes only.
JsonDict = dict[str, JsonValue]


def to_dict(fragment: Fragment) -> dict[str, Any]:
    """Serialize a Fragment to a wire-format ``dict``.

    Args:
        fragment: The fragment to serialize.

    Returns:
        A plain ``dict`` carrying ``schema_version``, the full
        :class:`~membrane.identity.PayloadIdentity` (expanded into a
        sub-dict), the payload reference, the lifecycle metadata,
        the consistency + HLC fields added at 2.0, and the
        v2.0+ ``fingerprint_compat`` field.
    """
    return {
            "schema_version": SCHEMA_VERSION,
            "identity": fragment.identity.to_dict(),
            "payload_ref": fragment.payload_ref,
            "payload_size": fragment.payload_size,
            "ttl": fragment.ttl,
            "reuse_score": fragment.reuse_score,
            "version_id": fragment.version_id,
            "consistency": fragment.consistency,
            "hlc": fragment.hlc,
            "fingerprint_compat": fragment.fingerprint_compat,
        }


def from_dict(data: dict[str, Any]) -> Fragment:
    """Deserialize a Fragment from a wire-format ``dict``.

    Args:
        data: The dict produced by :func:`to_dict`.

    Returns:
        The reconstructed ``Fragment``.

    Raises:
        SchemaError: If ``data["schema_version"]`` does not match
            ``SCHEMA_VERSION``, or if a required field is missing.
            Schema 1 and 2 payloads are deliberately rejected; the
            2.0 contract carries no shims for older shapes.
    """
    if "schema_version" not in data:
        raise SchemaError("serialized fragment missing schema_version")
    if data["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"incompatible schema_version={data['schema_version']}; expected {SCHEMA_VERSION}"
        )
    try:
        identity_obj: dict[str, Any] = data["identity"]
        identity = PayloadIdentity.from_dict(identity_obj)
        # 2.0 added consistency and hlc; both are required for
        # every fragment the wire carries.
        if "consistency" not in data:
            raise SchemaError("missing required field: consistency")
        if "hlc" not in data:
            raise SchemaError("missing required field: hlc")
        if "fingerprint_compat" not in data:
            raise SchemaError("missing required field: fingerprint_compat")
        return Fragment(
            identity=identity,
            payload_ref=data["payload_ref"],
            payload_size=int(data["payload_size"]),
            ttl=float(data["ttl"]),
            reuse_score=float(data["reuse_score"]),
            version_id=int(data["version_id"]),
            consistency=str(data["consistency"]),
            hlc=int(data["hlc"]),
            fingerprint_compat=str(data["fingerprint_compat"]),
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
