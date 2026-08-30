"""Compatibility fingerprint for fragment reuse (Phase 1.1).

At 2.0+ every fragment carries a :class:`ModelCompatibilityFingerprint`
that uniquely identifies the model and tokenizer that produced
its bytes. The fingerprint is computed at prefill time and
re-validated at retrieval time; a mismatch raises
:class:`MembraneIncompatibleError` so the cluster never installs
a payload built by one model into a different model's engine.

Fields (the ``model_id`` and ``tokenizer_name`` are separate so
the fingerprint distinguishes a Mistral model run through a
HuggingFace tokenizer from the same model run through the vLLM
default tokenizer):

* ``model_id``: Stable identifier of the model that produced
  (or can consume) the bytes.
* ``model_revision``: Pinning commit hash, ``""`` when unpinned.
* ``model_layout_version``: Monotonic counter bumped when a model
  archive's KV layout changes between revisions of the same
  ``model_id`` (e.g. rope scaling tweaks, attention head
  re-partitioning). A v4 producer can still read v3
  layout-pinned bytes; the counter is the source of truth.
* ``tokenizer_name``: Tokenizer identifier. Often equal to
  ``model_id`` but the two diverge when the tokenizer ships
  separately.
* ``tokenizer_revision``: Pinning commit hash, ``""`` when
  unpinned.
* ``tokenizer_layout_version``: Layout-version counter for the
  tokenizer (vocab diffs, added special tokens).
* ``dtype``: Element dtype, one of the values supported by
  :class:`~membrane.identity.PayloadIdentity`.
* ``config_hash``: ``sha256`` of the canonical JSON encoding of
  the underlying model config (``json.dumps(cfg, sort_keys=True,
  separators=(",", ":"))``). Operators that swap the model
  archive without bumping the digest will catch it here.

The class is frozen + hashable so it can be a dictionary key
or set member. :meth:`compatibility_hash` produces a
content-independent digest used as the wire field
``Fragment.fingerprint_compat``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCompatibilityFingerprint:
    """Stable identity fingerprint for fragment compatibility.

    Attributes:
        model_id: Model identifier.
        model_revision: Pinning revision hash, ``""`` when unpinned.
        model_layout_version: Monotonic layout counter, ``0`` when
            unused.
        tokenizer_name: Tokenizer identifier. Often equal to
            ``model_id`` but the two diverge when the tokenizer
            ships separately.
        tokenizer_revision: Pinning revision hash, ``""`` when
            unpinned.
        tokenizer_layout_version: Monotonic layout counter.
        dtype: Element dtype.
        config_hash: ``sha256`` of the model config JSON.
    """

    model_id: str
    model_revision: str
    model_layout_version: int
    tokenizer_name: str
    tokenizer_revision: str
    tokenizer_layout_version: int
    dtype: str
    config_hash: str

    def __post_init__(self) -> None:
        """Validate invariants.

        Raises:
            ValueError: When any non-empty string field is empty
                or any counter is negative.
        """
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.tokenizer_name, str) or not self.tokenizer_name:
            raise ValueError("tokenizer_name must be a non-empty string")
        if self.model_layout_version < 0 or self.tokenizer_layout_version < 0:
            raise ValueError("layout versions must be non-negative")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("dtype must be a non-empty string")
        # ``config_hash`` may be the empty string when callers have
        # not yet computed the model config hash. Operators that
        # pre-compute the config hash are encouraged to populate
        # it; the empty value is a documented escape hatch.
        if not isinstance(self.config_hash, str):
            raise ValueError("config_hash must be a string")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dict for the wire field."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_layout_version": self.model_layout_version,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_layout_version": self.tokenizer_layout_version,
            "dtype": self.dtype,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelCompatibilityFingerprint:
        """Reconstruct from the output of :meth:`to_dict`.

        Args:
            data: A ``dict`` previously produced by
                :meth:`to_dict` or received over the wire.

        Returns:
            ModelCompatibilityFingerprint: The reconstructed
            fingerprint.
        """
        def _coerce_int(value: object, default: int) -> int:
            try:
                return int(value)  # type: ignore[call-overload]
            except (TypeError, ValueError):
                return default

        return cls(
            model_id=str(data["model_id"]),
            model_revision=str(data.get("model_revision", "")),
            model_layout_version=_coerce_int(
                data.get("model_layout_version", 0), 0
            ),
            tokenizer_name=str(data.get("tokenizer_name", data["model_id"])),
            tokenizer_revision=str(data.get("tokenizer_revision", "")),
            tokenizer_layout_version=_coerce_int(
                data.get("tokenizer_layout_version", 0), 0
            ),
            dtype=str(data["dtype"]),
            config_hash=str(data.get("config_hash", "")),
        )

    def compatibility_hash(self) -> str:
        """Return the 64-character hex compatibility digest.

        The digest is :func:`compat_hash` applied to the canonical
        JSON of :meth:`to_dict`. The wire field
        ``Fragment.fingerprint_compat`` is this string.

        Returns:
            str: 64-character hex digest.
        """
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compat_hash(
    model_id: str,
    *,
    model_revision: str = "",
    model_layout_version: int = 0,
    tokenizer_name: str | None = None,
    tokenizer_revision: str = "",
    tokenizer_layout_version: int = 0,
    dtype: str = "float16",
    config_hash: str = "",
) -> ModelCompatibilityFingerprint:
    """Convenience factory.

    When ``tokenizer_name`` is ``None`` the function falls back to
    ``model_id`` (the most common case: a HuggingFace causal LM
    where the tokenizer ships in the same repository).

    Args:
        model_id: Model identifier.
        model_revision: Pinning revision hash, default ``""``.
        model_layout_version: Layout counter, default ``0``.
        tokenizer_name: Tokenizer identifier, default ``model_id``.
        tokenizer_revision: Pinning revision hash, default ``""``.
        tokenizer_layout_version: Layout counter, default ``0``.
        dtype: Element dtype, default ``"float16"``.
        config_hash: ``sha256`` of the model config JSON, default
            ``""`` when not yet computed.

    Returns:
        ModelCompatibilityFingerprint: A new fingerprint.
    """
    return ModelCompatibilityFingerprint(
        model_id=model_id,
        model_revision=model_revision,
        model_layout_version=model_layout_version,
        tokenizer_name=tokenizer_name or model_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_layout_version=tokenizer_layout_version,
        dtype=dtype,
        config_hash=config_hash,
    )


def compute_config_hash(model_config: dict[str, object] | object) -> str:
    """Compute the canonical config hash for a HuggingFace config.

    The function accepts either a ``dict`` or any object with a
    ``to_dict()`` method. The canonical encoding is the same
    one the wire uses, so the on-disk config hash matches the
    wire's compatibility field.

    Args:
        model_config: A transformers config object or a plain
            ``dict``. Plain ``dict`` is the common case for tests
            that don't load a real model.

    Returns:
        str: 64-character hex digest.
    """
    if isinstance(model_config, dict):
        config_dict = dict(model_config)
    else:
        to_dict = getattr(model_config, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                "model_config must be a dict or expose to_dict()"
            )
        config_dict = to_dict()
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "MembraneIncompatibleError",
    "MembraneValidator",
    "ModelCompatibilityFingerprint",
    "compat_hash",
    "compute_config_hash",
]


class MembraneIncompatibleError(RuntimeError):
    """Raised when a fragment's compatibility fingerprint disagrees
    with the live engine's fingerprint.

    This is the v2.0+ counterpart to a model-version mismatch.
    Operators that swap a model archive without bumping the
    digest trigger this error at retrieval time so the
    cluster refuses to install a payload built by the previous
    model into the new engine.
    """


class MembraneValidator:
    """Validates that a stored fragment matches the live engine.

    The validator is constructed with a live
    :class:`ModelCompatibilityFingerprint` (the one stamped by
    the active compute backend on every :func:`op_store` call).
    Each :meth:`validate` compares the stored fragment's
    ``fingerprint_compat`` against the live fingerprint's
    :meth:`ModelCompatibilityFingerprint.compatibility_hash`.

    Usage::

        fingerprint = adapter.active_fingerprint()
        validator = MembraneValidator(fingerprint)
        # ...
        validator.validate(stored_fragment)
    """

    def __init__(self, fingerprint: ModelCompatibilityFingerprint) -> None:
        """Initialize with the live engine fingerprint.

        Args:
            fingerprint: The active engine's compatibility
                fingerprint; :meth:`validate` compares incoming
                fragments to this.
        """
        self.fingerprint = fingerprint
        self._hash = fingerprint.compatibility_hash()

    def validate(self, fragment: object) -> None:
        """Verify ``fragment.fingerprint_compat`` matches the live hash.

        Args:
            fragment: Any object with a ``fingerprint_compat`` field.

        Raises:
            MembraneIncompatibleError: When the live hash and the
                stored hash disagree, or when the stored fragment
                ships with the empty string (legacy / tests; a
                v2.0+ deployment must populate the field).
        """
        stored = getattr(fragment, "fingerprint_compat", "")
        if not stored:
            raise MembraneIncompatibleError(
                "fragment has no compatibility fingerprint; v2.0+ deployments must "
                "populate fragment.fingerprint_compat before op_store."
            )
        if stored != self._hash:
            raise MembraneIncompatibleError(
                f"fragment.fingerprint_compat={stored!r} disagrees with the live "
                f"engine fingerprint {self._hash!r}"
            )
