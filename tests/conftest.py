"""Shared pytest fixtures for the Membrane test suite.

The previous test layout had many copies of local ``make_fragment``
helpers scattered across the test tree. This conftest consolidates
them into a single factory function (``make_fragment``) that every
test module imports explicitly. There is no autouse injection: each
test file declares ``from tests.conftest import make_fragment`` (or
imports it via a shared helper) so the function name is visible to
static analysis tools (ruff, mypy) and so test failures identify the
failing fixture unambiguously.

The factory supports several historical calling conventions:

* ``make_fragment(content_hash)`` — just a hash.
* ``make_fragment(content_hash, size)`` — integer second positional.
* ``make_fragment(content_hash, token_span)`` — int-tuple is
  interpreted as ``token_span``.
* ``make_fragment(content_hash, embedding)`` — float-tuple is
  interpreted as ``embedding`` (silently dropped on the new schema;
  embeddding is no longer on Fragment).
* ``make_fragment(content_hash=..., payload_size=..., token_span=...)``
  — keyword form, preferred for new tests.
"""

from __future__ import annotations

from typing import Any

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity


def make_fragment(*args: Any, **kwargs: Any) -> Fragment:
    """Build a :class:`Fragment` with sensible test defaults.

    The factory dispatches on the *type* of the second positional
    argument: ``tuple[int, int]`` is ``token_span``; ``tuple[float, ...]``
    is ``embedding`` (no longer on Fragment, so it is dropped silently).
    This matches every historical local helper found across the suite
    without requiring test rewrites.
    """
    if len(args) > 3:
        raise TypeError(f"make_fragment() takes at most 3 positional args, got {len(args)}")

    if len(args) >= 1:
        kwargs.setdefault("content_hash", args[0])
    if len(args) >= 2:
        second = args[1]
        if isinstance(second, (bool, int)):
            kwargs.setdefault("payload_size", second)
        elif isinstance(second, tuple):
            if second and all(isinstance(x, int) for x in second):
                kwargs.setdefault("token_span", second)
            else:
                # legacy "embedding" positional; ignored on new schema.
                kwargs.setdefault("_legacy_embedding", second)
        else:
            kwargs.setdefault("_legacy_embedding", second)
    if len(args) >= 3:
        kwargs.setdefault("model_id", args[2])

    content_hash: str = kwargs.pop("content_hash", "h1")
    token_span: tuple[int, int] = kwargs.pop("token_span", (0, 1))
    model_id: str = kwargs.pop("model_id", "test-model")
    layer_range: tuple[int, int] = kwargs.pop("layer_range", (0, 1))
    payload_size: int = kwargs.pop("payload_size", kwargs.pop("size", 100))
    ttl: float = kwargs.pop("ttl", 3600.0)
    reuse_score: float = kwargs.pop("reuse_score", 0.5)
    version_id: int = kwargs.pop("version_id", 1)
    payload_ref: str | None = kwargs.pop("payload_ref", None)
    kwargs.pop("_legacy_embedding", None)
    kwargs.pop("embedding", None)
    kwargs.pop("size", None)

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"unexpected keyword args: {unexpected}")

    identity = PayloadIdentity(
        payload_hash=content_hash,
        model_id=model_id,
        model_revision="",
        tokenizer_name=model_id,
        tokenizer_revision="",
        layer_range=layer_range,
        head_range=(-1, -1),
        token_span=token_span,
        dtype="float16",
        shape=(1, 1, 1, 128, 64),
    )

    if payload_ref is None:
        payload_ref = f"blob-{content_hash}"

    return Fragment(
        identity=identity,
        payload_ref=payload_ref,
        payload_size=payload_size,
        ttl=ttl,
        reuse_score=reuse_score,
        version_id=version_id,
    )


__all__ = ["make_fragment"]
