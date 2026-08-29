"""Shared pytest fixtures for the Membrane test suite.

The previous test layout had 28 copies of a local ``make_fragment``
helper scattered across the test tree. This conftest consolidates
them into a single fixture and uses pytest's autouse hook to inject
it as ``make_fragment(...)`` callable in every test module's
namespace, preserving the existing call-site shape.

The factory supports several historical calling conventions:

* ``make_fragment(content_hash)`` — just a hash.
* ``make_fragment(content_hash, token_span)`` — int-tuple is
  interpreted as ``token_span``.
* ``make_fragment(content_hash, embedding)`` — float-tuple is
  interpreted as ``embedding``.
* ``make_fragment(content_hash=..., token_span=..., size=..., ...)``
  — keyword form, preferred for new tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from membrane.fragment import Fragment
from membrane.signature import Signature


def make_fragment_factory(*args: Any, **kwargs: Any) -> Fragment:
    """Build a :class:`Fragment` with sensible test defaults.

    The factory dispatches on the *type* of the second positional
    argument: ``tuple[int, int]`` is ``token_span``; ``tuple[float, ...]``
    is ``embedding``. This matches every historical local helper
    found across the suite without requiring test rewrites.
    """
    if len(args) > 3:
        raise TypeError(f"make_fragment() takes at most 3 positional args, got {len(args)}")

    # Resolve positional args into kwargs. The second positional arg
    # can be ``size`` (int), ``token_span`` (tuple[int, int]), or
    # ``embedding`` (tuple[float, ...]). Disambiguate by runtime type.
    if len(args) >= 1:
        kwargs.setdefault("content_hash", args[0])
    if len(args) >= 2:
        second = args[1]
        if isinstance(second, bool):
            # bool is an int subclass — guard first to keep it from
            # the integer branch below.
            kwargs.setdefault("size", second)
        elif isinstance(second, int):
            kwargs.setdefault("size", second)
        elif isinstance(second, tuple):
            if second and all(isinstance(x, int) for x in second):
                kwargs.setdefault("token_span", second)
            else:
                kwargs.setdefault("embedding", second)
        else:
            kwargs.setdefault("embedding", second)
    if len(args) >= 3:
        kwargs.setdefault("model_id", args[2])

    content_hash: str = kwargs.pop("content_hash", "h1")
    token_span: tuple[int, int] = kwargs.pop("token_span", (0, 1))
    model_id: str = kwargs.pop("model_id", "test-model")
    embedding: tuple[float, ...] = kwargs.pop("embedding", (0.0,))
    layer_range: tuple[int, int] = kwargs.pop("layer_range", (0, 1))
    size: int = kwargs.pop("size", 100)
    ttl: float = kwargs.pop("ttl", 3600.0)
    reuse_score: float = kwargs.pop("reuse_score", 0.5)
    version_id: int = kwargs.pop("version_id", 1)

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"unexpected keyword args: {unexpected}")

    return Fragment(
        content_hash=content_hash,
        embedding=embedding,
        structural_signature=Signature(
            model_id=model_id,
            layer_range=layer_range,
            token_span=token_span,
        ),
        size=size,
        ttl=ttl,
        reuse_score=reuse_score,
        version_id=version_id,
    )


@pytest.fixture(autouse=True)
def inject_make_fragment_factory(request):
    """Make ``make_fragment(...)`` available as a module-level callable.

    Pytest doesn't have a built-in way to expose a fixture as a bare
    function name (the standard pattern requires every test to declare
    the fixture in its parameter list). This autouse fixture takes
    a different route: it injects ``make_fragment`` into the test
    module's namespace so callers can write ``make_fragment(...)``
    directly without a fixture parameter.

    Existing tests that already accept ``make_fragment`` as a
    fixture parameter continue to work because pytest treats the
    fixture name as available both ways.
    """
    request.module.make_fragment = make_fragment_factory
