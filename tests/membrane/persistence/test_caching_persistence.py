"""CachingPersistence: write-through over a canonical backend, and serving through its outages."""

import pytest

from membrane.persistence.cache import CachingPersistence
from membrane.persistence.memory import Memory
from tests.conftest import make_fragment


class Down:
    """A backend whose every call fails, like Redis during an outage."""

    def __getattr__(self, name: str):
        def fail(*args, **kwargs):
            raise ConnectionError(f"{name}: backend down")

        return fail


def test_writes_through_and_reads_through() -> None:
    inner = Memory()
    cache = CachingPersistence(inner)
    fragment = make_fragment("a" * 32, (0, 3))
    assert cache.ping()
    assert cache.store_fragment(fragment, "n1", is_primary=True)
    assert inner.retrieve_fragment(fragment.key) == fragment
    assert cache.retrieve_fragment(fragment.key) == fragment
    assert list(cache.list_node_fragments("n1")) == [fragment.key]
    assert cache.get_primary(fragment.key) == "n1"
    assert fragment.key in cache.inventory_digest("n1")
    cache.record_location(fragment.key, "n2")
    assert "n2" in cache.locate(fragment.key)
    assert cache.lru_candidates(5) == [fragment.key]
    # A miss in the cache is filled from the inner backend.
    cache.cache.clear()
    assert cache.retrieve_fragment(fragment.key) == fragment and fragment.key in cache.cache
    assert cache.forget_on_node(fragment.key, "n1")
    assert cache.delete_fragment(fragment.key)
    assert cache.retrieve_fragment(fragment.key) is None
    cache.store_fragment(fragment, "n1")
    cache.flush()
    assert cache.cache == {} and inner.retrieve_fragment(fragment.key) is None


def test_serves_from_cache_while_the_backend_is_down() -> None:
    failures: list[str] = []
    cache = CachingPersistence(Down(), on_unavailable=lambda op, exc: failures.append(op))  # type: ignore[arg-type]
    fragment = make_fragment("b" * 32, (0, 3))
    assert cache.ping() is False
    assert cache.store_fragment(fragment, "n1") is False
    assert cache.retrieve_fragment(fragment.key) == fragment  # still served
    assert cache.retrieve_fragment("unknown") is None
    assert cache.inventory_digest("n1") == {}
    assert cache.list_node_fragments("n1") == []
    cache.record_location(fragment.key, "n1")
    assert cache.locate(fragment.key) == []
    assert cache.get_primary(fragment.key) is None
    assert cache.lru_candidates(3) == []
    assert cache.forget_on_node(fragment.key, "n1") is False
    assert cache.delete_fragment(fragment.key) is False
    cache.flush()
    assert set(failures) >= {
        "ping",
        "store_fragment",
        "retrieve_fragment",
        "inventory_digest",
        "list_node_fragments",
        "record_location",
        "locate",
        "get_primary",
        "lru_candidates",
        "forget_on_node",
        "delete_fragment",
        "flush",
    }


def test_forget_is_a_no_op_for_backends_without_it() -> None:
    class Minimal(Memory):
        forget_on_node = None  # type: ignore[assignment]

    assert CachingPersistence(Minimal()).forget_on_node("x", "n1") is True


@pytest.mark.parametrize("op", ["store_fragment", "retrieve_fragment"])
def test_default_unavailable_callback_is_silent(op: str) -> None:
    cache = CachingPersistence(Down())  # type: ignore[arg-type]
    fragment = make_fragment("c" * 32, (0, 3))
    if op == "store_fragment":
        assert cache.store_fragment(fragment, "n1") is False
    else:
        assert cache.retrieve_fragment(fragment.key) is None
