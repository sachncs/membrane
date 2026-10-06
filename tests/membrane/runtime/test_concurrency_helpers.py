"""Deferred reference counting helpers, on either build."""

import ctypes

from membrane.runtime import concurrency


def test_the_deferred_refcount_api_is_found_or_absent(monkeypatch) -> None:
    function = concurrency.deferred_refcount_api()
    assert function is None or callable(function)
    monkeypatch.delattr(ctypes.pythonapi, "PyUnstable_Object_EnableDeferredRefcount", raising=False)
    original = ctypes.pythonapi

    class NoApi:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "pythonapi", NoApi())
    assert concurrency.deferred_refcount_api() is None
    monkeypatch.setattr(ctypes, "pythonapi", original)


def test_share_skips_none_and_counts_switches(monkeypatch) -> None:
    switched: list[object] = []

    def enable(obj: object) -> int:
        switched.append(obj)
        return 1

    monkeypatch.setattr(concurrency, "ENABLE_DEFERRED", enable)
    first, second = [], {}
    assert concurrency.share(first, None, second) == 2
    assert switched == [first, second]
    monkeypatch.setattr(concurrency, "ENABLE_DEFERRED", None)
    assert concurrency.share([]) == 0


def test_gil_state_is_reported() -> None:
    assert isinstance(concurrency.gil_enabled(), bool)
    assert concurrency.FREE_THREADED in (True, False)
