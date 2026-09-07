from tests.conftest import make_fragment

"""Thread-safety stress tests for Node."""

import threading
import time

import pytest

from membrane.fragment import Fragment
from membrane.node import Node


class TestThreadSafety:
    """Stress tests for concurrent store/retrieve/evict."""

    def test_concurrent_store_and_retrieve(self):
        node = Node("stress", max_memory_bytes=10000)
        errors: list[Exception] = []

        def store_worker(n: int):
            try:
                for i in range(n):
                    frag = make_fragment(f"frag-{i}", size=50)
                    node.store(frag, is_primary=True)
            except Exception as exc:
                errors.append(exc)

        def retrieve_worker(n: int):
            try:
                for i in range(n):
                    node.retrieve(f"frag-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=store_worker, args=(100,)),
            threading.Thread(target=retrieve_worker, args=(100,)),
            threading.Thread(target=store_worker, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"
        stats = node.get_stats()
        assert stats.memory_used_bytes >= 0
        assert stats.fragment_count >= 0

    def test_concurrent_evict_during_store(self):
        node = Node("stress2", max_memory_bytes=500)
        errors: list[Exception] = []

        def store_worker():
            try:
                for i in range(20):
                    frag = make_fragment(f"evict-{i}", size=100)
                    node.store(frag, is_primary=True)
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=store_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"
        stats = node.get_stats()
        assert stats.memory_used_bytes <= node.max_memory_bytes
        assert stats.fragment_count >= 0
