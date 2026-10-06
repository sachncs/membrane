"""One node on many cores: multi-loop HTTP, lock-free reads, and their thread safety."""

import os
import random
import sys
import threading
import time
import urllib.request

import pytest

from membrane.content_store import InProcessBytes
from membrane.node import Node
from membrane.runtime.concurrency import FREE_THREADED, gil_enabled, share
from membrane.server import Server
from membrane.store.table import FragmentTable
from membrane.transport.limits import ConcurrencyLimitMiddleware, InFlight, TransportLimits
from tests.conftest import make_fragment

PARALLEL = FREE_THREADED and not gil_enabled()


def wait_for(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("condition not met in time")
        time.sleep(0.05)


def test_several_event_loops_share_one_listener_and_node() -> None:
    server = Server(
        node=Node("loops"), port=0, load_hooks=False, limits=TransportLimits(http_threads=3, max_concurrency=8)
    )
    server.start()
    try:
        wait_for(lambda: len(server.transport.servers) == 3 and all(s.started for s in server.transport.servers))
        url = f"http://127.0.0.1:{server.transport.port}"
        server.node.content_store.put("blob-a", b"x")
        server.node.store(make_fragment("a" * 32, (0, 3)))
        for _ in range(30):
            with urllib.request.urlopen(f"{url}/retrieve?content_hash={'a' * 32}") as response:
                assert response.status == 200
        with urllib.request.urlopen(f"{url}/metrics") as response:
            text = response.read().decode()
        assert "membrane_http_event_loops 3" in text
        assert "membrane_gil_enabled" in text
    finally:
        assert server.stop() in (True, False)
    wait_for(lambda: not any(t.name.startswith("membrane-http-") for t in threading.enumerate()))


def test_concurrency_bound_is_split_between_loops() -> None:
    middleware = ConcurrencyLimitMiddleware(app=None, max_concurrency=10, queue_timeout_sec=0.1, loops=4)  # type: ignore[arg-type]
    assert middleware.per_loop == 3
    assert ConcurrencyLimitMiddleware(app=None, max_concurrency=2, queue_timeout_sec=0.1, loops=8).per_loop == 1  # type: ignore[arg-type]


def test_in_flight_counters_are_thread_safe() -> None:
    counters = InFlight()

    def churn() -> None:
        for _ in range(20_000):
            counters.add(active=1, waiting=1)
            counters.add(active=-1, waiting=-1)

    threads = [threading.Thread(target=churn) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (counters.active, counters.waiting) == (0, 0)


def test_reads_from_many_threads_are_merged_into_recency() -> None:
    table = FragmentTable()
    fragments = [make_fragment(f"{i:032x}", (0, 3)) for i in range(16)]
    for fragment in fragments:
        table.add(fragment, now=1.0)

    def read(offset: int) -> None:
        for fragment in fragments[offset::4]:
            table.record_access(fragment.identity.payload_hash, 100.0 + offset)

    threads = [threading.Thread(target=read, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    recency = table.recency()
    assert {recency[f.identity.payload_hash] for f in fragments} == {100.0, 101.0, 102.0, 103.0}
    table.record_access("not-resident", 5.0)
    assert "not-resident" not in table.recency()


def test_store_evict_and_read_race_keeps_accounting_exact() -> None:
    """Writers evict under the lock while readers take none; totals must stay exact."""
    node = Node("race", max_memory_bytes=40 * 1024)
    hashes = [f"{i:032x}" for i in range(400)]
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(seed: int) -> None:
        rnd = random.Random(seed)
        try:
            while not stop.is_set():
                h = hashes[rnd.randrange(len(hashes))]
                node.content_store.put(f"blob-{h}", b"v")
                node.store(make_fragment(h, (0, 3)))
                if rnd.random() < 0.2:
                    with node.lock:
                        if h in node.fragments:
                            node.remove_fragment(h)
        except BaseException as exc:
            errors.append(exc)

    def reader(seed: int) -> None:
        rnd = random.Random(seed)
        try:
            while not stop.is_set():
                fragment = node.retrieve(hashes[rnd.randrange(len(hashes))])
                assert fragment is None or fragment.identity.payload_hash in hashes
        except BaseException as exc:
            errors.append(exc)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
        threads += [threading.Thread(target=reader, args=(n,)) for n in range(5)]
        for thread in threads:
            thread.start()
        time.sleep(1.5)
        stop.set()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(interval)
    assert not errors
    with node.lock:
        assert node.memory_usage == sum(f.payload_size for f in node.fragments.values())
        assert node.memory_usage <= node.max_memory_bytes
        assert set(node.access_times) <= set(node.fragments)


def test_in_process_store_tracks_bytes_incrementally() -> None:
    store = InProcessBytes()
    store.put("a", b"12345")
    store.put("b", b"12")
    store.put("a", b"1")
    assert store.size() == 3
    assert store.delete("a") and not store.delete("a")
    assert store.size() == 2


def test_share_switches_objects_only_on_free_threaded_builds() -> None:
    # Only objects the garbage collector tracks can switch; None is skipped.
    switched = share([], None, {})
    assert switched == (2 if FREE_THREADED else 0)


@pytest.mark.skipif(not PARALLEL, reason="needs a free-threaded Python running without the GIL")
@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs at least 4 cores")
def test_retrieve_throughput_scales_with_threads() -> None:
    node = Node("bench", max_memory_bytes=1 << 34)
    hashes = [f"{i:032x}" for i in range(20_000)]
    for h in hashes:
        node.store(make_fragment(h, (0, 3)))

    def work(count: int) -> None:
        rnd = random.Random()
        retrieve = node.retrieve
        for _ in range(count):
            retrieve(hashes[rnd.randrange(len(hashes))])

    def throughput(threads: int, per_thread: int = 60_000) -> float:
        workers = [threading.Thread(target=work, args=(per_thread,)) for _ in range(threads)]
        started = time.perf_counter()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        return threads * per_thread / (time.perf_counter() - started)

    throughput(1)  # warm up
    single = max(throughput(1) for _ in range(2))
    four = max(throughput(4) for _ in range(2))
    assert four / single >= 1.8, f"4 threads: {four:,.0f}/s vs 1 thread: {single:,.0f}/s"
