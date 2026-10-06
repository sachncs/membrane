"""Measure ``GET /retrieve`` throughput of one node under concurrent clients.

Starts ``membrane serve`` with the given interpreter and flags, stores
fragments, then runs client processes (each with keep-alive connections)
for a fixed time and prints requests per second::

    python scripts/bench_http.py --python python3.14t --http-threads 4
    python scripts/bench_http.py --python python3.14 --http-threads 1

Clients run in separate processes so the load generator is not the
bottleneck.
"""

import argparse
import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
import urllib.request

FRAGMENTS = 2_000

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("bench_http")


def free_port() -> int:
    """Pick an unused local TCP port.

    Returns:
        int: The port.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(url: str, proc: subprocess.Popen) -> None:
    """Block until the node answers ``/readyz``.

    Args:
        url: Node URL.
        proc: The node process.

    Raises:
        RuntimeError: When the node exits or does not come up in 60 s.
    """
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("membrane serve exited")
        try:
            urllib.request.urlopen(f"{url}/readyz", timeout=1)
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("membrane serve did not come up")


def seed(url: str) -> list[str]:
    """Prefill prompts so there are fragments to read.

    Args:
        url: Node URL.

    Returns:
        list[str]: Content hashes stored.
    """
    hashes: list[str] = []
    for i in range(FRAGMENTS // 4):
        body = json.dumps({"prompt_tokens": list(range(i * 1000, i * 1000 + 512)), "model_id": "bench"}).encode()
        request = urllib.request.Request(f"{url}/prefill", data=body, headers={"content-type": "application/json"})
        reply = json.load(urllib.request.urlopen(request))
        hashes += [f["identity"]["payload_hash"] for f in reply["fragments"]]
    return hashes


def client(url: str, hashes: list[str], seconds: float, connections: int, results: multiprocessing.Queue) -> None:
    """Issue reads for ``seconds`` over ``connections`` keep-alive connections.

    Args:
        url: Node URL.
        hashes: Hashes to read.
        seconds: Run time.
        connections: Concurrent connections in this process.
        results: Receives the number of successful reads.
    """
    import asyncio

    import httpx

    async def run() -> int:
        done = 0
        limits = httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
        async with httpx.AsyncClient(base_url=url, limits=limits, timeout=10) as http:
            deadline = time.monotonic() + seconds

            async def worker(offset: int) -> None:
                nonlocal done
                i = offset
                while time.monotonic() < deadline:
                    response = await http.get("/retrieve", params={"content_hash": hashes[i % len(hashes)]})
                    if response.status_code == 200:
                        done += 1
                    i += 7

            await asyncio.gather(*(worker(n) for n in range(connections)))
        return done

    results.put(asyncio.run(run()))


def main() -> None:
    """Run the benchmark and print the result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter for membrane serve")
    parser.add_argument("--http-threads", type=int, default=1)
    parser.add_argument("--clients", type=int, default=os.cpu_count() or 4, help="client processes")
    parser.add_argument("--connections", type=int, default=16, help="connections per client process")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        args.python, "-m", "membrane", "serve", "--daemon", "--port", str(port),
        "--http-threads", str(args.http_threads), "--max-concurrency", "0", "--log-level", "WARNING",
    ]  # fmt: skip
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(url, proc)
        hashes = seed(url)
        results: multiprocessing.Queue = multiprocessing.Queue()
        workers = [
            multiprocessing.Process(target=client, args=(url, hashes, args.seconds, args.connections, results))
            for _ in range(args.clients)
        ]
        for worker in workers:
            worker.start()
        total = sum(results.get() for _ in workers)
        for worker in workers:
            worker.join()
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    result = {"http_threads": args.http_threads, "python": args.python, "requests_per_sec": total / args.seconds}
    logger.info("%s", json.dumps(result))


if __name__ == "__main__":
    main()
