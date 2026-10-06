"""Cluster load and verification for the kind end-to-end test (scripts/kind_e2e.sh).

Runs inside the cluster, from the Membrane image.

``write`` mode issues strong writes. Each write uploads deterministic KV
bytes (``PUT /blobs``) and then the fragment (``POST /store``) to the
same pod (the bytes must be on the node that stores the fragment), and
moves to the next pod on a retryable failure, as a cluster-aware client
would. It ends with a ``SUMMARY`` line holding the counts and a
sample of written hashes.

``verify`` mode reads those hashes back from each pod directly, through
the headless Service, and checks that at least one pod still holds the
fragment with byte-identical KV data.

``seed`` mode writes ``--count`` fragments the same way and stops.

``capacity`` mode measures how many reads per second the cluster serves.
Each pod gets its own client process with ``--connections`` keep-alive
connections, each reading fragments that pod holds (from its
``/inventory``), so every pod is driven to its limit independently and the
total is the cluster's read capacity. It ends with a ``CAPACITY`` line.

Usage::

    python kind_load.py write --duration 120 --key KEY
    python kind_load.py verify --hashes h1,h2,... --key KEY --pods 3 --skip 0
    python kind_load.py seed --count 600 --key KEY --pods 3
    python kind_load.py capacity --duration 30 --key KEY --pods 5 --connections 8
"""

import argparse
import hashlib
import http.client
import json
import logging
import multiprocessing
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity
from membrane.serialization import to_dict

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("kind_load")

NAMESPACE = os.environ.get("POD_NAMESPACE", "default")
PAYLOAD_BYTES = 4096


def pod_host(ordinal: int) -> str:
    """Return the DNS name of StatefulSet pod ``ordinal``.

    Args:
        ordinal: Pod ordinal.

    Returns:
        str: ``membrane-N.membrane-headless.<namespace>.svc.cluster.local``.
    """
    return f"membrane-{ordinal}.membrane-headless.{NAMESPACE}.svc.cluster.local"


def pod_url(ordinal: int) -> str:
    """Return the base URL of StatefulSet pod ``ordinal``.

    Args:
        ordinal: Pod ordinal.

    Returns:
        str: ``http://membrane-N.membrane-headless...:8080``.
    """
    return f"http://{pod_host(ordinal)}:8080"


def payload_for(content_hash: str) -> bytes:
    """Return the deterministic KV bytes written for ``content_hash``.

    Args:
        content_hash: Fragment content hash.

    Returns:
        bytes: :data:`PAYLOAD_BYTES` bytes derived from the hash.
    """
    out = bytearray()
    counter = 0
    while len(out) < PAYLOAD_BYTES:
        out += hashlib.sha256(f"{content_hash}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:PAYLOAD_BYTES])


def fragment(content_hash: str) -> Fragment:
    """Build a strong-consistency fragment whose payload is :func:`payload_for`.

    Args:
        content_hash: Fragment content hash.

    Returns:
        Fragment: The fragment.
    """
    identity = PayloadIdentity(
        payload_hash=content_hash,
        model_id="kind",
        model_revision="",
        tokenizer_name="kind",
        tokenizer_revision="",
        layer_range=(0, 1),
        head_range=(-1, -1),
        token_span=(0, 1),
        dtype="float16",
        shape=(1, 1, 1, 32, 64),
    )
    return Fragment(
        identity=identity,
        payload_ref=f"kv-{content_hash}",
        payload_size=PAYLOAD_BYTES,
        ttl=3600.0,
        reuse_score=0.5,
        version_id=1,
        consistency="strong",
    )


def request(
    method: str, url: str, key: str, body: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, bytes, str | None]:
    """Issue one HTTP request.

    Args:
        method: HTTP method.
        url: Full URL.
        key: Bearer API key.
        body: Request body.
        headers: Extra headers.

    Returns:
        tuple[int, bytes, str | None]: Status (0 on a network error), body,
        and the ``Retry-After`` header.
    """
    req = urllib.request.Request(
        url, data=body, method=method, headers={"Authorization": f"Bearer {key}", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Retry-After")
    except OSError:
        return 0, b"", None


def write_one(content_hash: str, key: str, pods: int) -> tuple[bool, int]:
    """Write one fragment with its bytes, retrying retryable failures on the next pod.

    Args:
        content_hash: Fragment content hash.
        key: Bearer API key (admin: blob uploads are peer-scoped).
        pods: Number of StatefulSet pods.

    Returns:
        tuple[bool, int]: Success, and the number of retries used.
    """
    frag = fragment(content_hash)
    data = payload_for(content_hash)
    digest = hashlib.sha256(data).hexdigest()
    store_body = json.dumps({"fragment": to_dict(frag), "is_primary": True}).encode()
    first = int(content_hash[:8], 16) % pods
    for attempt in range(10):
        base = pod_url((first + attempt) % pods)
        status, _, retry_after = request(
            "PUT", f"{base}/blobs/{frag.payload_ref}", key, data, {"X-Content-SHA256": digest}
        )
        if status == 200:
            status, body, retry_after = request(
                "POST", f"{base}/store", key, store_body, {"Content-Type": "application/json"}
            )
            if status == 200 and json.loads(body).get("success") is True:
                return True, attempt
        if status not in (0, 502, 503, 504):
            log.warning("write %s failed with HTTP %s", content_hash, status)
            return False, attempt
        time.sleep(float(retry_after or 0.5))
    return False, 10


def run_write(duration: float, key: str, pods: int) -> int:
    """Write continuously for ``duration`` seconds.

    Args:
        duration: Seconds to write for.
        key: Bearer API key.
        pods: Number of StatefulSet pods.

    Returns:
        int: Exit status (0 when no write failed).
    """
    start = time.monotonic()
    written: list[str] = []
    failures = retries = 0
    last_report = start
    i = 0
    while time.monotonic() - start < duration:
        content_hash = hashlib.sha256(f"kind-{start}-{i}".encode()).hexdigest()
        ok, used = write_one(content_hash, key, pods)
        retries += used
        if ok:
            written.append(content_hash)
        else:
            failures += 1
        i += 1
        if time.monotonic() - last_report >= 10:
            last_report = time.monotonic()
            log.info(json.dumps({"t": round(last_report - start), "writes": i, "failures": failures}))
    sample = written[:: max(1, len(written) // 50)][:50]
    summary = {"writes": i, "ok": len(written), "failures": failures, "retries": retries, "sample": sample}
    log.info("SUMMARY %s", json.dumps(summary))
    return 0 if failures == 0 and written else 1


def run_verify(hashes: list[str], key: str, pods: int, skip: set[int]) -> int:
    """Check that every hash is still served, with identical bytes, by some pod.

    Args:
        hashes: Content hashes written earlier.
        key: Bearer API key.
        pods: Number of StatefulSet pods.
        skip: Pod ordinals not to ask (e.g. a pod that was just killed).

    Returns:
        int: Exit status (0 when every hash is served).
    """
    missing = []
    for content_hash in hashes:
        ref = fragment(content_hash).payload_ref
        expected = payload_for(content_hash)
        served = False
        for ordinal in (o for o in range(pods) if o not in skip):
            status, body, _ = request("GET", f"{pod_url(ordinal)}/blobs/{ref}", key)
            if status == 200 and body == expected:
                served = True
                break
        if not served:
            missing.append(content_hash)
    log.info("VERIFY %s", json.dumps({"checked": len(hashes), "missing": len(missing)}))
    return 0 if not missing else 1


def run_seed(count: int, key: str, pods: int) -> int:
    """Write ``count`` fragments with their bytes.

    Args:
        count: Fragments to write.
        key: Bearer API key.
        pods: Number of StatefulSet pods.

    Returns:
        int: Exit status (0 when every write succeeded).
    """
    failures = 0
    for i in range(count):
        ok, _ = write_one(hashlib.sha256(f"seed-{i}".encode()).hexdigest(), key, pods)
        failures += not ok
    log.info("SEED %s", json.dumps({"written": count - failures, "failures": failures}))
    return 0 if failures == 0 else 1


def holdings(ordinal: int, key: str) -> list[str]:
    """Return the fragment keys pod ``ordinal`` holds.

    Args:
        ordinal: Pod ordinal.
        key: Bearer API key.

    Returns:
        list[str]: Keys from its ``/inventory``.
    """
    status, body, _ = request("GET", f"{pod_url(ordinal)}/inventory", key)
    return sorted(json.loads(body)["digest"]) if status == 200 else []


def read_pod(
    ordinal: int, keys: list[str], key: str, connections: int, warmup: float, duration: float
) -> tuple[int, int]:
    """Read fragments from one pod as fast as it answers (runs in its own process).

    Args:
        ordinal: Pod ordinal.
        keys: Fragment keys the pod holds.
        key: Bearer API key.
        connections: Concurrent keep-alive connections.
        warmup: Seconds to read before counting.
        duration: Seconds to count.

    Returns:
        tuple[int, int]: Reads answered ``found`` and failed reads, in the window.
    """
    start = time.monotonic() + warmup
    stop = start + duration
    counts = [[0, 0] for _ in range(connections)]
    headers = {"Authorization": f"Bearer {key}"}

    def reader(slot: int) -> None:
        rnd = random.Random(slot)
        conn = http.client.HTTPConnection(pod_host(ordinal), 8080, timeout=15)
        while (now := time.monotonic()) < stop:
            try:
                conn.request("GET", f"/retrieve?content_hash={rnd.choice(keys)}", headers=headers)
                response = conn.getresponse()
                body = response.read()
                ok = response.status == 200 and (b'"found":true' in body or b'"found": true' in body)
            except OSError, http.client.HTTPException:
                conn.close()
                conn = http.client.HTTPConnection(pod_host(ordinal), 8080, timeout=15)
                ok = False
            if now >= start:
                counts[slot][0 if ok else 1] += 1
        conn.close()

    threads = [threading.Thread(target=reader, args=(slot,)) for slot in range(connections)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(c[0] for c in counts), sum(c[1] for c in counts)


def run_capacity(key: str, pods: int, duration: float, connections: int) -> int:
    """Measure the cluster's read capacity: every pod driven to its limit at once.

    Args:
        key: Bearer API key.
        pods: Number of StatefulSet pods.
        duration: Seconds to count.
        connections: Keep-alive connections per pod.

    Returns:
        int: Exit status (0 when every pod served reads).
    """
    held = [holdings(ordinal, key) for ordinal in range(pods)]
    if not all(held):
        log.error("pods without fragments: %s", [o for o, keys in enumerate(held) if not keys])
        return 1
    with multiprocessing.Pool(pods) as pool:
        results = pool.starmap(read_pod, [(o, held[o], key, connections, 5.0, duration) for o in range(pods)])
    per_pod = [round(found / duration) for found, _ in results]
    errors = sum(failed for _, failed in results)
    summary = {"pods": pods, "rps": sum(per_pod), "per_pod": per_pod, "held": [len(k) for k in held], "errors": errors}
    log.info("CAPACITY %s", json.dumps(summary))
    return 0 if all(per_pod) else 1


def main() -> int:
    """Parse arguments and run the selected mode.

    Returns:
        int: Exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("write", "verify", "seed", "capacity"))
    parser.add_argument("--key", default=os.environ.get("KEY", ""))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--hashes", default="")
    parser.add_argument("--pods", type=int, default=3)
    parser.add_argument("--skip", default="", help="comma-separated pod ordinals to leave out of verify")
    parser.add_argument("--count", type=int, default=600, help="fragments to write (seed)")
    parser.add_argument("--connections", type=int, default=8, help="keep-alive connections per pod (capacity)")
    args = parser.parse_args()
    if args.mode == "write":
        return run_write(args.duration, args.key, args.pods)
    if args.mode == "seed":
        return run_seed(args.count, args.key, args.pods)
    if args.mode == "capacity":
        return run_capacity(args.key, args.pods, args.duration, args.connections)
    skip = {int(o) for o in args.skip.split(",") if o}
    return run_verify([h for h in args.hashes.split(",") if h], args.key, args.pods, skip)


if __name__ == "__main__":
    sys.exit(main())
