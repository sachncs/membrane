"""Every script and example a new user is pointed at must run.

Several demos had silently broken against API changes; running them in
the test suite keeps the onboarding path honest.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SELF_CONTAINED = [
    "scripts/demo.py",
    "scripts/demo_full.py",
    "scripts/demo_membrane.py",
    "scripts/demo_quantization.py",
    "examples/vllm_e2e.py",
]


def _run(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("script", SELF_CONTAINED)
def test_script_runs(script: str) -> None:
    result = _run(script)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() or result.stderr.strip(), "script produced no output"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_rag_pipeline_against_live_server() -> None:
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "membrane", "serve", "--daemon", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline or server.poll() is not None:
                    pytest.fail("membrane serve did not come up")
                time.sleep(0.2)
        result = _run("examples/rag_pipeline.py", {"MEMBRANE_URL": f"http://127.0.0.1:{port}"})
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stderr.count("hit") == 2
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_demo_matches_published_numbers() -> None:
    """site/src/components/Performance.astro quotes these; update both together."""
    result = _run("scripts/demo.py")
    out = result.stdout + result.stderr
    for expected in ("Throughput gain      : 1.39x", "Mean TTFT reduction  : 42%", "P90 TTFT reduction   : 10%"):
        assert expected in out, f"demo output changed; update the site: {expected!r} missing"
    assert "Avg egress bandwidth : 4.8 Gbps" in out


def _cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "membrane", *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _start_node(port: int, *extra: str) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "membrane", "serve", "--daemon", "--port", str(port), *extra],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=1)
            return proc
        except OSError:
            if time.monotonic() > deadline or proc.poll() is not None:
                proc.kill()
                pytest.fail("membrane serve did not come up")
            time.sleep(0.2)


def test_quickstart_cli_flow(tmp_path: Path) -> None:
    """docs/getting-started.md steps 2, 3 and 5."""
    import json

    port = _free_port()
    node = _start_node(port)
    base = f"http://127.0.0.1:{port}"
    try:
        assert _cli("client", "prefill", "--prompt-tokens", "1 2 3 4 5 6 7 8", "--base-url", base).returncode == 0
        digest = json.loads(_cli("client", "inventory", "--base-url", base).stdout)["digest"]
        assert list(digest) == ["b5ca3ba695d8ec8ce47bf6e7a2b579d0"]
        found = json.loads(_cli("client", "retrieve", "--hash", next(iter(digest)), "--base-url", base).stdout)
        assert found["found"] is True
    finally:
        node.terminate()
        node.wait(timeout=30)

    refused = _cli("serve", "--host", "0.0.0.0", "--daemon", "--port", str(_free_port()))
    assert refused.returncode == 2
    assert "Refusing to serve unauthenticated" in refused.stderr

    keyfile = tmp_path / "api-keys"
    keyfile.write_text("quickstart-key:acme:read,write\n")
    keyfile.chmod(0o600)
    port = _free_port()
    node = _start_node(port, "--api-key-file", str(keyfile))
    base = f"http://127.0.0.1:{port}"
    try:
        assert _cli("client", "inventory", "--base-url", base).returncode != 0
        authed = _cli("client", "inventory", "--base-url", base, "--api-key", "quickstart-key")
        assert authed.returncode == 0, authed.stderr
    finally:
        node.terminate()
        node.wait(timeout=30)


def test_quickstart_local_cluster_forms() -> None:
    """docs/getting-started.md step 6: three local nodes find each other."""
    import json

    ports = [_free_port() for _ in range(3)]
    nodes = []
    try:
        for i, port in enumerate(ports):
            peers = [arg for other in ports if other != port for arg in ("--peer", f"localhost:{other}")]
            nodes.append(_start_node(port, "--node-id", f"n{i + 1}", "--heartbeat-interval", "0.5", *peers))
        deadline = time.monotonic() + 45
        while True:
            healthy = []
            for port in ports:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/peers", timeout=2).read()
                healthy.append(sum(1 for p in json.loads(body)["peers"] if p["healthy"]))
            if healthy == [2, 2, 2]:
                break
            if time.monotonic() > deadline:
                pytest.fail(f"cluster did not converge: healthy peer counts {healthy}")
            time.sleep(0.5)
    finally:
        for node in nodes:
            node.terminate()
        for node in nodes:
            node.wait(timeout=30)


def test_sigterm_drains_and_exits_cleanly(tmp_path: Path) -> None:
    """Containers stop the server with SIGTERM: it must drain, log it, and exit 0."""
    import signal

    port = _free_port()
    log = tmp_path / "serve.log"
    with log.open("wb") as sink:
        proc = subprocess.Popen(
            [sys.executable, "-m", "membrane", "serve", "--daemon", "--port", str(port), "--drain-timeout", "5"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=sink,
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    proc.kill()
                    pytest.fail("membrane serve did not come up")
                time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)
        try:
            assert proc.wait(timeout=15) == 0
        finally:
            proc.kill()
    text = log.read_text()
    assert "SIGTERM received; draining" in text
    assert "drain complete" in text


def test_replicas_serve_kv_bytes_after_primary_loss() -> None:
    """Replication copies the KV bytes, so a replica can serve a fragment after its primary dies."""
    import json
    import signal

    ports = [_free_port() for _ in range(3)]
    nodes = []

    def call(port: int, path: str, body: dict | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    try:
        for i, port in enumerate(ports):
            peers = [arg for other in ports if other != port for arg in ("--peer", f"localhost:{other}")]
            nodes.append(
                _start_node(
                    port, "--node-id", f"n{i + 1}", "--heartbeat-interval", "0.5", "--gossip-interval", "0.5", *peers
                )
            )
        status, body = call(ports[0], "/prefill", {"prompt_tokens": list(range(512)), "model_id": "e2e"})
        assert status == 200
        fragments = json.loads(body)["fragments"]
        assert fragments and all(f["payload_ref"] for f in fragments)
        originals = {f["payload_ref"]: call(ports[0], f"/blobs/{f['payload_ref']}")[1] for f in fragments}

        def replicated(frag: dict) -> bool:
            content_hash = frag["identity"]["payload_hash"]
            for port in ports[1:]:
                found = json.loads(call(port, f"/retrieve?content_hash={content_hash}")[1]).get("found")
                if found:
                    return True
            return False

        deadline = time.monotonic() + 60
        while not all(replicated(f) for f in fragments):
            if time.monotonic() > deadline:
                pytest.fail("fragments were not replicated with their bytes")
            time.sleep(0.5)

        nodes[0].send_signal(signal.SIGKILL)
        nodes[0].wait(timeout=10)
        for frag in fragments:
            ref = frag["payload_ref"]
            served = [call(port, f"/blobs/{ref}") for port in ports[1:]]
            assert any(status == 200 and data == originals[ref] for status, data in served), ref
    finally:
        for node in nodes:
            if node.poll() is None:
                node.terminate()
        for node in nodes:
            node.wait(timeout=30)


def test_sigterm_drain_hands_primaries_to_peers(tmp_path: Path) -> None:
    """A draining node hands every primary, bytes included, to a peer before exiting."""
    import json
    import re
    import signal

    ports = [_free_port() for _ in range(2)]
    log = tmp_path / "n1.log"
    procs = []
    try:
        with log.open("wb") as sink:
            for i, port in enumerate(ports):
                other = ports[1 - i]
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "membrane",
                            "serve",
                            "--daemon",
                            "--port",
                            str(port),
                            "--node-id",
                            f"n{i + 1}",
                            "--peer",
                            f"localhost:{other}",
                            "--heartbeat-interval",
                            "0.5",
                            "--gossip-interval",
                            "0.5",
                            "--drain-timeout",
                            "20",
                        ],
                        cwd=ROOT,
                        stdout=subprocess.DEVNULL,
                        stderr=sink if i == 0 else subprocess.DEVNULL,
                    )
                )
            deadline = time.monotonic() + 45
            while True:
                try:
                    peers = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{ports[0]}/peers", timeout=2).read())
                    if any(p["healthy"] for p in peers["peers"]):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    pytest.fail("cluster did not form")
                time.sleep(0.5)
            req = urllib.request.Request(
                f"http://127.0.0.1:{ports[0]}/prefill",
                data=json.dumps({"prompt_tokens": list(range(512)), "model_id": "drain"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            fragments = json.loads(urllib.request.urlopen(req, timeout=10).read())["fragments"]
            procs[0].send_signal(signal.SIGTERM)
            assert procs[0].wait(timeout=60) == 0
        match = re.search(r"drain complete: \{'migrated': (\d+), 'stragglers': (\d+)", log.read_text())
        assert match, log.read_text()[-2000:]
        assert int(match.group(2)) == 0
        assert int(match.group(1)) >= 1
        for frag in fragments:
            ref = frag["payload_ref"]
            with urllib.request.urlopen(f"http://127.0.0.1:{ports[1]}/blobs/{ref}", timeout=5) as resp:
                assert resp.status == 200
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            proc.wait(timeout=30)
