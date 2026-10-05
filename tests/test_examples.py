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
