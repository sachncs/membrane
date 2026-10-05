"""Every script and example a new user is pointed at must run.

Several demos had silently broken against API changes; running them in
the test suite keeps the onboarding path honest.
"""

from __future__ import annotations

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
        assert result.stdout.count("hit") == 2
    finally:
        server.terminate()
        server.wait(timeout=30)
