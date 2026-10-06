"""Entry points: ``python -m membrane``, the demo, the container health check, lazy exports."""

import logging
import runpy
import sys
import time

import pytest

import membrane
from membrane import demo, healthcheck
from membrane.node import Node
from membrane.server import Server
from tests.tls_helpers import ca_pem, issue, make_ca


def test_python_dash_m_runs_the_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["membrane", "--help"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("membrane", run_name="__main__")
    assert exited.value.code == 0
    assert "Usage" in capsys.readouterr().out


def test_demo_stores_then_hits(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="membrane.demo"):
        assert demo.main() == 0
    messages = [r.getMessage() for r in caplog.records]
    assert "stored slot=alpha" in messages and "hit slot=alpha" in messages


def test_health_check_answers_from_a_live_server(monkeypatch) -> None:
    server = Server(node=Node("hc"), port=0, load_hooks=False)
    server.start()
    try:
        monkeypatch.delenv("MEMBRANE_TLS_CERT_FILE", raising=False)
        deadline = time.monotonic() + 15
        while True:  # the listener binds its port asynchronously
            monkeypatch.setenv("MEMBRANE_PORT", str(server.transport.port))
            if server.transport.port and healthcheck.main() == 0:
                break
            assert time.monotonic() < deadline, "server never answered /livez"
            time.sleep(0.05)
        assert healthcheck.livez_url()[1] is None
    finally:
        server.stop()


def test_health_check_fails_when_nothing_listens(monkeypatch) -> None:
    monkeypatch.setenv("MEMBRANE_PORT", "1")
    monkeypatch.delenv("MEMBRANE_TLS_CERT_FILE", raising=False)
    assert healthcheck.main() == 1


def test_health_check_presents_the_node_certificate_under_mtls(monkeypatch, tmp_path) -> None:
    ca = make_ca()
    cert, key, _serial = issue(ca, "node")
    (tmp_path / "ca.pem").write_text(ca_pem(ca))
    (tmp_path / "cert.pem").write_text(cert)
    (tmp_path / "key.pem").write_text(key)
    monkeypatch.setenv("MEMBRANE_PORT", "9443")
    monkeypatch.setenv("MEMBRANE_TLS_CERT_FILE", str(tmp_path / "cert.pem"))
    monkeypatch.setenv("MEMBRANE_TLS_KEY_FILE", str(tmp_path / "key.pem"))
    monkeypatch.setenv("MEMBRANE_TLS_CA_FILE", str(tmp_path / "ca.pem"))
    url, context = healthcheck.livez_url()
    assert url == "https://127.0.0.1:9443/livez"
    assert context is not None and context.check_hostname is False


def test_lazy_exports_load_on_first_use() -> None:
    assert "Node" in dir(membrane)
    assert membrane.Node is Node
    with pytest.raises(AttributeError, match="no attribute"):
        _ = membrane.NotAnExport  # type: ignore[attr-defined]
