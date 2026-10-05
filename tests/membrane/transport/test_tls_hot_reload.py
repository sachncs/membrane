"""A running mTLS server serves a rotated certificate without a restart."""

import socket
import ssl
import time
from pathlib import Path

from cryptography import x509

from membrane.node import Node
from membrane.server import Server
from membrane.transport.tls import MTLSConfig
from tests.tls_helpers import ca_pem, issue, make_ca


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def served_serial(port: int, ca: str, client_cert: Path, client_key: Path) -> int:
    context = ssl.create_default_context(cadata=ca)
    context.check_hostname = False
    context.load_cert_chain(client_cert, client_key)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw, context.wrap_socket(raw) as tls:
        der = tls.getpeercert(binary_form=True)
    assert der is not None
    return x509.load_der_x509_certificate(der).serial_number


def test_rotated_certificate_is_served_without_restart(tmp_path: Path) -> None:
    ca = make_ca()
    cert1, key1, serial1 = issue(ca, "admin-n1")
    cert_file, key_file = tmp_path / "node.crt", tmp_path / "node.key"
    cert_file.write_text(cert1)
    key_file.write_text(key1)
    key_file.chmod(0o600)
    tls = MTLSConfig.allow_all_signed_by_ca(
        server_cert_pem=cert1, server_key_pem=key1, ca_bundle_pem=ca_pem(ca), client_cert_pem=cert1, client_key_pem=key1
    )
    port = free_port()
    server = Server(node=Node("n1"), host="127.0.0.1", port=port, tls=tls, tls_files=(str(cert_file), str(key_file)))
    server.start()
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                first = served_serial(port, ca_pem(ca), cert_file, key_file)
                break
            except OSError:
                assert time.monotonic() < deadline, "server did not start"
                time.sleep(0.1)
        assert first == serial1

        cert2, key2, serial2 = issue(ca, "admin-n1")
        cert_file.write_text(cert2)
        key_file.write_text(key2)
        assert server.cert_watcher is not None
        server.cert_watcher.reload()  # what the poll loop and SIGHUP do
        assert served_serial(port, ca_pem(ca), cert_file, key_file) == serial2
        assert server.tls is not None and server.tls.server_cert_pem == cert2
        server.refresh_metrics()
        assert server.metrics_transport.tls_cert_expiry.value > 20 * 86400
    finally:
        server.stop(2.0)


def test_expired_certificate_is_rejected_on_reload(tmp_path: Path) -> None:
    ca = make_ca()
    good, good_key, _ = issue(ca, "admin-n1")
    expired, expired_key, _ = issue(ca, "admin-n1", days=-1)
    rotated: list[str] = []
    from membrane.transport.tls_rotation import CertRotationWatcher

    cert_file, key_file = tmp_path / "c", tmp_path / "k"
    cert_file.write_text(good)
    key_file.write_text(good_key)
    watcher = CertRotationWatcher(str(cert_file), str(key_file), on_rotate=lambda c, k: rotated.append(c))
    watcher.reload()
    cert_file.write_text(expired)
    key_file.write_text(expired_key)
    watcher.reload()
    assert rotated == [good]
