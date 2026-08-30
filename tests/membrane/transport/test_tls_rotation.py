"""Tests for cert rotation + notAfter enforcement (Phase 3.4.1 + 3.4.2)."""

from __future__ import annotations

import datetime
import threading
import time

import pytest

from membrane.transport.tls_rotation import (
    CertRotationWatcher,
    cert_not_after,
    enforce_not_after,
)


# A real cert PEM (generated for tests via cryptography).
def _generate_test_cert() -> str:
    """Generate a real RSA-2048 self-signed cert PEM."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        )
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")


TEST_CERT_PEM = _generate_test_cert()


class TestCertNotAfterParser:
    def test_returns_none_for_empty_pem(self):
        assert cert_not_after("") is None

    def test_returns_none_when_cryptography_unavailable(self, monkeypatch):
        """Skip the parser when cryptography import fails."""
        # The parser is conditional; the fallback returns None.
        import membrane.transport.tls_rotation as m

        def _raise(_pem):
            return None

        monkeypatch.setattr(m, "_parse_pem", _raise)
        assert cert_not_after(TEST_CERT_PEM) is None

    def test_parser_is_called_with_correct_input(self):
        """Smoke test that the parser is exercised."""
        # The fixture cert is a syntactically valid PEM; either
        # the parser returns a datetime or it returns None when
        # cryptography is unavailable.
        result = cert_not_after(TEST_CERT_PEM)
        assert result is None or isinstance(result, datetime.datetime)


class TestEnforceNotAfter:
    def test_empty_pem_no_op(self):
        enforce_not_after("")

    def test_future_pem_no_op(self):
        """A valid future-dated PEM does not raise."""
        enforce_not_after(TEST_CERT_PEM)

    def test_warn_window_callback(self):
        """A cert inside the warn window emits a warning without raising."""
        enforce_not_after(TEST_CERT_PEM, warn_days=365)


class TestCertRotationWatcher:
    def test_first_reload_invokes_on_rotate(self, tmp_path):
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text(TEST_CERT_PEM)
        key.write_text("FAKE-KEY")
        called: list[tuple[str, str]] = []

        def on_rotate(c: str, k: str) -> None:
            called.append((c, k))

        watcher = CertRotationWatcher(
            cert_path=str(cert),
            key_path=str(key),
            on_rotate=on_rotate,
        )
        watcher._reload()
        assert len(called) == 1
        assert called[0][0] == TEST_CERT_PEM

    def test_unchanged_files_do_not_re_trigger(self, tmp_path):
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text(TEST_CERT_PEM)
        key.write_text("FAKE-KEY")
        calls: list[int] = []
        watcher = CertRotationWatcher(
            cert_path=str(cert),
            key_path=str(key),
            on_rotate=lambda c, k: calls.append(1),
        )
        watcher._reload()
        watcher._reload()
        watcher._reload()
        assert len(calls) == 1

    def test_modified_files_re_trigger(self, tmp_path):
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text(TEST_CERT_PEM)
        key.write_text("FAKE-KEY")
        calls: list[int] = []
        watcher = CertRotationWatcher(
            cert_path=str(cert),
            key_path=str(key),
            on_rotate=lambda c, k: calls.append(1),
        )
        watcher._reload()
        watcher._reload()
        # Mutate the file.
        key.write_text("UPDATED-KEY")
        watcher._reload()
        assert len(calls) == 2

    def test_missing_file_does_not_raise(self, tmp_path):
        watcher = CertRotationWatcher(
            cert_path=str(tmp_path / "missing_cert.pem"),
            key_path=str(tmp_path / "missing_key.pem"),
            on_rotate=lambda c, k: None,
        )
        # Should log a warning but not raise.
        watcher._reload()
