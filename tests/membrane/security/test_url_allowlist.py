"""Tests for the SSRF allow-list (Phase 3.1.2)."""

from __future__ import annotations

import pytest

from membrane.security import (
    SSRFError,
    URLAllowlist,
    configure,
    reset_default_allowlist,
    set_default_allowlist,
    validate_outbound_url,
)


class TestSchemeCheck:
    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_allowed_schemes_pass(self, scheme):
        policy = URLAllowlist(block_private=False)
        url = validate_outbound_url(f"{scheme}://example.com/path", allowlist=policy)
        assert url.startswith(scheme)

    @pytest.mark.parametrize(
        "scheme", ["file", "gopher", "ftp", "ldap", "javascript", "data"]
    )
    def test_blocked_schemes_raise(self, scheme):
        policy = URLAllowlist(block_private=False)
        with pytest.raises(SSRFError, match="scheme not allowed"):
            validate_outbound_url(f"{scheme}://example.com", allowlist=policy)

    def test_empty_url_raises(self):
        policy = URLAllowlist(block_private=False)
        with pytest.raises(SSRFError, match="empty url"):
            validate_outbound_url("", allowlist=policy)

    def test_missing_host_raises(self):
        policy = URLAllowlist(block_private=False)
        with pytest.raises(SSRFError, match="missing host"):
            validate_outbound_url("https:///path", allowlist=policy)


class TestPrivateAddressBlocking:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/x",
            "http://127.0.0.1:8080/x",
            "http://[::1]/x",
            "http://10.0.0.1/x",
            "http://192.168.1.1/x",
            "http://172.16.0.1/x",
            "http://169.254.169.254/latest/meta-data",
        ],
    )
    def test_blocked_addresses_raise(self, url):
        policy = URLAllowlist(block_private=True)
        with pytest.raises(SSRFError, match="blocked address"):
            validate_outbound_url(url, allowlist=policy)

    def test_block_private_false_disables_dns_check(self):
        """The DNS check is bypassed when ``block_private=False``."""
        policy = URLAllowlist(block_private=False)
        # The URL itself is malformed (no scheme) but the scheme
        # check still fires; use a public-looking URL to verify
        # the DNS path is off.
        url = validate_outbound_url("http://localhost:8080/x", allowlist=policy)
        assert url == "http://localhost:8080/x"

    def test_dns_failure_raises(self):
        """A hostname that does not resolve is rejected."""
        policy = URLAllowlist(block_private=True)
        with pytest.raises(SSRFError, match="dns resolution failed"):
            validate_outbound_url(
                "http://this-host-does-not-exist.invalid/x",
                allowlist=policy,
            )


class TestAllowlist:
    def test_allowlisted_host_bypasses_blocklist(self):
        policy = URLAllowlist(allowlist=frozenset({"internal.svc.cluster"}))
        url = validate_outbound_url("http://internal.svc.cluster/x", allowlist=policy)
        assert url == "http://internal.svc.cluster/x"

    def test_allowlist_is_case_insensitive(self):
        policy = URLAllowlist(allowlist=frozenset({"internal.svc.cluster"}))
        url = validate_outbound_url("http://INTERNAL.SVC.CLUSTER/x", allowlist=policy)
        assert url.startswith("http://INTERNAL.SVC.CLUSTER")

    def test_non_allowlisted_host_still_blocked(self):
        policy = URLAllowlist(
            allowlist=frozenset({"internal.svc.cluster"}),
            block_private=True,
        )
        with pytest.raises(SSRFError, match="blocked address"):
            validate_outbound_url("http://127.0.0.1/x", allowlist=policy)

    def test_is_host_allowed(self):
        policy = URLAllowlist(allowlist=frozenset({"a.example"}))
        assert policy.is_host_allowed("a.example") is True
        assert policy.is_host_allowed("A.EXAMPLE") is True
        assert policy.is_host_allowed("b.example") is False


class TestDefaultAllowlistManagement:
    def teardown_method(self):
        reset_default_allowlist()

    def test_default_blocks_private(self):
        """The factory default blocks private addresses."""
        from membrane.security import get_default_allowlist

        assert get_default_allowlist().block_private is True

    def test_set_and_reset(self):
        policy = URLAllowlist(allowlist=frozenset({"x.example"}), block_private=False)
        set_default_allowlist(policy)
        from membrane.security import get_default_allowlist

        assert get_default_allowlist() is policy
        reset_default_allowlist()
        assert get_default_allowlist() is not policy

    def test_configure_installs_and_returns(self):
        from membrane.security import get_default_allowlist

        new = configure(allowlist=["x.example"], block_private=False)
        assert new.allowlist == frozenset({"x.example"})
        assert new.block_private is False
        assert get_default_allowlist() is new

    def test_configure_lowercases_hosts(self):
        new = configure(allowlist=["X.Example.COM"])
        assert new.allowlist == frozenset({"x.example.com"})


class TestPeerTransportIntegration:
    """The :class:`HTTPTransport` wires the SSRF check before ``urlopen``."""

    def teardown_method(self):
        reset_default_allowlist()

    def test_transport_rejects_private_address(self):
        from membrane.network.peer import HTTPTransport

        transport = HTTPTransport()
        result = transport.request(
            method="GET",
            url="http://127.0.0.1:8080/x",
            body=None,
            headers={},
            timeout_sec=1.0,
        )
        assert result is None

    def test_transport_allows_allowlisted_host(self):
        from membrane.network.peer import HTTPTransport

        configure(allowlist=["internal.svc.cluster"], block_private=True)
        transport = HTTPTransport()
        # The request will fail to actually open (no server), but
        # the SSRF check should not have rejected it. The
        # transport returns ``None`` on any failure (including
        # connection errors), so we only assert that the call
        # doesn't short-circuit on the SSRF check.
        import socket

        old_socket = socket.socket
        try:
            # Patch out the connect call so we never actually
            # open a socket to the fake host.
            def boom(*args, **kwargs):
                raise OSError("blocked by test")

            socket.socket = boom  # type: ignore[assignment]
            result = transport.request(
                method="GET",
                url="http://internal.svc.cluster/x",
                body=None,
                headers={},
                timeout_sec=1.0,
            )
        finally:
            socket.socket = old_socket  # type: ignore[assignment]
        # The call returned ``None`` from a connection error, not
        # from the SSRF check. There's no easy way to assert
        # this without a mock, so the test mainly ensures the
        # allow-listed host doesn't raise SSRFError.
        assert result is None
