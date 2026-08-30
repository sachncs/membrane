"""Tests for the per-route scope wiring (Phase 3.1.1).

The v3.0.0 release wires :func:`membrane.auth.require_scope` into
every HTTP route via :mod:`membrane.transport.authz`. The
:class:`ROUTE_SCOPES` table is the single source of truth for
which scope a given ``(method, path)`` requires.
"""

from __future__ import annotations

import pytest

from membrane.auth import (
    AuthBackendError,
    AuthContext,
    AuthRequest,
)
from membrane.auth.apikey import APIKeyAuthenticator, NoopAuthenticator
from membrane.transport.authz import (
    DEFAULT_SCOPE,
    ROUTE_SCOPES,
    enforce_route_scope,
    required_scope,
)


class _StubAuthenticator:
    """Authenticator that always returns the configured context."""

    def __init__(self, context: AuthContext) -> None:
        self._context = context

    def authenticate(self, request: AuthRequest) -> AuthContext:
        return self._context


class TestRequiredScope:
    def test_public_probes(self):
        assert required_scope("GET", "/livez") == "public"
        assert required_scope("GET", "/readyz") == "public"

    def test_reads(self):
        assert required_scope("GET", "/heartbeat") == "read"
        assert required_scope("GET", "/retrieve") == "read"
        assert required_scope("GET", "/inventory") == "read"
        assert required_scope("GET", "/peers") == "read"
        assert required_scope("GET", "/metrics") == "read"

    def test_writes(self):
        assert required_scope("POST", "/store") == "write"
        assert required_scope("POST", "/replicate") == "write"
        assert required_scope("POST", "/prefill") == "write"
        assert required_scope("POST", "/sync") == "write"
        assert required_scope("POST", "/gossip") == "write"
        assert required_scope("POST", "/join") == "write"
        assert required_scope("POST", "/leave") == "write"

    def test_admin(self):
        assert required_scope("POST", "/delete") == "admin"
        assert required_scope("POST", "/tombstone") == "admin"
        assert required_scope("POST", "/purge") == "admin"
        assert required_scope("POST", "/verify") == "admin"

    def test_unknown_route_fails_closed_to_read(self):
        """A route not in the table defaults to ``read`` to fail closed."""
        assert required_scope("GET", "/new-endpoint") == DEFAULT_SCOPE
        assert DEFAULT_SCOPE == "read"

    def test_path_normalized(self):
        assert required_scope("get", "livez") == "public"
        assert required_scope("POST", "store") == "write"


class TestEnforceRouteScope:
    def test_no_authenticator_bypasses(self):
        """When no authenticator is configured the helper returns empty."""
        ctx = enforce_route_scope(None, "GET", "/retrieve")
        assert ctx.scopes == frozenset()

    def test_public_probes_bypass_even_with_authenticator(self):
        auth = _StubAuthenticator(AuthContext(subject="n", scopes=frozenset({"read"})))
        ctx = enforce_route_scope(auth, "GET", "/livez", headers={})
        assert ctx.subject == ""

    def test_admin_scope_accepted_for_admin_holder(self):
        auth = _StubAuthenticator(
            AuthContext(subject="root", scopes=frozenset({"admin"}))
        )
        ctx = enforce_route_scope(auth, "POST", "/delete", headers={})
        assert ctx.subject == "root"

    def test_read_scope_rejected_for_write_holder(self):
        auth = _StubAuthenticator(
            AuthContext(subject="reader", scopes=frozenset({"read"}))
        )
        with pytest.raises(AuthBackendError, match="missing required scope"):
            enforce_route_scope(auth, "POST", "/store", headers={})

    def test_admin_scope_required_for_admin_endpoint(self):
        auth = _StubAuthenticator(
            AuthContext(subject="writer", scopes=frozenset({"read", "write"}))
        )
        with pytest.raises(AuthBackendError, match="missing required scope"):
            enforce_route_scope(auth, "POST", "/delete", headers={})

    def test_hierarchical_scope_expansion(self):
        """An admin holder passes a read check via the SCOPES hierarchy."""
        auth = _StubAuthenticator(
            AuthContext(subject="root", scopes=frozenset({"admin"}))
        )
        ctx = enforce_route_scope(auth, "GET", "/retrieve", headers={})
        assert ctx.subject == "root"


class TestApiKeyAuthenticatorIntegration:
    def test_api_key_holder_passes_write_check(self):
        """APIKeyAuthenticator grants the key's scopes; integration test."""
        auth = APIKeyAuthenticator(
            keyfile_text="test-key:reader1:read\nadmin-key:admin1:admin\n"
        )
        ctx_read = enforce_route_scope(
            auth, "GET", "/retrieve", headers={"authorization": "Bearer test-key"}
        )
        assert "read" in ctx_read.scopes
        ctx_admin = enforce_route_scope(
            auth, "POST", "/delete", headers={"authorization": "Bearer admin-key"}
        )
        assert "admin" in ctx_admin.scopes

    def test_api_key_holder_rejected_for_admin_when_read_only(self):
        auth = APIKeyAuthenticator(keyfile_text="ro-key:reader1:read\n")
        with pytest.raises(AuthBackendError):
            enforce_route_scope(
                auth, "POST", "/delete", headers={"authorization": "Bearer ro-key"}
            )

    def test_api_key_missing_header_rejected(self):
        auth = APIKeyAuthenticator(keyfile_text="k:s:r\n")
        with pytest.raises(AuthBackendError, match="unauthorized"):
            enforce_route_scope(auth, "GET", "/retrieve", headers={})

    def test_api_key_unknown_key_rejected(self):
        auth = APIKeyAuthenticator(keyfile_text="known-key:s:r\n")
        with pytest.raises(AuthBackendError, match="unauthorized"):
            enforce_route_scope(
                auth, "GET", "/retrieve", headers={"authorization": "Bearer unknown-key"}
            )


class TestNoopAuthenticator:
    def test_noop_authenticator_returns_empty_context(self):
        """The NoopAuthenticator returns an empty context for any request.

        NoopAuthenticator is the test sentinel that signals "no
        authentication is configured". It does NOT grant the
        scope hierarchy; production deployments must use
        ``authenticator=None`` (which bypasses the check) rather
        than ``NoopAuthenticator()`` (which exercises the check
        and finds the caller is missing every scope).
        """
        ctx = NoopAuthenticator().authenticate(
            AuthRequest(method="GET", path="/retrieve", headers={}, client="")
        )
        assert ctx.subject == ""
        assert ctx.scopes == frozenset()

    def test_noop_authenticator_fails_scope_check(self):
        """A NoopAuthenticator explicitly wired in fails the scope check."""
        with pytest.raises(AuthBackendError, match="missing required scope"):
            enforce_route_scope(
                NoopAuthenticator(), "POST", "/store", headers={}
            )
