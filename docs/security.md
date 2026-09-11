# Security & authentication model

Membrane's authentication and authorisation model is split
across four modules; this page is the single landing page
that ties them together.

## Module map

| Concern | Module | Notes |
|---------|--------|-------|
| mTLS / TLS configuration | `membrane.transport.tls` | `MTLSConfig`, `TLSConfig`, peer-CN allow-list. |
| Per-op authorisation (tenant scopes) | `membrane.transport.authz` | `AuthContext`, `require_scopes` decorators. |
| Per-tenant fragment policy | `membrane.security.tenant` | `TenantAuthorizer`, `TenantPolicy`. |
| Outbound URL allow-list (SSRF) | `membrane.security.url_allowlist` | `URLAllowlist`, `validate_outbound_url`. |
| Secrets vault (AWS / GCP / Vault / in-process) | `membrane.secrets` | Backend-agnostic secret fetch. |
| Encryption at rest | `membrane.security.encryption` | AES-256-GCM, `DecryptError`. |
| Key rotation | `membrane.security.key_rotation` | Master-key envelope rotation. |
| Append-only audit log | `membrane.audit` | Hash-chained `AuditRecord`. |
| Authentication protocol | `membrane.auth` | `Authenticator` + `APIKeyAuthenticator`. |

## Authentication

The runtime accepts two authentication modes:

* **API key** — `APIKeyAuthenticator`
  (`membrane.auth.apikey`) checks the `X-Membrane-API-Key`
  header against the configured key set. Suitable for
  service-to-service calls where the caller is already
  inside a trusted boundary.
* **mTLS** — `MTLSConfig`
  (`membrane.transport.tls`) gates inbound HTTP at the
  TLS handshake. Every peer must present a certificate
  signed by the cluster CA, and the certificate's CN must
  appear in `MTLSConfig.allowed_cns`. Production
  multi-node clusters must supply an `MTLSConfig`; the
  v3.0.0 release treats the single-node deployment as
  the only path that may run with `mtls=None`.

The two modes are not mutually exclusive: a single
deployment can require both an API key on the inbound
request and a valid client certificate on the TLS layer.

## Authorisation

`membrane.transport.authz` exposes `AuthContext` (the
parsed identity of the caller) and a `require_scopes`
decorator that enforces per-route scope checks:

* `fragment:read` — read access to `op_get`,
  `op_inventory`, and the public retrieve HTTP route.
* `fragment:write` — write access to `op_store` and the
  public store HTTP route.
* `cluster:admin` — access to the admin endpoints
  (`/admin/snapshot`, `/admin/rotate-keys`, etc.).

`TenantAuthorizer` (`membrane.security.tenant`) layers a
second check: a fragment carrying `tenant_id="acme"` is
readable only by callers whose `AuthContext.subject`
matches `acme` or who carry an explicit `tenant:acme` scope.

## SSRF / outbound URL guard

Every outbound HTTP request from the cluster layer is
routed through `validate_outbound_url`
(`membrane.security.url_allowlist`). The check restricts
the scheme to `http`/`https`, blocks RFC 1918 private
ranges, link-local `169.254.0.0/16`, `127.0.0.0/8`,
`::1`, and the IPv6 ULA `fc00::/7`, and resolves the
hostname to confirm the resolved IP is not on the
blocklist.

The resolver pins the validated IP to the outbound socket
so a DNS-rebinding attack cannot smuggle a private
address into the second resolution (`urllib` /
`httpx` would otherwise re-resolve on their own).

The outbound client disables redirect-following; every
3xx response is surfaced to the caller as an explicit
redirect that must be re-validated.

## Encryption at rest

`membrane.security.encryption` provides AES-256-GCM
authenticated encryption for the canonical payload bytes.
The `Encryption` class is constructed from a master key
(or a `SecretsBackend` that produces one); every payload
is encrypted with a per-record IV and carries a 16-byte
GCM tag.

The storage layer raises `DecryptError` (typed) when the
GCM tag fails to verify; the storage layer's
`EncryptedInProcessBytes.get` and `FilesystemBlob.get`
propagate the error as a typed signal (added in 3.0.1)
so corruption is not silently conflated with a miss.

## Key rotation

`membrane.security.key_rotation` provides envelope
encryption with key versioning: every encrypted payload
carries the key version that produced it, and the
rotation path reads the new master key from the
configured `SecretsBackend` without re-encrypting
existing payloads in place. Old payloads decrypt with
the prior key version until they age out.

## Audit log

`membrane.audit` writes every privileged operation
(`op_store`, `op_delete`, key-rotation events, admin
endpoints) to an append-only, hash-chained `AuditRecord`.
The chain is verifiable end-to-end: the operator can
walk the chain from any point and confirm the digest
matches the prior record's `prev_hash`.

## See also

* `docs/api-stability.md` — stability classification of the
  security modules.
* `docs/operations/incident-response.md` — what to do when
  the audit chain reports a tamper or the URL allow-list
  rejects an outbound call.
* `membrane/security/url_allowlist.py` — outbound URL
  policy.
* `membrane/security/encryption.py` — encryption primitives.
