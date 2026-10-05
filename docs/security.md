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

`membrane serve` refuses to listen on a non-loopback address unless
one of these modes is configured (or `--allow-unauthenticated` is
passed explicitly):

* **API key**: `APIKeyAuthenticator` (`membrane.auth.apikey`),
  enabled with `--api-key-file`. Clients send
  `Authorization: Bearer <key>`. Keyfile lines are
  `<key>:<subject>:<scope,...>`; lines with an empty key or subject
  are ignored, because an empty subject would bypass the tenant check.
* **mTLS**: `MTLSConfig` (`membrane.transport.tls`), enabled with
  `--tls-cert/--tls-key/--tls-ca`. Every connection must present a
  certificate signed by the CA bundle, and `MTLSAuthenticator` admits
  only CNs in `allowed_cns`. The CN is read from the **verified peer
  certificate** of the TLS handshake
  (`membrane.transport.tls_protocol`); a client-supplied
  `X-SSL-Client-CN` header is always discarded. Scopes come from the CN
  prefix (`admin-`, `write-`, `read-`). On `/join` the CN must equal
  the joining node id, optionally with a role prefix.

Authentication failures return `401` with `WWW-Authenticate: Bearer`;
a valid caller without the required scope gets `403`.

## Authorisation

`membrane.transport.authz.ROUTE_SCOPES` maps every route to a scope,
and `enforce_route_scope` runs it at the top of every handler:

| Scope | Routes |
|-------|--------|
| public | `GET /livez`, `GET /readyz` |
| `read` | `GET /retrieve`, `/inventory`, `/peers`, `/heartbeat`, `/metrics`, `/metrics.json` |
| `write` | `POST /store`, `/replicate`, `/prefill`, `/sync`, `/gossip`, `/join`, `/leave` |
| `admin` | `POST /delete`, `/tombstone`, `/purge`, `/verify`, and everything under `/admin/` |

`admin` implies `write` implies `read`. Unlisted routes default to
`read`, so a new route fails closed.

`TenantAuthorizer` (`membrane.security.tenant`) layers a second
check: a fragment carrying `tenant_id="acme"` is readable only by
callers whose `AuthContext.subject` matches `acme`, who carry an
explicit `tenant:acme` scope, or who hold `admin`. A cross-tenant read
is indistinguishable from a miss.

### Peer-to-peer calls

Nodes call each other's routes for join, heartbeat, gossip,
replication, and delete propagation, so they authenticate like any
client: with the mTLS client certificate, or in API-key clusters with
the key from `--peer-api-key-file`, which must carry `admin`.

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

Cluster peers usually live on private addresses. Seed hosts given
with `--peer` are allowed by name, and `--peer-network` (or
`MEMBRANE_PEER_NETWORKS`) admits a CIDR such as the Kubernetes pod
network for peers learned later. A host passes only when every address
it resolves to is inside an allowed network. Keep the range as narrow
as the deployment allows and never include `169.254.0.0/16`.

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
