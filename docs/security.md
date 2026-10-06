# Security & authentication model

This page describes how Membrane authenticates callers, authorizes
them, isolates tenants, and protects data at rest.

## Module map

| Concern | Module | Key names |
|---------|--------|-----------|
| Authentication protocol | `membrane.auth` | `Authenticator`, `AuthContext`, `require_scope` |
| API keys | `membrane.auth.apikey` | `APIKeyAuthenticator`, `generate_key`, `hash_key` |
| mTLS | `membrane.auth.mtls`, `membrane.transport.tls`, `membrane.transport.tls_protocol` | `MTLSAuthenticator`, `MTLSConfig`, `PeerCertH11Protocol` |
| Route scopes | `membrane.transport.authz` | `ROUTE_SCOPES`, `enforce_route_scope` |
| Tenant isolation | `membrane.security.tenant` | `TenantAuthorizer` |
| Secret file permissions | `membrane.security.files` | `require_private_file` |
| Backpressure and rate limits | `membrane.transport.limits` | `TransportLimits`, `ConcurrencyLimitMiddleware`, `RateLimitMiddleware` |
| Request IDs | `membrane.transport.request_id` | `RequestIdMiddleware` |
| Outbound URL guard (SSRF) | `membrane.security.url_allowlist` | `URLAllowlist`, `validate_outbound_url` |
| Encryption at rest | `membrane.security.encryption`, `membrane.content_store` | `encrypt_payload`, `FilesystemBlob`, `DecryptError` |
| Key rotation | `membrane.security.key_rotation` | versioned master keys |
| Secret backends | `membrane.secrets` | AWS, GCP, Vault |
| Audit log | `membrane.audit` | `AuditLog`, `FileAuditStorage`, `verify_chain` |
| Data keys and rotation | `membrane.security.keyring` | `DirectoryKeyring`, `write_next_key` |
| ACME certificates | `membrane.transport.acme` | `ACMEClient`, `ensure_certificate` |
| SPIFFE identity | `membrane.transport.spiffe`, `membrane.auth.spiffe` | `SPIFFEClient`, `SPIFFEAuthenticator` |
| Certificate hot reload | `membrane.transport.tls_rotation` | `CertRotationWatcher` |

## Authentication

`membrane serve` refuses to listen on a non-loopback address unless
one of these modes is configured (or `--allow-unauthenticated` is
passed explicitly):

* **API key**: `APIKeyAuthenticator` (`membrane.auth.apikey`),
  enabled with `--api-key-file`. Clients send
  `Authorization: Bearer <key>`. Keyfile lines are
  `sha256:<hex digest of the key>:<subject>:<scope,...>`, so a leaked
  keyfile holds no usable credentials. `membrane keys generate
  --subject S --scope read` prints a new key and its line. Plaintext
  `<key>:<subject>:<scope,...>` lines are still accepted, with a
  warning. Only digests are kept in memory, and a presented key is
  compared against them in constant time (`hmac.compare_digest`).
  Lines with an empty subject are ignored, because an empty subject
  would bypass the tenant check.
* **Plugin**: `--authenticator NAME --auth-config PATH` loads an
  authenticator registered under the `membrane.authenticators` entry
  point ([Plugins](plugins.md)).
* **mTLS**: `MTLSConfig` (`membrane.transport.tls`), enabled with
  `--tls-cert/--tls-key/--tls-ca`. Every connection must present a
  certificate signed by the CA bundle, and `MTLSAuthenticator` admits
  only CNs in `allowed_cns`. The CN is read from the **verified peer
  certificate** of the TLS handshake
  (`membrane.transport.tls_protocol`); a client-supplied
  `X-SSL-Client-CN` header is always discarded. Scopes come from the CN
  prefix (`admin-`, `write-`, `read-`). On `/join` the CN must equal
  the joining node id, optionally with a role prefix.
* **SPIFFE**: `--tls-spiffe-socket /run/spire/agent.sock` takes the
  node's certificate, key, and trust bundle from the SPIFFE Workload
  API (SPIRE). Callers present SVIDs; `SPIFFEAuthenticator` admits the
  SPIFFE IDs listed with `--tls-spiffe-allow spiffe://td/path=read,write`
  and grants those scopes. SVIDs are re-fetched every 5 minutes and a
  renewed one is served without a restart. Peers trust each other through
  the bundle (their certificates name workloads, not hosts). Install
  `membrane[tls-spiffe]`.

### TLS certificates

* **Hot reload.** With `--tls-cert/--tls-key`, the files are checked
  every minute and on `SIGHUP`; a changed pair is served on new
  connections and presented to peers without a restart. An expired
  certificate is refused at startup and on reload, and
  `membrane_tls_cert_expiry_seconds` tracks the time left.
* **ACME.** `--tls-acme-domain example.com` (repeatable) gets the
  listener certificate from an ACME CA (Let's Encrypt by default,
  `--tls-acme-directory` for others) through HTTP-01 challenges, keeps
  the account and certificate in `--tls-acme-state-dir` (default
  `<data-dir>/acme`, mode 0600), and renews it when less than 30 days
  remain. The domain's port 80 must reach `--tls-acme-http-port`.
  ACME certificates serve clients that authenticate with API keys;
  they are for single-node public listeners (peers use mTLS or SPIFFE).

### Secrets from a secret manager

The secret settings `--api-key-file`, `--peer-api-key-file`,
`--tls-cert`, `--tls-key`, `--tls-ca`, `--data-key-file`, and the LLM
`--api-key` accept `secret://NAME` in place of a file or value. It is
resolved through `--secret-provider`:
`env` (environment variables, the default), `aws` (Secrets Manager;
`AWS_REGION`), `gcp` (Secret Manager; `GOOGLE_CLOUD_PROJECT`), or
`vault` (`VAULT_ADDR`, `VAULT_TOKEN`), or an installed
`membrane.secret_providers` plugin. The secret never touches disk.

Authentication failures return `401` with `WWW-Authenticate: Bearer`;
a valid caller without the required scope gets `403`.

### Secret files

The server refuses to start when the API keyfile, the TLS private key,
the `--auth-config` file, or the data key can be read by other users
or written by anyone but their owner. The error tells you to run
`chmod 600`. Group read is allowed for the Kubernetes `fsGroup`
pattern (secret volumes mounted 0440); it is logged when the file's
group is not one of the server's own groups.

### API schema

FastAPI's interactive docs (`/docs`, `/redoc`) and `/openapi.json` are
disabled, because FastAPI serves them outside route authentication.
`--enable-api-docs` serves `/openapi.json` behind the `read` scope.

## Authorisation

`membrane.transport.authz.ROUTE_SCOPES` maps every route to a scope,
and `enforce_route_scope` runs it at the top of every handler:

| Scope | Routes |
|-------|--------|
| public | `GET /livez`, `GET /readyz`, `GET /disagg/healthz` |
| `read` | `GET /retrieve`, `/inventory`, `/peers`, `/heartbeat`, `/metrics`, `/metrics.json`, `/openapi.json` (with `--enable-api-docs`), `/prefix/lookup`, `/sessions/{id}`, `/objects/{hash}`, `GET`/`HEAD /kv/{handle}`; `POST /reconstruct` (without `prefill`), `/prefix/lookup`, `/route` |
| `write` | `POST /store`, `/replicate`, `/prefill`, `/gossip`, `/join`, `/leave`, `/objects`, `/reconstruct` with `prefill`, `/disagg/prefill`, `/disagg/prefill/batch`, `/disagg/decode` (and the matching gRPC calls); `PUT /kv/{handle}`; `DELETE /sessions/{id}` |
| `admin` | `POST /sync`, `/delete`, `/tombstone`, `/purge`, `/verify`, `PUT`/`GET`/`HEAD /blobs/{payload_ref}` (peer-to-peer KV bytes), and everything under `/admin/` |

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

### Prefill and tenants

`POST /prefill` stamps the caller's tenant (the key's subject or the
certificate CN) on every fragment it creates, so one tenant's prefill
is not readable by another.

### Known limitation: identical content across tenants

A node keys fragments by content hash alone. When two tenants store or
prefill byte-identical content, the node keeps the first tenant's copy;
the second tenant's write succeeds but its reads miss, exactly as if the
fragment were absent. Nothing is exposed across tenants, but the second
tenant gets no cache benefit for that content. Tenant-scoped fragment
keys are planned.

## Abuse and overload protection

* **Backpressure.** At most `--max-concurrency` requests (default 64)
  are handled at once. A request that cannot get a slot within 100 ms
  gets `503` with `Retry-After: 1`, so a saturated node sheds load
  instead of queueing it without bound.
* **Rate limiting.** `--rate-limit R` (requests per second) and
  `--rate-limit-burst B` apply a token bucket per credential: the
  bearer key's digest, or the client address for unauthenticated
  callers. A caller over its limit gets `429` with `Retry-After`.
* **Connections.** `--max-connections` caps open connections, and
  `--keep-alive-timeout` closes idle keep-alive connections.

Probes (`/livez`, `/readyz`) bypass these limits. Rejections are
counted in `membrane_requests_rejected_total{reason}`. Every response
carries an `X-Request-ID` (UUIDv7, or the caller's own if well formed),
and every log line written while handling the request includes it.

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

With `--data-dir`, KV bytes are stored by `FilesystemBlob`: each blob
is encrypted with AES-256-GCM under a key derived from the node's
master key and the blob's content hash, and written atomically
(temp file, `fsync`, rename). The master key comes from
`--data-key-file` (a file, a key directory, or `secret://NAME`) or is
generated once into `<data-dir>/master.key` with mode 0600; a key file
other users can read is refused. Keep the
key outside the volume in production; a
snapshot that contains both the blobs and the key protects nothing.

A blob that fails authentication (tampering, or the wrong key) is
never returned: `/retrieve` reports it as `found: false`, and the
in-memory encrypted store (`EncryptedInProcessBytes`) raises
`DecryptError`, which `/retrieve` reports as `"corrupt": true`.

Without `--data-dir`, KV bytes live only in process memory.

## Key rotation

Point `--data-key-file` at a directory of versioned keys (`v1.key`,
`v2.key`, ...; move an existing key to `DIR/v1.key` to adopt this).
The highest version encrypts new blobs and every version still
decrypts. To rotate:

```bash
membrane keys rotate-data-key /etc/membrane/data-keys   # writes v2.key (0600)
```

Each node re-reads the directory every minute, switches to the new
version, and re-encrypts its existing blobs under it in the background
(logged as `re-encrypted N blobs under data key version 2`). After that
the old `v1.key` can be deleted.

## Audit log

Every `/admin/*` operation (inspect, placement, evict, repair, policy
changes) is appended to a hash-chained `AuditLog`, which admins can
read with `GET /admin/audit`. With `--data-dir` the log is persisted to
`<data-dir>/audit.jsonl` (mode 0600) and survives restarts: on start the
node reloads it, verifies the chain, and continues it. A broken chain
(an edited, removed, or reordered entry) is logged at CRITICAL and
`membrane_audit_chain_valid` drops to 0. Without `--data-dir` the log
is in memory only.

## See also

* `docs/api-stability.md` — stability classification of the
  security modules.
* `docs/operations/incident-response.md` — what to do when
  the audit chain reports a tamper or the URL allow-list
  rejects an outbound call.
* `membrane/security/url_allowlist.py` — outbound URL
  policy.
* `membrane/security/encryption.py` — encryption primitives.
