# Security

How Membrane authenticates callers, authorizes every request, isolates tenants, protects data in transit and at rest, and records administrative actions. Membrane is secure by default: a node refuses to serve unauthenticated traffic on any address other than loopback.

## Security at a glance

| Control | What Membrane does |
|---------|--------------------|
| **Authentication** | API keys (stored as SHA-256 digests), mTLS, or SPIFFE workload identity on every route except health probes |
| **Authorization** | Every route requires a scope (`read`, `write`, `admin`); unlisted routes fail closed |
| **Tenant isolation** | Each tenant keeps and reads its own fragments; a cross-tenant read is indistinguishable from a miss |
| **Encryption in transit** | TLS with hot-reloaded certificates, ACME, or SPIFFE SVIDs |
| **Encryption at rest** | AES-256-GCM per blob, with versioned, rotatable data keys |
| **Secrets** | Read from AWS Secrets Manager, Google Secret Manager, Vault, or the environment; never written to disk |
| **Audit** | A hash-chained, tamper-evident log of every administrative action |
| **Abuse protection** | Concurrency limits, per-credential rate limits, and connection caps |
| **Outbound requests** | An SSRF guard on every request a node makes to a peer |

## Hardening checklist

- [ ] Use mTLS (or SPIFFE) between nodes, and API keys or mTLS for clients.
- [ ] Give each service its own key with the narrowest scope it needs.
- [ ] Keep keyfiles, TLS keys, and data keys `chmod 600`, or load them with `secret://` from a secret manager.
- [ ] Mount the data key from outside the data volume (`--data-key-file`).
- [ ] Rotate the data key periodically with `membrane keys rotate-data-key`.
- [ ] Set `--rate-limit` on nodes shared by several clients.
- [ ] Keep `MEMBRANE_PEER_NETWORKS` as narrow as the deployment allows.
- [ ] Alert on `membrane_audit_chain_valid == 0` and on certificate expiry.

## Authentication

`membrane serve` refuses to listen on a non-loopback address unless one of these modes is configured. `--allow-unauthenticated` overrides the check for throwaway development setups only.

### API keys

Enable with `--api-key-file`. Clients send `Authorization: Bearer <key>`.

- Keyfile lines have the form `sha256:<hex digest>:<subject>:<scope,...>`, so a leaked keyfile contains no usable credentials. `membrane keys generate --subject S --scope read` prints a new key and its line.
- The server keeps only digests in memory and compares a presented key against them in constant time.
- The subject is the tenant the key reads and writes. Lines with an empty subject are ignored, because an empty subject would bypass the tenant check.
- Legacy plaintext lines (`<key>:<subject>:<scope,...>`) are still accepted, with a warning.

### mTLS

Enable with `--tls-cert`, `--tls-key`, and `--tls-ca`. Every connection must present a certificate signed by the CA bundle, and only allowed common names (CNs) are admitted.

- The CN is read from the **verified peer certificate** of the TLS handshake. A client-supplied `X-SSL-Client-CN` header is always discarded.
- Scopes come from the CN prefix: `admin-`, `write-`, or `read-`.
- When a node joins a cluster, its CN must equal its node id, optionally with a role prefix.

### SPIFFE

Enable with `--tls-spiffe-socket /run/spire/agent.sock` (install `membrane[tls-spiffe]`). The node takes its certificate, key, and trust bundle from the SPIFFE Workload API (SPIRE).

- Callers present SVIDs. Only the SPIFFE IDs listed with `--tls-spiffe-allow spiffe://td/path=read,write` are admitted, with the scopes given.
- SVIDs are refreshed every 5 minutes, and a renewed one is served without a restart.
- Peers trust each other through the bundle, because their certificates name workloads rather than hosts.

### Custom authenticators

`--authenticator NAME --auth-config PATH` loads an authenticator registered under the `membrane.authenticators` entry point. See [Plugins](plugins.md).

### Responses

An authentication failure returns `401` with `WWW-Authenticate: Bearer`. A valid caller without the required scope gets `403`.

## Authorization

Every route maps to a scope, enforced at the top of every handler. `admin` implies `write`, which implies `read`. A route that is not listed defaults to `read`, so a new route fails closed.

| Scope | Routes |
|-------|--------|
| public | `GET /livez`, `GET /readyz`, `GET /disagg/healthz` |
| `read` | `GET /retrieve`, `/inventory`, `/inventory/buckets`, `/peers`, `/heartbeat`, `/metrics`, `/metrics.json`, `/openapi.json` (with `--enable-api-docs`), `/prefix/lookup`, `/sessions/{id}`, `/objects/{hash}`, `GET`/`HEAD /kv/{handle}`; `POST /reconstruct` (without `prefill`), `/prefix/lookup`, `/route` |
| `write` | `POST /store`, `/replicate`, `/prefill`, `/gossip`, `/join`, `/leave`, `/objects`, `/reconstruct` with `prefill`, `/disagg/prefill`, `/disagg/prefill/batch`, `/disagg/decode` (and the matching gRPC calls); `PUT /kv/{handle}`; `DELETE /sessions/{id}` |
| `admin` | `POST /sync`, `/delete`, `/tombstone`, `/purge`, `/verify`, `PUT`/`GET`/`HEAD /blobs/{payload_ref}` (peer-to-peer KV bytes), and everything under `/admin/` |

FastAPI's interactive API docs (`/docs`, `/redoc`, `/openapi.json`) are disabled, because FastAPI would serve them outside route authentication. `--enable-api-docs` serves `/openapi.json` behind the `read` scope.

### Peer-to-peer calls

Nodes call each other for joining, heartbeats, gossip, replication, and delete propagation, and they authenticate like any other client: with their mTLS client certificate, or in API-key clusters with the key from `--peer-api-key-file`. The peer key needs `admin`, because peers replicate every tenant's fragments.

## Tenant isolation

Every fragment belongs to a tenant. A fragment of tenant `acme` is readable only by callers whose identity is `acme`, or who hold `admin`. Fragments of the default tenant, `public`, are readable by everyone. A cross-tenant read returns exactly what a miss returns, so a caller cannot probe for another tenant's content.

`POST /prefill` stamps the caller's tenant (the key's subject or the certificate CN) on every fragment it creates.

### Identical content across tenants

Each tenant keeps its own copy of a content hash, so two tenants that cache the same prompt both get cache hits without sharing access.

- A node stores a fragment under its *key*: the bare content hash for the `public` tenant, and `<tenant>:<hash>` for every other tenant. Tenant ids cannot contain `:`.
- `GET /retrieve?content_hash=<hash>` returns the caller's own copy, else the `public` copy. A caller without a tenant (authentication off) or with `admin` falls back to any tenant's copy.
- `<tenant>:<hash>` names one copy directly. The tenant check still applies, so it does not expose another tenant's copy to a non-admin.
- `/inventory`, the persistence backends, the warm tier, and fragment events use keys. `/admin/fragments/{key}` and `/admin/evict` accept a key or a bare hash.
- The copies share one encrypted KV blob, because the bytes are identical. The blob is deleted when the last copy that references it is removed.
- The hash ring places every tenant's copy of a hash on the same nodes.

## Encryption in transit

### Certificate hot reload

With `--tls-cert` and `--tls-key`, the files are checked every minute and on `SIGHUP`. A changed pair is served on new connections, and presented to peers, without a restart. An expired certificate is refused at startup and on reload, and `membrane_tls_cert_expiry_seconds` reports the time remaining.

### ACME

`--tls-acme-domain example.com` (repeatable) obtains the listener certificate from an ACME CA through HTTP-01 challenges: Let's Encrypt by default, or another CA with `--tls-acme-directory`.

- The account and certificate are kept in `--tls-acme-state-dir` (default `<data-dir>/acme`, mode 0600).
- The certificate is renewed when fewer than 30 days remain.
- The domain's port 80 must reach `--tls-acme-http-port`.

> [!NOTE]
> ACME certificates suit single-node public listeners whose clients authenticate with API keys. Peers use mTLS or SPIFFE.

## Encryption at rest

With `--data-dir`, KV bytes are encrypted with AES-256-GCM under a key derived from the node's master key and the blob's content hash, and written atomically (temporary file, `fsync`, rename).

- The master key comes from `--data-key-file` (a file, a key directory, or `secret://NAME`), or is generated once into `<data-dir>/master.key` with mode 0600. A key file that other users can read is refused.
- A blob that fails authentication, because it was tampered with or the key is wrong, is never returned. `/retrieve` reports it as not found, or as `"corrupt": true` for the in-memory encrypted store.
- Without `--data-dir`, KV bytes live only in process memory.

> [!WARNING]
> Keep the data key outside the data volume. A snapshot that contains both the blobs and the key protects nothing.

### Rotating the data key

Point `--data-key-file` at a directory of versioned keys (`v1.key`, `v2.key`, ...). To adopt this, move an existing key to `DIR/v1.key`. The highest version encrypts new blobs, and every version still decrypts.

```bash title="Rotate"
membrane keys rotate-data-key /etc/membrane/data-keys   # writes v2.key (0600)
```

Each node re-reads the directory every minute, switches to the new version, and re-encrypts its existing blobs in the background (logged as `re-encrypted N blobs under data key version 2`). After that, the old key can be deleted.

## Secrets management

Every secret setting accepts `secret://NAME` in place of a file or value: `--api-key-file`, `--peer-api-key-file`, `--tls-cert`, `--tls-key`, `--tls-ca`, `--data-key-file`, and the LLM `--api-key`. The secret is resolved in memory through `--secret-provider` and never touches disk.

| Provider | Configuration |
|----------|---------------|
| `env` (default) | Environment variables |
| `aws` | AWS Secrets Manager; `AWS_REGION`, and `AWS_PROFILE` for a named profile |
| `gcp` | Google Secret Manager; `GOOGLE_CLOUD_PROJECT` |
| `vault` | HashiCorp Vault; `VAULT_ADDR`, `VAULT_TOKEN`. Reads the `value` field of a KV v2 secret on the `secret` mount |
| Plugin | Anything registered under the `membrane.secret_providers` entry point |

### Secret file permissions

The server refuses to start when the API keyfile, a TLS private key, the `--auth-config` file, or the data key can be read by other users or written by anyone but the owner, and the error says to run `chmod 600`. Group read is allowed, for the Kubernetes `fsGroup` pattern (secret volumes mounted 0440); it is logged when the file's group is not one of the server's own.

## Audit log

Every `/admin/*` operation (inspect, placement, evict, repair, and policy changes) is appended to a hash-chained audit log, which admins read with `GET /admin/audit`.

- With `--data-dir`, the log is persisted to `<data-dir>/audit.jsonl` (mode 0600) and survives restarts: on start, the node reloads it, verifies the chain, and continues it.
- A broken chain, meaning an edited, removed, or reordered entry, is logged at CRITICAL, and `membrane_audit_chain_valid` drops to 0.
- Without `--data-dir`, the log is held in memory only.

## Abuse and overload protection

| Protection | Setting | Behavior |
|------------|---------|----------|
| Backpressure | `--max-concurrency` (64) | A request that cannot get a slot within 100 ms gets `503` with `Retry-After: 1`, so a saturated node sheds load instead of queueing it without bound |
| Rate limiting | `--rate-limit`, `--rate-limit-burst` | A token bucket per credential (the key's digest, or the client address when unauthenticated); over-limit callers get `429` with `Retry-After` |
| Connections | `--max-connections`, `--keep-alive-timeout` | Caps open connections and closes idle keep-alive connections |

Health probes bypass these limits. Rejections are counted in `membrane_requests_rejected_total{reason}`.

Every response carries an `X-Request-ID` (a UUIDv7, or the caller's own if well formed), and every log line written while handling the request includes it.

## Outbound request guard

Every outbound HTTP request a node makes passes an SSRF guard:

- Only `http` and `https` are allowed.
- Private (RFC 1918), loopback (`127.0.0.0/8`, `::1`), link-local (`169.254.0.0/16`), and IPv6 unique-local (`fc00::/7`) addresses are blocked, after resolving the hostname.
- The validated IP is pinned to the socket, so DNS rebinding cannot substitute a private address on a second resolution.
- Redirects are not followed; each one must be validated again.

Cluster peers usually live on private addresses. Seed hosts given with `--peer` are allowed by name, and `--peer-network` (or `MEMBRANE_PEER_NETWORKS`) admits a CIDR, such as the Kubernetes pod network, for peers learned later. A host passes only when every address it resolves to is inside an allowed network.

> [!CAUTION]
> Keep the peer network as narrow as the deployment allows, and never include `169.254.0.0/16`, which covers cloud metadata endpoints.

## Reporting a vulnerability

Report vulnerabilities privately as described in [SECURITY.md](../SECURITY.md), not in public issues.

## Implementation reference

| Concern | Module |
|---------|--------|
| Authentication protocol | `membrane.auth` (`Authenticator`, `AuthContext`, `require_scope`) |
| API keys | `membrane.auth.apikey` |
| mTLS | `membrane.auth.mtls`, `membrane.transport.tls`, `membrane.transport.tls_protocol` |
| SPIFFE | `membrane.transport.spiffe`, `membrane.auth.spiffe` |
| Route scopes | `membrane.transport.authz` (`ROUTE_SCOPES`) |
| Tenant isolation | `membrane.security.tenant`, `membrane.store.tenant_guard` |
| Secret file permissions | `membrane.security.files` |
| Limits and request IDs | `membrane.transport.limits`, `membrane.transport.request_id` |
| Outbound request guard | `membrane.security.url_allowlist` |
| Encryption and key rotation | `membrane.security.encryption`, `membrane.security.keyring`, `membrane.content_store` |
| Secret providers | `membrane.secrets` |
| Audit log | `membrane.audit` |
| ACME and certificate reload | `membrane.transport.acme`, `membrane.transport.tls_rotation` |

See also [API stability](api-stability.md) for the stability of these modules, and [Incident response](operations/incident-response.md) for what to do when the audit chain breaks or the outbound guard rejects a call.
