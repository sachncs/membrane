# Membrane wire & storage format (v5)

This document specifies the on-wire / on-disk formats that
Membrane uses for fragment payloads, identity fingerprints,
canonical bytes, and per-cluster metadata. Everything in this
document is part of the **v5** contract; breaking changes
require a major version bump and a ``schema_version`` bump in
:data:`membrane.serialization.SCHEMA_VERSION`.

The v5 contract was introduced in v3.0.0 and supersedes the
v2 / v4 contracts. ``from_dict`` and ``parse_canonical``
reject any other ``schema_version`` with
:class:`membrane.errors.SchemaError`.

## 1. Fragment identity — ``PayloadIdentity``

Stable ten-field fingerprint that uniquely identifies a
fragment in storage, on the wire, and in gossip digests:

```python
PayloadIdentity(
    payload_hash: str,            # 64 hex characters (SHA-256)
    model_id: str,
    model_revision: str,          # "" when unpinned
    tokenizer_name: str,
    tokenizer_revision: str,
    layer_range: (int, int),      # [start, end], inclusive
    head_range: (int, int),       # (-1, -1) for "all heads"
    token_span: (int, int),       # [start, end], inclusive
    dtype: str,                   # see ``docs/compat-matrix.md``
    shape: tuple[int, ...],       # (batch, layers, heads, seq, head_dim)
)
```

Serialised to JSON as a sub-dict:

```json
{
    "payload_hash": "abc…",
    "model_id": "llama-3-8b",
    "model_revision": "",
    "tokenizer_name": "llama-3-8b",
    "tokenizer_revision": "",
    "layer_range": [0, 32],
    "head_range": [-1, -1],
    "token_span": [0, 128],
    "dtype": "float16",
    "shape": [1, 32, 32, 128, 64]
}
```

``PayloadIdentity.fingerprint()`` returns the SHA-256 of the
JSON-canonical form (``sort_keys=True``). Two fragments
collide only when every field is identical; the ten fields
combined disambiguate model / tokenizer revisions, layer /
head spans, dtype, and tensor shape.

## 2. Wire format — ``FragmentMessage``

**Schema version 5.** Every request body that carries a
fragment uses this shape:

```json
{
    "schema_version": 5,
    "tenant_id": "public",
    "identity": { ... PayloadIdentity sub-dict ... },
    "payload_ref": "abc…",
    "payload_size": 8388608,
    "ttl": 3600.0,
    "reuse_score": 0.87,
    "version_id": 1,
    "consistency": "strong",
    "hlc": 1735600000123456789,
    "fingerprint_compat": "5b6e..."
}
```

The body never carries the canonical bytes inline over HTTP /
FastAPI: clients and servers stream the bytes through the
separate ``ContentStore`` API (``PUT /payload/{key}`` style
out of scope for the v5 wire). The bytes are addressable
through ``payload_ref`` (the SHA-256 hex digest).

The fields added at v2.0 (``consistency``, ``hlc``) and
v3.0.0 (``tenant_id``, ``fingerprint_compat``) are required:
``from_dict`` raises ``SchemaError`` if any are missing.

### Consistency levels

The ``consistency`` field is one of ``"strong"``, ``"quorum"``,
or ``"eventual"``. See ``docs/consistency.md`` for the full
contract; the v5 wire carries the literal string in the
envelope and :func:`membrane.serialization.from_dict` does not
constrain it (the typed enforcement lives on the cluster side
in :class:`membrane.network.config.ClusterConfig`).

### gRPC variant

The gRPC envelope is defined by ``membrane/wire/v3/wire_v3.proto``
(``Envelope``, ``TensorPayload``, ``ChunkRequest``, ``Chunk``).
The protobuf stub is generated into
``membrane/wire/v3/wire_v3_pb2.py`` and the servicer is
implemented at ``membrane.transport.grpc``. gRPC servers set
``max_receive_message_length`` and ``max_send_message_length``
to ``DEFAULT_MAX_BODY_BYTES`` (100 MiB) so inline payloads can
carry large frames without truncation.

## 3. Canonical byte framing — ``canonicalize`` / ``parse_canonical``

Frames on disk:

```
+-----------------------------------+
| MAGIC       4 B  = 0xC0DE0105    |   (last byte = 0x05 for v5)
+-----------------------------------+
| schema      2 B                   |   (= 5 for v5)
+-----------------------------------+
| reserved    4 B   (= 0)           |
+-----------------------------------+
| identity_len 4 B (u32 LE)         |
+-----------------------------------+   offset = 14
| identity_json   (identity_len B)  |   UTF-8 JSON of
+-----------------------------------+   PayloadIdentity.to_dict()
| payload_len  u64 (LE)             |
+-----------------------------------+
| payload      (payload_len B)      |
+-----------------------------------+
| trailer      8 B                  |   first 8 bytes of SHA-256
+-----------------------------------+   of payload; cheap verify
```

Header = 14 bytes. Total size =
``14 + identity_len + 8 + payload_len + 8``.

``parse_canonical(buf)`` reverses the round-trip and rejects
any trailer / magic / header mismatch with:

* ``SchemaError`` — magic, schema, or identity length is wrong.
* ``CorruptPayloadError`` — magic + schema + length match but
  the truncated trailer disagrees with the payload's hash.

## 4. Cluster metadata — Snapshot

Durable cluster state is written by ``Server.checkpoint_state()``
and read by ``Server.restore_state()`` to a single JSON file
per node:

```
{state_dir}/{node_id}.json
```

```json
{
  "schema_version": 5,
  "cluster_epoch": 17,
  "captured_at": 1735600000.123,
  "membership": [
    {"node_id": "peer-1", "host": "10.0.0.1", "port": 8080,
     "cluster_epoch": 17, "healthy": true, "suspect": false,
     "missed_heartbeats": 0},
    ...
  ],
  "shards": {
    "primary_map": {"hash1": "node-1", ...},
    "replica_map": {"hash1": ["node-2", "node-3"], ...}
  },
  "server": {"request_count": 123, "error_count": 4}
}
```

The file is rewritten atomically
(``tempfile.NamedTemporaryFile + os.fsync + os.replace +
fsync on the parent dir``). A ``cluster_epoch`` more than one
step behind the live value is rejected on restore (see
``ClusterEpochGuard``); the stale file is then deleted.

## 5. Tombstone propagation

Every soft-delete writes a ``Tombstone(content_hash, until,
nodes)`` to the local ``TombstoneTable`` *before* removing the
fragment. Gossip piggybacks the active tombstone set on its
next state delivery so peers can:

* refuse to re-add the hash via stale store requests,
* converge on a single expiry across replicas (the larger
  ``until`` wins),
* sweep expired entries via the daemon ``Sweeper``.

The default ``tombstone_until`` is **60 s** after the delete
and the wire op ``op_tombstone`` carries the value explicitly
so the deadline survives truncation on the producer side.

## 6. Ref-count semantics

In-process ``RefCount`` tracks the set of node identifiers
holding each ``payload_hash``. ``release(hash, node_id)``
returns ``True`` only when the last reference is gone; the
caller decides what to do with that signal (typically a
``ContentStore.delete``) so the wire contract stays free of
hidden side effects.

Cross-process ref counts are out of scope for v5; the
``InventoryDigest`` returned by ``op_inventory`` plus
tombstone gossip carry the equivalent information at the
cluster level.

## 7. Backward compatibility & migration

The current contract is **v5** (magic ``0xC0DE0105``, schema
``5``). Older wire breaks and their migration paths:

| From | To | Migration | Tool |
|------|----|-----------|------|
| v0.x | v1.0 | Fleet restart with the v1.0 image. | none |
| v1.x | v2.0 | Additive: ``consistency`` + ``hlc`` introduced; v2 readers accept v1 envelopes with the defaults applied at write time. | ``tools/upgrade_v1_to_v2.py`` |
| v2.x | v5 (3.0.0) | Breaking: ``tenant_id`` and ``fingerprint_compat`` added; v5 readers reject v2 envelopes outright. Convert at the proxy or in a one-shot migration pass. | ``tools/upgrade_v2_to_v5.py`` |
| v4    | v5 (3.0.0) | Breaking: same as v2 -> v5; v4 was an internal pre-release schema that the public never used. | ``tools/upgrade_v2_to_v5.py`` |
| v3    | v5 (3.0.0) | Breaking: v3 was a transient schema that the public never used; conversion follows the v2 -> v5 path. | ``tools/upgrade_v2_to_v5.py`` |

Operators upgrading from a 2.x or earlier deployment must run
``tools/upgrade_v2_to_v5.py`` (and the equivalent JSON
helper ``tools/upgrade_v2_to_v5_json.py`` for stored
payloads) before booting a 3.0.0 cluster. The conversion
tool reads v2 / v4 envelopes, fills in ``tenant_id`` (default
``"public"``) and ``fingerprint_compat`` (recomputed from
the identity), and writes v5 envelopes back to the same
storage backend.

See also:

* ``docs/compat-matrix.md`` — runtime / engine compat
* ``CHANGELOG.md`` — release history of every wire break
* ``tools/upgrade_v2_to_v5.py`` — one-shot migration tool
