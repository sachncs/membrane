# Membrane wire & storage format (v1.0)

This document specifies the on-wire / on-disk formats that
Membrane uses for fragment fragments, identity fingerprints,
canonical payloads, and per-cluster metadata. Everything in this
document is part of the v1.0 contract; breaking changes require a
major version bump.

## 1. Fragment identity — `PayloadIdentity`

Stable ten-field fingerprint that uniquely identifies a fragment in
storage, on the wire, and in gossip digests:

```python
PayloadIdentity(
    payload_hash: str,            # 64 hex characters (SHA-256)
    model_id: str,
    model_revision: str,          # "" when unpinned
    tokenizer_name: str,
    tokenizer_revision: str,
    layer_range: (int, int),      # [start, end), inclusive
    head_range: (int, int),       # (-1, -1) for "all heads"
    token_span: (int, int),       # [start, end], inclusive
    dtype: str,                   # "float16" | "bfloat16" | "float32" | "float64"
    shape: tuple[int, ...],       # (batch, layers, heads, seq, head_dim)
)
```

Serialized to JSON as a sub-dict:

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

`PayloadIdentity.fingerprint()` returns the SHA-256 of the JSON-
canonical form (sort_keys=True). Two fragments collide only when
every field is identical; the ten fields combined are enough to
disambiguate model / tokenizer revisions, layer / head spans,
dtype, and tensor shape.

## 2. Wire format — `FragmentMessage`

Schema version 2. Every request body that carries a fragment uses
this shape:

```json
{
    "schema_version": 2,
    "identity": { ... PayloadIdentity sub-dict ... },
    "payload_ref": "abc…",
    "payload_size": 8388608,
    "ttl": 3600.0,
    "reuse_score": 0.87,
    "version_id": 1
}
```

The body never carries the canonical bytes inline over HTTP /
FastAPI: clients and servers stream the bytes through the
separate `ContentStore` API (`PUT /payload/{key}` style out of
scope for the v1 wire). The bytes are addressable through
`payload_ref` (the SHA-256 hex digest).

### gRPC variant

`membrane.proto` `FragmentMessage` field set:

```
1    int32  schema_version         = 2
2    bytes   payload_hash           = 64-byte SHA-256 (raw)
3    string  model_id
4    string  model_revision
5    string  tokenizer_name
6    string  tokenizer_revision
7    repeated int32  layer_range    // [start, end]
8    repeated int32  head_range
9    repeated int32  token_span
10   string  dtype
11   repeated int64  shape
12   string  payload_ref
13   int64   payload_size
14   bytes   payload               // optional inline payload
15   double  ttl
16   double  reuse_score
17   int32   version_id
```

gRPC servers set `max_receive_message_length` and
`max_send_message_length` to `DEFAULT_MAX_BODY_BYTES` (100 MiB) so
inline `payload` can carry large frames without truncation.

### Backward compatibility

Schema version 1 is **not** retained. A v1 client receives a
`SchemaError` on every v2 reader; a v2 client refuses every v1
fragment. Operators rolling out v1.0 must drain their v0.8 fleet
first or convert at the proxy layer.

## 3. Canonical byte framing — `canonicalize` / `parse_canonical`

Frames on disk:

```
+-----------------+
| MAGIC 4 B       |  = 0xC0DE0102
+-----------------+
| schema 2 B      |  = 2 today
| reserved 4 B    |  = 0
+-----------------+
| identity_len 4 B (u32 LE)
+-----------------+
| identity_json   |  (PayloadIdentity.to_dict())
+-----------------+
| payload_len 8 B (u64 LE)
+-----------------+
| payload         |  (canonical K/V bytes)
+-----------------+
| trailer 8 B     |  = first 8 bytes of SHA-256(payload)
+-----------------+
```

Header = 14 bytes. Total size =
`14 + identity_len + 8 + payload_len + 8`.

`parse_canonical(buf)` reverses the round-trip and rejects any
trailer / magic / header mismatch with:

* `SchemaError` — magic, schema, or identity length is wrong.
* `CorruptPayloadError` — magic + schema + length match but the
  truncated trailer disagrees with the payload's hash.

## 4. Cluster metadata — Snapshot

Durable cluster state is written by `Server.checkpoint_state()`
and read by `Server.restore_state()` to a single JSON file per
node:

```
{state_dir}/{node_id}.json
```

```json
{
  "schema_version": 2,
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

The file is rewritten atomically (`tempfile.NamedTemporaryFile +
os.fsync + os.replace + fsync on the parent dir`). A `cluster_epoch`
more than one step behind the live value is rejected on restore
(see `ClusterEpochGuard`); the stale file is then deleted.

## 5. Tombstone propagation

Every soft-delete writes a `Tombstone(content_hash, until,
nodes)` to the local `TombstoneTable` *before* removing the
fragment. Gossip piggybacks the active tombstone set on its next
state delivery so peers can:

* refuse to re-add the hash via stale store requests,
* converge on a single expiry across replicas (the larger
  `until` wins),
* sweep expired entries via the daemon `Sweeper`.

The default `tombstone_until` is **60 s** after the delete and
the wire op `op_tombstone` carries the value explicitly so the
deadline survives truncation on the producer side.

## 6. Ref-count semantics

In-process `RefCount` tracks the set of node identifiers holding
each `payload_hash`. `release(hash, node_id)` returns `True` only
when the last reference is gone; the caller decides what to do
with that signal (typically a `ContentStore.delete`) so the wire
contract stays free of hidden side effects.

Cross-process ref counts are out of scope for v1.0; the
`InventoryDigest` returned by `op_inventory` plus tombstone
gossip carry the equivalent information at the cluster level.

## 7. Backward compatibility & migration

There is none. v0.x → v1.0 is a wire-format break. Operators must
restart the fleet with the same v1.0 image after every node is
upgraded. Persisted on-disk state (Redis metadata, FilesystemBlob
payloads, snapshot files) is laid out so that a v1.0 server can
read v0.8's Redis keys and snapshot files, but it cannot read
v0.8's wire payloads (the framing differs).
