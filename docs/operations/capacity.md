# Capacity Planning

## What uses memory

A node's budget (`--max-memory`, default 1 GiB) covers the
`payload_size` of the fragments it holds. When a store would exceed
the budget, the node evicts expired fragments first, then the
least valuable ones (weighted by last access and `reuse_score`), then
cold graph neighbours. A full node is therefore normal; watch
`membrane_evictions_total{reason="capacity"}` to see whether it is
evicting fragments that are still wanted.

Where the bytes live:

| Configuration | KV bytes | Fragment metadata |
|---------------|----------|-------------------|
| default | process memory | process memory |
| `--data-dir` | encrypted files under `<data-dir>/blobs` | process memory |
| `--data-dir` + `--redis` | encrypted files | process memory, written through to Redis |

Budget process memory for the fragment payloads you keep in memory
plus roughly 1 KiB of metadata and index entries per fragment. With
`--data-dir`, size the volume for the payload bytes instead.

## KV cache size per token

For a decoder model in fp16 / bf16:

```text
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes
```

| Model | Layers | KV heads × head dim | Per token | Per 1k tokens |
|-------|--------|---------------------|-----------|---------------|
| Llama-3-8B | 32 | 8 × 128 | 128 KiB | 128 MiB |
| Llama-3-70B | 80 | 8 × 128 | 320 KiB | 320 MiB |
| Mistral-7B | 32 | 8 × 128 | 128 KiB | 128 MiB |
| Mixtral-8x7B | 32 | 8 × 128 | 128 KiB | 128 MiB |

So caching a 32k-token prefix of Llama-3-8B takes about 4 GiB, and a
node with 64 GiB of fragment budget holds roughly 16 such prefixes, or
proportionally more shorter ones. FP8 or the `transfer` extra's
quantization halves or quarters these figures.

## Redis

Redis holds only fragment metadata (well under 1 KiB per fragment),
never KV bytes. Records expire with each fragment's TTL. A single small
Redis instance serves many nodes; enable AOF (`appendonly yes`) so the
metadata survives a Redis restart. For failover, point every node at
Sentinel (`redis+sentinel://...`); to spread the metadata over several
servers, use Redis Cluster (`redis+cluster://...`), where each write
is a non-transactional pipeline because its keys live in different
slots.

## Adding nodes

Each node's cost of keeping the cluster consistent depends on what
changes, not on how much the cluster holds:

- **Inventory digests.** A node maintains a digest of its fragments in
  1,024 buckets, updated on every store and removal. Gossip carries its
  root at no extra cost. Repair compares a peer's bucket digests (one
  small request) and pages only the buckets that changed since they
  were last verified, so a quiet cluster repairs in one request per peer.
- **Location registry.** A node remembers where at most
  `max_location_entries` (200,000) hashes live, least recently recorded
  forgotten first. A hash it no longer records is found through its ring
  owners. Its memory does not grow with the cluster's data.
- **Ownership.** The consistent-hash ring spreads primaries evenly. The
  kind e2e test scales 4 → 5 nodes and fails if any node still owns more
  than 30% of the primaries after rebalancing.

Read capacity grows with the node count. `scripts/kind_capacity.sh`
(the `kind-capacity` CI job) gives every pod the same CPU limit
(250m, requests = limits), drives every pod to that limit with reads of
fragments it holds, and compares 3 nodes with 5 after rebalancing:

| Nodes | Reads/s | Per pod |
|-------|---------|---------|
| 3 | 1,848 | 616 / 612 / 620 |
| 5 | 2,955 | 579 / 583 / 613 / 590 / 590 |

That is 1.60x for 5/3 = 1.67x more CPU, with no failed reads, measured
on a 4-CPU Docker VM while another workload shared the host. The job
also checks that primary ownership settles before it measures (here
within 15 s, 600 primaries split 105-147 per pod), and fails below
1.5x. A pod's read rate depends on its CPU budget, not on
the cluster size, so plan read capacity per pod and add pods for more.

## Network

- Gossip: a few KiB per peer every `--gossip-interval` (5 s).
  Negligible.
- Strong writes send each fragment's metadata to `replica_count` peers.
- Moving KV bytes between nodes or datacenters is the dominant cost.
  `python scripts/demo.py` estimates it for the paper's workload
  (about 4.8 Gbps average egress at the threshold the model picks).

## Replicas and quorum

- `--replica-count` (default 2) peers receive each strong write.
- `--quorum-count` (default 2) copies, the local one included, must
  exist before a strong write succeeds, so strong writes keep working
  while at least `quorum_count` nodes are up.
- Spread replicas across hosts (the StatefulSet uses pod
  anti-affinity) and, for regional failure tolerance, across zones.
