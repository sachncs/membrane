# Capacity planning

Size memory, storage, Redis, and node count for a Membrane deployment. A node's memory budget holds the KV cache for the prompts you want to keep hot, read capacity grows linearly with nodes, and Redis holds only small metadata records.

## Sizing worksheet

1. **Estimate KV size per token** for your model, from the [table below](#kv-cache-size-per-token).
2. **Multiply by the prefixes you want to keep hot** (system prompts, documents, conversation histories) and their lengths. This is your working set.
3. **Set `--max-memory`** on each node to its share of the working set, times your replica count, plus about 1 KiB per fragment for metadata and indexes.
4. **Size `--data-dir` volumes** for the payload bytes, if you persist them.
5. **Choose a node count** from your read rate, using the per-node figures in [Adding nodes](#adding-nodes), with headroom for losing one node.

## What uses memory

A node's budget, `--max-memory` (default 1 GiB), covers the `payload_size` of the fragments it holds. When a store would exceed the budget, the node evicts expired fragments first, then the least valuable ones (weighted by last access and `reuse_score`), then cold graph neighbors.

> [!TIP]
> A full node is normal and stays ready. Watch `membrane_evictions_total{reason="capacity"}` to see whether it is evicting fragments that are still wanted.

| Configuration | KV bytes | Fragment metadata |
|---------------|----------|-------------------|
| Default | Process memory | Process memory |
| `--data-dir` | Encrypted files under `<data-dir>/blobs` | Process memory |
| `--data-dir` and `--redis` | Encrypted files | Process memory, written through to Redis |
| `--warm-tier-bytes` | Evicted fragments move to an encrypted disk tier and are promoted back on read | |

Budget process memory for the fragment payloads you keep in memory, plus roughly 1 KiB of metadata and index entries per fragment. With `--data-dir`, size the volume for the payload bytes.

## KV cache size per token

For a decoder model in fp16 or bf16:

```text
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes
```

| Model | Layers | KV heads × head dim | Per token | Per 1k tokens |
|-------|--------|---------------------|-----------|---------------|
| Llama-3-8B | 32 | 8 × 128 | 128 KiB | 128 MiB |
| Llama-3-70B | 80 | 8 × 128 | 320 KiB | 320 MiB |
| Mistral-7B | 32 | 8 × 128 | 128 KiB | 128 MiB |
| Mixtral-8x7B | 32 | 8 × 128 | 128 KiB | 128 MiB |

For example, a 32k-token prefix of Llama-3-8B takes about 4 GiB, so a node with a 64 GiB budget holds roughly 16 such prefixes, or proportionally more shorter ones. Storing KV as FP8 or with `--kv-quantization` halves or quarters these figures.

## Redis

Redis holds only fragment metadata, well under 1 KiB per fragment, and never KV bytes. Records expire with each fragment's TTL, so a single small Redis instance serves many nodes.

- Enable AOF (`appendonly yes`) so metadata survives a Redis restart.
- For failover, point every node at Sentinel (`redis+sentinel://...`).
- To spread metadata over several servers, use Redis Cluster (`redis+cluster://...`).

## Adding nodes

Read capacity grows linearly with the node count. A node's read rate depends on its CPU budget, not on the cluster size, so plan read capacity per node and add nodes for more.

The `kind-capacity` CI job (`scripts/kind_capacity.sh`) verifies this on every change. It gives every pod the same CPU limit (250m, with requests equal to limits), drives every pod to that limit with reads of fragments it holds, and compares 3 nodes with 5 once primary ownership has settled:

| Nodes | Reads/s | Per pod |
|-------|---------|---------|
| 3 | 1,848 | 616 / 612 / 620 |
| 5 | 2,955 | 579 / 583 / 613 / 590 / 590 |

That is 1.60x for 1.67x more CPU, with no failed reads, measured on a 4-CPU Docker VM shared with another workload. The job fails below 1.5x, or if ownership does not settle.

The cost of keeping a cluster consistent depends on what changes, not on how much it holds:

- **Inventory digests.** Each node keeps a digest of its fragments in 1,024 buckets, updated on every store and removal. Repair compares a peer's bucket digests in one small request and pages only the buckets that changed, so a quiet cluster repairs with one request per peer.
- **Location registry.** A node remembers where at most `max_location_entries` (200,000) hashes live, forgetting the least recently recorded first. A hash it no longer records is found through its ring owners, so this memory does not grow with the cluster's data.
- **Ownership.** Every node shares the same consistent-hash ring, which spreads primaries evenly. When nodes join or leave, primaries move to their new owners and settle.

### Scaling up a single node

On free-threaded Python, one node scales across cores with `--http-threads`: in-process reads run 3.2x faster on 4 threads than on 1. See the [free-threaded image](../deployment.md#free-threaded-image).

## Network

| Traffic | Volume |
|---------|--------|
| Gossip | A few KiB per peer every `--gossip-interval` (5 seconds); negligible |
| Strong writes | Each fragment's metadata to `replica_count` peers, plus its bytes |
| KV movement between nodes or datacenters | The dominant cost. `python scripts/demo.py` estimates it for the paper's workload: about 4.8 Gbps average egress at the threshold the model picks |

## Replicas and quorum

- `--replica-count` (default 2) peers receive each strong write.
- `--quorum-count` (default 2) copies, the local one included, must exist before a strong write succeeds. Strong writes keep working while at least `quorum_count` nodes are up.
- Spread replicas across hosts (the StatefulSet uses pod anti-affinity) and, for regional failure tolerance, across zones.

See [Consistency levels](../consistency.md) for the trade-offs.
