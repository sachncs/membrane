# Capacity Planning

This document provides sizing guidance for Membrane deployments. Numbers
are derived from the model's throughput equations (Eqs. 1–6 in the
referenced paper) and validated against the local benchmark
(`scripts/demo.py`).

## Memory sizing

Approximate fragment sizes per model layer (KV cache, fp16):

| Model | Layers | KV per 1k tokens / layer |
|---|---|---|
| Llama-3-8B | 32 | 1.0 MiB |
| Llama-3-70B | 80 | 2.5 MiB |
| Mistral-7B | 32 | 1.0 MiB |
| Mixtral-8x7B | 32 | 1.8 MiB |

For a 128k-token context window on Llama-3-8B:
128 × 32 × 1.0 MiB = 4 GiB per fragment.

Add 30 % headroom for overhead (indexes, Python objects, fragmentation).
**Budget: 5 GiB per fragment** is a safe starting point.

## CPU sizing

One Membrane worker thread per ~500 active fragments. For a cluster
serving 100k fragments, plan for 200 worker threads, which on x86 means
about 8 cores.

## Redis sizing

* Fragment payloads dominate Redis memory; size Redis at 2× the per-node
  fragment budget to account for replicas.
* Use `maxmemory-policy allkeys-lru` so Redis evicts when its own
  memory cap is reached.
* Enable AOF (`appendonly yes`) for durability; budget 10 % overhead for
  the AOF file.

## Network sizing

* Inter-node gossip: ~50 entries × 200 bytes = 10 KiB per gossip
  interval (5 s default). Negligible.
* Cross-cluster prefill transfer: ~5–50 MiB per prefill, depending on
  context length. Plan for 10 Gbps inter-node links.

## Replica count

* `replica_count` defaults to 2. Increase to 3 for higher read
  availability; further increases are diminishing returns.
* Place replicas on different hosts (anti-affinity in K8s).

## Worked example

Target: 10k active fragments, 99.9 % availability, p99 retrieval < 50 ms.

| Resource | Estimate |
|---|---|
| Memory per node | 50 GiB (10k × 5 GiB) |
| Cluster size | 3 nodes × 2 replicas = 6 nodes |
| Redis | 60 GiB RAM, 1 Gbps network |
| Compute | 16 vCPU per node, GPU optional |

## References

* `scripts/demo.py` — local throughput demonstration
* `membrane/model/throughput.py` — analytical model
