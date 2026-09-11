# Membrane compatibility matrix (Phase 3.6.6)

The v3.0.0 release drops the v0.8 + 1.0.x compat shims. The
matrix below is the single source of truth for which versions
of the supported runtimes, models, and GPUs are known to
work with the v3.0.0 series.

## Python

| Python | Status |
|--------|--------|
| 3.10   | supported |
| 3.11   | supported |
| 3.12   | supported (CI smoke) |
| 3.13   | supported (primary CI) |

## Optional runtime deps

| Package              | Optional dep group      | Required by                |
|----------------------|-------------------------|---------------------------|
| ``cryptography``      | (always required at 3.0+)| encryption at rest        |
| ``lmcache>=0.5,<0.6``| ``membrane[lmcache]``    | LMCache backend storage    |
| ``vllm>=0.10,<0.12`` | ``membrane[vllm]``       | vLLM KVConnector v1 backend |
| ``sglang>=0.4,<0.6`` | ``membrane[sglang]``     | SGLang radix-cache backend |
| ``tensorrt-llm>=0.20,<0.22`` | ``membrane[trtllm]`` | TensorRT-LLM backend |
| ``fastapi`` / ``uvicorn`` / ``grpcio`` | ``membrane[server]`` | FastAPI + gRPC transport |
| ``redis>=8.1.0``     | ``membrane[server]``     | Redis persistence backend  |
| ``torch>=2.13.0``    | ``membrane[gpu]``        | GPU compute backend        |
| ``transformers`` / ``tokenizers`` / ``sentencepiece`` / ``protobuf`` | ``membrane[local-llm]`` | HuggingFace local LLM backend |
| ``opentelemetry-sdk``| (always available)      | OTel tracer                |
| ``opentelemetry-exporter-otlp-proto-grpc`` | (always available) | OTel OTLP exporter |

> Production deployments pin each of the optional deps to the
> exact version that ships with the deployment image. The
> CI matrix exercises a single canonical version of each
> optional dep and reports drift in the smoke logs.

The historical ``[secrets-aws]``, ``[secrets-gcp]``, and
``[secrets-vault]`` extras have been folded into
``[server]`` at 3.0.0; the ``membrane.secrets`` backend
imports ``boto3`` / ``google-cloud-secret-manager`` /
``hvac`` lazily so the runtime dep is only required when
the corresponding backend is configured.

## Engines

| Engine      | Version          | Plugin                                |
|-------------|------------------|---------------------------------------|
| vLLM        | 0.10.x – 0.11.x  | ``membrane.adapters.vllm`` (Phase 5)  |
| SGLang      | 0.4.x – 0.5.x    | ``membrane.adapters.sglang`` (Phase 6)|
| TensorRT-LLM | 0.20.x – 0.21.x| ``membrane.adapters.trtllm`` (Phase 6)|

## Transports

| Transport     | Version | Notes                                  |
|---------------|---------|----------------------------------------|
| FastAPI HTTP  | 0.110+  | The v3.0.0 production transport         |
| gRPC          | n/a     | Removed wholesale in 3.0.0; new gRPC ships in a 3.0.1 release |

## Wire schema

| Schema version | Read  | Write |
|-----------------|-------|-------|
| v5              | yes   | yes (3.0.0+) |
| v4 / v3 / v2     | rejected | rejected (3.0.0+) |

Operators upgrading from 2.0.x must convert legacy blobs via
the migration script before booting a 3.0.0 cluster. The
conversion tool ships in ``tools/upgrade_v2_to_v5.py``.

## GPU matrix

| GPU | Notes |
|-----|-------|
| NVIDIA A100 / H100 | CUDA 12 + torch 2.3+ recommended |
| Apple Silicon (MPS) | CPU fallback is the supported path |

GPUDirect stage is gated behind a feature flag; the smoke
test runs on CPU and skips the GPU path.
