# Compatibility matrix

The v3.0.0 release drops the v0.8 + 1.0.x compat shims. The
matrix below is the single source of truth for which versions
of the supported runtimes, models, and GPUs are known to
work with the 3.x series.

## Python

| Python | Status |
|--------|--------|
| 3.10   | supported |
| 3.11   | supported |
| 3.12   | supported |
| 3.13   | supported |

All four versions run the full test suite in CI.

## Optional runtime deps

| Package              | Optional dep group      | Required by                |
|----------------------|-------------------------|---------------------------|
| ``cryptography``      | ``membrane[server]``     | TLS, encryption at rest   |
| ``numpy`` / ``lz4`` / ``zstandard`` | ``membrane[transfer]`` | KV transfer engine, quantization |
| ``lmcache>=0.5,<0.6``| ``membrane[lmcache]``    | LMCache backend storage    |
| ``vllm>=0.10,<0.12`` | ``membrane[vllm]``       | vLLM KVConnector v1 backend |
| ``sglang>=0.4,<0.6`` | ``membrane[sglang]``     | SGLang radix-cache backend |
| ``tensorrt-llm>=0.20,<0.22`` | ``membrane[trtllm]`` | TensorRT-LLM backend |
| ``fastapi`` / ``uvicorn`` / ``httpx`` | ``membrane[server]`` | HTTP transport and clients |
| ``grpcio`` | ``membrane[disagg]`` | Disaggregation gRPC surface |
| ``redis>=8.1.0``     | ``membrane[server]``     | Redis persistence backend  |
| ``torch>=2.13.0``    | ``membrane[gpu]``        | GPU compute backend        |
| ``transformers`` / ``tokenizers`` / ``sentencepiece`` / ``protobuf`` | ``membrane[local-llm]`` | HuggingFace local LLM backend |
| ``opentelemetry-*``  | ``membrane[otel]``       | OpenTelemetry tracing      |
| ``boto3`` / ``google-cloud-secret-manager`` / ``hvac`` | ``membrane[secrets-aws]`` / ``[secrets-gcp]`` / ``[secrets-vault]`` | Secret backends |

> Production deployments pin each of the optional deps to the
> exact version that ships with the deployment image. The
> CI matrix exercises a single canonical version of each
> optional dep and reports drift in the smoke logs.

## Engines

| Engine      | Version          | Plugin                                |
|-------------|------------------|---------------------------------------|
| vLLM        | 0.10.x – 0.11.x  | ``membrane.adapters.vllm``   |
| SGLang      | 0.4.x – 0.5.x    | ``membrane.adapters.sglang`` |
| TensorRT-LLM | 0.20.x – 0.21.x | ``membrane.adapters.trtllm`` |

CI exercises the adapters against their in-memory clients; the engines
themselves are not installed in CI.

## Transports

| Transport     | Version | Notes                                  |
|---------------|---------|----------------------------------------|
| HTTP (FastAPI) | 0.141+ | The only ``membrane serve`` transport; HTTPS with mTLS optional |

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
