# Compatibility matrix

The v3.0.0 release drops the v0.8 + 1.0.x compat shims. The
matrix below is the single source of truth for which versions
of the supported runtimes, models, and GPUs are known to
work with the 3.x series.

## Python

| Python | Status |
|--------|--------|
| 3.14   | required |
| 3.13 and older | not supported |

`requires-python = ">=3.14,<3.15"`. Membrane uses 3.14 features
throughout: deferred annotations (PEP 649), `compression.zstd`
(PEP 784), template strings in structured logging (PEP 750),
`InterpreterPoolExecutor` (PEP 734) for the threshold optimizer, and
`uuid.uuid7()` for request and audit IDs. `.python-version` pins 3.14,
`uv.lock` pins every dependency, and CI and the container image install
from the lock.

## Optional runtime deps

| Package              | Optional dep group      | Required by                |
|----------------------|-------------------------|---------------------------|
| ``cryptography``      | ``membrane[server]``     | TLS, encryption at rest   |
| ``numpy`` / ``lz4`` | ``membrane[transfer]`` | KV transfer engine, quantization |
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

The engines are not extras. They pin their own dependency stacks (vLLM
caps FastAPI below what the server needs) and do not all ship Python
3.14 wheels. Install the engine in its own environment;
``membrane.adapters`` imports it lazily. CI exercises the adapters
against their in-memory clients. The LMCache integration was removed in
the Python 3.14 release (LMCache has no 3.14 wheels).

## Transports

| Transport     | Version | Notes                                  |
|---------------|---------|----------------------------------------|
| HTTP (FastAPI) | 0.142+ | The only ``membrane serve`` transport; HTTPS with mTLS optional |

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
