# Plugins

Compute backends, authenticators, and content stores are looked up by
name in registries (`membrane.runtime.plugins`). The built-ins are
registered there. Any installed package can add more through a Python
entry point, with no change to Membrane.

| Group | Selected by | Factory signature | Built-ins |
|-------|-------------|-------------------|-----------|
| `membrane.compute` | `--compute NAME` | `(llm_url: str, llm_model: str, api_key: str) -> Backend` | `cpu`, `gpu`, `ollama`, `openai`, `anthropic`, `transformers` |
| `membrane.authenticators` | `--authenticator NAME --auth-config PATH` | `(config_path: str) -> Authenticator` | `apikey` |
| `membrane.content_stores` | `--content-store NAME` (with `--data-dir`) | `(location: str, key_file: str) -> ContentStore` | `filesystem`, `memory` |

A built-in name always wins over an entry point with the same name, so
an installed package cannot silently replace a built-in. An unknown
name fails at startup with the list of available names:

```text
membrane serve --compute vlm
... unknown compute backend 'vlm'; available: anthropic, cpu, gpu, ollama, openai, transformers
```

## Writing a compute backend

A compute backend implements `membrane.compute.base.Backend` (see
`membrane/compute/cpu.py` for a complete example). Package it with an
entry point:

```toml
# pyproject.toml of your package
[project.entry-points."membrane.compute"]
my-engine = "my_package.membrane_plugin:make_backend"
```

```python
# my_package/membrane_plugin.py
from typing import override

from membrane.compute.base import Backend


class MyEngine(Backend):
    """Prefill through my inference engine."""

    def __init__(self, url: str, model: str) -> None:
        self.url, self.model = url, model

    @override
    def device_name(self) -> str:
        return f"my-engine({self.model})"

    # ... implement prefill(), generate(), and available() ...


def make_backend(llm_url: str, llm_model: str, api_key: str) -> Backend:
    return MyEngine(llm_url or "http://localhost:9000", llm_model or "default")
```

Install it into the same environment as Membrane (`uv pip install -e
path/to/my_package`), then run `membrane serve --compute my-engine
--llm-url http://engine:9000`.

## Writing an authenticator

An authenticator implements the `membrane.auth.Authenticator`
protocol. Its `authenticate(request)` method returns an `AuthContext`
(subject and scopes) or raises `AuthBackendError` (401) or
`AuthForbiddenError` (403). The factory receives the `--auth-config`
path. The server refuses to start when that file is readable by other
users, because it usually holds secrets.

```toml
[project.entry-points."membrane.authenticators"]
oidc = "my_package.auth:make_authenticator"
```

```bash
membrane serve --host 0.0.0.0 --authenticator oidc --auth-config /etc/membrane/oidc.toml
```

Scopes are `read`, `write`, and `admin`; the subject is the tenant the
caller reads and writes. See [Security](security.md).

## Writing a content store

A content store holds the KV bytes behind each fragment's `payload_ref`
and implements the `membrane.content_store.ContentStore` protocol
(`put`, `get`, `has`, `delete`, `size`). The factory receives `--data-dir`
and `--data-key-file`:

```toml
[project.entry-points."membrane.content_stores"]
s3 = "my_package.store:make_store"
```

```bash
membrane serve --data-dir s3://bucket/prefix --content-store s3
```

## Registering in code

Embedding applications can register a factory directly instead of
through an entry point:

```python
from membrane.runtime.plugins import COMPUTE_BACKENDS

COMPUTE_BACKENDS.register("my-engine", make_backend)
```
