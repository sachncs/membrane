# Plugins

Extend Membrane without forking it. Compute backends, authenticators, content stores, persistence backends, eviction policies, placement policies, and secret providers are looked up by name, and any installed package can add its own through a standard Python entry point. Hooks let you observe server events, for example to export audit data.

## Extension points

| Group | Selected by | Factory signature | Built-ins |
|-------|-------------|-------------------|-----------|
| `membrane.compute` | `--compute NAME` | `(llm_url: str, llm_model: str, api_key: str) -> Backend` | `cpu`, `gpu`, `ollama`, `openai`, `anthropic`, `transformers` |
| `membrane.authenticators` | `--authenticator NAME --auth-config PATH` | `(config_path: str) -> Authenticator` | `apikey` |
| `membrane.content_stores` | `--content-store NAME` (with `--data-dir`) | `(location: str, key_file: str) -> ContentStore` | `filesystem`, `memory` |
| `membrane.persistence` | `--persistence NAME` (URL from `--redis`) | `(url: str) -> backend` | `redis`, `memory` |
| `membrane.eviction` | `--eviction NAME` | `() -> EvictionPolicy` | `weighted-lru`, `tinylfu` |
| `membrane.secret_providers` | used to resolve secrets | `() -> SecretProvider` (reads its own environment) | `env`, `aws`, `gcp`, `vault` |
| `membrane.placement` | `--placement NAME` | `() -> PlacementPolicy` | `ring`, `latency`, `selector`, `economic`, `joint` |
| `membrane.hooks` | every installed hook runs (`--no-hooks` disables) | `(bus: EventBus, server: Server) -> None` | none |

> [!IMPORTANT]
> A built-in name always wins over an entry point with the same name, so an installed package cannot silently replace a built-in.

An unknown name fails at startup with the list of available names:

```text
membrane serve --compute vlm
... unknown compute backend 'vlm'; available: anthropic, cpu, gpu, ollama, openai, transformers
```

## Writing a compute backend

A compute backend implements `membrane.compute.base.Backend` (see
`membrane/compute/cpu.py` for a complete example). Package it with an
entry point:

```toml title="pyproject.toml (your package)"
[project.entry-points."membrane.compute"]
my-engine = "my_package.membrane_plugin:make_backend"
```

```python title="my_package/membrane_plugin.py"
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

## Writing a hook

Hooks observe the server without changing it. Every installed
`membrane.hooks` entry point is called once while the server is built,
with the server's `EventBus` and the `Server` itself:

```toml
[project.entry-points."membrane.hooks"]
audit-export = "my_package.hooks:install"
```

```python
from membrane.runtime.events import FragmentStored, PeerLeft


def install(bus, server):
    bus.subscribe(FragmentStored, lambda event: ship(event.content_hash, event.tenant_id))
    bus.subscribe(PeerLeft, lambda event: page_oncall(event.node_id))
```

| Event | Fields | Published when |
|-------|--------|----------------|
| `FragmentStored` | `content_hash`, `tenant_id`, `is_primary` | A fragment becomes resident |
| `FragmentRemoved` | `content_hash` | A fragment leaves (eviction, expiry, delete, or rollback) |
| `PeerJoined`, `PeerLeft` | `node_id` | The membership table changes |
| `DrainStarted` | `deadline_sec` | A `SIGTERM` drain begins |
| `DrainFinished` | `migrated`, `stragglers` | The drain's hand-offs are done |

In fragment events, `content_hash` is the storage key: the bare content hash for the `public` tenant, and `<tenant>:<hash>` for others (see [Security](security.md#identical-content-across-tenants)).

> [!WARNING]
> Events are delivered in order on one dispatcher thread, never on the request path. A handler that raises is logged and skipped, and once 10,000 events are queued new ones are dropped. Keep handlers fast, and hand slow work to your own queue.

## Writing an eviction policy

A policy orders eviction candidates; the node removes them until enough
bytes are free (expired fragments always go first):

```python
class OldestFirst:
    def order(self, candidates, access_times, now):
        return [h for h, _ in sorted(candidates, key=lambda c: access_times.get(c[0], now))]

    def touch(self, content_hash):
        pass
```

```toml
[project.entry-points."membrane.eviction"]
oldest-first = "my_package.eviction:OldestFirst"
```

## Writing a placement policy

A placement policy answers `POST /route` ([Memory API](memory-api.md)).
`place` receives a `ClusterView` (`membrane.services.placement`) with
this node's fragments, the gossiped location directory, ring owners, and
each healthy node's last reported load, plus the recently read hashes:

```python
from membrane.services.placement import ClusterView, Placement


class LeastPressure:
    def place(self, view: ClusterView, content_hash: str, history: list[str]) -> Placement:
        telemetry = view.telemetry()
        coolest = min(telemetry, key=lambda node_id: telemetry[node_id].memory_pressure)
        holders = view.holders(content_hash)
        return Placement(
            fetch_from=holders[0] if holders else "",
            store_on=coolest,
            prefill_on=view.local_id,
            reason="least memory pressure",
        )


def make_policy():
    return LeastPressure()
```

```toml
[project.entry-points."membrane.placement"]
least-pressure = "my_plugin.placement:make_policy"
```

`membrane serve --placement least-pressure` then routes with it.

