"""Named plugin registries for compute backends, authenticators, and content stores.

Built-in implementations are registered here. Third-party packages add
their own through :mod:`importlib.metadata` entry points; no change to
Membrane is needed::

    # pyproject.toml of a plugin package
    [project.entry-points."membrane.compute"]
    vllm = "my_plugin.compute:make_backend"

Then ``membrane serve --compute vllm`` loads ``make_backend``. The
groups and factory signatures are:

=========================== ===================================================
Group                       Factory
=========================== ===================================================
``membrane.compute``        ``(llm_url, llm_model, api_key) -> Backend``
``membrane.authenticators`` ``(config_path) -> Authenticator``
``membrane.content_stores`` ``(location, key_file) -> ContentStore``
=========================== ===================================================

A built-in name wins over an entry point with the same name, so an
installed package cannot silently replace a built-in.
"""

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from threading import Lock
from typing import Any, cast

from membrane.auth import Authenticator
from membrane.compute.base import Backend

logger = logging.getLogger(__name__)

type ComputeBackendFactory = Callable[[str, str, str], Backend]
type AuthenticatorFactory = Callable[[str], Authenticator]
type ContentStoreFactory = Callable[[str, str], Any]


class UnknownPluginError(ValueError):
    """Raised when a plugin name is neither built in nor installed."""


class PluginRegistry[F]:
    """Factories of one plugin kind, looked up by name.

    Attributes:
        group: Entry-point group third-party plugins register under.
        kind: Human-readable plugin kind, used in error messages.
    """

    def __init__(self, group: str, kind: str) -> None:
        """Create an empty registry.

        Args:
            group: Entry-point group (e.g. ``"membrane.compute"``).
            kind: Human-readable plugin kind (e.g. ``"compute backend"``).
        """
        self.group = group
        self.kind = kind
        self.__builtins: dict[str, F] = {}
        self.__loaded: dict[str, F] = {}
        self.__lock = Lock()

    def register(self, name: str, factory: F) -> None:
        """Register a built-in factory.

        Args:
            name: Name selected on the command line.
            factory: The factory.
        """
        with self.__lock:
            self.__builtins[name] = factory

    def names(self) -> list[str]:
        """Return every available name, built-in and installed.

        Returns:
            list[str]: Sorted plugin names.
        """
        installed = {ep.name for ep in entry_points(group=self.group)}
        return sorted(self.__builtins.keys() | installed)

    def get(self, name: str) -> F:
        """Return the factory registered as ``name``.

        Args:
            name: Plugin name.

        Returns:
            F: The built-in factory, or the loaded entry point.

        Raises:
            UnknownPluginError: When no plugin has that name.
        """
        with self.__lock:
            if name in self.__builtins:
                return self.__builtins[name]
            if name in self.__loaded:
                return self.__loaded[name]
        matches = entry_points(group=self.group, name=name)
        if not matches:
            raise UnknownPluginError(f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}")
        entry = next(iter(matches))
        factory = cast(F, entry.load())
        logger.info("loaded %s plugin %r from %s", self.kind, name, entry.value)
        with self.__lock:
            self.__loaded[name] = factory
        return factory

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is available.

        Args:
            name: Plugin name.

        Returns:
            bool: True when :meth:`get` would succeed.
        """
        return isinstance(name, str) and name in self.names()


def optional_backend(module: str, class_name: str) -> Any:
    """Import a compute backend class whose dependencies are optional.

    Args:
        module: Module under :mod:`membrane.compute` (e.g. ``"ollama"``).
        class_name: Class to return from it.

    Returns:
        Any: The class, typed loosely because provider constructors differ.
    """
    from importlib import import_module

    return getattr(import_module(f"membrane.compute.{module}"), class_name)


def cpu_backend(_url: str, _model: str, _key: str) -> Backend:
    """Build the CPU backend.

    Args:
        _url: Unused.
        _model: Unused.
        _key: Unused.

    Returns:
        Backend: A :class:`~membrane.compute.cpu.CPU` backend.
    """
    from membrane.compute.cpu import CPU

    return CPU()


def gpu_backend(_url: str, _model: str, _key: str) -> Backend:
    """Build the GPU backend.

    Args:
        _url: Unused.
        _model: Unused.
        _key: Unused.

    Returns:
        Backend: A :class:`~membrane.compute.gpu.GPU` backend.
    """
    return cast(Backend, optional_backend("gpu", "GPU")())


def ollama_backend(url: str, model: str, _key: str) -> Backend:
    """Build the Ollama backend.

    Args:
        url: Ollama base URL; defaults to ``http://localhost:11434``.
        model: Model name; defaults to ``llama3.2``.
        _key: Unused.

    Returns:
        Backend: An Ollama backend.
    """
    return cast(
        Backend,
        optional_backend("ollama", "Ollama")(base_url=url or "http://localhost:11434", model=model or "llama3.2"),
    )


def openai_backend(_url: str, model: str, key: str) -> Backend:
    """Build the OpenAI backend.

    Args:
        _url: Unused.
        model: Model name; defaults to ``gpt-4o-mini``.
        key: OpenAI API key.

    Returns:
        Backend: An OpenAI backend.
    """
    return cast(Backend, optional_backend("openai", "OpenAI")(model=model or "gpt-4o-mini", api_key=key))


def anthropic_backend(_url: str, model: str, key: str) -> Backend:
    """Build the Anthropic backend.

    Args:
        _url: Unused.
        model: Model name; defaults to ``claude-sonnet-5-5``.
        key: Anthropic API key.

    Returns:
        Backend: An Anthropic backend.
    """
    return cast(Backend, optional_backend("anthropic", "Anthropic")(model=model or "claude-sonnet-5-5", api_key=key))


def transformers_backend(_url: str, model: str, _key: str) -> Backend:
    """Build the Hugging Face Transformers backend.

    Args:
        _url: Unused.
        model: Model id; defaults to ``gpt2``.
        _key: Unused.

    Returns:
        Backend: A Transformers backend.
    """
    return cast(Backend, optional_backend("transformers", "Transformers")(model_id=model or "gpt2"))


def apikey_authenticator(config_path: str) -> Authenticator:
    """Build the API-key authenticator from a keyfile.

    Args:
        config_path: Path of the keyfile.

    Returns:
        Authenticator: An :class:`~membrane.auth.apikey.APIKeyAuthenticator`.
    """
    from membrane.auth.apikey import APIKeyAuthenticator

    return APIKeyAuthenticator(Path(config_path).read_text())


def filesystem_content_store(location: str, key_file: str) -> Any:
    """Build the encrypted on-disk content store.

    Args:
        location: Data directory.
        key_file: Optional master key file.

    Returns:
        Any: A :class:`~membrane.content_store.FilesystemBlob`.
    """
    from membrane.runtime.components import build_content_store

    return build_content_store(location, key_file)


def memory_content_store(_location: str, _key_file: str) -> Any:
    """Build the in-process content store (bytes are lost on restart).

    Args:
        _location: Unused.
        _key_file: Unused.

    Returns:
        Any: A :class:`~membrane.content_store.InProcessBytes`.
    """
    from membrane.content_store import InProcessBytes

    return InProcessBytes()


COMPUTE_BACKENDS = PluginRegistry[ComputeBackendFactory]("membrane.compute", "compute backend")
AUTHENTICATORS = PluginRegistry[AuthenticatorFactory]("membrane.authenticators", "authenticator")
CONTENT_STORES = PluginRegistry[ContentStoreFactory]("membrane.content_stores", "content store")

COMPUTE_BACKENDS.register("cpu", cpu_backend)
COMPUTE_BACKENDS.register("gpu", gpu_backend)
COMPUTE_BACKENDS.register("ollama", ollama_backend)
COMPUTE_BACKENDS.register("openai", openai_backend)
COMPUTE_BACKENDS.register("anthropic", anthropic_backend)
COMPUTE_BACKENDS.register("transformers", transformers_backend)
AUTHENTICATORS.register("apikey", apikey_authenticator)
CONTENT_STORES.register("filesystem", filesystem_content_store)
CONTENT_STORES.register("memory", memory_content_store)

__all__ = [
    "AUTHENTICATORS",
    "COMPUTE_BACKENDS",
    "CONTENT_STORES",
    "AuthenticatorFactory",
    "ComputeBackendFactory",
    "ContentStoreFactory",
    "PluginRegistry",
    "UnknownPluginError",
    "anthropic_backend",
    "apikey_authenticator",
    "cpu_backend",
    "filesystem_content_store",
    "gpu_backend",
    "memory_content_store",
    "ollama_backend",
    "openai_backend",
    "optional_backend",
    "transformers_backend",
]
