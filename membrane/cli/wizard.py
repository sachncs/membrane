"""Interactive setup wizard.

Prompted when ``membrane serve`` is invoked with all defaults in a
TTY, or explicitly with ``--interactive``. Returns a plain ``dict``
of configuration values; the caller is responsible for building
:class:`~membrane.network.config.ClusterConfig` and
:class:`~membrane.server.Server` from it.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

console = Console()


def interactive_setup() -> dict[str, Any]:
    """Prompt the user for configuration values interactively.

    Each question shows the current default in brackets; pressing
    Enter accepts the default. Invalid values are re-prompted until
    a valid one is supplied.

    Returns:
        dict[str, Any]: Configuration dictionary with the
        following keys: ``node_id``, ``host``, ``port``,
        ``transport``, ``compute``, ``llm_url``, ``llm_model``,
        ``api_key``, ``redis_url``, ``peers``, ``max_memory``,
        ``log_level``.
    """
    console.print("[bold cyan]Membrane Setup Wizard[/bold cyan]")
    console.print("Press Enter to accept defaults (shown in brackets).\n")

    def ask(prompt: str, default: str = "") -> str:
        full = f"{prompt} [{default}]: " if default else f"{prompt}: "
        val = input(full).strip()
        return val if val else default

    def ask_bool(prompt: str, default: bool = True) -> bool:
        suffix = "Y/n" if default else "y/N"
        val = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes", "true", "1")

    node_id = ask("Node ID", "membrane-0")
    host = ask("Bind host", "0.0.0.0")
    port = int(ask("Listen port", "8080"))

    transport = ask("Transport (http/grpc)", "http")
    while transport not in ("http", "grpc"):
        console.print("[red]Invalid transport. Choose 'http' or 'grpc'.[/red]")
        transport = ask("Transport (http/grpc)", "http")

    valid_backends = ("cpu", "gpu", "ollama", "openai", "anthropic", "transformers")
    compute = ask("Compute backend (cpu/gpu/ollama/openai/anthropic/transformers)", "cpu")
    while compute not in valid_backends:
        console.print(f"[red]Invalid compute. Choose one of: {', '.join(valid_backends)}.[/red]")
        compute = ask("Compute backend", "cpu")

    llm_url = ""
    llm_model = ""
    api_key = ""
    if compute == "ollama":
        llm_url = ask("Ollama URL", "http://localhost:11434")
        llm_model = ask("Ollama model", "llama3.2")
    elif compute == "openai":
        api_key = ask("OpenAI API key", "")
        llm_model = ask("OpenAI model", "gpt-4o-mini")
    elif compute == "anthropic":
        api_key = ask("Anthropic API key", "")
        llm_model = ask("Anthropic model", "claude-3-sonnet-20240229")
    elif compute == "transformers":
        llm_model = ask("HuggingFace model ID", "gpt2")

    use_redis = ask_bool("Use Redis persistence?")
    redis_url = ""
    if use_redis:
        redis_url = ask("Redis URL", "redis://localhost:6379/0")

    peers_input = ask(
        "Seed peers (comma-separated host:port, or leave empty)",
        "",
    )
    peers = [p.strip() for p in peers_input.split(",") if p.strip()]

    max_memory = int(ask("Max memory (bytes)", str(1 << 30)))
    log_level = ask("Log level", "INFO")

    return {
        "node_id": node_id,
        "host": host,
        "port": port,
        "transport": transport,
        "compute": compute,
        "llm_url": llm_url,
        "llm_model": llm_model,
        "api_key": api_key,
        "redis_url": redis_url,
        "peers": peers,
        "max_memory": max_memory,
        "log_level": log_level,
    }


__all__ = ["interactive_setup"]
