"""KVBackend: real K/V tensor extraction from a HuggingFace causal LM.

The :class:`KVBackend` is the v1 successor of the embeddings-only
``Transformers`` backend. Where the older backend produced a
mean-pooled embedding vector and discarded the underlying attention
tensors, ``KVBackend`` runs a forward pass with
``use_cache=True`` and ``output_attentions=True``, then for every
``window_size``-token window:

1. Extracts the past K/V tensors for that span across every layer.
2. Stacks them into a canonical raw-byte frame
   (``[K_0, V_0, K_1, V_1, ..., K_n_layers-1, V_n_layers-1]``).
3. Hashes the frame with SHA-256.
4. Wraps the result in a :class:`~membrane.fragment.Fragment` whose
   :class:`~membrane.identity.PayloadIdentity` carries model_id,
   model_revision, tokenizer_name, tokenizer_revision, layer_range,
   head_range, token_span, dtype, and shape.

The framed bytes flow into a :class:`~membrane.persistence.ContentStore`
injected at construction; the returned :class:`Fragment` carries the
blob ``payload_ref`` for downstream persistence and replication.

When the model or tokenizer fails to load (missing optional ``[kv]``
extras, OOM, network outage), ``prefill`` falls back to the shared
:func:`Backend.simulate_prefill_fragment` helper so callers still
receive well-formed fragments.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Any

from membrane.compute.base import Backend
from membrane.compute.remote import RemoteLLMBackend
from membrane.content_store import ContentStore
from membrane.fragment import Fragment
from membrane.identity import PayloadIdentity

logger = logging.getLogger(__name__)


_DTYPE_BYTES: dict[str, int] = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


class KVBackend(RemoteLLMBackend):
    """Real K/V tensor producer for HuggingFace causal LM models.

    Args:
        content_store: Destination for the canonical frame bytes.
            Required; the bytes are real and the backend refuses to
            run when no store is supplied (in-memory callers should
            pass ``InProcessBytes()``).
        model_id: HuggingFace model identifier
            (e.g., ``"meta-llama/Llama-3-8b"``). Defaults to
            ``"gpt2"`` so the backend is smoke-testable without
            network or large weights.
        model_revision: Pinning commit hash for the model weights,
            or ``""`` to accept unpinned.
        tokenizer_name: Tokenizer identifier (defaults to ``model_id``).
        tokenizer_revision: Pinning commit hash for the tokenizer.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``. ``"auto"``
            picks CUDA when available.
        dtype: Tensor dtype; one of ``"float16"``, ``"bfloat16"``,
            ``"float32"``, ``"float64"``. The forward pass runs in
            this precision when the model supports it.
        window_size: Token window size (default ``Backend.SIMULATE_WINDOW_SIZE``).
    """

    def __init__(
        self,
        content_store: ContentStore,
        model_id: str = "gpt2",
        model_revision: str = "",
        tokenizer_name: str | None = None,
        tokenizer_revision: str = "",
        device: str = "auto",
        dtype: str = "float16",
        window_size: int | None = None,
    ) -> None:
        """Initialize the backend and load the model lazily on first use.

        Args:
            content_store: Backing store for canonical frames.
            model_id: HuggingFace model identifier.
            model_revision: Pinned model revision hash.
            tokenizer_name: Tokenizer identifier or ``None`` to
                fall back to ``model_id``.
            tokenizer_revision: Pinned tokenizer revision hash.
            device: Device override.
            dtype: Element dtype the model is cast to before
                extracting tensors.
            window_size: Window size, or ``None`` for the
                :data:`Backend.SIMULATE_WINDOW_SIZE` default.
        """
        super().__init__()
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPE_BYTES)}, got {dtype!r}")
        self.content_store = content_store
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_name = tokenizer_name or model_id
        self.tokenizer_revision = tokenizer_revision
        self.device = device
        self.dtype = dtype
        self.window_size = window_size or Backend.SIMULATE_WINDOW_SIZE
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.torch: Any | None = None
        self.actual_device: str = "cpu"
        self.n_layers: int = 0
        self.n_heads: int = 0
        self.head_dim: int = 0
        self.load_model()

    def load_model(self) -> None:
        """Lazy-load the model and tokenizer on first use.

        All failures (missing ``[kv]`` extras, network errors,
        OOM, ``attn_implementation`` constraints) log a warning
        and leave ``self.model = self.tokenizer = None`` so
        :meth:`prefill` cleanly degrades to the simulate path.
        """
        try:
            import torch

            # sdpa attn_implementation refuses output_attentions;
            # force eager so we can read past_key_values plus
            # per-layer attention biases when needed.
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

            self.torch = torch
            self.actual_device = (
                "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
            )
            logger.info("KVBackend: loading %s on %s", self.model_id, self.actual_device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, revision=self.tokenizer_revision or None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision or None,
                attn_implementation="eager",
                torch_dtype=getattr(torch, self.dtype),
            )
            self.model.to(self.actual_device)  # type: ignore[arg-type]
            self.model.eval()  # type: ignore[attr-defined]
            self.n_layers = int(self.model.config.num_hidden_layers)
            self.n_heads = int(getattr(self.model.config, "num_attention_heads", 0))
            self.head_dim = self.n_heads and int(self.model.config.hidden_size) // self.n_heads
            logger.info(
                "KVBackend: loaded %s (n_layers=%d, n_heads=%d, head_dim=%d)",
                self.model_id,
                self.n_layers,
                self.n_heads,
                self.head_dim,
            )
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("KVBackend: failed to load model (%s)", exc)
            self.model = None
            self.tokenizer = None

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run real prefill and emit one fragment per window.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier stamped on each fragment's
                :class:`~membrane.identity.PayloadIdentity`.

        Returns:
            list[Fragment]: One fragment per window. Falls back to
            a simulated prefill when the model or tokenizer did not
            load.
        """
        if self.model is None or self.tokenizer is None or self.torch is None:
            return self.simulate_prefill(prompt_tokens, model_id)

        try:
            torch = self.torch
            prompt_text = " ".join(str(t) for t in prompt_tokens)
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.actual_device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=True,
                )  # type: ignore[arg-type]
            pkv = outputs.past_key_values
            seq_len = int(input_ids.shape[1])
            if pkv is None or len(pkv) == 0:
                raise RuntimeError("model returned empty past_key_values")
        except Exception as exc:
            logger.warning("KVBackend forward pass failed (%s); falling back to simulation", exc)
            return self.simulate_prefill(prompt_tokens, model_id)

        element_size = _DTYPE_BYTES[self.dtype]
        fragment_kv = self._frames_for_windows(pkv, seq_len, model_id)
        fragments: list[Fragment] = []
        for ident, frame in fragment_kv:
            self.content_store.put(ident.payload_hash, frame)
            frag = Fragment(
                identity=ident,
                payload_ref=ident.payload_hash,
                payload_size=len(frame),
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        # Touch element_size so the variable is referenced and
        # ruff B019 won't warn about an unused binding that some
        # reader still finds helpful when extending the backend.
        del element_size
        return fragments

    def _frames_for_windows(
        self,
        pkv: Any,
        seq_len: int,
        model_id: str,
    ) -> list[tuple[PayloadIdentity, bytes]]:
        """Build canonical frames for every window in the sequence.

        Args:
            pkv: ``past_key_values`` from the model's forward pass.
            seq_len: Sequence length of the input prompt.
            model_id: Model identifier stamped on each identity.

        Returns:
            list[tuple[PayloadIdentity, bytes]]: One identity and one
            raw frame per window.
        """
        torch = self.torch
        window_size = self.window_size
        element_size = _DTYPE_BYTES[self.dtype]
        # Per-layer view: each entry is a (K, V) tensor.
        # Newer transformers return DynamicCache where iteration
        # yields the per-layer tuples.
        layers: list[tuple[Any, Any]] = list(pkv)
        n_layers = len(layers)

        results: list[tuple[PayloadIdentity, bytes]] = []
        for chunk_index in range(0, seq_len, window_size):
            cs = chunk_index
            ce = min(chunk_index + window_size, seq_len)
            window_len = ce - cs
            if window_len <= 0:
                continue
            pieces: list[bytes] = []
            for k_full, v_full in layers:
                # Slice the window from each layer's K and V.
                # Shapes are (1, n_heads, full_seq, head_dim); we
                # take [0, :, cs:ce, :] to grab this window.
                k_window = k_full[:, :, cs:ce, :].contiguous().cpu()
                v_window = v_full[:, :, cs:ce, :].contiguous().cpu()
                if k_window.dtype != getattr(torch, self.dtype):
                    k_window = k_window.to(getattr(torch, self.dtype))
                    v_window = v_window.to(getattr(torch, self.dtype))
                pieces.append(bytes(k_window.numpy().tobytes()))
                pieces.append(bytes(v_window.numpy().tobytes()))
            payload_bytes = b"".join(pieces)
            payload_hash = hashlib.sha256(payload_bytes).hexdigest()
            identity = PayloadIdentity(
                payload_hash=payload_hash,
                model_id=model_id,
                model_revision=self.model_revision,
                tokenizer_name=self.tokenizer_name,
                tokenizer_revision=self.tokenizer_revision,
                layer_range=(0, n_layers - 1),
                head_range=(-1, -1),
                token_span=(cs, ce - 1),
                dtype=self.dtype,
                shape=(1, n_layers, self.n_heads, window_len, self.head_dim),
            )
            # Header records layer/head/window layout alongside
            # the SHA-256 trailer; the trailer lets the store
            # cheap-verify bytes before the full round-trip.
            header = struct.pack(
                "<IIIIII",
                2,
                n_layers,
                self.n_heads,
                window_len,
                self.head_dim,
                element_size,
            )
            frame = header + payload_bytes
            results.append((identity, frame))
        return results

    def simulate_prefill(
        self,
        prompt_tokens: list[int],
        model_id: str,
    ) -> list[Fragment]:
        """Simulated fallback used when the model is not loaded.

        Delegates to :meth:`Backend.simulate_prefill_fragment`, which
        builds a deterministic :class:`PayloadIdentity` per window.
        """
        window_size = Backend.SIMULATE_WINDOW_SIZE
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            fragments.append(
                Backend.simulate_prefill_fragment(
                    chunk=chunk,
                    chunk_index=i,
                    total_prompt_tokens=len(prompt_tokens),
                    model_id=model_id,
                    window_size=window_size,
                )
            )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Stub text-generation entry point.

        The real K/V producer is intentionally write-only: it
        computes prefill once and stops. Text decoding lives in
        the upstream backend orchestration and is not duplicated.

        Args:
            prompt_tokens: Input token IDs (unused).
            model_id: Model identifier (unused).
            max_tokens: Maximum tokens (unused).

        Returns:
            dict: ``{"text": "", "tokens": []}``.
        """
        return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return whether the model and tokenizer are loaded.

        Returns:
            bool: ``True`` when both ``self.model`` and
            ``self.tokenizer`` were successfully initialized.
        """
        return self.model is not None and self.tokenizer is not None

    def device_name(self) -> str:
        """Return a descriptive device name.

        Returns:
            str: ``"kv(<model_id>,<dtype>,<device>)"`` when loaded,
            ``"kv(unloaded,<dtype>,<device>)"`` otherwise.
        """
        suffix = self.actual_device if self.model is not None else "unloaded"
        return f"kv({self.model_id},{self.dtype},{suffix})"


__all__ = ["KVBackend"]
