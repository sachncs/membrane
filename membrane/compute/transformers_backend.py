"""TransformersBackend: compute backend loading real HuggingFace model weights.

Requires ``transformers`` and ``torch`` (installed via
``pip install membrane[local-llm]``).

This backend wraps a HuggingFace Transformers model and tokenizer
and exposes the standard :class:`~membrane.compute.backend
.ComputeBackend` interface. On
:meth:`TransformersBackend.prefill` it runs a forward pass and
uses the last hidden state as the fragment embedding — the
average over each 128-token window becomes the embedding for
that fragment's :class:`~membrane.fragment.Fragment`. On
:meth:`TransformersBackend.generate` it invokes
``model.generate`` and decodes the freshly produced tokens.

Failure modes:

* ``transformers`` or ``torch`` is missing → the model and
  tokenizer remain ``None``; prefill and generate fall back to a
  small simulation.
* The forward pass raises for any reason → the prefill result is
  produced by :meth:`simulate_prefill` and a warning is logged.

The embedding is truncated to 256 dimensions to keep
:attr:`membrane.fragment.Fragment.embedding` bounded regardless
of the underlying model's hidden size.
"""

import hashlib
import logging
from typing import Any

from membrane.compute.backend import ComputeBackend
from membrane.fragment import Fragment
from membrane.structural_signature import StructuralSignature

logger = logging.getLogger(__name__)


class TransformersBackend(ComputeBackend):
    """Compute backend using HuggingFace Transformers for local inference.

    Loads a real model and tokenizer, runs forward passes to
    extract hidden states as embeddings, and uses
    ``model.generate()`` for text generation.

    Args:
        model_id: HuggingFace model identifier
            (e.g., ``"meta-llama/Llama-2-7b-hf"``). Defaults to
            ``"gpt2"`` so the backend can be smoke-tested
            without downloading large weights.
        device: Override device (``"cpu"``, ``"cuda"``, or
            ``"auto"``). When ``"auto"``, CUDA is preferred
            when available.
    """

    def __init__(self, model_id: str = "gpt2", device: str = "auto") -> None:
        """Initialize the backend and load the model.

        Args:
            model_id: HuggingFace model identifier.
            device: Device selection. ``"auto"`` picks CUDA if
                available.
        """
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._actual_device: str = "cpu"
        self.load_model()

    def load_model(self) -> None:
        """Load ``model_id`` and its tokenizer.

        All failures (missing dependencies, network errors,
        OOM) are caught and converted into a logged warning so
        the backend can still be instantiated in degraded
        environments.
        """
        try:
            import torch
            # transformers is an optional dependency. mypy cannot see
            # its type stubs without an explicit [transformers] extra,
            # hence the type: ignore on the import line below.
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

            self._torch = torch
            if self.device == "auto":
                # Prefer CUDA when available, fall back to CPU.
                self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._actual_device = self.device

            logger.info("TransformersBackend: loading %s on %s", self.model_id, self._actual_device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to(self._actual_device)
            self._model.eval()
            logger.info("TransformersBackend: loaded %s", self.model_id)
        except ImportError:
            logger.warning("TransformersBackend: transformers or torch not installed")
        except (OSError, RuntimeError, ValueError) as exc:
            # Network errors, OOM, or invalid model IDs land here.
            logger.warning("TransformersBackend: failed to load model (%s)", exc)

    def hash_tokens(self, tokens: list[int]) -> str:
        """MD5-hash a token chunk.

        Args:
            tokens: Token IDs to hash.

        Returns:
            str: Hexadecimal digest.
        """
        payload = ",".join(str(t) for t in tokens)
        return hashlib.md5(payload.encode()).hexdigest()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run a forward pass to obtain hidden-state embeddings.

        If the model failed to load, falls back to a small
        simulation. The simulation produces fragments with a
        two-element embedding so downstream tests still receive
        well-formed fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier stamped on each
                fragment's structural signature.

        Returns:
            list[Fragment]: One fragment per 128-token window.
        """
        if self._model is None or self._tokenizer is None:
            return self.simulate_prefill(prompt_tokens, model_id)

        try:
            import torch

            with torch.no_grad():
                inputs = self._tokenizer(
                    " ".join(str(t) for t in prompt_tokens),
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}
                outputs = self._model(**inputs, output_hidden_states=True)
                # Last layer's hidden state: (batch, seq_len, hidden_dim).
                hidden_states = outputs.hidden_states[-1]
                # Drop batch dim and move to CPU as numpy.
                embeddings = hidden_states[0].cpu().numpy()
        except (RuntimeError, ValueError, IndexError) as exc:
            # OOM, device errors, or shape mismatches degrade to
            # simulated prefill.
            logger.warning(
                "Transformers forward pass failed (%s); falling back to simulation", exc
            )
            return self.simulate_prefill(prompt_tokens, model_id)

        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self.hash_tokens(chunk)
            # Average embeddings over the chunk window so each
            # fragment is represented by a single fixed-size
            # vector.
            emb_slice = embeddings[i : i + window_size]
            if len(emb_slice) > 0:
                avg_emb = emb_slice.mean(axis=0).tolist()
            else:
                avg_emb = [0.0]
            frag = Fragment(
                content_hash=h,
                # Cap the embedding at 256 dimensions regardless
                # of the model's hidden size.
                embedding=tuple(avg_emb[:256]),
                structural_signature=StructuralSignature(
                    model_id=model_id,
                    layer_range=(0, 1),
                    token_span=(i, min(i + window_size, len(prompt_tokens)) - 1),
                ),
                size=len(chunk) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        logger.debug(
            "TransformersBackend: prefill %s tokens into %s fragments",
            len(prompt_tokens),
            len(fragments),
        )
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate text with ``model.generate``.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier (currently unused —
                ``self.model_id`` is the source of truth).
            max_tokens: Maximum number of new tokens to
                generate.

        Returns:
            dict: ``{"text": ..., "tokens": [...]}``. Empty
            strings and lists when the model is not loaded or
            generation fails.
        """
        if self._model is None or self._tokenizer is None:
            return {"text": "", "tokens": []}
        try:
            import torch

            prompt_text = " ".join(str(t) for t in prompt_tokens)
            inputs = self._tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=2048
            )
            inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            # Slice off the prompt portion to keep only the
            # newly generated tokens.
            new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            return {"text": text, "tokens": new_ids.tolist()}
        except (RuntimeError, ValueError, IndexError) as exc:
            logger.warning("Transformers generate failed: %s", exc)
            return {"text": "", "tokens": []}

    def available(self) -> bool:
        """Return whether the model and tokenizer are loaded.

        Returns:
            bool: True when both ``_model`` and ``_tokenizer``
            were successfully initialized.
        """
        return self._model is not None and self._tokenizer is not None

    def device_name(self) -> str:
        """Return a descriptive device name.

        Returns:
            str: ``"transformers(<model>,<device>)"`` when
            loaded, ``"transformers(unloaded)"`` otherwise.
        """
        if self._model is None:
            return "transformers(unloaded)"
        return f"transformers({self.model_id},{self._actual_device})"

    def simulate_prefill(
        self,
        prompt_tokens: list[int],
        model_id: str,
    ) -> list[Fragment]:
        """Produce a simulated prefill (used when no model is loaded).

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier for the structural
                signature.

        Returns:
            list[Fragment]: One fragment per 128-token window,
            with a placeholder embedding.
        """
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self.hash_tokens(chunk)
            frag = Fragment(
                content_hash=h,
                embedding=(float(i), float(len(chunk))),
                structural_signature=StructuralSignature(
                    model_id=model_id,
                    layer_range=(0, 1),
                    token_span=(i, min(i + window_size, len(prompt_tokens)) - 1),
                ),
                size=len(chunk) * 64,
                ttl=3600.0,
                reuse_score=0.5,
                version_id=1,
            )
            fragments.append(frag)
        return fragments
