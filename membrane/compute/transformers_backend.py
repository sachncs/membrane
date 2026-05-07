"""TransformersBackend: compute backend loading real HuggingFace model weights.

Requires ``transformers`` and ``torch`` (installed via ``pip install membrane[local-llm]``).
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

    Loads a real model and tokenizer, runs forward passes to extract
    hidden states as embeddings, and uses ``model.generate()`` for text generation.

    Args:
        model_id: HuggingFace model identifier (e.g. ``"meta-llama/Llama-2-7b-hf"``).
        device: Override device (``"cpu"``, ``"cuda"``, or ``"auto"``).
    """

    def __init__(self, model_id: str = "gpt2", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._actual_device: str = "cpu"
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

            self._torch = torch
            if self.device == "auto":
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
        except Exception as exc:
            logger.warning("TransformersBackend: failed to load model (%s)", exc)

    def _hash_tokens(self, tokens: list[int]) -> str:
        payload = ",".join(str(t) for t in tokens)
        return hashlib.md5(payload.encode()).hexdigest()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        """Run forward pass to get hidden states as real embeddings."""
        if self._model is None or self._tokenizer is None:
            return self._simulate_prefill(prompt_tokens, model_id)

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
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                embeddings = hidden_states[0].cpu().numpy()  # (seq_len, hidden_dim)
        except Exception as exc:
            logger.warning("Transformers forward pass failed (%s); falling back to simulation", exc)
            return self._simulate_prefill(prompt_tokens, model_id)

        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self._hash_tokens(chunk)
            # Average embeddings over the chunk window
            emb_slice = embeddings[i : i + window_size]
            if len(emb_slice) > 0:
                avg_emb = emb_slice.mean(axis=0).tolist()
            else:
                avg_emb = [0.0]
            frag = Fragment(
                content_hash=h,
                embedding=tuple(avg_emb[:256]),  # cap embedding dimension
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
        logger.debug("TransformersBackend: prefill %s tokens into %s fragments", len(prompt_tokens), len(fragments))
        return fragments

    def generate(self, prompt_tokens: list[int], model_id: str, max_tokens: int = 128) -> dict:
        """Generate text using model.generate()."""
        if self._model is None or self._tokenizer is None:
            return {"text": "", "tokens": []}
        try:
            import torch

            prompt_text = " ".join(str(t) for t in prompt_tokens)
            inputs = self._tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self._actual_device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            # Decode only the newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            return {"text": text, "tokens": new_ids.tolist()}
        except Exception as exc:
            logger.warning("Transformers generate failed: %s", exc)
            return {"text": "", "tokens": []}

    def available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def device_name(self) -> str:
        if self._model is None:
            return "transformers(unloaded)"
        return f"transformers({self.model_id},{self._actual_device})"

    def _simulate_prefill(self, prompt_tokens: list[int], model_id: str) -> list[Fragment]:
        window_size = 128
        fragments: list[Fragment] = []
        for i in range(0, len(prompt_tokens), window_size):
            chunk = prompt_tokens[i : i + window_size]
            h = self._hash_tokens(chunk)
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
