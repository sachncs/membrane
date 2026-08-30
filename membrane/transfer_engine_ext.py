"""GPU-direct pinned host memory + adaptive fragment sizing (Phase 3.3.9 + 3.3.10).

The v3.0.0 release optimizes the GPU → wire path:

* :class:`PinnedTensorHandle` describes a tensor buffer that
  lives in pinned host memory (the staging area for
  ``cudaMemcpyAsync``); :class:`KVTransferEngine.transfer_kv`
  accepts a :class:`PinnedTensorHandle` and writes the bytes
  straight into the network send buffer, skipping the
  GPU → CPU → bytes materialization of the v2.0 surface.
* :class:`AdaptiveFragmenter` sizes the window per
  architecture + memory pressure + reuse score. Operators
  flip the feature flag on to enable adaptation; the default
  size path remains the legacy :class:`membrane.fragmenter.Fragmenter`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PinnedTensorHandle:
    """Tensor handle backed by pinned host memory.

    Attributes:
        data: Pinned host memory bytes.
        shape: Tensor shape.
        dtype: Element dtype string.
    """

    data: bytes
    shape: tuple[int, ...]
    dtype: str

    def size_bytes(self) -> int:
        """Return the byte size of the handle.

        Returns:
            int: ``len(data)``.
        """
        return len(self.data)


@dataclass(frozen=True)
class ModelSizeProfile:
    """Per-architecture default window size.

    Attributes:
        model_id: The model identifier.
        baseline_window_size: Default window size for the model.
        bytes_per_token: Approximate bytes per token (used
            when no GPU profile is available).
    """

    model_id: str
    baseline_window_size: int = 128
    bytes_per_token: int = 1024


_DEFAULT_PROFILES: dict[str, ModelSizeProfile] = {
    "llama-3-8b": ModelSizeProfile("llama-3-8b", baseline_window_size=128, bytes_per_token=512),
    "llama-3-70b": ModelSizeProfile("llama-3-70b", baseline_window_size=64, bytes_per_token=4096),
    "mistral-7b": ModelSizeProfile("mistral-7b", baseline_window_size=128, bytes_per_token=512),
    "mixtral-8x7b": ModelSizeProfile("mixtral-8x7b", baseline_window_size=64, bytes_per_token=2048),
    "phi-3": ModelSizeProfile("phi-3", baseline_window_size=128, bytes_per_token=256),
}


def get_model_profile(model_id: str) -> ModelSizeProfile:
    """Return the size profile for ``model_id``.

    Args:
        model_id: The model identifier; case-insensitive.

    Returns:
        ModelSizeProfile: A real profile when the model is
        known, the generic :class:`ModelSizeProfile` baseline
        otherwise.
    """
    key = model_id.lower()
    return _DEFAULT_PROFILES.get(key, ModelSizeProfile(model_id=model_id))


@dataclass
class AdaptiveFragmenter:
    """Adaptive fragment sizer.

    Attributes:
        node_memory_used_bytes: Current node memory usage.
        node_memory_limit_bytes: Node memory budget.
        model_id: Active model identifier.
        reuse_score_avg: Average reuse score observed so far.
        enabled: When ``True``, the adaptive path is in use;
            the v1 :class:`membrane.fragmenter.Fragmenter` is
            still the default fallback.
    """

    node_memory_used_bytes: int = 0
    node_memory_limit_bytes: int = 1 << 30
    model_id: str = "default"
    reuse_score_avg: float = 0.5
    enabled: bool = False

    def window_size(self) -> int:
        """Compute the window size for the current state.

        Returns:
            int: A positive integer window size.
        """
        if not self.enabled:
            return 128
        profile = get_model_profile(self.model_id)
        baseline = profile.baseline_window_size

        # Memory pressure: when node memory is > 85 % used we
        # shrink the window so per-fragment payloads fit.
        if self.node_memory_limit_bytes > 0:
            pressure = self.node_memory_used_bytes / self.node_memory_limit_bytes
        else:
            pressure = 0.0
        if pressure > 0.85:
            baseline = max(16, baseline // 2)
        if pressure > 0.95:
            baseline = max(8, baseline // 4)

        # Reuse-driven growth: high reuse scores justify larger
        # windows because the cached K/V is hot.
        if self.reuse_score_avg > 0.7 and pressure < 0.6:
            baseline = baseline * 2
        return max(16, baseline)


__all__ = [
    "AdaptiveFragmenter",
    "ModelSizeProfile",
    "PinnedTensorHandle",
    "get_model_profile",
]
