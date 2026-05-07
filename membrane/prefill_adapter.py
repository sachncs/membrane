"""PrefillAdapter: wraps the Membrane analytical model."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.fragmentation_engine import FragmentationEngine
from membrane.model.profiler import kv_size_mib, prefill_time_seconds
from membrane.model.router import Router, RoutingDecision


@dataclass(frozen=True)
class PrefillResult:
    """Outcome of a simulated prefill operation.

    Attributes:
        kv_size_mib: Estimated KV cache size in MiB.
        latency_seconds: Estimated prefill latency in seconds.
        routing_decision: Optional routing decision from Membrane Router.
        fragments: Fragments produced from the KV output.
    """

    kv_size_mib: float
    latency_seconds: float
    routing_decision: RoutingDecision | None
    fragments: list[Fragment]


class PrefillAdapter:
    """Adapts the Membrane analytical model for integration.

    Treats profiler functions as a model-based prefill service. Returns
    synthetic KV metadata that is immediately converted to fragments.
    """

    def __init__(
        self,
        router: Router | None = None,
        compute_scale: float = 1.0,
        fragmentation_engine: FragmentationEngine | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            router: Optional Membrane Router for offloading decisions.
            compute_scale: Hardware compute scale factor (1.0 = H200).
            fragmentation_engine: Engine to convert KV output into fragments.
        """
        self.router = router
        self.compute_scale = compute_scale
        self.fragmentation_engine = fragmentation_engine or FragmentationEngine()

    def prefill(self, prompt_tokens: list[int], model_id: str) -> PrefillResult:
        """Simulate prefill and return KV metadata plus fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.

        Returns:
            PrefillResult with estimated sizes, latency, and fragments.
        """
        length = len(prompt_tokens)
        size = kv_size_mib(length)
        latency = prefill_time_seconds(length, self.compute_scale)

        decision = None
        if self.router is not None:
            decision = self.router.route(length)

        fragments = self.convert_kv_to_fragments(prompt_tokens, model_id, size)

        return PrefillResult(
            kv_size_mib=size,
            latency_seconds=latency,
            routing_decision=decision,
            fragments=fragments,
        )

    def convert_kv_to_fragments(
        self,
        prompt_tokens: list[int],
        model_id: str,
        kv_size_mib: float,
    ) -> list[Fragment]:
        """Convert simulated KV output into content-addressed fragments.

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            kv_size_mib: Total KV size to distribute across fragments.

        Returns:
            Fragments representing the KV tensor windows.
        """
        if not prompt_tokens:
            return []

        frags = self.fragmentation_engine.create_windows(prompt_tokens, model_id)
        if not frags:
            return []

        total_prompt_tokens = len(prompt_tokens)
        bytes_per_token = (kv_size_mib * 1024.0 * 1024.0) / total_prompt_tokens

        sized_frags = []
        for frag in frags:
            span = frag.structural_signature.token_span
            num_tokens = span[1] - span[0] + 1
            frag_size = int(num_tokens * bytes_per_token)

            sized_frag = Fragment(
                content_hash=frag.content_hash,
                embedding=frag.embedding,
                structural_signature=frag.structural_signature,
                size=max(1, frag_size),
                ttl=frag.ttl,
                reuse_score=frag.reuse_score,
                version_id=frag.version_id,
            )
            sized_frags.append(sized_frag)

        return sized_frags
