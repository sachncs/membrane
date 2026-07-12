"""PrefillAdapter: wraps the Membrane analytical model.

This module defines :class:`PrefillAdapter` and the supporting
:class:`PrefillResult` dataclass. The adapter is the bridge
between the analytical throughput model from the
"Prefill-as-a-Service" paper (see
:mod:`membrane.model.profiler`) and the content-addressed fragment
fabric.

Given a prompt, the adapter:

1. Estimates the KV cache size and prefill latency using the
   analytical model.
2. Optionally consults a :class:`~membrane.model.router.Router`
   for offload routing decisions.
3. Converts the simulated KV output into a list of fragments
   sized to reflect the estimated per-token footprint.

The adapter is the *only* place where the analytical model's
output crosses into the fragment store. Higher-level components
(e.g., the reconstruction engine) can therefore rely on a
consistent fragment representation regardless of which underlying
backend produced the prefill.
"""

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
        routing_decision: Optional routing decision from
            :class:`Router`. ``None`` when no router is
            configured.
        fragments: Fragments produced from the KV output.
    """

    kv_size_mib: float
    latency_seconds: float
    routing_decision: RoutingDecision | None
    fragments: list[Fragment]


class PrefillAdapter:
    """Adapts the Membrane analytical model for integration.

    Treats profiler functions as a model-based prefill service.
    Returns synthetic KV metadata that is immediately converted
    to fragments.

    Attributes:
        router: Optional :class:`Router` consulted for offload
            decisions.
        compute_scale: Hardware compute scale factor.
            ``1.0`` represents the H200 reference hardware.
        fragmentation_engine: Engine used to convert the
            simulated KV output into fragments.
    """

    def __init__(
        self,
        router: Router | None = None,
        compute_scale: float = 1.0,
        fragmentation_engine: FragmentationEngine | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            router: Optional Membrane Router for offloading
                decisions.
            compute_scale: Hardware compute scale factor
                (``1.0`` corresponds to the H200 reference).
            fragmentation_engine: Engine to convert KV output
                into fragments. A default
                :class:`FragmentationEngine` is used when
                ``None``.
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
            PrefillResult: Estimated sizes, latency, optional
            routing decision, and the corresponding fragment
            chain.
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

        The fragment chain mirrors the prompt windows produced by
        :class:`FragmentationEngine`. Each fragment's ``size``
        is set to reflect its share of the total estimated KV
        footprint (``kv_size_mib`` distributed by token count).

        Args:
            prompt_tokens: Input token IDs.
            model_id: Model identifier.
            kv_size_mib: Total KV size to distribute across
                fragments.

        Returns:
            list[Fragment]: Fragments representing the KV tensor
            windows. Empty when ``prompt_tokens`` is empty or
            fragmentation yields no windows.
        """
        if not prompt_tokens:
            return []

        # Window the prompt into content-addressable fragments.
        frags = self.fragmentation_engine.create_windows(prompt_tokens, model_id)
        if not frags:
            return []

        # Distribute the estimated KV size uniformly across
        # tokens, then size each fragment by its token span.
        total_prompt_tokens = len(prompt_tokens)
        bytes_per_token = (kv_size_mib * 1024.0 * 1024.0) / total_prompt_tokens

        sized_frags = []
        for frag in frags:
            span = frag.structural_signature.token_span
            num_tokens = span[1] - span[0] + 1
            frag_size = int(num_tokens * bytes_per_token)

            # Rebuild the fragment with the size override; all
            # other fields are preserved from the fragmentation
            # engine's output.
            sized_frag = Fragment(
                content_hash=frag.content_hash,
                embedding=frag.embedding,
                structural_signature=frag.structural_signature,
                # max(1, ...) guards against zero-sized
                # fragments that would otherwise be filtered by
                # downstream stores.
                size=max(1, frag_size),
                ttl=frag.ttl,
                reuse_score=frag.reuse_score,
                version_id=frag.version_id,
            )
            sized_frags.append(sized_frag)

        return sized_frags
