"""Dual-timescale scheduler from Section 3.4.3.

Short-term:
    * Bandwidth- and cache-aware routing adjustments.
    * When Membrane egress utilization nears the bandwidth
      ceiling, the effective routing threshold is raised so only
      longer requests are offloaded.

Long-term:
    * Traffic-driven reallocation of PD instances between
      prefill and decode.
    * Periodically re-runs the grid search in
      :mod:`membrane.model.optimizer` to restore the optimality
      conditions of Equations (7) and (8) from the paper, then
      updates threshold ``t`` and the ``N_p`` / ``N_d`` split.

References:
    * "Prefill-as-a-Service", arXiv:2604.15039v2, §3.4.3.
"""

import logging

logger = logging.getLogger(__name__)


from collections import deque
from dataclasses import dataclass, field
from typing import List

from membrane.model import optimizer

# ---------------------------------------------------------------------------
# Constants derived from the analytical model (Section 3.4.1 & 3.4.3)
# ---------------------------------------------------------------------------

#: Default egress utilization fraction that triggers short-term
#: threshold raising. Chosen at ``0.85`` to provide headroom before
#: bandwidth saturation.
DEFAULT_CONGESTION_THRESHOLD: float = 0.85

#: Hard cap on effective threshold (tokens). Matches
#: :data:`membrane.model.workload.MAX_LENGTH` and prevents
#: unbounded growth during prolonged congestion.
MAX_EFFECTIVE_THRESHOLD_TOKENS: int = 131_072

#: Multiplier applied to ``effective_threshold`` when congestion
#: is detected. A 1.10x (10%) raise quickly throttles short
#: offload traffic.
THRESHOLD_RAISE_MULTIPLIER: float = 1.10

#: Multiplier applied when relaxing ``effective_threshold`` back
#: toward the base value. A 0.99x (1%) decay per call yields
#: smooth convergence without oscillation.
THRESHOLD_RELAX_MULTIPLIER: float = 0.99


@dataclass
class EgressMonitor:
    """Tracks moving-average Membrane egress utilization.

    Attributes:
        window_size: Maximum number of samples retained in
            the rolling window.
        history: Bounded :class:`collections.deque` of recent
            utilization samples in ``[0.0, 1.0]``.
    """

    window_size: int = 100
    history: deque[float] = field(default_factory=lambda: deque())

    def __post_init__(self):
        # Ensure the deque has the correct maxlen so that
        # historical samples are dropped automatically.
        object.__setattr__(
            self, "history", deque(self.history, maxlen=self.window_size)
        )

    def record(self, utilization: float) -> None:
        """Record a new utilization sample.

        Args:
            utilization: Sample in ``[0.0, 1.0]``. The deque
                silently drops the oldest sample once the
                window is full.
        """
        self.history.append(utilization)

    def average(self) -> float:
        """Return current moving-average utilization.

        Returns:
            float: Mean of the retained samples, or ``0.0``
            when the deque is empty.
        """
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


@dataclass
class SchedulerState:
    """Mutable state for the dual-timescale scheduler.

    Attributes:
        threshold: Base routing threshold ``t``.
        num_pd_p: Current number of PD-P instances.
        num_pd_d: Current number of PD-D instances.
        monitor: Egress utilization monitor.
        bandwidth_abundant: Whether the scheduler operates in
            the bandwidth-abundant regime.
        effective_threshold: Currently-applied threshold; may
            deviate from ``threshold`` during congestion.
    """

    threshold: int
    num_pd_p: int
    num_pd_d: int
    monitor: EgressMonitor = field(default_factory=lambda: EgressMonitor())
    bandwidth_abundant: bool = False
    effective_threshold: int = 0  # set dynamically from threshold

    def __post_init__(self):
        if self.effective_threshold == 0:
            self.effective_threshold = self.threshold


class DualTimescaleScheduler:
    """Scheduler that reacts on two time scales.

    Short-term:
        * Monitors Membrane egress utilization and queue depth.
        * When congestion nears the bandwidth ceiling, raises
          the effective routing threshold so only longer
          requests are offloaded.

    Long-term:
        * Periodically rebalances the PD cluster by converting
          nodes between prefill and decode roles to restore
          the optimality conditions of Equations (7) and (8),
          then re-optimizes threshold ``t``.

    Attributes:
        state: Mutable scheduler state.
        congestion_threshold: Egress utilization fraction that
            triggers short-term threshold raising.
    """

    def __init__(
        self,
        state: SchedulerState,
        congestion_threshold: float = DEFAULT_CONGESTION_THRESHOLD,
    ):
        """Initialize the scheduler.

        Args:
            state: Initial scheduler state.
            congestion_threshold: Egress utilization fraction
                that triggers short-term threshold raising.
        """
        self.state = state
        self.congestion_threshold = congestion_threshold

    def short_term_adjust(
        self,
        current_queue_depth: int,
        max_queue_depth: int = 50,
    ) -> int:
        """Return the effective threshold for the next routing decision.

        Args:
            current_queue_depth: Current number of queued
                requests at Membrane.
            max_queue_depth: Queue depth considered critical.

        Returns:
            int: Effective threshold in tokens, after the
            short-term adjustment. Also written back to
            ``state.effective_threshold``.
        """
        util = self.state.monitor.average()
        queue_ratio = (
            current_queue_depth / max_queue_depth if max_queue_depth > 0 else 0.0
        )

        if util >= self.congestion_threshold or queue_ratio >= 1.0:
            # Raise effective threshold to reduce bandwidth
            # pressure. The result is clamped so it never
            # exceeds MAX_EFFECTIVE_THRESHOLD_TOKENS.
            new_threshold = int(
                self.state.effective_threshold * THRESHOLD_RAISE_MULTIPLIER
            )
            self.state.effective_threshold = min(
                new_threshold, MAX_EFFECTIVE_THRESHOLD_TOKENS
            )
            return self.state.effective_threshold

        # Otherwise, gradually relax back toward the base
        # threshold. The relaxation never goes below the base.
        if self.state.effective_threshold > self.state.threshold:
            self.state.effective_threshold = int(
                self.state.effective_threshold * THRESHOLD_RELAX_MULTIPLIER
            )
            self.state.effective_threshold = max(
                self.state.effective_threshold, self.state.threshold
            )
        return self.state.effective_threshold

    def long_term_reoptimize(
        self,
        lengths: List[int],
        total_pd_instances: int,
    ) -> None:
        """Re-run the grid search and update threshold and ``N_p`` / ``N_d``.

        Args:
            lengths: Recent workload lengths.
            total_pd_instances: Total PD instances available.
        """
        best_t, best_n_p, best_n_d, _unused = optimizer.search(lengths, total_pd_instances)
        self.state.threshold = best_t
        self.state.effective_threshold = best_t
        self.state.num_pd_p = best_n_p
        self.state.num_pd_d = best_n_d
