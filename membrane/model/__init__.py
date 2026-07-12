"""Membrane analytical model and simulator.

This subpackage reproduces the throughput model, scheduling
policies, and case-study evaluation from the paper:

    "Prefill-as-a-Service: KVCache of Next-Generation Models
    Could Go Cross-Datacenter" (arXiv:2604.15039v2).

Modules:

* :mod:`membrane.model.throughput_model` — Equations (1)–(6)
  from the paper.
* :mod:`membrane.model.profiler` — KV-size and prefill-time
  estimators built on top of the throughput model.
* :mod:`membrane.model.workload` — Log-normal workload
  generator.
* :mod:`membrane.model.router` — Throughput-optimal routing
  policy.
* :mod:`membrane.model.scheduler` — Dual-timescale scheduling.
* :mod:`membrane.model.optimizer` — Grid-search optimizer over
  routing threshold and PD split.
* :mod:`membrane.model.simulator` — End-to-end simulation that
  combines all of the above.
* :mod:`membrane.model.metrics` — Aggregated simulation metrics.

Each module is independently testable and can be reused outside
the simulator.
"""

import logging

logger = logging.getLogger(__name__)
