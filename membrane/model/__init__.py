"""Membrane analytical model and simulator.

This subpackage reproduces the throughput model, scheduling policies, and
case-study evaluation from the paper:

    "Prefill-as-a-Service: KVCache of Next-Generation Models Could Go
     Cross-Datacenter" (arXiv:2604.15039v2).
"""

import logging

logger = logging.getLogger(__name__)
