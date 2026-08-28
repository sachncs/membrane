"""Public type façade for Membrane.

The 40+ deep imports of ``Fragment`` (and similar for ``MembraneNode``,
``StructuralSignature``, etc.) scattered through the codebase made every rename
painful. This module re-exports the core data model from a single, stable
location so internal layout changes do not ripple into every consumer.

Consumers should prefer::

    from membrane.types import Fragment, Node, Signature, Store, ...

over::

    from membrane.fragment import Fragment
"""

from __future__ import annotations

from membrane.fragment import Fragment
from membrane.kv_segment import KVSegment
from membrane.network.config import ClusterConfig
from membrane.network.gossip_state import GossipState, PeerEndpoint
from membrane.structural_signature import StructuralSignature
from membrane.transfer_service import TransferService

__all__ = [
    "ClusterConfig",
    "Fragment",
    "GossipState",
    "KVSegment",
    "PeerEndpoint",
    "Signature",
    "StructuralSignature",
    "TransferService",
]


Signature = StructuralSignature
"""Alias for ``StructuralSignature`` so call sites read ``membrane.types.Signature``."""
