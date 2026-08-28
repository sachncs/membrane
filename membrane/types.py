"""Public type façade for Membrane.

The 40+ deep imports of ``Fragment`` (and similar for ``Node``,
``Signature``, etc.) scattered through the codebase made every rename
painful. This module re-exports the core data model from a single, stable
location so internal layout changes do not ripple into every consumer.

Consumers should prefer::

    from membrane.types import Fragment, Node, Signature, Store, ...

over::

    from membrane.fragment import Fragment
"""

from __future__ import annotations

from membrane.fragment import Fragment
from membrane.network.config import ClusterConfig
from membrane.network.gossip import GossipState, PeerEndpoint
from membrane.segment import Segment
from membrane.signature import Signature
from membrane.transfer import TransferService

__all__ = [
    "ClusterConfig",
    "Fragment",
    "GossipState",
    "Segment",
    "PeerEndpoint",
    "Signature",
    "Signature",
    "TransferService",
]


Signature = Signature
"""Alias for ``Signature`` so call sites read ``membrane.types.Signature``."""
