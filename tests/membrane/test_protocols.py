"""Tests that concrete classes satisfy their declared protocols."""

import pytest

from membrane.canonical_store import CanonicalStore
from membrane.economic_router import EconomicRouter
from membrane.global_directory import GlobalDirectory
from membrane.index_system import IndexSystem
from membrane.membrane_node import MembraneNode
from membrane.protocols import DirectoryProtocol, IndexProtocol, RouterProtocol, TransportProtocol
from membrane.transfer_service import TransferService


def test_index_system_satisfies_index_protocol():
    idx = IndexSystem()
    assert isinstance(idx, IndexProtocol)


def test_economic_router_satisfies_router_protocol():
    router = EconomicRouter()
    assert isinstance(router, RouterProtocol)


def test_global_directory_satisfies_directory_protocol():
    gd = GlobalDirectory()
    assert isinstance(gd, DirectoryProtocol)


def test_transfer_service_satisfies_transport_protocol():
    ts = TransferService()
    assert isinstance(ts, TransportProtocol)


def test_canonical_store_does_not_satisfy_index_protocol():
    cs = CanonicalStore()
    assert not isinstance(cs, IndexProtocol)
