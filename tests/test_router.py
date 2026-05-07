"""Tests for the request router."""

from membrane.model import router


def test_short_request_routed_to_pd():
    """Requests with incremental length <= threshold go to PD-P."""
    rtr = router.Router(threshold=1000)
    decision = rtr.route(total_length=800, cached_prefix_membrane=0, cached_prefix_pd=0)
    assert decision.target == "pd-p"
    assert decision.incremental_length == 800


def test_long_request_routed_to_membrane():
    """Requests with incremental length > threshold go to Membrane."""
    rtr = router.Router(threshold=1000)
    decision = rtr.route(
        total_length=1200, cached_prefix_membrane=0, cached_prefix_pd=0
    )
    assert decision.target == "membrane"
    assert decision.incremental_length == 1200


def test_bandwidth_scarce_independent_cache():
    """When bandwidth is scarce, PD cache is checked first."""
    rtr = router.Router(threshold=1000, bandwidth_abundant=False)
    # total=2000, pd cache=1500 -> incremental=500 <= threshold -> PD-P
    decision = rtr.route(
        total_length=2000, cached_prefix_membrane=0, cached_prefix_pd=1500
    )
    assert decision.target == "pd-p"
    assert decision.incremental_length == 500
    assert not decision.cross_cluster_cache_transfer


def test_bandwidth_abundant_best_cache():
    """When bandwidth is abundant, best cache wins even if cross-cluster."""
    rtr = router.Router(threshold=1000, bandwidth_abundant=True)
    # total=2000, best cache=1800 (in Membrane) -> incremental=200 <= threshold -> PD-P
    # but cache is in Membrane, so cross-cluster transfer needed.
    decision = rtr.route(
        total_length=2000, cached_prefix_membrane=1800, cached_prefix_pd=0
    )
    assert decision.target == "pd-p"
    assert decision.cross_cluster_cache_transfer is True


def test_zero_prefix():
    """Routing with zero prefix cache falls back to total length."""
    rtr = router.Router(threshold=500)
    decision = rtr.route(total_length=300)
    assert decision.target == "pd-p"
    assert decision.incremental_length == 300
