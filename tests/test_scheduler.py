"""Tests for the dual-timescale scheduler."""

import math

from membrane.model import scheduler


def test_monitor_window():
    """Monitor should keep only the last window_size samples."""
    mon = scheduler.EgressMonitor(window_size=3)
    mon.record(0.5)
    mon.record(0.6)
    mon.record(0.7)
    mon.record(0.8)
    assert math.isclose(mon.average(), (0.6 + 0.7 + 0.8) / 3)


def test_short_term_raises_threshold_on_congestion():
    """When utilization is high, effective threshold should increase."""
    state = scheduler.SchedulerState(threshold=10000, num_pd_p=3, num_pd_d=5)
    state.monitor.record(0.9)
    sched = scheduler.DualTimescaleScheduler(state, congestion_threshold=0.85)
    new_t = sched.short_term_adjust(current_queue_depth=10, max_queue_depth=50)
    assert new_t > 10000


def test_short_term_relaxes_threshold_when_clear():
    """When utilization is low, effective threshold should move toward base."""
    state = scheduler.SchedulerState(threshold=10000, num_pd_p=3, num_pd_d=5, effective_threshold=12000)
    state.monitor.record(0.1)
    sched = scheduler.DualTimescaleScheduler(state, congestion_threshold=0.85)
    new_t = sched.short_term_adjust(current_queue_depth=0, max_queue_depth=50)
    assert new_t <= 12000
    assert new_t >= 10000


def test_long_term_changes_state():
    """Long-term reoptimization should update threshold and instance counts."""
    from membrane.model import workload

    lengths = workload.generate_request_lengths(5000, seed=7)
    state = scheduler.SchedulerState(threshold=5000, num_pd_p=2, num_pd_d=6)
    sched = scheduler.DualTimescaleScheduler(state)
    sched.long_term_reoptimize(lengths, total_pd_instances=8)
    assert sched.state.threshold != 5000 or sched.state.num_pd_p != 2
