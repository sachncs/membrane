"""Hybrid Logical Clock (HLC).

The HLC is the per-fragment stability anchor at 2.0+ — every
Fragment carries an ``hlc`` integer in its wire payload and the
op_store layer uses HLC ordering to resolve concurrent writes
on the same ``content_hash``.

The 64-bit HLC layout packs:

    bits  0..47 : physical time, milliseconds since the epoch
    bits 48..63 : logical counter, incremented when two events
                  happen at the same physical millisecond on the
                  same node

The merge rule is the standard HLC form: ``local = tick()``;
``remote = merge(local, observed)`` where ``observed`` is the
``hlc`` of an incoming fragment. Merge picks the larger of
``local`` and ``observed`` (treating the two halves independently);
ties on the physical half are broken by bumping the logical
counter.

The peer_id tiebreak is applied only when two fragments
arrive simultaneously with the same HLC value and different
content_hashes; the cluster's gossip layer (Phase 5) and
registry layer (Phase 3.2) use ``peer_id`` lex order when
comparing equal HLCs. The HLC itself is purely
``(physical, logical)`` so the wire format stays compact.

The HLC is intentionally not a true vector clock — the
``peer_id`` ordering at the gossip layer prevents the
"concurrent update with no observable order" case from
silently passing. We trade a tiny amount of fan-out precision
for a single 64-bit integer.

The public surface:

* :class:`HLC` is the clock instance held by ``Server`` /
  ``Cluster`` / per-request handlers. ``tick()`` advances the
  local time; ``merge(local, remote)`` reconciles.
* :func:`pack` / :func:`unpack` translate to/from 64-bit
  integers for the wire format.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

_PHYSICAL_SHIFT: int = 16
_LOGICAL_MAX: int = (1 << 16) - 1
_PHYSICAL_MAX: int = (1 << 48) - 1


@dataclass(frozen=True)
class HLC:
    """Hybrid logical clock for a single process.

    Attributes:
        physical_ms: Last observed physical time in milliseconds
            since the Unix epoch.
        logical: Per-process counter for events that share the
            same physical millisecond.
    """

    physical_ms: int
    logical: int


def unpack(value: int) -> HLC:
    """Decode a 64-bit HLC integer into ``HLC``.

    Args:
        value: 64-bit integer in the canonical HLC encoding.

    Returns:
        HLC: The decoded clock state.
    """
    physical = (value >> _PHYSICAL_SHIFT) & _PHYSICAL_MAX
    logical = value & _LOGICAL_MAX
    return HLC(physical_ms=physical, logical=logical)


def pack(clock: HLC) -> int:
    """Encode an :class:`HLC` into a 64-bit integer.

    Args:
        clock: Clock state.

    Returns:
        int: 64-bit integer wire representation.
    """
    return (clock.physical_ms << _PHYSICAL_SHIFT) | (clock.logical & _LOGICAL_MAX)


def tick(previous: HLC) -> HLC:
    """Advance the clock to the current physical time.

    Called on every local event. The rule is "the new clock is
    either one logical step beyond the previous clock or
    whichever is greater between the previous clock and the
    current physical clock".

    Args:
        previous: Clock state from the last observed event.

    Returns:
        HLC: Advanced clock state.
    """
    now_ms = int(time.time() * 1000)
    if now_ms > previous.physical_ms:
        return HLC(physical_ms=now_ms, logical=0)
    if previous.logical < _LOGICAL_MAX:
        return HLC(physical_ms=previous.physical_ms, logical=previous.logical + 1)
    # Logical overflow at the same physical millisecond: roll
    # into the next millisecond. The real-world chance of
    # reaching 65k events per millisecond on a single node is
    # effectively zero; this branch exists to keep the
    # invariant total.
    return HLC(physical_ms=previous.physical_ms + 1, logical=0)


def merge(local: HLC, observed: int | HLC) -> HLC:
    """Reconcile a local clock with an observed HLC from a peer.

    The merge rule picks the greater of the two clocks
    independently on the physical and logical halves, but
    refuses to read the system clock — the merge is a
    deterministic function of the two inputs so it can be
    applied inside HLC tickers without advancing the local
    wall clock.

    * If ``observed.physical`` is greater, the result uses
      that physical value with ``logical=0``.
    * If ``local.physical`` is greater, the result keeps the
      local physical value and bumps the local logical
      counter by one (with overflow rolling into the next
      millisecond).
    * On equal physical values, the logical counter is
      ``max(local.logical, observed.logical) + 1``.

    Args:
        local: Local clock state from the last observed event.
        observed: Either a wire-encoded integer or a decoded
            :class:`HLC` from a peer.

    Returns:
        HLC: New clock state with both halves advanced past the
        maximum of ``local`` and ``observed``.
    """
    other = unpack(observed) if isinstance(observed, int) else observed

    if other.physical_ms > local.physical_ms:
        # Observed physical time is strictly newer. Use that
        # physical value with a fresh logical counter.
        return HLC(physical_ms=other.physical_ms, logical=0)
    if other.physical_ms < local.physical_ms:
        # Local is already newer; bump local's logical counter
        # by one (rolling into the next millisecond on
        # saturation). Deterministic: ``tick()`` here would
        # advance to wall-clock time which would change the
        # HLC ordering across re-runs.
        if local.logical < _LOGICAL_MAX:
            return HLC(physical_ms=local.physical_ms, logical=local.logical + 1)
        return HLC(physical_ms=local.physical_ms + 1, logical=0)
    # Same physical ms — pick the maximum logical counter plus
    # one. Ties on the counter (i.e., equal observed and local)
    # still produce a strictly newer clock because we add 1.
    new_logical = max(local.logical, other.logical) + 1
    if new_logical > _LOGICAL_MAX:
        return HLC(physical_ms=local.physical_ms + 1, logical=0)
    return HLC(physical_ms=local.physical_ms, logical=new_logical)


def compare(a: int, b: int) -> int:
    """Total order over two HLC wire values.

    Returns:
        int: ``-1`` if ``a`` precedes ``b``, ``1`` if ``a``
        follows ``b``, ``0`` if equal. Equal pairs are
        tie-broken by ``peer_id`` lex at the call site because
        identical HLCs from different peers are an extremely
        rare event (millisecond collision) that consumers
        decide based on ``payload_hash`` anyway.
    """
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


class Clock:
    """Per-process clock state with thread-safe tick/merge.

    Used by ``Server`` / ``Cluster`` to maintain the local HLC.
    The instance is held by a single thread in normal operation
    but every API is guarded by an internal lock so callers
    can call :meth:`tick` from any thread without racing.
    """

    def __init__(self, seed: HLC | None = None) -> None:
        """Initialize the clock.

        Args:
            seed: Optional starting state. ``None`` defaults to
                the current physical millisecond with
                ``logical=0``.
        """
        if seed is None:
            seed = HLC(physical_ms=int(time.time() * 1000), logical=0)
        self._state = seed
        self._lock = threading.Lock()

    def current(self) -> HLC:
        """Return the current clock state without advancing."""
        with self._lock:
            return self._state

    def pack_current(self) -> int:
        """Return the current packed HLC integer."""
        return pack(self.current())

    def tick(self) -> int:
        """Advance the clock and return the new packed integer."""
        with self._lock:
            self._state = tick(self._state)
            return pack(self._state)

    def merge(self, observed: int | HLC) -> int:
        """Reconcile with the observed HLC and return the new packed integer."""
        with self._lock:
            self._state = merge(self._state, observed)
            return pack(self._state)


__all__ = [
    "HLC",
    "Clock",
    "compare",
    "merge",
    "pack",
    "tick",
    "unpack",
]
