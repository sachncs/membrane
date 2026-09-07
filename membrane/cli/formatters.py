"""Formatters for CLI output.

Single source of truth for byte-size and duration formatting. Both
the dashboard rendering layer and the static config command use
these helpers so the output is consistent everywhere.
"""

from __future__ import annotations


def fmt_bytes(n: int) -> str:
    """Format an integer byte count using human-readable units.

    Args:
        n: Byte count.

    Returns:
        str: ``"{value:.1f} {unit}"`` where ``unit`` is one of
        ``B``, ``KiB``, ``MiB``, ``GiB``, ``TiB``, ``PiB``.
    """
    size = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``Ns`` / ``Nm`` / ``Nh``.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Compact duration string.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


__all__ = ["fmt_bytes", "fmt_duration"]
