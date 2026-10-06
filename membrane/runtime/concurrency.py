"""Helpers for running on many cores, especially on a free-threaded Python.

On a free-threaded (PEP 703) build, every request thread that reaches a
long-lived object (the node, its fragment table, the content store) adds
and drops a reference to it. All threads then update one shared reference
count, and that contention, not the work, caps throughput: four threads
reading through ``node.table`` run no faster than one. :func:`share`
switches such objects to deferred reference counting
(``PyUnstable_Object_EnableDeferredRefcount``, CPython 3.14), after which
reads scale with cores. The objects are then reclaimed only by the
cyclic garbage collector, which suits objects that live as long as the
server.
"""

import ctypes
import logging
import sys
import sysconfig

logger = logging.getLogger(__name__)

#: Whether this interpreter was built without the GIL (it may still be
#: re-enabled at run time; see :func:`gil_enabled`).
FREE_THREADED: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_enabled() -> bool:
    """Whether the GIL is enabled right now.

    A free-threaded build turns the GIL back on when it imports an
    extension that has not declared free-threading support.

    Returns:
        bool: True on a GIL build, or a free-threaded build that re-enabled it.
    """
    return bool(getattr(sys, "_is_gil_enabled", lambda: True)())


def deferred_refcount_api() -> object | None:
    """Look up ``PyUnstable_Object_EnableDeferredRefcount``.

    Returns:
        object | None: The C function, or ``None`` when unavailable.
    """
    try:
        function = ctypes.pythonapi.PyUnstable_Object_EnableDeferredRefcount
    except AttributeError:
        return None
    function.argtypes = [ctypes.py_object]
    function.restype = ctypes.c_int
    return function


ENABLE_DEFERRED = deferred_refcount_api() if FREE_THREADED else None


def share(*objects: object) -> int:
    """Mark long-lived objects that every request thread reads.

    A no-op on GIL builds.

    Args:
        *objects: Objects that live as long as the server (``None`` is skipped).

    Returns:
        int: How many objects switched to deferred reference counting.
    """
    if ENABLE_DEFERRED is None:
        return 0
    switched = 0
    for obj in objects:
        if obj is not None:
            switched += int(ENABLE_DEFERRED(obj))  # type: ignore[operator]
    return switched


__all__ = ["ENABLE_DEFERRED", "FREE_THREADED", "deferred_refcount_api", "gil_enabled", "share"]
