"""Building blocks of a node's fragment storage.

* :mod:`membrane.store.table`: :class:`FragmentTable`, resident fragments
  and their bookkeeping.
* :mod:`membrane.store.eviction`: eviction policies (:class:`WeightedLRU`,
  :class:`FrequencyLRU`).
* :mod:`membrane.store.tenant_guard`: per-tenant read and write checks.

:class:`membrane.node.Node` composes them.
"""

__all__: list[str] = []
