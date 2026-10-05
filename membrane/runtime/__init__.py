"""Building blocks of a running Membrane server.

* :mod:`membrane.runtime.plugins`: named registries for compute
  backends, authenticators, and content stores, extendable through
  entry points.
* :mod:`membrane.runtime.components`: builders for persistence, the
  content store, authentication, and peer access.
* :mod:`membrane.runtime.persistence_writer`: ordered, bounded
  write-behind persistence.
* :mod:`membrane.runtime.lifecycle`: periodic background tasks and
  signal-driven graceful drain.
* :mod:`membrane.runtime.observability`: dashboard events and
  diagnostics.

:class:`membrane.server.Server` wires these together.
"""

__all__: list[str] = []
