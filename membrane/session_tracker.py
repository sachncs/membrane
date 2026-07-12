"""SessionTracker: per-session context history.

This module provides :class:`SessionTracker` and its supporting
:class:`Session` dataclass. The tracker keeps an in-memory history of
which content hashes each active session has accessed in temporal
order, and exposes it to other components for prediction and value
estimation.

Typical consumers:

* :class:`~membrane.predictor.Predictor` uses session histories to
  forecast the next likely prefix for a given session.
* :class:`~membrane.value_density.ValueDensity` reads recent
  accesses when computing the value of a candidate fragment.
* The :class:`~membrane.co_access_index.CoAccessIndex` ingests
  session-level access patterns to learn which fragments are
  frequently accessed together.

Thread safety:
    The current implementation is *not* thread-safe. Callers running
    in multi-threaded contexts should serialize access via an
    external lock or replace ``self.sessions`` with a
    thread-safe mapping. The classes here are mutable by design so
    that callers can manage their own concurrency primitives.

Limitations:
    * All state is held in memory; no persistence to the canonical
      store or to disk is performed. Restarting the process discards
      the history.
"""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field


@dataclass
class Session:
    """A single session's access history.

    Attributes:
        session_id: Unique identifier for the session. Typically
            derived from the upstream chat/orchestration layer.
        access_history: Ordered list of ``content_hash`` values
            accessed during the session, oldest first. The list is
            mutated in place by :meth:`SessionTracker.record_access`;
            callers should not rely on it being immutable.
    """

    session_id: str
    access_history: list[str] = field(default_factory=list)


class SessionTracker:
    """Tracks per-session context history for prediction.

    The tracker is a thin wrapper around a ``dict[session_id,
    Session]``. It is intentionally minimal so it can be replaced or
    wrapped without touching its callers.

    Attributes:
        sessions: Mapping from ``session_id`` to the corresponding
            :class:`Session`. Mutations should go through the
            tracker methods to preserve invariants.
    """

    def __init__(self) -> None:
        """Initialize an empty session tracker.

        The internal ``sessions`` dict starts empty; sessions are
        created lazily by :meth:`record_access`.
        """
        self.sessions: dict[str, Session] = {}

    def record_access(self, session_id: str, content_hash: str) -> None:
        """Record that ``session_id`` accessed ``content_hash``.

        Creates the session on first access. Accesses are appended
        to the tail of the session's history; no deduplication is
        performed — repeated accesses will produce repeated entries,
        which is the desired input for downstream components such as
        :class:`~membrane.value_density.ValueDensity` that compute
        frequency-weighted scores.

        Args:
            session_id: Identifier of the accessing session.
            content_hash: Hash of the fragment that was accessed.
        """
        session = self.sessions.setdefault(session_id, Session(session_id=session_id))
        session.access_history.append(content_hash)

    def get_session_history(self, session_id: str) -> list[str]:
        """Return the access history for a session.

        A defensive copy is returned so that callers cannot mutate
        the tracker's internal state through the returned list.

        Args:
            session_id: Identifier of the session to query.

        Returns:
            list[str]: Ordered list of ``content_hash`` values
            accessed by the session. Returns an empty list for
            unknown sessions.
        """
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return list(session.access_history)

    def get_unique_accesses(self, session_id: str) -> set[str]:
        """Return unique content hashes accessed in a session.

        Implemented in terms of :meth:`get_session_history` so the
        two views cannot diverge.

        Args:
            session_id: Identifier of the session to query.

        Returns:
            set[str]: Set of unique hashes accessed during the
            session. Empty for unknown sessions.
        """
        return set(self.get_session_history(session_id))
