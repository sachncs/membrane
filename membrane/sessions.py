"""Sessions: per-session context history.

This module provides :class:`Sessions` and its supporting
:class:`Session` dataclass. The tracker keeps an in-memory history of
which content hashes each active session has accessed in temporal
order, and exposes it to other components for prediction and value
estimation.

Typical consumers:

* :class:`~membrane.predictor.Predict` uses session histories to
  forecast the next likely prefix for a given session.
* :class:`~membrane.value_density.density` reads recent
  accesses when computing the value of a candidate fragment.
* The :class:`~membrane.coaccess.Coaccess` ingests
  session-level access patterns to learn which fragments are
  frequently accessed together.

Thread safety:
    :class:`Sessions` serializes its own updates with a lock, so the
    HTTP handlers of a running server share one tracker.

Limitations:
    * All state is held in memory; no persistence to the canonical
      store or to disk is performed. Restarting the process discards
      the history.
    * The tracker is bounded: at most ``max_sessions`` sessions (the
      least recently active are dropped) of at most ``max_history``
      accesses each (the oldest are dropped).
"""

import logging

logger = logging.getLogger(__name__)


import threading
from collections import OrderedDict
from dataclasses import dataclass, field

MAX_SESSIONS = 10_000
MAX_HISTORY = 1_000


@dataclass
class Session:
    """A single session's access history.

    Attributes:
        session_id: Unique identifier for the session. Typically
            derived from the upstream chat/orchestration layer.
        access_history: Ordered list of ``content_hash`` values
            accessed during the session, oldest first. The list is
            mutated in place by :meth:`Sessions.record_access`;
            callers should not rely on it being immutable.
    """

    session_id: str
    access_history: list[str] = field(default_factory=list)


class Sessions:
    """Tracks per-session context history for prediction.

    The tracker is a thin wrapper around a ``dict[session_id,
    Session]``. It is intentionally minimal so it can be replaced or
    wrapped without touching its callers.

    Attributes:
        sessions: Mapping from ``session_id`` to the corresponding
            :class:`Session`. Mutations should go through the
            tracker methods to preserve invariants.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS, max_history: int = MAX_HISTORY) -> None:
        """Initialize an empty session tracker.

        The internal ``sessions`` dict starts empty; sessions are
        created lazily by :meth:`record_access`.

        Args:
            max_sessions: Sessions kept; the least recently active go first.
            max_history: Accesses kept per session; the oldest go first.
        """
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.max_sessions = max_sessions
        self.max_history = max_history
        self.__lock = threading.Lock()

    def record_access(self, session_id: str, content_hash: str) -> None:
        """Record that ``session_id`` accessed ``content_hash``.

        Creates the session on first access. Accesses are appended
        to the tail of the session's history; no deduplication is
        performed — repeated accesses will produce repeated entries,
        which is the desired input for downstream components such as
        :class:`~membrane.value_density.density` that compute
        frequency-weighted scores.

        Args:
            session_id: Identifier of the accessing session.
            content_hash: Hash of the fragment that was accessed.
        """
        with self.__lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = self.sessions[session_id] = Session(session_id=session_id)
                while len(self.sessions) > self.max_sessions:
                    self.sessions.popitem(last=False)
            else:
                self.sessions.move_to_end(session_id)
            session.access_history.append(content_hash)
            if len(session.access_history) > self.max_history:
                del session.access_history[: len(session.access_history) - self.max_history]

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
        with self.__lock:
            session = self.sessions.get(session_id)
            return [] if session is None else list(session.access_history)

    def forget(self, session_id: str) -> bool:
        """Drop a session's history.

        Args:
            session_id: Identifier of the session.

        Returns:
            bool: True when the session existed.
        """
        with self.__lock:
            return self.sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        """Number of tracked sessions.

        Returns:
            int: Session count.
        """
        return len(self.sessions)

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


__all__ = [
    "MAX_HISTORY",
    "MAX_SESSIONS",
    "Session",
    "Sessions",
]
