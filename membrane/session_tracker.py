"""SessionTracker: per-session context history."""

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field


@dataclass
class Session:
    """A single session's access history.

    Attributes:
        session_id: Unique session identifier.
        access_history: Ordered list of accessed content hashes.
    """

    session_id: str
    access_history: list[str] = field(default_factory=list)


class SessionTracker:
    """Tracks per-session context history for prediction."""

    def __init__(self) -> None:
        """Initialize an empty session tracker."""
        """Initialize an empty session tracker."""
        self.sessions: dict[str, Session] = {}

    def record_access(self, session_id: str, content_hash: str) -> None:
        """Record that a session accessed a fragment.

        Args:
            session_id: Session identifier.
            content_hash: Accessed fragment hash.
        """
        session = self.sessions.setdefault(session_id, Session(session_id=session_id))
        session.access_history.append(content_hash)

    def get_session_history(self, session_id: str) -> list[str]:
        """Return the access history for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Ordered list of accessed content hashes.
        """
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return list(session.access_history)

    def get_unique_accesses(self, session_id: str) -> set[str]:
        """Return unique content hashes accessed in a session.

        Args:
            session_id: Session identifier.

        Returns:
            Set of unique accessed hashes.
        """
        return set(self.get_session_history(session_id))
