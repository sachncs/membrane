"""Tests for session_tracker module."""

import pytest

from membrane.sessions import Session, Sessions


class TestSessionTracker:
    """Test suite for Sessions."""

    def test_record_access(self):
        st = Sessions()
        st.record_access("s1", "h1")
        assert st.get_session_history("s1") == ["h1"]

    def test_multiple_accesses_ordered(self):
        st = Sessions()
        st.record_access("s1", "h1")
        st.record_access("s1", "h2")
        assert st.get_session_history("s1") == ["h1", "h2"]

    def test_unknown_session_returns_empty(self):
        st = Sessions()
        assert st.get_session_history("unknown") == []

    def test_unique_accesses(self):
        st = Sessions()
        st.record_access("s1", "h1")
        st.record_access("s1", "h1")
        st.record_access("s1", "h2")
        assert st.get_unique_accesses("s1") == {"h1", "h2"}

    def test_session_object_created_lazily(self):
        st = Sessions()
        assert "s1" not in st.sessions
        st.record_access("s1", "h1")
        assert "s1" in st.sessions
        assert isinstance(st.sessions["s1"], Session)
