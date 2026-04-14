from __future__ import annotations

from datetime import datetime, timezone

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.state.in_memory import InMemoryStateStore


def _sample_session(session_id: str) -> BackendSession:
    now = datetime.now(timezone.utc)
    return BackendSession(
        session_id=session_id,
        backend_name="mock_online",
        model_name=None,
        status="created",
        created_at=now,
        updated_at=now,
        metadata={},
    )


def _sample_state(session_id: str) -> BackendState:
    return BackendState(
        session_id=session_id,
        last_update_time=datetime.now(timezone.utc),
        observation_count=0,
        state_version=0,
        internal_state={},
        recent_observations=[],
    )


def test_in_memory_create_and_get_session():
    store = InMemoryStateStore()
    session = _sample_session("s1")
    state = _sample_state("s1")

    store.create_session(session, state)

    assert store.get_session("s1") is not None


def test_in_memory_save_and_get_state():
    store = InMemoryStateStore()
    session = _sample_session("s1")
    state = _sample_state("s1")
    store.create_session(session, state)

    state.state_version = 2
    store.save_state("s1", state)

    assert store.get_state("s1").state_version == 2


def test_in_memory_save_session_updates_record():
    store = InMemoryStateStore()
    session = _sample_session("s1")
    store.create_session(session, _sample_state("s1"))

    session.status = "updated"
    store.save_session(session)

    assert store.get_session("s1").status == "updated"


def test_in_memory_list_sessions():
    store = InMemoryStateStore()
    store.create_session(_sample_session("s1"), _sample_state("s1"))
    store.create_session(_sample_session("s2"), _sample_state("s2"))

    sessions = store.list_sessions()

    assert len(sessions) == 2


def test_in_memory_delete_session():
    store = InMemoryStateStore()
    store.create_session(_sample_session("s1"), _sample_state("s1"))

    store.delete_session("s1")

    assert store.get_session("s1") is None
    assert store.get_state("s1") is None
