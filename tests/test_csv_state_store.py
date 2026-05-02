from __future__ import annotations

from datetime import datetime, timezone

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.state.csv_store import CsvStateStore


def test_csv_state_store_round_trip(tmp_path):
    now = datetime.now(timezone.utc)
    session = BackendSession(
        session_id="s1",
        backend_name="mock_online",
        model_name="m1",
        status="created",
        created_at=now,
        updated_at=now,
        metadata={"k": [1, 2]},
        runtime_metadata={"r": {"a": True}},
    )
    state = BackendState(session_id="s1", last_update_time=now, observation_count=1, state_version=2, metadata={"s": 1})

    store = CsvStateStore(tmp_path)
    store.create_session(session, state)

    reloaded = CsvStateStore(tmp_path)
    loaded = reloaded.get_session("s1")
    loaded_state = reloaded.get_state("s1")

    assert loaded is not None
    assert loaded_state is not None
    assert loaded.metadata == {"k": [1, 2]}
    assert loaded.runtime_metadata == {"r": {"a": True}}
    assert loaded_state.state_version == 2


def test_csv_state_store_latest_forecast_linkage_persists(tmp_path):
    now = datetime.now(timezone.utc)
    store = CsvStateStore(tmp_path)
    store.create_session(
        BackendSession("s1", "mock_online", None, "created", now, now),
        BackendState(session_id="s1", last_update_time=now, observation_count=0, state_version=0),
    )
    store.save_latest_forecast_linkage("s1", "f-1", "artifacts/f-1")

    reloaded = CsvStateStore(tmp_path)
    link = reloaded.get_latest_forecast_linkage("s1")
    assert link is not None
    assert link["latest_forecast_id"] == "f-1"
