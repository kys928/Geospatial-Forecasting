from __future__ import annotations

from plume.api import deps
from plume.state.csv_store import CsvStateStore
from plume.state.in_memory import InMemoryStateStore


def test_default_state_store_is_in_memory(monkeypatch):
    monkeypatch.delenv("PLUME_STATE_STORE", raising=False)
    monkeypatch.setattr(deps, "_STATE_STORE_SINGLETON", None)
    store = deps.get_state_store()
    assert isinstance(store, InMemoryStateStore)


def test_env_can_select_csv_state_store(monkeypatch, tmp_path):
    monkeypatch.setenv("PLUME_STATE_STORE", "csv")
    monkeypatch.setenv("PLUME_SESSION_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(deps, "_STATE_STORE_SINGLETON", None)
    store = deps.get_state_store()
    assert isinstance(store, CsvStateStore)
