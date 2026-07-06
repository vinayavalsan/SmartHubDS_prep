"""Tests for the Tier-2 config store (SQLite-backed, no Postgres needed)."""

import pytest

from smarthub.core.config_store import ConfigError, ConfigStore, REGISTRY_BY_KEY


def _store(tmp_path):
    return ConfigStore(f"sqlite:///{tmp_path / 'config.db'}")


def test_get_returns_registry_default(tmp_path):
    store = _store(tmp_path)
    assert store.get("target_cm") == 0.25
    assert store.get("recency_window_days") == 21


def test_set_then_get_typed(tmp_path):
    store = _store(tmp_path)
    store.set("target_cm", "0.4", updated_by="nimesh")   # string coerces to float
    assert store.get("target_cm") == 0.4
    store.set("recency_window_days", 30)
    assert store.get("recency_window_days") == 30 and isinstance(
        store.get("recency_window_days"), int
    )


def test_validation_rejects_out_of_range(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.set("target_cm", 1.5)          # > max 1.0
    with pytest.raises(ConfigError):
        store.set("recency_window_days", 0)  # < min 1


def test_validation_rejects_bad_choice(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.set("holiday_calendar", "IN")  # not in ('US','NONE')


def test_unknown_key_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.get("does_not_exist")
    with pytest.raises(ConfigError):
        store.set("does_not_exist", 1)


def test_env_scoping_is_independent(tmp_path):
    store = _store(tmp_path)
    store.set("target_cm", 0.3, env="staging")
    store.set("target_cm", 0.5, env="prod")
    assert store.get("target_cm", env="staging") == 0.3
    assert store.get("target_cm", env="prod") == 0.5


def test_history_is_appended(tmp_path):
    store = _store(tmp_path)
    store.set("target_cm", 0.2)
    store.set("target_cm", 0.3)
    store.set("target_cm", 0.4)
    from sqlalchemy import func, select

    from smarthub.core.config_store import history_table

    with store.engine.begin() as conn:
        count = conn.execute(
            select(func.count()).select_from(history_table)
        ).scalar()
    assert count == 3  # every write recorded


def test_resolved_lists_all_params_with_metadata(tmp_path):
    store = _store(tmp_path)
    store.set("target_cm", 0.33, updated_by="vinaya")
    resolved = {r["key"]: r for r in store.resolved()}
    assert set(resolved) == set(REGISTRY_BY_KEY)          # all params present
    assert resolved["target_cm"]["overridden"] is True
    assert resolved["target_cm"]["value"] == 0.33
    assert resolved["target_cm"]["updated_by"] == "vinaya"
    assert resolved["bid_floor"]["overridden"] is False   # still default
    assert resolved["bid_floor"]["value"] == 0.0
