"""Tests for the Tier-2 business config store (SQLite-backed, no Postgres)."""

import pytest

from smarthub.core.config_store import (
    ConfigError,
    ConfigParam,
    ConfigStore,
    REGISTRY_BY_KEY,
)


def _store(tmp_path):
    """Return a ConfigStore backed by a temp SQLite database."""
    return ConfigStore(f"sqlite:///{tmp_path / 'config.db'}")


def test_get_returns_registry_default(tmp_path):
    """get returns the registry default when no override is stored."""
    store = _store(tmp_path)
    assert store.get("target_cm") == 0.25
    assert store.get("bid_max_cap") == 100.0


def test_set_then_get_typed(tmp_path):
    """set coerces string values to the param's typed default (float)."""
    store = _store(tmp_path)
    store.set("target_cm", "0.4", updated_by="nimesh")   # string coerces to float
    assert store.get("target_cm") == 0.4
    store.set("bid_max_cap", "30")                        # coerces to float
    assert store.get("bid_max_cap") == 30.0
    assert isinstance(store.get("bid_max_cap"), float)


def test_validation_rejects_out_of_range(tmp_path):
    """set raises ConfigError when a value falls outside the allowed range."""
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.set("target_cm", 1.5)     # > max 1.0
    with pytest.raises(ConfigError):
        store.set("bid_floor", -1)      # < min 0.0


def test_configparam_cast_int_and_choices():
    """ConfigParam.cast handles int coercion and choice validation."""
    p_int = ConfigParam("n", "int", 1, "", minimum=1, maximum=10)
    assert p_int.cast("5") == 5 and isinstance(p_int.cast("5"), int)
    with pytest.raises(ConfigError):
        p_int.cast(0)   # below min
    p_choice = ConfigParam("c", "str", "US", "", choices=("US", "NONE"))
    assert p_choice.cast("NONE") == "NONE"
    with pytest.raises(ConfigError):
        p_choice.cast("IN")


def test_unknown_key_raises(tmp_path):
    """get/set raise ConfigError for a key not in the registry."""
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.get("does_not_exist")
    with pytest.raises(ConfigError):
        store.set("does_not_exist", 1)


def test_registry_is_business_only(tmp_path):
    """Registry holds only business settings, never task knobs."""
    for task_key in ("model_type", "recency_window_days", "active_model_version",
                     "holiday_calendar", "exploration_variance_pct"):
        assert task_key not in REGISTRY_BY_KEY
    assert set(REGISTRY_BY_KEY) == {
        "target_cm", "bid_floor", "bid_max_cap", "min_source_quality"
    }


def test_env_scoping_is_independent(tmp_path):
    """Values set per environment are stored and read independently."""
    store = _store(tmp_path)
    store.set("target_cm", 0.3, env="staging")
    store.set("target_cm", 0.5, env="prod")
    assert store.get("target_cm", env="staging") == 0.3
    assert store.get("target_cm", env="prod") == 0.5


def test_history_is_appended(tmp_path):
    """Every set writes an audit row to the history table."""
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
    """resolved lists every param with its value and override metadata."""
    store = _store(tmp_path)
    store.set("target_cm", 0.33, updated_by="vinaya")
    resolved = {r["key"]: r for r in store.resolved()}
    assert set(resolved) == set(REGISTRY_BY_KEY)          # all params present
    assert resolved["target_cm"]["overridden"] is True
    assert resolved["target_cm"]["value"] == 0.33
    assert resolved["target_cm"]["updated_by"] == "vinaya"
    assert resolved["bid_floor"]["overridden"] is False   # still default
    assert resolved["bid_floor"]["value"] == 0.0
