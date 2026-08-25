"""Tests for the hardened promotion/registry workflow.

Covers the three reliability guarantees:
  * promotion is atomic — a failed production publish must not mark the model
    promoted or switch the serving pointer;
  * production-storage failures are visible, not silently masked;
  * production version numbers are collision-free (authoritative + atomic claim).
"""

import pytest

from smarthub.train_and_predict import registry
from smarthub.train_and_predict.model_storage import FilesystemModelStore


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    registry.reset_production_store_cache()
    yield
    registry.reset_production_store_cache()


def _save(lead_type_name="auto"):
    return registry.save_version(
        {"fake": "model"},
        lead_type_name,
        feature_cols=["bid", "state"],
        metrics={"roc_auc": 0.7},
        optimizer_summary={"recommended_bid_total_expected_profit": 100.0},
        lineage={"model_type": "lightgbm"},
        model_params={},
        training_config={},
        promotion_mode="manual",
        eligibility_status="eligible",
        promotion_status="awaiting_manual_promotion",
        promotion_decision_reason="test",
    )


class _FaultyStore(FilesystemModelStore):
    """Filesystem production store that raises when writing chosen keys."""

    backend = "faulty"

    def __init__(self, root, fail_keys=()):
        super().__init__(root)
        self.fail_keys = set(fail_keys)

    def write_bytes(self, key, data):
        if key in self.fail_keys:
            raise RuntimeError(f"boom writing {key}")
        return super().write_bytes(key, data)


# --- Atomic promotion ---------------------------------------------------------


def test_promote_success_is_consistent_across_all_three(tmp_path, monkeypatch):
    store = FilesystemModelStore(tmp_path / "prod")
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: store)

    v = _save()["version"]
    registry.promote("auto", v, reason="ok")

    # production storage has artifact + manifest + pointer
    assert store.exists(f"auto/{v}.pkl")
    assert store.exists(f"auto/{v}.json")
    assert registry.production_serving_pointer("auto")["training_run_id"] == v
    # local manifest marked promoted + serving pointer switched
    lm = registry.load_manifest("auto", v)
    assert lm["promoted"] is True
    assert lm["promotion_status"] == "promoted"
    assert registry.currently_serving_version("auto") == v


def test_publish_failure_does_not_promote_or_switch(tmp_path, monkeypatch):
    store = _FaultyStore(tmp_path / "prod")
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: store)

    # First model promotes cleanly and is serving.
    a = _save()["version"]
    registry.promote("auto", a, reason="A")
    assert registry.production_serving_pointer("auto")["training_run_id"] == a

    # Second model: make the production pointer write fail.
    b = _save()["version"]
    store.fail_keys = {"auto/current.json"}
    with pytest.raises(Exception):
        registry.promote("auto", b, reason="B")

    # Serving pointer still points at the previous model (A), not B.
    assert registry.production_serving_pointer("auto")["training_run_id"] == a
    # B was NOT marked promoted.
    bm = registry.load_manifest("auto", b)
    assert bm.get("promoted") is not True
    assert bm.get("promotion_status") != "promoted"


# --- Production-storage failures are visible ----------------------------------


def test_production_store_raises_when_configured_but_broken(monkeypatch):
    class _Cfg:
        production_storage = {"backend": "s3"}

        def production_model_store(self):
            raise RuntimeError("bad creds / boto3 missing")

    monkeypatch.setattr(
        "smarthub.train_and_predict.config.load_training_config",
        lambda lead_type_id: _Cfg(),
    )
    registry.reset_production_store_cache()
    with pytest.raises(registry.RegistryError):
        registry._production_store("auto")


def test_production_store_none_when_disabled(monkeypatch):
    class _Cfg:
        production_storage = None

        def production_model_store(self):  # pragma: no cover - shouldn't be called
            raise AssertionError("should not build a store when disabled")

    monkeypatch.setattr(
        "smarthub.train_and_predict.config.load_training_config",
        lambda lead_type_id: _Cfg(),
    )
    registry.reset_production_store_cache()
    assert registry._production_store("auto") is None


def test_serving_model_path_no_local_fallback_when_production_configured(
    tmp_path, monkeypatch
):
    store = FilesystemModelStore(tmp_path / "prod")
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: store)

    # Local artifact exists, but production (authoritative) does not have it.
    m = _save()
    assert registry.serving_model_path("auto", m["version"]) is None


# --- Collision-free version assignment ---------------------------------------


def test_version_not_reused_across_stores(tmp_path, monkeypatch):
    store = FilesystemModelStore(tmp_path / "prod")
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: store)

    # Pretend auto_v1.0.5 was already assigned elsewhere (only a production
    # marker); the next assignment must bump the patch off the highest seen.
    assert store.claim("auto/versions/auto_v1.0.5.json") is True

    assert registry._next_production_model_version("auto") == "auto_v1.0.6"
    # The claim above reserved v1.0.6; the next assignment must not reuse it.
    assert registry._next_production_model_version("auto") == "auto_v1.0.7"


def test_claim_is_atomic_exclusive(tmp_path):
    store = FilesystemModelStore(tmp_path / "prod")
    assert store.claim("auto/versions/auto_v1.0.0.json") is True
    assert store.claim("auto/versions/auto_v1.0.0.json") is False


def test_local_only_promotion_still_works(monkeypatch):
    """Regression: production disabled -> promote via local pointer, unchanged."""
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: None)
    v = _save()["version"]
    registry.promote("auto", v, reason="local")
    assert registry.currently_serving_version("auto") == v
    lm = registry.load_manifest("auto", v)
    assert lm["promoted"] is True
    assert lm["production_model_version"] == "auto_v1.0.0"
