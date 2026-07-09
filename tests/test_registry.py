"""Tests for the versioned model registry + promotion gate.

Pure logic + filesystem — no sklearn/joblib model actually needs to be a real
estimator here, any picklable object stands in for "the model".
"""

import pytest

from smarthub.train_and_predict import registry


@pytest.fixture(autouse=True)
def _isolated_model_dir(tmp_path, monkeypatch):
    """Redirect the registry at a temp dir, same pattern as test_io.py."""
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")


def _save(lead_type_name="auto", roc_auc=0.70, profit=100.0, feature_cols=None):
    manifest = registry.save_version(
        {"fake": "model"},
        lead_type_name,
        feature_cols=feature_cols or ["bid", "state"],
        metrics={"roc_auc": roc_auc},
        optimizer_summary={"recommended_bid_total_expected_profit": profit},
        lineage={"model_type": "lightgbm"},
        model_params={"n_estimators": 10},
    )
    return manifest


def test_load_currently_serving_resolves_path_in_current_env():
    # A manifest saved in another environment records an absolute model_path
    # (e.g. /app/... from a Docker run). Loading must resolve via version_path
    # in THIS environment, not trust that stale absolute string.
    import json

    m = _save()
    registry.promote("auto", m["version"])
    mf = registry.manifest_path("auto", m["version"])
    data = json.loads(mf.read_text())
    data["model_path"] = "/app/data/models/auto/does-not-exist.pkl"
    mf.write_text(json.dumps(data))

    model, manifest = registry.load_currently_serving_model("auto")
    assert model == {"fake": "model"}          # loaded via version_path
    assert manifest["version"] == m["version"]


def test_load_currently_serving_none_when_file_missing():
    m = _save()
    registry.promote("auto", m["version"])
    registry.version_path("auto", m["version"]).unlink()   # pkl gone
    # Graceful bootstrap, not a crash.
    assert registry.load_currently_serving_model("auto") == (None, None)


# --- Versioning ---------------------------------------------------------------


def test_versions_are_numbered_and_timestamped_and_never_overwritten():
    m1 = _save()
    m2 = _save()
    m3 = _save()

    assert m1["version"].startswith("v1_")
    assert m2["version"].startswith("v2_")
    assert m3["version"].startswith("v3_")
    assert registry.list_versions("auto") == [
        m1["version"], m2["version"], m3["version"],
    ]
    # all three pkl files exist independently -- nothing was overwritten
    for m in (m1, m2, m3):
        assert registry.version_path("auto", m["version"]).exists()
        assert registry.manifest_path("auto", m["version"]).exists()


def test_version_numbering_is_per_lead_type():
    auto1 = _save("auto")
    home1 = _save("home")
    auto2 = _save("auto")

    assert auto1["version"].startswith("v1_")
    assert home1["version"].startswith("v1_")  # independent counter
    assert auto2["version"].startswith("v2_")


def test_nothing_currently_serving_before_any_promotion():
    _save()
    assert registry.currently_serving_version("auto") is None
    assert registry.currently_serving_model_path("auto") is None
    model, manifest = registry.load_currently_serving_model("auto")
    assert model is None and manifest is None


# --- Promotion / rollback ------------------------------------------------------


def test_promote_sets_serving_pointer():
    m1 = _save()
    registry.promote("auto", m1["version"], reason="first model")

    assert registry.currently_serving_version("auto") == m1["version"]
    model, manifest = registry.load_currently_serving_model("auto")
    assert model == {"fake": "model"}
    assert manifest["promoted"] is True
    assert manifest["promotion_reason"] == "first model"


def test_promote_unknown_version_raises():
    with pytest.raises(FileNotFoundError):
        registry.promote("auto", "v99_doesnotexist", reason="x")


def test_rollback_to_previous_version():
    m1 = _save()
    m2 = _save()
    registry.promote("auto", m1["version"])
    registry.promote("auto", m2["version"])
    assert registry.currently_serving_version("auto") == m2["version"]

    registry.rollback("auto")
    assert registry.currently_serving_version("auto") == m1["version"]


def test_rollback_to_explicit_version():
    m1 = _save()
    _save()
    m3 = _save()
    registry.promote("auto", m3["version"])

    registry.rollback("auto", to_version=m1["version"], reason="bad m3")
    assert registry.currently_serving_version("auto") == m1["version"]


def test_rollback_at_earliest_version_raises():
    m1 = _save()
    registry.promote("auto", m1["version"])
    with pytest.raises(ValueError, match="nothing to roll back to"):
        registry.rollback("auto")


def test_rollback_without_anything_serving_raises():
    _save()
    with pytest.raises(ValueError, match="No currently-serving model"):
        registry.rollback("auto")


# --- Promotion policy (decide_promotion) --------------------------------------


def test_decide_promotion_bootstraps_with_nothing_currently_serving():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.5},
        challenger_optimizer={},
        currently_serving_metrics=None,
        currently_serving_optimizer=None,
    )
    assert decision.promote is True
    assert "Nothing currently serving" in decision.reason


def test_decide_promotion_blocks_on_roc_auc_regression():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.60},
        challenger_optimizer={"recommended_bid_total_expected_profit": 1000},
        currently_serving_metrics={"roc_auc": 0.70},
        currently_serving_optimizer={"recommended_bid_total_expected_profit": 500},
        min_roc_auc_regression=0.01,
    )
    # AUC dropped by 0.10, way past the 0.01 tolerance -- profit doesn't matter
    assert decision.promote is False
    assert "ROC AUC regressed" in decision.reason


def test_decide_promotion_blocks_on_profit_regression():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.70},
        challenger_optimizer={"recommended_bid_total_expected_profit": 80},
        currently_serving_metrics={"roc_auc": 0.70},
        currently_serving_optimizer={"recommended_bid_total_expected_profit": 100},
        min_profit_ratio=0.98,
    )
    # 80/100 = 80% of the currently-serving model's profit, below the 98% floor
    assert decision.promote is False
    assert "expected profit" in decision.reason
    assert decision.comparison["profit_ratio"] == pytest.approx(0.8)


def test_decide_promotion_allows_small_profit_dip_within_tolerance():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.70},
        challenger_optimizer={"recommended_bid_total_expected_profit": 99},
        currently_serving_metrics={"roc_auc": 0.70},
        currently_serving_optimizer={"recommended_bid_total_expected_profit": 100},
        min_profit_ratio=0.98,
    )
    # 99/100 = 99% >= 98% floor
    assert decision.promote is True


def test_decide_promotion_promotes_on_clear_improvement():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.75},
        challenger_optimizer={"recommended_bid_total_expected_profit": 150},
        currently_serving_metrics={"roc_auc": 0.70},
        currently_serving_optimizer={"recommended_bid_total_expected_profit": 100},
    )
    assert decision.promote is True
    assert decision.comparison["profit_ratio"] == pytest.approx(1.5)


def test_decide_promotion_falls_back_to_auc_when_no_optimizer_data():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.70},
        challenger_optimizer={},
        currently_serving_metrics={"roc_auc": 0.69},
        currently_serving_optimizer={},
        min_roc_auc_regression=0.01,
    )
    assert decision.promote is True
    assert "no optimizer comparison available" in decision.reason


def test_decide_promotion_unprofitable_serving_model_any_nonnegative_challenger_ok():
    decision = registry.decide_promotion(
        challenger_metrics={"roc_auc": 0.70},
        challenger_optimizer={"recommended_bid_total_expected_profit": 0},
        currently_serving_metrics={"roc_auc": 0.70},
        currently_serving_optimizer={"recommended_bid_total_expected_profit": -10},
    )
    assert decision.promote is True
