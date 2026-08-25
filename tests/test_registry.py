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
    """Save a fake model version and return its manifest."""
    manifest = registry.save_version(
        {"fake": "model"},
        lead_type_name,
        feature_cols=feature_cols or ["bid", "state"],
        metrics={"roc_auc": roc_auc},
        optimizer_summary={"recommended_bid_total_expected_profit": profit},
        lineage={"model_type": "lightgbm"},
        model_params={"n_estimators": 10},
        training_config={},
        promotion_mode="manual",
        eligibility_status="eligible",
        promotion_status="awaiting_manual_promotion",
        promotion_decision_reason="not evaluated",
    )
    return manifest


def test_load_currently_serving_resolves_path_in_current_env():
    """Serving model loads via version_path, ignoring a stale absolute path."""
    import json

    m = _save()
    registry.promote("auto", m["version"])
    mf = registry.manifest_path("auto", m["version"])
    data = json.loads(mf.read_text())
    data["model_path"] = "/app/data/models/auto/does-not-exist.pkl"
    mf.write_text(json.dumps(data))

    model, manifest = registry.load_currently_serving_model("auto")
    assert model == {"fake": "model"}  # loaded via version_path
    assert manifest["version"] == m["version"]


def test_load_currently_serving_none_when_file_missing():
    """Missing pkl yields (None, None) instead of crashing."""
    m = _save()
    registry.promote("auto", m["version"])
    registry.version_path("auto", m["version"]).unlink()  # pkl gone
    # Graceful bootstrap, not a crash.
    assert registry.load_currently_serving_model("auto") == (None, None)


# --- Production storage routing -----------------------------------------------


def test_promote_publishes_only_promoted_model_to_production(tmp_path, monkeypatch):
    """promote() copies artifact + manifest + pointer to the production store."""
    from smarthub.train_and_predict.model_storage import FilesystemModelStore

    prod = FilesystemModelStore(tmp_path / "prod")
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: prod)

    m = _save()
    version = m["version"]
    # A saved-but-unpromoted run must NOT be in production yet.
    assert not prod.exists(f"auto/{version}.pkl")

    registry.promote("auto", version, reason="test")

    assert prod.exists(f"auto/{version}.pkl")
    assert prod.exists(f"auto/{version}.json")
    pointer = registry.production_serving_pointer("auto")
    assert pointer["training_run_id"] == version
    assert pointer["production_model_version"] == "auto_v1.0.0"


def test_load_serving_model_prefers_production(tmp_path, monkeypatch):
    """load_serving_model resolves from the production store when configured."""
    from smarthub.train_and_predict.model_storage import FilesystemModelStore

    prod = FilesystemModelStore(tmp_path / "prod")
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: prod)

    m = _save()
    registry.promote("auto", m["version"])

    model, manifest = registry.load_serving_model("auto")
    assert model == {"fake": "model"}
    assert manifest["version"] == m["version"]


def test_load_serving_model_falls_back_to_local_when_no_production(monkeypatch):
    """With production disabled, serving uses local storage unchanged."""
    monkeypatch.setattr(registry, "_production_store", lambda lead_type_name: None)
    m = _save()
    registry.promote("auto", m["version"])
    model, manifest = registry.load_serving_model("auto")
    assert model == {"fake": "model"}
    assert manifest["version"] == m["version"]


# --- Versioning ---------------------------------------------------------------


def test_training_run_ids_are_unique_and_never_overwritten():
    """Every saved candidate gets a unique immutable training-run ID."""
    m1 = _save()
    m2 = _save()
    m3 = _save()

    for manifest in (m1, m2, m3):
        assert manifest["training_run_id"].startswith("run_")
        assert manifest["version"] == manifest["training_run_id"]
        assert manifest["production_model_version"] is None

    assert len({m1["version"], m2["version"], m3["version"]}) == 3
    assert registry.list_versions("auto") == [
        m1["version"],
        m2["version"],
        m3["version"],
    ]
    for manifest in (m1, m2, m3):
        assert registry.version_path("auto", manifest["version"]).exists()
        assert registry.manifest_path("auto", manifest["version"]).exists()


def test_production_version_numbering_is_per_lead_type():
    """Only promoted models consume per-lead-type production versions."""
    auto1 = _save("auto")
    home1 = _save("home")
    auto2 = _save("auto")

    pointer_auto1 = registry.promote("auto", auto1["version"])
    pointer_home1 = registry.promote("home", home1["version"])
    pointer_auto2 = registry.promote("auto", auto2["version"])

    assert pointer_auto1["production_model_version"] == "auto_v1.0.0"
    assert pointer_home1["production_model_version"] == "home_v1.0.0"
    assert pointer_auto2["production_model_version"] == "auto_v1.0.1"
    assert registry.list_production_versions("auto") == ["auto_v1.0.0", "auto_v1.0.1"]
    assert registry.list_production_versions("home") == ["home_v1.0.0"]


def test_nothing_currently_serving_before_any_promotion():
    """Nothing is serving until a version is promoted."""
    _save()
    assert registry.currently_serving_version("auto") is None
    assert registry.currently_serving_model_path("auto") is None
    model, manifest = registry.load_currently_serving_model("auto")
    assert model is None and manifest is None


# --- Promotion / rollback ------------------------------------------------------


def test_promote_sets_serving_pointer():
    """promote sets the serving pointer and records the reason."""
    m1 = _save()
    registry.promote("auto", m1["version"], reason="first model")

    assert registry.currently_serving_version("auto") == m1["version"]
    model, manifest = registry.load_currently_serving_model("auto")
    assert model == {"fake": "model"}
    assert manifest["promoted"] is True
    assert manifest["promotion_reason"] == "first model"


def test_promote_unknown_version_raises():
    """promote raises FileNotFoundError for an unknown version."""
    with pytest.raises(FileNotFoundError):
        registry.promote("auto", "v99_doesnotexist", reason="x")


def test_rollback_to_previous_version():
    """rollback reverts the serving pointer to the previous version."""
    m1 = _save()
    m2 = _save()
    registry.promote("auto", m1["version"])
    registry.promote("auto", m2["version"])
    assert registry.currently_serving_version("auto") == m2["version"]

    registry.rollback("auto")
    assert registry.currently_serving_version("auto") == m1["version"]


def test_rollback_to_explicit_version():
    """rollback can target an explicit version."""
    m1 = _save()
    _save()
    m3 = _save()
    registry.promote("auto", m3["version"])

    registry.rollback("auto", to_version=m1["version"], reason="bad m3")
    assert registry.currently_serving_version("auto") == m1["version"]


def test_rollback_at_earliest_version_raises():
    """rollback raises when there is no earlier version to revert to."""
    m1 = _save()
    registry.promote("auto", m1["version"])
    with pytest.raises(ValueError, match="earliest promoted model"):
        registry.rollback("auto")


def test_rollback_without_anything_serving_raises():
    """rollback raises when nothing is currently serving."""
    _save()
    with pytest.raises(ValueError, match="No currently-serving promoted model"):
        registry.rollback("auto")


# --- Promotion policy (decide_promotion) --------------------------------------


POLICY = {
    "max_log_loss_regression": 0.01,
    "min_profit_ratio": 0.98,
    "max_absolute_profit_loss_tolerance": 5.0,
    "max_log_loss": 0.55,
    "min_expected_profit": 0.0,
}


def _optimizer(profit, cm=0.25):
    return {
        "recommended_bid_total_expected_profit": profit,
        "avg_recommended_bid_cm_if_won": cm,
    }


def test_decide_promotion_bootstraps_when_absolute_gates_pass():
    """The first model is eligible only after passing all absolute gates."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.50},
        challenger_optimizer=_optimizer(100, cm=0.25),
        currently_serving_metrics=None,
        currently_serving_optimizer=None,
        **POLICY,
    )
    assert decision.promote is True
    assert "First model passed all absolute promotion thresholds" in decision.reason


def test_decide_promotion_blocks_first_model_above_max_log_loss():
    """The first model must satisfy the absolute log-loss threshold."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.60},
        challenger_optimizer=_optimizer(100),
        currently_serving_metrics=None,
        currently_serving_optimizer=None,
        **POLICY,
    )
    assert decision.promote is False
    assert "exceeds the maximum allowed" in decision.reason


def test_decide_promotion_blocks_negative_challenger_profit():
    """A challenger with negative expected profit is never eligible."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.50},
        challenger_optimizer=_optimizer(-1),
        currently_serving_metrics=None,
        currently_serving_optimizer=None,
        **POLICY,
    )
    assert decision.promote is False
    assert "below the required minimum" in decision.reason


def test_decide_promotion_blocks_on_log_loss_regression():
    """Profit improvement cannot override excessive log-loss regression."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.52},
        challenger_optimizer=_optimizer(1000),
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(500),
        **POLICY,
    )
    assert decision.promote is False
    assert "log loss regressed" in decision.reason
    assert decision.comparison["log_loss_regression"] == pytest.approx(0.02)


def test_decide_promotion_blocks_on_profit_regression():
    """A challenger below the configured profit-ratio floor is rejected."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.50},
        challenger_optimizer=_optimizer(80),
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(100),
        **POLICY,
    )
    assert decision.promote is False
    assert "expected profit" in decision.reason
    assert decision.comparison["profit_ratio"] == pytest.approx(0.8)


def test_decide_promotion_allows_small_profit_dip_within_tolerance():
    """A small profit dip is allowed when both profit tolerances pass."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.50},
        challenger_optimizer=_optimizer(99),
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(100),
        **POLICY,
    )
    assert decision.promote is True


def test_decide_promotion_blocks_when_absolute_profit_loss_exceeds_tolerance():
    """A challenger is rejected when its absolute profit loss is too large."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.50},
        challenger_optimizer=_optimizer(94),
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(100),
        min_profit_ratio=0.90,
        max_absolute_profit_loss_tolerance=5.0,
        max_log_loss=0.55,
        min_expected_profit=0.0,
        max_log_loss_regression=0.01,
    )
    assert decision.promote is False
    assert "absolute profit-loss tolerance" in decision.reason
    assert decision.comparison["absolute_profit_loss"] == pytest.approx(6.0)
    assert decision.comparison["max_absolute_profit_loss_tolerance"] == pytest.approx(
        5.0
    )


def test_decide_promotion_promotes_on_clear_improvement():
    """Profit and log-loss requirements all passing permit promotion."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.45},
        challenger_optimizer=_optimizer(150, cm=0.30),
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(100),
        **POLICY,
    )
    assert decision.promote is True
    assert decision.comparison["profit_ratio"] == pytest.approx(1.5)


def test_decide_promotion_blocks_when_challenger_profit_is_unavailable():
    """Every challenger must provide expected-profit evaluation results."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.49},
        challenger_optimizer={},
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(100),
        **POLICY,
    )
    assert decision.promote is False
    assert (
        "Challenger probability-weighted expected profit is unavailable"
        in decision.reason
    )


def test_decide_promotion_blocks_invalid_nonpositive_serving_profit():
    """A nonpositive serving profit is an invalid relative-comparison state."""
    decision = registry.decide_promotion(
        challenger_metrics={"log_loss": 0.50},
        challenger_optimizer=_optimizer(10),
        currently_serving_metrics={"log_loss": 0.50},
        currently_serving_optimizer=_optimizer(-10),
        **POLICY,
    )
    assert decision.promote is False
    assert "must be greater than zero" in decision.reason
