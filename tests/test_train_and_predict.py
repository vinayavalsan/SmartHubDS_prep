"""Tests for the Anton train/predict module — pure logic only.

Covers the feature-parity contract, training-table preparation, and the bid
optimizer math. sklearn/mlflow/fastapi are NOT required here (predict keeps
those imports lazy/guarded), so these run in the base env.
"""

import numpy as np
import pandas as pd
import pytest

from smarthub.feature_engineering import features as fe
from smarthub.train_and_predict import config, predict, preprocessing, registry


# --- Feature schema (single source of truth) --------------------------------


def test_model_feature_columns_auto_vs_home():
    num_auto, cat_auto = fe.model_feature_columns(fe.LEAD_TYPE_AUTO)
    num_home, cat_home = fe.model_feature_columns(fe.LEAD_TYPE_HOME)

    # auto-only features present for auto, absent for home
    assert "multi_vehicle" in num_auto and "num_vehicles" in num_auto
    assert "multi_vehicle" not in num_home and "num_vehicles" not in num_home
    assert "home_owner" in cat_auto and "home_owner" not in cat_home
    # home-only features present for home, absent for auto
    assert "num_home_claims" in num_home and "num_home_claims" not in num_auto
    assert "home_property_type" in cat_home
    assert "home_property_type" not in cat_auto
    # shared features present in both
    for col in fe.AGE_COHORT_COLUMNS + ["is_married", "created_hour", "is_workday"]:
        assert col in num_auto and col in num_home
    assert "state" in cat_auto and "state" in cat_home
    assert "traffic_tier" in cat_auto and "traffic_tier" in cat_home
    # high-cardinality / redundant identifiers are excluded
    assert "source_type_id" not in cat_auto
    assert "account_id" not in cat_auto


def test_derive_serving_features_parity():
    raw = pd.DataFrame(
        [{"marital_status": "Married", "num_vehicles": 2, "age": 40,
          "created_hour": 9, "created_dayofweek": 2}]
    )
    out = fe.derive_serving_features(raw)
    assert out.loc[0, "is_married"] == 1
    assert out.loc[0, "multi_vehicle"] == 1
    assert out.loc[0, "age_cohort_35_44"] == 1
    # time parts already supplied are preserved
    assert out.loc[0, "created_hour"] == 9


# --- Serving frame -----------------------------------------------------------


def test_serving_frame_selects_and_normalizes():
    record = {
        "campaign_id": 123, "account_id": 7, "source_type_id": 2,
        "state": "TX", "gender": "Female", "marital_status": "",
        "insured": "true", "home_owner": "false", "dui": "false",
        "military_affiliation": "false",
        "num_vehicles": 2, "num_drivers": 1,
        "num_auto_violations": 0, "num_auto_accidents": 0,
        "continuous_coverage_months": 24, "age": 34,
        "created_hour": 14, "created_dayofweek": 2, "bid": 0.25,
    }
    frame = preprocessing.serving_frame([record], fe.LEAD_TYPE_AUTO)
    numeric, categorical = config.feature_columns(fe.LEAD_TYPE_AUTO)

    assert list(frame.columns) == numeric + categorical
    # ids normalised to strings for one-hot; blank -> NAvail
    assert frame.loc[0, "campaign_id"] == "123"
    assert frame.loc[0, "marital_status"] == "NAvail"
    # derived features computed from raw
    assert frame.loc[0, "multi_vehicle"] == 1
    assert frame.loc[0, "age_cohort_25_34"] == 1


# --- Training-table preparation ---------------------------------------------


def _fake_training_table():
    numeric, categorical = fe.model_feature_columns(fe.LEAD_TYPE_AUTO)
    n = 6
    data = {"id": range(n),
            "created_at": pd.to_datetime([f"2026-06-2{i} 01:00" for i in range(n)]),
            "won_flag": [1, 0, 1, 0, 1, 0],
            "expected_revenue": [20.0, 18.0, 25.0, 22.0, 30.0, 15.0]}
    for c in numeric:
        data[c] = np.arange(n, dtype=float) + 1
    for c in categorical:
        data[c] = ["A", "B"] * (n // 2)
    return pd.DataFrame(data)


def test_prepare_training_data(monkeypatch):
    table = _fake_training_table()
    monkeypatch.setattr(
        preprocessing.io, "load_training_table", lambda name, v=None: table
    )
    monkeypatch.setattr(
        preprocessing.io, "training_versions", lambda name: ["2026-07-06T095148Z"]
    )
    monkeypatch.setattr(
        preprocessing.io,
        "load_training_metadata",
        lambda name, v=None: {
            "data_min_created_at": "2026-06-20",
            "data_max_created_at": "2026-07-06",
            "row_count": 999,
        },
    )

    frame, numeric, categorical, summary = preprocessing.prepare_training_data(
        fe.LEAD_TYPE_AUTO, "auto"
    )
    assert summary["training_rows"] == len(table)
    assert summary["win_rate"] == pytest.approx(0.5)
    # lineage captured
    assert summary["training_table_version"] == "2026-07-06T095148Z"
    assert summary["data_max_created_at"] == "2026-07-06"
    assert summary["source_row_count"] == 999
    # frame carries features + target + expected_revenue, sorted by created_at
    assert config.TARGET_COL in frame.columns
    assert config.REVENUE_COL in frame.columns
    assert set(numeric + categorical).issubset(frame.columns)
    assert frame["created_at"].is_monotonic_increasing


def test_model_type_from_ini_and_fallback(monkeypatch):
    from smarthub.core import task_config

    # Missing config -> model_type falls back to the code default.
    monkeypatch.setenv("SMARTHUB_TASK_CONFIG", "/nonexistent/smarthub.yaml")
    task_config.reload()
    assert config.model_type() == config.MODEL_TYPE
    task_config.reload()  # restore real config for other tests


def test_target_cm_falls_back_when_store_unavailable(monkeypatch):
    # target_cm stays a BUSINESS knob (config store); unreachable -> code default.
    monkeypatch.setenv(
        "SMARTHUB_CONFIG_DB_URL", "postgresql+psycopg2://x:x@127.0.0.1:1/none"
    )
    assert config.target_cm_value() == config.TARGET_CM


def test_drop_zero_variance():
    frame = pd.DataFrame(
        {"a": [1, 2, 3], "b": [5, 5, 5], "c": ["x", "y", "x"], "d": ["k", "k", "k"]}
    )
    num, cat, dropped = preprocessing.drop_zero_variance(frame, ["a", "b"], ["c", "d"])
    assert num == ["a"]
    assert cat == ["c"]
    assert set(dropped) == {"b", "d"}


def test_assert_trainable_rejects_single_class():
    # win_rate=1.0 (all wins) -> not trainable, clear message.
    frame = pd.DataFrame({config.TARGET_COL: [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="only ONE target class"):
        preprocessing.assert_trainable(frame, "auto")


def test_assert_trainable_accepts_two_classes():
    frame = pd.DataFrame({config.TARGET_COL: [1, 0, 1, 0]})
    assert preprocessing.assert_trainable(frame, "auto") == [0, 1]


def test_prepare_training_data_missing_target(monkeypatch):
    table = _fake_training_table().drop(columns=["won_flag"])
    monkeypatch.setattr(
        preprocessing.io, "load_training_table", lambda name, v=None: table
    )
    with pytest.raises(ValueError, match="missing target"):
        preprocessing.prepare_training_data(fe.LEAD_TYPE_AUTO, "auto")


# --- Bid optimizer math ------------------------------------------------------


class _StubModel:
    """Constant win-probability model: predict_proba -> p for every row."""

    def __init__(self, p=0.5):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def test_optimize_bid_picks_max_profit():
    # Constant win rate -> profit = p*(rev - bid) is maximised at the lowest bid.
    row = pd.Series({"bid": 5.0, "state": "TX"})
    result = predict.optimize_bid_for_row(
        row=row, model=_StubModel(0.5), expected_revenue=25.0,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
    )
    assert result["max_bid"] == pytest.approx(25.0 * 0.75)  # 18.75
    assert result["recommended_bid"] == pytest.approx(0.25)  # lowest bid wins
    assert result["recommended_bid_predicted_win_rate"] == pytest.approx(0.5)
    assert result["n_candidate_bids"] > 0


def test_optimize_bid_skips_when_no_room():
    row = pd.Series({"bid": 1.0})
    # expected_revenue tiny -> max_bid below min_bid -> no candidates
    result = predict.optimize_bid_for_row(
        row=row, model=_StubModel(), expected_revenue=0.1,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
    )
    assert np.isnan(result["recommended_bid"])
    assert result["n_candidate_bids"] == 0


# --- bid_curve_around (explanatory "shape of the market" view) --------------


def test_bid_curve_around_spans_center_bid_symmetrically():
    """Points bracket center_bid on the bid_step grid, sorted, deduplicated."""
    row = pd.Series({"bid": 5.0, "state": "TX"})
    curve = predict.bid_curve_around(
        row=row, model=_StubModel(0.5), expected_revenue=20.0,
        min_bid=0.25, max_bid=15.0, bid_step=0.25, center_bid=5.0, n_points=2,
    )
    bids = [pt["bid"] for pt in curve]
    assert bids == sorted(bids)
    assert bids == pytest.approx([4.5, 4.75, 5.0, 5.25, 5.5])
    for pt in curve:
        assert pt["predicted_win_rate"] == pytest.approx(0.5)
        assert pt["expected_profit"] == pytest.approx(0.5 * (20.0 - pt["bid"]))


def test_bid_curve_around_clips_to_bounds_without_duplicates():
    """Near an edge, out-of-range offsets clip to the bound, not duplicate it."""
    row = pd.Series({"bid": 0.25, "state": "TX"})
    curve = predict.bid_curve_around(
        row=row, model=_StubModel(0.5), expected_revenue=20.0,
        min_bid=0.25, max_bid=15.0, bid_step=0.25, center_bid=0.25, n_points=2,
    )
    bids = [pt["bid"] for pt in curve]
    assert min(bids) == pytest.approx(0.25)
    assert len(bids) == len(set(round(b, 6) for b in bids))  # no dupes


def test_bid_curve_around_empty_when_no_viable_bid():
    """No room between min_bid and max_bid, or a NaN center -> empty, not a crash."""
    row = pd.Series({"bid": 1.0})
    assert predict.bid_curve_around(
        row=row, model=_StubModel(), expected_revenue=0.1,
        min_bid=0.25, max_bid=0.1, bid_step=0.25, center_bid=0.25,
    ) == []
    assert predict.bid_curve_around(
        row=row, model=_StubModel(), expected_revenue=20.0,
        min_bid=0.25, max_bid=15.0, bid_step=0.25, center_bid=float("nan"),
    ) == []


# --- Model resolution (MODEL_URI env > pinned version > currently serving) --


def test_resolve_model_uri_prefers_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.setenv("MODEL_URI", "s3://somewhere/pinned-model.pkl")
    resolved = predict.resolve_model_uri(fe.LEAD_TYPE_AUTO)
    assert resolved == "s3://somewhere/pinned-model.pkl"


def test_resolve_model_uri_uses_pinned_ini_version_over_currently_serving(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    manifest = registry.save_version(
        {"m": 1}, "auto", feature_cols=["bid"], metrics={"roc_auc": 0.7},
        optimizer_summary={}, lineage={}, model_params={},
    )
    registry.promote("auto", manifest["version"])
    monkeypatch.setattr(config, "active_model_version", lambda: manifest["version"])

    resolved = predict.resolve_model_uri(fe.LEAD_TYPE_AUTO)
    assert resolved == str(registry.version_path("auto", manifest["version"]))


def test_resolve_model_uri_falls_back_to_currently_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    manifest = registry.save_version(
        {"m": 1}, "auto", feature_cols=["bid"], metrics={"roc_auc": 0.7},
        optimizer_summary={}, lineage={}, model_params={},
    )
    registry.promote("auto", manifest["version"])

    resolved = predict.resolve_model_uri(fe.LEAD_TYPE_AUTO)
    assert resolved == str(registry.currently_serving_model_path("auto"))


def test_resolve_model_uri_nothing_serving_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    with pytest.raises(FileNotFoundError):
        predict.resolve_model_uri(fe.LEAD_TYPE_AUTO)


def test_run_bid_optimizer_evaluation_summary():
    numeric, categorical = fe.model_feature_columns(fe.LEAD_TYPE_AUTO)
    feature_cols = numeric + categorical
    n = 4
    df = pd.DataFrame({c: np.ones(n) for c in numeric})
    for c in categorical:
        df[c] = "A"
    df["bid"] = [5.0, 6.0, 7.0, 8.0]
    df[config.REVENUE_COL] = [20.0, 22.0, 25.0, 30.0]

    out = predict.run_bid_optimizer_evaluation(
        test_eval_df=df, model=_StubModel(0.5), feature_cols=feature_cols,
    )
    assert out is not None
    eval_df, summary = out
    assert summary["optimizer_rows"] == n
    # constant win rate -> optimizer lowers bids -> decreases dominate
    assert summary["bid_decrease_count"] == n
    keys = {"expected_profit_lift_pct", "avg_recommended_bid_cm_if_won"}
    assert keys.issubset(summary)


# --- Cold-start / exploration bidding policy (predict.decide_bid) -----------


def test_cold_start_fallback_bid_snaps_to_grid_within_bounds():
    """The fallback bid is a fixed % of [min_bid, ceiling], on the bid grid."""
    result = predict.cold_start_fallback_bid(
        expected_revenue=20.0, target_cm=0.25, min_bid=0.25, bid_step=0.25,
        fallback_pct=0.5,
    )
    max_bid = 20.0 * 0.75
    assert result["max_bid"] == pytest.approx(max_bid)
    assert 0.25 <= result["recommended_bid"] <= max_bid
    # on the {0.25, 0.5, 0.75, ...} grid
    assert (result["recommended_bid"] - 0.25) % 0.25 == pytest.approx(0.0)
    assert result["recommended_bid_predicted_win_rate"] is None
    assert result["recommended_bid_expected_profit"] is None


def test_cold_start_fallback_bid_empty_when_revenue_too_low():
    """No room between min_bid and the ceiling -> empty result, like the optimizer."""
    result = predict.cold_start_fallback_bid(
        expected_revenue=0.1, target_cm=0.25, min_bid=0.25, bid_step=0.25,
    )
    assert np.isnan(result["recommended_bid"])


def test_exploration_slot_is_deterministic_and_alternates_direction():
    """10% variance -> 1-in-10 hour-of-week buckets explore, alternating direction."""
    # bucket 0 = Monday 00:00 -> triggers, direction +1
    explore, direction = predict.exploration_slot(0, 0, variance_pct=0.10)
    assert explore is True and direction == 1
    # bucket 10 -> triggers, direction flips to -1
    explore, direction = predict.exploration_slot(0, 10, variance_pct=0.10)
    assert explore is True and direction == -1
    # bucket 5 -> not a multiple of 10 -> no explore
    explore, direction = predict.exploration_slot(0, 5, variance_pct=0.10)
    assert explore is False and direction is None
    # same inputs always give the same answer (reproducible/auditable)
    assert predict.exploration_slot(0, 0, variance_pct=0.10) == (True, 1)


def test_exploration_slot_disabled_when_variance_pct_zero():
    """A zero/negative variance_pct means "no exploration", ever."""
    assert predict.exploration_slot(0, 0, variance_pct=0.0) == (False, None)


def test_model_recency_flags_stale_model_past_window():
    """A model trained on data older than the recency window is flagged stale."""
    manifest = {"lineage": {"data_max_created_at": "2026-06-01T00:00:00"}}
    age_days, is_stale = predict.model_recency(
        manifest, recency_window_days=30, now=pd.Timestamp("2026-07-14", tz="UTC")
    )
    assert age_days == pytest.approx(43.0, abs=0.1)
    assert is_stale is True


def test_model_recency_within_window_not_stale():
    manifest = {"lineage": {"data_max_created_at": "2026-07-10T00:00:00"}}
    age_days, is_stale = predict.model_recency(
        manifest, recency_window_days=30, now=pd.Timestamp("2026-07-14", tz="UTC")
    )
    assert is_stale is False


def test_model_recency_none_when_lineage_missing():
    """No manifest / no lineage data date -> (None, False), not a crash."""
    assert predict.model_recency(None) == (None, False)
    assert predict.model_recency({"lineage": {}}) == (None, False)


def test_decide_bid_cold_start_when_no_model():
    """model=None -> the defined cold-start fallback, flagged explicitly."""
    row = pd.Series({"bid": 5.0, "state": "TX"})
    result = predict.decide_bid(
        row=row, model=None, manifest=None, expected_revenue=20.0,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
    )
    assert result["decision_path"] == "cold_start_fallback"
    assert "cold start" in result["decision_reason"]
    assert not np.isnan(result["recommended_bid"])


def test_decide_bid_normal_path_when_not_an_explore_slot():
    """A non-explore hour -> the plain optimizer bid, decision_path='model'."""
    row = pd.Series({"bid": 5.0, "state": "TX"})
    result = predict.decide_bid(
        row=row, model=_StubModel(0.5), manifest=None, expected_revenue=20.0,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
        created_dayofweek=0, created_hour=5,  # bucket 5 -> not an explore slot
    )
    assert result["decision_path"] == "model"
    assert result["recommended_bid"] == pytest.approx(0.25)


class _RisingWinRateModel:
    """win rate rises with bid (toy model, for an interior profit optimum).

    p = bid / 20, so profit = p*(rev - bid) peaks at an interior bid rather
    than always at the floor -- needed so a 10% exploration perturbation
    actually moves the bid onto a different grid point.
    """

    def predict_proba(self, X):
        bids = X["bid"].to_numpy(dtype=float)
        p = np.clip(bids / 20.0, 0.0, 0.95)
        return np.column_stack([1 - p, p])


def test_decide_bid_exploration_path_perturbs_optimum_bid():
    """A scheduled explore hour perturbs the bid and says so, explicitly."""
    row = pd.Series({"bid": 5.0, "state": "TX"})
    result = predict.decide_bid(
        row=row, model=_RisingWinRateModel(), manifest=None, expected_revenue=20.0,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
        created_dayofweek=0, created_hour=0,  # bucket 0 -> explore, direction +1
    )
    assert result["decision_path"] == "exploration"
    assert result["pre_exploration_optimum_bid"] == pytest.approx(10.0, abs=0.25)
    assert result["recommended_bid"] > result["pre_exploration_optimum_bid"]
    assert "Scheduled exploration probe" in result["decision_reason"]


def test_decide_bid_flags_stale_model_in_reason():
    """A stale currently-serving model surfaces that fact in decision_reason."""
    row = pd.Series({"bid": 5.0, "state": "TX"})
    manifest = {"lineage": {"data_max_created_at": "2020-01-01T00:00:00"}}
    result = predict.decide_bid(
        row=row, model=_StubModel(0.5), manifest=manifest, expected_revenue=20.0,
        target_cm=0.25, min_bid=0.25, bid_step=0.25,
        created_dayofweek=0, created_hour=5,  # not an explore slot
    )
    assert result["decision_path"] == "model"
    assert "due for retraining" in result["decision_reason"]
    assert result["model_data_age_days"] > config.RECENCY_WINDOW_DAYS


# --- load_model_and_manifest (cold-start-aware model resolution) ------------


def test_load_model_and_manifest_returns_none_none_when_nothing_serving(
    tmp_path, monkeypatch
):
    """True cold start: no version ever saved -> (None, None), not a raise."""
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)

    model, manifest = predict.load_model_and_manifest(fe.LEAD_TYPE_AUTO)
    assert model is None
    assert manifest is None


def test_load_model_and_manifest_loads_currently_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR_ROOT", tmp_path / "models")
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setattr(config, "active_model_version", lambda: None)
    saved = registry.save_version(
        {"m": 1}, "auto", feature_cols=["bid"], metrics={"roc_auc": 0.7},
        optimizer_summary={}, lineage={"data_max_created_at": "2026-07-01"},
        model_params={},
    )
    registry.promote("auto", saved["version"])

    model, manifest = predict.load_model_and_manifest(fe.LEAD_TYPE_AUTO)
    assert model == {"m": 1}
    assert manifest["version"] == saved["version"]
