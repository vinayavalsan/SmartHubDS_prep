"""Tests for SmartHub training reports and comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from smarthub.train_and_predict import training_artifacts as ta


def _evaluation_frame():
    return pd.DataFrame(
        {
            "lead_id": [101, 102, 103],
            "created_at": pd.to_datetime(
                [
                    "2026-08-01 10:00:00",
                    "2026-08-01 11:00:00",
                    "2026-08-01 12:00:00",
                ]
            ),
            "won_flag": [0, 1, 0],
            "prediction": [0.1, 0.8, 0.2],
        }
    )


def _optimizer_frame():
    return pd.DataFrame(
        {
            "lead_id": [101, 102, 103],
            "bid": [1.0, 2.0, 3.0],
            "recommended_bid": [1.5, 2.0, 2.5],
            "recommended_bid_expected_profit": [2.0, 3.0, 4.0],
        }
    )


def test_build_feature_summary_reports_types_missing_and_numeric_stats():
    frame = pd.DataFrame(
        {
            "age": [20.0, None, 40.0, 60.0],
            "vehicles": [1, 2, 2, 3],
            "state": ["CA", "", None, "TX"],
        }
    )

    result = ta.build_feature_summary_dataframe(
        frame,
        continuous_features=["age"],
        discrete_features=["vehicles"],
        categorical_features=["state"],
    ).set_index("feature")

    assert result.loc["age", "type"] == "continuous"
    assert result.loc["age", "missing_count"] == 1
    assert result.loc["age", "missing_pct"] == pytest.approx(25.0)
    assert result.loc["age", "mean"] == pytest.approx(40.0)
    assert result.loc["age", "median"] == pytest.approx(40.0)
    assert result.loc["age", "min"] == pytest.approx(20.0)
    assert result.loc["age", "max"] == pytest.approx(60.0)

    assert result.loc["vehicles", "type"] == "discrete"
    assert result.loc["vehicles", "mode"] == 2
    assert pd.isna(result.loc["vehicles", "mean"])

    assert result.loc["state", "type"] == "categorical"
    assert result.loc["state", "missing_count"] == 2
    assert result.loc["state", "missing_pct"] == pytest.approx(50.0)


def test_build_feature_summary_skips_configured_columns_not_in_frame():
    frame = pd.DataFrame({"age": [20, 30]})

    result = ta.build_feature_summary_dataframe(
        frame,
        continuous_features=["age", "missing_numeric"],
        discrete_features=[],
        categorical_features=["missing_category"],
    )

    assert result["feature"].tolist() == ["age"]


def test_feature_value_counts_tracks_na_empty_percent_and_top_n():
    frame = pd.DataFrame(
        {
            "state": ["CA", "CA", "TX", None, ""],
            "tier": ["A", "B", "A", "C", "D"],
        }
    )

    result = ta.build_feature_value_counts_dataframe(
        frame,
        features=["state", "tier"],
        top_n_per_feature=2,
    )

    assert (result.groupby("feature").size() <= 2).all()

    state_rows = result[result["feature"] == "state"].set_index("feature_value")
    assert state_rows.loc["CA", "count"] == 2
    assert state_rows.loc["CA", "percent"] == pytest.approx(40.0)

    full_state = ta.build_feature_value_counts_dataframe(
        frame,
        features=["state"],
        top_n_per_feature=10,
    ).set_index("feature_value")
    assert "<NA>" in full_state.index
    assert "<EMPTY>" in full_state.index


def test_build_test_set_id_is_deterministic():
    frame = _evaluation_frame()
    kwargs = {
        "training_table_version": "2026-08-19T000000Z",
        "split_settings": {"strategy": "time", "test_size": 0.2},
    }

    first = ta.build_test_set_id(frame, **kwargs)
    second = ta.build_test_set_id(frame.copy(), **kwargs)

    assert first == second
    assert len(first) == 64


def test_build_test_set_id_ignores_split_dict_key_order():
    frame = _evaluation_frame()

    first = ta.build_test_set_id(
        frame,
        training_table_version="v1",
        split_settings={"strategy": "time", "test_size": 0.2},
    )
    second = ta.build_test_set_id(
        frame,
        training_table_version="v1",
        split_settings={"test_size": 0.2, "strategy": "time"},
    )

    assert first == second


@pytest.mark.parametrize(
    "change",
    ["rows", "index", "table_version", "split_settings"],
)
def test_build_test_set_id_changes_when_comparison_identity_changes(change):
    frame = _evaluation_frame()
    table_version = "v1"
    split_settings = {"strategy": "time", "test_size": 0.2}

    baseline = ta.build_test_set_id(
        frame,
        training_table_version=table_version,
        split_settings=split_settings,
    )

    changed_frame = frame.copy()
    changed_version = table_version
    changed_split = dict(split_settings)

    if change == "rows":
        changed_frame.loc[0, "prediction"] = 0.99
    elif change == "index":
        changed_frame.index = [10, 11, 12]
    elif change == "table_version":
        changed_version = "v2"
    elif change == "split_settings":
        changed_split["test_size"] = 0.25

    changed = ta.build_test_set_id(
        changed_frame,
        training_table_version=changed_version,
        split_settings=changed_split,
    )

    assert changed != baseline


def test_build_test_set_id_is_sensitive_to_row_order():
    frame = _evaluation_frame()

    original = ta.build_test_set_id(
        frame,
        training_table_version="v1",
        split_settings={"strategy": "time", "test_size": 0.2},
    )
    reversed_rows = ta.build_test_set_id(
        frame.iloc[::-1],
        training_table_version="v1",
        split_settings={"strategy": "time", "test_size": 0.2},
    )

    assert reversed_rows != original


def test_save_comparison_artifacts_writes_complete_round_trip(tmp_path):
    evaluation = _evaluation_frame()
    optimizer = _optimizer_frame()
    metadata = {
        "training_run_id": "run_123",
        "training_table_version": "table_v4",
        "test_set_id": "abc123",
        "split": {"strategy": "time", "test_size": 0.2},
    }

    saved = ta.save_comparison_artifacts(
        output_dir=tmp_path / "comparison",
        evaluation_df=evaluation,
        optimizer_df=optimizer,
        metadata=metadata,
    )

    assert set(saved) == {
        "artifact_dir",
        "evaluation_dataset",
        "optimizer_results",
        "metadata",
    }

    artifact_dir = Path(saved["artifact_dir"])
    assert artifact_dir == tmp_path / "comparison"
    assert artifact_dir.exists()

    evaluation_loaded = pd.read_parquet(saved["evaluation_dataset"])
    optimizer_loaded = pd.read_parquet(saved["optimizer_results"])
    metadata_loaded = json.loads(Path(saved["metadata"]).read_text())

    pd.testing.assert_frame_equal(evaluation_loaded, evaluation)
    pd.testing.assert_frame_equal(optimizer_loaded, optimizer)
    assert metadata_loaded == metadata


def test_save_comparison_artifacts_overwrites_same_named_files(tmp_path):
    output_dir = tmp_path / "comparison"
    first_eval = _evaluation_frame()
    second_eval = first_eval.copy()
    second_eval["prediction"] = [0.9, 0.9, 0.9]

    ta.save_comparison_artifacts(
        output_dir=output_dir,
        evaluation_df=first_eval,
        optimizer_df=_optimizer_frame(),
        metadata={"run": 1},
    )
    saved = ta.save_comparison_artifacts(
        output_dir=output_dir,
        evaluation_df=second_eval,
        optimizer_df=_optimizer_frame(),
        metadata={"run": 2},
    )

    loaded = pd.read_parquet(saved["evaluation_dataset"])
    metadata = json.loads(Path(saved["metadata"]).read_text())

    assert loaded["prediction"].tolist() == [0.9, 0.9, 0.9]
    assert metadata == {"run": 2}


def test_save_evaluation_summary_writes_json_and_optimizer_rows(tmp_path):
    optimizer = _optimizer_frame()
    summary = {
        "metrics": {"log_loss": 0.42},
        "optimizer": {"expected_profit_lift_total": 12.5},
    }

    path = ta.save_evaluation_summary(
        tmp_path,
        summary,
        optimizer_eval_df=optimizer,
    )

    assert json.loads(path.read_text()) == summary
    saved_optimizer = pd.read_csv(tmp_path / "bid_optimizer_test_rows.csv")
    assert len(saved_optimizer) == len(optimizer)


def test_save_evaluation_summary_omits_optimizer_csv_when_unavailable(tmp_path):
    path = ta.save_evaluation_summary(
        tmp_path,
        {"metrics": {"log_loss": 0.42}},
        optimizer_eval_df=None,
    )

    assert path.exists()
    assert not (tmp_path / "bid_optimizer_test_rows.csv").exists()


def test_save_feature_summary_files_writes_both_tables(tmp_path):
    summary = pd.DataFrame(
        [{"feature": "age", "type": "continuous", "missing_count": 0}]
    )
    counts = pd.DataFrame([{"feature": "state", "feature_value": "CA", "count": 2}])

    ta.save_feature_summary_files(tmp_path, summary, counts)

    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "feature_summary.csv"),
        summary,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "feature_value_counts.csv"),
        counts,
    )


def test_plot_histogram_skips_missing_or_all_nan_column(tmp_path):
    frame = pd.DataFrame({"other": [1, 2, 3]})
    assert (
        ta._plot_histogram(
            tmp_path,
            frame,
            "missing",
            "missing.png",
            "x",
            "title",
        )
        is None
    )

    frame["all_nan"] = [None, None, None]
    assert (
        ta._plot_histogram(
            tmp_path,
            frame,
            "all_nan",
            "all_nan.png",
            "x",
            "title",
        )
        is None
    )


def test_plot_histogram_writes_file_for_valid_values(tmp_path):
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    path = ta._plot_histogram(
        tmp_path,
        frame,
        "value",
        "hist.png",
        "Value",
        "Histogram",
    )

    assert path == tmp_path / "hist.png"
    assert path.exists()


def test_save_optimizer_plots_only_writes_available_diagnostics(tmp_path):
    frame = pd.DataFrame(
        {
            "expected_profit_lift": [1.0, 2.0],
            "bid_change": [0.5, -0.5],
        }
    )

    files = ta._save_optimizer_plots(tmp_path, frame)

    assert set(files) == {
        "optimizer_expected_profit_lift.png",
        "recommended_bid_change.png",
    }
    assert not (tmp_path / "recommended_cm_distribution.png").exists()
    assert not (tmp_path / "current_vs_recommended_win_rate.png").exists()
