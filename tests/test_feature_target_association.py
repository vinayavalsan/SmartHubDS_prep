"""Tests for pre-training feature-to-target association diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarthub.train_and_predict import feature_target_association as fta


def test_numeric_values_coerces_and_median_fills_missing():
    series = pd.Series([1, "2", None, "bad", 5])

    values = fta._numeric_values(series)

    assert values.shape == (5, 1)
    assert values[:, 0].tolist() == [1.0, 2.0, 2.0, 2.0, 5.0]
    assert np.isfinite(values).all()


def test_numeric_values_all_missing_falls_back_to_zero():
    series = pd.Series([None, np.nan, "bad"])

    values = fta._numeric_values(series)

    assert values.shape == (3, 1)
    assert values[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_categorical_values_handles_missing_and_is_deterministic():
    series = pd.Series(["TX", None, "CA", "TX", ""])

    first = fta._categorical_values(series)
    second = fta._categorical_values(series)

    assert first.shape == (5, 1)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4


def test_constant_feature_has_zero_mutual_information():
    frame = pd.DataFrame(
        {
            "constant_numeric": [1.0] * 20,
            "constant_categorical": ["A"] * 20,
            "won_flag": [0, 1] * 10,
        }
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=["constant_numeric"],
        categorical_features=["constant_categorical"],
        target_column="won_flag",
        random_seed=42,
    )

    by_feature = {row["feature"]: row for row in result}
    assert by_feature["constant_numeric"]["mutual_information"] == 0.0
    assert by_feature["constant_categorical"]["mutual_information"] == 0.0


def test_strong_categorical_signal_ranks_above_unrelated_feature():
    n = 200
    target = np.array([0, 1] * (n // 2))
    rng = np.random.default_rng(123)

    frame = pd.DataFrame(
        {
            "strong_signal": np.where(target == 1, "high", "low"),
            "noise": rng.choice(["A", "B", "C", "D"], size=n),
            "won_flag": target,
        }
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=[],
        categorical_features=["strong_signal", "noise"],
        target_column="won_flag",
        random_seed=42,
    )

    assert result[0]["feature"] == "strong_signal"
    assert result[0]["rank"] == 1
    assert result[0]["mutual_information"] > result[1]["mutual_information"]


def test_strong_numeric_signal_ranks_above_unrelated_feature():
    n = 300
    target = np.array([0, 1] * (n // 2))
    rng = np.random.default_rng(456)

    frame = pd.DataFrame(
        {
            "strong_numeric": target.astype(float),
            "noise_numeric": rng.normal(size=n),
            "won_flag": target,
        }
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=["strong_numeric", "noise_numeric"],
        categorical_features=[],
        target_column="won_flag",
        random_seed=42,
    )

    assert result[0]["feature"] == "strong_numeric"
    assert result[0]["mutual_information"] > result[1]["mutual_information"]


def test_missing_configured_feature_is_skipped():
    frame = pd.DataFrame(
        {
            "present": [0.0, 1.0, 0.0, 1.0],
            "won_flag": [0, 1, 0, 1],
        }
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=["present", "missing_numeric"],
        categorical_features=["missing_categorical"],
        target_column="won_flag",
        random_seed=42,
    )

    assert [row["feature"] for row in result] == ["present"]


def test_rows_with_missing_target_are_excluded():
    frame = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 999.0, 1.0, 0.0],
            "won_flag": [0, 1, None, 1, 0],
        }
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=["feature"],
        categorical_features=[],
        target_column="won_flag",
        random_seed=42,
    )

    assert len(result) == 1
    assert result[0]["feature"] == "feature"
    assert np.isfinite(result[0]["mutual_information"])


def test_feature_type_is_reported_correctly():
    frame = pd.DataFrame(
        {
            "numeric_feature": [0.0, 1.0] * 20,
            "categorical_feature": ["A", "B"] * 20,
            "won_flag": [0, 1] * 20,
        }
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=["numeric_feature"],
        categorical_features=["categorical_feature"],
        target_column="won_flag",
        random_seed=42,
    )

    by_feature = {row["feature"]: row for row in result}
    assert by_feature["numeric_feature"]["feature_type"] == "numeric"
    assert by_feature["categorical_feature"]["feature_type"] == "categorical"


def test_results_are_sorted_by_mutual_information_then_feature_name(monkeypatch):
    frame = pd.DataFrame(
        {
            "z_feature": [0.0, 1.0] * 10,
            "a_feature": [1.0, 0.0] * 10,
            "won_flag": [0, 1] * 10,
        }
    )

    monkeypatch.setattr(
        fta,
        "mutual_info_classif",
        lambda values, target, discrete_features, random_state: np.array([0.25]),
    )

    result = fta.build_feature_target_association(
        frame=frame,
        numeric_features=["z_feature", "a_feature"],
        categorical_features=[],
        target_column="won_flag",
        random_seed=42,
    )

    assert [row["feature"] for row in result] == ["a_feature", "z_feature"]
    assert [row["rank"] for row in result] == [1, 2]


def test_repeated_calls_with_same_seed_are_identical():
    rng = np.random.default_rng(999)
    n = 100
    frame = pd.DataFrame(
        {
            "numeric_feature": rng.normal(size=n),
            "categorical_feature": rng.choice(["A", "B", "C"], size=n),
            "won_flag": rng.integers(0, 2, size=n),
        }
    )

    kwargs = dict(
        frame=frame,
        numeric_features=["numeric_feature"],
        categorical_features=["categorical_feature"],
        target_column="won_flag",
        random_seed=17,
    )

    first = fta.build_feature_target_association(**kwargs)
    second = fta.build_feature_target_association(**kwargs)

    assert first == second


def test_target_column_missing_raises_key_error():
    frame = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(KeyError):
        fta.build_feature_target_association(
            frame=frame,
            numeric_features=["feature"],
            categorical_features=[],
            target_column="won_flag",
            random_seed=42,
        )


def test_log_feature_target_association_logs_empty_message():
    class _Logger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = _Logger()

    fta.log_feature_target_association([], logger)

    assert logger.messages[0] == "Feature-Target Association"
    assert any(
        "No feature-target association diagnostics available." in message
        for message in logger.messages
    )


def test_log_feature_target_association_logs_ranked_table():
    class _Logger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = _Logger()
    diagnostics = [
        {
            "rank": 1,
            "feature": "state",
            "feature_type": "categorical",
            "mutual_information": 0.123456789,
        }
    ]

    fta.log_feature_target_association(diagnostics, logger)

    rendered = "\n".join(logger.messages)
    assert "Feature-Target Association" in rendered
    assert "state" in rendered
    assert "categorical" in rendered
    assert "0.123457" in rendered
