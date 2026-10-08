"""Reproducible candidate splits without persisted row lists."""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from smarthub.train_and_predict import model_parameters


@pytest.mark.parametrize(
    "strategy,stratify", [("time", False), ("random", False), ("random", True)]
)
def test_candidate_split_matches_original_hpo_algorithm(strategy, stratify):
    frame = (
        pd.DataFrame(
            {
                "created_at": pd.to_datetime(["2026-10-01"] * 10 + ["2026-10-02"] * 10),
                "won_flag": [0, 1] * 10,
                "row_id": range(20),
            }
        )
        .sample(frac=1, random_state=5)
        .reset_index(drop=True)
    )
    settings = {"strategy": strategy, "test_size": 0.25, "stratify": stratify}
    if strategy == "time":
        ordered = frame.sort_values("created_at", kind="stable")
        expected_fit, expected_test = ordered.iloc[:-5], ordered.iloc[-5:]
    else:
        expected_fit, expected_test = train_test_split(
            frame,
            test_size=0.25,
            random_state=17,
            shuffle=True,
            stratify=frame["won_flag"] if stratify else None,
        )
    data = {
        "frame_fingerprint": model_parameters.dataset_fingerprint(frame),
        "split_version": 1,
        "split_settings": settings,
        "random_seed": 17,
    }
    fit, test = model_parameters.candidate_partitions(frame, data, "won_flag")
    pd.testing.assert_frame_equal(fit, expected_fit)
    pd.testing.assert_frame_equal(test, expected_test)
    daily_fit, daily_test = model_parameters.split_training_data(
        frame, "won_flag", settings, 17
    )
    pd.testing.assert_frame_equal(fit, daily_fit)
    pd.testing.assert_frame_equal(test, daily_test)
    with pytest.raises(ValueError, match="changed"):
        model_parameters.candidate_partitions(frame.iloc[::-1], data, "won_flag")
    with pytest.raises(ValueError, match="split_version"):
        model_parameters.candidate_partitions(
            frame, {**data, "split_version": 99}, "won_flag"
        )


def test_legacy_partition_artifacts_remain_readable():
    frame = pd.DataFrame({"row_id": range(4)})
    data = {
        "frame_fingerprint": model_parameters.dataset_fingerprint(frame),
        "training_positions": [2, 0, 1],
        "test_positions": [3],
    }
    fit, test = model_parameters.candidate_partitions(frame, data)
    assert fit.index.tolist() == [2, 0, 1]
    assert test.index.tolist() == [3]
    with pytest.raises(ValueError, match="overlap"):
        model_parameters.candidate_partitions(frame, {**data, "test_positions": [1]})


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_invalid_split_fraction_is_rejected(test_size):
    with pytest.raises(ValueError, match="test_size"):
        model_parameters.split_training_data(
            pd.DataFrame({"row_id": range(4)}),
            "won_flag",
            {"strategy": "time", "test_size": test_size},
            17,
        )


def test_split_cannot_leave_empty_training_partition():
    frame = pd.DataFrame({"created_at": pd.to_datetime(["2026-10-01"])})
    with pytest.raises(ValueError, match="empty"):
        model_parameters.split_training_data(
            frame, "won_flag", {"strategy": "time", "test_size": 0.2}, 17
        )
