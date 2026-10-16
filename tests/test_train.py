"""Tests for train.py's promotion-gate helper.

`train.run_training` itself is a heavy integration function (real
sklearn/lightgbm fit + report/plot generation) not covered here; this file
targets `_evaluate_currently_serving_model` in isolation. Importing
`train.py` pulls in `metrics.py` -> sklearn at module level (unlike the
lazy-import modules elsewhere in this package), so these tests need the
`ml` extra and are skipped in the base test env.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from smarthub.train_and_predict import registry, train  # noqa: E402


def _monotonicity_config():
    """Return the minimal monotonicity config required by the helper."""
    return SimpleNamespace(
        tolerance=1e-8,
        max_violation_rate=0.0,
    )


def test_evaluate_currently_serving_model_none_when_nothing_serving(monkeypatch):
    """No promoted version yet -> (None, None), logged as the bootstrap case."""
    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: None
    )
    logged_messages = []
    monkeypatch.setattr(
        train.logger,
        "info",
        lambda message, *args: logged_messages.append(
            message % args if args else message
        ),
    )
    result = train._evaluate_currently_serving_model(
        "auto",
        pd.DataFrame(),
        pd.Series(dtype=int),
        0.25,
        0.25,
        0.25,
        100,
        _monotonicity_config(),
    )
    assert result == (None, None)
    assert any("first model" in message for message in logged_messages)


def test_evaluate_currently_serving_model_propagates_registry_load_error(monkeypatch):
    """Registry loading errors currently propagate to the training workflow."""

    def _boom(lead_type_name):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v1_x"
    )
    monkeypatch.setattr(registry, "load_currently_serving_model", _boom)
    with pytest.raises(ValueError, match="Expecting value"):
        train._evaluate_currently_serving_model(
            "auto",
            pd.DataFrame(),
            pd.Series(dtype=int),
            0.25,
            0.25,
            0.25,
            100,
            _monotonicity_config(),
        )


def test_evaluate_currently_serving_model_skips_incompatible_schema(monkeypatch):
    """An incompatible serving feature schema is treated as not comparable
    (logged) rather than raising -- so a feature-pipeline migration (e.g. the
    one-hot age_cohort_* / age_missing columns replaced by a single age_cohort)
    doesn't fail the whole training run. The challenger is then judged on the
    absolute promotion gates alone (decide_promotion handles None serving
    metrics)."""
    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v1_x"
    )
    monkeypatch.setattr(
        registry,
        "load_currently_serving_model",
        lambda lead_type_name: (
            object(),
            {"version": "v1_x", "feature_cols": ["not_a_real_column"]},
        ),
    )
    test_df = pd.DataFrame({"bid": [1.0, 2.0]})
    logged = []
    monkeypatch.setattr(
        train.logger,
        "warning",
        lambda message, *args: logged.append(message % args if args else message),
    )
    result = train._evaluate_currently_serving_model(
        "auto",
        test_df,
        pd.Series([0, 1]),
        0.25,
        0.25,
        0.25,
        100,
        _monotonicity_config(),
    )
    assert result == (None, None)
    assert any(
        "not present in the current training data" in message for message in logged
    )


def test_evaluate_currently_serving_model_skips_on_scoring_failure(monkeypatch):
    """If the champion pipeline cannot score, treat it as not comparable."""

    class _Boom:
        def predict_proba(self, X):
            raise RuntimeError("incompatible encoders")

        def predict(self, X):
            raise RuntimeError("incompatible encoders")

    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v2_x"
    )
    monkeypatch.setattr(
        registry,
        "load_currently_serving_model",
        lambda lead_type_name: (
            _Boom(),
            {"version": "v2_x", "feature_cols": ["bid"]},
        ),
    )

    logged = []
    monkeypatch.setattr(
        train.logger,
        "warning",
        lambda message, *args: logged.append(message % args if args else message),
    )

    test_df = pd.DataFrame({"bid": [1.0, 2.0]})
    result = train._evaluate_currently_serving_model(
        "auto",
        test_df,
        pd.Series([0, 1]),
        0.25,
        0.25,
        0.25,
        100,
        _monotonicity_config(),
    )
    assert result == (None, None)
    assert any("not comparable" in message for message in logged)
