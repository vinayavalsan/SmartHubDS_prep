"""Tests for train.py's promotion-gate helper.

`train.run_training` itself is a heavy integration function (real
sklearn/lightgbm fit + report/plot generation) not covered here; this file
targets `_evaluate_currently_serving_model` in isolation. Importing
`train.py` pulls in `metrics.py` -> sklearn at module level (unlike the
lazy-import modules elsewhere in this package), so these tests need the
`ml` extra and are skipped in the base test env.
"""

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from smarthub.train_and_predict import registry, train  # noqa: E402


class _StubLogger:
    """Minimal stand-in for a Prefect/stdlib logger -- just records calls."""

    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def test_evaluate_currently_serving_model_none_when_nothing_serving(monkeypatch):
    """No promoted version yet -> (None, None), logged as the bootstrap case."""
    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: None
    )
    log = _StubLogger()
    result = train._evaluate_currently_serving_model(
        "auto", pd.DataFrame(), pd.Series(dtype=int), 0.25, 0.25, 0.25, 100, log
    )
    assert result == (None, None)
    assert any("first model" in m for m in log.infos)


def test_evaluate_currently_serving_model_propagates_registry_load_error(monkeypatch):
    """Registry loading errors currently propagate to the training workflow."""

    def _boom(lead_type_name):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        registry, "currently_serving_version", lambda lead_type_name: "v1_x"
    )
    monkeypatch.setattr(registry, "load_currently_serving_model", _boom)
    log = _StubLogger()
    with pytest.raises(ValueError, match="Expecting value"):
        train._evaluate_currently_serving_model(
            "auto", pd.DataFrame(), pd.Series(dtype=int), 0.25, 0.25, 0.25, 100, log
        )


def test_evaluate_currently_serving_model_propagates_schema_mismatch(monkeypatch):
    """An incompatible serving feature schema raises a clear KeyError."""
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
    log = _StubLogger()
    test_df = pd.DataFrame({"bid": [1.0, 2.0]})
    with pytest.raises(KeyError, match="not_a_real_column"):
        train._evaluate_currently_serving_model(
            "auto", test_df, pd.Series([0, 1]), 0.25, 0.25, 0.25, 100, log
        )
