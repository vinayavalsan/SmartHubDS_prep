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
        registry, "load_currently_serving_model", lambda lead_type_name: (None, None)
    )
    log = _StubLogger()
    result = train._evaluate_currently_serving_model(
        "auto", pd.DataFrame(), pd.Series(dtype=int), 0.25, 0.25, log
    )
    assert result == (None, None)
    assert any("first model" in m for m in log.infos)


def test_evaluate_currently_serving_model_survives_registry_crash(monkeypatch):
    """A raise from registry.load_currently_serving_model (e.g. a corrupt
    current.json's JSONDecodeError) must degrade to (None, None), not
    propagate and crash the training run -- this exact failure mode took
    down a real training flow (see docs/CHANGELOG.md).
    """
    def _boom(lead_type_name):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(registry, "load_currently_serving_model", _boom)
    log = _StubLogger()
    result = train._evaluate_currently_serving_model(
        "auto", pd.DataFrame(), pd.Series(dtype=int), 0.25, 0.25, log
    )
    assert result == (None, None)
    assert any("Could not load/score" in m for m in log.warnings)


def test_evaluate_currently_serving_model_survives_schema_mismatch(monkeypatch):
    """An incompatible feature schema (test_df missing a trained-on column)
    also degrades gracefully rather than raising a KeyError."""
    monkeypatch.setattr(
        registry, "load_currently_serving_model",
        lambda lead_type_name: (
            object(), {"version": "v1_x", "feature_cols": ["not_a_real_column"]}
        ),
    )
    log = _StubLogger()
    test_df = pd.DataFrame({"bid": [1.0, 2.0]})
    result = train._evaluate_currently_serving_model(
        "auto", test_df, pd.Series([0, 1]), 0.25, 0.25, log
    )
    assert result == (None, None)
    assert any("Could not load/score" in m for m in log.warnings)
