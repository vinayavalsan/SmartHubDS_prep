"""Focused tests for SmartHub model-family categorical preprocessing."""

import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from smarthub.train_and_predict import models


class _FakeTreeClassifier(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-compatible classifier used instead of XGBoost/LightGBM."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_params(self, deep=True):
        return dict(self.kwargs)

    def set_params(self, **params):
        self.kwargs.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(
        self,
        X,
        y,
        eval_set=None,
        eval_metric=None,
        callbacks=None,
        **kwargs,
    ):
        self.classes_ = np.array(sorted(pd.Series(y).dropna().unique()))
        self.best_iteration_ = 1
        metric = eval_metric or "binary_logloss"
        self.best_score_ = {"valid_0": {metric: 0.5}}
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.tile(np.array([[0.5, 0.5]]), (n, 1))


@pytest.fixture(autouse=True)
def _fake_tree_model_packages(monkeypatch):
    """Provide tiny stand-ins so these preprocessing tests need no ML extras."""
    xgboost = types.ModuleType("xgboost")
    xgboost.XGBClassifier = _FakeTreeClassifier

    lightgbm = types.ModuleType("lightgbm")
    lightgbm.LGBMClassifier = _FakeTreeClassifier
    lightgbm.early_stopping = lambda *args, **kwargs: object()
    lightgbm.log_evaluation = lambda *args, **kwargs: object()
    lightgbm.record_evaluation = lambda *args, **kwargs: object()

    monkeypatch.setitem(sys.modules, "xgboost", xgboost)
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)


def _frame():
    return pd.DataFrame(
        {
            "bid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "age": [20, 30, 40, 50, 60, 25, 35, 45],
            "source_type_id": ["100", "200", "100", "300", "200", "300", "100", "200"],
            "traffic_tier": ["a", "b", "a", "c", "b", "c", "a", "b"],
        }
    )


def _target():
    return pd.Series([0, 0, 0, 1, 0, 1, 1, 1])


def _unknown_frame():
    return pd.DataFrame(
        {
            "bid": [4.5],
            "age": [40],
            "source_type_id": ["999"],
            "traffic_tier": ["new_tier"],
        }
    )


@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm"])
def test_tree_models_keep_registry_categoricals_native(model_type):
    # These tree libs are the `ml` extra, not installed in the base CI env
    # (pip install .[dev,validation]); skip rather than error there, same
    # convention as the fastapi-gated serving tests.
    pytest.importorskip(model_type)
    params = {"n_estimators": 5}
    if model_type == "xgboost":
        params.update({"tree_method": "hist", "verbosity": 0})
    else:
        params.update({"verbose": -1})

    model = models.build_model(
        model_type,
        ["bid", "age"],
        ["source_type_id", "traffic_tier"],
        params,
        calibration_enabled=False,
        calibration_method=None,
        calibration_cv=None,
    )
    model.fit(_frame(), _target())

    transformed = model.named_steps["preprocessor"].transform(_unknown_frame())

    assert str(transformed["source_type_id"].dtype) == "category"
    assert str(transformed["traffic_tier"].dtype) == "category"
    assert transformed.loc[0, "source_type_id"] is pd.NA or pd.isna(
        transformed.loc[0, "source_type_id"]
    )
    assert pd.isna(transformed.loc[0, "traffic_tier"])
    assert model.predict_proba(_unknown_frame()).shape == (1, 2)


def test_logistic_regression_unknown_category_still_scores():
    model = models.build_model(
        "logistic_regression",
        ["bid", "age"],
        ["source_type_id", "traffic_tier"],
        {"max_iter": 200},
        calibration_enabled=False,
        calibration_method=None,
        calibration_cv=None,
    )
    model.fit(_frame(), _target())
    assert model.predict_proba(_unknown_frame()).shape == (1, 2)


def test_native_category_vocabulary_survives_joblib_round_trip(tmp_path):
    pytest.importorskip("lightgbm")  # `ml` extra; skip in the base CI env
    model = models.build_model(
        "lightgbm",
        ["bid", "age"],
        ["source_type_id", "traffic_tier"],
        {"n_estimators": 5, "verbose": -1},
        calibration_enabled=False,
        calibration_method=None,
        calibration_cv=None,
    )
    model.fit(_frame(), _target())

    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    restored = joblib.load(path)

    preprocessor = restored.named_steps["preprocessor"]
    assert preprocessor.categories_["source_type_id"] == ["100", "200", "300"]
    assert restored.predict_proba(_unknown_frame()).shape == (1, 2)


def test_lightgbm_early_stopping_uses_fit_vocabulary_for_validation():
    pytest.importorskip("lightgbm")  # `ml` extra; skip in the base CI env
    pipeline = models.build_lightgbm_model(
        ["bid", "age"],
        ["source_type_id", "traffic_tier"],
        {"n_estimators": 10, "verbose": -1},
    )
    X = _frame()
    y = _target()

    validation = X.iloc[6:].copy()
    validation.loc[:, "source_type_id"] = ["unseen_a", "unseen_b"]

    result = models.fit_lightgbm_with_early_stopping(
        pipeline,
        X.iloc[:6],
        y.iloc[:6],
        validation,
        y.iloc[6:],
        stopping_rounds=2,
        eval_metric="binary_logloss",
    )

    transformed_validation = pipeline.named_steps["preprocessor"].transform(validation)
    assert transformed_validation["source_type_id"].isna().all()
    assert result["best_iteration"] >= 1
