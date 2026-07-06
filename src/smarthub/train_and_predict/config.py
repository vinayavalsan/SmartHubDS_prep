"""Configuration for Anton model training and prediction.

Feature definitions are NOT duplicated here — they come from
``smarthub.feature_engineering.features`` (the single source of truth shared by
the feature build, training, and serving). This module only holds model
hyper-parameters, the bid-optimizer settings, and output locations.
"""

from __future__ import annotations

import os

from smarthub.core import paths
from smarthub.feature_engineering import features as fe

RANDOM_SEED = 42

# Target + revenue come from the feature_engineering contract.
TARGET_COL = fe.TARGET_COLUMN          # "won_flag"
REVENUE_COL = fe.REVENUE_COLUMN        # "expected_revenue"


def lead_type_name(lead_type_id: int) -> str:
    """Human name for a lead type id (auto / home / ...)."""
    return fe.lead_type_name(lead_type_id)


def feature_columns(lead_type_id: int) -> tuple[list[str], list[str]]:
    """(numeric, categorical) model feature names for a lead type."""
    return fe.model_feature_columns(lead_type_id)


# Model family: "logistic_regression" | "lightgbm". Runtime-overridable from the
# Tier-2 config store (`model_type`); this is the fallback default.
MODEL_TYPE = "logistic_regression"

# Logistic Regression settings — passed straight into sklearn. `penalty` is
# omitted (its default is 'l2') to avoid the sklearn 1.8 deprecation warning.
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": RANDOM_SEED,
}

# LightGBM settings. A monotonic-increasing constraint on `bid` is added by the
# model builder (models.build_lightgbm_model), so P(win) rises with the bid —
# required for the bid optimizer to behave sensibly.
LIGHTGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}


# --- Runtime (Tier-2) knobs — editable from the Streamlit Config page ---------
# These read the shared Postgres config store when available and fall back to the
# constants above if the store is unreachable/unset, so training never breaks.
_CONFIG_ENV = os.getenv("SMARTHUB_CONFIG_ENV", "prod")


def _runtime(key, default):
    try:
        from smarthub.core.config_store import ConfigStore

        return ConfigStore().get(key, env=_CONFIG_ENV)
    except Exception:  # noqa: BLE001 - store down/unset -> use code default
        return default


def model_type() -> str:
    """Active model family (Tier-2 `model_type`, else MODEL_TYPE)."""
    return _runtime("model_type", MODEL_TYPE)


def target_cm_value() -> float:
    """Target contribution margin (Tier-2 `target_cm`, else TARGET_CM)."""
    return _runtime("target_cm", TARGET_CM)


def min_bid_value() -> float:
    """Minimum bid (Tier-2 `bid_floor`, else MIN_BID)."""
    return _runtime("bid_floor", MIN_BID)


# Drop feature columns with no variance (e.g. all-'false' insured/home_owner) at
# train time — cosmetic + removes dead one-hot columns.
DROP_ZERO_VARIANCE = True

# Wrap the classifier in probability calibration (isotonic) so predict_proba is
# trustworthy for the profit optimizer. ~3x slower predict_proba; falls back to
# uncalibrated automatically if calibration errors.
CALIBRATE = True

# Bid optimizer settings. Target CM = 25% -> max_bid = expected_revenue*(1-.25).
TARGET_CM = 0.25
MIN_BID = 0.25
BID_STEP = 0.25

# Test-split fraction (time-ordered split — most recent rows held out).
TEST_SIZE = 0.20

# Output locations (kept under data/ so they sit beside pulls + training tables).
REPORT_ROOT = "data/training_report"
MODEL_ROOT = "data/models"


def report_dir(lead_type_name: str) -> str:
    """Absolute per-lead-type training-report directory."""
    return str(paths.resolve(f"{REPORT_ROOT}/{lead_type_name}"))


def model_path(lead_type_name: str) -> str:
    """Absolute per-lead-type model file (so auto/home don't clobber)."""
    return str(paths.resolve(f"{MODEL_ROOT}/anton_model_{lead_type_name}.pkl"))


MLFLOW_EXPERIMENT_NAME = "anton_win_probability"
MLFLOW_RUN_NAME = "logistic_regression_v1"
MLFLOW_REGISTERED_MODEL_NAME = "anton-win-probability-model"
