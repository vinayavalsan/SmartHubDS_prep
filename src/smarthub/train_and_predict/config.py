"""Configuration for Anton model training and prediction.

Config tiers (team decision — Kiran/Vinaya):
- **Secrets / DB connection** -> ``.env``.
- **Business settings** -> Postgres config store, edited in the Streamlit Config
  page (``target_cm``, ``bid_floor``, …). Read here via ``target_cm_value()`` /
  ``min_bid_value()``.
- **Task configs** -> ``config/smarthub.ini`` (``smarthub.core.task_config``),
  ``[training]`` / ``[prediction]`` sections. Read here for model_type, bid_step,
  calibration, etc. Code constants below are the fallback defaults.

Feature definitions come from ``smarthub.feature_engineering.features`` (single
source of truth), not this module.
"""

from __future__ import annotations

import os

from smarthub.core import paths, task_config
from smarthub.feature_engineering import features as fe

# Target + revenue come from the feature_engineering contract.
TARGET_COL = fe.TARGET_COLUMN          # "won_flag"
REVENUE_COL = fe.REVENUE_COLUMN        # "expected_revenue"


def lead_type_name(lead_type_id: int) -> str:
    """Human name for a lead type id (auto / home / ...)."""
    return fe.lead_type_name(lead_type_id)


def feature_columns(lead_type_id: int) -> tuple[list[str], list[str]]:
    """(numeric, categorical) model feature names for a lead type."""
    return fe.model_feature_columns(lead_type_id)


# =============================================================================
# TASK configs — from config/smarthub.ini ([training] / [prediction]).
# The constants are the fallback defaults if the ini/key is absent.
# =============================================================================
MODEL_TYPE = "lightgbm"                 # fallback; ini [training] model_type wins
RANDOM_SEED = task_config.get_int("training", "random_seed", 42)
DROP_ZERO_VARIANCE = task_config.get_bool("training", "drop_zero_variance", True)
CALIBRATE = task_config.get_bool("training", "calibrate", True)
TEST_SIZE = task_config.get_float("training", "test_size", 0.20)
BID_STEP = task_config.get_float("prediction", "bid_step", 0.25)

# Promotion gate (see `registry.decide_promotion`). A newly trained model
# ("challenger") only replaces the model currently serving traffic if it
# clears both checks below, evaluated on the SAME held-out test set. Loosen/
# tighten per lead type behaviour via the ini, not code.
PROMOTION_MIN_ROC_AUC_REGRESSION = task_config.get_float(
    "training", "promotion_min_roc_auc_regression", 0.01
)
PROMOTION_MIN_PROFIT_RATIO = task_config.get_float(
    "training", "promotion_min_profit_ratio", 0.98
)


def active_model_version() -> str | None:
    """Explicit per-lead-type pin (ini ``[prediction] active_model_version``,
    e.g. ``"v3_2026-07-09T140501Z"``) to serve a specific version instead of
    whatever is currently promoted. ``"none"`` (default) = serve that model.
    """
    raw = task_config.get("prediction", "active_model_version", "none")
    raw = (raw or "none").strip()
    return None if raw.lower() == "none" else raw


def model_type() -> str:
    """Model family (ini ``[training] model_type``, else MODEL_TYPE)."""
    return task_config.get("training", "model_type", MODEL_TYPE)


# Logistic Regression settings — `penalty` omitted (default 'l2') to avoid the
# sklearn 1.8 deprecation warning.
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": RANDOM_SEED,
}

# LightGBM settings. A monotonic-increasing constraint on `bid` is added by the
# model builder so P(win) rises with the bid — required for the optimizer.
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


# =============================================================================
# BUSINESS settings — from the Postgres config store (Streamlit Config page).
# Fall back to the constants if the store is unreachable/unset.
# =============================================================================
TARGET_CM = 0.25   # fallback for business `target_cm`
MIN_BID = 0.25     # fallback for business `bid_floor`
_CONFIG_ENV = os.getenv("SMARTHUB_CONFIG_ENV", "prod")


def _store(key, default):
    try:
        from smarthub.core.config_store import ConfigStore

        return ConfigStore().get(key, env=_CONFIG_ENV)
    except Exception:  # noqa: BLE001 - store down/unset -> use code default
        return default


def target_cm_value() -> float:
    """Target contribution margin (business `target_cm`, else TARGET_CM)."""
    return _store("target_cm", TARGET_CM)


def min_bid_value() -> float:
    """Minimum bid (business `bid_floor`, else MIN_BID)."""
    return _store("bid_floor", MIN_BID)


# --- Output locations (under data/, beside pulls + training tables) ----------
REPORT_ROOT = "data/training_report"
MODEL_ROOT = "data/models"


def report_dir(lead_type_name: str) -> str:
    """Absolute per-lead-type training-report directory."""
    return str(paths.resolve(f"{REPORT_ROOT}/{lead_type_name}"))


def model_path(lead_type_name: str) -> str:
    """Legacy single-file model path (pre-versioning).

    No longer written by ``train.py`` — models are now versioned under
    ``data/models/<type>/v<N>_<timestamp>.pkl`` with a currently-serving
    model per lead type; see ``registry.py``. Kept only so an old artifact at
    this path can still be loaded manually via ``MODEL_URI`` if one exists.
    """
    return str(paths.resolve(f"{MODEL_ROOT}/anton_model_{lead_type_name}.pkl"))


MLFLOW_EXPERIMENT_NAME = "anton_win_probability"
MLFLOW_RUN_NAME = "anton"
MLFLOW_REGISTERED_MODEL_NAME = "anton-win-probability-model"
