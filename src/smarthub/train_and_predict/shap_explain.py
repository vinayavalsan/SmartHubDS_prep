"""SHAP-based factor breakdown for one lead's win-probability prediction.

Split out of ``explain.py`` (2026-07-24) — this module owns everything that
turns a fitted model + one lead's features into a ranked list of "why the
model predicted what it predicted" factors. ``explain.py`` remains the thin
orchestrator that calls into this module (for the numeric breakdown) and
``llm_explain.py`` (to turn that breakdown into plain English); no logic
changed in this split, only which file each piece lives in.

Heavy/optional deps (shap, lightgbm) are imported lazily so the rest of
`train_and_predict` keeps working without the `explain` extra installed —
same pattern as `predict.py`'s lazy joblib/mlflow/fastapi imports.
"""

from __future__ import annotations

import warnings

import numpy as np

from smarthub.core import task_config
from smarthub.core.logging_utils import get_logger

from . import config, preprocessing

logger = get_logger(__name__)

# Silence a harmless, repetitive SHAP notice on LightGBM binary classifiers
# ("... TreeExplainer shap values output has changed to a list of ndarray").
# It fires on EVERY shap_values() call and floods the shap-worker logs; we handle
# the output shape explicitly below, so the notice is pure noise. Scoped to this
# exact message so real UserWarnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r"LightGBM binary classifier with TreeExplainer",
    category=UserWarning,
)
# Task config: smarthub.yaml `explain` section — used only by the explain
# pipeline, not the live bidding path, so it's kept local rather than in
# train_and_predict/config.py.
TOP_N_FACTORS = task_config.get_int("explain", "top_n_factors", 5)

# Speed knob: how many calibration folds to explain. The dominant SHAP cost is
# building a TreeExplainer per fold; a calibrated model has cv=3 folds. Isotonic
# calibration is a monotonic rescaling that doesn't change which features matter
# or their ranking (see _fitted_lgbm_estimators' docstring), so explaining ONE
# fold is a sound, ~3x-cheaper default for the logged factors. Set to a large
# number (e.g. 99) to restore averaging across all folds for max fidelity.
SHAP_MAX_FOLDS = task_config.get_int("explain", "shap_max_folds", 1)

# Per-process cache of built SHAP explainers, keyed by the classifier object's
# id. Building the TreeExplainer is the expensive step and depends only on the
# fitted trees, not the row -- so build once per loaded model and reuse it for
# every row. (In the shap-worker, models are cached per version, so their
# classifiers stay alive and these explainers stay valid.)
_EXPLAINER_CACHE: dict = {}


def _get_explainer(classifier):
    """Return a cached ``shap.TreeExplainer`` for ``classifier`` (build once)."""
    import shap

    key = id(classifier)
    exp = _EXPLAINER_CACHE.get(key)
    if exp is None:
        exp = shap.TreeExplainer(classifier)
        _EXPLAINER_CACHE[key] = exp
    return exp


def _fitted_lgbm_estimators(model):
    """Return the fitted (preprocessor, LGBMClassifier) pair(s) inside ``model``.

    Handles both a plain sklearn ``Pipeline`` (preprocessor + classifier, see
    ``models.build_lightgbm_model``) and a ``CalibratedClassifierCV`` wrapping
    one — isotonic calibration is a monotonic rescaling of the final
    probability, so it doesn't change *which* features mattered or their
    ranking, only the final number; that's why explaining the underlying
    LightGBM model(s) is still valid even though the served model is
    calibrated. When calibrated, there's one fitted pipeline per CV fold
    (``models.py``'s ``cv=3``) — all are returned so callers can average.

    Raises
    ------
    ValueError
        If ``model`` isn't (or doesn't wrap) a LightGBM pipeline — SHAP
        explanations here only support ``model_type=lightgbm`` for now.
    """
    from lightgbm import LGBMClassifier

    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        pipelines = [
            getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            for cc in calibrated_classifiers
        ]
        pipelines = [p for p in pipelines if p is not None]
    else:
        pipelines = [model]

    pairs = []
    for pipe in pipelines:
        preprocessor = pipe.named_steps["preprocessor"]
        classifier = pipe.named_steps["classifier"]
        if not isinstance(classifier, LGBMClassifier):
            raise ValueError(
                "SHAP explanations currently support model_type='lightgbm' "
                f"only (got {type(classifier).__name__}). Train an LGBM "
                "model to use /explain_bid for this lead type."
            )
        pairs.append((preprocessor, classifier))

    if not pairs:
        raise ValueError("Could not find a fitted LightGBM estimator in this model.")
    return pairs


def _shap_for_row(model, row_frame, feature_cols):
    """SHAP values for one row, averaged across calibration folds if any.

    ``shap.TreeExplainer`` on an ``LGBMClassifier`` works in **margin
    (log-odds) space**, not probability space: raw feature contributions and
    ``explainer.expected_value`` are unbounded reals that sum to the model's
    raw score, not a 0-1 probability (e.g. a base value of ``-2.0`` is a
    normal log-odds figure — ``sigmoid(-2.0) ≈ 0.12``). Per-feature
    contributions are left in log-odds units (their *sign* and *relative
    magnitude* — which factor mattered most — are unaffected by the
    sigmoid's monotonicity, so ranking/direction stay valid), but the base
    value is converted to an actual probability here since callers use it as
    "the model's average predicted win rate".

    Returns
    -------
    tuple[dict[str, float], float]
        ``(feature -> log-odds shap contribution, base_win_rate)`` where
        ``base_win_rate`` IS a 0-1 probability — the model's average
        predicted win rate before this lead's specific factors are applied.
    """
    import numpy as np

    pairs = _fitted_lgbm_estimators(model)
    if SHAP_MAX_FOLDS and SHAP_MAX_FOLDS > 0:
        pairs = pairs[:SHAP_MAX_FOLDS]  # speed: explain fewer calibration folds
    per_fold_shap = []
    base_values = []
    for preprocessor, classifier in pairs:
        transformed = preprocessor.transform(row_frame)
        explainer = _get_explainer(classifier)  # cached; built once per model
        raw = explainer.shap_values(transformed)

        # SHAP's return shape for binary classifiers has changed across
        # versions: a [class0, class1] list (older), or a single array with a
        # trailing class axis (newer). Normalize to "one row of feature
        # contributions to the positive (win) class".
        if isinstance(raw, list):
            values = raw[1][0]
        elif np.asarray(raw).ndim == 3:
            values = np.asarray(raw)[0, :, 1]
        else:
            values = np.asarray(raw)[0]
        per_fold_shap.append(np.asarray(values, dtype=float))

        base = explainer.expected_value
        base = base[1] if isinstance(base, (list, np.ndarray)) else base
        base_values.append(float(base))

    avg_shap = np.mean(per_fold_shap, axis=0)
    avg_base_margin = float(np.mean(base_values))
    avg_base_win_rate = float(1.0 / (1.0 + np.exp(-avg_base_margin)))
    # SHAP additivity (log-odds): the model's margin for this row is the base
    # margin plus the sum of feature contributions. Convert to a probability so
    # callers get "the predicted win rate this explanation accounts for".
    predicted_margin = avg_base_margin + float(np.sum(avg_shap))
    predicted_win_rate = float(1.0 / (1.0 + np.exp(-predicted_margin)))
    return (
        dict(zip(feature_cols, avg_shap.tolist())),
        avg_base_win_rate,
        predicted_win_rate,
    )


def _to_native(value):
    """Convert a numpy scalar (int64/float64/bool_) to a plain Python type.

    ``frame.iloc[0][name]`` returns numpy scalars for numeric columns, and
    some FastAPI/pydantic version combinations can't JSON-encode those (seen
    in the wild as ``TypeError: 'numpy.int64' object is not iterable`` from
    ``jsonable_encoder``) — cast explicitly rather than depend on the
    installed encoder's numpy support.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rank_factors(shap_values: dict, frame, top_n: int) -> list[dict]:
    """Rank feature contributions by |shap| into the top-N factor dicts."""
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {
            "feature": name,
            "value": _to_native(frame.iloc[0][name]),
            "shap": round(value, 4),
            "direction": "increased" if value > 0 else "decreased",
        }
        for name, value in ranked[:top_n]
    ]


def _all_contributions(shap_values: dict, frame) -> list[dict]:
    """Every model feature's contribution, sorted by |contribution| desc.

    Unlike :func:`_rank_factors` this is **not** truncated -- the stored SHAP
    payload records the contribution of every feature the model scored (per the
    prediction-log SHAP spec), so a later step can reconstruct the full picture
    without recomputing SHAP. Contributions are in log-odds units, matching
    ``base_win_rate`` / ``base_prediction`` (a probability)."""
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {
            "feature": name,
            "value": _to_native(frame.iloc[0][name]),
            "contribution": round(value, 4),
        }
        for name, value in ranked
    ]


def explain_row(model, record, lead_type_id, top_n=None):
    """Build the structured 'why' facts for one lead from a **raw** record.

    Derives the model features from ``record`` (via ``serving_frame``) and then
    runs SHAP. Use this in the local/dev path where you start from a raw lead;
    the production path should instead reuse an already-computed prediction's
    ``model_input_features`` via :func:`explain_prepared_row` (no re-derivation).

    Inputs
    ------
    model : fitted sklearn Pipeline or CalibratedClassifierCV
        Trained Anton model (LightGBM only — see ``_fitted_lgbm_estimators``).
    record : dict
        Raw lead attributes (same shape as ``BidRequest`` in ``predict.py``).
    lead_type_id : int
        6=auto, 1=home — selects the model feature schema.
    top_n : int | None
        How many top factors to keep; ``TOP_N_FACTORS`` when ``None``.

    Returns
    -------
    dict
        ``{"top_factors": [...], "base_win_rate": float}``. Each factor is
        ``{"feature", "value", "shap", "direction"}``, sorted by |shap|.
    """
    top_n = top_n or TOP_N_FACTORS
    numeric, categorical = config.feature_columns(lead_type_id)
    feature_cols = list(numeric) + list(categorical)

    frame = preprocessing.serving_frame([record], lead_type_id)
    shap_values, base_value, predicted = _shap_for_row(model, frame, feature_cols)
    return {
        "base_prediction": base_value,
        "prediction": predicted,
        "feature_contributions": _all_contributions(shap_values, frame),
        "top_factors": _rank_factors(shap_values, frame, top_n),
        "base_win_rate": base_value,
    }


def explain_prepared_row(
    model, model_input_features, lead_type_id, recommended_bid=None, top_n=None
):
    """SHAP for an **already-prepared** model-input row — no re-derivation.

    Consumes the exact feature vector a prediction already scored (the
    ``model_input_features`` persisted in the prediction log), so the
    explanation corresponds to that prediction rather than a fresh
    re-computation. When ``recommended_bid`` is given, the ``bid`` feature is
    set to it, so SHAP reflects the chosen bid rather than whatever placeholder
    the logged row carried.

    Inputs
    ------
    model : fitted sklearn Pipeline or CalibratedClassifierCV
        The model that produced the prediction (LightGBM only).
    model_input_features : dict
        The prepared feature row (post-``serving_frame``) from the prediction.
    lead_type_id : int
        6=auto, 1=home.
    recommended_bid : float | None
        Overwrites the ``bid`` feature so SHAP is taken at the chosen bid.
    top_n : int | None
        How many top factors to keep; ``TOP_N_FACTORS`` when ``None``.

    Returns
    -------
    dict
        Same shape as :func:`explain_row`.
    """
    import pandas as pd

    top_n = top_n or TOP_N_FACTORS
    numeric, categorical = config.feature_columns(lead_type_id)
    feature_cols = list(numeric) + list(categorical)

    row = dict(model_input_features)
    if recommended_bid is not None and "bid" in feature_cols:
        row["bid"] = recommended_bid
    # Project to exactly the model's feature columns, in order (ignoring any
    # extra keys like id/created_at the caller may carry along). Numeric columns
    # are coerced back to numbers: a prediction reloaded from the log has its
    # model_input_features JSON-serialized, so numeric values arrive as strings.
    frame = pd.DataFrame([{col: row.get(col) for col in feature_cols}])
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[feature_cols]
    shap_values, base_value, predicted = _shap_for_row(model, frame, feature_cols)
    return {
        "base_prediction": base_value,
        "prediction": predicted,
        "feature_contributions": _all_contributions(shap_values, frame),
        "top_factors": _rank_factors(shap_values, frame, top_n),
        "base_win_rate": base_value,
    }
