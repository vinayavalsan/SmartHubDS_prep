"""Model builders for Anton.

The feature lists are passed in (from ``feature_engineering.model_feature_columns``)
rather than imported, so the same builder serves auto and home models.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_regression_model(
    numeric_features, categorical_features, model_params, calibrate=False
):
    """Build the baseline Anton win-probability model (a full sklearn Pipeline).

    Numeric: median-impute + standardize. Categorical: most-frequent-impute +
    one-hot (unknown categories ignored at predict time). When ``calibrate`` is
    set, the whole pipeline is wrapped in isotonic probability calibration so
    ``predict_proba`` is well-calibrated for the profit optimizer.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(numeric_features)),
            ("categorical", categorical_transformer, list(categorical_features)),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**model_params)),
        ]
    )
    return _maybe_calibrate(pipeline, calibrate)


def build_lightgbm_model(
    numeric_features, categorical_features, model_params, calibrate=False
):
    """Build a LightGBM win-probability model with a monotonic bid constraint.

    Numeric features are imputed; categoricals are imputed + ordinal-encoded
    (kept one column each so the transformed matrix order matches the feature
    order). ``P(win)`` is constrained to be **non-decreasing in `bid`** — without
    this a tree can predict a lower win probability at a higher bid, which breaks
    the bid optimizer.
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import OrdinalEncoder

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(numeric_features)),
            ("categorical", categorical_transformer, list(categorical_features)),
        ]
    )

    # ColumnTransformer output order = numeric block then categorical block,
    # one column each -> aligns with this ordered list, so the constraint vector
    # matches the transformed matrix.
    ordered = list(numeric_features) + list(categorical_features)
    monotone = [1 if col == "bid" else 0 for col in ordered]

    classifier = LGBMClassifier(monotone_constraints=monotone, **model_params)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )
    return _maybe_calibrate(pipeline, calibrate)


def _maybe_calibrate(pipeline, calibrate):
    """Wrap in isotonic probability calibration when requested."""
    if not calibrate:
        return pipeline
    from sklearn.calibration import CalibratedClassifierCV

    # Positional estimator arg works across sklearn versions
    # (estimator= / base_estimator=). Isotonic is monotonic, so it preserves
    # the bid monotonicity above.
    return CalibratedClassifierCV(pipeline, method="isotonic", cv=3)


def build_model(model_type, numeric_features, categorical_features, model_params,
                calibrate=False):
    """Dispatch to the requested model family."""
    if model_type == "logistic_regression":
        return build_logistic_regression_model(
            numeric_features, categorical_features, model_params, calibrate
        )
    if model_type == "lightgbm":
        return build_lightgbm_model(
            numeric_features, categorical_features, model_params, calibrate
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")
