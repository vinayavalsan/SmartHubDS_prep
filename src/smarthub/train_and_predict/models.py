"""Model builders for SmartHub win-probability training.

This module creates preprocessing pipelines and supported classifier families.
"""

from __future__ import annotations

import warnings

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)

# sklearn/LightGBM interoperability warning:
# CalibratedClassifierCV may call a fitted LGBMClassifier with an ndarray even
# when LightGBM recorded feature names during fit. Our pipeline fixes feature
# order explicitly through ColumnTransformer and the warning does not indicate
# a schema mismatch. Suppress only this exact warning; all other warnings remain.
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, "
    "but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def _to_numpy_array(value):
    """Return model input as a NumPy-compatible array."""
    return value.to_numpy() if hasattr(value, "to_numpy") else value


def build_logistic_regression_model(
    numeric_features,
    categorical_features,
    model_params,
):
    """Build an unfitted logistic-regression pipeline.

    Inputs
    ------
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.
    model_params : dict
        Parameters passed to the classifier.
    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted classifier pipeline.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

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
    return pipeline


def build_xgboost_model(
    numeric_features,
    categorical_features,
    model_params,
):
    """Build an unfitted XGBoost pipeline.

    Inputs
    ------
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.
    model_params : dict
        Parameters passed to the classifier.
    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted classifier pipeline.
    """
    from sklearn.preprocessing import OrdinalEncoder
    from xgboost import XGBClassifier

    numeric_transformer = "passthrough"
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(numeric_features)),
            (
                "categorical",
                categorical_transformer,
                list(categorical_features),
            ),
        ]
    )

    ordered = list(numeric_features) + list(categorical_features)
    monotone = tuple(1 if col == "bid" else 0 for col in ordered)

    classifier = XGBClassifier(
        monotone_constraints=monotone,
        **model_params,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def build_lightgbm_model(
    numeric_features,
    categorical_features,
    model_params,
):
    """Build an unfitted LightGBM pipeline.

    Inputs
    ------
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.
    model_params : dict
        Parameters passed to the classifier.
    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted classifier pipeline.
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import OrdinalEncoder

    numeric_transformer = "passthrough"
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, list(numeric_features)),
            ("categorical", categorical_transformer, list(categorical_features)),
        ]
    )
    # Explicitly convert the transformed feature matrix to an array before
    # LightGBM. This keeps fit and predict input types identical even when a
    # global sklearn setting requests pandas transformer output.
    to_numpy = FunctionTransformer(
        _to_numpy_array,
        validate=False,
    )

    # ColumnTransformer output order = numeric block then categorical block,
    # one column each -> aligns with this ordered list, so the constraint vector
    # matches the transformed matrix.
    ordered = list(numeric_features) + list(categorical_features)
    monotone = [1 if col == "bid" else 0 for col in ordered]

    classifier = LGBMClassifier(monotone_constraints=monotone, **model_params)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("to_numpy", to_numpy),
            ("classifier", classifier),
        ]
    )
    return pipeline


def _maybe_calibrate(pipeline, calibration_enabled, calibration_method, calibration_cv):
    """Apply the configured probability calibration when enabled.

    Inputs
    ------
    pipeline : Any
        Classifier pipeline to calibrate.
    calibration_enabled : bool
        Whether to apply probability calibration.
    calibration_method : str
        Calibration method passed to ``CalibratedClassifierCV``. Supported
        values are ``sigmoid`` and ``isotonic``.
    calibration_cv : int
        Number of calibration cross-validation folds.

    Returns
    -------
    Any
        Original pipeline or calibrated classifier.
    """
    if not calibration_enabled:
        return pipeline
    from sklearn.calibration import CalibratedClassifierCV

    # Positional estimator arg works across sklearn versions
    # (estimator= / base_estimator=). Isotonic is monotonic, so it preserves
    # the bid monotonicity above.
    return CalibratedClassifierCV(
        pipeline, method=calibration_method, cv=calibration_cv
    )


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression_model,
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
}


def build_model(
    model_type,
    numeric_features,
    categorical_features,
    model_params,
    *,
    calibration_enabled,
    calibration_method,
    calibration_cv,
):
    """Build the configured classifier pipeline.

    Inputs
    ------
    model_type : str
        Configured model family name.
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.
    model_params : dict
        Parameters passed to the classifier.
    calibration_enabled : bool
        Whether to apply probability calibration.
    calibration_method : str
        Calibration method passed to ``CalibratedClassifierCV``. Supported
        values are ``sigmoid`` and ``isotonic``.
    calibration_cv : int
        Number of calibration cross-validation folds.

    Returns
    -------
    Any
        Unfitted configured classifier pipeline.

    Raises
    ------
    ValueError
        If the configured model family is unsupported.
    """
    try:
        builder = MODEL_BUILDERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Supported: {supported}."
        ) from exc

    pipeline = builder(
        numeric_features,
        categorical_features,
        model_params,
    )
    return _maybe_calibrate(
        pipeline,
        calibration_enabled=calibration_enabled,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
    )
