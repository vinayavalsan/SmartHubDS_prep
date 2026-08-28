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


def fit_lightgbm_with_early_stopping(
    pipeline,
    X_fit,
    y_fit,
    X_validation,
    y_validation,
    stopping_rounds,
    eval_metric,
):
    """Fit a raw LightGBM pipeline and return its best boosting iteration.

    The validation frame is transformed with the preprocessor fitted only on
    ``X_fit``. It is used solely to choose the boosting iteration and is never
    used to fit tree splits or leaf values.
    """
    from lightgbm import early_stopping, log_evaluation, record_evaluation

    preprocessor = pipeline.named_steps["preprocessor"]
    to_numpy = pipeline.named_steps["to_numpy"]
    classifier = pipeline.named_steps["classifier"]

    transformed_fit = preprocessor.fit_transform(X_fit, y_fit)
    transformed_fit = to_numpy.fit_transform(transformed_fit, y_fit)
    transformed_validation = preprocessor.transform(X_validation)
    transformed_validation = to_numpy.transform(transformed_validation)

    evaluation_history = {}
    classifier.fit(
        transformed_fit,
        y_fit,
        eval_set=[(transformed_validation, y_validation)],
        eval_metric=eval_metric,
        callbacks=[
            early_stopping(int(stopping_rounds), verbose=False),
            record_evaluation(evaluation_history),
            log_evaluation(period=0),
        ],
    )

    max_estimators = int(classifier.get_params()["n_estimators"])
    best_iteration = int(getattr(classifier, "best_iteration_", 0) or 0)
    if best_iteration <= 0:
        best_iteration = max_estimators

    validation_history = evaluation_history.get("valid_0", {})
    metric_history = validation_history.get(eval_metric)
    if metric_history is None and validation_history:
        metric_history = next(iter(validation_history.values()))
    metric_history = list(metric_history or [])

    stopped_iteration = len(metric_history) or best_iteration
    best_score = None
    best_scores = getattr(classifier, "best_score_", {}) or {}
    validation_scores = best_scores.get("valid_0", {})
    if eval_metric in validation_scores:
        best_score = float(validation_scores[eval_metric])
    elif metric_history and 0 < best_iteration <= len(metric_history):
        best_score = float(metric_history[best_iteration - 1])

    return {
        "best_iteration": best_iteration,
        "stopped_iteration": int(stopped_iteration),
        "best_score": best_score,
        "stopped_early": int(stopped_iteration) < max_estimators,
    }


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
