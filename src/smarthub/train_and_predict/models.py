"""Model builders for SmartHub win-probability training.

This module creates preprocessing pipelines and supported classifier families.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
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
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def _maybe_calibrate(pipeline, calibrate, calibration_method, calibration_cv):
    """Apply the configured probability calibration when enabled.

    Inputs
    ------
    pipeline : Any
        Classifier pipeline to calibrate.
    calibrate : bool
        Whether to apply probability calibration.
    calibration_method : str
        Calibration method passed to ``CalibratedClassifierCV``.
    calibration_cv : int
        Number of calibration cross-validation folds.

    Returns
    -------
    Any
        Original pipeline or calibrated classifier.
    """
    if not calibrate:
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
    calibrate,
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
    calibrate : bool
        Whether to apply probability calibration.
    calibration_method : str
        Calibration method passed to ``CalibratedClassifierCV``.
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
        calibrate=calibrate,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
    )
