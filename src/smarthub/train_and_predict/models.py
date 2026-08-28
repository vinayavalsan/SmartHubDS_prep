"""Model builders for SmartHub win-probability training.

This module creates preprocessing pipelines and supported classifier families.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)


class NativeCategoricalFrameTransformer(BaseEstimator, TransformerMixin):
    """Prepare model inputs for native categorical tree-model handling.

    Inputs
    ------
    numeric_features : list[str] | tuple[str, ...]
        Numeric feature names.
    categorical_features : list[str] | tuple[str, ...]
        Categorical feature names.
    """

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    @property
    def feature_names(self) -> list[str]:
        """Return model-input columns in stable order.

        Returns
        -------
        list[str]
            Numeric feature names followed by categorical feature names.
        """
        return list(self.numeric_features) + list(self.categorical_features)

    @staticmethod
    def _categorical_strings(series: pd.Series) -> pd.Series:
        """Normalize one categorical feature series.

        Inputs
        ------
        series : pandas.Series
            Categorical feature values.

        Returns
        -------
        pandas.Series
            String-normalized categorical values.
        """
        return series.astype("string").str.strip()

    def fit(self, X, y=None):
        """Learn categorical vocabularies from the fit partition.

        Inputs
        ------
        X : pandas.DataFrame
            Model feature matrix.
        y : Any, optional
            Target values. Accepted for sklearn compatibility and otherwise unused.

        Returns
        -------
        NativeCategoricalFrameTransformer
            Fitted transformer.

        Raises
        ------
        ValueError
            If a configured model feature is missing from the input frame.
        """
        frame = pd.DataFrame(X).copy()
        missing = [
            column for column in self.feature_names if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "Model input is missing registered feature column(s): " f"{missing}."
            )

        self.categories_ = {}
        for column in self.categorical_features:
            values = self._categorical_strings(frame[column])
            self.categories_[column] = sorted(values.dropna().unique().tolist())

        self.feature_names_in_ = list(self.feature_names)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        """Apply fitted numeric dtypes and categorical vocabularies.

        Inputs
        ------
        X : pandas.DataFrame
            Model feature matrix.

        Returns
        -------
        pandas.DataFrame
            Model-ready dataframe with native categorical dtypes.

        Raises
        ------
        ValueError
            If a fitted model feature is missing from the input frame.
        """
        check_is_fitted(self, attributes=["categories_", "feature_names_in_"])

        frame = pd.DataFrame(X).copy()
        missing = [
            column for column in self.feature_names_in_ if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "Model input is missing registered feature column(s): " f"{missing}."
            )

        out = frame[self.feature_names_in_].copy()

        for column in self.numeric_features:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

        for column in self.categorical_features:
            values = self._categorical_strings(out[column])
            out[column] = pd.Categorical(
                values,
                categories=self.categories_[column],
            )

        return out

    def get_feature_names_out(self, input_features=None):
        """Return fitted model-input feature names.

        Inputs
        ------
        input_features : Any, optional
            Accepted for sklearn compatibility and otherwise unused.

        Returns
        -------
        numpy.ndarray
            Fitted model-input feature names.
        """
        check_is_fitted(self, attributes=["feature_names_in_"])
        return pd.Index(self.feature_names_in_, dtype="object").to_numpy()


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


def _native_tree_preprocessor(numeric_features, categorical_features):
    """Build the native-categorical tree-model preprocessor.

    Inputs
    ------
    numeric_features : list[str]
        Numeric feature names.
    categorical_features : list[str]
        Categorical feature names.

    Returns
    -------
    NativeCategoricalFrameTransformer
        Unfitted native-categorical preprocessor.
    """
    return NativeCategoricalFrameTransformer(
        numeric_features=tuple(numeric_features),
        categorical_features=tuple(categorical_features),
    )


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

    Raises
    ------
    ValueError
        If native categorical handling is explicitly disabled.
    """
    from xgboost import XGBClassifier

    ordered = list(numeric_features) + list(categorical_features)
    monotone = tuple(1 if col == "bid" else 0 for col in ordered)

    params = dict(model_params)
    configured_enable_categorical = params.pop("enable_categorical", True)
    if configured_enable_categorical is not True:
        raise ValueError(
            "SmartHub XGBoost models require enable_categorical=True because "
            "registry categorical features are modeled as unordered categories."
        )

    preprocessor = _native_tree_preprocessor(numeric_features, categorical_features)
    classifier = XGBClassifier(
        enable_categorical=True,
        monotone_constraints=monotone,
        **params,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


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

    ordered = list(numeric_features) + list(categorical_features)
    monotone = [1 if col == "bid" else 0 for col in ordered]

    preprocessor = _native_tree_preprocessor(numeric_features, categorical_features)
    classifier = LGBMClassifier(monotone_constraints=monotone, **model_params)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


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

    Inputs
    ------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted LightGBM pipeline.
    X_fit : pandas.DataFrame
        Model-fitting feature matrix.
    y_fit : pandas.Series
        Model-fitting target values.
    X_validation : pandas.DataFrame
        Early-stopping validation feature matrix.
    y_validation : pandas.Series
        Early-stopping validation target values.
    stopping_rounds : int
        Number of rounds without improvement before stopping.
    eval_metric : str
        LightGBM evaluation metric.

    Returns
    -------
    dict
        Best iteration and early-stopping diagnostics.
    """
    from lightgbm import early_stopping, log_evaluation, record_evaluation

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_fit = preprocessor.fit_transform(X_fit, y_fit)
    transformed_validation = preprocessor.transform(X_validation)

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
