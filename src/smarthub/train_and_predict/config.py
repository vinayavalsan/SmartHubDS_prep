"""
Configuration constants for Anton model training and prediction.
"""

LEAD_TYPE_ID = 6

RANDOM_SEED = 42

TARGET_COL = "won"


COMMON_FEATURES = {
    # Continuous
    "bid": "continuous",
    "age": "continuous",
    # Discrete
    "continuous_coverage_months": "discrete",
    "created_hour": "discrete",
    "created_dayofweek": "discrete",
    # Categorical
    "campaign_id": "categorical",
    "lead_type_id": "categorical",
    "source_type_id": "categorical",
    "account_id": "categorical",
    "state": "categorical",
    "insured": "categorical",
    "gender": "categorical",
    "marital_status": "categorical",
    "military_affiliation": "categorical",
}

AUTO_FEATURES = {
    "num_vehicles": "discrete",
    "num_drivers": "discrete",
    "num_auto_violations": "discrete",
    "num_auto_accidents": "discrete",
    "home_owner": "categorical",
    "dui": "categorical",
}

if LEAD_TYPE_ID == 6:
    FEATURES = {
        **COMMON_FEATURES,
        **AUTO_FEATURES,
    }
else:
    raise ValueError(f"Lead type {LEAD_TYPE_ID} is not configured.")

CONTINUOUS_FEATURES = [
    feature
    for feature, feature_type in FEATURES.items()
    if feature_type == "continuous"
]

DISCRETE_FEATURES = [
    feature for feature, feature_type in FEATURES.items() if feature_type == "discrete"
]

CATEGORICAL_FEATURES = [
    feature
    for feature, feature_type in FEATURES.items()
    if feature_type == "categorical"
]

FEATURE_COLS = CONTINUOUS_FEATURES + DISCRETE_FEATURES + CATEGORICAL_FEATURES


# Logistic Regression model settings.
# These values are passed directly into sklearn.linear_model.LogisticRegression.
LOGISTIC_REGRESSION_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": RANDOM_SEED,
}


# Bid optimizer settings.
# Target CM = 25% means max_bid = expected_revenue * (1 - 0.25).
TARGET_CM = 0.25
MIN_BID = 0.25
BID_STEP = 0.25

REPORT_DIR = "training_report"
MODEL_DIR = "models"
LOCAL_MODEL_PATH = f"{MODEL_DIR}/anton_model.pkl"

MLFLOW_EXPERIMENT_NAME = "anton_win_probability"
MLFLOW_RUN_NAME = "logistic_regression_v1"
MLFLOW_REGISTERED_MODEL_NAME = "anton-win-probability-model"
