# SmartHub Train and Predict

## Overview

The SmartHub machine-learning code is organized around three separate workflows:

1. **Model training** — trains, evaluates, versions, and optionally promotes a model.
2. **Hyperparameter search** — runs a manual Optuna search and produces recommended parameters for a later training run.
3. **Production prediction** — loads the selected serving model, evaluates candidate bids, returns a recommendation, and records the prediction.

Hyperparameter search is an optional preparation step. A model can be trained directly from the parameters already defined in `config/training.yaml`, or the search workflow can first produce improved parameters that are copied into `training.yaml` before training.

The model predicts the probability of winning a lead at a given bid. The bid optimizer then evaluates candidate bids and selects the bid with the highest predicted expected profit while respecting the configured contribution-margin ceiling.

---

## Folder structure

```text
src/smarthub/
├── train_and_predict/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── metrics.py
│   ├── train.py
│   ├── flow.py
│   ├── hyperparameter_search.py
│   ├── optimizer.py
│   ├── optimizer_evaluation.py
│   ├── registry.py
│   ├── mlflow_utils.py
│   ├── plots_and_reports.py
│   ├── explain.py
│   ├── prediction_log_schema.py
│   └── manual_api_check.py
│
└── server/
    └── predict.py

config/
├── training.yaml
├── hyperparameter_search.yaml
└── smarthub.yaml

data/
├── models/
├── model_evaluations/
├── hyperparameter_tuning/
└── mlflow/
```

The exact data paths are configuration-driven and may differ by environment.

---

## High-level architecture

```text
                         Training table
                               │
               ┌───────────────┴────────────────┐
               │                                │
               │ Optional                       │ Direct training
               ▼                                │
      Hyperparameter search                     │
               │                                │
               ▼                                │
     Best parameter recommendation              │
               │                                │
               └──── copy selected values ──────┘
                               │
                               ▼
                         Model training
                               │
                               ▼
                         Model evaluation
                               │
                               ▼
                  Offline optimizer evaluation
                               │
                               ▼
                     Model version is saved
                               │
                               ▼
                    Promotion policy evaluated
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             Keep current model     Promote challenger
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Serving-model registry
                               │
                               ▼
                 Production prediction service
                  `src/smarthub/server/predict.py`
```

Important points:

- Hyperparameter search does not train or promote the production model.
- Hyperparameter search only recommends parameter values.
- Training can run without performing hyperparameter search first.
- Production prediction is separate from both training and hyperparameter search.

---

# Configuration files

## `config/training.yaml`

Used by the model-training workflow.

Typical sections include:

```yaml
training:
  model_type: lightgbm
  random_seed: 42
  drop_zero_variance: true

calibration:
  method: isotonic  # isotonic, sigmoid, or none
  cv: 3

split:
  strategy: time
  time:
    test_size: 0.20
  random:
    test_size: 0.20
    stratify: true

models:
  logistic_regression: {}
  xgboost: {}
  lightgbm: {}

optimizer:
  target_cm: 0.25
  minimum_bid: 0.25
  bid_step: 0.25
  chunk_size: 500

promotion:
  mode: automatic

  criteria:
    max_log_loss: 0.55
    min_expected_profit: 0.0
    min_profit_ratio: 0.95
    max_absolute_profit_loss_tolerance: 5000.0
    max_log_loss_regression: 0.01

output:
  report_root: data/model_evaluations
  model_root: data/models

mlflow:
  tracking_db_path: data/mlflow/mlflow.db
  artifact_root: data/mlflow/artifacts
  experiment_name: anton_win_probability
  registered_model_name: anton-win-probability-model
```

The actual model parameters belong under the selected model in the `models` section. Only the configuration for `training.model_type` is loaded for the run.

The loader also stores resolved metadata in memory, including the source config path, selected split, calibration settings, selected model parameters, optimizer settings, and promotion criteria. The optional environment variable `SMARTHUB_TRAINING_CONFIG` can override the packaged `config/training.yaml` path.

### Main responsibilities

- Select the model family.
- Define the train/test split.
- Configure probability calibration using `isotonic`, `sigmoid`, or `none`.
- Define the parameters passed to the selected estimator.
- Configure candidate-bid generation and offline scoring chunk size.
- Define promotion mode and promotion guardrails.
- Define report, model-registry, and MLflow locations.

---

## `config/hyperparameter_search.yaml`

Used only by the manual hyperparameter-search workflow.

Typical structure:

```yaml
search:
  model_type: lightgbm
  scoring: neg_log_loss
  n_trials: 50
  cv_folds: 3
  timeout_seconds: null
  random_seed: 42
  n_jobs: 1

models:
  lightgbm:
    fixed_parameters: {}
    search_space:
      n_estimators:
        type: int
        low: 100
        high: 800
      learning_rate:
        type: float
        low: 0.01
        high: 0.20
        log: true

output:
  root: data/hyperparameter_tuning
```

### Main responsibilities

- Select one model family to tune.
- Define the cross-validation scoring metric.
- Define fixed parameters and the Optuna search space.
- Define trial count, CV folds, seed, timeout, and parallelism.
- Define where tuning results are written.

The default scoring metric should remain:

```yaml
scoring: neg_log_loss
```

Scikit-learn returns negative Log Loss because its scorer convention assumes that larger values are better. Optuna therefore maximizes the cross-validation score while effectively minimizing Log Loss.

The search output includes a copy-paste-ready `best_parameters.yaml`. Review those values before copying them into `config/training.yaml`.

---

## `config/smarthub.yaml`

Used by production and operational workflows, including the real-time prediction service in:

```text
src/smarthub/server/predict.py
```

Depending on the deployed version, this file can contain settings for:

- the explicitly pinned serving-model version,
- prediction behavior,
- cold-start fallback behavior,
- scheduled exploration,
- model-data recency checks,
- prediction logging,
- explainability and local LLM settings,
- monitoring and other production runtime configuration.

Training-only choices belong in `training.yaml`, not `smarthub.yaml`. Hyperparameter-search choices belong in `hyperparameter_search.yaml`.

---

# Workflow 1: Model training

## Purpose

The training workflow:

1. loads a versioned training table,
2. prepares model features,
3. creates the train/test split,
4. trains the configured model,
5. evaluates probability and classification metrics,
6. evaluates the bid optimizer on held-out rows,
7. writes reports,
8. compares the challenger with the currently serving model,
9. saves an immutable model version,
10. optionally promotes it,
11. optionally logs the run to MLflow.

## Entry points

```text
src/smarthub/train_and_predict/train.py
src/smarthub/train_and_predict/flow.py
```

`train.py` contains the main training implementation and command-line entry point. `flow.py` wraps the training operation in a Prefect flow and publishes run summaries and notifications.

## Relevant scripts

### `train.py`

Coordinates the end-to-end training workflow.

Main responsibilities:

- Load `TrainingConfig`.
- Load and prepare the training table.
- Split data into training and test sets.
- Remove zero-variance features when enabled.
- Build and fit the selected model.
- Evaluate the model.
- Run offline optimizer evaluation.
- Generate reports.
- Re-score the serving model on the same challenger test rows.
- Request a promotion decision from `registry.py`.
- Save the challenger as a new registry version.
- Promote automatically when configured and eligible.
- Log the run to MLflow.

### `flow.py`

Prefect orchestration for training.

It:

- invokes `train.run_training`,
- records a Prefect artifact,
- summarizes model quality and optimizer results,
- sends success or failure notifications.

### `preprocessing.py`

Prepares training and serving data.

It:

- loads the requested training-table version,
- resolves numeric and categorical feature columns,
- normalizes data types,
- adds missing configured columns,
- retains the target, expected revenue, and lineage fields,
- sorts time-based training data,
- validates that both target classes are present,
- removes zero-variance columns when requested,
- produces model-ready serving frames.

### `models.py`

Builds supported estimator pipelines:

- Logistic Regression
- XGBoost
- LightGBM

The model pipeline includes preprocessing and the classifier. XGBoost and LightGBM apply a monotonic constraint to `bid`, ensuring predicted win probability does not decrease as bid increases.

When `calibration.method` is `isotonic` or `sigmoid`, the pipeline is wrapped with `CalibratedClassifierCV` using the configured number of folds. When the method is `none`, the base pipeline is returned without calibration.

### `metrics.py`

Calculates held-out model metrics.

Probability-quality metrics:

- Log Loss
- Brier Score
- calibration error

Ranking metrics:

- ROC AUC
- PR AUC

Threshold-based diagnostics at the current `0.5` threshold:

- Accuracy
- Precision
- Recall
- F1
- F2

F2 weights recall more heavily than precision. It is retained as a diagnostic metric and is not a direct promotion gate.

### `optimizer.py`

Contains the reusable bid-optimization logic.

For each lead, it:

1. calculates the maximum permitted bid:

   ```text
   max_bid = expected_revenue × (1 - target_cm)
   ```

2. creates candidate bids from `minimum_bid` through `max_bid`, using `bid_step`,
3. obtains a predicted win probability for each candidate,
4. calculates expected profit:

   ```text
   expected_profit = predicted_win_probability × (expected_revenue - bid)
   ```

5. returns the candidate with the highest expected profit.

The module supports both one-row real-time optimization and chunked offline evaluation.

### `optimizer_evaluation.py`

Evaluates optimizer behavior on held-out historical rows.

It compares:

- expected profit at the historical bid,
- expected profit at the recommended bid,
- predicted win rate at each bid,
- bid increases, decreases, and unchanged bids,
- average and median bid change,
- recommended contribution margin if won.

Its summary is used for reports, MLflow, and promotion evaluation.

### `plots_and_reports.py`

Writes training and optimizer reports, including:

```text
feature_summary.csv
feature_value_counts.csv
roc_curve.png
precision_recall_curve.png
calibration_curve.png
probability_histogram.png
confusion_matrix.png
model_evaluation_summary.json
optimizer_expected_profit_lift.png
recommended_bid_change.png
recommended_cm_distribution.png
current_vs_recommended_win_rate.png
bid_optimizer_test_rows.csv
```

The exact set depends on whether optimizer evaluation completed successfully.

### `registry.py`

Manages immutable model versions and the currently serving model pointer.

Example layout:

```text
data/models/auto/
├── current.json
├── run_20260727T162136Z_a1b2c3d4.pkl
├── run_20260727T162136Z_a1b2c3d4.json
├── run_20260728T101500Z_e5f6a7b8.pkl
└── run_20260728T101500Z_e5f6a7b8.json
```

Every saved candidate receives an immutable `training_run_id` such as
`run_20260727T162136Z_a1b2c3d4`. Only a promoted candidate receives a
sequential production version such as `auto_v1`, `auto_v2`, and so on.

Each manifest records:

- training run ID and optional production model version,
- feature columns,
- model metrics,
- optimizer summary,
- training parameters,
- data and model lineage,
- promotion mode,
- eligibility and promotion status,
- promotion comparison details and reason,
- MLflow identifiers,
- promotion timestamp when promoted.

`current.json` points to the currently serving training run and its assigned production version. A manual promotion is allowed only for a candidate whose saved `eligibility_status` is `eligible`.

### `mlflow_utils.py`

Logs the training run to MLflow.

It records:

- feature list,
- model parameters,
- model metrics including F2,
- optimizer metrics,
- lineage parameters and lifecycle tags,
- generated report artifacts,
- serialized model artifacts.

Model registration is performed only when a run is promoted. Automatic and manual promotion use the same idempotent MLflow registration path, so promoting the same training run again does not create a duplicate registered-model version.

---

# Model-promotion policy

## Promotion modes

The `promotion.mode` setting supports:

### `manual`

- Evaluate the challenger.
- Save the model and its eligibility decision.
- Do not change `current.json` automatically.
- A user may promote the saved version later with the registry command.

### `automatic`

- Evaluate the challenger.
- Save the challenger.
- Automatically promote it only when the policy returns an eligible decision.

### `disabled`

- Skip promotion comparison.
- Save the model version.
- Leave the currently serving model unchanged.

## Promotion criteria

The promotion policy is business-driven. Every challenger is first evaluated independently and, when a serving model exists, compared with the currently serving model.

When a serving model exists, it is re-evaluated on the challenger's **same held-out test rows**, using the challenger's optimizer settings. This keeps the Log Loss and expected-profit comparisons on identical data.

### Absolute promotion gates

Every challenger must satisfy all of the following:

- Log Loss ≤ `max_log_loss`
- Total recommended expected profit ≥ `min_expected_profit`
- Average recommended contribution margin ≥ `optimizer.target_cm`

These gates apply even when no serving model exists.

`max_log_loss`, `min_expected_profit`, and `max_absolute_profit_loss_tolerance` have centralized defaults in `config.py` when omitted. Other required promotion values are read from YAML and validated.

### Relative promotion gates

When a serving model exists, the challenger must also satisfy all of the following:

- Profit ratio ≥ `min_profit_ratio`
- Absolute profit loss ≤ `max_absolute_profit_loss_tolerance`
- Log Loss regression ≤ `max_log_loss_regression`

where:

```text
profit_ratio =
challenger_expected_profit /
serving_expected_profit
```

```text
absolute_profit_loss =
serving_expected_profit -
challenger_expected_profit
```

A challenger is eligible only if **all** configured promotion criteria pass. Missing challenger Log Loss, expected profit, or recommended CM causes failure. For an existing serving model, missing serving Log Loss or expected profit also causes failure, and serving expected profit must be greater than zero for the relative comparison.

### Diagnostic metrics

The following are logged and reported but do not independently determine promotion:

- ROC AUC
- PR AUC
- Accuracy
- Precision
- Recall
- F1
- F2
- Brier Score
- calibration error

When no serving model exists, only the absolute promotion gates are evaluated. In `automatic` mode, the first model is promoted only if those gates pass. In `manual` mode, an eligible first model is saved as `awaiting_manual_promotion` until the registry command is run.

---

# Workflow 2: Hyperparameter search

## Purpose

Hyperparameter search is a separate, manually triggered experimentation workflow. It searches for parameter values that improve cross-validated probability quality.

It does not:

- save a production model version,
- modify `current.json`,
- promote a model,
- change `training.yaml` automatically.

## Entry point

```text
src/smarthub/train_and_predict/hyperparameter_search.py
```

## Relevant scripts

### `hyperparameter_search.py`

It:

1. loads `config/hyperparameter_search.yaml`,
2. loads and prepares the training table,
3. removes zero-variance features,
4. builds stratified cross-validation folds,
5. asks Optuna to select candidate parameters,
6. evaluates each trial using the configured scorer,
7. writes a run summary,
8. writes copy-paste-ready best parameters,
9. copies the search YAML used for the run,
10. writes interactive Optuna history, importance, and contour plots when visualization succeeds.

### Shared scripts

The workflow reuses:

- `config.py`
- `preprocessing.py`
- `models.py`

This ensures the search evaluates the same feature preparation and model construction used by official training.

## Output

Typical output layout:

```text
data/hyperparameter_tuning/
└── auto/
    └── lightgbm/
        └── 20260723_143000/
            ├── summary.json
            ├── best_parameters.yaml
            ├── hyperparameter_search.yaml
            └── plots/
                ├── optimization_history.html
                ├── parameter_importance.html
                └── contour_matrix.html
```

`best_parameters.yaml` is a recommendation. Review it and copy the selected model block into `config/training.yaml` before starting the official training workflow.

## Relationship to training

```text
Option A: train directly

training.yaml parameters
        │
        ▼
     train.py


Option B: tune, then train

hyperparameter_search.yaml
        │
        ▼
hyperparameter_search.py
        │
        ▼
best_parameters.yaml
        │
        ▼
review and copy into training.yaml
        │
        ▼
     train.py
```

---

# Workflow 3: Production prediction

## Purpose

The production workflow returns bid recommendations for individual leads in real time.

The production API entry point is:

```text
src/smarthub/server/predict.py
```

This is intentionally outside the `train_and_predict` package because it is part of the production server application. It imports reusable training-package components rather than running training logic.

## Relevant scripts

### `src/smarthub/server/predict.py`

The production server is responsible for:

- exposing the FastAPI endpoints,
- resolving and loading the serving model and manifest,
- validating incoming requests,
- applying serving policy,
- invoking bid optimization,
- handling cold-start or exploration behavior when configured,
- returning the bid recommendation,
- writing the prediction audit log,
- scheduling non-blocking explainability work when supported.

### `train_and_predict/optimizer.py`

Generates and evaluates candidate bids using the loaded serving model.

### `train_and_predict/preprocessing.py`

Transforms the incoming lead record into the exact feature structure expected by the model.

### `train_and_predict/registry.py`

Resolves the currently serving model and loads its manifest.

### `train_and_predict/config.py`

Exposes shared configuration helpers and serving-model selection helpers.

### `train_and_predict/prediction_log_schema.py`

Defines the single-table prediction log and its read/write interface.

One row is written for each prediction or explanation call, whether it succeeds or fails. The row contains request metadata, model lineage, input-feature snapshots, optimizer results, decision details, and optional SHAP data.

### `train_and_predict/explain.py`

Provides slower, on-demand explanation functionality. It is not responsible for selecting or training the production model.

It can:

- calculate SHAP contributions for supported tree models,
- identify the most influential features for one lead,
- build nearby bid-curve facts,
- optionally use a local Ollama model to turn structured numeric facts into a short explanation.

### `train_and_predict/manual_api_check.py`

Provides a small sample client for manually testing the recommendation endpoint.

---

# Basic run commands

Run commands from the repository root with the project environment activated.

## Train a model

Using the installed CLI:

```bash
smarthub-train --lead-type-id 6
```

Using the Python module directly:

```bash
python -m smarthub.train_and_predict.train --lead-type-id 6
```

Train a specific training-table version:

```bash
python -m smarthub.train_and_predict.train \
  --lead-type-id 6 \
  --version <training-table-version>
```

Skip MLflow for a local diagnostic run:

```bash
python -m smarthub.train_and_predict.train \
  --lead-type-id 6 \
  --no-mlflow
```

## Run the Prefect training flow

The exact deployment command depends on the configured Prefect deployment. For direct local execution:

```bash
python -m smarthub.train_and_predict.flow
```

The module's direct execution currently defaults to Auto, `lead_type_id=6`.

## Run hyperparameter search

Using the installed CLI:

```bash
smarthub-hyperparameter-search --lead-type-id 6
```

Using the Python module directly:

```bash
python -m smarthub.train_and_predict.hyperparameter_search \
  --lead-type-id 6
```

Use an alternate search configuration:

```bash
python -m smarthub.train_and_predict.hyperparameter_search \
  --lead-type-id 6 \
  --config config/hyperparameter_search.yaml
```

Tune against a specific training-table version:

```bash
python -m smarthub.train_and_predict.hyperparameter_search \
  --lead-type-id 6 \
  --version <training-table-version>
```

## Start the production prediction API

The production application lives in `src/smarthub/server/predict.py`:

```bash
uvicorn smarthub.server.predict:app --host 0.0.0.0 --port 8000
```

For local development with automatic reload:

```bash
uvicorn smarthub.server.predict:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

## Check API health

```bash
curl "http://127.0.0.1:8000/health?lead_type_id=6"
```

## Test a bid request

```bash
python -m smarthub.train_and_predict.manual_api_check
```

## Manually promote a saved model

```bash
python -m smarthub.train_and_predict.registry promote \
  --lead-type-name auto \
  --version <model-version> \
  --reason "manual review approved"
```

---

# Script reference

| Script | Workflow | Responsibility | Main configuration |
|---|---|---|---|
| `train_and_predict/config.py` | Training, search | Loads and validates training and search YAML and resolves selected model/split settings | `training.yaml`, `hyperparameter_search.yaml` |
| `train_and_predict/preprocessing.py` | Shared | Creates normalized training and serving frames | Feature definitions and training config |
| `train_and_predict/models.py` | Training, search | Builds Logistic Regression, XGBoost, and LightGBM pipelines | `training.yaml`, `hyperparameter_search.yaml` |
| `train_and_predict/metrics.py` | Training | Computes Log Loss, Brier, calibration, ranking, F1, and F2 metrics | None directly |
| `train_and_predict/train.py` | Training | Runs the complete model-training and promotion workflow | `training.yaml` |
| `train_and_predict/flow.py` | Training | Prefect orchestration, artifact, and notifications | `training.yaml` and operational notification config |
| `train_and_predict/hyperparameter_search.py` | Search | Runs Optuna cross-validation and writes best parameters | `hyperparameter_search.yaml` |
| `train_and_predict/optimizer.py` | Training, prediction | Generates candidate bids and selects maximum expected profit | Optimizer values from training or serving request/config |
| `train_and_predict/optimizer_evaluation.py` | Training | Compares current and recommended bids on held-out rows | `training.yaml` |
| `train_and_predict/registry.py` | Training, prediction | Saves versions, manages serving pointer, promotes, and rolls back | Promotion settings supplied by training |
| `train_and_predict/mlflow_utils.py` | Training | Logs model, metrics, parameters, and reports | `training.yaml` |
| `train_and_predict/plots_and_reports.py` | Training | Writes plots, CSVs, and JSON evaluation reports | Output paths from `training.yaml` |
| `train_and_predict/prediction_log_schema.py` | Prediction | Defines and writes the single-table prediction audit log | `SMARTHUB_PREDICTION_LOG_DB_URL` plus values supplied by serving |
| `train_and_predict/explain.py` | Prediction support | Produces SHAP and optional Ollama explanations | `smarthub.yaml` explain settings |
| `train_and_predict/manual_api_check.py` | Prediction support | Sends a sample request to the local API | URL and sample payload in the script |
| `server/predict.py` | Prediction | Production FastAPI application and serving-policy entry point | `smarthub.yaml` |

---

# Design principles

- Training, hyperparameter search, and prediction are separate workflows.
- Hyperparameter search is optional and never changes production by itself.
- Official model parameters come from `training.yaml`.
- Every completed training run saves an immutable candidate identified by a `training_run_id`.
- Only promoted candidates receive sequential production versions such as `auto_v1`.
- Production loads the model selected by the registry or explicit production pin.
- The model estimates probabilities for bidding. The `0.5` class threshold is used only for diagnostic classification metrics and plots.
- Log Loss is the primary probability-quality objective for tuning.
- Every challenger must satisfy absolute business and model-quality thresholds.
- Existing serving models are protected by profit-ratio, absolute-profit-loss, and Log Loss regression guardrails.
- Promotion requires both probability-quality and business guardrails; no single metric can bypass the others.
- Log Loss is the primary probability-quality guardrail.
- F2, PR AUC, ROC AUC, F1, precision, and recall remain diagnostics.
- Production prediction does not retrain models.
- Prediction logs preserve the model, configuration, inputs, and optimizer decision needed for auditing and debugging.
