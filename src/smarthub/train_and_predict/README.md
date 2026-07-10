# SmartHub Train and Predict

This package trains Anton win-probability models, evaluates model quality, runs
offline bid-optimizer evaluation, versions models, controls promotion, and serves
bid recommendations.

## Workflow

The package is Step 3 of the SmartHub data-science pipeline:

1. `data_pull` creates the raw lead dataset.
2. `feature_engineering` creates versioned training tables.
3. `train_and_predict`:
   - loads a training-table version;
   - prepares model features;
   - splits the data into training and test sets;
   - trains the configured classifier;
   - evaluates predicted win probability;
   - evaluates the bid optimizer on held-out rows;
   - writes reports and model artifacts;
   - evaluates promotion eligibility;
   - optionally logs the run to MLflow.

## Supported models

The training configuration supports:

- Logistic regression
- XGBoost
- LightGBM

XGBoost and LightGBM apply a positive monotonic constraint to `bid`, so the
predicted win probability cannot decrease when the bid increases while all
other inputs remain unchanged.

Probability calibration can be enabled in `training.yaml`. The current
implementation uses isotonic calibration. If calibration fails during training,
the pipeline logs a warning and refits the model without calibration.

## Configuration

The primary configuration file is:

```text
config/training.yaml
```

Set `SMARTHUB_TRAINING_CONFIG` to use a different file.

The configuration defines:

- selected model and model parameters;
- random seed;
- calibration and zero-variance handling;
- random or time-based train/test splitting;
- optimizer target contribution margin, minimum bid, and bid step;
- promotion mode and thresholds;
- report and model directories;
- MLflow tracking, artifact, experiment, and registered-model settings.

Supported promotion modes are:

- `manual`: evaluate eligibility, save the model, and wait for an explicit
  promotion command;
- `automatic`: evaluate eligibility and automatically promote an eligible
  model;
- `disabled`: save the model without evaluating promotion eligibility.

## Training

Run training through the installed command:

```bash
smarthub-train --lead-type-id 6
```

Supported lead type IDs currently include:

- `6`: Auto
- `1`: Home
- `5`: Commercial

Use a specific versioned training table:

```bash
smarthub-train --lead-type-id 6 --version <training-table-version>
```

Skip MLflow logging and registration:

```bash
smarthub-train --lead-type-id 6 --no-mlflow
```

The Prefect flow calls the same training workflow and publishes a markdown
artifact containing model, optimizer, lineage, and promotion information.

## Data preparation and missing data

`preprocessing.prepare_training_data` loads the requested training table and
uses the feature definitions from `feature_engineering.features`.

The preparation rules are:

- The target column is required. Training stops with an error when it is
  missing.
- Rows with a missing or nonnumeric target are removed.
- Expected revenue is retained when available because it is required for the
  offline optimizer evaluation, but it is not a win-probability model feature.
- `created_at` is retained when available and is used to sort rows before a
  time-based split.
- Configured feature columns that are completely absent from the training table
  are added as missing columns and listed in `missing_feature_columns`.
- Numeric values are converted with `pandas.to_numeric(..., errors="coerce")`.
  Invalid values therefore become missing.
- Blank or missing categorical values are normalized to `NAvail`.
- Features with zero observed variance may be removed after the train/test
  split when `drop_zero_variance` is enabled.

Missing values that remain after normalization are handled inside each model
pipeline:

- Logistic regression: numeric median imputation and categorical
  most-frequent imputation, followed by scaling and one-hot encoding.
- XGBoost and LightGBM: numeric median imputation and categorical
  most-frequent imputation, followed by ordinal encoding.
- Previously unseen categorical values are supported during prediction.

The same normalization rules are applied when constructing the serving frame.
This keeps training and online prediction inputs consistent.

## Train/test split

The split strategy is selected in `training.yaml`.

### Random split

The dataframe is shuffled using the configured random seed. Optional target
stratification can preserve the overall win-rate distribution.

### Time split

The prepared dataframe is sorted by `created_at`. The newest configured
fraction becomes the test set and the earlier rows become the training set.

The pipeline raises an error if the selected split produces an empty training
or test dataset.

## Model evaluation

The held-out test set produces:

- ROC AUC
- PR AUC
- log loss
- Brier score
- accuracy
- precision
- recall
- F1 score
- observed win rate
- average predicted win rate at historical bids
- calibration error
- average predicted win rate at recommended bids, when optimizer evaluation is
  available

The classification threshold used for accuracy, precision, recall, F1, and the
confusion matrix is `0.5`.

## Bid optimizer

The optimizer evaluates candidate bids from the configured minimum bid through:

```text
maximum bid = expected revenue * (1 - target CM)
```

Candidate bids are separated by the configured bid step. For each candidate:

```text
expected profit = predicted win probability * (expected revenue - bid)
```

The candidate with the largest expected profit is selected.

Offline optimizer evaluation compares the historical bid with the recommended
bid on held-out rows. It reports:

- current and recommended total expected profit;
- total and percentage expected-profit lift;
- average predicted win rates at current and recommended bids;
- average and median bid change;
- percentages of bids increased, decreased, and unchanged;
- average recommended contribution margin if the lead is won.

These values are model-based offline estimates. They are not realized production
results.

## Reports

Reports are written under the configured report root by lead type. Generated
artifacts include:

```text
feature_summary.csv
feature_value_counts.csv
model_evaluation_summary.json
optimizer_evaluation.csv                 # when optimizer evaluation is available
roc_curve.png
precision_recall_curve.png
calibration_curve.png
probability_histogram.png
confusion_matrix.png
optimizer_expected_profit_lift.png       # when available
recommended_bid_change.png               # when available
recommended_cm_distribution.png          # when available
current_vs_recommended_win_rate.png       # when available
```

`feature_summary.csv` includes feature type, missing count, missing percentage,
number of unique values, mode, and numeric summary statistics where applicable.
`feature_value_counts.csv` contains the most frequent values for discrete and
categorical features.

## Model versioning and promotion

Every completed training run saves an immutable model artifact and JSON
manifest under the configured model directory. The manifest records:

- model version and creation time;
- feature columns;
- model metrics;
- optimizer summary;
- training-data lineage;
- model parameters;
- promotion mode, eligibility, decision reason, and promotion status.

The serving model is identified by a `current.json` pointer for each lead type.
Training compares a challenger with the currently-serving model when promotion
evaluation is enabled. The currently-serving model is re-scored on the new
run's held-out test set so the comparison uses the same rows.

In manual mode, promote a saved version with:

```bash
python -m smarthub.train_and_predict.registry promote \
  --lead-type-name auto \
  --version <model-version> \
  --reason "approved after review"
```

Promotion updates the selected version manifest and the serving pointer. The
registry module also supports programmatic rollback to a previous saved
version.

## MLflow

When MLflow logging is enabled, the training workflow logs:

- model parameters and selected features;
- model evaluation metrics;
- retained optimizer metrics;
- lineage and configuration values;
- report artifacts;
- the fitted sklearn-compatible model;
- an optional registered-model version.

The experiment artifact location must match the configured artifact root. The
pipeline raises an error rather than silently writing an existing experiment to
a different location.

## Hyperparameter search

Manual Optuna search is available separately from the official training run:

```bash
python -m smarthub.train_and_predict.hyperparameter_search \
  --lead-type-id 6 \
  --model-type lightgbm
```

Optional arguments:

```text
--version <training-table-version>
--config <hyperparameter-search-yaml>
```

The search writes:

```text
summary.json
best_parameters.yaml
```

The generated YAML block is intended to be reviewed and copied into the normal
training configuration. Hyperparameter search does not automatically replace
the production training configuration or promote a model.

## Prediction API

Start the FastAPI application using the project deployment command or an ASGI
server pointed at:

```text
smarthub.train_and_predict.predict:app
```

Endpoints:

- `GET /health`: returns service status and the resolved model artifact.
- `POST /recommend_bid`: validates lead and optimizer inputs, loads the serving
  model, and returns the expected-profit-maximizing bid.

Model resolution order is:

1. `MODEL_URI` environment variable;
2. explicitly pinned `prediction.active_model_version`;
3. the currently-serving registry pointer.

The request supplies `expected_revenue`, optimizer controls, and lead features.
The API initializes the row with the minimum bid and then evaluates the complete
candidate-bid grid.

A sample request is available in `manual_api_check.py`.

## Explainability

`explain.py` is an optional, on-demand workflow and is not part of the live bid
path. It currently supports LightGBM models and uses SHAP to identify the
features that most affected one lead's prediction.

SHAP contributions are calculated in the underlying model's log-odds space.
The optional local LLM only converts the supplied numeric facts into plain
language; it does not calculate the recommendation or model contribution.

Heavy explainability dependencies are imported lazily so normal training and
serving can run without the explainability extras installed.

## Main modules

- `train.py`: end-to-end training workflow and CLI.
- `preprocessing.py`: training and serving input preparation.
- `models.py`: sklearn-compatible model pipelines.
- `metrics.py`: held-out model evaluation.
- `optimizer.py`: candidate-bid generation and optimization.
- `optimizer_evaluation.py`: offline optimizer comparison and summary.
- `plots_and_reports.py`: report tables, plots, and saved summaries.
- `registry.py`: immutable model versions, promotion, serving pointer, rollback.
- `mlflow_utils.py`: MLflow experiment and artifact logging.
- `hyperparameter_search.py`: manual Optuna search.
- `predict.py`: serving model resolution and FastAPI endpoints.
- `flow.py`: Prefect orchestration, artifacts, and notifications.
- `explain.py`: optional LightGBM/SHAP explanation workflow.
