# SmartHub DS - ML Training and Prediction Folder

## Overview

This folder contains the machine learning training and prediction components for the SmartHub Data Science project. Its primary objective is to train a model that predicts the probability of winning a lead at a given bid and to use that model to recommend the bid that maximizes expected profit.

This is **not** the complete SmartHub project. Instead, it is a self-contained module within the larger SmartHub codebase that focuses specifically on model development, evaluation, and prediction.

The current structure is intended as a **preliminary starting point**. As the project matures and new requirements emerge, the organization of this folder may evolve. Modules may be added, merged, or split to better support maintainability, testing, deployment, and future functionality. The goal of the current structure is simply to establish a clean separation of responsibilities while the project is still under active development.

---

## Folder Structure

```text
ml_training_prediction/

├── train.py
├── predict.py
├── preprocessing.py
├── models.py
├── metrics.py
├── plots_and_reports.py
├── mlflow_utils.py
├── config.py

├── models/
│   └── anton_model.pkl

├── model_evaluation/

├── mlruns/

└── data/
```

---

## Overall Workflow

Training workflow:

```text
Training Data
      │
      ▼
Preprocessing
      │
      ▼
Train ML Model
      │
      ▼
Evaluate Model
      │
      ▼
Run Bid Optimization Evaluation
      │
      ▼
Generate Reports
      │
      ▼
Save Model
      │
      ▼
Log Training Run to MLflow
```

Prediction workflow:

```text
Incoming Lead
      │
      ▼
Predict Win Probability
      │
      ▼
Generate Candidate Bids
      │
      ▼
Predict Win Probability
for Every Candidate Bid
      │
      ▼
Calculate Expected Profit
      │
      ▼
Recommend Optimal Bid
```

---

## Usage

Train a model for a specific lead type by passing `lead_type_id` as a command-line argument.

For Auto leads:

```bash
python train.py --lead_type_id 6
```

For Home leads:

```bash
python train.py --lead_type_id 1
```

The training script passes this value into `preprocessing.clean_training_data()`, so the preprocessing module no longer hard-codes a specific lead type.

## Module Description

### `train.py`

The main entry point for model training.

This script orchestrates the complete training workflow while keeping implementation details inside the supporting modules. It is intentionally designed to be easy to read and should describe the overall pipeline rather than contain complex logic at this stage of development; by the MVP phase, this design may evolve.

Responsibilities include:

- Load training data
- Accept `lead_type_id` as a command-line argument
- Clean and prepare data for the requested lead type
- Generate dataset summaries
- Split train/test data
- Train the machine learning model
- Evaluate model performance
- Run bid optimization evaluation
- Generate reports and plots
- Save the trained model
- Log the training run to MLflow

---

### `predict.py`

Contains the prediction and bid optimization logic.

Responsibilities include:

- Loading the trained model
- FastAPI prediction endpoint
- Bid optimization logic
- Bid recommendation
- Offline bid optimizer evaluation used by the training script

The same bid optimization functions are reused during training to evaluate how much improvement the optimizer predicts over the current bids.

The bid optimizer follows this process:

```text
Lead Features
      │
      ▼
Generate Candidate Bids
      │
      ▼
Predict Win Rate for Each Candidate Bid
      │
      ▼
Calculate Expected Profit
      │
      ▼
Choose Candidate Bid with Highest Expected Profit
```

Expected profit is calculated as:

```text
Expected Profit = Predicted Win Rate × (Expected Revenue - Bid)
```

The offline optimizer evaluation now reports additional metrics, including:

- Expected profit using the current bid
- Expected profit using the recommended bid
- Expected profit lift
- Expected profit lift percentage
- Predicted win-rate lift
- Average, median, 10th percentile, and 90th percentile bid changes
- Bid increase / decrease / unchanged counts and percentages
- Average number of candidate bids evaluated
- Selected bid percentile
- Recommended CM distribution
- Direction-level summaries for increased, decreased, and unchanged bids

Important note: these are **offline predicted optimizer metrics**. They compare the current bid to the recommended bid using the trained model's predicted win rates. They do not represent measured production lift.

---

### `preprocessing.py`

Contains all preprocessing and feature engineering logic.

Examples include:

- Filtering to the requested lead type passed from `train.py`
- Missing value handling
- Business default values
- Target conversion
- Feature preparation

No model training or prediction logic should exist in this module.

---

### `models.py`

Defines the machine learning models used by the project.

Currently this contains the baseline Logistic Regression model.

As additional models are introduced, such as XGBoost or LightGBM, they can be added here without changing the rest of the training pipeline.

---

### `metrics.py`

Computes numerical evaluation metrics for the trained model.

Examples include:

- ROC AUC
- PR AUC
- Log Loss
- Brier Score
- Accuracy
- Precision
- Recall
- F1 Score
- Calibration Error

This module focuses only on numerical evaluation and does not generate plots.

---

### `plots_and_reports.py`

Generates all reports produced during training.

This includes:

Dataset summaries:

- Feature statistics
- Missing value summaries
- Feature value distributions

Model evaluation:

- ROC Curve
- Precision-Recall Curve
- Calibration Curve
- Probability Histogram
- Confusion Matrix

Bid optimization evaluation:

- Expected profit improvement
- Recommended bid changes
- Predicted win-rate lift
- Recommended CM distribution
- Current vs recommended predicted win-rate scatter plot

Output files include:

```text
model_evaluation/

feature_summary.csv
feature_value_counts.csv
roc_curve.png
precision_recall_curve.png
calibration_curve.png
probability_histogram.png
confusion_matrix.png
optimizer_expected_profit_lift.png
recommended_bid_change.png
predicted_win_rate_lift.png
recommended_cm_distribution.png
current_vs_recommended_win_rate.png
bid_optimizer_test_rows.csv
model_evaluation_summary.json
```

The `model_evaluation/` directory stores three types of outputs:

- **Dataset summaries**: `feature_summary.csv` and `feature_value_counts.csv` describe the raw data before cleaning.
- **Model evaluation artifacts**: ROC, precision-recall, calibration, probability, and confusion-matrix outputs describe classifier quality.
- **Bid optimizer artifacts**: optimizer plots and `bid_optimizer_test_rows.csv` describe the offline predicted behavior of recommended bids versus historical bids.

---

### `mlflow_utils.py`

Provides helper functions for logging training runs to MLflow.

Responsibilities include:

- Creating experiments
- Starting runs
- Logging parameters
- Logging metrics
- Uploading reports
- Uploading trained models
- Registering model versions

Keeping all MLflow-related code in a single module keeps the training pipeline clean and easier to maintain.

---

### `config.py`

Stores project-wide configuration.

Examples include:

- Feature definitions
- Target variable
- Runtime training inputs such as `lead_type_id` are passed as script arguments, not hard-coded in `config.py`
- Random seed
- Bid optimization parameters
- Model paths
- Report paths
- MLflow configuration

No implementation logic should be placed in this file.

---

## Design Principles

The current organization follows several design principles:

- **Single responsibility:** Each module is responsible for one well-defined task.
- **Readability:** The training pipeline should be easy to follow without understanding every implementation detail.
- **Reusability:** Common functionality, such as bid optimization and reporting, should be reusable across training and production prediction.
- **Extensibility:** The structure should make it straightforward to introduce additional models, reports, evaluation metrics, or deployment methods.
- **Reproducibility:** Training runs should be reproducible through fixed configuration and MLflow experiment tracking.

As the SmartHub platform grows, this folder is expected to evolve alongside it. The current layout should be viewed as a foundation rather than a final architecture.
