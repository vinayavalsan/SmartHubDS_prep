# `train_and_predict/` — analysis & integration notes

Analysis of the Anton model layer (`src/smarthub/train_and_predict/`), a
colleague's first cut at training the win-probability model and recommending
bids. This doc records **what the module does**, the **issues found**, and the
**changes applied** to make it integrate with the rest of the package.

---

## What it is

STEP 3 of the pipeline (after `data-pull` → `build-features`). It trains
`P(won | bid, lead features)`, evaluates it, runs an offline bid-optimizer
evaluation, saves the model + reports, logs to MLflow, and serves bid
recommendations over a FastAPI endpoint.

**Model:** scikit-learn Logistic Regression inside a `Pipeline`
(median-impute + standardize numerics; most-frequent-impute + one-hot
categoricals via `ColumnTransformer`). `bid` **is** a feature (so the optimizer
can sweep it); `expected_revenue` is **not** a feature (used only in the profit
objective) — matching MODELING.md.

**Bid optimizer:** for a lead, sweep candidate bids from `MIN_BID` up to
`max_bid = expected_revenue × (1 − target_cm)` in `BID_STEP` steps, predict win
rate at each, and pick the bid maximizing
`expected_profit = P(win) × (expected_revenue − bid)`.

**Modules:** `config` (settings), `preprocessing` (data prep), `models`
(pipeline builder), `metrics` (numeric evaluation), `plots_and_reports`
(artifacts), `mlflow_utils` (tracking), `train` (orchestration),
`predict` (optimizer + API), `flow` (Prefect), `manual_api_check` (client).

---

## Strengths

- Clean single-responsibility split across modules; readable `train` pipeline.
- Correct optimizer math and correct leakage discipline around
  `expected_revenue`.
- Thoughtful **offline** optimizer evaluation (predicted profit / win-rate lift,
  bid-change direction breakdown, CM distribution) with an honest caveat that it
  is predicted, not measured, lift.
- Shared cleaning intended for both training and serving (train/serve parity).

---

## Issues found (original state)

1. **Broken import against the current package.** `train.py` did
   `import smarthub.io`, but `io` lives at `smarthub.core.io`. It would fail
   immediately.
2. **Not an installable subpackage.** No `__init__.py`; flat sibling imports
   (`import config`, `import metrics`, …) only resolve when run from inside the
   folder. It could not be imported as `smarthub.train_and_predict.*`.
3. **README CLI not implemented.** `train.py` imported `argparse` but never
   parsed args; it read a hard-coded `config.LEAD_TYPE_ID = 6` and raised for any
   other type. `--lead_type_id 1` (home) did nothing.
4. **All logic ran at module top level** — no `main()` guard; importing `train`
   executed a full run.
5. **Bypassed the feature pipeline.** It re-cleaned `load_leads()` instead of
   consuming the leakage-safe training tables from `feature_engineering`,
   creating **two divergent feature definitions** (raw `age`, `marital_status`,
   … vs. your `age_cohort_*`, `is_married`, `multi_vehicle`, `won_flag`).
6. **Random train/test split** on a temporal problem (look-ahead leakage,
   optimistic metrics).
7. **ML dependencies undeclared** in `pyproject.toml`.
8. **`test_predict.py` would break `pytest`** (collected as a test, made a live
   HTTP call at import).
9. Minor: duplicate `recommended_bid_change.png` plot; CWD-relative output
   paths; `mlflow.log_model(name=...)` keyword varies by MLflow version;
   `iterrows()` + per-row `predict_proba` is slow at scale.

---

## Changes applied

**(a) This document.**

**(b) Integration-ready.**
- Added `__init__.py`; converted sibling imports to relative (`from . import …`)
  and fixed `smarthub.io` → `smarthub.core.io`.
- `train.py` refactored into a reusable `run_training(...)` + a `main()` with a
  real `--lead-type-id` (6=auto, 1=home) and `--version` / `--no-mlflow` flags;
  execution no longer happens at import.
- Added an `ml` extra to `pyproject.toml`
  (`scikit-learn, mlflow, matplotlib, joblib, fastapi, uvicorn, pydantic,
  requests`); the worker image installs `.[orchestration,ml]`.
- `predict.py` made import-safe: `joblib`/`mlflow` are imported lazily and the
  FastAPI app is guarded, so the pure optimizer functions work (and are tested)
  without the full extra.
- Renamed `test_predict.py` → `manual_api_check.py` so pytest won't run it.
- Removed the duplicate plot; output paths now resolve under `data/` and are
  **per lead type** (`data/models/anton_model_<type>.pkl`).

**(c) Reconciled feature pipelines.** `feature_engineering.features` is now the
**single source of truth**: it exposes `model_feature_columns(lead_type_id)`
(numeric/categorical split, auto-only features dropped for home),
`add_time_features`, and `derive_serving_features` (the train/serve parity hook,
reused by `build_training_table`). Training now loads the versioned training
table via `io.load_training_table` — no re-cleaning — and the API derives
features the same way. Target is `won_flag`. The split is **time-ordered**
(most-recent tail held out).

**New Prefect flow (STEP 3).** `train_and_predict/flow.py:train_flow` trains +
runs the offline optimizer, with `log_prints=True` (all pipeline output shows in
the run logs), a markdown artifact, a Slack success notification, and the shared
`flow_failure_hook` for failures. Registered in `prefect.yaml` as `train-model`
(two schedules: auto/home) on a new `training` work queue.

---

## Still open / future

- **Model choice:** LR is the baseline; XGBoost/LightGBM slot into `models.py`.
- **Calibration:** current "calibration error" is a global bias, not ECE;
  consider isotonic/Platt calibration for trustworthy probabilities feeding the
  optimizer.
- **Optimizer performance:** vectorize the per-row candidate scoring for large
  test sets.
- **MLflow `log_model` keyword** may need to match the installed MLflow version.
- **`expected_revenue`** still comes from the interim SUM-over-selected-listings
  logic; switch to the backend column when it lands (CONTEXT §4).
