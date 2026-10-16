# SmartHub (Anton) — Developer Documentation

> A practical guide to the SmartHub bid-recommendation system: what it does, how the
> pieces fit together, how each pipeline works, how to configure it, and how to run it
> locally. Aimed at a developer joining the project — read Section 1 first (≈10–15 min)
> for the whole picture, then dip into the rest as reference.

**Contents**

1. [System Overview](#1-system-overview)
2. [Pipeline Documentation](#2-pipeline-documentation)
3. [Production Automation](#3-production-automation)
4. [Configuration](#4-configuration)
5. [Getting Started (Local Development)](#5-getting-started-local-development)

---

## 1. System Overview

### 1.1 What SmartHub is

SmartHub (codename **Anton**) recommends how much to **bid** for an incoming sales lead so
that the business buys leads it is likely to win while keeping a target profit margin. For
each lead it predicts the probability of winning the auction at different bid levels —
`P(won | bid, lead features)` — and then picks the bid that maximises expected profit.

The system has two halves that deliberately stay separate:

- **Offline (training):** scheduled jobs that pull historical lead data, build a training
  table, train and evaluate a model, and promote the best one into a versioned registry.
- **Online (serving):** a continuously-running API that loads the promoted model and
  answers real-time `recommend a bid` requests in under a second, logging every decision.

Everything the online service needs is published to a shared **model registry**; the two
halves never call each other directly. This means training can run, fail, or be re-run
without ever affecting live bidding, and the live API keeps serving the last promoted
model regardless of what the offline jobs are doing.

### 1.2 Major components

| Component | Role |
|---|---|
| **FastAPI serve app** (`smarthub.server`) | The live bid API (`/recommend_bid`, `/explain_bid`, `/health`). Loads promoted models, decides the bid, logs every call. |
| **nginx** | The only publicly-exposed entry point. Reverse-proxies to the serve app with rate limits and tight timeouts. |
| **Prefect** | Orchestrates the offline pipeline (data-pull → build-features → train-model) on cron schedules via a worker + work pool. |
| **Postgres** | One shared database behind everything: Prefect state, the prediction log, the config store, API keys, and MLflow's backend store. |
| **MinIO / S3** | The model registry's object store — where promoted models and the "currently serving" pointer live. |
| **MLflow** | Experiment tracking for training runs (metrics, params, artifacts). Read-only UI. |
| **Streamlit dashboards** | A monitoring app (Leads / Monitoring / Config pages) and a separate model-diagnostics app. |
| **Ollama** | A small local LLM that turns model factors into a plain-English "why this bid" explanation. |
| **shap-worker** | Background worker that computes SHAP factor breakdowns off the request path. |
| **slo-alerts** | Background loop that watches API health and posts Slack alerts. |

### 1.3 End-to-end data flow

```
 Redshift (lead_pings)
        │  (1) data-pull  — SSH tunnel, rolling window, upsert on id
        ▼
 Local storage: DuckDB + Parquet (data/raw_datasets/)
        │  (2) build-features — leakage-safe training table
        ▼
 data/training_datasets/<type>/<version>.parquet
        │  (3) train-model — fit, evaluate, optimise bids
        ▼
 Promotion gate ──► Model registry (MinIO/S3): <type>/current.json  ◄─┐
        │                                                              │
        ▼                                                              │
 (5) serve (FastAPI) loads promoted model, decides bid                 │ reads pointer
        │                                                              │
        │  (6) logs every call ──► Postgres  smarthub_prediction_log   │
        ▼                                                              │
 Clients  ◄── nginx ◄── serve                                          │
        │                                                              │
 (7) prediction-log-refresh joins logs to realised outcomes ──► prediction_monitoring.parquet
        ▼
 Dashboards + slo-alerts read monitoring data ───────────────────────┘
```

1. **Data pull** queries Redshift `lead_pings` over an SSH tunnel for a rolling time
   window and **upserts on `id`** into DuckDB and partitioned Parquet, so re-pulled
   windows update late-resolving outcomes rather than duplicating rows.
2. **Feature engineering** reads a rolling window of pulled data and writes a versioned,
   leakage-safe **training table**.
3. **Training** fits the win-probability model, evaluates it, runs an offline bid
   optimiser, and asks the **promotion gate** whether the new model beats the one
   currently serving.
4. If it passes, the model is written to the **registry** and the `current.json` pointer
   is flipped — this is the hand-off from offline to online.
5. **Serving** resolves the promoted model for the request's lead type, decides a bid, and
   returns it.
6. Every call is **logged** to Postgres (off the response path, so it never slows bidding).
7. **Monitoring** periodically joins logged predictions to realised outcomes into a
   Parquet dataset that dashboards and alerts read.

### 1.4 Offline vs online — which code belongs where

**Offline (runs on the Prefect worker, on a schedule):**
`data_pull/`, `feature_engineering/`, and the training side of `train_and_predict/`
(`train.py`, `flow.py`, `models.py`, `optimizer*`, `metrics.py`, `registry.py`,
`mlflow_utils.py`, `hyperparameter_search.py`).

**Online (the live request path):**
`server/` (the FastAPI app, auth, explanations, shap-worker), which consumes the serving
side of `train_and_predict/` (`registry.py`, `model_storage.py`, `preprocessing.py`,
`optimizer.py`, `shap_explain.py`, `llm_explain.py`, `prediction_log_schema.py`).

**Neither (operational / visualisation):**
`monitoring/` (dashboards + SLO alerts) and `model_diagnostics/`.

### 1.5 Repository structure

```
SmartHubDS_prep/
├── src/smarthub/            # the installable package (src-layout)
│   ├── core/                # cross-cutting: config, storage, paths, logging, notifications
│   ├── data_pull/           # STEP 1 — Redshift pull, windowing, validation
│   ├── feature_engineering/ # STEP 2 — build the training table
│   ├── train_and_predict/   # STEP 3 — training, registry, optimiser, serving logic
│   ├── server/              # the FastAPI bid API + shap-worker
│   ├── monitoring/          # Streamlit dashboards + SLO alerting
│   └── model_diagnostics/   # standalone post-training diagnostics app
├── config/                  # file configs: smarthub.yaml, training.yaml,
│                            #   hyperparameter_search.yaml, holidays.json
├── tests/                   # pytest suite (+ tests/sim/)
├── docker/                  # Dockerfiles, worker-entrypoint.sh, nginx/nginx.conf
├── docs/                    # CONTEXT.md (domain), API docs, diagrams, operations guide
├── scripts/                 # operational/manual scripts (smoke tests, load tests)
├── data/                    # (gitignored) pulled data, training tables, models, mlruns
├── .github/                 # CI/CD workflows + CI Slack notifier
├── prefect.yaml             # Prefect deployment + schedule definitions
├── docker-compose*.yml      # the service stack (see §5.7)
├── deploy.sh / install.sh   # deploy (pull images) / local bring-up
├── pyproject.toml           # package metadata, deps, console scripts
└── .env.example             # template for secrets/connection settings
```

> Domain background (what a lead, ping, auction, and expected revenue mean) lives in
> `docs/CONTEXT.md`. This document focuses on the system and its pipelines.

### 1.6 Console commands

Installed by `pip install -e .` (from `pyproject.toml [project.scripts]`):

| Command | Runs |
|---|---|
| `smarthub-pull` | `smarthub.data_pull.pull:main` — the data pull |
| `smarthub-build-features` | `smarthub.feature_engineering.build:main` — build the training table |
| `smarthub-train` | `smarthub.train_and_predict.train:main` — train + evaluate + promote |
| `smarthub-hyperparameter-search` | `smarthub.train_and_predict.hyperparameter_search:main` — Optuna HPO |
| `smarthub-apikey` | `smarthub.server.manage_keys:main` — manage API keys |

The serve app is launched with `uvicorn smarthub.server.app:app` (not a console script).
Two more background processes run as `python -m`: `smarthub.server.shap_worker` and
`smarthub.monitoring.slo_alerts`.

---

## 2. Pipeline Documentation

The three core pipelines run strictly in order — **data-pull → build-features →
train-model** — and each fails fast with a "run the previous step first" message if its
input is missing. Prediction and monitoring run continuously alongside them.

### 2.1 Data Pull / ETL

**Purpose.** Pull `lead_pings` (leads, their auction outcomes, and expected revenue) from
Redshift into local storage so the rest of the pipeline can read them without touching the
warehouse.

**When it runs.** Prefect deployment `data-pull` (flow `smarthub-data-pull`), every 4
hours, staggered per lead type: `0 */4 * * *` for auto (`lead_type_id 6`) and
`15 */4 * * *` for home (`lead_type_id 1`). Also runnable manually via `smarthub-pull`.

**Inputs.** A time window over `created_at`. On a scheduled run the window is derived from
a per-lead-type **watermark** (a Prefect Variable `smarthub_last_pull_timestamp_<type>`):
`[watermark − overlap_hours, now]`. On the very first run (no watermark) it backfills
`[now − default_lookback_hours, now]` (default 168 h = 7 days). A manual run takes explicit
`--min-created-at` / `--max-created-at` and does **not** move the watermark.

**Key behaviour.**
- Connects to Redshift through an **SSH bastion tunnel** by default (`SSH_TUNNEL=true`), or
  directly when the host is in the same VPC (`SSH_TUNNEL=false` — then no SSH key needed).
- The window deliberately **overlaps** the previous pull so late-resolving outcomes (won,
  revenue, listing payouts) get updated in place.
- Writes are **upserts keyed on `id`** in both backends, so overlapping windows never
  duplicate rows. The watermark only advances to the newest `created_at` actually pulled;
  an empty window keeps the previous watermark (never skips a gap).
- Data validation runs on every batch but is **warn-only** — it never blocks the pull
  (see §2.3).

**Outputs.**
- DuckDB table `lead_pings` (default `data/raw_datasets/leads.duckdb`) and/or partitioned
  Parquet (`data/raw_datasets/leads/YYYY/MM/DD-MM-YYYY.parquet`), per `STORAGE_BACKEND`.
- Prefect markdown artifacts (`data-pull-<type>`, `data-quality-<type>`).
- Slack: a success summary to the updates channel; data-quality issues to warnings.

**Configuration.** `.env` (Redshift/SSH credentials, `STORAGE_BACKEND`, paths) +
`config/smarthub.yaml` `data_pull` section (`overlap_hours`, `default_lookback_hours`,
`with_expected_revenue`, `selected_only`).

**Related modules.** `data_pull/pull.py` (CLI core, `fetch_leads`/`run`),
`data_pull/flow.py` (Prefect flow + `prediction_log_refresh_flow`),
`data_pull/windowing.py` (window math), `data_pull/models.py` (ORM queries + dtype
coercion), `core/storage.py` (`save_pull`, DuckDB/Parquet upsert).

### 2.2 Feature Engineering

**Purpose.** Turn the accumulated raw `lead_pings` into a **versioned, leakage-safe
training table** — the single input to training.

**When it runs.** Prefect deployment `build-features` (flow `smarthub-build-features`),
every 4 hours, ~30 min after the pull: `30 */4 * * *` (auto) and `45 */4 * * *` (home).
Also runnable manually via `smarthub-build-features`.

**Inputs.** A rolling window of stored raw leads (`training_window_days`, default 21 days;
`0` = all data), column-projected to keep memory low. If no raw data exists it stops with a
"run data-pull first" message.

**What it produces (high level).** A table of model-ready columns per lead:
- Raw lead attributes (bid, age, state, demographics, coverage, vehicle/driver/claim
  counts, campaign/source/traffic/device fields).
- Derived time features from `created_at` in Pacific time (`created_hour`,
  `created_dayofweek`, `is_workday` using `config/holidays.json`) plus a few helpers
  (`is_married`, `age_valid`, `age_cohort`).
- The target column (`won_flag`).

Each feature is declared once in a **feature registry** (`feature_engineering/
feature_registry.py`) — the single source of truth for which columns are model features,
which apply to which lead type, and which are mandatory API inputs at serving time.
Post-bid outcome columns are explicitly excluded as leakage.

**Outputs.** A versioned Parquet training table at `data/training_datasets/<type>/
<version>.parquet` plus a lineage/metadata manifest (row counts, win rate, coverage, data
range, feature list), a Prefect artifact, and a Slack success notification.

**Configuration.** `config/smarthub.yaml` `feature_engineering` section
(`training_window_days`, `training_campaign_ids`). Holidays file override `SMARTHUB_HOLIDAYS`.

**Related modules.** `feature_engineering/build.py` (CLI core, build stages),
`feature_engineering/flow.py` (Prefect wrapper), `feature_engineering/features.py`
(`build_training_table`, leakage controls), `feature_engineering/feature_registry.py`.

### 2.3 Data Validation

**Purpose.** Flag data-quality problems in each freshly pulled batch **without ever
changing or dropping data, and without failing the pull**. Detection and reporting only.

**When it runs.** Inside the data-pull flow (and the CLI pull path), after dtype coercion
and before persisting, run separately per lead type.

**What it checks.**
- **Schema** (via pandera when installed): ranges, domains, uniqueness; degrades to
  warn-only if pandera isn't present.
- **Raw-kind violations:** validates raw values against each registered field's declared
  type *before* coercion, so malformed values stay distinct from missing ones.
- **Custom per-field rules** and **cross-field integrity** (errored-row and
  auction-eligibility consistency).
- **Missingness:** columns at/above `high_missing_threshold` (default 0.5) are flagged,
  scoped per lead type so another product's columns aren't falsely flagged; plus
  constant/single-value columns.
- **Population accounting:** errored rows and auction-ineligible rows are counted and
  excluded from ordinary validation (reported as `erred_rows` / `auction_excluded_rows` /
  `validated_rows`), and a set of batch metrics is computed (expected-revenue coverage,
  won/sold counts, etc.).

**How issues are reported.** A single count, `issue_count = schema_issues + rule_violations
+ cross_field_hits` (0 = clean). A full report goes to a Prefect markdown artifact
(`data-quality-<type>`); when `issue_count > 0` a separate alert is posted to the Slack
**warnings** channel. A clean pull stays quiet. The CLI path logs the report instead.

**Configuration.** `config/smarthub.yaml` `validation.high_missing_threshold` (default 0.5).

**Related modules.** `data_pull/validation_runner.py` (orchestration, `validate_leads`,
`ValidationReport`), `data_pull/validation_report.py` (`issue_count`, `slack_group`,
`to_markdown`), `data_pull/validation_rules.py` + `validation_custom.py` + `field_registry.py`.

### 2.4 Model Training & Evaluation

**Purpose.** Train and evaluate the per-lead-type win-probability model, run the offline
bid optimiser, decide whether to promote it, version and store it, and log the run to
MLflow.

**When it runs.** Prefect deployment `train-model` (flow `smarthub-train-model`), daily:
`0 5 * * *` (auto) and `20 5 * * *` (home) UTC, with `register_mlflow: true`. Also runnable
manually via `smarthub-train --lead-type-id N`.

**Inputs.** The latest versioned training table from §2.2 (or a pinned `--version`). Needs
at least ~50 rows with both outcome classes present, or it stops with a clear message.

**What it does.**
- **Split** (time-based by default, 20% test) and basic diagnostics.
- **Fit** a LightGBM model with probability calibration (per `config/training.yaml`),
  with optional early stopping.
- **Evaluate** probability quality (ROC-AUC, PR-AUC, log loss, Brier, calibration error,
  etc.) and run the **offline bid optimiser**, which sweeps candidate bids to estimate
  expected-profit lift and checks that higher bids never lower the win probability
  (monotonicity).
- **Decide promotion** (see below), **save** the versioned model, and **log** to MLflow
  (best-effort — MLflow problems never fail the run).

**Promotion gate (the offline→online hand-off).** A new "challenger" model is compared
against the model currently serving, scored on the same held-out rows:
- **Absolute gates** (always): log loss ≤ `max_log_loss` (0.55) and expected profit ≥
  `min_expected_profit` (0.0).
- **Relative gates** (when a serving model exists): profit ratio ≥ `min_profit_ratio`
  (0.95), absolute profit loss ≤ `max_absolute_profit_loss_tolerance`, and log-loss
  regression ≤ `max_log_loss_regression` (0.01).
- **Monotonicity** must hold (max violation rate 0.0).
- The **first** model for a lead type is promoted unconditionally once the absolute gates
  pass. In `automatic` mode an eligible challenger is promoted immediately; in `manual`
  mode it is marked "awaiting manual promotion"; `disabled` turns promotion off. A
  challenger that fails is still saved as a version, just not promoted.

**Model storage & versioning.** Every run is saved locally under `data/models/<type>/` as
a `.pkl` + JSON manifest, with a `current.json` serving pointer. Promoted models get a
semantic production version (`<type>_v<major>.<minor>.<patch>`). When production storage is
configured (S3/MinIO), `promote()` publishes the artifact, manifest, and pointer to
production **first** (the commit), then mirrors locally — so a publish failure leaves the
previously-serving model authoritative.

**Outputs.** Versioned model + manifest, updated serving pointer, an evaluation report
under `data/model_evaluations/<type>/`, a Prefect artifact, a Slack notification, and an
MLflow run (experiment `anton_win_probability_<type>`).

**Configuration.** `config/training.yaml` (`defaults` + per-lead-type blocks); see §4.2.

**Related modules.** `train_and_predict/train.py` (training stages, CLI core),
`train_and_predict/flow.py` (Prefect wrapper), `models.py` / `preprocessing.py` /
`metrics.py`, `optimizer.py` / `optimizer_evaluation.py`, `registry.py` (`decide_promotion`,
`promote`, `rollback`), `model_storage.py` (filesystem + S3 stores), `mlflow_utils.py`.

### 2.5 Prediction (the serving API)

**Purpose.** Answer real-time bid requests ("Anton Bid Prediction API") with a ≤ 1-second
turnaround target.

**When it runs.** Continuously, as the `serve` container (`uvicorn smarthub.server.app:app`,
`SERVE_WORKERS` processes, default 4), behind nginx. Companion continuous processes:
`shap-worker` (SHAP backfill) and `slo-alerts` (health alerts).

**Endpoints.**
- `GET /health` — open (no auth); readiness + optional per-lead-type model check.
- `POST /recommend_bid` — the main endpoint; returns the recommended bid and an auditable
  decision path. API-key protected when auth is enabled.
- `POST /explain_bid` — returns a plain-English explanation for an already-logged
  prediction (it never re-decides the bid). API-key protected.

**How a request is served.**
1. Validate the request (universal fields plus the lead-type-specific inputs the feature
   registry marks mandatory).
2. Resolve the model for the lead type — `MODEL_URI` env override → pinned
   `active_model_version` → the registry's currently-serving pointer — and load it from an
   in-memory cache (models are eager-loaded on startup).
3. Run `decide_bid`, which always returns exactly one of three auditable paths:
   - `model` — the profit-maximising bid from the optimiser;
   - `cold_start_fallback` — used when no model has ever been promoted for the lead type;
   - `exploration` — a scheduled probe that perturbs the optimal bid to gather data.
4. Generate a `prediction_id`, measure turnaround (TAT), **enqueue** a log row, optionally
   attach SHAP factors, and return the result.

**Prediction logging.** Every call (success or failure) becomes one row in the Postgres
table `smarthub_prediction_log`. For `/recommend_bid` the write happens **off the response
path**: the row is put on an in-process queue and a background writer thread batches the
inserts, so database latency never affects TAT (a full queue drops the row with a warning
rather than blocking). The row captures the full candidate-bid sweep, decision path,
recommended bid, predicted win-rate/profit, and a config snapshot.

**SHAP offload.** Computing SHAP factor breakdowns (~1.5 s) is kept off the bid path by
default (`SMARTHUB_SHAP_MODE=offload`): serve logs the row with SHAP empty and the
dedicated `shap-worker` backfills it, claiming rows with `FOR UPDATE SKIP LOCKED` so
multiple replicas don't double-process. Modes: `offload` (default), `inprocess`, `off`.

**Configuration.** `config/smarthub.yaml` `prediction` and `explain` sections; env:
`MODEL_URI`, `SMARTHUB_SHAP_MODE`, `SERVE_WORKERS`, `SMARTHUB_PREDICTION_LOG_DB_URL`, API-key
settings.

**Related modules.** `server/predict.py` (the app, routes, cache, `decide_bid`, log writer),
`server/app.py` (stable import target), `server/auth.py`, `server/explain.py`,
`server/shap_worker.py`, `train_and_predict/optimizer.py`,
`train_and_predict/prediction_log_schema.py`.

### 2.6 Monitoring

**Purpose.** Watch the live service's health (service-level indicators) and surface it on
dashboards and via Slack alerts; and maintain a dataset that joins predictions to their
realised outcomes.

**When it runs.**
- `slo-alerts` container loops every 60 s (`python -m smarthub.monitoring.slo_alerts
  --loop --interval 60`), computing SLIs over a rolling window (default 15 min).
- The Streamlit **Health** page shows the same numbers on demand.
- The hourly `prediction-log-refresh` Prefect flow refreshes the monitoring dataset.

**What it measures (SLIs).** Request count and rate, TAT percentiles (p50/p95/p99/max),
the share of requests within 1 s, error count and rate, SHAP backlog, freshness (age of
the last successful request), the mix of decision paths, and the count/examples of any
predictions that exceeded 1 s.

**Alert thresholds** (`config/smarthub.yaml` `slo` section):

| Metric | Default | Severity |
|---|---|---|
| `tat_p99_seconds` | 0.8 | critical |
| `error_rate_pct` | 1.0 | critical |
| `shap_backlog` | 1000 | warning |
| `no_requests_minutes` | 10 | critical (only if there was recent traffic) |
| `predictions_over_1s` | 0 | failure (alert on even one prediction over 1 s) |

The `predictions_over_1s` alert includes the offending `prediction_id`s (slowest first) so
you can look them up in `smarthub_prediction_log` and in the logs.

**Alert state machine.** Alerts are debounced in memory per metric: a breach notifies once
when it starts (or when its severity changes), re-notifies unresolved **critical** breaches
every `SMARTHUB_SLO_REMINDER_MINUTES` (default 60), and sends a "recovered" message when it
clears. Breaches are routed to the right Slack channel by severity (see §3.5).

**The monitoring dataset.** `prediction_monitoring.parquet` is built by joining logged
predictions to realised lead outcomes on `lead_ping_id = lead_pings.id`, upserted on
`prediction_id`, and trimmed to the most recent 30 days. Dashboards read this file rather
than querying the live log database directly.

**Related modules.** `monitoring/slo.py` (`thresholds`, `compute_slis`, `evaluate_alerts`),
`monitoring/slo_alerts.py` (loop, state machine, Slack dispatch), `monitoring/app.py` +
page modules, `data_pull/prediction_logs.py` (builds the dataset), `core/storage.py`
(`save_monitoring`/`load_monitoring`).

> **Model-degradation monitor.** A separate `smarthub-model-degradation` flow exists in
> code (`monitoring/flow.py`, backed by `monitoring/model_degradation.py`) to compare
> predicted vs realised win rates per cohort. Its Prefect deployment is **currently
> disabled** in `prefect.yaml` (the ML bidding strategy it watches isn't live yet), so it
> runs only if invoked manually. The live monitoring path today is the `slo-alerts` loop.

---

## 3. Production Automation

### 3.1 Scheduled workflows (Prefect)

All deployments register against the `smarthub-pool` work pool and are created at worker
startup (`prefect deploy --all`). All crons are **UTC**. Two lead types run throughout:
**auto = 6**, **home = 1**.

| Deployment | Flow entrypoint | Queue | Schedule | Notes |
|---|---|---|---|---|
| `data-pull` | `data_pull/flow.py:data_pull_flow` | `default` | `0 */4 * * *` (auto), `15 */4 * * *` (home) | every 4 h, staggered |
| `build-features` | `feature_engineering/flow.py:build_features_flow` | `features` | `30 */4 * * *` (auto), `45 */4 * * *` (home) | ~30 min after pull |
| `train-model` | `train_and_predict/flow.py:train_flow` | `training` | `0 5 * * *` (auto), `20 5 * * *` (home) | daily; `register_mlflow: true` |
| `prediction-log-refresh` | `data_pull/flow.py:prediction_log_refresh_flow` | `monitoring` | `0 * * * *` | hourly; refreshes monitoring dataset |

A `model-degradation` deployment is present but **commented out / disabled** (see §2.6).

### 3.2 Real-time prediction workflow

Clients reach the API only through **nginx** (host port `8000` → internal `80`); the
`serve` container has no published port and is reachable only on the Docker network at
`serve:8000`. nginx re-resolves `serve`'s address every 10 s (so it self-heals after a
serve rebuild), applies per-IP and per-key rate limits (rejecting overflow with `429`),
and holds `/recommend_bid` to tight timeouts. Each uvicorn worker also caps concurrency and
returns `503` on overflow.

`/recommend_bid` and `/explain_bid` require an API key when `SMARTHUB_API_AUTH_ENABLED=true`
(`Authorization: Bearer <key>`); keys are stored SHA-256-hashed in Postgres and verified
against a short-TTL in-memory cache. `/health` is always open. The serve app loads promoted
models from the registry and logs every call, as described in §2.5.

### 3.3 Model promotion & deployment

Two separate things move into production:

**Models** flow through the registry, not through a deploy. When training promotes a model,
`registry.promote()` writes the artifact, manifest, and finally the `current.json` pointer
to the S3/MinIO store (in that commit order). The serve app reads that pointer; because
registry URIs are immutable, a new promotion naturally invalidates the in-memory model
cache. No redeploy, no restart — the live service picks up the newly promoted model on its
own.

**Code** flows through container images. CI builds and pushes three images to one Docker
Hub repo (`<account>/smarthub`), tag-prefixed `worker-*`, `dashboard-*`, `serve-*`. On the
server, **Watchtower** (enabled only under the `prod` profile) polls Docker Hub and
auto-pulls the labelled containers (worker, dashboard, serve, shap-worker, slo-alerts,
model-diagnostics, mlflow-ui); Postgres, prefect-server, and nginx are left alone.

`deploy.sh <version>` pulls pre-built versioned images (no source build on the box), ensures
the MLflow database exists, waits for Prefect to be healthy, brings the stack up, and posts
a Slack result. `docker/worker-entrypoint.sh` runs at worker start: it creates the work
pool and queues (idempotent), runs `prefect deploy --all` (so a worker restart re-registers
everything from `prefect.yaml`), then starts the worker.

### 3.4 How the pieces interact once deployed

- **Shared Postgres** backs Prefect state, the MLflow backend store, the prediction log,
  the config store, and the API-key store.
- **Prefect worker** runs the three-stage pipeline on schedule, writing raw data and
  training tables to `data/`, logging to MLflow, and promoting models into the registry.
- **Registry (S3/MinIO)** is the serving source of truth via `current.json`.
- **serve** handles live requests and logs them (off the response path); **shap-worker**
  backfills SHAP; **slo-alerts** watches health.
- **prediction-log-refresh** joins logs to outcomes into `prediction_monitoring.parquet`,
  which the **dashboards** read; **MLflow UI** and **model-diagnostics** read training
  artifacts.

### 3.5 Failure handling & notifications

Slack notifications are centralised in `core/notifications.py` — standard-library only,
best-effort (errors are logged and swallowed), and a no-op when no webhook is configured.
Alerts are routed by **severity → category → channel**:

| Severity | Channel | Webhook env var | `@here`? |
|---|---|---|---|
| `success` | updates | `SLACK_WEBHOOK_UPDATES_URL` | no |
| `warning` | warnings | `SLACK_WEBHOOK_WARNINGS_URL` | no |
| `critical` | critical | `SLACK_WEBHOOK_CRITICAL_URL` | **yes** |
| `failure` | failures | `SLACK_WEBHOOK_FAILURES_URL` | **yes** |

Each category falls back to a single legacy `SLACK_WEBHOOK_URL` if its specific webhook is
unset. Message titles follow a consistent shape: `SmartHub <ENV> · <area> · <status> ·
<subject>`, where `<ENV>` comes from `SLACK_ENV_LABEL` and the `@here` mention on
critical/failure can be overridden with `SLACK_MENTION_ON_FAILURE`.

Where alerts come from:
- **Pipeline flows:** every flow has a Prefect `on_failure` hook that posts to **failures**
  (`@here`) with the lead type and a link to the Prefect run; flows post success summaries
  to **updates** and data-quality issues to **warnings**.
- **SLO alerts:** `slo-alerts` routes each breach to its severity's channel and sends
  "recovered" messages when breaches clear (§2.6).
- **CI:** `.github/scripts/notify_ci.py` posts build results — success (with `docker pull`
  commands) to **updates**, failures (with the failed stage and trimmed error) to
  **failures**.
- **Release / deploy:** `version-bump.yml` posts "released vX.Y.Z" to updates (and failures
  on error); `deploy.sh` posts "deployed" / "deploy FAILED" similarly.

### 3.6 CI/CD

**`ci_cd.yml`** runs on every push/PR across Python 3.11 and 3.12: three independent lint
jobs (`isort`, `black`, `flake8` over `src/smarthub tests`), then `pytest`. Image build +
push jobs exist in the same file but are gated to manual `workflow_dispatch`.

**`version-bump.yml`** is the release workflow: on a merged PR into the deploy branch
(`smarthub.etl.pipeline`) it picks a SemVer bump level from PR labels, updates the version
in `pyproject.toml`, builds and pushes all three images tagged `:<prefix>-v<new>` and
`:<prefix>-latest`, commits + tags the release, and posts to Slack.

Required GitHub secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, and the Slack webhooks
(`SLACK_WEBHOOK_URL` plus the category webhooks used by routing).

---

## 4. Configuration

SmartHub uses a **three-tier** configuration model:

| Tier | What | Where | Edited by |
|---|---|---|---|
| **Secrets & connections** | DB/SSH/warehouse credentials, webhook URLs, API keys, connection URLs | `.env` (never committed; `.env.example` is the template) | a developer / operator |
| **Business settings** | `target_cm`, `bid_floor`, `bid_max_cap`, `min_source_quality` | Postgres config store | the Streamlit **Config** page (UI) |
| **Task configs** | per-stage pipeline knobs, model + HPO settings | `config/*.yaml` | a developer (version-controlled) |

The business-settings tier is small and deliberately the only thing the UI exposes. Its
values are environment-scoped (`staging`/`prod` with a `global` fallback) and versioned for
audit/rollback. Everything else is either a secret (`.env`) or a file config (`config/`).

> **Reading precedence.** For model/training configs, an environment variable override (if
> present) wins over the YAML value, which wins over the code default. The YAML files are
> optional — every key has a code-level fallback — but the committed files are the intended
> source of truth.

### 4.1 `config/smarthub.yaml` (per-stage task config)

Read by `core/task_config.py`; path override `SMARTHUB_TASK_CONFIG`.

**`data_pull`**

| Key | Default | Meaning |
|---|---|---|
| `overlap_hours` | `1` | Hours re-pulled before the watermark each run (catches late outcomes). |
| `default_lookback_hours` | `168` | First-run backfill window (7 days). |
| `with_expected_revenue` | `true` | Include the expected-revenue column. |
| `selected_only` | `true` | Pull only selected rows. |

**`feature_engineering`**

| Key | Default | Meaning |
|---|---|---|
| `training_window_days` | `21` | Rolling window the build reads; `0` = all stored data. |
| `training_campaign_ids` | `[]` | Campaign IDs to keep; empty = all. |

**`validation`**

| Key | Default | Meaning |
|---|---|---|
| `high_missing_threshold` | `0.5` | Null/blank rate at/above which a column is flagged "high-missing". |

**`prediction`**

| Key | Default | Meaning |
|---|---|---|
| `bid_step` | `0.25` | Bid increment granularity. |
| `exploration_variance_pct` | `0.00` | How far to probe around the optimal bid on an explore slot; also sets explore frequency (`N = round(1/this)`). |
| `recency_window_days` | `30` | Age beyond which a serving model's training data is flagged stale (informational). |
| `cold_start_fallback_bid_pct` | `0.50` | Where between floor and ceiling to bid when a lead type has no model. |
| `active_model_version` | `"none"` | Pin serving to one local version; `"none"` = serve the promoted model. |
| `shap_enrichment_mode` | `offload` | `inprocess` \| `offload` \| `off` (env `SMARTHUB_SHAP_MODE` overrides). |

**`explain`**

| Key | Default | Meaning |
|---|---|---|
| `llm_model` | `qwen2.5:1.5b-instruct` | Local Ollama model for explanations. |
| `ollama_host` | `http://localhost:11434` | Ollama endpoint (env `SMARTHUB_OLLAMA_HOST` overrides). |
| `top_n_factors` | `5` | SHAP factors surfaced in an explanation. |
| `timeout_seconds` | `30` | LLM call timeout. |

**`slo`** — see the table in §2.6.

**`model_degradation`** (used only if that flow is enabled): `enabled` (true),
`ml_bidding_strategy_ids` (`[4,5,7]`), `cohort_features` (`[campaign_id]`), `window_hours`
(1), `persistence_windows` (3), `required_bad_windows` (2), `min_opportunities_per_window`
(100), `warning_winrate_deviation` (0.10), `warning_zscore` (−2.0),
`critical_winrate_deviation` (0.20), `critical_zscore` (−3.0).

**Example (abridged):**

```yaml
data_pull:
  overlap_hours: 1
  default_lookback_hours: 168
feature_engineering:
  training_window_days: 21
  training_campaign_ids: []
validation:
  high_missing_threshold: 0.5
prediction:
  bid_step: 0.25
  exploration_variance_pct: 0.00
  recency_window_days: 30
  cold_start_fallback_bid_pct: 0.50
  active_model_version: "none"
  shap_enrichment_mode: offload
slo:
  tat_p99_seconds: 0.8
  error_rate_pct: 1.0
  shap_backlog: 1000
  no_requests_minutes: 10
  predictions_over_1s: 0
```

### 4.2 `config/training.yaml`

Read by `train_and_predict/config.py`; path override `SMARTHUB_TRAINING_CONFIG`. Structure:
a shared `defaults` block deep-merged with a per-lead-type `lead_types.<id>` block (the
per-type values win). Supported models: `logistic_regression`, `xgboost`, `lightgbm`; split
strategies: `time`, `random`; promotion modes: `manual`, `automatic`, `disabled`.

**`defaults` — key options**

| Path | Default | Meaning |
|---|---|---|
| `random_seed` | `42` | Seed (injected as `random_state`). |
| `split.strategy` | `time` | `time` or `random`. |
| `split.time.test_size` | `0.2` | Test fraction (time split). |
| `split.random.test_size` / `.stratify` | `0.2` / `true` | Test fraction / stratify (random split). |
| `optimizer.target_cm` | `0.25` | Target contribution margin for the bid optimiser. |
| `optimizer.minimum_bid` | `0.25` | Minimum bid. |
| `optimizer.bid_step` | `0.25` | Bid increment. |
| `optimizer.chunk_size` | `500` | Optimiser batch size. |
| `promotion.mode` | `automatic` | `automatic` \| `manual` \| `disabled`. |
| `promotion.criteria.max_log_loss` | `0.55` | Absolute gate: max log loss to promote. |
| `promotion.criteria.min_expected_profit` | `0.0` | Absolute gate: min expected profit. |
| `promotion.criteria.min_profit_ratio` | `0.95` | Relative gate: challenger/incumbent profit ratio. |
| `promotion.criteria.max_absolute_profit_loss_tolerance` | `500.0` | Relative gate: max absolute profit loss. |
| `promotion.criteria.max_log_loss_regression` | `0.01` | Relative gate: max log-loss regression vs incumbent. |
| `promotion.criteria.monotonicity.{enabled,tolerance,max_violation_rate}` | `true` / `1e-8` / `0.0` | Require bid→win-prob monotonicity. |
| `early_stopping.{enabled,validation_fraction,stopping_rounds,max_estimators,metric}` | `true` / `0.15` / `50` / `2000` / `binary_logloss` | LightGBM early stopping (LightGBM only). |
| `output.{report_root,model_root}` | `data/model_evaluations` / `data/models` | Where reports and models are written. |
| `mlflow.{tracking_db_path,artifact_root,experiment_name,registered_model_name}` | `data/mlflow.db` / `data/mlruns` / `anton_win_probability` / `anton-win-probability-model` | Local MLflow. |
| `mlflow.production.tracking_uri` | `''` | Prod MLflow (env `SMARTHUB_MLFLOW_PROD_TRACKING_URI`; empty disables). |
| `storage.production.backend` | `''` | `s3` / `filesystem` / empty (local-only). Env `SMARTHUB_PRODUCTION_STORAGE_BACKEND`. |
| `storage.production.{bucket,prefix,endpoint_url,region}` | `smarthub-models` / `models` / `''` / `''` | S3/MinIO registry location (env `SMARTHUB_S3_*` override). |

**`lead_types.<id>`** sets `name`, `model_type` (`lightgbm`), `calibration.{enabled,method,
cv}` (currently `true` / `isotonic` / `3`), and `models.lightgbm` hyperparameters
(`num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`, `subsample`,
`colsample_bytree`, `reg_alpha`, `reg_lambda`, plus `verbosity: -1`, `n_jobs: 1`,
`subsample_freq: 1`). Note: when early stopping is on, `n_estimators` must **not** appear in
`models.lightgbm` (`max_estimators` is the ceiling).

**Example (abridged):**

```yaml
training:
  defaults:
    random_seed: 42
    split: { strategy: time, time: { test_size: 0.2 } }
    optimizer: { target_cm: 0.25, minimum_bid: 0.25, bid_step: 0.25, chunk_size: 500 }
    promotion:
      mode: automatic
      criteria:
        max_log_loss: 0.55
        min_expected_profit: 0.0
        min_profit_ratio: 0.95
        max_log_loss_regression: 0.01
        monotonicity: { enabled: true, max_violation_rate: 0.0 }
    early_stopping: { enabled: true, max_estimators: 2000, stopping_rounds: 50 }
    storage:
      production: { backend: s3, bucket: smarthub-models, endpoint_url: "" }
  lead_types:
    6:
      name: auto
      model_type: lightgbm
      calibration: { enabled: true, method: isotonic, cv: 3 }
      models:
        lightgbm: { num_leaves: 102, max_depth: 6, learning_rate: 0.028, ... }
```

### 4.3 `config/hyperparameter_search.yaml`

Read by `train_and_predict/config.py` and consumed by `hyperparameter_search.py`; path
override `SMARTHUB_HYPERPARAMETER_SEARCH_CONFIG`. Same `defaults` + per-lead-type merge
pattern. Drives an Optuna search that selects hyperparameters; its output
(`best_parameters.yaml`) is what you copy into `training.yaml`.

**`defaults.search`**

| Key | Default | Meaning |
|---|---|---|
| `scoring` | `neg_log_loss` | Objective (must be `neg_log_loss` or `neg_brier_score`). |
| `n_trials` | `50` | Number of Optuna trials. |
| `cv_folds` | `3` | CV folds per trial. |
| `timeout_seconds` | `null` | Overall study timeout (none = unlimited). |
| `random_seed` | `42` | Seed. |
| `n_jobs` | `1` | Parallel trials. |

Other blocks: `validation.strategy` (`time` or `stratified_random`), `split.*` (final-test
reservation), `finalists.*` (shortlist sizes, log-loss guardrail, monotonicity),
`optimizer.*` (bid-optimiser evaluation of finalists), `calibration.{enabled,methods,cv}`,
`output.root` (`data/hyperparameter_tuning`), `early_stopping.*`.

**`lead_types.<id>.models.lightgbm`** has `fixed_parameters` (`verbosity: -1`, `n_jobs: 1`,
`subsample_freq: 1`) and a `search_space`. The shipped search space:

| Param | type | low | high | log |
|---|---|---|---|---|
| `num_leaves` | int | 15 | 127 | — |
| `max_depth` | int | 3 | 12 | — |
| `learning_rate` | float | 0.005 | 0.2 | yes |
| `min_child_samples` | int | 10 | 150 | — |
| `subsample` | float | 0.6 | 1.0 | — |
| `colsample_bytree` | float | 0.6 | 1.0 | — |
| `reg_alpha` | float | 1e-06 | 10.0 | yes |
| `reg_lambda` | float | 1e-06 | 20.0 | yes |

### 4.4 `.env` variables

Copy `.env.example` to `.env` and fill in. Grouped by purpose (defaults in parentheses):

**Redshift access**
- `SSH_TUNNEL` (`true`) — reach Redshift via the SSH bastion; `false` = direct connection.
- `SSH_HOST`, `SSH_USER` — bastion host/user (required when tunnelling).
- `SSH_PRIVATE_KEY_PATH` (`~/.ssh/id_ed25519`) — must exist when tunnelling.
- `SSH_PORT` (`22`), `SSH_PRIVATE_KEY_PASSWORD` (optional, only if the key has a passphrase).
- `REDSHIFT_HOST`, `REDSHIFT_DB`, `REDSHIFT_USER`, `REDSHIFT_PASSWORD` — required.
- `REDSHIFT_PORT` (`5439`), `REDSHIFT_CONNECT_TIMEOUT` (`10`).

**Storage backend**
- `STORAGE_BACKEND` (`both`) — `duckdb` \| `parquet` \| `both`.
- `DUCKDB_PATH` (`data/raw_datasets/leads.duckdb`), `PARQUET_DIR` (`data/raw_datasets/leads`),
  `PARTITION_DATE_COL` (`created_at`).

**Serving / prediction logging**
- `SMARTHUB_SHAP_MODE` (`offload`) — `inprocess` \| `offload` \| `off`.
- `SERVE_WORKERS` (`4`) — uvicorn worker processes.
- `SMARTHUB_PREDICTION_LOG_DB_URL` (shared Postgres) — the prediction-log DB.
- `SMARTHUB_SHAP_WORKER_POLL_SECS` (`1.0`), `SMARTHUB_SHAP_WORKER_BATCH` (`50`).

**API authentication**
- `SMARTHUB_API_AUTH_ENABLED` (`false`) — require a Bearer key on `/recommend_bid` and
  `/explain_bid`.
- `SMARTHUB_AUTH_DB_URL` (defaults to the prediction-log DB), `SMARTHUB_API_KEY_CACHE_TTL`
  (`60`).

**Model registry (S3 / MinIO)** — these override `storage.production.*` in `training.yaml`.
- `SMARTHUB_PRODUCTION_STORAGE_BACKEND` (`s3`) — `s3` \| `filesystem` \| empty.
- `SMARTHUB_S3_BUCKET`, `SMARTHUB_S3_PREFIX`, `SMARTHUB_S3_ENDPOINT_URL`
  (empty = real AWS S3; a URL = MinIO/S3-compatible), `SMARTHUB_S3_REGION`.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` — AWS credentials.
  **On EC2 with an IAM instance role, leave the access key and secret unset** so the
  containers use the role automatically (see §5.2). `AWS_DEFAULT_REGION` is still set.

**MLflow**
- `SMARTHUB_MLFLOW_PROD_TRACKING_URI` — production MLflow (empty = disabled).

**Ollama / LLM**
- `SMARTHUB_LLM_MODEL` (`qwen2.5:1.5b-instruct`), `SMARTHUB_OLLAMA_HOST`
  (compose sets `http://ollama:11434`).

**Config store (business settings)**
- `SMARTHUB_CONFIG_DB_URL` (shared Postgres), `CONFIG_ADMIN_PASSWORD` (blank = Config page
  locked).

**Prefect**
- `PREFECT_API_URL` (`http://prefect-server:4200/api`), `PREFECT_WORK_POOL` (`smarthub-pool`),
  `PREFECT_WORK_QUEUE` (`default`).

**Slack notifications** (see §3.5)
- `SLACK_WEBHOOK_UPDATES_URL`, `SLACK_WEBHOOK_WARNINGS_URL`, `SLACK_WEBHOOK_CRITICAL_URL`,
  `SLACK_WEBHOOK_FAILURES_URL` — per-channel webhooks.
- `SLACK_WEBHOOK_URL` — legacy single webhook used as fallback for any unset category.
- `SLACK_ENV_LABEL` — environment label in titles/footers (e.g. `PROD`, `STAGING`).
- `SLACK_MENTION_ON_FAILURE` — override the default `@here` on critical/failure
  (e.g. `<@U123ABC>` or `<!subteam^S123>`).

**SLO**
- `SMARTHUB_SLO_WINDOW_MINUTES` (`15`), `SMARTHUB_SLO_REMINDER_MINUTES` (`60`).

**Environment / images**
- `SMARTHUB_ENV` (`local`) — `local` builds from source (no Watchtower); `staging`/`prod`
  pull CD-built images and run Watchtower.
- `IMAGE_REPO` (`<account>/smarthub`), `IMAGE_TAG` (`latest`), `WATCHTOWER_POLL_INTERVAL`
  (`300`).

**Misc**
- `LOG_LEVEL` (`INFO`), `SMARTHUB_ROOT` (override project root, e.g. `/app` in a container).

**Path overrides** (read by the loaders, not in `.env.example`): `SMARTHUB_TASK_CONFIG`,
`SMARTHUB_TRAINING_CONFIG`, `SMARTHUB_HYPERPARAMETER_SEARCH_CONFIG`, `SMARTHUB_HOLIDAYS`.

---

## 5. Getting Started (Local Development)

### 5.1 Prerequisites

- **Python ≥ 3.10** (CI runs 3.11 and 3.12; containers use 3.11).
- **Docker + Docker Compose v2** (Docker Desktop with ≥ 4 GB RAM for the worker).
- For the explanation path only: a local **Ollama** server (not a pip dependency).

### 5.2 Environment setup

```bash
cp .env.example .env
```

Fill in the minimum to pull data: `REDSHIFT_HOST`, `REDSHIFT_DB`, `REDSHIFT_USER`,
`REDSHIFT_PASSWORD`, plus — when `SSH_TUNNEL=true` — `SSH_HOST`, `SSH_USER`,
`SSH_PRIVATE_KEY_PATH` (and the key file must exist). Set `SSH_TUNNEL=false` to connect to
Redshift directly (no SSH vars needed). Everything else has sensible compose defaults.

> **Credentials note.** Task configs are **not** in `.env` — they live in
> `config/smarthub.yaml`. `.env` holds only secrets and connection settings. On a cloud
> host with an IAM role for S3, leave `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` unset so
> the containers fall back to the role (the compose files default them to empty for exactly
> this reason).

### 5.3 Installing dependencies

For local development and tests without Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # editable install + pytest, flake8, black, isort, pre-commit
```

Optional extras (`pyproject.toml`): `dev`, `orchestration` (Prefect ≥ 3), `validation`
(pandera), `ml` (LightGBM, XGBoost, `mlflow==3.16.0`, FastAPI, uvicorn, boto3, joblib),
`explain` (SHAP; also needs `ml` + Ollama). To run the pipelines manually you'll want
`pip install -e ".[orchestration,ml]"`; for the full test suite, `pip install -e ".[dev]"
joblib`.

For the Docker path instead: `cp .env.example .env` then `./install.sh` (or
`./install.sh --check` to validate prerequisites only).

### 5.4 Running each pipeline locally

Run in order. Manual pulls take an explicit window and do not move the watermark.

```bash
# 1) data-pull
smarthub-pull --lead-type-id 6 \
  --min-created-at "2026-07-01 00:00:00" \
  --max-created-at "2026-07-09 00:00:00"
#   add --include-prediction-logs to also refresh the monitoring join

# 2) build-features
smarthub-build-features --lead-type-id 6          # auto
smarthub-build-features --lead-type-id 1          # home
smarthub-build-features --lead-type-id 6 --window-days 0   # all stored data

# 3) train-model
smarthub-train --lead-type-id 6
smarthub-train --lead-type-id 6 --no-mlflow       # skip MLflow logging

# hyperparameter search (writes best_parameters.yaml under data/hyperparameter_tuning/)
smarthub-hyperparameter-search --lead-type-id 6
```

Use `--help` on any command to see all options.

### 5.5 Starting the prediction API

Directly:

```bash
uvicorn smarthub.server.app:app --port 8000
curl "http://localhost:8000/health?lead_type_id=6"
```

With no `MODEL_URI` set it serves the currently-promoted model per lead type. For
`/explain_bid` locally, install the explain extras and run Ollama:

```bash
pip install -e ".[explain,ml]"
ollama pull qwen2.5:1.5b-instruct
ollama serve
```

Manage API keys (when auth is enabled) with `smarthub-apikey`.

### 5.6 Running tests

```bash
pip install -e ".[dev]" joblib
pytest                      # or: pytest -q
```

Linting matches CI (line length 88):

```bash
isort --check-only --diff src/smarthub tests
black  --check --diff src/smarthub tests
flake8 src/smarthub tests
```

Drop `--check`/`--diff` to autofix. Optionally enable the git hooks (lint on commit, tests
on push):

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
pre-commit run --all-files
```

### 5.7 The Docker stack

Three compose files:

- **`docker-compose.yaml`** — the consolidated single-file stack; used by `deploy.sh`
  (pull mode) and also supports `docker compose up -d --build` for an all-in-one local
  build. Local object storage (`minio`, `minio-init`) is behind the **`minio` profile**
  here, so a plain `up` uses real S3 / the instance role; add `--profile minio` for a local
  MinIO.
- **`docker-compose.prefect.yml`** — the full Prefect-oriented stack (Postgres,
  prefect-server, worker, MinIO, MLflow UI, dashboards, Ollama, serve, shap-worker,
  slo-alerts, nginx); used by `install.sh`. MinIO runs unconditionally here.
- **`docker-compose.local.yml`** — a thin override adding `build:` blocks so images build
  from source when `SMARTHUB_ENV=local`.

Bring the whole stack up locally:

```bash
./install.sh                 # SMARTHUB_ENV=local → builds from source, no Watchtower
# equivalent:
docker compose -f docker-compose.prefect.yml -f docker-compose.local.yml up -d --build
./install.sh --down          # stop and free host ports
```

For local S3-backed model storage with the single-file stack:

```bash
docker compose --profile minio up -d   # MinIO S3 API on :9000, console on :9001 (minioadmin/minioadmin)
```

**Ports:** Prefect UI `4200`; MLflow UI `5001`; monitoring dashboard `8500`;
model-diagnostics `8511`; bid API (via nginx) `8000`; MinIO `9000` / `9001`.

---

*This document reflects the repository as of package version 0.2.2. Flow names, schedules,
and config defaults are drawn directly from `prefect.yaml`, `config/*.yaml`, and
`.env.example`; when in doubt, those files are authoritative.*
