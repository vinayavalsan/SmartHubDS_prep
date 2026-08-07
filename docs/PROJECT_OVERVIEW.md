# SmartHub / Anton — Project Overview

*Idea, implementation, and toolchain, end to end.*

**Status:** living document · **Scope:** the full system from Redshift pull to the
live bid API and its CI/CD. For business domain detail see
[`CONTEXT.md`](./CONTEXT.md); for modelling detail see [`MODELING.md`](./MODELING.md);
for the serving-scaling effort see [`SCALING_PLAN.md`](./SCALING_PLAN.md).

---

## 1. The idea

Anton is an automated bidder for SmartFinancial insurance **lead-pings**. When a
lead becomes available, partners bid for it in real time; if Anton wins, the
lead is delivered and may convert into revenue. The core question the system
answers, per incoming lead, is: **what price should we bid?**

The answer is framed as a profit-maximisation problem rather than a raw
prediction. For a given lead, the model estimates the **probability of winning
as a function of the bid** (higher bids win more often), and the system then
picks the bid that maximises **expected profit** — `win_probability × (expected_revenue − bid)`
— while respecting a target **contribution margin (CM)** and a partner-side bid
floor. Win-probability is required to be **monotonically non-decreasing in bid**,
which is enforced in the model so the economics never invert.

Everything is organised **per lead type** (`LEAD_TYPES = {"auto": 6, "home": 1}`).
Each lead type gets its own training data, model, evaluation, and serving
pointer; adding a new type (e.g. commercial) is designed to be a single registry
entry, not a code fork.

The design philosophy throughout is **auditability over cleverness**: every bid
follows one of a small number of explicit, logged decision paths; validation
reports rather than silently mutates data; and every prediction is logged with
the exact model version, features, and config that produced it.

---

## 2. System at a glance

The system is a four-stage offline batch pipeline that produces a promoted
model, plus an online serving path that uses it:

```
Redshift ─▶ STEP 1 Data Pull ─▶ Raw Store ─▶ STEP 2 Feature Eng ─▶ STEP 3 Train & Predict ─▶ Model Registry
                    │ (+ Validation, warn-only)                         │ (promotion gate)          │
                    ▼                                                    ▼                           ▼
              DuckDB + Parquet                                    MLflow + S3/MinIO           current.json + .pkl
                                                                                                    │
   Online:  nginx ─▶ FastAPI /recommend_bid ─▶ decide_bid ─▶ bid (≤1s TAT)  ◀── loads promoted model
                          └─ async: prediction log → Postgres, SHAP enrichment (offloaded), /explain_bid → Ollama
```

All of it runs on a **single node** via `docker-compose`, orchestrated by
**Prefect**, with **GitHub Actions** building and shipping images that
**watchtower** auto-deploys. A companion architecture diagram is available on
request.

---

## 3. Implementation, stage by stage

### 3.1 Data pull (STEP 1 — `smarthub/data_pull`)

Data is pulled from **Amazon Redshift** over an **SSH tunnel**. `pull.py` opens
an `SSHTunnelForwarder` to the bastion host and forwards to Redshift, then builds
a SQLAlchemy engine on the `redshift+redshift_connector` dialect against the
local tunnel endpoint. The query itself is assembled from a single **field
registry** (the source of truth for which columns and in what order), producing a
`SELECT … WHERE created_at ∈ [lower, upper) ORDER BY created_at`, optionally
LEFT-JOINed to an expected-revenue subquery that sums listing payouts.

The pull is **incremental with overlapping windows**. A per-lead-type
**watermark** (stored in a Prefect Variable) records the latest `created_at`
seen; each run re-reads from `watermark − overlap_hours` to now. The overlap is
deliberate: outcomes like `won`, revenue, and listing payouts resolve late, so
re-pulling recent rows lets those values update in place. Updates land via an
**upsert keyed on `id`**, so overlapping windows never duplicate rows.

Persistence is handled by a pluggable storage layer (`core/storage.py`) selected
by `STORAGE_BACKEND`:

- **DuckDB** — a single file with native SQL upsert (delete-by-id + insert) and
  schema evolution.
- **Parquet** — one file per day (`YYYY/MM/DD-MM-YYYY.parquet`), merged and
  de-duped per partition; a lock-free copy the dashboards can read without
  contending with the writer.
- **both** (default) — write to each, so serving/dashboards read Parquet while
  the pipeline reads DuckDB.

### 3.2 Data validation (`smarthub/data_pull` validation modules)

Every pull validates the freshly-fetched batch, but in **detect-and-report
mode only** — it never drops, imputes, caps, or blocks the pull. A **pandera**
schema layer checks types, numeric ranges (e.g. `age` 1–120, `bid ≥ 0`), and
categorical domains (state, gender, marital status); a pandas layer adds
cross-field integrity checks (e.g. `won=true` with no bid) and a per-column
null/blank-rate catalogue. Results are rendered to a per-lead-type Prefect
artifact, a "Data quality" section in the Slack notification, and a CLI log
summary. If pandera isn't installed the schema checks degrade to a warning and
the pandas checks still run.

### 3.3 Feature engineering (STEP 2 — `smarthub/feature_engineering`)

A single registry (`FEATURES`, a dict of frozen `FeatureSpec` records) is the
one source of truth for every feature: its kind (numeric/categorical/binary),
whether it's raw or derived, which lead types it applies to, whether it's
mandatory, and the derivation function for computed features (e.g. `is_workday`
from the holiday calendar, `age_cohort` from `age`, `multi_vehicle` from
`num_vehicles`). Insertion order is the canonical model-feature order, which
keeps the monotone-constraint vector aligned at train and serve time.

`build_training_table` assembles a leakage-safe training frame: it drops errored
pings, coalesces revenue, derives the win label, applies the registry
derivations, and explicitly excludes known leakage columns. Feature **selection**
is configurable per run — each lead type has a locked **mandatory core** plus an
**optional set** toggled in YAML (`all` / `none` / an explicit list) — and both
training and serving resolve columns through the same registry function so they
never drift. The output is an **immutable, versioned training-table Parquet**
(timestamp version + a `.json` lineage manifest), never overwritten, bounded by a
rolling **recency window** (default 21 days).

### 3.4 Training and promotion (STEP 3 — `smarthub/train_and_predict`)

Training builds an scikit-learn `Pipeline` of a `ColumnTransformer`
(median-impute numeric; ordinal/one-hot encode categorical) plus a classifier.
Three model families are supported — **LightGBM** (default), **XGBoost**, and
**logistic regression** — chosen in config. The tree models apply a **monotone
constraint of 1 on `bid`**; optional **isotonic calibration**
(`CalibratedClassifierCV`, cv=3) keeps probabilities well-scaled while preserving
that monotonicity.

The training flow loads a training-table version, splits chronologically (or
stratified-random), fits the model, and evaluates with ROC/PR AUC, log-loss,
Brier score, and calibration error. It then runs an **offline bid-optimiser
evaluation** — replaying the profit-maximising bid choice over the holdout — and
compares the challenger against the **currently-serving** model on the same data.
A **promotion gate** (automatic mode, with absolute and relative thresholds on
log-loss and expected profit) decides whether to promote.

Model versioning uses a **file-based registry**: every run writes a
`run_<ts>_<uuid>.pkl` (joblib) plus a `.json` manifest capturing the full
resolved config; a promoted run also gets a sequential production version
(`auto_v6`, `home_v1`) and updates a `current.json` **serving pointer**.
Everything is mirrored into **MLflow** (params, metrics, artifacts, model
registry) and, when configured, published to **S3/MinIO** production storage.
Hyperparameter tuning is a separate, manual **Optuna** (TPE) search whose best
params are pasted into the training config.

### 3.5 Serving (`smarthub/server`)

The bid API is a **FastAPI** app with three routes:

- **`GET /health`** — service + resolved-model status.
- **`POST /recommend_bid`** — the hot path. It loads the cached promoted model,
  builds one model-ready feature row, generates the candidate-bid grid
  (`min_bid … expected_revenue × (1 − target_cm)` in `bid_step` increments),
  scores every candidate with a single `predict_proba`, and returns the
  **max-expected-profit bid**. `decide_bid` always takes exactly one explicit,
  logged path: **model** (normal), **cold_start_fallback** (no model yet), or
  **exploration** (a scheduled, reproducible probe that perturbs the optimal bid
  to keep learning the market). Models are cached per worker and warm-loaded at
  startup, so no disk load sits on the request.
- **`POST /explain_bid`** — on-demand explanation of an already-logged
  prediction: SHAP factor breakdown plus an optional local-LLM narrative. Never
  on the bid path.

Two things are kept **off the response path** so they never inflate bid latency:
the **prediction log** (one row per call, written asynchronously to Postgres) and
**SHAP enrichment** (see §5). A single `nginx` reverse proxy is the only
host-exposed entry point; the API itself runs `SERVE_WORKERS` (default 4)
preforked uvicorn processes.

### 3.6 Monitoring (`smarthub/monitoring`)

A single multipage **Streamlit** app: a **Leads** explorer (filters, funnel,
win-rate curves), a **Monitoring** page (revenue/CM/win-rate over time), and a
password-gated **Config** page that reads/writes the business-settings store.
Dashboards read the lock-free **Parquet** copy so they never contend with the
pipeline's DuckDB write lock.

---

## 4. Configuration model (three tiers)

Configuration is split by *who owns it*:

| Tier | What | Where | Edited by |
| --- | --- | --- | --- |
| Secrets / connection | SSH + Redshift creds, DB URLs, storage paths, passwords | `.env` | ops |
| Business settings | `target_cm`, `bid_floor`, `bid_max_cap`, `min_source_quality` | Postgres config store, via the Streamlit **Config** page | business |
| Task configs | model type, training window, calibration, feature selection, bid step, SHAP mode… | `config/smarthub.yaml` (+ `training.yaml`) | developers (git) |

Task-config keys fall back to code defaults when absent, and business settings
are typed, validated, and history-tracked in Postgres.

---

## 5. The serving-scaling work (1 s TAT)

The concrete requirement was that `/recommend_bid` serve **200 requests/minute
with a ≤ 1 s turnaround** per request. Benchmarking the real model showed the bid
decision itself is cheap (**7–17 ms**), but the **per-prediction SHAP
enrichment** — which ran in the serving process as a background task — costs
**~1.5 s** and, being CPU/GIL-bound, starved the bid responses under load.

The fix keeps the bid path untouched and **decouples SHAP from serving**, behind
a config flag (`shap_enrichment_mode`: `inprocess` | `offload` | `off`):

- **inprocess** (legacy, retained): serve computes SHAP after the response.
- **offload** (new default target): serve logs the prediction with SHAP empty; a
  separate **`shap-worker`** process drains those rows from the prediction-log
  table (a DB-as-queue, claimed with `FOR UPDATE SKIP LOCKED` for safe
  multi-replica concurrency) and backfills the identical explanation.
- **off**: no SHAP enrichment.

A load-test harness (`scripts/loadtest_recommend_bid.py`) proved it, **on the
same 4 serving workers**:

| Mode | Rate | p50 | p99 | > 1 s | SLA |
| --- | --- | --- | --- | --- | --- |
| inprocess | 200/min | 45 ms | 7.6 s | 7.5 % | ❌ |
| offload | 200/min | 37 ms | 55–80 ms | 0 % | ✅ |
| offload | 1200/min (6×) | 25 ms | 69 ms | 0 % | ✅ |

The headline: the 1 s TAT was achieved by **relocating SHAP, not by adding
serving capacity** — same 4 workers, same box. Both paths coexist behind the flag
so the change is reversible. (A follow-up "Phase 1" speedup — caching the SHAP
explainer per model version and explaining a single calibration fold — lets a
single `shap-worker` keep pace with the explanation backlog.)

---

## 6. Orchestration, infrastructure, and CI/CD

**Orchestration.** Each pipeline stage has a Prefect-free core plus a thin
**Prefect** flow wrapper. Deployments (`data-pull → build-features →
train-model`) are cron-scheduled per lead type onto a process work-pool, served
by a single Prefect worker.

**Infrastructure (single node, `docker-compose.prefect.yml`).** The stack is
one host running: **Postgres** (shared backing store for Prefect, MLflow
metadata, the config store, and the prediction log), the Prefect **server** and
**worker**, the **serve** API, the new **shap-worker**, **nginx**, **MinIO**
(S3-compatible model storage), an **MLflow UI**, **Ollama** (local LLM for
`/explain_bid`), the Streamlit **dashboard**, and **watchtower** (image-based
continuous deploy).

**CI/CD (`.github/workflows/ci_cd.yml`).** On every push / pull request, GitHub
Actions runs the quality gate — **isort**, **black**, **flake8**, and **pytest**
across a Python **3.11 / 3.12** matrix. Only on the deploy branch, and only
after the whole test matrix passes (`needs: pytest`), it **builds and pushes** the
`serve`, `worker`, and `dashboard` images to **Docker Hub** (tagged `:latest` and
`:<sha>`, with GHA build cache). **watchtower** on the node polls Docker Hub and
recreates those containers, closing the deploy loop. A `notify_ci.py` job reports
CI status (Slack).

---

## 7. Tools and technology stack

| Area | Tools | Purpose in the project |
| --- | --- | --- |
| Language / runtime | Python 3.10+ (CI: 3.11, 3.12) | Whole codebase (src layout, installable `smarthub` package) |
| Data wrangling | pandas, numpy, pyarrow | In-memory tables, feature building, Parquet I/O |
| Local analytics store | DuckDB | Single-file raw store with SQL upsert + window reads |
| Columnar files | Parquet (via pyarrow) | Per-day partitions; lock-free dashboard reads; training tables |
| Source DB / connectivity | Amazon Redshift, redshift-connector, SQLAlchemy, sqlalchemy-redshift, psycopg2-binary | Pull lead data; Postgres access for config/log |
| Secure access | sshtunnel, paramiko | SSH tunnel to reach Redshift |
| Data validation | pandera (+ pandas checks) | Warn-only schema / range / integrity / null-rate checks on each pull |
| ML training | scikit-learn, LightGBM, XGBoost | Pipelines, encoders, calibration; gradient-boosted + linear models |
| Calibration | scikit-learn `CalibratedClassifierCV` (isotonic) | Well-scaled, monotonicity-preserving probabilities |
| Hyperparameter search | Optuna (TPE) | Offline, manual tuning of model params |
| Experiment tracking | MLflow | Params, metrics, artifacts, model registry |
| Model persistence | joblib, boto3 (S3/MinIO), filesystem | Versioned `.pkl` artifacts; production model storage |
| Explainability | SHAP, Ollama (local LLM: qwen2.5:1.5b-instruct) | Factor breakdowns; plain-English `/explain_bid` narrative |
| Serving | FastAPI, uvicorn, pydantic, requests | Bid API, request validation, model serving |
| Dashboards | Streamlit, plotly | Leads / monitoring / config UI |
| Orchestration | Prefect 3 | Scheduled per-lead-type pipeline flows + watermarks |
| Object storage | MinIO (S3 API) | Production model artifacts on-node |
| Config | PyYAML, python-dotenv | Task configs; env secrets |
| Notifications | Slack | Data-quality + CI status messages |
| Containerisation | Docker, docker-compose, nginx | Single-node stack; reverse proxy |
| Shared database | PostgreSQL | Prefect + MLflow + config store + prediction log |
| Continuous deploy | watchtower | Auto-pull + recreate updated containers |
| CI/CD | GitHub Actions, Docker Hub | Lint + test matrix; build + push images |
| Code quality | black, isort, flake8, pre-commit | Formatting, import order, linting (line length 88) |
| Testing | pytest | Unit/integration tests (metrics, features, storage, registry, serving, logging…) |
| Load testing | httpx (async) — `scripts/loadtest_recommend_bid.py` | TAT / throughput verification against the 1 s SLA |

---

## 8. Deployment

The whole stack is a single-node `docker-compose` deployment. For the current
200 req/min target the recommended host is an **AWS `m7i.2xlarge` (8 vCPU /
32 GB)** with a **gp3 200 GB** data volume, placed in the same region as the
Redshift cluster to minimise pull latency. The bid path is CPU-trivial; the size
is driven by the co-located training worker, the SHAP offload workers, Ollama,
and the shared Postgres/MinIO. See [`SCALING_PLAN.md`](./SCALING_PLAN.md) §8 for
the sizing rationale and the levers that change it.

---

## 9. Repository layout

```text
src/smarthub/
  core/               shared foundations: config, config_store, paths, logging,
                      notifications, storage, io, transforms, lead_types
  data_pull/          STEP 1: pull, query_builder, windowing, field_registry,
                      validation_*, flow (Prefect), cli
  feature_engineering/STEP 2: features, feature_registry, build, flow
  train_and_predict/  STEP 3: train, models, registry, model_storage, optimizer,
                      metrics, mlflow_utils, hyperparameter_search, shap_explain,
                      llm_explain, prediction_log_schema, flow
  server/             FastAPI bid API: predict, app, explain, shap_worker
  monitoring/         Streamlit multipage app (leads / monitoring / config)
config/               smarthub.yaml, training.yaml, holidays.json, …
docker/               Dockerfile.app / .worker / .serve, nginx/
docs/                 CONTEXT, MODELING, SCALING_PLAN, PROJECT_OVERVIEW, CHANGELOG…
scripts/              loadtest_recommend_bid.py, …
.github/workflows/    ci_cd.yml (lint + test + build/push)
docker-compose.prefect.yml   the single-node stack
prefect.yaml          Prefect deployments (per lead type)
```
