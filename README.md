# SmartHub Anton

Data-science toolkit for SmartHub / Anton: pull lead data from Redshift and
explore it through Streamlit dashboards. For the business domain (what the data
means and what Anton is solving), see [CONTEXT.md](./docs/CONTEXT.md).

## Project layout

```text
.
├── src/smarthub/                 # installable package (src layout), organised by pipeline stage
│   ├── core/                     # shared foundations: config, config_store, paths, logging,
│   │                             #   notifications + persistence (storage, io) + transforms
│   ├── data_pull/                # STEP 1: pull, models, cli, windowing, flow (Prefect)
│   ├── feature_engineering/      # STEP 2: features + flow (Prefect)
│   └── monitoring/               # Streamlit multipage app (leads, monitoring, config)
├── config/                       # smarthub.ini (task configs) + holidays.json (is_workday calendar)
├── docker/                       # Dockerfile.app, Dockerfile.worker, worker-entrypoint.sh
├── docs/                         # CONTEXT, MODELING, PLAN_July2026, CHANGELOG
├── tests/                        # pytest unit tests
├── data/                         # accumulated data (gitignored): leads/, training/, duckdb
├── prefect.yaml                  # Prefect deployments (data-pull, build-features)
├── docker-compose.prefect.yml    # Postgres + Prefect server + worker
├── install.sh                    # validate prerequisites/.env, then start the stack
├── pyproject.toml                # packaging, deps, scripts, pytest config
└── .env.example                  # copy to .env and fill in
```

## Setup

**Quickest path (Docker + Prefect):** fill in `.env`, then run the installer —
it validates prerequisites and `.env`, then brings up the stack:

```bash
cp .env.example .env        # fill in SSH + Redshift credentials
./install.sh                # validate + start (or: ./install.sh --check to validate only)
# Prefect UI: http://localhost:4200
```

**For local dev / running tests** (no Docker):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # editable install + dev tools (pytest, flake8)
```

## Usage

### 1. Pull data

The date range is now passed on the command line (no editing source):

```bash
smarthub-pull \
    --min-created-at "2026-06-07 00:00:00" \
    --max-created-at "2026-06-20 00:00:00"
# or:  python -m smarthub.data_pull.pull --min-created-at ... --max-created-at ...
```

Pulls **upsert on `id`** into the configured storage, so you can run on
overlapping windows (e.g. last 8h every 4h) and late-resolving outcomes
(`won`, `rev`, listing payouts) update in place instead of duplicating. Use
`--no-expected-revenue` to skip the listings join, `--all-listings` to aggregate
expected revenue over all listings.

### Storage (set in `.env`)

`STORAGE_BACKEND` selects where pulls are persisted:

- `duckdb` — single file at `DUCKDB_PATH` (default `data/smarthub.duckdb`); native
  upsert + SQL window reads.
- `parquet` — per-day files under `PARQUET_DIR` laid out as
  `data/leads/YYYY/MM/DD-MM-YYYY.parquet`; same-day pulls merge + dedupe.
- `both` (default) — write to both.

`PARTITION_DATE_COL` (default `created_at`) buckets rows into the per-day Parquet
files. For training, `io.load_leads_window(days=N)` reads just the most recent
`N` days (the rolling recency window from CONTEXT §7).

### 2. Dashboard

A **single multipage Streamlit app** (`dashboard` service) at
**http://localhost:8501**, with three pages:

- **Leads** — lead-ping explorer (filters, funnel, win-rate curves).
- **Monitoring** — performance over time (revenue/CM/win-rate), aggregated live
  from the pulled leads.
- **Config** — Anton's runtime tuning knobs; **password-gated** (set
  `CONFIG_ADMIN_PASSWORD` in `.env`). Reads/writes the shared Postgres.

Dashboards read the **lock-free Parquet copy** (`STORAGE_BACKEND=parquet`) so
they never contend with the worker's DuckDB write lock — keep
`STORAGE_BACKEND=both` in `.env` so Parquet is written.

Run it manually instead:

```bash
streamlit run src/smarthub/monitoring/app.py        # all three pages
```

The leads dashboard reads from the configured storage automatically (DuckDB if
present, else Parquet); click **🔄 Reload Data** after a new pull. Stop a
dashboard with `Ctrl+C`.

## Configuration (three tiers)

Config is split by *who owns it* (team decision — Kiran/Vinaya: "business
settings in the UI and nothing else; secrets in env"):

| Tier | What | Where | Edited by |
| --- | --- | --- | --- |
| **Secrets / connection** | SSH + Redshift creds, storage paths, passwords, DB URLs | **`.env`** | ops |
| **Business settings** | `target_cm`, `bid_floor`, `bid_max_cap`, `min_source_quality` | **Postgres config store**, via the **Streamlit Config page** | business (Kiran) |
| **Task configs** | model_type, training window, calibration, bid_step, feature selection, data-pull knobs… | **`config/smarthub.ini`** (`[data_pull]`/`[feature_engineering]`/`[features]`/`[training]`/`[prediction]`) | devs (git) |

### Business settings (UI)
Only business knobs live in the typed registry (`core/config_store.py`) and are
editable at **http://localhost:8501 → Config** (password `CONFIG_ADMIN_PASSWORD`,
pick `staging`/`prod`, Save — validated + history-tracked). Read in code:

```python
from smarthub.core.config_store import ConfigStore
ConfigStore().get("target_cm", env="prod")     # typed, validated, with fallback
```

### Task configs (ini file)
Edit `config/smarthub.ini` — sections per pipeline stage. Missing keys fall back
to code defaults, so the file is optional. Example — switch the model to LR:

```ini
[training]
model_type = logistic_regression   ; or lightgbm
calibrate  = true
```

Read in code via `smarthub.core.task_config` (e.g. `config.model_type()`,
`config.BID_STEP`, `training_window_days()`). Override the file path with
`SMARTHUB_TASK_CONFIG`. The ini ships in the Docker images (`COPY config`), so a
worker rebuild picks up edits.

### Feature selection (mandatory vs optional)
Which features the **model** trains on is configurable per run via the
`[features]` section of `config/smarthub.ini`, without touching code. Each lead
type has a **mandatory core** (locked in `features.py`, always trained on, never
toggleable) and an **optional set** listed in the ini:

```ini
[features]
# auto mandatory core (locked, cannot be removed): home_owner, multi_vehicle,
# num_vehicles, insured, num_auto_accidents, dui, sr22_required, age (+ bands), bid
auto_optional = state, gender, marital_status, campaign_id, traffic_tier,
    num_drivers, num_auto_violations, continuous_coverage_months, is_married,
    created_hour, created_dayofweek, is_workday
```

The auto mandatory core is SmartFinancial's lead-matching criteria (home owner,
multiple vehicles, currently insured, accidents, DUI, SR-22, age) plus `bid`.
`auto_optional` accepts a comma list (train on exactly those), `all` (every
optional feature — the default if the key is absent), or `none` (mandatory core
only). Unknown names are ignored with a warning; a mandatory feature can never be
dropped. Toggling changes only what the **model consumes** — every feature is
still built into the training table, so no re-pull/re-build is needed, just a
retrain. Training and serving both resolve features through
`features.model_feature_columns`, so they stay in lock-step. (Home selection is
not enabled yet — its mandatory core is TBD with Kiran, so home keeps all
features.)

### Holiday calendar (`is_workday`)
`is_workday` is a model feature. Weekends (Sat/Sun) are non-workdays computed in
code; **observed holidays** are listed in **`config/holidays.json`** (edit to
add/remove dates — `SMARTHUB_HOLIDAYS` overrides the path, and the file is
mountable to edit without a rebuild). Derived from `pst_date`.

### Data validation (on every pull)

Each data-pull validates the freshly-fetched `lead_pings` batch — **warn +
report only**: it flags bad rows and catalogues missing-value patterns but never
drops, imputes, or caps anything, and never blocks the pull. Lives in
`smarthub/validation` (pandera for schema/range/domain rules + a pandas layer for
cross-field integrity and the missing-value catalogue), invoked from both the
Prefect flow and the `smarthub-pull` CLI.

It checks: schema drift; `id` uniqueness; numeric ranges (e.g. `age` 1–120,
`bid ≥ 0`); categorical domains (`state`, `gender`, `marital_status`); cross-field
integrity (`current_carrier` populated while `insured=false`; `won=true` with no
bid; lead-type completeness); and per-column null/blank rates (surfaces things
like `pst_hour` being empty). Output: a per-lead-type `data-quality-<type>`
Prefect artifact, a "Data quality" section in the data-pull Slack notification,
and a log summary on the CLI. Tune the high-missing call-out in
`config/smarthub.ini [validation] high_missing_threshold`. Needs the `validation`
extra (`pip install -e ".[validation]"`); if pandera is absent, the schema checks
degrade to a warning and the pandas checks still run.

## Testing & linting

```bash
pip install -e ".[dev]" joblib   # test deps (joblib: model-registry tests)
pytest          # unit tests (metrics, features, storage, registry, notifications…)
flake8          # style (.flake8: max line length 88)
```

The suite runs on the **base env** — the heavier `ml`/`orchestration` extras
(sklearn, lightgbm, mlflow, prefect) aren't needed because those imports are
lazy/guarded; only `joblib` is required (the model registry).

**CI** (`.github/workflows/ci.yml`) runs flake8 + pytest on every push/PR across
Python 3.11 and 3.12. **pre-commit** (`.pre-commit-config.yaml`) runs flake8 +
hygiene hooks locally — enable it once with:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files   # optional: check everything now
```

## CI/CD

**CI** — see above (flake8 + pytest on push/PR).

**CD** (`.github/workflows/cd.yml`) — on every push to the deploy branch: run
the quality gate, then build the `worker` + `dashboard` images and push them to
Docker Hub. Both images share **one repo** (`<account>/smarthub`) — the free
tier's single private repo — differentiated by a **tag prefix**:

```
<account>/smarthub:worker-latest      <account>/smarthub:worker-<sha>
<account>/smarthub:dashboard-latest   <account>/smarthub:dashboard-<sha>
```

**No inbound access to the server is needed** — the server pulls the new images
itself:

```
push → flake8 + pytest → build worker + dashboard → push to <account>/smarthub
     → Watchtower (on the server) pulls the -latest tags → recreates the containers
     → worker re-runs `prefect deploy --all` on boot
```

Set up once:

- **GitHub repo secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (a Docker Hub
  access token with **Read & Write** — a read-only token fails the push).
- **On the server `.env`:** `IMAGE_REPO=<account>/smarthub` (must equal
  `${DOCKERHUB_USERNAME}/smarthub`); `IMAGE_TAG=latest` for rolling deploys.
- **Private repo:** `docker login` on the host so Watchtower can pull it.
- The `watchtower` service in `docker-compose.prefect.yml` polls Docker Hub
  (default 300s) and updates only the labelled containers (worker, dashboard);
  Postgres and the Prefect server are left untouched.

Rollback: set `IMAGE_TAG=<older sha>` on the server and `docker compose up -d`,
or push a revert. Note pushing deploys straight to the environment — switch the
CD trigger to tags if you later want a manual release gate.

## Docker

```bash
docker build -f docker/Dockerfile.app -t smarthub .
# leads dashboard (default):
docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/app/data" smarthub
# data pull:
docker run --rm --env-file .env -v "$PWD/data:/app/data" smarthub \
    smarthub-pull --min-created-at "2026-06-07 00:00:00" \
                  --max-created-at "2026-06-20 00:00:00"
```

## Orchestration (Prefect, local via Docker)

> **Run order (important): 1) `data-pull` → 2) `build-features` → 3) `train-model`.**
> `data-pull` loads leads into storage; `build-features` reads that data to build
> the training table; `train-model` trains + evaluates the model from that table.
> In the Prefect UI the deployments are labelled **STEP 1/2/3** (with matching
> `step-1-run-first` / `step-2-run-after-data-pull` /
> `step-3-run-after-build-features` tags). Each downstream step fails fast with a
> clear "run the previous step first" message (and a blocked-run artifact) if its
> input is missing.

The pull also runs as a scheduled **Prefect** flow, broken into tasks
(`resolve_window → fetch → persist → update_watermark`) in
`src/smarthub/data_pull/flow.py`. Everything runs locally in Docker:

```bash
docker compose -f docker-compose.prefect.yml up --build
# Prefect UI: http://localhost:4200
```

The stack is **Postgres + server + worker**. Postgres backs the Prefect server
(the default SQLite throws "database is locked" under the scheduler + worker
concurrency).

What happens on startup (`docker/worker-entrypoint.sh`): wait for the server →
create work pool `smarthub-pool` + queue `default` → `prefect deploy --all`
(reads `prefect.yaml`) → start the worker. So flows + deployment + pool + queue
are wired the moment the stack is up.

- **Per lead type**: one `data-pull` deployment with **two schedules** (Prefect
  per-schedule parameters) — auto (`lead_type_id=6`) and home (`lead_type_id=1`)
  — so each type is pulled separately. Tell them apart by each run's parameters.
- **Schedule / params** live in `prefect.yaml` (default: every 4h, 8h overlap,
  7-day first-run backfill).
- **Watermark (per type)**: the last record's timestamp is stored in a Prefect
  Variable `smarthub_last_pull_timestamp_<type>` (e.g. `..._auto`, `..._home`);
  each run resumes from it minus the overlap, so late-resolving outcomes get
  re-pulled and upserted. First run with no watermark backfills
  `default_lookback_hours`. (A window with no rows keeps the watermark unchanged.)
- The worker mounts your SSH key (`SSH_PRIVATE_KEY_PATH` on the host →
  `/keys/id_ed25519` in the container) and `./data` for persistence.

**Feature extraction** runs as a second deployment, `build-features`, on the
**same work pool but a separate queue** (`features`). It reads the accumulated
data and writes a **versioned** leakage-safe training table to
`data/training/<type>/<timestamp>.parquet` (per lead type, two schedules) — each
build is kept so a model traces to its exact snapshot; loaders default to the
latest. Trains on a rolling window set by `training_window_days` in
`config/smarthub.ini` (`[feature_engineering]`; default **21**; `0` = all data) —
raw data is always retained, only the window read is limited.

**Model training** runs as a third deployment, `train-model`, on the `training`
queue. It reads the latest training table, trains `P(won | bid, features)`,
evaluates it, runs the offline bid-optimizer evaluation, saves the model as a
new **version**, writes a report under `data/training_report/<type>/`, and
logs to MLflow. It runs with `log_prints=True`, so the full training +
optimizer output shows in the Prefect run logs, and it posts a summary
artifact + Slack notification (including the promotion decision below). Needs
the `ml` extra (installed in the worker image). One worker serves all three
queues: `default`, `features`, `training` (`PREFECT_WORK_QUEUES`).

**Model versioning & promotion gate.** Every training run is saved as an
immutable version — `data/models/<type>/v<N>_<UTC timestamp>.pkl` (e.g.
`v3_2026-07-09T140501Z.pkl`) plus a `.json` manifest of its metrics and
lineage — never overwritten. It is **not** automatically served: the newly
trained model ("challenger") is re-compared against the model **currently
serving** that lead type **on the same held-out test set** before anything
changes (`train_and_predict/registry.py:decide_promotion`). The challenger is
only promoted (repointing `data/models/<type>/current.json`, which is what
serving reads) if its ROC AUC doesn't regress beyond
`promotion_min_roc_auc_regression` *and* its offline expected profit on that
test set is at least `promotion_min_profit_ratio` of the currently-serving
model's (`config/smarthub.ini [training]`, defaults 0.01 / 0.98). If it fails
either check, the currently-serving model is left untouched and the run is
logged as **held**, not silently discarded — the model + its metrics are
still saved as a version for inspection. The first model trained for a lead
type is always promoted (nothing to compare against yet).

```python
from smarthub.train_and_predict import registry

registry.list_versions("auto")                 # ["v1_...", "v2_...", "v3_..."]
registry.currently_serving_version("auto")     # currently-serving version, or None
registry.rollback("auto")                      # repoint at the prior version
registry.rollback("auto", to_version="v1_2026-07-01T050000Z")  # or a specific one
```

### Run a stage manually (no schedule / no worker)

The whole pipeline runs on the Prefect schedule, but you can also run any stage
**directly from the command line** for ad-hoc runs, backfills, or debugging —
same code as the deployments, just in-process. Keep the pipeline order in mind:
**1) data-pull → 2) build-features → 3) train-model** (each reads the previous
stage's output). All read `.env` for credentials/paths; build-features and train
need the `ml` / `orchestration` extras.

```bash
pip install -e ".[orchestration,ml]"    # everything needed for manual runs

# 1) data-pull — pull leads from Redshift into storage.
#    Manual pull takes an explicit window and does NOT move the watermark.
smarthub-pull --lead-type-id 6 --min-created-at "2026-07-01 00:00:00" \
                                --max-created-at "2026-07-09 00:00:00"
smarthub-pull --help                    # all options (window, listings, log level)

# 2) build-features — rebuild the training table for a lead type / window.
smarthub-build-features --lead-type-id 6                # auto (default)
smarthub-build-features --lead-type-id 1                # home
smarthub-build-features --lead-type-id 6 --window-days 0   # use ALL stored data
#   (equivalent: python -m smarthub.feature_engineering.build --lead-type-id 6)

# 3) train-model — train + evaluate + (maybe) promote one lead type's model.
smarthub-train --lead-type-id 6                         # auto
smarthub-train --lead-type-id 1                         # home
smarthub-train --lead-type-id 6 --no-mlflow             # skip MLflow logging
smarthub-train --lead-type-id 6 --version 2026-07-09T073241Z   # pin training table
#   (equivalent: python -m smarthub.train_and_predict.train --lead-type-id 6)
```

Console scripts (`smarthub-pull`, `smarthub-build-features`, `smarthub-train`)
are installed by `pip install -e .`. **All three manual entry points are
Prefect-free** — each stage has a Prefect-free core (`data_pull/pull.py`,
`feature_engineering/build.py`, `train_and_predict/train.py`) that the CLI calls
directly, while the matching `flow.py` wraps it for the scheduled deployment
(Prefect run + artifact + Slack notification). To reproduce a scheduled run
end-to-end, run the three in order for the lead type you want.

The bid-recommendation API (FastAPI) serves the trained model. With no
`MODEL_URI` set it serves whichever version is currently promoted for the
request's `lead_type_id`; set `MODEL_URI` to pin a specific `.pkl`/MLflow URI
regardless of the registry (or pin per lead type via `config/smarthub.ini
[prediction] active_model_version`):

```bash
uvicorn smarthub.train_and_predict.predict:app --port 8000
# POST /recommend_bid  ·  GET /health?lead_type_id=6
```

## Slack notifications

Both pipelines send Slack alerts — a success message when a run finishes and a
failure alert on any error. Set an [Incoming Webhook](https://api.slack.com/messaging/webhooks)
in `.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
SLACK_ENV_LABEL=prod            # optional; shown on every message (default: hostname)
SLACK_MENTION_ON_FAILURE=<@U123>  # optional; @-mention on failures only
```

Leave `SLACK_WEBHOOK_URL` blank to disable — notifications become a clean no-op.
Sending is **best-effort**: a Slack/network problem is logged and swallowed, so
it can never break or fail a pull / feature build.

**Layout.** All three success messages share one **grouped** layout: the lead
type sits in the header, a bold **headline** leads with the key outcome, the rest
is split into titled sections (bold subheader + 2-column fields) separated by
dividers, and long paths / definitions go in the small footer. Built by
`notifications.notify_success_grouped`.

**On success:**

- *data-pull* — headline `N rows pulled · window`; sections **Volume** (rows
  fetched, DuckDB/Parquet row counts), **Watermark** (before → after), **Run**
  (start/finish UTC); footer carries the Parquet + DuckDB paths.
- *build-features* — headline `N training rows · version (win rate)`; sections
  **Rows** (raw→training, dropped, wins/losses), **Coverage**
  (`expected_revenue`, age-missing, `traffic_tier` distinct), **Time mix**
  (weekday/weekend/`is_workday` share), **Build** (window, feature count, data
  range); footer carries the day-metric definitions + table path.
- *train-model* — headline is the **promotion decision** (promoted/held +
  version + reason); sections **Model** (algo, rows, data range, table),
  **Performance** (ROC/PR AUC, log loss, calibration error), **Bid optimizer**
  (profit lift %, avg CM), **Features** (count split + optional included /
  excluded); footer carries the model file path.

**On failure:** every failure point is caught by a Prefect `on_failure` hook on
each flow (covers window resolution, the Redshift/SSH fetch, storage writes,
watermark updates, the feature build, and the "no data — run data-pull first"
guard). The alert identifies the pipeline, lead type, run name and a link to the
run in the Prefect UI, plus the error message. Manual CLI pulls
(`smarthub-pull`) also alert on failure.

## What the dashboards show

The dashboards visualise win rate, contribution margin, profit and revenue
across price points, time, states and campaigns — the "find the shelves"
analysis described in [CONTEXT.md](./docs/CONTEXT.md). Metric definitions live in one
place (`transforms.py`) so the two dashboards stay consistent.

### Open item

Expected revenue lives in a **separate table**, not in the `lead_pings` table
that `data_pull.py` currently queries. To model the bid ceiling for Anton, that
table will need to be joined into the pull. See CONTEXT.md §4.
