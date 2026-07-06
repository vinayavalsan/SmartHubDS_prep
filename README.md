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

## Anton runtime config

Anton's tunable knobs (target CM, recency window, bid bounds, active model
version, …) live in the **shared Postgres** — **not** in `.env`. `.env` holds
only secrets/connection settings. Each parameter is defined once in the typed
**registry** (`src/smarthub/core/config_store.py`), which supplies its **type,
default, allowed values, and description** (used for both validation and the UI).

### Defining a new parameter (code — one line)
A parameter must exist in the registry before it can be set (the UI can only edit
known params; it can't add new keys). Add a `ConfigParam` and read it where
needed:

```python
# src/smarthub/core/config_store.py
ConfigParam("max_daily_spend", "float", 5000.0,
            "Max total spend per day across all bids ($).", minimum=0.0)
```
```python
from smarthub.core.config_store import ConfigStore
cap = ConfigStore().get("max_daily_spend", env="prod")   # default until set
```
No DB migration is needed — reads fall back to the registry default until a value
is saved, and the new param appears in the Config UI automatically.

### Setting a value (three ways)
1. **Config UI** (normal path): open **http://localhost:8501 → Config**, enter the
   admin password (`CONFIG_ADMIN_PASSWORD` in `.env`), pick the environment
   (`staging` / `prod`), edit values, **Save**. Every change is validated and
   recorded (who/when) in a history table.
2. **Python / REPL:**
   ```python
   from smarthub.core.config_store import ConfigStore
   ConfigStore().set("target_cm", 0.30, env="prod", updated_by="nimesh")
   ```
3. **SQL** (last resort) — values live in the `smarthub_config` table on the
   shared Postgres.

### Reading a value in code
```python
from smarthub.core.config_store import ConfigStore
store = ConfigStore()
store.get("recency_window_days", env="prod")   # typed, validated, with fallback
```

> Note: the current registry is a **draft** — confirm with Kiran which knobs
> Anton should actually expose before treating it as final.

## Testing & linting

```bash
pytest          # unit tests for metric math and config validation
flake8          # style (max line length 88)
```

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
latest. Trains on a rolling window set by `TRAINING_WINDOW_DAYS` in `.env`
(default **21**; `0` = all data) — raw data is always retained, only the window
read is limited.

**Model training** runs as a third deployment, `train-model`, on the `training`
queue. It reads the latest training table, trains `P(won | bid, features)`,
evaluates it, runs the offline bid-optimizer evaluation, saves the model to
`data/models/anton_model_<type>.pkl`, writes a report under
`data/training_report/<type>/`, and logs to MLflow. It runs with
`log_prints=True`, so the full training + optimizer output shows in the Prefect
run logs, and it posts a summary artifact + Slack notification. Needs the `ml`
extra (installed in the worker image). One worker serves all three queues:
`default`, `features`, `training` (`PREFECT_WORK_QUEUES`).

Run any flow once locally without Docker:

```bash
pip install -e ".[orchestration]"           # data-pull + build-features
python -m smarthub.data_pull.flow           # pull (defaults to auto)
python -m smarthub.feature_engineering.flow # build features (defaults to auto)

pip install -e ".[orchestration,ml]"        # + model training
python -m smarthub.train_and_predict.train --lead-type-id 6   # train auto
python -m smarthub.train_and_predict.flow                     # train via Prefect
```

The bid-recommendation API (FastAPI) serves the trained model:

```bash
export MODEL_URI="data/models/anton_model_auto.pkl"   # or an MLflow model URI
uvicorn smarthub.train_and_predict.predict:app --port 8000
# POST /recommend_bid  ·  GET /health
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

**On success:**

- *data-pull* — lead type (auto/home + id), data window (`created_at` min→max),
  run start/finish (UTC), rows fetched, watermark before→after, DuckDB/Parquet
  row counts, and the exact stored file paths (per-day Parquet + DuckDB file).
- *build-features* — lead type, version, row & feature counts, win rate,
  training window, data date range, and the training-table output path.

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
