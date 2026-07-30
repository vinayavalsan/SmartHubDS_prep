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
│   ├── train_and_predict/        # STEP 3: train, registry, optimizer, explain, flow (Prefect)
│   ├── server/                   # FastAPI bid-recommendation API (predict.py) -- see docker/Dockerfile.serve
│   └── monitoring/               # Streamlit multipage app (leads, monitoring, config)
├── config/                       # smarthub.yaml (task configs) + holidays.json (is_workday calendar)
├── docker/                       # Dockerfile.app, Dockerfile.worker, Dockerfile.serve,
│                                 #   worker-entrypoint.sh, nginx/nginx.conf
├── docs/                         # CONTEXT, MODELING, PLAN_July2026, CHANGELOG
├── tests/                        # pytest unit tests
├── data/                         # accumulated data (gitignored):
│                                 #   raw_datasets/ (leads.duckdb + leads/ parquet),
│                                 #   training_datasets/<auto|home>/, models/, model_evaluation/
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

- `duckdb` — single file at `DUCKDB_PATH` (default
  `data/raw_datasets/leads.duckdb`); native upsert + SQL window reads.
- `parquet` — per-day files under `PARQUET_DIR` laid out as
  `data/raw_datasets/leads/YYYY/MM/DD-MM-YYYY.parquet`; same-day pulls merge +
  dedupe.
- `both` (default) — write to both.

`PARTITION_DATE_COL` (default `created_at`) buckets rows into the per-day Parquet
files. For training, `io.load_leads_window(days=N)` reads just the most recent
`N` days (the rolling recency window from CONTEXT §7).

The `data/` folder (gitignored) is laid out by role:

```text
data/
├── raw_datasets/                 # pulled leads (STEP 1 output)
│   ├── leads.duckdb              #   DUCKDB_PATH
│   └── leads/YYYY/MM/DD-MM-YYYY.parquet   # PARQUET_DIR partitions
├── training_datasets/            # versioned training tables (STEP 2 output)
│   ├── auto/<version>.parquet (+ .json lineage)
│   └── home/<version>.parquet
├── models/                       # versioned model registry (STEP 3)
│   ├── auto/  (vN_*.pkl, vN_*.json, current.json)
│   └── home/
└── model_evaluation/             # per-run eval reports (plots, metrics, csvs)
    ├── auto/
    └── home/
```

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
| **Task configs** | model_type, training window, calibration, bid_step, feature selection, data-pull knobs… | **`config/smarthub.yaml`** (sections: `data_pull` / `feature_engineering` / `features` / `training` / `prediction` / `explain`) | devs (git) |

### Business settings (UI)
Only business knobs live in the typed registry (`core/config_store.py`) and are
editable at **http://localhost:8501 → Config** (password `CONFIG_ADMIN_PASSWORD`,
pick `staging`/`prod`, Save — validated + history-tracked). Read in code:

```python
from smarthub.core.config_store import ConfigStore

ConfigStore().get("target_cm", env="prod")     # typed, validated, with fallback
```

### Task configs (YAML file)
Edit `config/smarthub.yaml` — one mapping per pipeline stage. Missing keys fall
back to code defaults, so the file is optional. Example — switch the model to LR:

```yaml
training:
  model_type: logistic_regression   # or lightgbm
  calibrate: true
```

Read in code via `smarthub.core.task_config` (e.g. `config.model_type()`,
`config.BID_STEP`, `training_window_days()`). Override the file path with
`SMARTHUB_TASK_CONFIG`. The YAML ships in the Docker images (`COPY config`), so a
worker rebuild picks up edits.

### Lead-type registry
All per-lead-type configuration lives in one lookup dictionary,
`features.LEAD_TYPES`, keyed by `lead_type_id`. Each entry is a frozen
`LeadTypeSpec` declaring that type's `name`, its `numeric` and `categorical`
features (a shared base plus the type's own extras), its `mandatory` core, and
`required_raw` (the raw signature columns validation requires for that type).
Feature selection is **inclusion-based**: everything downstream — model feature
columns, mandatory/optional resolution, and the validation completeness checks —
reads this dict for the requested type. There are no per-type variables or
`if auto / if home` branches, and an unregistered `lead_type_id` fails fast with
a clear error. **Adding a new lead type (e.g. commercial) is a single entry
here**; nothing else needs to change.

### Feature selection (mandatory vs optional)
Which features the **model** trains on is configurable per run via the
`features` section of `config/smarthub.yaml`, without touching code. Each lead
type has a **mandatory core** (locked in the registry —
`features.LEAD_TYPES[<id>].mandatory` — always trained on, never toggleable) and
an **optional set** listed in the YAML:

```yaml
features:
  # auto mandatory core (locked, cannot be removed): home_owner, multi_vehicle,
  # num_vehicles, insured, num_auto_accidents, dui, sr22_required, age (+ bands), bid
  auto_optional: >-
    state, gender, marital_status, campaign_id, traffic_tier, num_drivers,
    num_auto_violations, continuous_coverage_months, is_married, created_hour,
    created_dayofweek, is_workday
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
`config/smarthub.yaml validation.high_missing_threshold`. Needs the `validation`
extra (`pip install -e ".[validation]"`); if pandera is absent, the schema checks
degrade to a warning and the pandas checks still run.

## Testing & linting

```bash
pip install -e ".[dev]" joblib   # test deps (joblib: model-registry tests)
isort --check-only --diff src/smarthub tests   # import order (profile = black)
black --check --diff src/smarthub tests        # formatting (line length 88)
flake8          # style (.flake8: max line length 88)
pytest          # unit tests (metrics, features, storage, registry, notifications…)
```

Drop `--check-only`/`--check --diff` to have isort/black fix things in place
instead of just reporting them:

```bash
isort src/smarthub tests
black src/smarthub tests
```

The suite runs on the **base env** — the heavier `ml`/`orchestration` extras
(sklearn, lightgbm, mlflow, prefect) aren't needed because those imports are
lazy/guarded; only `joblib` is required (the model registry).

**isort + black.** Import sorting (`isort`, `profile = "black"`) always runs
*before* formatting (`black`, line length 88) — isort's black profile already
matches black's style, so the two never fight over the same line. Config
lives in `pyproject.toml` (`[tool.isort]` / `[tool.black]`); `.flake8`'s
`E203`/`W503` ignores exist so flake8 doesn't flag black's own style choices.

**CI** (`.github/workflows/ci_cd.yml`) runs isort, black and flake8 as three
independent jobs, then pytest (which needs all three), across Python 3.11 and
3.12, on every push/PR — a PR fails if any of the four fail. **pre-commit**
(`.pre-commit-config.yaml`) runs the same isort + black + flake8 (plus basic
hygiene hooks) on every commit, and the full `pytest` suite on every push
(kept out of the commit hook since it's slower). Enable both once with:

```bash
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push
pre-commit run --all-files                        # optional: check everything now
pre-commit run --all-files --hook-stage pre-push   # optional: run pytest now
```

## CI/CD

**One pipeline** (`.github/workflows/ci_cd.yml`):

- **`isort` / `black` / `flake8`** — three independent jobs, each across
  Python 3.11 and 3.12, on every push/PR.
- **`pytest`** — `needs` all three lint jobs, same Python matrix.
- **`build-ready`** — a no-op gate: `needs: pytest`, and only runs on a push
  to the deploy branch (`smarthub.etl.pipeline`). Exists purely so
  `build-worker`/`build-dashboard` below can both depend on "tests passed AND
  this is the deploy branch" without repeating that condition twice.
- **`build-worker` / `build-dashboard` / `build-serve`** — all three
  `needs: build-ready`, so they run **in parallel** once it's clear the images
  should be built (never before pytest passes, never on a PR). All three
  images share **one repo** (`<account>/smarthub`) — the free tier's single
  private repo — differentiated by a **tag prefix**:

```
<account>/smarthub:worker-latest      <account>/smarthub:worker-<sha>
<account>/smarthub:dashboard-latest   <account>/smarthub:dashboard-<sha>
<account>/smarthub:serve-latest       <account>/smarthub:serve-<sha>
```

- **`notify`** — `needs` all of the above, `if: always()` (so it still runs
  when something upstream failed), and gated the same way as the build jobs
  (push to `smarthub.etl.pipeline` only — PRs stay quiet; GitHub's own
  per-check status already covers those). Posts one Slack message: a
  `🔹`-bulleted metadata list (branch/commit/actor/runner/time), then either
  a `🐳 Pull the latest images:` section with the `docker pull` command for
  each image's `-<sha>` tag (pinned to this exact build — worker, dashboard,
  and serve), or — on failure — which stage failed and a trimmed excerpt of
  its actual output (the isort/black diff, the flake8 findings, or the pytest
  failure summary; a `build-worker`/`build-dashboard`/`build-serve` failure
  falls back to a pointer at that job's log, since `docker/build-push-action`
  doesn't expose its build log as a capturable value), then a `🔗 Workflow:`
  link. Goes through `smarthub.core.notifications.notify_raw` — same
  webhook/best-effort behavior as the Prefect flow alerts, just a custom
  layout. See `.github/scripts/notify_ci.py`.

**No inbound access to the server is needed** — the server pulls the new images
itself:

```
push → isort/black/flake8 + pytest → build worker + dashboard + serve (parallel) → push to <account>/smarthub
     → Watchtower (on the server) pulls the -latest tags → recreates the containers
     → worker re-runs `prefect deploy --all` on boot
```

**Local vs server (build vs pull).** `SMARTHUB_ENV` in `.env` decides where the
worker/dashboard images come from, and `install.sh` acts on it:

- `SMARTHUB_ENV=local` (default) — **builds from source**, no pull, no
  Watchtower: `docker compose -f docker-compose.prefect.yml -f
  docker-compose.local.yml up -d --build`.
- `SMARTHUB_ENV=staging|prod` — **pulls** the CD-built images from Docker Hub and
  runs Watchtower: `docker compose -f docker-compose.prefect.yml --profile prod
  up -d --pull always`.

The base compose pulls (image-only); `docker-compose.local.yml` adds `build:`
for the local path. Watchtower sits behind the `prod` profile so it never runs
locally (it would pull the Hub image and clobber your local build).

Set up once:

- **GitHub repo secrets:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (a Docker Hub
  access token with **Read & Write** — a read-only token fails the push);
  `SLACK_WEBHOOK_URL` for the `notify` job (same value as the runtime
  `SLACK_WEBHOOK_URL` env var used by `core/notifications.py` — GitHub Actions
  can't read that one directly, it needs its own copy as a repo secret).
  Without it, `notify` simply no-ops (same "disabled when unconfigured"
  behavior as the Prefect alerts).
- **On the server `.env`:** `IMAGE_REPO=<account>/smarthub` (must equal
  `${DOCKERHUB_USERNAME}/smarthub`); `IMAGE_TAG=latest` for rolling deploys.
- **Private repo:** `docker login` on the host so Watchtower can pull it.
- The `watchtower` service in `docker-compose.prefect.yml` polls Docker Hub
  (default 300s) and updates only the labelled containers (worker, dashboard,
  serve); Postgres, the Prefect server, and nginx are left untouched.

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
`data/training_datasets/<type>/<timestamp>.parquet` (per lead type, two schedules) — each
build is kept so a model traces to its exact snapshot; loaders default to the
latest. Trains on a rolling window set by `training_window_days` in
`config/smarthub.yaml` (`[feature_engineering]`; default **21**; `0` = all data) —
raw data is always retained, only the window read is limited.

**Model training** runs as a third deployment, `train-model`, on the `training`
queue. It reads the latest training table, trains `P(won | bid, features)`,
evaluates it, runs the offline bid-optimizer evaluation, saves the model as a
new **version**, writes a report under `data/model_evaluation/<type>/`, and
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
model's (`config/smarthub.yaml (training section)`, defaults 0.01 / 0.98). If it fails
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
regardless of the registry (or pin per lead type via `config/smarthub.yaml
[prediction] active_model_version`). It caches the loaded model in memory
(see "Model caching" below), so repeated requests don't re-unpickle from disk.

**Local/dev — run it directly:**

```bash
uvicorn smarthub.server.predict:app --port 8000
# POST /recommend_bid  ·  GET /health?lead_type_id=6
```

**Production — `serve` + `nginx` (part of `docker-compose.prefect.yml`).**
The API runs in its own container (`docker/Dockerfile.serve`, built/pushed by
CI the same way as `worker`/`dashboard` — see CI/CD above), auto-updated by
Watchtower like the other two. It has **no host `ports:` mapping** — it's only
reachable on the internal Docker network (`serve:8000`). `nginx` (plain HTTP
reverse proxy, `docker/nginx/nginx.conf`) is the only container with a
published port and is the sole way to reach the API from outside:

```bash
docker compose -f docker-compose.prefect.yml up -d serve nginx   # brings both up
curl http://localhost:8000/health?lead_type_id=6                 # via nginx, not the container directly
```

TLS isn't set up yet — nginx currently proxies plain HTTP. Add a `listen 443
ssl` server block (plus a cert volume mount) to `docker/nginx/nginx.conf` if
this ever needs to be reachable outside a trusted network.

**Scaling.** `serve` runs multiple `uvicorn` worker processes
(`SERVE_WORKERS`, default 4 — set it in `.env` to change), each with its own
copy of the in-memory model cache; that's fine, since models are small,
read-only, and each worker eager-loads its own copy at startup, so there's no
shared state to coordinate. For more headroom than one container's workers
can give, scale the container itself (`docker compose ... up -d --scale
serve=N`) and point `nginx.conf`'s `upstream smarthub_serve` at all the
replicas (Docker's service-name DNS round-robins across them). `nginx`
itself is unlikely to be the bottleneck before `serve`'s model-inference
code is — its tight timeouts (`proxy_connect_timeout 3s`, `proxy_read_timeout
5s`) already fail fast under overload rather than queuing, which suits a
real-time bid path where a slow bid is often worse than a dropped one.

**Model caching.** `predict.load_model` keeps loaded models in an in-process
cache keyed by the resolved model URI (and, for local `.pkl` paths, the
file's mtime as a safety net against a pinned `MODEL_URI` being overwritten in
place). Registry-resolved URIs are immutable versioned filenames
(`v<N>_<timestamp>.pkl`), so promoting a new model naturally busts the cache —
no manual invalidation needed in the normal (registry-driven) path.
`predict.clear_model_cache()` clears it manually if ever needed. The app also
**eager-loads** every configured lead type's model into this cache on
startup (a FastAPI `lifespan` hook), so the first real request after a
deploy/restart doesn't pay the load cost — a lead type with no promoted model
yet just logs a warning and is picked up lazily once one exists.
`GET /health?lead_type_id=<id>` reports `model_loaded: true/false` (a cheap
cache check, not a forced load) alongside the existing `model_uri`.

`load_model_and_manifest` (used by `/recommend_bid`/`/explain_bid`) also
caches the resolved **manifest** by `(lead_type_name, version)` — manifests
are immutable once written, so this needs no invalidation logic at all. It
still reads `current.json` exactly once per request (that's the
promotion-detection check, same role as the model cache's mtime check), but
no longer re-derives that same version 3 separate times or re-parses the
manifest JSON on every call — found and fixed after a live TAT investigation
traced an unexpectedly high `/recommend_bid` latency to this redundant,
uncached registry I/O rather than the actual model-scoring cost.

### Cold-start + exploration bidding policy

`/recommend_bid` doesn't just return the raw profit-maximizing optimizer bid —
it runs every lead through `predict.decide_bid`, which picks one of three
explicit, auditable paths (never emergent/random behavior — see
docs/CONTEXT.md §7):

- **`model`** — the normal case: the profit-maximizing bid from the
  currently-serving model. If that model's training data is older than
  `[prediction] recency_window_days` (default 30), the response flags it as
  stale (`model_data_age_days`) as a retraining-cadence signal — it still
  bids normally.
- **`cold_start_fallback`** — a brand-new lead type/partner with **no model
  ever trained/promoted yet**. Bids a fixed, configurable fraction of the way
  from the floor to the CM-respecting ceiling
  (`[prediction] cold_start_fallback_bid_pct`, default 50%) instead of
  guessing or erroring. Self-terminating: the first real model promotes
  unconditionally, so this path stops firing once training has run once.
- **`exploration`** — a deliberate probe around the optimum to keep learning
  the market's shape, per a **defined, reproducible schedule**: every lead is
  bucketed by hour-of-week (0-167 from `created_dayofweek`/`created_hour`),
  and 1-in-`N` buckets (`N = round(1 / exploration_variance_pct)`, default
  10 → ~10% of hours) are scheduled explore slots, perturbing the bid
  ±`exploration_variance_pct` and alternating direction each time a bucket
  triggers. Reproducible by design — the same lead always gets the same
  explore/exploit decision, so it can be recomputed and audited later
  instead of relying on a per-request coin flip.

Every response carries `decision_path` and a plain-English `decision_reason`
saying which path applied and why — the same fields `/explain_bid` (below)
surfaces to a human.

### Bid explanations (`/explain_bid`)

`POST /explain_bid` answers "why did Anton bid $X for this lead?" in plain
English — offline/on-demand only (a separate, slower endpoint;
`/recommend_bid` doesn't call it), but it runs the **same** `decide_bid`
policy above, so the bid and `decision_path` always match what live serving
would return for the same inputs. For a normal (`model`) or `exploration`
bid, it explains the model's prediction at that bid with SHAP feature
attributions and asks a small local LLM (via [Ollama](https://ollama.com)) to
turn those factors — plus a note when the bid was a scheduled exploration
probe — into 2-3 plain-English sentences. For a `cold_start_fallback` bid
there's no model to run SHAP against, so it skips straight to the policy's
own (fully deterministic) explanation. Same request body as `/recommend_bid`
(`lead_type_id` + lead attributes + `expected_revenue`).

Requires the `explain` extra (SHAP; only supports `model_type=lightgbm`) plus
`ml`, and a running local Ollama server with the configured model pulled.
`docker/Dockerfile.serve` installs both extras (`pip install -e ".[ml,explain]"`)
since the same container serves this route and /recommend_bid's background
SHAP enrichment (see below) — a plain `.[ml]` build silently breaks both
(caught and logged as a warning, never a crash).

**Docker:** `docker-compose.prefect.yml` runs a dedicated `ollama` service
(no host ports — internal only, like `serve`) with a persistent volume for
pulled models. `serve` checks whether the configured model
(`[explain] llm_model`) is already pulled there at its own startup, and
pulls it if not — entirely in a background thread (`explain
.ensure_model_pulled_async`), so neither container startup nor any
in-flight `/recommend_bid`/`/explain_bid` request ever waits on a pull
(a multi-GB model can take minutes the first time). `SMARTHUB_OLLAMA_HOST`
points `serve` at `http://ollama:11434`; `config/smarthub.yaml`'s
`ollama_host` stays `localhost` as the default for non-Docker/local-dev use.

**Local/dev (no Docker):**

```bash
pip install -e ".[explain,ml]"
ollama pull qwen2.5:1.5b-instruct   # or whatever [explain] llm_model is set to
ollama serve                        # if not already running

curl -X POST localhost:8000/explain_bid -H 'content-type: application/json' \
  -d '{"lead_type_id": 6, "bid": 0.25, "expected_revenue": 20, ...}'
```

Response: everything from `decide_bid` (`recommended_bid`, `decision_path`,
`decision_reason`, `model_data_age_days`, …) plus `predicted_win_rate`,
`base_win_rate` (the model's average win rate), `expected_profit`,
`top_factors` (ranked `{feature, value, shap, direction}`), `bid_curve`
(predicted win rate/profit at a few nearby bids — "the shape of the market"
around the chosen bid, not just the one number; empty for `cold_start_fallback`
since there's no model to score other bids with), and `explanation` (the
LLM's prose, the cold-start policy text, or a clear fallback message — never
a hard error — if Ollama isn't reachable). The prompt feeds `bid_curve` to the
LLM as real numbers and states the model's monotonic bid constraint
explicitly, so it can't reason backwards about whether a different bid would
have won more or less often. Configurable in `config/smarthub.yaml (explain section)`:
`llm_model`, `ollama_host`, `top_n_factors`, `timeout_seconds`.

### Prediction logging

Every `/recommend_bid` and `/explain_bid` call — success or failure — is
logged as one row in `smarthub_prediction_log`
(`smarthub/train_and_predict/prediction_log_schema.py`). Full design + worked
example: `docs/PREDICTION_LOG_SCHEMA.md`. In short: no per-candidate-bid
table and no separate SHAP table — the candidate-bid sweep is summarized as
JSON (`candidate_bid_generation`: method/bounds/count) and SHAP folds into a
single JSON column (`shap_explanation`), rather than one row per candidate or
a 1:1 child table. Failed requests log too (`status: "error"`,
`error_message`), and `serving_config` snapshots the exploration/recency/
cold-start policy values in effect for that specific prediction, so it stays
reproducible even if `config/smarthub.yaml` changes later. Points at the
shared Postgres by default (`$SMARTHUB_PREDICTION_LOG_DB_URL` to override;
defaults to SQLite in tests) — a logging-DB outage only ever logs a warning,
never breaks live bidding.

`recommended_bid_predicted_profit` (renamed 2026-07-23 from
`recommended_bid_expected_profit`, for consistency with
`recommended_bid_predicted_win_rate`'s naming) and a new
`recommended_bid_predicted_cm` (`= recommended_bid_predicted_profit /
expected_revenue`) are both on the row for every prediction that reached a
model-scored bid.

Both endpoints accept an optional `lead_ping_id` (never used in the bid
decision itself, purely a correlation key) and echo it straight back in the
response, alongside a `prediction_id` generated for every response —
either one is enough to join a response back to its log row later; supply
`lead_ping_id` up front if you have it, or just hang onto the returned
`prediction_id` if you don't.

**`/recommend_bid`'s logging is entirely off the response path (2026-07-23).**
Not just the SHAP enrichment below — the row insert itself, and every
derived metric computed for it (including `recommended_bid_predicted_cm`),
run as a `BackgroundTasks` job scheduled to fire only after the response has
already been sent. `prediction_id` is generated in the route (`uuid.uuid4()`)
and returned immediately, before the write happens — so it's a receipt for
"log under this id," not a guarantee the row exists yet; a lookup by it can
(rarely) come up empty if the background write itself fails. `/explain_bid`
is unchanged: it still logs synchronously, since it already runs SHAP (and
sometimes an Ollama call) inline and was never the fast path.

`/recommend_bid` also backfills `shap_explanation` for every prediction a
real model served (`top_factors`/`base_win_rate` only — no `bid_curve` or
LLM narrative, those stay `/explain_bid`-only) via a second background task,
scheduled to run after the logging insert above. Expect a brief window
where a freshly-logged `/recommend_bid` row still has `shap_explanation:
null` before that task fills it in.

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
