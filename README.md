# SmartHub Anton

Data-science toolkit for SmartHub / Anton: pull lead data from Redshift and
explore it through Streamlit dashboards. For the business domain (what the data
means and what Anton is solving), see [CONTEXT.md](./CONTEXT.md).

## Project layout

```text
.
├── src/smarthub/                 # installable package (src layout)
│   ├── config.py                 # env-driven, validated settings
│   ├── cli.py                    # argument parsing
│   ├── paths.py                  # project-root path resolution
│   ├── logging_utils.py          # logging setup
│   ├── io.py                     # data loading/saving (friendly errors)
│   ├── transforms.py             # shared metric definitions
│   ├── models.py                 # SQLAlchemy ORM + query builders
│   ├── storage.py                # DuckDB + partitioned-Parquet persistence
│   ├── features.py               # leakage-safe training-table extraction
│   ├── data_pull.py              # Redshift -> storage pull
│   ├── dashboards/               # Streamlit apps (leads, monitoring)
│   └── flows/                    # Prefect flows (data_pull, features) + windowing
├── docker/                       # Dockerfile.app, Dockerfile.worker, worker-entrypoint.sh
├── tests/                        # pytest unit tests
├── data/                         # accumulated data (gitignored) + etl/sample_data.csv
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
# or:  python -m smarthub.data_pull --min-created-at ... --max-created-at ...
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

### 2. Launch a dashboard

```bash
streamlit run src/smarthub/dashboards/leads_app.py        # lead-ping explorer
streamlit run src/smarthub/dashboards/monitoring_app.py   # DS performance
```

Streamlit prints a local URL and opens your browser — by default
**http://localhost:8501**. To run both at once, give the second one a different
port:

```bash
streamlit run src/smarthub/dashboards/monitoring_app.py --server.port 8502
# -> http://localhost:8502
```

The leads dashboard reads from the configured storage automatically (DuckDB if
present, else Parquet); click **🔄 Reload Data** after a new pull. Stop a
dashboard with `Ctrl+C`.

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

The pull also runs as a scheduled **Prefect** flow, broken into tasks
(`resolve_window → fetch → persist → update_watermark`) in
`src/smarthub/flows/data_pull_flow.py`. Everything runs locally in Docker:

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
read is limited. One worker serves both `default` and `features`
queues (`PREFECT_WORK_QUEUES`).

Run either flow once locally without Docker (needs `pip install -e ".[orchestration]"`):

```bash
python -m smarthub.flows.data_pull_flow      # pull (defaults to auto)
python -m smarthub.flows.features_flow       # build features (defaults to auto)
```

## What the dashboards show

The dashboards visualise win rate, contribution margin, profit and revenue
across price points, time, states and campaigns — the "find the shelves"
analysis described in [CONTEXT.md](./CONTEXT.md). Metric definitions live in one
place (`transforms.py`) so the two dashboards stay consistent.

### Open item

Expected revenue lives in a **separate table**, not in the `lead_pings` table
that `data_pull.py` currently queries. To model the bid ceiling for Anton, that
table will need to be joined into the pull. See CONTEXT.md §4.
