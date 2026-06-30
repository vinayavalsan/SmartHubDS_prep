# Changelog


## 2026-06-25

### Understanding & docs
- Analyzed the original repo and produced a code-flow overview.
- Wrote `CONTEXT.md` from Kiran's DS-weekly walkthrough: SmartHub as a
  reseller, partners vs. buyers, the ping → bid → win → resell flow, money
  mechanics, and Anton's goal.
- Corrected `CONTEXT.md` per Vinaya/Kiran's Slack feedback: bid bounds
  (`ceiling = expected_revenue × (1 − target_CM)`, partner-side floor, bounds
  may not exist), profit maximization as a *single* objective, and a new
  exploration + recency section.
- Reconciled the doc to the real warehouse schema (the "payout is overloaded"
  finding; concept → column map).

### Productionization (pre-MVP → production-grade)
- Restructured into an installable `src/smarthub/` package (config, CLI, paths,
  logging, IO, transforms, models, storage, dashboards).
- Replaced the raw SQL with a SQLAlchemy ORM (`LeadPing`, `LeadPingListing`)
  reaching Redshift through the SSH tunnel.
- Expanded the model to the full real schema, excluded PII columns, and added an
  expected-revenue join (aggregating `est_payout` from the listings).
- Hardened the pull: env validation, `main()` guard, deterministic
  tunnel/connection cleanup, logging, CLI date range, optional SSH key
  passphrase, and `--no-expected-revenue` / `--all-listings` flags.

### Orchestration (Prefect, local via Docker)
- Data pull now runs as a scheduled **Prefect 3** flow, split into tasks
  (`resolve_window → fetch → persist → update_watermark`) reusing the existing
  pull/storage code.
- Locally hosted via `docker-compose.prefect.yml` (server + worker);
  `docker/worker-entrypoint.sh` wires work pool + queue + deployment + worker on
  startup, deployment declared in `prefect.yaml`.
- **Per-lead-type pulls**: flow parametrized by `lead_type_id`; one `data-pull`
  deployment with two **schedules** (per-schedule parameters) for auto (6) and
  home (1). ORM query builders take an optional `lead_type_id` filter.
- Last-record timestamp stored **per type** in a Prefect **Variable**
  (`smarthub_last_pull_timestamp_<type>`); each run resumes from it (minus an
  overlap so late-resolving outcomes are re-pulled), 7-day backfill on first run.
  A window with no rows keeps the watermark unchanged.
- Pure window logic in `flows/windowing.py` with tests.
- Postgres backs the Prefect server (avoids SQLite "database is locked").

### Feature extraction
- `features.build_training_table` builds the leakage-safe training table per lead
  type: keeps real bidding decisions (`won` true/false), ping-time features +
  `bid` + `expected_revenue` + `won_flag`, drops leakage + zero-variance columns,
  adds time features.
- `build-features` Prefect deployment on the **same work pool, separate queue**
  (`features`); two schedules (auto/home). One worker serves both `default` and
  `features` queues.
- Output is **versioned**: `data/training/<type>/<timestamp>.parquet` (each build
  kept; loaders default to latest) with a `<version>.json` lineage manifest
  (window, date range, rows, win rate, features). Rolling window via
  `TRAINING_WINDOW_DAYS` (`.env`, default 21; 0 = all data).
- Both flows publish **Prefect markdown artifacts** (keyed `data-pull-<type>` /
  `build-features-<type>`) summarising each run — window, rows, watermarks, win
  rate, date range, features — visible per-run and as history in the UI.

### Storage
- Added a DuckDB + partitioned-Parquet storage layer, switchable via `.env`
  (`STORAGE_BACKEND`).
- Both backends upsert on `id` so overlapping re-pulls update late-resolving
  outcomes instead of duplicating.
- Parquet layout: `data/leads/YYYY/MM/DD-MM-YYYY.parquet`, bucketed by
  `created_at`; DuckDB auto-migrates new columns.
- Added `io.load_leads_window(days=N)` for rolling-recency training reads.

### Live pull verified
- First real pull succeeded end-to-end (tunnel → ORM query → expected-revenue
  join → both sinks): 276 rows, 55 columns.

### Dashboards (Streamlit)
- Ported both dashboards onto the shared library; fixed relative imports so
  `streamlit run` works.
- Added Plot Type 4 — cumulative win-rate "shelves" curves (bid ≤ X vs bid > X,
  plus delta), an accept/reject funnel, and partner / bidding-strategy / insured
  filters.

### Ops / structure
- `restart: unless-stopped` on all compose services.
- Dashboards run in the compose: **leads** at http://localhost:8502, **monitoring**
  at http://localhost:8503 (renamed "SmartHub Leads" / "SmartHub Monitoring");
  both read the lock-free Parquet copy (`STORAGE_BACKEND=parquet`).
- **Monitoring dashboard now uses real data** — `transforms.leads_to_monitoring_base`
  aggregates pulled `lead_pings` into the time-series performance shape; the
  sample CSV (`data/etl/sample_data.csv`) and `io.load_monitoring` were removed.
- `install.sh --down` stops the stack and frees the host ports (4200/8502/8503).
- Tidied Docker files into `docker/` (`Dockerfile.app`, `Dockerfile.worker`,
  `worker-entrypoint.sh`); compose + README updated.
- Added `install.sh` — validates prerequisites (docker/compose, `.env` present,
  required vars set, SSH key exists, valid `STORAGE_BACKEND`) then brings the
  stack up; `--check` validates only.

### Quality & cleanup
- Unit tests throughout (41 total: transforms, config, ORM SQL, storage);
  flake8 clean; added a `Dockerfile`.
- Reset `.env.example` to placeholders; removed all legacy/dead files
  (`prep/`, `src/monitoring/`, empty `src/utils/` stubs, empty data placeholders).

