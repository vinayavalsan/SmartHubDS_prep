# Changelog


## 2026-07-08

### expected_revenue backend column: confirmed present but unpopulated (blocked)
- `lead_pings.exp_rev numeric(10,2)` exists (the intended authoritative expected
  revenue), but it is 0 non-null / 0 positive across all 572,047 rows. So we keep
  the interim listings-sum for R; switching to `exp_rev` is blocked on the backend
  populating it. Recorded in CONTEXT §4. Verified via SQL on prod.

### is_workday feature + holiday calendar; age missingness-as-signal
- Added **`is_workday`** as a model feature (per Kiran/Vinaya): weekends
  (Sat/Sun) are non-workdays computed in code; observed holidays live in a
  git-versioned **`config/holidays.json`** (mountable; `SMARTHUB_HOLIDAYS`
  overrides). New `core/holidays.py` (`is_workday`/`is_holiday`). Derived from
  `pst_date` (Pacific business day). Populated with **SmartFinancial's 2026
  company paid holidays** (8 dates incl. 2027-01-01); not the generic US federal
  set — MLK/Presidents'/Juneteenth/Veterans Day are working days here.
- **age**: per Vinaya, replaced the NaN-clamp with a `-1` sentinel + an
  **`age_missing`** flag (set on null OR implausible/default age); no mean
  imputation. `age_missing` + `is_workday` added to the model feature set.
- Dropped the placeholder `holiday_calendar` ini knob (holidays now in the JSON).
- Not doing (per Kiran): per-ping completeness / reliability / data-quality
  features — `traffic_tier` already carries source quality at the aggregate
  level.

### Config split into three tiers (team decision: UI = business only)
- Per Kiran/Vinaya ("business settings in the UI and nothing else; secrets in
  env"), reorganized config into: **secrets → `.env`**, **business → Postgres
  config store/UI** (trimmed to `target_cm`, `bid_floor`, `bid_max_cap`,
  `min_source_quality`), **task configs → `config/smarthub.ini`** with
  `[data_pull]`/`[feature_engineering]`/`[training]`/`[prediction]` sections.
- Added `core/task_config.py` (ini loader with typed getters + defaults).
- Moved `model_type`, `calibrate`, `drop_zero_variance`, `test_size`,
  `random_seed`, `bid_step`, `training_window_days`, `holiday_calendar`,
  `exploration_variance_pct`, `active_model_version` out of the UI registry into
  the ini. Training reads model/window/bid_step from the ini; `target_cm`/
  `bid_floor` still come from the business store.
- Dockerfiles now `COPY config` so the ini ships in the images. README config
  section rewritten. `TRAINING_WINDOW_DAYS` removed from `.env`/`.env.example`
  entirely — the training window now comes solely from the ini
  (`[feature_engineering] training_window_days`). Fixed a latent `paths.py`
  `project_root()` bug (pointed at `src/` after the reorg → now repo root).

### Quieted LightGBM feature-name warning flood
- The offline optimizer calls `predict_proba` once per test row, so sklearn's
  harmless "X does not have valid feature names" warning flooded the logs
  (tens of thousands of lines). Silenced it at the optimizer's prediction sites
  (`_quiet_feature_name_warning`). Training behavior unchanged.

### LightGBM, runtime model config, and model→data lineage
- Added a **LightGBM** model (`models.build_lightgbm_model` + `build_model`
  dispatch) with a **monotonic-increasing constraint on `bid`** (so P(win) rises
  with the bid — safe for the optimizer) and optional calibration. `MODEL_TYPE`
  now defaults to `lightgbm`; LR still available via the switch. Added
  `lightgbm` to the `ml` extra.
- Wired training to the **Tier-2 config store**: added a `model_type` knob to the
  registry (Config page), and training now reads `model_type`, `target_cm`, and
  `bid_floor` from the store with a safe fallback to code constants if it's
  unreachable.
- **Model lineage**: `prepare_training_data` records the training-table version
  + data date range; every train logs `model_type`, `training_table_version`,
  `data_min/max_created_at`, `source_row_count` to MLflow (params + tags), the
  Prefect artifact, the Slack notification, and the report JSON — so a model
  traces back to the exact data it was trained on.

### Model cleanup pass (data quality + generalization)
- Clamp implausible `age` (outside 1–200) to NaN in `feature_engineering`
  (fixes raw ages like -7648 / 1828 feeding the scaler); applies to train +
  serve.
- Dropped `source_type_id` (~9k unique -> memorization, inflated ROC AUC) and
  redundant `account_id` (1:1 with `campaign_id`) from the model feature set.
- Auto-drop zero-variance feature columns at train time
  (`preprocessing.drop_zero_variance`) — removes dead constants (all-'false'
  insured/home_owner/dui/military_affiliation) while keeping them in the schema
  for when the data varies.
- Optional isotonic probability calibration (`CALIBRATE`, default on) so
  predict_proba is trustworthy for the profit optimizer, with automatic
  fallback to uncalibrated if it errors.
- Removed `penalty='l2'` from LR params (it's the default) to silence the
  sklearn 1.8 deprecation warning.

### Fixed DuckDB timestamp precision mismatch on pull
- `append_duckdb` failed with "Unimplemented type for cast
  (TIMESTAMP_NS -> TIMESTAMP_S) ... expiration_date": the table stored a
  datetime column at second precision while pandas produces nanosecond
  datetimes, and DuckDB can't downcast on INSERT.
- Fix: align the incoming frame's datetime precision to the existing table
  columns (done in pandas, which can downcast) before insert. No DB reset
  needed; robust for any timestamp column, either direction.

### Fixed the training label (single-class -> proper win/loss)
- Diagnosed why train-model failed: the warehouse `won` column is only ever
  `'true'` or NULL (no `'false'`), so the old "keep true/false" logic produced a
  single-class target (all wins) and LogisticRegression couldn't fit.
- Redefined the target in `build_training_table`: a bid is **placed** when
  `bid > 0`; a placed bid **won** (`won=='true'` -> 1) or **lost** (null/blank
  -> 0); no-bid pings are excluded. For auto this yields ~38k wins + ~68k losses
  (~36% win rate) instead of 38k wins only.
- Added `preprocessing.assert_trainable` so a single-class target fails with a
  clear message (+ Slack alert) instead of a raw sklearn error.
- TODO (confirm with Kiran): whether a NULL `won` on a recent, still-settling
  ping means "lost" vs "unresolved" — may warrant excluding the most recent
  window from training.


## 2026-07-07

### Anton model layer (train_and_predict) integrated
- Analyzed the colleague's `train_and_predict/` module (see
  `docs/TRAIN_AND_PREDICT_ANALYSIS.md`).
- Made it integration-ready: added `__init__.py`, fixed `smarthub.io` →
  `smarthub.core.io` and flat sibling imports → relative, wrapped `train.py` in
  `run_training()` + a real `--lead-type-id` CLI, added the `ml` optional extra,
  made `predict.py` import-safe (lazy joblib/mlflow, guarded FastAPI), renamed
  `test_predict.py` → `manual_api_check.py`, removed a duplicate plot, moved
  outputs under `data/`.
- Reconciled the two feature pipelines: `feature_engineering.features` is now
  the single source of truth (`model_feature_columns`, `add_time_features`,
  `derive_serving_features`); training consumes the versioned training table and
  uses a time-ordered split; serving derives features identically.
- New Prefect flow `train-model` (STEP 3) with `log_prints=True`, a summary
  artifact, Slack success/failure notifications, on a new `training` queue.
  Worker image now installs `.[orchestration,ml]`.


## 2026-07-06

### Slack notifications
- Added `smarthub.core.notifications`: best-effort Slack (Incoming Webhook)
  alerts, disabled cleanly when `SLACK_WEBHOOK_URL` is unset, dependency-free.
- Both Prefect flows send a success notification (data-pull: lead type, data
  window, run start/finish, rows, watermark, stored Parquet/DuckDB paths;
  build-features: version, row/feature counts, win rate, window, data range,
  output path) and a failure alert via an `on_failure` hook covering every
  failure point. Manual `smarthub-pull` runs also alert on failure.
- `storage.save_pull` now also returns `duckdb_path` + `parquet_paths`.

### Pipeline ordering guardrails
- data-pull/build-features labelled STEP 1 / STEP 2 (Prefect deployment
  descriptions + tags); build-features fails fast with a clear "run data-pull
  first" message + artifact when no data exists.

### Package reorganised by pipeline stage (hybrid)
- `core/` keeps shared foundations + persistence (storage, io) + transforms.
- `data_pull/` (pull, models, cli, windowing, flow), `feature_engineering/`
  (features, flow), `monitoring/` (the Streamlit app). Removed `data/`,
  `flows/`, `dashboards/`. Updated imports, `pyproject` entry point
  (`smarthub.data_pull.pull:main`), `prefect.yaml` entrypoints, Dockerfile,
  scripts, and docs. 78 tests green, flake8 clean.

### Training features
- Added derived `is_married`, `multi_vehicle`, and one-hot `age_cohort_*`
  bands; `home_owner`/`insured` retained (zero-variance drop now off by
  default — "don't drop anything").


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

### Package restructure
- Grouped modules into subpackages: **`core/`** (config, config_store, paths,
  logging) and **`data/`** (models, data_pull, storage, io, transforms, features,
  cli); `flows/` and `dashboards/` unchanged. All imports updated to the new
  paths (`smarthub.core.*`, `smarthub.data.*`); `smarthub-pull` entry point →
  `smarthub.data.data_pull:main`. Tests/scripts/docs updated; 66 tests green.

### Runtime config (Tier-2)
- `config_store.py` — typed, validated, **versioned** config store backed by the
  **shared Postgres** (`smarthub_config` + history tables); env-scoped
  (staging/prod) with global fallback. Registry covers target CM, bid floor/cap,
  recency window, exploration variance, active model version, source-quality
  threshold, holiday calendar.
- **Single multipage dashboard** (`app.py`, `st.navigation`) — **Leads /
  Monitoring / Config** in one Streamlit app on **http://localhost:8501**
  (`dashboard` service); the three separate dashboard services are consolidated.
- **Config page is password-gated** (`_auth.require_password`, `CONFIG_ADMIN_PASSWORD`
  env); Leads/Monitoring stay open (read-only).
- Secrets & DB connection stay in env (Tier-1); only business knobs are in the DB/UI.

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

