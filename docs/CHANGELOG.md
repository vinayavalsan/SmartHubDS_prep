# Changelog


## 2026-07-09 (later)

### Fix: build-features OOM (SIGKILL -9) — lean reads + worker memory
- **Column-projected storage reads.** `read_duckdb_table` / `read_duckdb_window`
  / `read_parquet_dataset` / `load_leads_raw` / `load_window_raw` take an
  optional `columns=`; build-features now requests only the ~42 raw columns
  `build_training_table` actually consumes (from `PRE_BID_FEATURES` + time/label
  cols), dropping ~12 wide unused ones (`naics_code`, `sic_code`,
  `health_conditions`, `life_*`, `annual_revenue`, outcome cols) *before* they
  enter pandas. Output table unchanged (those columns were discarded anyway).
  The DuckDB window still filters rows in SQL, so both column and row pushdown
  keep peak memory down as the store grows (currently ~800k rows).
- **Worker memory limit.** `docker-compose.prefect.yml` sets `mem_limit: 4g` /
  `mem_reservation: 2g` on the worker (a too-low cap surfaced as flow runs
  killed with exit -9). The Docker Desktop VM must have at least this allocated.

### Data validation on the pull (D1)
- New `smarthub/validation` package — validates each freshly-pulled
  `lead_pings` batch, **warn + report only** (flags bad rows + catalogues
  missing-value patterns; never drops/imputes/caps, never blocks the pull).
  pandera for schema/range/domain rules; a pandas layer for cross-field
  integrity and the null/blank catalogue. Detect-only, never mutates the frame.
- Checks: schema drift, `id` uniqueness, numeric ranges (`age` 1–120,
  `bid`/`exp_rev` ≥ 0, vehicle/driver/claim counts), categorical domains
  (`state`/`gender`/`marital_status`), boolean-ish domains, cross-field rules
  (`current_carrier` while `insured=false`; `won=true` without a bid;
  `erred`+bid; lead-type completeness), and per-column missing rates. Batch
  metrics include `pst_hour` populated %, `exp_rev` coverage, age-implausible
  rate, and `won=false` count (should be 0).
- Wired into the data-pull flow (per-lead-type `data-quality-<type>` Prefect
  artifact + a "Data quality" group in the Slack notification) and the
  `smarthub-pull` CLI (log summary). Threshold in
  `config/smarthub.ini [validation] high_missing_threshold`. New `validation`
  extra (pandera); worker image installs it; degrades gracefully if absent.

### CI/CD + quality gates
- **GitHub Actions** (`.github/workflows/ci.yml`) — runs `flake8` + `pytest` on
  every push / PR across Python 3.11 and 3.12. Installs `.[dev]` + `joblib`
  (the suite runs on the base env; ml/orchestration imports are lazy/guarded).
- **pre-commit** (`.pre-commit-config.yaml`) — flake8 + hygiene hooks
  (trailing-whitespace, end-of-file, yaml/toml, merge-conflict, large-file
  guard). Enable with `pre-commit install`.
- **Fixed the duplicate/conflicting `paramiko`** pin in `requirements.txt`
  (kept `>=3,<4`, matching `pyproject.toml`; dropped the stray `==3.5.1`).

### build-features manual run is now Prefect-free (Vinaya)
- Extracted the build core into `feature_engineering/build.py` — load → build →
  save with **no Prefect dependency** — mirroring `data_pull/pull.py` and
  `train_and_predict/train.py`. `flow.py` is now a thin Prefect wrapper around
  `build.run_build_features` (adds the run artifact, Slack notification, failure
  hook, and the "run data-pull first" guidance artifact).
- `smarthub-build-features` / `python -m smarthub.feature_engineering.build` now
  run **without importing Prefect** — so all three manual stage entry points are
  Prefect-free, and Prefect is only used for the scheduled deployments. Console
  script repointed `flow:main` → `build:main`.

### Fix: registry loads the serving model portably (Docker → local)
- `registry.load_currently_serving_model` no longer trusts the **absolute**
  `model_path` recorded in the manifest (e.g. `/app/data/models/...` from a
  Docker training run) — it now resolves the file via `version_path` in the
  *current* `data/models` location. This fixes a `FileNotFoundError` crash when
  running `train` locally against a model that was previously trained in the
  container. Missing file → graceful `(None, None)` (bootstrap) instead of a
  crash. `predict.resolve_model_uri` was already portable.

### Manual run entry points for every stage
- Each pipeline stage can now be run **directly from the CLI** (not just on the
  Prefect schedule), for ad-hoc runs / backfills / debugging. Added console
  scripts `smarthub-build-features` and `smarthub-train` (alongside the existing
  `smarthub-pull`).
- `feature_engineering/flow.py` gained an argparse `main()`
  (`--lead-type-id` / `--lead-type-name` / `--window-days`), replacing the
  hardcoded `__main__`. train already had `train.py:main`
  (`--lead-type-id` / `--version` / `--no-mlflow`); data-pull already had
  `smarthub-pull`.
- All three manual entry points are Prefect-free (see the build-features core
  extraction above). Pipeline order still applies: data-pull → build-features →
  train-model. README + docs/MANUAL.md document the commands.

### Slack notifications: grouped layout (all three pipelines)
- **Success messages are now grouped** instead of one long flat field list. New
  `notifications.notify_success_grouped` renders a bold **headline** + titled
  sections (bold subheader + 2-column fields) separated by dividers, with long
  paths / definitions in the footer; the lead type moves into the header. A
  text fallback is still sent for no-block clients.
- **train-model** leads with the promotion decision; sections Model /
  Performance / Bid optimizer / Features (count split + optional included /
  excluded). **data-pull** → Volume / Watermark / Run. **build-features** →
  Rows / Coverage / Time mix / Build.
- `train.run_training` now returns `feature_cols`; the flow reports which
  optional features were included vs excluded (mandatory is implied). The old
  flat `notify_success` is unchanged and still used for failures.

### Mandatory / optional feature selection (auto)
- **Feature selection is now configurable per training run.** `[features]`
  section in `config/smarthub.ini` (`auto_optional` / `home_optional`):
  `all` (default) = every optional feature; `none` = mandatory core only;
  a comma list = exactly those optional features (unknown names ignored +
  warned).
- **Mandatory auto core** (SmartFinancial's lead-matching criteria — home owner,
  multiple vehicles, currently insured, accidents, DUI, SR-22, age — plus `bid`)
  is always trained on and cannot be toggled off, enforced in
  `features.mandatory_features` / `model_feature_columns`.
- **Added `sr22_required`** as an auto model feature (was in the training table
  but not the model input); also added to the serving `BidRequest`. Auto-only.
- Toggling changes only what the **model consumes** — every feature is still
  built into the training table, so flipping features needs **no re-pull or
  re-build**. Training and serving both read `model_feature_columns`, so they
  stay in lock-step. Home selection not enabled yet (mandatory core TBD with
  Kiran); home keeps every feature.


## 2026-07-09 (cont'd)

### Model versioning + promotion gate
- **Problem:** `train.py` used to `joblib.dump` straight over
  `data/models/anton_model_<type>.pkl` on every run — no comparison against
  what was already serving, no history, no way back if a training run (e.g. a
  data glitch) produced a worse model.
- **New `train_and_predict/registry.py`.** Every training run is now saved as
  an immutable, numbered, timestamped version:
  `data/models/<type>/v<N>_<UTC timestamp>.pkl` + a `.json` manifest (metrics,
  optimizer summary, lineage, model params). A `current.json` pointer per lead
  type records the model **currently serving** that lead type — the version
  `predict.load_model` actually uses. `registry.rollback(...)` repoints it at
  an earlier version with no retraining.
- **`train.run_training` now gates promotion.** Before saving, it re-scores
  the *currently-serving* model on this run's exact held-out test set (not a
  stored metric from a different data snapshot — the same rows), then
  `registry.decide_promotion(...)` promotes the new ("challenger") model only
  if its ROC AUC doesn't regress beyond `promotion_min_roc_auc_regression` and
  its offline expected profit on that test set is >= `promotion_min_profit_ratio`
  of the currently-serving model's (new `config/smarthub.ini [training]`
  knobs, default 0.01 / 0.98). First model for a lead type is always promoted
  (bootstrap case). A held (non-promoted) run is still saved as a version and
  logged — visible in the Prefect artifact + Slack notification — not
  silently dropped.
- **`predict.load_model` / the FastAPI service** now resolve the model per
  `lead_type_id`: `MODEL_URI` env (explicit override) > `smarthub.ini
  [prediction] active_model_version` (pin a specific version) > the model
  currently serving that lead type (default). Previously `MODEL_URI` defaulted
  to a single hardcoded `auto` path regardless of the requested lead type.
  `/health` now reports the resolved model URI for a given `lead_type_id`.
- `config.model_path()` (the old flat-file path) is kept only for loading a
  pre-existing legacy artifact manually; nothing writes to it anymore.


## 2026-07-09

### Docs: captured 7 Jul meeting insights
- CONTEXT §12 and **MODELING §8 (new "current spec")** now reflect the meeting:
  label/erred logic, Pacific time, `exp_rev`, per-lead-type features,
  `current_carrier` data issue, the "no losses / lower profit" baseline to beat,
  and the full-payload serving decision. MODELING's stale `won='false'` cleaning
  rule is marked superseded.

### Applied DS-meeting (7 Jul) decisions
- **Exclude errored pings** from training (`erred` true → dropped); a placed bid
  (`bid > 0`) with `won` null/blank stays a **loss** (confirmed: null ≠ lost
  *unless* a bid was placed). Non-errored losses are kept — they carry the
  competitive-pricing signal.
- **Pacific time features** (Kiran: UTC flips the date since the call centre runs
  to ~5:30pm PT). `created_dayofweek` now from `pst_date`, `created_hour` from
  `pst_hour` (fallback to `created_at` UTC when the Pacific columns are absent).
  Added `pst_hour` to the ORM/pull.
- **expected_revenue now prefers the backend `exp_rev`** (added to ORM/pull) when
  populated (>0), else falls back to the interim listings-sum — auto-switches as
  `exp_rev` fills in (meeting: the column is now populating).
- **Home vs auto feature split.** Added `HOME_ONLY_FEATURES`
  (`home_property_type`, `num_home_claims`); `model_feature_columns` drops
  home-only for auto and auto-only for home. Added the two home columns to the
  training table (PRE_BID).

### Open items from the meeting (no code yet)
- `current_carrier` populated when `insured` = false (Kiran: bad data, but the
  field is critical for bidding — don't sell to a lead's current carrier).
  Kiran investigating; it's currently excluded from the model feature set —
  revisit once the data issue is resolved.
- Commercial lead type not yet handled (auto/home only for now).
- Serving will use the **full lead payload** (not ping-id DB lookups), pared to
  the needed features — API request schema to mirror the auto/home payloads.


## 2026-07-08

### expected_revenue backend column: confirmed present but unpopulated (blocked)
- `lead_pings.exp_rev numeric(10,2)` exists (the intended authoritative expected
  revenue), but it is 0 non-null / 0 positive across all 572,047 rows. So we keep
  the interim listings-sum for R; switching to `exp_rev` is blocked on the backend
  populating it. Recorded in CONTEXT §4. Verified via SQL on prod.

### Added traffic_tier as a model feature
- Per Kiran, `traffic_tier` (partner-subsource) carries source-quality /
  competitor-bidding signal and belongs in the model — added to the categorical
  feature set. (`source_type_id` stays excluded for cardinality; `account_id`
  stays excluded as a `campaign_id` duplicate.) Check its cardinality on the next
  build — if very high, group rare values (LightGBM's ordinal path handles it
  fine; LR one-hots it).

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

