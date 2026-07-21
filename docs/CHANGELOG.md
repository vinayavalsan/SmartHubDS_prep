# Changelog


## 2026-07-21

### CI/CD: Slack notification on the deploy pipeline (success = image names, failure = stage + log)
- New **`notify`** job in `.github/workflows/ci_cd.yml` — `needs` every
  other job (`isort`, `black`, `flake8`, `pytest`, `build-worker`,
  `build-dashboard`), `if: always()`, gated to a push on the deploy branch
  (`smarthub.etl.pipeline`) only — PRs and other branches stay quiet, same
  as the existing build jobs. Reuses `smarthub.core.notifications`
  (`notify_success`/`notify_failure`) so the message looks and behaves
  exactly like the existing Prefect flow-failure Slack alerts: same webhook
  env var (`SLACK_WEBHOOK_URL`), same Block Kit formatting, same
  disabled-when-unconfigured behavior.
- **On success**: `✅ SmartHub CI/CD Pipeline Succeeded` header, a bulleted
  metadata list (`🔹 Branch` / `Commit` / `Triggered by` / `Environment` /
  `Time`), a `🐳 Pull the latest images:` section with the `docker pull`
  command for each image's `-<sha>` tag (pinned to this exact build), and a
  `🔗 Workflow:` section with the run link — a specific custom layout the
  team asked for after iterating on a couple of earlier formats (a 4-line
  field grid, then a compact grouped layout) live in Slack.
- **On failure**: `🔴 SmartHub CI/CD Pipeline Failed`, the same bulleted
  metadata plus `Failed stage` (and `Skipped as a result` when a failure
  short-circuited downstream jobs), then a trimmed excerpt of the actual
  tool output (the isort/black diff, the flake8 findings, or the pytest
  failure summary) — added an output-capture step to `isort`/`black`/
  `flake8`/`pytest` (each keeps failing normally; it also now exposes an
  `error_message` job output with its last ~4000 chars of output).
  `build-worker`/`build-dashboard` failures fall back to "see this job's
  log" since `docker/build-push-action` doesn't expose its build log as a
  capturable output — flagged explicitly rather than silently omitted.
- New `.github/scripts/notify_ci.py` — reads the six jobs' results/messages
  from env vars and builds this exact Block Kit layout directly (custom
  enough — emoji section headers, non-code-fenced pull commands, a flat
  bulleted metadata section — that it doesn't fit the existing structured
  templates). Sends via a new `notify_raw(payload)` escape hatch added to
  `core/notifications.py`, which still goes through the same webhook
  config check / timeout / best-effort-and-swallow-errors behavior as
  every other function in that module, just without imposing a layout.
  (An intermediate version added a symmetric `notify_failure_grouped` next
  to the existing `notify_success_grouped` — removed again once the final
  layout stopped using it, so no unused public API was left behind.)
  Verified locally (monkeypatched `notifications._post`) for a clean run
  and a mid-pipeline failure with `SLACK_MENTION_ON_FAILURE` configured
  (downstream jobs correctly reported as "skipped as a result", not as
  additional failures).
- **New required GitHub secret**: `SLACK_WEBHOOK_URL` (mirrors the runtime
  env var of the same name; GitHub Actions can't read the runtime one
  directly).
- README's "Testing & linting" and "CI/CD" sections updated: fixed the
  stale `lint-test`/`build-push` job names left over from the user's own
  `isort`/`black`/`flake8`/`pytest`/`build-ready`/`build-worker`/
  `build-dashboard` job split, and documented the new `notify` job and
  secret.

## 2026-07-17

### Enforced formatting/import-order/lint/test checks (pre-commit + CI)
- **Black** (`[tool.black]`, `pyproject.toml`) — line length 88, targeting
  py3.10/3.11/3.12. **isort** (`[tool.isort]`) — `profile = "black"`, line
  length 88, so it sorts/groups imports in a form black already agrees with
  (isort always runs before black, so black gets the final say on any
  remaining formatting either way). `.flake8`'s existing `E203`/`W503`
  ignores already matched black's style, so no lint-config changes needed
  there.
- **Reformatted the whole codebase** with isort then black (39 files) so the
  checks start green — otherwise every future PR would fail on pre-existing
  formatting rather than just new changes. One case needed a manual tweak
  (`flow.py`): black's preferred join of two adjacent f-strings landed at 90
  chars, over flake8's 88 — pulled the f-string into its own `model_label`
  variable instead of fighting the two tools against each other on one line.
  Full suite green after reformatting: 186 passed / 8 skipped (base env),
  194 passed / 2 skipped (`ml`+`explain` extras).
- **`.pre-commit-config.yaml`**: added `isort` and `black` hooks (before
  `flake8`, all on the `pre-commit` stage), and a new local `pytest` hook
  scoped to the `pre-push` stage only (slower than the formatting/lint
  checks, so it runs once per push rather than every commit). Added
  `default_install_hooks: [pre-commit, pre-push]` so one `pre-commit install`
  wires up both — documented as an explicit two-`--hook-type` command too,
  since that key is the more version-proof way to install both.
- **CI** (`.github/workflows/ci_cd.yml`): `lint-test` now runs `isort
  --check-only --diff`, then `black --check --diff`, then `flake8`, then
  `pytest`, across the existing Python 3.11/3.12 matrix — a PR fails if any
  one of the four fails (and `build-push` still `needs: lint-test`, so a
  formatting/import/lint failure also blocks the image build same as a test
  failure always did).
- **`pyproject.toml`**: added `black`/`isort`/`pre-commit` to the `dev`
  extra, so `pip install -e ".[dev]"` is still the one command that covers
  everything CI and the hooks need.
- README's "Testing & linting" and "CI/CD" sections updated to document
  isort/black, the pre-commit + pre-push hook setup, and (in passing) fixed
  a stale `ci.yml` filename reference left over from the CI/CD merge — the
  actual file has been `ci_cd.yml` for a while.


## 2026-07-16

### Lead-type feature selection moved to a single registry
- Replaced the scattered per-type variables and `if auto / if home` branches in
  `feature_engineering/features.py` with one lookup dict, **`LEAD_TYPES:
  dict[int, LeadTypeSpec]`**, keyed by `lead_type_id`. Each `LeadTypeSpec`
  (frozen dataclass) declares that type's `name`, `numeric`, `categorical`, and
  `mandatory` feature sets. `_SHARED_NUMERIC` / `_SHARED_CATEGORICAL` hold the
  columns common to every type so a spec only lists its extras.
- Selection is now **inclusion-based**: `model_feature_columns` /
  `mandatory_features` / `optional_features` read the requested type's spec and
  return exactly its features (ordered by the canonical `MODEL_NUMERIC_ORDER` /
  `MODEL_CATEGORICAL_ORDER`, with any spec-only columns appended sorted so a new
  feature is never silently dropped). No more "load everything, then drop the
  other type's columns."
- **Fail-fast:** an unregistered `lead_type_id` raises
  `ValueError("Unknown lead_type_id …; register it in features.LEAD_TYPES")`
  instead of silently returning a wrong feature set.
- **Adding a lead type = one entry.** e.g. Commercial → add
  `LEAD_TYPES[<id>] = LeadTypeSpec(...)`; no other code path changes.
- Behaviour verified identical for auto (6) and home (1) against a pre-refactor
  snapshot (byte-for-byte). Fixed a latent bug where `_configured_optional`
  used a hardcoded auto mandatory set instead of the per-type one. Added
  `test_unknown_lead_type_raises` and
  `test_new_lead_type_needs_only_one_registry_entry`. Suite: 188 passed.

### Lead-type validation checks now driven by the same registry
- `validation/rules.py` `cross_field_checks` no longer hardcodes
  `lead_type_id == 6 → num_vehicles` / `== 1 → home_property_type`. Each type's
  signature raw columns now live on the registry as
  `LeadTypeSpec.required_raw` (auto → `num_vehicles`, home →
  `home_property_type`), and the completeness checks are generated by iterating
  `features.LEAD_TYPES`. Rule keys are unchanged
  (`auto_missing_num_vehicles`, `home_missing_property_type`), so dashboards and
  history keep working. A new lead type's data-quality rule is now part of its
  single registry entry.


## 2026-07-15

### Restructured the data/ folder
- Raw datasets grouped under **`data/raw_datasets/`** and training tables under
  **`data/training_datasets/<auto|home>/`**:
  - `data/smarthub.duckdb` → `data/raw_datasets/leads.duckdb` (moved + renamed)
  - `data/leads/` (parquet partitions) → `data/raw_datasets/leads/`
  - `data/training/<type>/` → `data/training_datasets/<type>/`
  - `data/models/` and `data/model_evaluation/` unchanged (artifacts, not
    datasets).
- Code defaults updated: `config.StorageSettings` (`DUCKDB_PATH` /
  `PARQUET_DIR`), `storage.duckdb_path()`, `io.TRAINING_DIR`. Existing on-disk
  data was physically moved, so serving + dashboards keep working. `.env` /
  `.env.example` defaults and the README storage/layout docs updated. All
  paths still overridable via `DUCKDB_PATH` / `PARQUET_DIR`.

### Task config moved from INI to YAML
- `config/smarthub.ini` → **`config/smarthub.yaml`** (per-stage mappings:
  `data_pull` / `validation` / `feature_engineering` / `features` / `training` /
  `prediction` / `explain`). `core/task_config.py` now loads YAML
  (`yaml.safe_load`) instead of `configparser`, **keeping the exact same public
  API** (`get` / `get_int` / `get_float` / `get_bool` / `reload` /
  `config_path`), so no call site changed — only the backend + the file. Native
  YAML booleans are honored by `get_bool`; `SMARTHUB_TASK_CONFIG` still
  overrides the path.
- Added `PyYAML>=6.0` to base deps (pyproject + requirements). Updated
  `test_task_config.py` / `test_config.py` to write YAML; refreshed the config
  examples/paths in README + docs/MANUAL + docs/validation_rules. Full suite
  green (186 passed / 6 skipped on the base env).

### Fix: corrupt `current.json` crashed an entire training run
- A scheduled `train-model` Prefect run failed with a raw
  `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
  from `registry.currently_serving_version`, propagating uncaught through
  `train._evaluate_currently_serving_model` and taking down the whole flow —
  not just skipping the (optional) promotion-gate comparison it was supposed
  to guard. Root cause: `current.json` existed but was empty/truncated, most
  likely from a process being interrupted mid-`write_text()` (plain
  `Path.write_text()` isn't atomic — a kill/crash between opening and
  finishing the write can leave a 0-byte or partial file).
- **Fix 1 — atomic writes.** New `registry._atomic_write_text()` (write to a
  `.tmp` file, then `os.replace`) used for every manifest/pointer write
  (`save_version`, `promote`'s manifest + `current.json` writes), so a
  killed process can no longer corrupt these files at all.
- **Fix 2 — tolerate a corrupt pointer if one already exists.**
  `currently_serving_version()` now catches `JSONDecodeError`/`OSError`,
  logs a warning, and returns `None` (same as "nothing promoted yet")
  instead of raising — defense in depth for pointer files that predate this
  fix, or corruption from any other cause.
- **Fix 3 — the real bug: the registry call was outside the try/except.**
  `train._evaluate_currently_serving_model`'s docstring already promised
  "any failure here... is treated as nothing comparable currently serving",
  but the `registry.load_currently_serving_model(...)` call sat *before*
  the `try:` block, so it alone couldn't honor that promise. Moved inside —
  now truly no failure mode in this optional comparison can crash training.
- **Tests:** `tests/test_registry.py` — corrupt pointer -> `None` (with the
  warning logged) at both `currently_serving_version` and
  `load_currently_serving_model`, and a no-`.tmp`-file-left-behind check on
  `promote()`. New `tests/test_train.py` (sklearn-gated, since `train.py`
  imports it at module level via `metrics.py`) — the registry-load-raises
  case (the exact bug) and a schema-mismatch case both degrade to
  `(None, None)` instead of propagating. 191 passed with `ml`+`explain`
  extras installed (previously 185).

## 2026-07-14

### `/explain_bid`: bid curve + fixed a real LLM reasoning mistake
- Live-tested `/explain_bid` on a real lead and found the LLM's prose
  reasoned backwards about the `bid` factor's SHAP sign — implying a *lower*
  bid would have won more often, which the model's monotonic bid constraint
  rules out. It wasn't inventing a number, just garbling the qualitative
  direction because it had no real numbers to reason from besides the one
  chosen bid.
- **Fix: give the LLM real numbers instead of letting it guess.** New
  `predict.bid_curve_around(row, model, expected_revenue, min_bid, max_bid,
  bid_step, center_bid, n_points=3)` computes predicted win rate + expected
  profit at a few bid points bracketing the chosen bid (one vectorized
  `predict_proba` call, same pattern as `optimize_bid_for_row`) — "the shape
  of the market" around the bid (Kiran, docs/CONTEXT.md §7), not just the
  single winning number. `explain.explain_bid` now computes this (skipped
  for `cold_start_fallback`, which has no model) and both returns it as
  `bid_curve` and feeds it into the LLM prompt as a fact ("Nearby bids
  explored: $X -> Y%, ...").
- **Also added an explicit guardrail line to the prompt**: "predicted win
  rate never decreases as the bid rises... never claim a lower bid would win
  more often than a higher one" — belt-and-suspenders alongside the real
  numbers now available.
- **Tests:** `bid_curve_around` (symmetric spanning, edge-clipping without
  duplicate bids, empty on no-viable-bid/NaN center) in
  `test_train_and_predict.py`; prompt-guardrail and bid-curve-rendering
  tests, plus updated `explain_bid` mocks, in `test_explain.py`. 185 passed
  with `ml`+`explain` extras installed.

### Fix: /explain_bid 500'd on numpy scalars (FastAPI couldn't JSON-encode them)
- Live-tested `/explain_bid` and hit `TypeError: 'numpy.int64' object is not
  iterable` from FastAPI's `jsonable_encoder`. Cause: `explain_row`'s
  `top_factors[].value` came straight from `frame.iloc[0][name]`, which is a
  numpy scalar (`int64`/`float64`) for numeric feature columns — this
  particular FastAPI/pydantic version combination can't encode those.
- Added `explain._to_native()` (`np.generic` -> `.item()`) and applied it to
  every `top_factors[].value`. New regression tests:
  `test_to_native_converts_numpy_scalars` (pure) and an assertion in
  `test_explain_row_returns_top_factors_ranked_by_shap` that no `value` is a
  `np.generic`, so this can't silently reappear.

### Cold-start + exploration bidding policy (`predict.decide_bid`)
- **The real gap Kiran flagged (docs/CONTEXT.md §3):** "'Recent' must be
  explicitly defined as a configurable window... when there is no recent
  data, the bidding pattern must be explicitly articulated... e.g. a defined
  exploration schedule or fallback bid" — not emergent/random behavior.
  `exploration_variance_pct` existed in the ini since the config-tiering pass
  but was never read anywhere in code; there was no cold-start detection at
  all (a lead type with no promoted model just 500'd).
- **New `predict.decide_bid`** — the one bidding decision, always returning
  an explicit `decision_path` (`"cold_start_fallback" | "exploration" |
  "model"`) + plain-English `decision_reason`:
  - `cold_start_fallback` — no model ever promoted for this lead type yet;
    bids a fixed, configurable fraction of the way from the floor to the
    CM-respecting ceiling (new `cold_start_fallback_bid_pct` ini key, default
    50%) instead of erroring. Self-terminating once a first model promotes.
  - `exploration` — a **deterministic, reproducible** explore/exploit
    schedule instead of a per-request coin flip: leads are bucketed by
    hour-of-week (0-167), and 1-in-`N` buckets (`N = round(1 /
    exploration_variance_pct)`) are scheduled probe slots, perturbing the
    optimum bid by `exploration_variance_pct` and alternating direction each
    time a bucket triggers — the same lead always gets the same
    explore/exploit answer, so it's auditable after the fact.
  - `model` — the normal profit-maximizing bid; flags `model_data_age_days`
    and a "due for retraining" note if the currently-serving model's
    training data is older than the new `recency_window_days` ini key
    (default 30) — the explicit, named "recent" window Kiran asked for.
    Informational only; doesn't change the bid itself.
- **New `predict.load_model_and_manifest`** — same resolution order as
  `resolve_model_uri` but returns `(None, None)` instead of raising when
  nothing's ever been promoted, so `decide_bid` can detect true cold start.
- **Wired into both `/recommend_bid` and `/explain_bid`** (previously
  `/recommend_bid` called the raw optimizer directly). `/explain_bid` now
  runs the identical policy, so its `decision_path`/bid always matches what
  live serving would do; a `cold_start_fallback` bid skips SHAP/the LLM
  entirely (no model to explain), and an `exploration` bid's LLM prompt gets
  a factual note about the probe so the prose reflects it.
- **New ini keys** (`config/smarthub.yaml [prediction]`):
  `recency_window_days` (30), `cold_start_fallback_bid_pct` (0.50).
  `exploration_variance_pct` (0.10, pre-existing) is now actually read.
- **Fix: SHAP base value was in log-odds space, not a probability.**
  `shap.TreeExplainer` on `LGBMClassifier` works in margin/log-odds space —
  `explain.py`'s `base_win_rate` was passing that straight through
  (observed in the wild as e.g. `-2.05`, not a valid 0-1 rate). Now converted
  through a sigmoid so it's a genuine probability comparable to
  `predicted_win_rate`; per-feature `shap` contributions are left in
  log-odds units (direction/ranking are unaffected by that monotonic
  transform) and documented as such.
- **Tests.** `tests/test_train_and_predict.py`: `cold_start_fallback_bid`,
  `exploration_slot` (determinism + alternating direction + disabled at
  `variance_pct=0`), `model_recency` (stale/not-stale/missing-lineage),
  `decide_bid` (all three paths + the stale-model note),
  `load_model_and_manifest` (cold start + currently-serving). Verified
  against the base test env and a full `ml`+`explain`-extras env
  (178 passed, 2 skipped).

### Bid explanations: `/explain_bid` (SHAP + local LLM via Ollama)
- **New `train_and_predict/explain.py`.** Offline/on-demand "why did Anton bid
  $X for this lead" explanations. Explains the model's prediction *at the
  chosen bid* with `shap.TreeExplainer` factor attributions
  (`model_type=lightgbm` only for now; handles both a plain Pipeline and a
  `CalibratedClassifierCV`, averaging SHAP values across its 3 CV folds).
  A small local LLM, called over Ollama's HTTP API (`POST /api/generate`),
  turns the ranked factors into 2-3 plain-English sentences from a tightly
  templated, "use only these facts, do not invent numbers" prompt — a
  formatting task, not a reasoning task, to keep a small model from
  hallucinating numbers. Ollama being unreachable degrades to a clear
  fallback message rather than an error; the numeric factors are unaffected.
- **New `predict.py` endpoint**, `POST /explain_bid` — same request shape as
  `/recommend_bid`; 503 if the `explain` extra isn't installed, 400 on a bad
  lead_type/model mismatch (e.g. a non-LightGBM model).
- **New `pyproject.toml` extra**, `explain = ["shap>=0.44"]` (also needs `ml`
  to explain a trained model, plus a running local Ollama server — not a
  Python dependency).
- **New `config/smarthub.yaml [explain]` section** — `llm_model`
  (`qwen2.5:1.5b-instruct` default), `ollama_host`, `top_n_factors`,
  `timeout_seconds`.
- **Tests (`tests/test_explain.py`).** Prompt formatting, Ollama success +
  graceful-fallback-on-unreachable (mocked `requests.post`), and the
  no-viable-bid short-circuit all run in the base env; SHAP/LightGBM-gated
  tests (`pytest.importorskip`) fit a real tiny LightGBM pipeline (plain and
  calibrated) and verify SHAP ranking, fold-averaging, and the
  non-LightGBM-model rejection. Verified against both the base test env and
  a full `ml`+`explain`-extras env (163 passed).


## 2026-07-09 (later)

### Local build vs server pull (SMARTHUB_ENV)
- `SMARTHUB_ENV` now decides image source. `local` (default) **builds from
  source** via a new `docker-compose.local.yml` override (adds `build:`) — no
  pull, no Watchtower. `staging`/`prod` **pulls** the CD-built images from
  Docker Hub and runs Watchtower. `install.sh` picks the right compose files +
  flags automatically.
- Base `docker-compose.prefect.yml` is now pull-only (removed `build:` from
  worker/dashboard); Watchtower moved behind a `prod` compose profile so it
  never runs locally and clobbers a local build. `.env.example` documents
  `SMARTHUB_ENV`.

### CI + CD merged into one gated pipeline
- Folded the build/push into `ci.yml` (renamed **CI/CD**) as a `build-push` job
  with `needs: lint-test` — images now build **only after the full test matrix
  passes**, once, instead of a separate `cd.yml` racing CI in parallel (which
  ran the tests twice). `build-push` is gated to pushes on the deploy branch;
  `cd.yml` removed. `cancel-in-progress: false` so a deploy is never interrupted.

### CD: automated image build + push + server auto-pull
- **`.github/workflows/cd.yml`** — on every push to `main`: run flake8 + pytest,
  then build the `worker` and `dashboard` images and push to **Docker Hub** as
  `:latest` and `:<sha>` (buildx + GH Actions layer cache). Secrets:
  `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
- **Single Docker Hub repo, tag-prefixed** (free tier = 1 private repo). Both
  images live in `${DOCKERHUB_USERNAME}/smarthub`, split by tag prefix:
  `worker-latest`/`worker-<sha>` and `dashboard-latest`/`dashboard-<sha>`.
  CD builds use separate buildx cache scopes so they don't thrash each other.
- **Server auto-pull (Watchtower).** `docker-compose.prefect.yml` worker +
  dashboard reference `${IMAGE_REPO}:worker-${IMAGE_TAG}` /
  `:dashboard-${IMAGE_TAG}` (`build:` kept for local) and carry the Watchtower
  enable label; a new `watchtower` service polls the registry (default 300s) and
  recreates only those two containers on a new `-latest`. No inbound access
  required. The worker re-runs `prefect deploy --all` on boot.
- `.env.example` documents `IMAGE_REPO` / `IMAGE_TAG` / `WATCHTOWER_POLL_INTERVAL`
  (+ private-repo `docker login` note); README gains a CI/CD section.

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
  `config/smarthub.yaml [validation] high_missing_threshold`. New `validation`
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
  section in `config/smarthub.yaml` (`auto_optional` / `home_optional`):
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
  of the currently-serving model's (new `config/smarthub.yaml [training]`
  knobs, default 0.01 / 0.98). First model for a lead type is always promoted
  (bootstrap case). A held (non-promoted) run is still saved as a version and
  logged — visible in the Prefect artifact + Slack notification — not
  silently dropped.
- **`predict.load_model` / the FastAPI service** now resolve the model per
  `lead_type_id`: `MODEL_URI` env (explicit override) > `smarthub.yaml
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
  `min_source_quality`), **task configs → `config/smarthub.yaml`** with
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

