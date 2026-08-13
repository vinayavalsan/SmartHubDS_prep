# Changelog


## 2026-08-13

### Canonical package version + per-prediction version provenance
- **`smarthub.__version__` is now canonical and runtime-resolved.** It reads the
  installed package metadata (`importlib.metadata.version("smarthub")`), so
  `pyproject.toml [project].version` is the single source of truth; the literal
  in `src/smarthub/__init__.py` is only a fallback for an un-installed checkout.
- **Semver policy documented** (`src/smarthub/__init__.py`): backward-incompatible
  changes to the serving contract — including the raw `field_registry` and the
  `feature_registry` (feature set/ordering/semantics that change how a stored
  model scores) — warrant a MAJOR bump; compatible additions MINOR; fixes PATCH.
- **Every prediction now records the package version.** Added a `package_version`
  column to `smarthub_prediction_log`, auto-filled from `smarthub.__version__` in
  `_row_values` (so success/error, `recommend_bid`/`explain_bid`, and batched
  writes all capture it), with a `log_prediction(package_version=...)` override.
  Pairs with the existing `model_version` so a row is traceable to **both** the
  code contract and the model that produced it. Existing tables pick up the new
  column via the idempotent `_add_missing_columns` migration — no manual DDL.

### Dual-output logging (readable `docker logs` + machine-parseable file)
- **`logging_utils` now writes two sinks:** human-readable text on stdout (so
  `docker logs` is normal again) and, when `SMARTHUB_LOG_JSON_DIR` is set, the
  rich JSON schema to `<dir>/<service>.jsonl`, rotated daily with
  `SMARTHUB_LOG_JSON_RETAIN_DAYS` (default 30) days retained.
- **Compose wiring:** the five app services (`worker`, `serve`, `shap-worker`,
  `slo-alerts`, `dashboard`) set `SMARTHUB_LOG_JSON_DIR=/app/logs/app` and mount
  `./logs:/app/logs`. Note: `LOG_FORMAT` must not be `json` in `.env` or stdout
  stays JSON (env_file overrides compose).


## 2026-08-12

### Config-driven campaign scoping for training (unblocks `home`)
- **Removed the hardcoded `campaign_id` allow-list** from the feature registry
  (`FeatureSpec("campaign_id", training_include_values=frozenset({13, 3, 6, 16}))`).
  It was an auto-campaign filter declared for `lead_types={"auto", "home"}`, so it
  was silently applied to `home` too. Home's campaigns are different (15/4/27/12),
  so the filter collapsed the home training table to **169 rows, all wins** →
  training aborted with "only ONE target class". Auto was unaffected because its
  campaigns overlap the list.
- **Campaign scoping now lives in config** — `config/smarthub.yaml` →
  `feature_engineering.training_campaign_ids` (list of ints). **Empty `[]` (the
  default) = all campaigns**; a non-empty list keeps only rows whose `campaign_id`
  is in the list. Change it without touching code (the `./config` mount picks it
  up on a worker restart).
- **Wiring** — new `training_campaign_ids()` in `core/config.py` (mirrors
  `training_window_days()`); `build_training_table()` gained a `campaign_ids`
  param that replaces the registry filter; `run_build_features` passes the config
  value in.
- **Result** — rebuilt `home` training table now has both classes
  (~50.6k losses / 20.1k wins over ~70k rows) and trains through; `auto` unchanged.

### Training resource caps + structured logging (infra)
- **`_feature_breakdown` fixed** (`train_and_predict/flow.py`) — the feature-
  registry refactor removed `mandatory_features`/`optional_features`, but the
  post-training report still called them (crashing with `AttributeError`). It now
  reports registry coverage via `model_feature_columns`.
- **Training can't starve/​hang the box** — `config/training.yaml` lightgbm
  `n_jobs: -1 → 4`; the `worker` service gets `cpus: 4`,
  `OMP/OPENBLAS/MKL_NUM_THREADS=4`, and `OMP_WAIT_POLICY=PASSIVE`, and mounts
  `./config` read-only. Fixes runs that pinned all 8 cores and never finished.
- **Structured JSON logs** — `core/logging_utils.py` adds an opt-in JSON mode
  (`LOG_FORMAT=json`) with per-line `service`, context binding, and an
  `error.fingerprint` + full `error.stack`, kept alongside the existing text
  format. App services set `LOG_FORMAT=json` + `SMARTHUB_SERVICE`.
- **Persistent logs** — `tools/log_collector.py` + `deploy/smarthub-logs.service`
  write date-stamped, container-tagged JSON log files (combined + per-container)
  with 30-day retention.
- **Noise** — silenced the repetitive LightGBM `TreeExplainer` `UserWarning` in
  `shap_explain.py` that flooded the shap-worker logs.


## 2026-08-03 (later)

### Local vs production model storage + MLflow separation
- **New `model_storage.py`** — a `ModelStore` abstraction with two backends:
  `FilesystemModelStore` (local dev/training, the default) and `S3ModelStore`
  (S3 or any S3-compatible endpoint via boto3, `endpoint_url`-configurable for
  MinIO, with a download+cache `local_path` for serving). boto3 is imported
  lazily so local/CI never needs it.
- **Config schema** — `training.yaml` gains `storage.production`
  (`backend`/`bucket`/`prefix`/`endpoint_url`/`region`/`root`) and
  `mlflow.production` (`tracking_uri`/`experiment_name`/`registered_model_name`).
  `TrainingConfig` parses them (env overrides: `SMARTHUB_S3_ENDPOINT_URL`,
  `SMARTHUB_PRODUCTION_STORAGE_BACKEND`, `SMARTHUB_MLFLOW_PROD_TRACKING_URI`)
  and exposes `local_model_store()` / `production_model_store()`. **Defaults
  keep today's local-only behaviour** — production is opt-in.
- **Registry routing** — training still writes every run to local storage;
  `promote()` now *publishes only the promoted model* (artifact + manifest +
  serving pointer) to production storage, and (when a production MLflow
  `tracking_uri` is set) logs + registers it in production MLflow. Production
  MLflow is best-effort (a promotion isn't failed by an audit-registry outage);
  a failed production-storage publish does raise, since that's serving-critical.
- **Production-aware serving** — the serving-read helpers
  (`currently_serving_version`, `load_manifest`, `serving_model_path`,
  `load_currently_serving_model`) prefer the production store's pointer /
  manifest / artifact when configured, and fall back to local otherwise.
  `predict.py` loads the serving model through `serving_model_path`, so it
  transparently pulls from S3/MinIO (cached) in production.
- **AWS S3 or MinIO, env-selectable (default MinIO).** The same `S3ModelStore`
  serves both: a non-empty `SMARTHUB_S3_ENDPOINT_URL` targets an S3-compatible
  service (MinIO, default in Docker), an **empty** endpoint targets real AWS S3
  (boto3 talks to AWS directly). Bucket/prefix/region are env-overridable
  (`SMARTHUB_S3_BUCKET`, `SMARTHUB_S3_PREFIX`, `SMARTHUB_S3_REGION` /
  `AWS_DEFAULT_REGION`); creds via the standard `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` (or an instance role on AWS). Switch to AWS with no
  code change: set `SMARTHUB_S3_ENDPOINT_URL=`, real creds, and the bucket.
- **Infra** — `boto3` added to the `ml` extra; docker-compose gains a `minio`
  service (+ one-shot `minio-init` to create the bucket) and threads the S3 env
  (`SMARTHUB_PRODUCTION_STORAGE_BACKEND`, `SMARTHUB_S3_ENDPOINT_URL`, AWS creds)
  into `worker` and `serve`. Flip `SMARTHUB_PRODUCTION_STORAGE_BACKEND=s3` to
  enable production publishing/serving.
- The three promotion modes (automatic/manual/disabled) and the two-tier
  versioning (immutable `run_<ts>` ids + `auto_vN` assigned only on promotion)
  are unchanged — this change is purely about *where* promoted models and their
  MLflow lineage are stored.


## 2026-08-03

### Structured SHAP payload for every prediction — full feature contributions, no LLM
- The stored `shap_explanation` payload now carries the SHAP spec shape:
  `base_prediction` (average win rate), `prediction` (the predicted win rate
  for the lead), and `feature_contributions` — **every** model feature's
  contribution as `{feature, value, contribution}`, sorted by `|contribution|`
  — instead of only the top-N. The existing `top_factors`/`base_win_rate`
  top-N view is retained for backward compatibility.
- `prediction` is **reconciled to the calibrated served win rate** (equals the
  `recommended_bid_predicted_win_rate` column), so the payload and the log
  column agree; the served value is threaded into the background explain task.
  The SHAP-reconstructed (uncalibrated) win rate is a fallback only for the
  raw-lead dev path. Contributions therefore describe the uncalibrated model
  output and won't exactly reconstruct a calibrated `prediction` — documented
  in `PREDICTION_LOG_SCHEMA.md` §4.
- `shap_explain._shap_for_row` now also returns the reconstructed predicted win
  rate (base margin + summed contributions → sigmoid); new `_all_contributions`
  emits the untruncated per-feature list. `explain_row` /
  `explain_prepared_row` return the superset payload.
- The automatic production path stays async and LLM-free: `/recommend_bid`
  computes and stores the payload in a post-response background task
  (`with_llm=False`), so prediction latency is unaffected. An LLM narrative is
  only ever produced on explicit `/explain_bid` request — and because the full
  contribution set is persisted, a natural-language explanation can be
  generated later from the stored payload without recomputing SHAP.
- Cold-start / no-viable-bid / non-LightGBM results now still write the spec
  shape (empty `feature_contributions` + a reason), so every logged prediction
  carries the payload keys. Non-finite SHAP floats are sanitized to `null`
  before serialization/persistence (`_json_safe`).
- Docs: `PREDICTION_LOG_SCHEMA.md` §4 updated to the new payload shape.


## 2026-07-30

### Separated prediction and explanation: `/explain_bid` consumes a logged prediction (no bid recompute)
- `/explain_bid` used to re-run the whole bidding policy (`explain.explain_bid`
  → `predict.decide_bid` → the candidate-bid optimizer sweep) just to re-derive
  a bid it usually already had -- duplicated logic, extra model execution, and a
  risk that the explanation didn't match the logged prediction (e.g. if a newer
  model got promoted in between).
- **New production explanation flow — consume, don't recompute.** `/explain_bid`
  now takes `{prediction_id}`, loads the persisted prediction, loads the **exact
  model version that served it** (by the logged `model_uri`), runs SHAP (+
  optional LLM narrative / nearby-bid curve) on the feature row it actually
  scored, and **persists the explanation back onto that same log row** -- it
  never re-decides the bid. So the explanation always corresponds to the logged
  bid and the model that produced it. Returns 404 for an unknown id and degrades
  gracefully (200 + fallback message) for a non-LightGBM model instead of 500.
- **New `server/explain.py`** — `PredictionOutput` + `explain_from_prediction()`,
  the reusable "explain an already-computed prediction" entry point. The SHAP /
  LLM logic itself stays in `train_and_predict` (`shap_explain` / `llm_explain`);
  added `shap_explain.explain_prepared_row()` to SHAP an already-prepared feature
  row (coercing numerics, since a log-reloaded row is JSON-serialized).
- `/recommend_bid`'s background SHAP enrichment now flows through the same
  `explain_from_prediction` path (still `with_llm=False`, still after the
  response is sent) -- one shared consume path for both endpoints.
- **API code consolidated in `server`.** `manual_api_check.py` moved into
  `server/` (now demonstrates the recommend → explain-by-id flow); new
  `server/app.py` is the stable serving entrypoint
  (`uvicorn smarthub.server.app:app`), and `Dockerfile.serve` + the run docs were
  updated to it. The raw-lead `explain.explain_bid` orchestrator stays in
  `train_and_predict` as the reusable local/dev path (the deferred "local mode").
- **Contract change:** `/explain_bid` now expects `prediction_id`, not a full
  lead payload -- callers must be updated. A raw-lead "predict then explain"
  local mode is deferred to a follow-up.

### Consolidated raw-data validation into `data_pull` + a centralized raw-field registry
- The validation layer moved out of its own `smarthub/validation` package into
  `smarthub/data_pull` (`validation_rules.py`, `validation_runner.py`,
  `validation_report.py`), since raw validation always runs immediately after
  extraction. The old package was removed; call sites (`pull.py`, `flow.py`)
  and `tests/` were repointed. Behaviour is unchanged.
- **New `data_pull/field_registry.py` — the single source of truth for every
  raw `lead_pings` field.** Each of the 54 pulled fields is declared exactly
  once (`RawFieldSpec` = `DataSourceSpec` + `ValidationSpec` + `lead_types` +
  `pii`), via small `_num`/`_cat`/`_bin`/`_dt` constructors. Mirrors the
  `feature_engineering.feature_registry` pattern.
- **Extraction is now registry-driven.** `models.LEADS_COLUMNS` is derived from
  `query_builder.leads_column_names()` (new `data_pull/query_builder.py`, which
  reads the registry) instead of a hand-maintained 54-line ORM tuple. The ORM
  is kept — `query_builder` resolves names to columns via `getattr(LeadPing,
  name)`, so parameterized/dialect-safe queries are preserved. A golden test
  pins the generated column set + order to the original pull.
- **Generic validation is now registry-driven.** `leads_schema()` builds the
  pandera schema from each field's `ValidationSpec` (`in_range` / `ge` /
  `isin` / `unique`); `batch_metrics` and `EXPECTED_COLUMNS` read the registry
  too. The duplicated `NUMERIC_RANGES` / `NON_NEGATIVE` / gender+marital domain
  constants were deleted from `validation_rules.py`.
- **Custom validation via `custom_rule`.** `validate_us_state` (+ `US_STATES`,
  a `ValidationResult` type) live in a new leaf module `validation_custom.py`
  — a leaf so the metadata-only registry can reference the function without an
  import cycle. `state`'s `ValidationSpec.custom_rule` points at it, and the
  runner executes custom rules as plain pandas, so they run even when pandera
  isn't installed (previously the state check was pandera-only).
- **Per-lead-type scoping of the high-missing catalogue.** `validate_leads`
  takes an optional `lead_type_id`; when given, columns that don't apply to
  that type (via `field_registry.columns_not_for_lead_type`) are no longer
  flagged as high-missing — so an auto pull stops reporting empty `home_*`
  columns (the noise noted in `validation_rules.md` §9). Threaded through
  `pull.py` and `flow.py`; `lead_type_id=None` (all-types pull) is unscoped,
  matching prior behaviour. The full per-column `missing` rates are unchanged.
- **Output preserved.** The existing validation tests pass unchanged (same
  detected issues, same check labels, same report shape); new tests cover the
  registry, `query_builder`, the custom rule, and the per-type scoping.

### Training promotion tolerates a stale / incompatible currently-serving model
- `train._evaluate_currently_serving_model` re-scores the currently-serving
  model on the new run's test set for the promotion comparison. When that
  model's saved `feature_cols` reference columns the current training data no
  longer produces (e.g. the old one-hot `age_cohort_*` / `age_missing` champion
  vs. the new single `age_cohort`), it used to raise `KeyError` and fail the
  whole training run — a feature-migration deadlock (you couldn't promote a
  new-schema model because the comparison against the old one crashed).
- It now treats a schema-drifted (or otherwise unscoreable) champion as **not
  comparable**: logs a warning and returns `None`, so `decide_promotion` judges
  the challenger on the absolute gates alone (same as the first-model case).
  Registry-load errors still propagate. The corresponding test was updated and
  a scoring-failure case added.


## 2026-07-29

### `age_cohort_*` one-hot columns collapsed into a single categorical `age_cohort`
- Feature engineering was manually one-hot-encoding age bands into seven
  `age_cohort_<band>` 0/1 columns, then declaring them as *numeric* model
  features -- the only feature built this way; every other categorical
  (`state`, `gender`, `traffic_tier`, ...) is left as a raw string and
  encoded downstream in the model pipeline, per model type. That was also a
  quiet correctness smell: the seven dummy columns went through the numeric
  preprocessing branch, so logistic regression's `StandardScaler` was
  standardizing 0/1 indicators as if they were continuous.
- `features.py`'s `_derive_features` now writes one categorical `age_cohort`
  column (values: `under_18`, `18_24`, `25_34`, `35_44`, `45_54`, `55_64`,
  `65_plus`, or null) instead of the seven dummy columns. Encoding is left to
  the training step -- `OneHotEncoder` for logistic regression,
  `OrdinalEncoder` for XGBoost/LightGBM -- exactly the machinery already used
  for every other categorical, no new preprocessing code needed.
- **`age_missing` folded into `age_cohort`, removed as a standalone feature.**
  A missing/implausible age now simply leaves `age_cohort` null;
  `preprocessing.normalize_model_frame` maps that to the same `"NAvail"`
  sentinel every other categorical column's missing values already get. The
  raw `age` column's own `-1` sentinel fill (Vinaya's missingness-as-signal
  decision) is unchanged -- this only removes the *duplicate* signal that
  existed between the dedicated flag and the (previously) all-zero dummy
  columns.
- Registry updates (`features.py`): `age_cohort` added to `MODEL_CATEGORICAL_ORDER`
  / `_SHARED_CATEGORICAL`; `age_missing` and the seven `age_cohort_*` names
  removed from `MODEL_NUMERIC_ORDER` / `_SHARED_NUMERIC`; auto's mandatory
  set now lists `age_cohort` in place of `age_missing` + the seven dummy
  names. No changes needed in `predict.py`/`explain.py`/`preprocessing.py` --
  they all read the feature list from `config.feature_columns()`, so the new
  schema flows through automatically.
- `build.py`'s `age_missing_rate` build-metadata metric now reads
  `age_cohort.isna().mean()` instead of the old flag column; the manifest key
  name is unchanged so `flow.py`'s Prefect artifact / Slack notification
  don't need updating.
- **Bonus for `/explain_bid`:** SHAP attributions for a lead's age used to be
  split across up to seven `age_cohort_<band>` entries in `top_factors`. With
  a single `age_cohort` column, that's one clean, interpretable attribution
  per lead instead of a fragmented one.
- **Breaking change to the model input schema** -- this is not a hot-swappable
  code change. The currently-promoted model's pickled pipeline was fit on the
  old 7-dummy-column schema; deploying this requires rebuilding the training
  table, retraining, and promoting a new model version alongside the code
  change.
- Updated: `config/smarthub.yaml`'s mandatory-features comment,
  `docs/MODELING.md` §8's feature/missing-values description,
  `tests/test_features.py` (3 tests rewritten + 1 reference fixed),
  `tests/test_train_and_predict.py` (3 assertions updated), `tests/test_build.py`
  (fixture + new assertion). Full suite: 224 passed both before and after this
  change (6 pre-existing failures in `test_explain.py`/`test_predict_logging.py`
  are unrelated -- a `build_model()` missing-kwarg signature drift already
  present on this branch -- confirmed identical with this change stashed out).
  Lint clean (isort/black/flake8).


## 2026-07-23

### `/recommend_bid`'s entire log write moved off the response path; new CM column; renamed profit field
- Three related asks, all in the prediction-logging table: add a
  contribution-margin column, guarantee logging never delays a bid, and
  rename a confusingly-worded field.
- **New column**: `recommended_bid_predicted_cm` =
  `recommended_bid_predicted_profit / expected_revenue`. Added to
  `prediction_log_schema.py`'s table + `log_prediction(...)`, computed via a
  new `_predicted_cm()` helper in `predict.py` (null-safe: `None` whenever
  there's no predicted profit to divide — cold start, no viable bid, or an
  error before a bid was reached). Wired into both `/recommend_bid` and
  `/explain_bid`'s logging calls.
- **Rename**: `recommended_bid_expected_profit` → `recommended_bid_predicted_profit`,
  at the API-response/logging boundary only — matches
  `recommended_bid_predicted_win_rate`'s "predicted_" naming, so both
  model-predicted metrics use the same word instead of two different ones
  ("expected" vs. "predicted") for the same idea. Scoped deliberately: the
  internal training-time "expected_profit" family in `optimizer.py` /
  `optimizer_evaluation.py` (`current_bid_expected_profit`,
  `expected_profit_lift`, `recommended_bid_total_expected_profit`, etc.) was
  **not** renamed — that's a separate, much wider-reaching convention baked
  into already-saved model manifests (`optimizer_summary`) and
  `registry.py`'s promotion-gate logic, out of scope here.
  `predict.decide_bid()` now translates the one key it cares about right
  after calling into `optimizer.py` (`result["recommended_bid_predicted_profit"]
  = result.pop("recommended_bid_expected_profit")`), keeping the two
  modules' naming decoupled at that one seam.
- **`/recommend_bid`'s entire log-row insert is now a `BackgroundTasks` job**,
  not just the SHAP enrichment from earlier this week. `prediction_id` is
  generated in the route itself (`uuid.uuid4()`, not by the store) and
  returned in the response immediately; `log_prediction(...)` — with every
  derived metric, including the new `recommended_bid_predicted_cm` — runs
  strictly *after* the response has already been sent. Explicit ask: "ensure
  this logging does not delay bid predictions... calculate all extra
  metrics after placing the bid." The SHAP-enrichment task is scheduled
  right after the logging-insert task, so it always finds its row already
  there (Starlette runs `BackgroundTasks` in registration order).
  - **Trade-off, called out clearly in docs**: `prediction_id` in
    `/recommend_bid`'s response is no longer a "confirmed persisted"
    signal — it's a receipt for "log under this id." If the background
    insert itself fails (logging DB down), it's caught and logged as a
    warning same as always, but the caller has no way to know except that a
    later lookup by that id would come up empty. The alternative (waiting
    for the DB write before responding) is exactly the delay this avoids.
  - The error path (an exception before a bid was even decided) is
    unchanged — still a plain synchronous log call, since there's no bid
    response to protect from delay in that case.
  - `/explain_bid`'s logging is completely unchanged (still synchronous) —
    it already runs SHAP (and sometimes Ollama) inline, so it was never the
    fast path this mattered for.
- **Tests**: renamed all `recommended_bid_expected_profit` references across
  `test_predict.py`, `test_explain.py`, `test_prediction_log_schema.py`.
  Added `test_recommend_bid_logs_predicted_profit_and_cm` (verifies the
  renamed field, the new field's math, and that the old field name is gone
  from the response), `test_recommend_bid_prediction_id_generated_before_background_write`
  (confirms `prediction_id` is a well-formed `uuid4` returned by the route,
  and that the background-written row uses that exact id), and updated the
  cold-start test to assert both new/renamed fields are `null`. Full suite:
  225 passed (up from 223), isort/black/flake8 clean.
- Docs: `docs/PREDICTION_LOG_SCHEMA.md` §4/§6/§7/§8/§10/§12 and `README.md`'s
  "Prediction logging" section rewritten to describe the rename, the new
  column, and the fully-async `/recommend_bid` write path (and its
  `prediction_id` trade-off).

### nginx: relaxed timeouts for non-bid routes, kept `/recommend_bid` tight
- With the 504 on `/docs` still not fully resolved, bumped nginx's proxy
  timeouts to give more headroom -- but scoped, not blanket: the 3s/5s/5s
  budget was a deliberate fail-fast choice specifically for `/recommend_bid`
  (the real-time bid path with the 2s TAT requirement, see docs/CHANGELOG.md
  2026-07-22); loosening it there would let a slow/stuck request tie up a
  bidding slot for a full minute instead of failing fast.
- Split `nginx.conf`: `proxy_connect/send/read_timeout` moved up to
  `server`-level as the default (now `10s`/`60s`/`60s`) -- covers `/docs`,
  `/openapi.json`, `/explain_bid` (already documented as offline/on-demand,
  not latency-sensitive), and `/health`. A new `location = /recommend_bid`
  exact-match block explicitly pins that one route back down to `3s`/`5s`/
  `5s` -- an exact-match `location` always wins over the `location /` prefix
  match regardless of file order, so this can't be accidentally loosened by
  a future edit to the default. `proxy_set_header` lines moved to
  `server`-level too (unchanged values), since a location that sets its own
  `proxy_pass` target with an appended path still needs them and nginx's
  header-inheritance rules are all-or-nothing per location.
- Still investigating the root cause of the 504 itself (undiagnosed as of
  this entry) -- this change only widens the failure window, it doesn't fix
  whatever is making `serve` slow to respond in the first place.

### Fix: all 4 `SERVE_WORKERS` were pulling the Ollama model at once (504s)
- After fixing the nginx stale-IP 502 (below), the user hit a new `504
  Gateway Time-out` on `/docs`. `docker logs smarthub-serve` showed the
  "Ollama model ... not found ... pulling now..." log line **four times**
  at the same startup timestamp -- one per uvicorn worker process.
- Root cause: `SERVE_WORKERS=4` (`docker/Dockerfile.serve`'s `uvicorn
  --workers 4`) forks 4 worker *processes* inside one container, and
  `predict._lifespan` runs independently in each one -- so all 4 called
  `explain.ensure_model_pulled_async()` at their own startup, and all 4
  fired a simultaneous `POST /api/pull` for the identical model. That's 4x
  the bandwidth/disk I/O for the exact same bytes, and enough CPU/network
  contention in the shared Docker Desktop VM to make `/docs` and `/health`
  themselves time out under nginx's 5s `proxy_read_timeout` -- the pull
  itself was never blocking any individual request; the contention it
  created was the real problem, exactly the risk flagged (and apparently
  not fully closed) in the original design.
- Fix: `explain.py` gained `_ensure_model_pulled_locked()` -- wraps
  `_ensure_model_pulled_sync()` in a non-blocking `flock` on
  `/tmp/smarthub_ollama_pull.lock`. Since uvicorn's worker *processes* share
  one container filesystem (they're forked, not separate containers), a
  plain advisory file lock is enough: whichever worker's background thread
  gets there first does the check/pull; the rest see the lock held and
  return immediately -- no waiting, no retry needed, since the winning
  worker pulling the model benefits every worker's future `/explain_bid` (or
  `/recommend_bid` background SHAP) call regardless of which one did it.
  `ensure_model_pulled_async()` now spawns this locked wrapper instead of
  calling `_ensure_model_pulled_sync()` directly.
- **Tests**: `tests/test_explain.py::test_ensure_model_pulled_locked_dedupes_concurrent_workers`
  -- two threads (standing in for two worker processes) race for the lock;
  asserts the underlying pull logic runs exactly once, and the second
  "worker" returns immediately rather than blocking. Full suite: 223 passed
  (up from 222), isort/black/flake8 clean.

### Fix: nginx kept proxying to a stale `serve` IP after every rebuild (502s)
- Live-debugged after the user reported `http://localhost:8000/docs`
  returning `502 Bad Gateway`. `docker compose ps` showed `smarthub-nginx`
  up for 3 hours (never restarted) while `smarthub-serve` had just been
  recreated 9 minutes earlier by a rebuild; `docker logs smarthub-nginx`
  confirmed it: `connect() failed (111: Connection refused) ... upstream:
  "http://172.20.0.2:8000/..."` -- nginx was still trying the container's
  *old* Docker-assigned IP.
- Root cause: `docker/nginx/nginx.conf`'s `upstream { server serve:8000; }`
  block resolves the `serve` hostname to an IP exactly once, at nginx's own
  process startup -- never again. Every `docker compose up --build serve`
  (a routine, frequent part of this dev loop) gives the recreated container
  a new internal IP; nginx has no way to notice until it's itself restarted.
- Fix: replaced the static `upstream` block with `resolver 127.0.0.11
  valid=10s;` (127.0.0.11 is Docker's embedded DNS server) plus a
  variable-based `proxy_pass $smarthub_serve` (`set $smarthub_serve
  http://serve:8000;`) -- forces nginx to re-resolve the hostname on that
  10s TTL instead of caching it for the container's entire lifetime, so it
  self-heals within ~10s of `serve` being recreated. No more manual `docker
  restart smarthub-nginx` needed after every `serve` rebuild.
- **Immediate unblock** (before the config fix takes effect): `docker
  restart smarthub-nginx` forces a fresh DNS resolution right away.
- Not caught by the automated test suite -- this is Docker/nginx runtime
  behavior, no Python involved. Verification is `docker exec smarthub-nginx
  nginx -t` (config syntax) plus a live `curl -I http://localhost:8000/docs`
  after restarting nginx.

### Ollama runs as its own Docker service, with a non-blocking auto-pull
- Added an `ollama` service to `docker-compose.prefect.yml` (`ollama/ollama`
  image, no host ports -- internal-only like `serve`, persistent
  `ollama-data` volume so pulled models survive container recreation).
- `explain.py` gained `is_model_pulled()` (`GET /api/tags`), `pull_model()`
  (`POST /api/pull`), `_ensure_model_pulled_sync()` (waits briefly for
  Ollama to become reachable, then checks-then-pulls), and
  `ensure_model_pulled_async()` (spawns `_ensure_model_pulled_sync` in a
  daemon thread and returns immediately).
- Wired `ensure_model_pulled_async()` into `predict.py`'s `_lifespan`
  startup hook: `serve` now checks whether `[explain] llm_model` is already
  pulled into the `ollama` service every time it starts, and pulls it if
  missing -- entirely off the main thread, so a multi-minute first-time pull
  never blocks `serve`'s own startup or any in-flight `/recommend_bid` /
  `/explain_bid` request (the exact concern raised: "while pulling do not
  hold other operation").
- Added `$SMARTHUB_OLLAMA_HOST` as an env-var override for
  `explain.OLLAMA_HOST` (same env-wins-over-file convention as
  `SMARTHUB_PREDICTION_LOG_DB_URL`/`SMARTHUB_CONFIG_DB_URL`) -- Docker sets
  it to `http://ollama:11434`; `config/smarthub.yaml`'s `ollama_host` stays
  `localhost` as the default for non-Docker/local-dev use, unchanged.
- `docker/Dockerfile.serve`'s `.[ml,explain]` install (today's earlier fix)
  is what makes `explain.py`'s `import requests`/`import shap` actually
  resolve in this container -- without it, this whole feature would fail
  the same silent way `shap` did.
- **Tests**: `tests/test_explain.py` -- `is_model_pulled`/`pull_model`
  against mocked Ollama responses (present/absent/unreachable, success/
  failure), `_ensure_model_pulled_sync`'s three branches (already pulled,
  missing, host never reachable within budget), and a threading test
  proving `ensure_model_pulled_async` returns in well under a second even
  when the underlying sync check would otherwise block for 5s. Full suite:
  222 passed (up from 213), isort/black/flake8 clean.
- Docs: `README.md`'s `/explain_bid` section and `config/smarthub.yaml`'s
  `[explain]` comments updated to describe the `ollama` service, the
  `SMARTHUB_OLLAMA_HOST` override, and the non-blocking pull.

### Fix: `serve` image was missing the `shap` dependency entirely
- Found while verifying the new `/recommend_bid` background SHAP task
  against the user's live Docker deployment: `docs/PREDICTION_LOG_SCHEMA.md`'s
  worked example row showed `decision_path: "model"` (a real, LightGBM
  model-served prediction) but `shap_explanation` stayed `null` even minutes
  after the response returned.
- Root cause: `pyproject.toml` puts `shap` in a separate `explain` extra,
  not `ml`. `docker/Dockerfile.serve` only ever ran
  `pip install -e ".[ml]"` -- so `shap` was never installed in the `serve`
  image at all. `explain.py`'s `import shap` then raises
  `ModuleNotFoundError` on any real SHAP attempt.
- This wasn't a regression from today's background-task work -- `/explain_bid`
  (already in production, already calling the exact same `explain.py` code)
  has almost certainly had this same silent gap the whole time. Both routes'
  failure-isolation design (catch-and-log-a-warning, never raise into the
  caller) is exactly why nobody saw an error: `/explain_bid` still returns
  200 with a `top_factors: []`/generic response, and `/recommend_bid`'s new
  background task just leaves `shap_explanation: null`, both silently.
- Fix: `docker/Dockerfile.serve` now installs `pip install -e ".[ml,explain]"`.
  No code changes needed -- this was purely a missing dependency in the
  image build.
- **Action needed**: rebuild the `serve` image
  (`docker compose -f docker-compose.prefect.yml -f docker-compose.local.yml
  up -d --build serve`) for this fix to take effect; re-test `/recommend_bid`
  against a real LightGBM model and confirm `shap_explanation` populates.
- Docs: README's `/explain_bid` section now calls out that `Dockerfile.serve`
  bakes in both `ml` and `explain` extras, and why a plain `.[ml]` build
  silently breaks both SHAP paths instead of erroring loudly.

### `/recommend_bid` now backfills `shap_explanation` (background task)
- Every `/explain_bid` call already computed SHAP factors, but
  `/recommend_bid` rows always logged `shap_explanation: null` -- SHAP is
  deliberately kept off `/recommend_bid`'s response path (that's the whole
  reason `/explain_bid` exists separately), so there was no cheap way to
  audit/calibrate a live bid's own SHAP factors after the fact.
- Added `PredictionLogStore.update_shap_explanation(prediction_id, dict)` to
  `prediction_log_schema.py` -- a plain `UPDATE ... WHERE prediction_id = ...`
  against the existing single-table schema (no new columns/tables needed).
- `/recommend_bid` now schedules a `fastapi.BackgroundTasks` job
  (`_log_shap_background` in `predict.py`) right before returning: computes
  `top_factors`/`base_win_rate` via `explain.explain_row` and writes them to
  the row it already logged. Runs *after* the response is sent (Starlette
  semantics), so it adds zero latency to the bid decision itself. Skipped
  entirely on cold start (`model is None`) or a non-viable bid (`NaN`
  `recommended_bid`) -- nothing to explain either way.
- Deliberately excludes the LLM narrative (`explanation`) and `bid_curve`
  that `/explain_bid`'s `shap_explanation` also carries -- those require an
  Ollama call, and running one per bid (even off the response path) risks
  piling up load on Ollama under concurrency, the same scaling concern from
  the earlier TAT work this week. Confirmed with the user before
  implementing (background-SHAP vs. inline vs. leave-as-is), and again on
  scope (SHAP-only vs. full bundle) -- both resolved in favor of the
  lowest-latency, lowest-load option.
- Any failure in the background task (non-LightGBM model, logging DB
  unreachable for the update) is caught and logged as a warning only --
  never surfaces to the caller, never touches the 200 response already sent.
- **Tests**: `tests/test_predict_logging.py` -- a real (tiny) LightGBM
  pipeline's prediction gets `shap_explanation` populated with
  `top_factors`/`base_win_rate` (and confirmed to exclude `bid_curve`/
  `explanation`) by the time `TestClient.post(...)` returns; cold start
  schedules no task at all; a non-LightGBM model (`_ConstantWinRateModel`)
  leaves `shap_explanation` null without breaking the 200 response. Full
  suite: 213 passed (up from 210), isort/black/flake8 clean.
- Docs: `docs/PREDICTION_LOG_SCHEMA.md` §2/§4/§7/§12 and `README.md`'s
  "Prediction logging" section updated to describe the two different
  `shap_explanation` shapes (full for `/explain_bid`, SHAP-only for
  `/recommend_bid`) and the background-write timing.
- **Debugging note (same day, live deployment)**: while verifying this in
  the user's local Docker stack, traced a "why is `lead_ping_id`/
  `prediction_id` still null" report through several layers -- a stray local
  Python process holding `localhost:8000` outside Docker (killed), then a
  `psycopg2.OperationalError: could not translate host name "postgres"`
  once requests were actually reaching `serve`, traced to the `postgres`
  container simply not being up (`docker compose ... ps` showed only
  `nginx`+`serve` running). Brought `postgres` up alongside the rest of the
  stack; prediction logging then wrote/read rows correctly, confirmed via a
  live `psql` query against `smarthub_prediction_log`.

### Fix: prediction log wasn't reliably joinable back to a lead or a caller
- Closed both gaps flagged in `docs/PREDICTION_LOG_SCHEMA.md` §8 right after
  writing it: `BidRequest` had no `lead_ping_id` field (so nothing could be
  sent even if a caller wanted to), and neither `/recommend_bid` nor
  `/explain_bid` returned `prediction_id` in the response (so a caller had no
  receipt to reference later either). Together, that meant a logged
  prediction couldn't be reliably tied back to "which lead/request was
  this" - the log's whole reason for existing.
- Added optional `lead_ping_id: int | None` to `BidRequest` (never used in
  the bid decision itself, purely a correlation key), threaded through to
  `log_prediction(...)` on both the success and error paths in both routes.
- `_log_prediction_safe` now returns the store's generated `prediction_id`
  (`None` only if the write itself failed), and both routes attach it to
  their response body as `result["prediction_id"]` -- a caller either
  supplies `lead_ping_id` up front or gets `prediction_id` back to correlate
  after the fact; the log is joinable either way now.
- **Tests**: `tests/test_predict_logging.py` -- response `prediction_id`
  matches the actual log row for both endpoints, `lead_ping_id` threads
  through correctly, and it's confirmed optional (omitting it is not a
  validation error). Full suite: 210 passed (up from 207), isort/black/
  flake8 clean.
- **Follow-up same day**: `lead_ping_id` was being logged and echoed via
  `prediction_id`, but not echoed back itself -- a caller who sent one had
  no way to see it round-trip in the response. Both routes now also set
  `result["lead_ping_id"] = request.lead_ping_id` (`None` when the caller
  didn't supply one). Tests extended to assert the echo; still 210 passed,
  isort/black/flake8 clean.

### Prediction logging implemented: single table, per the DS weekly decision
- Per the 2026-07-22 DS weekly (Vinaya + Nimesh): don't log the optimizer's
  full candidate-bid sweep or split SHAP into its own table -- one row per
  `/recommend_bid`/`/explain_bid` call, success or failure, everything else
  folded into JSON columns. Rewrote `docs/PREDICTION_LOG_SCHEMA.md` from the
  original 3-table design (parent + `smarthub_optimizer_bid_evaluations` +
  `smarthub_shap_explanations`, never wired into a live endpoint) to a
  single `smarthub_prediction_log` table -- see the doc's §11 for the full
  before/after mapping.
- New `src/smarthub/train_and_predict/prediction_log_schema.py` --
  `PredictionLogStore` (plain SQLAlchemy Core, same conventions as
  `smarthub.core.config_store`: portable Text/Numeric columns, JSON stored
  as serialized text, `$SMARTHUB_PREDICTION_LOG_DB_URL` env override
  defaulting to the shared Postgres). `candidate_bid_generation` and
  `shap_explanation` are JSON columns replacing the old child tables;
  `status`/`error_message` make failed requests first-class; new
  `serving_config` snapshots the exploration/recency/cold-start policy
  values in effect for that specific prediction (Vinaya's reproducibility
  ask, same meeting).
- Wired into `smarthub/server/predict.py`: both routes log exactly once
  after computing their response (success) or in an `except` block before
  re-raising (failure) -- logging never delays or replaces the real
  response, and a logging-DB outage only ever logs a warning, never breaks
  live bidding (same principle as SHAP not delaying `/recommend_bid`). The
  store itself is constructed lazily on first use, not at import/startup,
  so a missing logging DB can't block the API from starting.
- **Tests**: `tests/test_prediction_log_schema.py` (store unit tests --
  round-tripping JSON columns, success/error rows, validation, `recent()`
  filtering) and `tests/test_predict_logging.py` (route-level integration
  via `TestClient`, using the existing `_ConstantWinRateModel` fake so no
  sklearn/lightgbm is required -- success, cold-start, and two distinct
  real-error paths all asserted to produce a correctly-shaped log row).
  Full suite: 207 passed (up from 192), isort/black/flake8 clean.
- **Found, not fixed (out of scope of this change)**: a synthetic LightGBM
  smoke test surfaced a pre-existing bug -- `/explain_bid` can 500 with
  `ValueError: Out of range float values are not JSON compliant` when a
  `top_factors[].value` is `NaN` (an optional numeric feature the caller
  didn't supply). Root cause is in `explain.py`'s `_to_native()` /
  `explain_row`, not in this session's logging work -- confirmed the
  prediction-log row itself still gets written correctly (`status:
  "success"`) before FastAPI's own response serialization fails downstream.
  Worth a follow-up ticket.


## 2026-07-22

### Fix: `/recommend_bid`'s 2s TAT traced to redundant, uncached registry I/O
- Investigated a live report that `/recommend_bid` (the real-time bid path)
  had a ~2s turnaround per request — far higher than a vectorized
  `predict_proba` call over a small candidate-bid grid should cost. Traced it
  to `load_model_and_manifest`'s "currently serving" tier: it called three
  separate `registry` helpers (`currently_serving_version`,
  `currently_serving_manifest`, `currently_serving_model_path`) that each
  independently re-read `current.json` from disk to re-derive the *same*
  version string, plus an uncached `load_manifest` read — every request,
  never cached, unlike the model binary itself (which already had a proper
  in-memory cache). On a bind-mounted Docker volume (`./data:/app/data`),
  repeated small-file reads across the host/container boundary are a
  plausible source of exactly this kind of latency.
- **Fix**: resolve the version exactly once per call (down from 3x), and
  added `_MANIFEST_CACHE` keyed by `(lead_type_name, version)` — manifests
  are immutable once `save_version` writes them, so this cache needs no
  invalidation logic; a version's manifest can never go stale. Still reads
  `current.json` once per request (the actual promotion-detection check —
  same role as the model cache's mtime check), but the manifest itself is
  now read from disk only once total, ever, per version, instead of on
  every single request. Wired into `clear_model_cache()` so tests/manual
  invalidation clear both caches together.
- **Verified**: instrumented a live call count across 5 requests —
  `currently_serving_version` went from 3 calls/request to 1, and
  `load_manifest` went from 1 uncached call/request to 1 call total across
  all 5. Full suite still 192 passed, isort/black/flake8 clean.
- This was root-caused deliberately *before* reaching for more
  infrastructure (more `uvicorn` workers, more `serve` replicas) — those
  would have scaled around the bug (more processes each still paying the
  same artificial 2s tax) rather than fixing why a single request cost 2s
  in the first place.

### Model serving: dedicated `serve` container behind `nginx`, plus in-memory model caching
- New `docker/Dockerfile.serve` — runs the existing FastAPI bid-recommendation
  app (`train_and_predict/predict.py`) via uvicorn. Previously this had no
  Docker image at all and was only documented as a manual
  `uvicorn ... --port 8000` command.
- `docker-compose.prefect.yml`: new `serve` service (image
  `${IMAGE_REPO}:serve-${IMAGE_TAG}`, Watchtower-labelled like
  worker/dashboard) with **no host `ports:` mapping** — reachable only on the
  internal Docker network. New `nginx` service (`nginx:alpine`,
  `docker/nginx/nginx.conf`, plain HTTP reverse proxy to `serve:8000`) is the
  only container exposed to the host — the API can't be hit directly.
  `docker-compose.local.yml` gained the matching `build:` override for local
  source builds.
- `.github/workflows/ci_cd.yml`: new `build-serve` job, identical pattern to
  `build-worker`/`build-dashboard` (same output-capture/`continue-on-error`
  approach), pushing `<account>/smarthub:serve-latest` /
  `:serve-<sha>`. Added to the `notify` job's `needs`; `notify_ci.py`'s
  success message now lists all three images' pull commands (previously two).
- `predict.py`: `load_model` now caches the loaded model in memory, keyed by
  the resolved model URI (registry-resolved URIs are immutable versioned
  filenames, so a newly promoted model naturally busts the cache — no manual
  invalidation needed) plus, for local `.pkl` paths, the file's mtime as a
  safety net against a pinned `MODEL_URI` being overwritten in place. Added
  because `/recommend_bid` sits in the real-time bid path, where
  re-unpickling from disk on every request is latency worth avoiding. New
  `predict.clear_model_cache()` for manual invalidation/tests.
- TLS is not set up yet — `nginx` proxies plain HTTP; noted as a follow-up in
  both the nginx config and the README once this needs to be reachable
  outside a trusted network.
- **Gap closed later the same day** (see "Cold-start/exploration bidding
  policy implemented" below): at this point in the day, `predict.decide_bid`
  didn't exist yet and `/explain_bid` wasn't a registered route — that's
  fixed further down this same changelog date.

### Serving API moved to its own package (`smarthub.server`); eager model loading + `/health` cache status
- `predict.py` moved from `train_and_predict/` to a new top-level
  `smarthub/server/` package (`smarthub/server/predict.py` +
  `smarthub/server/__init__.py`) — the serving surface's dependency footprint
  and deployment lifecycle are independent of the training/orchestration
  pipeline, and it now lives at its own import path
  (`smarthub.server.predict:app`) rather than under `train_and_predict`.
  Updated every reference: `explain.py`'s import, `docker/Dockerfile.serve`'s
  `CMD`, `docker-compose.prefect.yml`'s comment, the top-level README (project
  layout tree, uvicorn command, model-caching note), `train_and_predict/README.md`,
  `docs/MANUAL.md`, `config/smarthub.yaml`'s comment, the `ml` extra's comment
  in `pyproject.toml`, and the imports in `tests/test_train_and_predict.py` /
  `tests/test_explain.py`. No behavior changed by the move itself.
- `predict.py` now **eager-loads** every configured lead type's model
  (`fe.LEAD_TYPES`) into the in-memory cache on startup, via a FastAPI
  `lifespan` context manager (`_eager_load_models`) — so the first real
  `/recommend_bid` after a deploy/restart doesn't pay the disk-unpickle cost.
  A lead type with no promoted model yet (cold start) or any other load
  failure just logs a warning and is picked up lazily on its first request,
  same as before this existed — startup itself never fails because of it.
- New `predict.is_model_cached(lead_type_id)` — resolves the model URI and
  checks the cache dict only (no load). `GET /health` now returns
  `model_loaded: true/false` using it, alongside the existing `model_uri`.

### Cold-start/exploration bidding policy implemented; `/explain_bid` wired up as a real route
- This closes the gap noted earlier in this same changelog date: the
  README/`explain.py` referenced `predict.decide_bid`,
  `predict.load_model_and_manifest`, and `predict.bid_curve_around` as if
  they existed — they didn't. All three are now implemented in
  `smarthub/server/predict.py`, matching the policy already documented in
  the README's "Cold-start + exploration bidding policy" section and
  `docs/CONTEXT.md §7`.
- **`load_model_and_manifest(lead_type_id)`** — same tiered resolution as
  `resolve_model_uri` (env override -> pinned version -> currently-serving),
  but returns `(None, None)` on cold start instead of raising, so callers
  can branch on "no model yet" instead of catching an exception.
- **`exploration_slot(created_dayofweek, created_hour, variance_pct=None)`**
  — buckets a lead into one of 168 hour-of-week slots and deterministically
  flags 1-in-`N` of them (`N = round(1 / exploration_variance_pct)`) as a
  scheduled explore slot, alternating perturbation direction each time a
  bucket fires. Same inputs always give the same explore/exploit decision —
  reproducible and auditable, not a per-request coin flip.
- **`decide_bid(row, model, manifest, expected_revenue, target_cm, min_bid,
  bid_step, created_dayofweek=None, created_hour=None)`** — the one bidding
  decision, returning `decision_path` (`"cold_start_fallback" |
  "exploration" | "model"`) + a plain-English `decision_reason`:
  `cold_start_fallback` when there's no model yet (bids
  `cold_start_fallback_bid_pct` of the way from floor to the CM-respecting
  ceiling); `exploration` on a scheduled explore slot (perturbs the
  profit-maximizing bid by `exploration_variance_pct` and rescores at the
  new bid); `model` otherwise (the raw `optimizer.optimize_bid_for_row`
  result), flagging `model_data_age_days` when the model is older than
  `recency_window_days`.
- **`/recommend_bid`** now calls `load_model_and_manifest` + `decide_bid`
  instead of calling `optimizer.optimize_bid_for_row` directly, so live
  serving gets the same cold-start/exploration handling as `/explain_bid`.
- **`POST /explain_bid`** is now a real registered FastAPI route (it was a
  plain callable in `explain.py` before, never exposed on `app`). Lazily
  imports `explain` inside the route body (not at module level) to avoid a
  circular import with `explain.py`'s own `from smarthub.server import
  predict`, and to keep `shap`/Ollama out of every `/recommend_bid` request.
- **Tests**: new `tests/test_predict.py` covering `exploration_slot`'s
  determinism/bucketing/direction-alternation, `decide_bid`'s three paths
  (including an exploration case that asserts the win-rate/profit are
  recomputed at the perturbed bid, not reused from the original), and all
  four `load_model_and_manifest` resolution tiers. Full suite: 184 passed,
  6 skipped.

### `serve` scaling: configurable uvicorn worker processes
- `docker/Dockerfile.serve`'s `CMD` switched to shell form so `--workers
  ${SERVE_WORKERS:-4}` actually expands; default 4, overridable per
  deployment via `SERVE_WORKERS` in `docker-compose.prefect.yml`'s `serve`
  environment (reads from `.env`, same pattern as the other env-driven
  settings). Was previously hardcoded to uvicorn's implicit single process.
- Each worker process gets its own copy of the in-memory model cache and
  runs its own startup eager-load — not a problem, since models are small
  and read-only, so there's no cache-coherency concern across workers. All
  workers share the container's one port, so `nginx.conf`'s `upstream
  smarthub_serve { server serve:8000; }` didn't need to change.
- Documented in the README's "Production — serve + nginx" section, including
  the next step if one container's workers aren't enough (scale the
  container itself and list all replicas in the nginx upstream).


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

