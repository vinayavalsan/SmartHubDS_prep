# SmartHub — Manual Run Guide

How to run the SmartHub / Anton pipeline **by hand**, stage by stage, without the
Prefect schedule or the worker. Useful for ad-hoc runs, backfills, and debugging.

The pipeline has three stages and they must run **in order** — each reads the
previous stage's output:

```
1) data-pull  ──►  2) build-features  ──►  3) train-model
   Redshift → storage   training table         model + bid optimizer
```

If you run a stage before its input exists, it fails fast with a clear
"run the previous step first" message.

---

## 1. Prerequisites (one-time)

Install the package with the extras needed for all three stages:

```bash
pip install -e ".[orchestration,ml,validation]"
```

This registers three console scripts:

| Command | Stage |
| --- | --- |
| `smarthub-pull` | 1 · data-pull |
| `smarthub-build-features` | 2 · build-features |
| `smarthub-train` | 3 · train-model |

> Re-run `pip install -e .` after pulling new code so the console scripts stay
> current. Every command below also has an equivalent `python -m ...` form.

### Environment (`.env`)

Credentials, connection details, and storage paths come from `.env` (never
committed). At minimum you need the Redshift/SSH credentials for data-pull and
the storage paths. Slack is optional:

```bash
# .env
SLACK_WEBHOOK_URL=            # optional — blank = notifications disabled (no-op)
SLACK_ENV_LABEL=local         # optional — label shown on every Slack message
```

### Config (`config/smarthub.yaml`)

Non-secret task knobs (model type, training window, bid step, feature selection).
Missing keys fall back to code defaults, so the file is optional. See
[Feature selection](#feature-selection) below.

---

## 2. Stage 1 — data-pull

Pulls `lead_pings` from Redshift into local storage (partitioned Parquet +
DuckDB). A **manual** pull takes an **explicit window** (`--min-created-at` /
`--max-created-at`) and does **not** move the watermark — so it's safe to run
alongside the scheduled pulls.

```bash
# Auto leads for a specific window
smarthub-pull --lead-type-id 6 \
  --min-created-at "2026-07-01 00:00:00" \
  --max-created-at "2026-07-09 00:00:00"

# Home leads
smarthub-pull --lead-type-id 1 \
  --min-created-at "2026-07-01 00:00:00" \
  --max-created-at "2026-07-09 00:00:00"

# All lead types (omit --lead-type-id)
smarthub-pull \
  --min-created-at "2026-07-08 00:00:00" \
  --max-created-at "2026-07-09 00:00:00"
```

Options:

| Flag | Meaning |
| --- | --- |
| `--min-created-at` | **required** — inclusive lower bound, `YYYY-MM-DD HH:MM:SS` |
| `--max-created-at` | **required** — exclusive upper bound, `YYYY-MM-DD HH:MM:SS` |
| `--lead-type-id` | restrict to one type (6=auto, 1=home); default: all |
| `--no-expected-revenue` | skip the `lead_ping_listings` join |
| `--all-listings` | aggregate expected revenue over ALL listings, not just selected |
| `--log-level` | logging level (default: env `LOG_LEVEL` or `INFO`) |

`smarthub-pull --help` lists them all.

Every pull runs **data validation** on the fetched batch (warn + report only —
it flags bad rows and missing-value patterns, never drops/fixes them or blocks
the pull). The CLI logs a summary; the scheduled flow also writes a
`data-quality-<type>` Prefect artifact and a "Data quality" section in the Slack
message. Tune it in the `validation` section of `config/smarthub.yaml`.

---

## 3. Stage 2 — build-features

Rebuilds the leakage-safe **training table** for one lead type from the
accumulated leads. Runs the Prefect-free core in-process (no worker/server).

```bash
smarthub-build-features --lead-type-id 6            # auto (default)
smarthub-build-features --lead-type-id 1            # home
smarthub-build-features --lead-type-id 6 --window-days 0    # use ALL stored data
```

Equivalent module form (Prefect-free core):

```bash
python -m smarthub.feature_engineering.build --lead-type-id 6
```

> Manual build-features runs the **Prefect-free** core in
> `feature_engineering/build.py` — no worker/server, no Prefect run — same as
> data-pull and train. The Prefect deployment (`feature_engineering/flow.py`)
> wraps the same core for automation and adds the run artifact + Slack
> notification.

Options:

| Flag | Meaning |
| --- | --- |
| `--lead-type-id` | 6=auto (default), 1=home |
| `--lead-type-name` | override the name (default: derived from the id) |
| `--window-days` | rolling training window in days; `0` = all data; default from the YAML (`feature_engineering.training_window_days`, 21) |

---

## 4. Stage 3 — train-model

Trains + evaluates one lead type's win-probability model, runs the offline bid
optimizer, versions the model, and promotes it only if it beats the
currently-serving model on the same held-out test set.

```bash
smarthub-train --lead-type-id 6                     # auto
smarthub-train --lead-type-id 1                     # home
smarthub-train --lead-type-id 6 --no-mlflow         # skip MLflow logging
smarthub-train --lead-type-id 6 --version 2026-07-09T073241Z   # pin a training table
```

Equivalent module form:

```bash
python -m smarthub.train_and_predict.train --lead-type-id 6
```

Options:

| Flag | Meaning |
| --- | --- |
| `--lead-type-id` | 6=auto (default), 1=home |
| `--version` | training-table version to train on (default: latest) |
| `--no-mlflow` | skip MLflow logging/registration |

---

## 5. Full run, end to end

To reproduce a complete scheduled cycle for one lead type, run the three stages
in order:

```bash
# auto, last ~3 weeks
smarthub-pull --lead-type-id 6 \
  --min-created-at "2026-06-18 00:00:00" \
  --max-created-at "2026-07-09 00:00:00"

smarthub-build-features --lead-type-id 6

smarthub-train --lead-type-id 6
```

Repeat with `--lead-type-id 1` for home.

---

## Feature selection

Which features the **model** trains on is controlled per lead type in
`config/smarthub.yaml` — no code change, no re-pull/re-build needed (every feature
is always built into the training table; this only changes what the model
consumes, so just retrain after editing).

```yaml
features:
  # auto MANDATORY core (locked in code, always trained on, cannot be removed):
  #   home_owner, multi_vehicle, num_vehicles, insured, num_auto_accidents,
  #   dui, sr22_required, age (+ age bands), bid
  auto_optional: >-
    state, gender, marital_status, military_affiliation, campaign_id,
    traffic_tier, num_drivers, num_auto_violations, continuous_coverage_months,
    is_married, created_hour, created_dayofweek, is_workday
  home_optional: all
```

`auto_optional` accepts:

- `all` — every optional feature (default if the key is absent)
- `none` — mandatory core only
- a comma list — exactly those optional features (unknown names are ignored with
  a warning; mandatory features can never be dropped)

After editing, just retrain:

```bash
smarthub-train --lead-type-id 6
```

The train-model Slack notification reports which optional features were included
vs excluded, so you can confirm the change took effect.

---

## Serving the model (bid API)

The FastAPI service serves whichever model version is currently promoted for the
request's `lead_type_id`:

```bash
uvicorn smarthub.train_and_predict.predict:app --reload --port 8000
```

- `GET  /health?lead_type_id=6` — reports which model artifact would be served.
- `POST /recommend_bid` — returns the profit-maximising bid for one lead.

Pin a specific model instead of the promoted one via `MODEL_URI` (a `.pkl` path
or MLflow URI), or `config/smarthub.yaml prediction.active_model_version`.

Send a sample request with the bundled client (start the API first):

```bash
python -m smarthub.train_and_predict.manual_api_check
```

---

## Model registry / rollback

Every training run is saved as an immutable, numbered version; a `current.json`
pointer marks the serving version. From Python:

```python
from smarthub.train_and_predict import registry

registry.list_versions("auto")                 # ["v1_...", "v2_...", "v3_..."]
registry.currently_serving_version("auto")     # serving version, or None
registry.rollback("auto")                      # repoint at the prior version
registry.rollback("auto", to_version="v1_2026-07-01T050000Z")  # a specific one
```

---

## Manual vs scheduled — quick reference

| | Scheduled (Prefect) | Manual (CLI) |
| --- | --- | --- |
| data-pull | window from watermark; advances it | explicit `--min/--max`; watermark untouched |
| build-features | runs on the `features` queue | runs the Prefect-free core in-process |
| train-model | runs on the `training` queue | runs training in-process |
| Slack / artifacts | yes | none — all three log to console (data-pull still alerts on failure) |
| Prefect needed | worker + server | no — all three manual entry points are Prefect-free |

---

## Troubleshooting

- **"run data-pull first" / no data** — build-features found nothing in storage.
  Run Stage 1 for that lead type, then retry.
- **"needs both wins and losses" / single class** — the training window has only
  wins (or only losses). Widen the window (`--window-days`) or pull more data.
- **Model not promoted ("Held")** — the challenger didn't beat the
  currently-serving model on ROC AUC / profit. Expected during early data churn;
  the previous model keeps serving. Force-serve a version with `MODEL_URI` or the
  `active_model_version` pin if needed.
- **Console script not found** — re-run `pip install -e ".[orchestration,ml,validation]"`.
