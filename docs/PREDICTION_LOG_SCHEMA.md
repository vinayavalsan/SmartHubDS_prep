# Anton Production Prediction Log - Schema Design

Status: implemented. Table + store:
`src/smarthub/train_and_predict/prediction_log_schema.py`. Tests:
`tests/test_prediction_log_schema.py`.

**Revision note (2026-07-22 DS weekly, Vinaya + Nimesh):** the original
version of this design (v1) used three tables - a parent plus a child table
for every candidate bid the optimizer swept, plus a child table for the SHAP
payload. Vinaya's call in that meeting: don't log the full candidate sweep
(hundreds of rows per prediction once bids grow - not useful enough to
justify the volume), and don't split SHAP into its own table either - fold
both into JSON columns on a single row. "Let's just stick to a single line
for every API request... whether it succeeded or failed." This document
describes v2, the single-table design that resulted. See §11 for exactly
what changed from v1 and why.

## 1. Purpose

Anton supports two prediction endpoints (`/recommend_bid` and
`/explain_bid`). Every call to either one is logged - success or failure -
so it can be reconstructed, monitored, audited, and explained later, without
depending on the model version that served it still existing or the
feature-engineering code being unchanged since.

- **Monitoring** can track win-rate calibration, bid distribution, and
  decision-path mix (`model` vs. `cold_start_fallback` vs. `exploration`)
  over time without re-running anything.
- **Debugging** a specific bid ("why did we bid $4.20 for lead 82931
  yesterday at 3am?") never requires the model or feature-engineering code
  to be unchanged since.
- **Auditing** can show exactly what data and what model produced a
  decision, on demand, indefinitely - including failed requests, which log
  just as reliably as successful ones.
- **Explainability** already has the structured SHAP facts persisted (for
  `/explain_bid` calls), so a future dashboard or explanation UI reads them
  straight out of the log - no recomputation.

## 2. Design principles

**One row per API call, success or failure.** Vinaya's framing: the
question this log answers is "what did Anton do for this request," and that
question has one answer whether the request succeeded, returned no viable
bid, or errored outright. `status` + `error_message` (§4) make failures
first-class instead of simply absent from the log.

**Snapshot, don't reference.** Every field a reconstruction needs is copied
into the log row at prediction time - model version *and* its lineage, the
feature columns actually used, the exact post-feature-engineering vector fed
to the model, and (new in v2) the serving-policy config values in effect for
this specific prediction (§4, `serving_config`). The model registry, the
feature-engineering code, and `config/smarthub.yaml` are all free to change
later without invalidating a single historical row.

**Two feature snapshots, not one.** `input_features` is the raw request as
the caller sent it (audit trail: "what did the caller ask for"). Separately,
`model_input_features` + `feature_cols` is the exact vector the
preprocessing step produced and the model actually scored (audit trail:
"what did the model see"). These can legitimately differ - optional features
get toggled on/off per lead type, and a placeholder bid gets replaced by the
actual chosen bid before SHAP runs - so both are logged rather than assuming
one can be derived from the other.

**Candidate-bid generation is a JSON summary, not a table of every
candidate.** Per the meeting: store enough to describe the sweep (method,
bounds, how many candidates) so it can be regenerated on demand if there's
ever a reason to, without paying to store every individual candidate on
every single request. See `candidate_bid_generation` in §4.

**SHAP is one JSON column, not a table.** Full on `/explain_bid` calls
(everything that endpoint computes, including the LLM narrative) - see
`shap_explanation` in §4. `/recommend_bid` also populates a lighter version
of the same column (`top_factors`/`base_win_rate` only, no LLM text), but
never on its response path: it's computed in a background task *after* the
response is sent (2026-07-23 follow-up), so it never adds latency to a live
bid. The row exists (with `shap_explanation = null`) the instant the response
returns, and gets filled in a moment later; a reader that queries
immediately after `/recommend_bid` responds may briefly see `null` there.

**Numbers carry their units.** SHAP values from a tree model are computed in
log-odds space, not probability space - `base_win_rate` inside
`shap_explanation` is the model's average predicted win rate (already
converted to a 0-1 probability by `explain.py` before it ever reaches this
log), documented as such rather than left ambiguous.

## 3. Table overview

```
smarthub_prediction_log   -- one row per call to /recommend_bid or /explain_bid,
                              success or failure. That's the whole schema.
```

No child tables. Every candidate-bid detail and every SHAP detail lives on
this one row, as JSON.

**Relationships (all 1:1 with the prediction row, embedded as JSON — no joins):**

- **Prediction record** — the row itself: metadata, model info, input snapshot,
  and the final decision (`recommended_bid` + predicted metrics + `decision_path`).
- **Optimizer evaluations** — `candidate_evaluations`: the full per-candidate
  sweep (`{bid, predicted_win_rate, expected_profit, selected}[]`), with
  `candidate_bid_generation` describing how that candidate set was produced. One
  prediction has many candidates, all held inside this single array on the row.
- **SHAP explanation** — `shap_explanation`: the structured per-feature
  contribution payload for the selected prediction, written inline (`/explain_bid`)
  or backfilled by the `shap-worker` (`/recommend_bid`) onto the same row.

Because optimizer evaluations and the SHAP payload have no lifecycle
independent of the prediction (always exactly one per row, never queried on
their own), they are columns on the parent row rather than separate related
tables — the relationship is composition, not a foreign-key join.

**Primary key:** `prediction_id` (a UUID4 generated by the API, returned to
the caller so a later `/explain_bid` call or a support ticket can reference
the same id).

**Partitioning:** not applied natively today - see §10. `log_date` is kept
as a plain indexed column (not part of the key) so time-range queries stay
cheap without requiring Postgres-native partitioning, which also means this
table creates identically on SQLite (tests) and Postgres (production) with
no dialect-specific DDL.

**Foreign keys:** none - `lead_ping_id` is a *logical* reference to
`public.lead_pings.id`, not a DB-enforced FK, since that table may live in a
different database/schema. See §8 for how it gets populated.

## 4. `smarthub_prediction_log`

| Column | Type | Notes |
| --- | --- | --- |
| `prediction_id` | UUID / text | Primary key. App-generated (`uuid4()`). |
| `served_at` | timestamptz | When this prediction was served. |
| `log_date` | date | `served_at::date` - plain indexed column for time-range queries (§3). |
| `schema_version` | smallint | `2` for this single-table design (v1 was the 3-table version, never wired into a live endpoint - see §11). |
| `package_version` | text, nullable | SmartHub package version (semver, `smarthub.__version__`) that served the row. Pairs with `model_version` so a prediction is traceable to both the code contract (field/feature registries, serving logic) and the model. Auto-filled at write time. |
| `request_id` | text, nullable | Caller-supplied correlation id, if the upstream system has one. |
| `lead_ping_id` | bigint, nullable | Reference to `public.lead_pings.id` - the primary way to map a prediction back to a specific lead. Optional on `BidRequest`; nullable when the caller doesn't supply one (see §8). |
| `endpoint` | text | `'recommend_bid'` \| `'explain_bid'`. |
| `served_by_host` | text, nullable | Pod/instance id - ops debugging for a specific bad request. |
| `status` | text | `'success'` \| `'error'` - **every call is logged, whether it succeeded or not** (Vinaya's explicit ask). |
| `error_message` | text, nullable | Populated only when `status = 'error'` - what actually went wrong. |
| `lead_type_id`, `lead_type_name` | smallint, text | Which lead-type model answered this prediction. |
| `campaign_id`, `account_id`, `source_type_id` | bigint | Business context from the request, straight through. |
| `input_features` | jsonb | The raw request payload (minus the optimizer knobs, which have their own columns below) - exactly what the caller sent. |
| `model_input_features` | jsonb, nullable | The exact row the preprocessing step produced and the model scored, at the recommended bid. Null when scoring never happened (e.g. an error before preprocessing). |
| `feature_cols` | jsonb, nullable | Ordered list of columns the serving model version actually used (from its manifest). |
| `expected_revenue`, `target_cm`, `min_bid`, `bid_step` | numeric | Optimizer inputs, straight from the request. |
| `candidate_bid_generation` | jsonb, nullable | How the candidate set was generated: `{"method", "min_bid", "max_bid", "bid_step", "n_candidates"}` - today's method is an equally-spaced grid (`method: "equally_spaced"`); a future method (e.g. sparse-then-refine sampling) adds its own keys here without a schema change. |
| `candidate_evaluations` | jsonb, nullable | **The full optimizer sweep** - a JSON array with one entry per candidate bid the optimizer scored, each `{bid, predicted_win_rate, expected_profit, selected}`; exactly one entry has `selected: true` (the argmax-profit bid). Retains the complete optimizer evaluation history so the decision can be reconstructed **without re-running the model**. Null on cold start / no-viable-bid (nothing was scored). See the shape and the note on size below. |
| `model_version`, `model_uri` | text, nullable | Which model artifact answered this prediction. Null on true cold start (no model exists yet). |
| `model_type`, `model_calibrated` | text, boolean, nullable | From the manifest's `lineage` (e.g. `lightgbm`/`logistic_regression`; isotonic calibration on/off). |
| `training_table_version` | text, nullable | From the manifest's `lineage` - which training-table snapshot produced this model. |
| `model_data_min_created_at`, `model_data_max_created_at` | text, nullable | From the manifest's `lineage`, copied verbatim (not parsed) - the training data's date range. Text rather than a typed timestamp because the manifest's own values pass through `json.dumps(..., default=str)` at save time and aren't guaranteed strict ISO 8601; a parse failure should never turn a logging call into an error. |
| `model_data_age_days` | integer, nullable | From `decide_bid` - days since the serving model's manifest was written; flags a stale model (informational only). |
| `decision_path` | text, nullable | From `decide_bid`: `'model'` \| `'cold_start_fallback'` \| `'exploration'`. |
| `decision_reason` | text, nullable | From `decide_bid` - plain-English reason for the path taken. |
| `recommended_bid` | numeric, nullable | NULL means "no viable bid" (or an error before a bid was reached). |
| `recommended_bid_predicted_win_rate`, `recommended_bid_predicted_profit` | numeric, nullable | At the recommended bid. `recommended_bid_predicted_profit` renamed from `recommended_bid_expected_profit` (2026-07-23) to match `recommended_bid_predicted_win_rate`'s "predicted_" naming - both are model predictions, not realized/observed values, so "expected" (a different, ambiguous word for the same idea) was dropped. |
| `recommended_bid_predicted_cm` | numeric, nullable | **New 2026-07-23.** `recommended_bid_predicted_profit / expected_revenue` - the predicted contribution margin at the recommended bid. Null under the same conditions as `recommended_bid_predicted_profit` (cold start, no viable bid, or an error before a bid was reached). |
| `shap_explanation` | jsonb, nullable | **Replaces v1's SHAP child table.** For `/explain_bid`: `{"top_factors", "base_win_rate", "bid_curve", "explanation"}` - exactly what `/explain_bid` adds on top of `/recommend_bid`'s response (see `explain.explain_bid`), written synchronously (before the response). For `/recommend_bid`: `{"top_factors", "base_win_rate"}` only (no `bid_curve`/`explanation` - those stay LLM/on-demand-only), written by a background task *after* the response is sent - see the shapes in §4 below and the timing note in §2. Null on cold start (no model to explain), on a non-viable-bid result, or if the model isn't a supported type (`explain_row` only supports `lightgbm` today). |
| `serving_config` | jsonb, nullable | **New in v2** - snapshot of the serving-policy config in effect for this prediction (`exploration_variance_pct`, `recency_window_days`, `cold_start_fallback_bid_pct`), so a prediction stays reproducible even if `config/smarthub.yaml` changes later (Vinaya's reproducibility ask, same meeting). |
| `created_at` | timestamptz | Row-insert time. |

### `candidate_bid_generation` shape (today)

```json
{"method": "equally_spaced", "min_bid": 0.25, "max_bid": 37.50, "bid_step": 0.25, "n_candidates": 150}
```

### `candidate_evaluations` shape

The complete optimizer sweep for the prediction: one object per candidate bid
evaluated, so the optimizer's decision is fully auditable and reconstructable
without the model. Each entry carries the bid, the model's predicted win
probability at that bid, the resulting expected profit
(`predicted_win_rate * (expected_revenue - bid)`), and whether it was the
selected bid. Exactly one entry has `selected: true` — the argmax-profit bid
(on the `exploration` path the *served* bid is a perturbation of this selected
bid; `recommended_bid` remains the source of truth for what was actually bid).
Values are rounded (win rate 6dp, money 4dp) to keep the payload compact.

```json
[
  {"bid": 0.25, "predicted_win_rate": 0.041000, "expected_profit": 4.0713, "selected": false},
  {"bid": 8.25, "predicted_win_rate": 0.780000, "expected_profit": 71.5350, "selected": true},
  {"bid": 8.50, "predicted_win_rate": 0.774000, "expected_profit": 70.8210, "selected": false}
]
```

**Note on size.** This is the one column whose size scales with `n_candidates`
(~tens to a couple hundred entries per prediction). It's stored on the same row
(no child-table fan-out) and rounded to stay compact. If storage/throughput ever
becomes a concern, the sweep can be capped or made opt-out via
`decide_bid(include_candidates=...)` without any schema change — the column
simply becomes null for those rows.

### `shap_explanation` shape

The canonical payload is a structured SHAP breakdown (no LLM required):
`base_prediction` (the model's average win rate before this lead's factors),
`prediction` (the predicted win rate for this lead), and
`feature_contributions` — **every** model feature's contribution to the
predicted win probability, each `{feature, value, contribution}`, sorted by
`|contribution|` descending. Contributions are in log-odds units (their sign
and relative magnitude are what matter); `base_prediction`/`prediction` are
0-1 probabilities. `top_factors`/`base_win_rate` are a backward-compatible
top-N view kept for existing consumers. Because the full contribution set is
stored, a natural-language explanation can be generated later from this payload
**without re-running the model or recomputing SHAP**.

> **Note on `prediction` vs the contributions.** `prediction` is reconciled to
> the **calibrated** win rate actually served and logged — it equals the
> `recommended_bid_predicted_win_rate` column, so there is a single consistent
> headline number. The SHAP `feature_contributions` (log-odds) and
> `base_prediction` describe the underlying tree model and therefore sum to the
> model's *uncalibrated* output; when calibration (isotonic/sigmoid) is applied
> they will **not** exactly reconstruct `prediction`. Use the contributions for
> *why* (relative feature influence and direction) and `prediction` for the
> served win probability. (On the raw-lead dev path, where no served value
> exists, `prediction` falls back to the SHAP-reconstructed win rate.)

**`/recommend_bid`** (written by a background task after the response, so
prediction latency is unaffected; no LLM, no `bid_curve`/`explanation`):

```json
{
  "base_prediction": 0.114,
  "prediction": 0.78,
  "feature_contributions": [
    {"feature": "num_auto_accidents", "value": 1, "contribution": 0.84},
    {"feature": "bid", "value": 8.25, "contribution": 0.13},
    {"feature": "state", "value": "CA", "contribution": 0.05},
    {"feature": "age", "value": 34.0, "contribution": -0.31}
  ],
  "top_factors": [
    {"feature": "num_auto_accidents", "value": 1, "shap": 0.84, "direction": "increased"},
    {"feature": "age", "value": 34.0, "shap": -0.31, "direction": "decreased"}
  ],
  "base_win_rate": 0.114
}
```

**`/explain_bid`** (same payload as above, written synchronously before the
response, plus the on-demand `bid_curve` and — only if explicitly requested —
an LLM `explanation`; neither is part of the automatic production workflow):

```json
{
  "base_prediction": 0.114,
  "prediction": 0.78,
  "feature_contributions": [
    {"feature": "num_auto_accidents", "value": 1, "contribution": 0.84},
    {"feature": "age", "value": 34.0, "contribution": -0.31}
  ],
  "top_factors": [
    {"feature": "num_auto_accidents", "value": 1, "shap": 0.84, "direction": "increased"},
    {"feature": "age", "value": 34.0, "shap": -0.31, "direction": "decreased"}
  ],
  "base_win_rate": 0.114,
  "bid_curve": [
    {"bid": 12.25, "predicted_win_rate": 0.33, "expected_profit": 12.4},
    {"bid": 12.50, "predicted_win_rate": 0.34, "expected_profit": 12.75}
  ],
  "explanation": "This lead's prior accident and other factors pushed the predicted win rate well above average, supporting a bid of $12.50 for an expected $12.75 profit."
}
```

### `serving_config` shape

```json
{"exploration_variance_pct": 0.10, "recency_window_days": 30, "cold_start_fallback_bid_pct": 0.50}
```

**Indexes:**
- `pk_smarthub_prediction_log` (unique) - `prediction_id`
- `ix_prediction_log_served_at` - `(served_at)` - time-range queries (the role partitioning would otherwise play, §10).
- `ix_prediction_log_lead_type_served` - `(lead_type_id, served_at)`
- `ix_prediction_log_campaign_served` - `(campaign_id, served_at)`
- `ix_prediction_log_lead_ping_id` - `(lead_ping_id)`
- `ix_prediction_log_status` - `(status, served_at)` - "show me recent failures" without a full-table scan.

## 5. Worked example - one prediction, as it sits in the table

One lead ping (`lead_ping_id = 82931`), one `/explain_bid` call:

| Column | Value |
| --- | --- |
| `prediction_id` | `a0460922-6579-4697-b8fd-263bb76a74b7` |
| `served_at` / `log_date` | `2026-07-22 03:00:12+00` / `2026-07-22` |
| `schema_version` | `2` |
| `lead_ping_id` | `82931` |
| `endpoint` | `explain_bid` |
| `status` / `error_message` | `success` / `null` |
| `lead_type_id` / `lead_type_name` | `6` / `auto` |
| `campaign_id` / `account_id` / `source_type_id` | `4021` / `118` / `3` |
| `input_features` | `{"state": "CA", "age": 34, "num_auto_accidents": 1, ...}` |
| `model_input_features` | `{"age": 34.0, "state": "CA", "num_auto_accidents": 1.0, "bid": 12.50, ...}` |
| `feature_cols` | `["age", "state", "num_auto_accidents", "bid", ...]` |
| `expected_revenue` / `target_cm` / `min_bid` / `bid_step` | `50.00` / `0.2500` / `0.25` / `0.25` |
| `candidate_bid_generation` | `{"method": "equally_spaced", "min_bid": 0.25, "max_bid": 37.50, "bid_step": 0.25, "n_candidates": 150}` |
| `model_version` / `model_uri` | `v4_2026-07-09T140501Z` / `/data/models/auto/v4_2026-07-09T140501Z.pkl` |
| `model_type` / `model_calibrated` | `lightgbm` / `true` |
| `training_table_version` | `auto_2026-07-09` |
| `model_data_min_created_at` / `model_data_max_created_at` | `2026-04-11` / `2026-07-08` |
| `model_data_age_days` | `13` |
| `decision_path` / `decision_reason` | `model` / `"Standard profit-maximizing bid from the currently-serving model."` |
| `recommended_bid` | `12.50` |
| `recommended_bid_predicted_win_rate` / `recommended_bid_predicted_profit` | `0.340` / `12.7500` |
| `recommended_bid_predicted_cm` | `0.2550` (`= 12.75 / 50.00`) |
| `shap_explanation` | see §4 shape above |
| `serving_config` | `{"exploration_variance_pct": 0.10, "recency_window_days": 30, "cold_start_fallback_bid_pct": 0.50}` |
| `created_at` | `2026-07-22 03:00:12.884+00` |

A failed request (say, a malformed `expected_revenue` that slipped past
validation, or an unexpected exception inside `decide_bid`) logs the same
row shape, with `status = 'error'`, `error_message` populated, and every
field downstream of the failure point (`model_input_features` onward) left
`null` - it's still one row, still queryable by `lead_type_id`/`campaign_id`/
`served_at` like any other.

## 6. Reading it end to end

Lead ping `82931` came in via `/explain_bid`. Anton's auto model
(`v4_2026-07-09T140501Z`, 13 days old) swept 150 candidate bids from `$0.25`
to `$37.50` (`candidate_bid_generation`), picked `$12.50` via the normal
`model` decision path, and computed a SHAP explanation at that bid
(`shap_explanation`): starting from an `11.4%` population-average win rate,
this lead's own factors (a prior accident, being in CA, etc.) pushed the
predicted win rate up to `34%`.

## 7. Where this is called from

`/explain_bid` calls `PredictionLogStore.log_prediction(...)` synchronously,
**after** the response is computed but still inside the request - reading
the returned `prediction_id` straight into its own response body. This is
fine there: `/explain_bid` already runs SHAP (and, when the LLM is
reachable, an Ollama call) inline, so a bit more synchronous DB I/O doesn't
change its latency character - it was never the fast path to begin with.

`/recommend_bid` is different (2026-07-23): **the entire log-row insert is a
`BackgroundTasks` job**, not just the SHAP enrichment. `prediction_id` is
generated in the route itself (`uuid.uuid4()`, not by the store) and
returned to the caller immediately; `log_prediction(...)` (with that exact
id passed through) is scheduled to run only *after* the response has
already been sent (Starlette semantics - see `_log_prediction_safe` and the
`recommend_bid` route in `predict.py`). Nothing about logging - not the
insert, not any derived metric computed for it (`recommended_bid_predicted_cm`
included) - is ever on `/recommend_bid`'s response path. The trade-off:
`prediction_id` in the response is a receipt for "this is the id we intend
to log under," not a guarantee the row exists yet; if the background insert
itself fails, it's caught and logged as a warning (same failure isolation as
always), and a caller has no way to know except that a later lookup by that
id would come up empty.

`/recommend_bid` additionally schedules `PredictionLogStore
.update_shap_explanation(...)` as a *second* `BackgroundTasks` job, added
right after the logging insert above - Starlette runs `BackgroundTasks` in
registration order, so this always finds its row already there (or, if the
insert itself failed, simply updates zero rows). Same failure isolation
applies: any exception in there (logging DB unreachable, or a model type
SHAP doesn't support) is caught and logged as a warning, leaving
`shap_explanation` `null` rather than surfacing anywhere the caller would
see it.

## 8. Correlating a response back to its log row

**`lead_ping_id` is now threaded through.** `BidRequest` accepts an optional
`lead_ping_id: int | None` (never used in the bid decision itself, purely a
correlation key), passed straight to `log_prediction(...)` on both the
success and error paths. This is what makes the log joinable back to
`public.lead_pings.won` for realized calibration ("was Anton's predicted win
rate actually right?") - the single most important monitoring question this
log exists to answer.

**`prediction_id` is now returned to the caller.** Both `/recommend_bid` and
`/explain_bid` include `prediction_id` in their response body, so a caller
(or a later support ticket) always has a receipt to reference, without
needing `lead_ping_id` at all if the caller doesn't have one yet. The two
routes differ slightly in what that guarantees, per §7's async split:
`/explain_bid` writes synchronously and returns the store's own generated
id, so `prediction_id` is `None` only if that write itself failed.
`/recommend_bid` generates its `prediction_id` up front and always returns
it (never `None`) - the actual row write happens in the background
afterward, so this id is a receipt for "log under this id," not a
guarantee the row exists yet; a lookup by it can (rarely) come up empty if
the background insert itself failed.

Both were gaps in the original version of this document; closed together
since a caller either supplies `lead_ping_id` up front or gets `prediction_id`
back to correlate later - the log is reliably joinable either way now.

## 8a. `/recommend_bid` response contract (lean default + `verbose`)

The **server is the single source of truth for the log** - it persists the
complete record itself (§7), so the response never needs to carry logging data
for the log to be complete. The response has two shapes:

**Default (lean) - the hot path.** Returns only what a bidding client needs:
`recommended_bid`, `recommended_bid_predicted_win_rate`,
`recommended_bid_predicted_profit`, `max_bid`, `n_candidate_bids`,
`decision_path`, `decision_reason`, `model_data_age_days`, `prediction_id`,
and `lead_ping_id`. No feature snapshots, no model identity, no candidate
sweep - keeping the payload small and the TAT profile flat at high QPS.

**`verbose: true` (opt-in) - full decision payload.** Set `verbose` on the
`BidRequest` to also receive the complete logging-schema payload, built from
the *same* record that is persisted (so response and logged row can't diverge):
prediction `served_at`, `tat_seconds`, `package_version`, lead ids/metadata,
`input_features`, `model_input_features`, `feature_cols`, optimizer config
(`expected_revenue`/`target_cm`/`min_bid`/`bid_step` + `candidate_bid_generation`),
model identity (`model_name`/`model_version`/`model_uri`/`model_type`/
`model_calibrated`/`training_table_version` + data range), the final
`recommended_bid_predicted_cm`, and `serving_config`.

Two deliberate constraints on `verbose`:

- **Candidate sweep is capped on the wire** to the **selected bid + the first
  19 by bid** (≤20 entries), so the chosen bid is always present without
  shipping the full ~300-row array. The log always keeps the complete sweep.
- **SHAP is never in the response** - it is computed asynchronously and attached
  to the log row afterward (§2/§7). Consumers retrieve it later by
  `prediction_id` (via the log, or a future `GET /prediction/{id}`).

`verbose` adds **no model inference** - it only packages values already computed
during the prediction - so latency is unchanged for default traffic and only
marginally larger (serialization of ≤20 candidates + metadata) when requested.

## 9. One remaining gap: no migration tool

`create_all()` is fine for a fresh environment/tests; a running production
database will want a real migration (Alembic or similar) once this ships and
needs to evolve further.

## 10. Scalability

**Partitioning.** Not applied today (§3) - a single table with a
`served_at` index is the pragmatic starting point, matching v1's own
documented fallback option. If/when retention (`DELETE ... WHERE log_date <
...`) or write volume genuinely becomes a bottleneck, native Postgres
partitioning by `log_date` (monthly) is the natural upgrade - at that point
`prediction_id` alone can no longer be the sole primary key (Postgres
requires the partition column in any unique constraint on a partitioned
table), so the key would become `(prediction_id, log_date)`. Not needed
until volume justifies the operational overhead.

**Write-volume control.** Collapsing v1's per-candidate *child table* into
this single row was the main volume-control decision from the redesign — the
dominant cost in v1 at any real QPS was 100+ child **rows** per prediction
(row overhead, index maintenance, fan-out on every insert). The full sweep is
still retained for auditing/explainability, but as **one JSON array**
(`candidate_evaluations`) on the parent row rather than N rows: same
information, a single write, no fan-out. `candidate_bid_generation` remains as
the compact description of *how* that candidate set was produced. If even the
JSON array's size becomes a concern, it can be capped or turned off per-request
via `decide_bid(include_candidates=...)` with no schema change (the column just
goes null).

**Decoupling the write path from the request path.** `/explain_bid`'s row
insert is still synchronous (simplest, and that route was never the fast
path to begin with - see §7). `/recommend_bid` is fully decoupled as of
2026-07-23: both the row insert and the SHAP enrichment run as
`BackgroundTasks` jobs, so nothing about logging - not write latency, not
any derived metric computed for the row - is ever on that response's
critical path. This wasn't primarily a volume-scaling decision (the
motivating case was SHAP being too slow to sit on the live bid path at
all); it happens to also remove the plain row insert's DB-write latency
from the response, for free. See §7 for the failure-isolation approach
(and the `prediction_id` trade-off it implies) that keeps this safe.

**Storage location.** Reuses the existing shared Postgres
(`smarthub.core.config_store`'s instance) by default via
`SMARTHUB_PREDICTION_LOG_DB_URL` (mirroring `SMARTHUB_CONFIG_DB_URL`);
defaults to SQLite in tests.

## 11. Explicitly not in scope here

- A real migration tool for evolving this schema in production (§9).
- Threading `lead_ping_id` through whatever upstream system calls
  `/recommend_bid`/`/explain_bid` today - the field exists on `BidRequest`
  now (§8), but the caller still has to actually start sending it.
- Async/queued writes (§10) - not needed at today's volume.
- Native Postgres partitioning (§10) - not needed at today's volume.

## 12. What changed from v1 (the 3-table design) and why

| v1 (3 tables) | v2 (this document, 1 table) | Why |
| --- | --- | --- |
| `smarthub_optimizer_bid_evaluations` - one row per candidate bid (100+ per prediction) | `candidate_evaluations` JSON array (full per-candidate sweep) + `candidate_bid_generation` (how the set was made), both on the parent row | Keeps the complete per-candidate evaluation history (needed to reconstruct the decision without the model) but as one JSON array instead of 100+ child rows — same information, a single write, no row/index fan-out. |
| `smarthub_shap_explanations` - separate 1:1 child table | `shap_explanation` JSON column | No independent lifecycle from the parent row (always 1:1, never queried alone in practice) - a plain column is simpler with no loss of information. |
| Composite PK `(prediction_id, log_date)` everywhere, driven by native partitioning | Plain PK `prediction_id`, `log_date` as an indexed column | No partitioning is applied yet (§10) - the composite key was only ever a Postgres partitioning requirement, not a modeling need, so it's dropped until partitioning is actually adopted. |
| Only successful predictions modeled | `status` / `error_message` added | Vinaya's explicit ask: "a single line for every API request... whether it has succeeded or failed." |
| No config-versioning | `serving_config` JSON column added | Vinaya's same-meeting ask: predictions need to be reproducible against the exact serving-policy config used, independent of `config/smarthub.yaml` changing later. |
| Never wired into a live endpoint (design doc only) | Actually implemented and called from `/recommend_bid` / `/explain_bid` | This revision is the implementation pass, not just a design pass. |
| `BidRequest` had no `lead_ping_id`; nothing returned to correlate a response back to its log row | Optional `lead_ping_id` added to `BidRequest`; `prediction_id` now returned in every response | Follow-up fix, same implementation pass: without one of these two, the log wasn't reliably joinable back to "which lead/request was this" - see §8. |
| `shap_explanation` populated only by `/explain_bid` - always `null` for `/recommend_bid` rows | `/recommend_bid` also populates it, via a `BackgroundTasks` job after the response (`{"top_factors", "base_win_rate"}` only - no `bid_curve`/LLM `explanation`) | 2026-07-23 follow-up: every prediction (not just on-demand `/explain_bid` calls) can now be audited/calibrated against its own SHAP factors, without adding an Ollama call to the live bid path - see §2/§7. |
| `recommended_bid_expected_profit` | `recommended_bid_predicted_profit` | 2026-07-23 rename: matches `recommended_bid_predicted_win_rate`'s "predicted_" naming - both are model predictions, not realized values, so "expected" (a different word for the same idea) was dropped to avoid confusion. |
| No CM prediction stored | `recommended_bid_predicted_cm` added (`recommended_bid_predicted_profit / expected_revenue`) | 2026-07-23: a directly useful business metric that's one division away from columns already on the row - cheap to add, no reason to make a caller recompute it. |
| `/recommend_bid`'s row insert ran synchronously, on the response path | Runs entirely as a `BackgroundTasks` job; `prediction_id` is generated up front (`uuid.uuid4()`) and returned before the insert happens | 2026-07-23: explicit ask that logging (including every derived metric, like the new `recommended_bid_predicted_cm`) never delay a bid response. Trade-off: `prediction_id` is now a receipt for "log under this id," not a guarantee the row exists yet - see §7/§8. `/explain_bid`'s logging is unchanged (still synchronous). |
