# SmartHub — Serving Scaling Plan: `/recommend_bid` to 200 req/min

**Owner:** Nimesh · **Date:** 2026-08-03 · **Status:** Proposed (pre-implementation)

**Goal:** `/recommend_bid` reliably serves **≥ 200 requests/minute (3.33 req/s)**
with a **per-request TAT of ≤ 1 s** (business SLA), with clear headroom for
growth — *without* depending on adding hardware as the primary lever.

> **SLA note — two distinct numbers, don't conflate them:**
> - **Design target: ≤ 1 s per request.** This is achieved by *engineering* —
>   the system is built so every `/recommend_bid` actually completes in ~1 s
>   (p99 < 1 s under load). This is what the work below delivers.
> - **nginx timeout: 10 s (safety net).** A generous circuit-breaker so a
>   pathological request is eventually cut rather than hanging forever — it is
>   **not** the SLA and should never be the thing that "enforces" 1 s. We set
>   `/recommend_bid` to **10 s** (up from today's 5 s; §3.3). If requests are
>   routinely anywhere near 10 s, the design has failed and the load test
>   (p99 < 1 s) is what catches it — long before the timeout would.

---

## 1. Diagnosis (measured, not assumed)

Benchmarked against the live promoted model `data/models/auto/` → `auto_v6`
(`CalibratedClassifierCV`, isotonic, `cv=3`, 18 features) on a 4-CPU box.

| Path | Where | Cost / request |
| --- | --- | --- |
| Bid decision — candidate-bid sweep + `predict_proba` | `optimizer.optimize_bid_for_row` (`optimizer.py:92`) via `decide_bid` (`predict.py:389`) | **7–17 ms** (p99 ≈ 31 ms even at 637 candidate bids) |
| Prediction-log insert | `_log_prediction_safe` → `PredictionLogStore.log_prediction` (Postgres, single-row insert) | ~1–5 ms, already off the response path |
| **SHAP enrichment** (runs for every model-served prediction) | `_log_shap_background` (`predict.py:831`) → `explain_from_prediction` | **~1,500 ms** |

### Root cause

The bid decision is **not** the bottleneck — it could serve thousands of
requests/minute. The ceiling is the **SHAP explanation** that fires for every
`model`/`exploration` prediction. Two properties make it fatal to throughput:

1. **It is expensive (~1.5 s).** The served model is a `CalibratedClassifierCV`
   with `cv=3`, so `shap_explain` builds a SHAP `TreeExplainer` and explains
   **all three** underlying LightGBM fold-pipelines per prediction
   (`shap_explain._fitted_lgbm_estimators`, `:29`). The dominant cost is
   *constructing the explainer per request*, which depends only on the trees —
   not on the row being explained.
2. **It runs inside the serving process.** It is scheduled as a Starlette
   `BackgroundTask` (`predict.py:1027`), which executes in the same uvicorn
   worker's threadpool *after* the response is sent. Because SHAP is
   CPU-bound Python/NumPy work that largely holds the GIL, it directly
   competes with incoming bid requests for CPU.

### Why 200/min fails today

```
Target load        = 3.33 req/s
SHAP cost          = ~1.5 CPU-s per prediction (fires on ~every request)
SHAP CPU demand    = 3.33 × 1.5  = ~5.0 CPU-s per wall-second
Serve capacity     = 4 CPUs (nproc=4, SERVE_WORKERS=4, mem_limit 4g)
```

SHAP **alone** oversubscribes the box (5.0 vs 4.0), leaving no CPU for the bid
responses. In isolation a response is ~15 ms — comfortably inside the **1 s**
TAT. But under load those cheap responses queue behind SHAP's GIL-bound CPU
work, and latency climbs. Against the **1 s** design target the SLA breaks well before any timeout would
fire: the safe sustained rate that keeps p99 < 1 s is only **~120–140 req/min**,
well short of the 200/min target. (The 10 s nginx timeout won't even trip at
that point — the requests aren't erroring, they're just too slow to be useful,
which is exactly why the timeout must not be treated as the SLA.)

**Conclusion:** hitting 200/min is primarily about **removing SHAP from the
serving hot path** and **making SHAP itself cheap**, then proving it with a
load test. Hardware scaling is explicitly rejected as the primary lever — it is
costlier, and it leaves the per-prediction 1.5 s tax in place, so any traffic
spike re-breaks the SLA.

---

## 2. Strategy

Three changes, in priority order. (1) and (2) are the reliability fix; (3)
proves it.

1. **Decouple SHAP from serving** — the serve process stops computing SHAP.
   Explanations are computed by a separate consumer and backfilled onto the
   already-logged prediction row. Explainability coverage stays at ~100 %; the
   1.5 s tax leaves the serving CPU entirely.
2. **Make SHAP cheap** — cache the SHAP explainer per model version and explain
   a single calibration fold instead of three; batch rows in the consumer. This
   takes per-row SHAP from ~1.5 s to single-digit ms.
3. **Load test to the SLA** — a repeatable harness that drives `/recommend_bid`
   at 200+ req/min, ramps to find the true ceiling, and asserts p95 < 5 s with
   zero timeouts. Baseline before, verify after.

The fast predict path is left as-is. The prediction-log insert stays an
in-process background task (it's already ~ms and off the response path).

---

## 3. Design

### 3.1 Decouple SHAP — DB-as-queue + dedicated consumer

The prediction log already lives in shared Postgres and already has the target
column: `shap_explanation` (Text/JSON), updated by
`PredictionLogStore.update_shap_explanation(prediction_id, dict)`
(`prediction_log_schema.py:356`). We reuse it as the work queue — no new broker
infra.

**Serve side (removal):**
- Delete the `background_tasks.add_task(_log_shap_background, …)` call
  (`predict.py:1027`). Keep the prediction-log insert task. The row is written
  with `shap_explanation = NULL`, which *is* the "pending" marker.

**Consumer side (new `shap-worker` service):** a small long-running loop that
claims pending rows safely and backfills them:

```sql
SELECT prediction_id, lead_type_id, model_input_features,
       recommended_bid, recommended_bid_predicted_win_rate,
       model_version                     -- to load the right model
FROM   prediction_log
WHERE  shap_explanation IS NULL
  AND  decision_path IN ('model', 'exploration')   -- cold-start has no model
  AND  status = 'ok'
ORDER  BY created_at
LIMIT  :batch
FOR UPDATE SKIP LOCKED;                   -- safe concurrency across N workers
```

For each claimed batch: load the model for that version (cached), compute SHAP
for all rows in one explainer pass, then `update_shap_explanation` per row.
`FOR UPDATE SKIP LOCKED` makes it safe to run multiple `shap-worker` replicas.
Failures simply leave `shap_explanation = NULL`, so the row is naturally
retried on the next pass — durable and idempotent by construction.

**Why a dedicated service (not the Prefect worker):** it isolates SHAP CPU from
both serving *and* the single Prefect training/ingestion worker (which already
runs at a 4 GB limit and can SIGKILL on big feature builds). The `shap-worker`
gets its own CPU budget and can be replicated independently. A Prefect
scheduled deployment (every ~1 min, batch drain) is a viable lighter-weight
alternative if we'd rather not add a container — noted as an option, not the
recommendation.

**Latency contract:** explanations become **eventually consistent** — available
seconds to (worst case, under backlog) ~a minute after the bid. This is
acceptable: SHAP factors are audit/enrichment, never on the real-time bid path,
and `/explain_bid` already computes explanations on demand for the interactive
case.

### 3.2 Make SHAP cheap

The ~1.5 s is dominated by rebuilding the explainer per request over 3 folds.
Two changes in `shap_explain.py`:

1. **Cache the explainer per model version.** A SHAP `TreeExplainer` depends
   only on the fitted trees, not on the row. Build it once per
   `(model_version, fold)` and reuse across all predictions. This alone removes
   the bulk of the cost.
2. **Explain one fold, not three.** The module's own docstring notes isotonic
   calibration is a monotonic rescaling that does not change *which* features
   matter or their ranking. For the logged `top_factors` / `base_win_rate`,
   explaining a single representative fold (or the base LightGBM) is a
   defensible ~3× reduction. (Keep the 3-fold average available behind a config
   flag for `/explain_bid` if we want maximum fidelity there.)
3. **Batch in the consumer.** SHAP over N rows in one call amortizes the
   (already-cached) explainer overhead further.

Expected result: per-row SHAP from ~1.5 s → **low single-digit ms**, at which
point even an in-process fallback would fit — but we keep it decoupled for
isolation and burst absorption.

### 3.3 Serving path — confirm and harden (no behavior change)

- Predict path unchanged (7–17 ms — ~60× under the 1 s TAT). With SHAP gone,
  `SERVE_WORKERS=4` × ~15 ms ≈ **hundreds of req/s** of headroom — 200/min at
  a 1 s TAT becomes trivial, and p99 stays well inside the budget.
- **Set the nginx `/recommend_bid` timeout to 10 s** (up from today's 5 s in
  `nginx/nginx.conf`) as a generous safety net. The 1 s TAT is met *by design*
  and verified by the load test (p99 < 1 s) — the timeout is only a
  circuit-breaker for pathological cases, deliberately loose so it never
  false-trips a healthy-but-slightly-slow request. Keep the 10 s/60 s budgets on
  `/explain_bid`, `/docs`, `/health` as-is.
- Add explicit CPU **reservations** for `serve` in compose so it isn't starved
  by co-located containers (MinIO, MLflow-UI, Ollama, dashboard) on the single
  node.
- Keep the prediction-log insert as an in-process background task; revisit only
  if Postgres write contention appears (it won't at 3.3 inserts/s).

### 3.4 Future lever (documented, not required now)

Horizontal `serve` replicas behind nginx: switch `nginx.conf` from a single
`server serve:8000` to an upstream with multiple backends (or Docker Compose
`deploy.replicas` + DNS round-robin, which the current `resolver` setup already
tolerates). This is the growth path beyond ~1000 req/min; not needed to hit
200/min once SHAP is decoupled.

---

## 4. Load-test harness

A repeatable script (`scripts/loadtest_recommend_bid.py`, Locust or
asyncio+httpx) that:

- Drives `/recommend_bid` through nginx at a configurable rate (default 200/min,
  ramp mode to find the ceiling).
- Uses a **realistic request mix** — varied `expected_revenue`/`bid_step` so the
  candidate-bid count (and thus per-request cost) spans the small→large range
  measured above.
- Reports throughput, p50/p95/p99 latency, and **count of >1 s / timeout /504
  responses**.
- **Pass criteria:** ≥ 200 req/min sustained for 5 min, **p99 < 1 s** (the
  business TAT), **zero** responses over 1 s; then report the true breaking
  point (rate at which p99 first crosses 1 s).

Run it (a) as a baseline against `main` to confirm the ~150/min ceiling, and
(b) after each phase to verify the improvement.

---

## 4a. Measured A/B result (implemented behind a flag)

Both paths now coexist behind `config.shap_enrichment_mode`
(`inprocess` | `offload` | `off`; env override `SMARTHUB_SHAP_MODE`), so we can
compare before removing anything. Measured locally against the live `auto_v6`
model, 4 uvicorn workers on a 4-CPU box, realistic request mix, open-loop
arrivals (`scripts/loadtest_recommend_bid.py`; raw runs in
`docs/loadtest_results.jsonl`):

| Mode | Rate | p50 | p90 | p99 | max | > 1 s TAT | non-200 | SLA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **inprocess** (current) | 200/min | 220 ms | 6.2 s | 11.9 s | 11.9 s | **40.3 %** | 4 | ❌ |
| **offload** (new) | 200/min | 37 ms | 48 ms | 55 ms | 55 ms | **0 %** | 0 | ✅ |
| **offload** (new) | 1200/min (6×) | 25 ms | 64 ms | 69 ms | 78 ms | **0 %** | 0 | ✅ |

The in-process path fails the 1 s TAT hard at the *target* rate — 40 % of bids
breach 1 s and tail latency runs to ~12 s — exactly the GIL/CPU contention
predicted. Offloading SHAP holds p99 at **55 ms at target and 69 ms at 6× target**
with zero breaches. Decision leans strongly toward making `offload` the default;
`inprocess` stays available behind the flag until we're confident in production.

**One caveat on the offload side (worker capacity, not serving TAT):** SHAP is
still ~1.5 s/row today, so a single `shap-worker` drains ~0.66 rows/s. At
200/min (3.3 preds/s) the *explanation* backlog would grow unless we either
apply the Phase 1 explainer-caching speedup (drops per-row cost to ~ms) or run
~5 worker replicas (`docker compose up --scale shap-worker=5`; safe via `FOR
UPDATE SKIP LOCKED`). This does **not** affect the bid TAT — bids are already
fast and unblocked — it only governs how quickly the audit explanation lands.
Phase 1 is the clean fix and is recommended next.

---

## 5. Phased rollout

| Phase | Change | Outcome | Risk |
| --- | --- | --- | --- |
| **0. Baseline** | Build load-test harness; measure current ceiling | Documented ~150/min ceiling + latency curve | none (read-only) |
| **1. Cheap SHAP** | Cache explainer per model version + single-fold, in `shap_explain.py` | Per-prediction SHAP ~1.5 s → ~ms; may already clear 200/min in-process | low — approximation of calibrated SHAP; guard behind flag |
| **2. Decouple** | Remove in-process SHAP task; add `shap-worker` DB-queue consumer + compose service | SHAP off serving CPU entirely; ~100 % coverage, eventually consistent | med — new service, backlog handling, multi-worker correctness (mitigated by `SKIP LOCKED`) |
| **3. Verify + tune** | Re-run load test; set `serve` CPU reservations; tune `SERVE_WORKERS`, batch size, poll interval | Proven ≥ 200/min with headroom; documented new ceiling | low |
| **4. (Future) Replicas** | nginx upstream with multiple `serve` backends | Path to ≫ 1000/min | deferred |

Phase 1 is low-risk and independently shippable; it may itself be enough for
200/min. Phase 2 is what makes the target **reliable under bursts** and is the
recommended end state.

---

## 6. Risks & mitigations

- **Explanation staleness / backlog.** Under a sustained spike, pending SHAP
  rows accumulate. *Mitigation:* batch consumer + `shap-worker` replicas;
  monitor backlog depth (`COUNT(*) WHERE shap_explanation IS NULL`) and alert.
  Explanations being minutes-late is acceptable by design.
- **Single-fold SHAP approximation.** Ranking is preserved under monotonic
  calibration, but absolute contribution magnitudes shift slightly.
  *Mitigation:* config flag to restore 3-fold averaging for `/explain_bid`;
  document that logged `top_factors` use the fast path.
- **Lost explanations on consumer failure.** *Mitigation:* NULL-means-pending +
  `SKIP LOCKED` makes retries automatic and idempotent; no work is lost.
- **Shared Postgres contention.** The log DB is co-tenant with Prefect + MLflow
  + config store. At 3.3 inserts/s + batched updates this is negligible; flagged
  as the next decoupling target if write volume grows an order of magnitude.
- **Model-version skew.** The consumer must explain with the *same* model
  version that served the bid (`model_version` on the row), not "current".
  *Mitigation:* load by version from the row; cache by version.

---

## 7. Observability (add alongside)

- Structured timing on the predict path and SHAP consumer (p50/p95).
- Gauge: SHAP backlog depth (pending rows).
- Counter: nginx 504s on `/recommend_bid` (the SLA breach signal).
- Load-test results checked in as the regression baseline.

---

## 8. Definition of done

1. Load-test harness in `scripts/`, baseline recorded.
2. SHAP explainer cached + single-fold path (Phase 1).
3. `shap-worker` consumer + compose service; in-process SHAP task removed
   (Phase 2).
4. Load test: **≥ 200 req/min, 5 min, p99 < 1 s (TAT), 0 responses > 1 s**,
   plus the new measured ceiling documented here.
5. nginx `/recommend_bid` timeout set to 10 s (safety net; TAT met by design).
6. Backlog metric + 504 counter wired up.

---

### Appendix — key code references

- Bid decision: `src/smarthub/server/predict.py:389` (`decide_bid`),
  `optimizer.py:92` (`optimize_bid_for_row`)
- In-process SHAP task to remove: `predict.py:831` (`_log_shap_background`),
  scheduled at `predict.py:1027`
- SHAP internals to cache/trim: `train_and_predict/shap_explain.py:29`
  (`_fitted_lgbm_estimators`, the 3-fold loop), `:76` (`_shap_for_row`)
- Backfill API (queue drain target): `prediction_log_schema.py:356`
  (`update_shap_explanation`), schema `:73`+ (`shap_explanation` col `:123`)
- Serving deploy: `docker/Dockerfile.serve` (`SERVE_WORKERS`),
  `docker/nginx/nginx.conf` (5 s `/recommend_bid` timeout),
  `docker-compose.prefect.yml:209` (`serve`, `mem_limit 4g`)
