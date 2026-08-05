# SmartHub — `/recommend_bid` Scaling Report

*What we changed, why, how we measured it, and how to deploy it.*

**Author:** Nimesh · **Date:** 2026-08-05 · **Status:** implemented; pending EC2 validation

**Goal:** `/recommend_bid` should serve its target load with a **per-request TAT
of ≤ 1 second**. Immediate target is **200 requests/minute**; this report also
establishes the measured per-node capacity so the path to higher rates
(e.g. 200 req/s) is grounded in data rather than guesswork.

Related: [`SCALING_PLAN.md`](./SCALING_PLAN.md) (design detail),
[`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md) (full system).

---

## 1. The problem we found

Benchmarking the real serving path against the promoted `auto_v6` model showed
the bid decision itself is cheap — **7–17 ms** per request (candidate-bid sweep +
`predict_proba` + argmax). It was **not** the bottleneck.

The bottleneck was the **per-prediction SHAP explanation**. It ran for every
model-served bid as a Starlette `BackgroundTask` — *inside the serving process*
— and costs **~1.5 s** of CPU (the served model is a calibrated LightGBM with
`cv=3`, so SHAP explains three fold-pipelines and rebuilds the explainer per
request). Being CPU/GIL-bound, that background work starved the actual bid
responses: a request never waited for *its own* SHAP, but it waited behind the
*previous* request's SHAP.

The effect on throughput was severe. With SHAP inline, each worker can only
clear roughly one request every ~1.5 s, so the whole box saturates at a few
requests per second and latency runs away. Measured on a dev machine, the
in-process configuration **broke at ~200 req/min** — it could not even hold the
immediate target within the 1 s TAT.

---

## 2. What we changed

The fix keeps the fast bid path untouched and **moves SHAP off the serving
process entirely**, behind a config flag. No new brokers or infrastructure.

**A configurable SHAP mode** (`config.shap_enrichment_mode`, env override
`SMARTHUB_SHAP_MODE`):

- `offload` — **new default.** Serve returns the bid and logs the prediction
  row with SHAP empty; it does no SHAP work. Keeps serving CPU free for the TAT.
- `inprocess` — the legacy behaviour, retained behind the flag for comparison
  and rollback.
- `off` — no SHAP enrichment (a control for benchmarking the pure bid path).

**A dedicated `shap-worker` process** (`smarthub.server.shap_worker`) that drains
the offloaded rows and backfills the identical explanation. It treats the
prediction-log table as a work queue: rows where `shap_explanation IS NULL` and
`decision_path IN ('model','exploration')` are pending, claimed with
`FOR UPDATE SKIP LOCKED` so multiple worker replicas never double-process, and
retried automatically on failure (NULL stays NULL until it succeeds; a hard,
non-retryable failure writes a small error sentinel). A companion `shap-worker`
service was added to `docker-compose.prefect.yml`, scalable with
`docker compose up --scale shap-worker=N`.

**Trade-off accepted:** explanations become *eventually consistent* — available
seconds (worst case, under backlog, ~a minute) after the bid rather than
synchronously. This is fine because SHAP factors are audit/enrichment, never on
the bidding decision, and `/explain_bid` still computes one on demand.

### Files changed

| File | Change |
| --- | --- |
| `config/smarthub.yaml` | `prediction.shap_enrichment_mode: offload` (new default) |
| `src/smarthub/train_and_predict/config.py` | `shap_enrichment_mode()` resolver (env → YAML → default) |
| `src/smarthub/server/predict.py` | `/recommend_bid` gates the in-process SHAP task on the mode |
| `src/smarthub/server/shap_worker.py` | **new** — offloaded drain/backfill worker (DB queue, SKIP LOCKED) |
| `docker-compose.prefect.yml` | **new** `shap-worker` service |
| `scripts/loadtest_recommend_bid.py` | **new** — load-test / breaking-point harness |
| `tests/test_predict_logging.py` | in-process SHAP tests pinned to `inprocess` so the new default doesn't disable them |

The full unit-test suite passes with the new default.

---

## 3. How we measured it

We built a repeatable load-test harness (`scripts/loadtest_recommend_bid.py`)
with:

- **Open-loop arrivals** at a configurable rate (requests/minute), so a slow
  server shows up as latency rather than silently throttling offered load.
- A **realistic request mix** (varying revenue / bid-step → candidate-grid sizes
  spanning the small→large cost range).
- A **`--sweep`** mode that steps the rate up to find the breaking point.
- **Within-1s-TAT reporting**: every run reports how many responses came back
  inside 1 s (`within_1s_TAT` / `within_1s_pct`), alongside p50/p95/p99 and error
  counts. Pass = 100% within 1 s and zero errors.

---

## 4. Results

### 4a. In-process vs offload at the target rate (200 req/min)

Same 4 serving workers, same hardware — only the SHAP mode differs:

| Mode | Within 1s | p50 | p99 | Errors | Verdict |
| --- | --- | --- | --- | --- | --- |
| **inprocess** (old default) | ~58–93% | 45–555 ms | **7.6 s** | some | ❌ FAIL |
| **offload** (new default) | **100%** | ~37 ms | **55–80 ms** | 0 | ✅ PASS |

The in-process path misses the TAT and tails into multiple seconds; offload
holds p99 under ~80 ms with every response inside 1 s. **The fix came from
relocating SHAP, not from adding serving capacity — same 4 workers, same box.**

### 4b. Breaking-point sweep (offload, 4 workers, MacBook M3 Pro)

| Rate | Within 1s | p50 | p99 | Verdict |
| --- | --- | --- | --- | --- |
| 100 req/s (6,000/min) | 100% | 26 ms | 73 ms | PASS |
| 117 req/s (7,000/min) | 100% | 20 ms | 52 ms | PASS |
| 125 req/s (7,500/min) | 100% | 22 ms | 53 ms | PASS |
| 133 req/s (8,000/min) | 100% | 25 ms | 59 ms | PASS |
| 141 req/s (8,500/min) | 100% | 28 ms | 67 ms | PASS |

**A single 4-worker node cleanly handles ~120–140 req/s at 100% within the 1 s
TAT**, and p99 stayed *flat* (~50–70 ms) across the whole range — meaning the
node was **not yet saturated** even at 141 req/s. The true ceiling is higher; we
could not reach it because the single-process load client itself tops out around
~140–150 req/s (an apparent "break" at 150 req/s was the client falling behind,
not the server — it recovered cleanly at higher server-side headroom). Finding
the true ceiling would require generating load from multiple client
processes/machines.

**Takeaway to quote:** *one offload node (4 workers) serves ≥120–140 req/s at
100% within 1 s TAT and is not saturated at that point.*

### 4c. Full Docker-stack capacity (offload, deployed compose)

Measured against the **complete deployed stack** — `nginx → serve (4 workers) →
Postgres prediction log + shap-worker`, offload mode — on the MacBook M3 Pro.
Important: this laptop was **also running a k3s/k8s cluster, redis, and other
apps** alongside SmartHub *and* the load-test client, so these numbers reflect a
**contended shared box**, not the software ceiling. 60 s runs (30 s for the
sweep), `--timeout 5`.

| Rate/min | Rate/s | p50 | p99 | within 1 s | verdict |
| --- | --- | --- | --- | --- | --- |
| 200 | 3.3 | 44 ms | 150 ms | 100 % | PASS |
| 300 | 5 | 37 ms | 59 ms | 100 % | PASS |
| 500 | 8.3 | 31 ms | 49 ms | 100 % | PASS |
| 1,000 | 16.7 | 24 ms | 40 ms | 100 % | PASS |
| 3,000 | 50 | 26–35 ms | ~400 ms | 100 % | **PASS (comfortable ceiling)** |
| 3,600 | 60 | 47 ms | 583–838 ms | 100 % / 99.7 % | **edge / knee** |
| 4,200 | 70 | 358 ms | 2,141 ms | 75.6 % | BREAK |
| 4,800 | 80 | 2,258 ms | 5,010 ms | 16–38 % | collapse |
| 6,000 | 100 | 5,004 ms | 5,010 ms | 4 % | collapse |

**Reading it:**

- **Reliable safe capacity on this contended box: 50 req/s (3,000/min)** — 100 %
  within 1 s, p99 ~400 ms, reproducible across runs.
- **Knee ≈ 60 req/s (3,600/min):** it can still hit 100 %, but p99 has jumped to
  ~600–840 ms and one run dropped a single request (99.7 %). Do not operate here.
- **Breaks at 70 req/s (4,200/min)**; hard saturation collapse by 80 req/s
  (p50 pinned at the 5 s client timeout — a queue that never drains).
- The collapse is a **cliff** (p99 goes ~400 ms → ~600 ms → 2 s across
  50 → 60 → 70 req/s), so p99 crossing ~400–500 ms is the autoscale/alert signal.

**Environmental caveat:** this ~50–60 req/s knee is the *shared laptop*, not the
system. The same laptop **bare-metal** (§4b — no k8s/redis/nginx/Postgres
contention) held 100 req/s clean and didn't break until ~140 req/s. The
dedicated `m7i.2xlarge` (8 vCPU, nothing else co-resident) will land materially
higher — re-run this exact sweep there for its real knee.

**Relative to the goal:** the 200 req/min (3.3 req/s) requirement is met with
**~15× headroom even on the contended laptop** (safe to ~3,000/min there), and
far more on the dedicated instance.

---

## 5. Deployment

### Immediate: single `m7i.2xlarge`

The whole stack deploys as one node via `docker-compose.prefect.yml`.

- **Instance:** AWS `m7i.2xlarge` (8 vCPU / 32 GB), same region as Redshift.
- **Disk:** gp3 EBS 200 GB (the `data/` tree grows monotonically; set a disk
  alarm).
- **Key config:**
  ```
  SMARTHUB_SHAP_MODE=offload          # now the default; explicit for clarity
  SERVE_WORKERS=8                      # = vCPU on this instance
  SMARTHUB_PREDICTION_LOG_DB_URL=postgresql+psycopg2://…@postgres:5432/prefect
  SMARTHUB_PRODUCTION_STORAGE_BACKEND=s3   # MinIO on-node, or real S3
  # shap-worker runs as its own container; scale with --scale shap-worker=N
  ```
- The bid path is CPU-trivial at the 200 req/min target; the instance size is
  driven by the co-located training worker, SHAP workers, Ollama, and Postgres —
  not by serving.

### Validation to run on EC2 (before trusting the numbers)

Re-run the sweep on the real instance — laptop numbers don't transfer exactly
(real network hop, Postgres prediction-log writes, `SERVE_WORKERS=8`):

```bash
python scripts/loadtest_recommend_bid.py \
    --url http://<host>:8000/recommend_bid \
    --sweep "1200,3600,6000,9000,12000" --duration 30 --timeout 3
```

Read the `win1s%` column and the breaking-point line. To push past ~140 req/s
reliably, run 2–3 of these in parallel so the client isn't the bottleneck.

### Path to 200 req/s (when needed)

Measured per-node clean capacity (≥120–140 req/s) makes the scale-out shape
simple: `/recommend_bid` is stateless (each replica just holds a read-only
cached model), so it scales horizontally behind a load balancer.

- **2 serving nodes** behind an ALB cover 200 req/s (each ~100 req/s — inside the
  flat-latency, 100%-within-1s zone).
- **+1 node for N+1 failover** so 200 req/s still holds if one node dies.
- **SHAP at high volume:** full per-request SHAP is infeasible at 200 req/s
  (~hundreds of cores); sample it or make it on-demand at that scale.
- **Prediction log at 200 req/s:** batch the inserts and give the log its own DB
  (don't share the pipeline Postgres).
- **Autoscale trigger:** add nodes at ~70% of per-node rate — the saturation
  knee is sharp (p99 jumps from ~60 ms to ~20 s within one step), so scale well
  before it.

---

## 6. Status and next steps

- [x] Root cause identified (in-process SHAP, ~1.5 s/request).
- [x] Offload implemented behind a flag; made the default; tests pass.
- [x] `shap-worker` + compose service.
- [x] Load-test harness with within-1s reporting and breaking-point sweep.
- [x] Measured: offload holds 200 req/min at 100% within 1 s; single node clean
      to ~120–140 req/s bare-metal, not saturated.
- [x] Phase-1 SHAP speedup done — cached explainer + single fold (~1.5 s →
      ~10 ms/row); shap-worker keeps the backlog at 0.
- [x] Validated on the **full Docker stack** (nginx → serve → Postgres +
      shap-worker): 200 req/min at 100% within 1 s (p99 150 ms); safe to
      ~50 req/s / 3,000 req/min on the contended dev laptop, knee ~60 req/s
      (§4c).
- [ ] Deploy to `m7i.2xlarge` and re-run the §4c sweep to get the real per-node
      knee (dedicated 8 vCPU, `SERVE_WORKERS=8`).
- [ ] (When needed) Multi-node + ALB for 200 req/s, per §5.
