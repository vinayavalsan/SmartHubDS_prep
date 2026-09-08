# SmartHub Serving — Rate Limiting & Concurrency Limits

This documents the overload protection in front of the bid API (`/recommend_bid`):
the nginx rate/connection limits and uvicorn's per-worker concurrency cap — what
they are, **why the thresholds are what they are**, and the load-test evidence
behind them.

Scope: the real-time bid path only. `/health`, `/explain_bid`, and the dashboards
are unaffected.

---

## 1. The problem this fixes

Before this change there was **nothing between a caller and the serving event
loop** — no `limit_req` in nginx, no limiter in FastAPI, no per-key quota. Two
consequences:

- **Latency collapse under load.** uvicorn queued requests without bound, so past
  a concurrency knee the p99 fell off a cliff — throughput stopped rising while
  latency ran away, blowing the 1-second bid SLA.
- **DoS exposure (critical issue #2).** A single client — or a single abusive IP —
  could saturate the service and take bidding down for everyone.

The fix is defense-in-depth: shed overflow **cheaply and early** at nginx, with a
hard concurrency backstop at uvicorn, so the service **fails fast (429/503)
instead of degrading for everyone**.

---

## 2. What we set

### nginx (`docker/nginx/nginx.conf`, on `location = /recommend_bid`)

Overflow is rejected with **HTTP 429**.

| Limit | Keyed by | Rate / cap | Burst | Purpose |
| --- | --- | --- | --- | --- |
| `limit_req perip` | client IP (`$binary_remote_addr`) | **50 r/s** | 100 (`nodelay`) | DoS / single-source flood shield. Works on **unauthenticated** traffic (before the key check), so it's what actually caps a flood. |
| `limit_req perkey` | API key (`$http_authorization`) | **100 r/s** | 200 (`nodelay`) | Per-tenant fairness — one client can't eat the whole serving budget. |
| `limit_conn perip_conn` | client IP | **20** concurrent connections | — | Slowloris / connection-exhaustion guard. |

### uvicorn (`docker/Dockerfile.serve`)

Overflow is rejected with **HTTP 503**.

| Setting | Value | Effect |
| --- | --- | --- |
| `--limit-concurrency` | **8 per worker** (env `SERVE_LIMIT_CONCURRENCY`, default 8) | With `SERVE_WORKERS=4` → **32 aggregate** in-flight requests. Above it, uvicorn returns 503 immediately instead of queuing. |

### How the two layers compose

A flood hits the nginx **429** first (per-IP 50 r/s). If traffic is *within* the
rate limit but still piles up concurrently (e.g. a burst of slow requests),
uvicorn's **503** is the deeper backstop at 32 in-flight. In normal production a
client sees neither — both are protection, not part of the happy path.

Both are tunable **without a code change**: `SERVE_LIMIT_CONCURRENCY` for the
concurrency cap; edit `rate=` in `nginx.conf` and reload nginx for the rate.

---

## 3. Why these thresholds

### `--limit-concurrency 8` (per worker)

Chosen from a **closed-loop concurrency sweep** (`scripts/concurrency_sweep.py`):
hold N concurrent clients firing back-to-back, measure throughput + latency, step
N up, and find the knee. Uncapped baseline (§4.1):

- Throughput **saturated at the knee (~16 aggregate concurrency)** — beyond it,
  adding concurrency produced **zero** extra throughput and only piled on latency.
- p99 climbed 206ms → 550ms → 990ms → ~2030ms as concurrency doubled from 16 to
  128, crossing the 1s SLA between 32 and 64.

So the useful ceiling is ~16 aggregate concurrency; past it you're only building
queue. We set the cap at **32 aggregate (8 × 4 workers)** — deliberately a little
**above** the throughput-optimal knee so brief bursts still get served, while
p99 at that point is still well under SLA (**416ms on EC2**, §4.2). Anything
beyond 32 — the region where p99 runs to the cliff — is shed as 503.

Per-worker math: the cap is enforced **per uvicorn process**, so the number is
`aggregate ÷ workers = 32 ÷ 4 = 8`. Change `SERVE_WORKERS` and the per-worker
value should track it.

### nginx `rate=` (per-IP 50 r/s, per-key 100 r/s)

Sized just under the measured **saturation throughput (~110–133 rps aggregate)**
so nginx sheds overflow before uvicorn has to. The per-IP zone (50 r/s) is the
tighter one because it's the single-source flood shield; the per-key zone (100
r/s) is looser because it's about fairness between tenants, not stopping a flood.

> **These two rates are placeholders.** They passed the tests, but a real
> high-volume client (Anton) calling from one egress IP could exceed 50 r/s and
> get 429'd on legitimate bids. **Before production, set both `rate=` values to
> the client's agreed peak rps** (per-IP ≥ per-key, since one client = one IP
> here). See §6.

`limit_conn 20` and `burst` values are standard hardening: enough headroom for a
legitimate client's connection pool and short spikes, tight enough to stop
slowloris and sustained floods.

---

## 4. Proof

All runs use the closed-loop / burst / soak scripts in `scripts/` against the
current bid contract, with a deletable sentinel `lead_ping_id = 2000000000`.

### 4.1 Baseline — the cliff, *without* the cap

Concurrency sweep, no `--limit-concurrency`. Throughput saturates at C≈16 and p99
runs away past it — the failure mode we're fixing:

| Concurrency | throughput/s | p99 (ms) | <1s % |
| --- | --- | --- | --- |
| 8 | 99.6 | 130 | 100% |
| 16 | 130.4 | 206 | 100% |
| 32 | 133.1 | 550 | 100% |
| 64 | 133.2 | **990** | 99.1% |
| 128 | ~133 | **~2030** | ~50% |

Throughput is flat from 16 onward (130 → 133 → 133) while p99 quadruples — classic
saturation. This is why the knee sits at ~16 and the cap belongs just above it.

### 4.2 With the cap — EC2, direct to serve (bypassing nginx)

`--limit-concurrency 8` live. p99 stays bounded at **every** load level, and
overload is shed as 503 instead of latency:

| Concurrency | throughput/s | p99 (ms) | <1s % | 503 shed |
| --- | --- | --- | --- | --- |
| 8 | 58.9 | 240 | 100% | — |
| 16 | 98.9 | 283 | 100% | — |
| 32 | 109.9 | 416 | 100% | 12,719 |
| 64 | 98.1 | 448 | 100% | 27,944 |
| 128 | 65.8 | **456** | **100%** | 35,476 |

p99 never exceeds **456ms** even at 128 concurrent (vs ~2030ms uncapped), and
`<1s` is **100%** throughout. The cliff is gone.

### 4.3 nginx rate limit — EC2, flood through the proxy

Burst of 6,000 requests at concurrency 64 through nginx:

| Result | Count | Share |
| --- | --- | --- |
| 200 (served) | 196 | 3.3% |
| 429 (nginx rate-limited) | 5,804 | 96.7% |
| 503 (uvicorn) | 0 | 0% |

Served-request p99 was **275ms**. The flood is shed at the proxy as 429 **before**
it can pressure serve — which is why no 503 was needed. This is the intended
"429 first, 503 as backstop" layering.

### 4.4 Sustained soak — EC2, production-rate through nginx

Steady 40 r/s (under the 50 r/s per-IP cap), through nginx:

| Duration | Requests | Served (200) | p50 / p95 / p99 (ms) | 429/503 |
| --- | --- | --- | --- | --- |
| 5 min | 12,000 | **100%** | 78 / 108 / 138 | 0 |
| 10 min | 24,000 | **100%** | 78 / 107 / 138 | 0 |

Identical latency distribution across double the duration — **no p99 drift, no
memory creep, zero rejections**. Legitimate steady traffic is completely
unaffected by the limits, with ~7× headroom under the 1s SLA.

> **Environment note.** §4.1 was measured on a contended dev laptop; §4.2–4.4 on
> the EC2 staging box. Absolute rps is machine-specific (CPU-bound), so the
> *shape* (knee ≈ 16 aggregate, cliff past it) is the transferable finding — the
> `--limit-concurrency 8` cap is tied to that shape, not to raw rps. Re-run the
> sweep on any new instance before trusting the absolute numbers.

---

## 5. How to re-measure / tune

Scripts (in `scripts/`, run from a sibling container so the generator doesn't
steal serve's CPU):

- **Find the concurrency knee** (direct to serve, bypass nginx):
  ```
  concurrency_sweep.py --url http://serve:8000 --key <KEY> \
      --concurrency 8,16,32,64,128 --duration 30 --warmup-secs 5
  ```
  Knee = highest C where p99 ≤ ~550ms and throughput is still rising. Set
  `SERVE_LIMIT_CONCURRENCY = knee ÷ SERVE_WORKERS`.

- **Verify the limits reject overload** (429 via nginx, 503 direct to serve):
  ```
  verify_limits.py --url http://nginx     --key <KEY> --concurrency 64 --total 6000   # expect 429
  verify_limits.py --url http://serve:8000 --key <KEY> --concurrency 64 --total 6000   # expect 503
  ```

- **Sustained soak at a chosen rate** (keep `--rpm` under the per-IP cap for a
  pure service test):
  ```
  verify_limits.py --url http://nginx --key <KEY> --rpm 2400 --duration 600
  ```

Applying changes:

- Concurrency: set `SERVE_LIMIT_CONCURRENCY` in `.env`, then
  `docker compose ... up -d --force-recreate serve` (no rebuild).
- nginx rate: edit `rate=` in `docker/nginx/nginx.conf`, then
  `docker compose ... up -d --force-recreate nginx` (config is a mounted volume).

Clean up test rows afterward:
```
DELETE FROM smarthub_prediction_log WHERE lead_ping_id = 2000000000;
```

---

## 6. Open items before production

1. **Set the real nginx rates.** Replace the placeholder `perip 50 r/s` /
   `perkey 100 r/s` with the client's agreed peak rps (per-IP ≥ per-key). A
   single high-volume client from one IP will otherwise hit the 50 r/s wall.
2. **Open the security group** on the external port for the client's IP — the
   soak validated the in-box path only; external callers reach nginx over the
   public interface.
3. **Re-run the sweep on the production instance** if it differs from staging;
   adjust `SERVE_LIMIT_CONCURRENCY` and the nginx `rate=` to that box's capacity.
4. **Document 429/503 for the client** so their integration retries with backoff
   (429 → honor `Retry-After`; 503 → retry with jitter).

## 7. The honest caveat

These limits are **blast-radius control, not more capacity**. They make the
service fail gracefully at its ceiling (~110–133 rps on the tested boxes); they do
not raise it. If legitimate aggregate demand exceeds capacity, the fix is
**horizontal scale** (more workers/replicas), with these limits protecting the
service until then.
