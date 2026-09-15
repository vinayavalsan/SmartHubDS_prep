# SmartHub bid-API replay / load test harness (`tests/sim`)

A controlled substitute for shadow deployment: replay **real historical
`lead_pings`** at `/recommend_bid` to prove the serving path end-to-end and hold
it under production-like load — for a full week — without touching production.

Everything runs where `smarthub` + pandas live (the **worker container** on EC2),
or anywhere with `pandas`, `pyarrow`, `requests`. Nothing here writes to production;
point it at a **stub** (safe) or a **staging** endpoint.

## Pieces

| File | Piece | What it does |
|------|-------|--------------|
| `extract_snapshot.py` | 1 | Bulk-pull months of `lead_pings` → one Parquet snapshot (chunked, low-memory). |
| `payload_builder.py`  | 2 | One snapshot row → a faithful `BidRequest` body (point-in-time; `created_at` verbatim). |
| `arrivals.py`         | 3 | Bursty arrival schedule (Poisson scatter + injectable bursts). |
| `sender.py`           | 4 | `requests` + thread-pool POST (open-loop) + per-request capture ledger (JSONL). |
| `stub_server.py`      | 5 | Dependency-free fake `/recommend_bid` — the safe first target. |
| `run.py`              | — | Orchestrator: snapshot → schedule → fire → ledger (+ summary). |
| `monitors.py`         | 6 | Host / container / Postgres health → CSV, for week-long creep. |
| `supervisor.py`       | 7 | Week-long runner: day segments, resume, restart, Slack heartbeat, daily rollup. |
| `analyze.py`          | 8 | Ledger → latency / errors / decision-path + parity backtest + PASS/FAIL. |
| `phase1_replayer.py`  | — | Standalone arrival-distribution verifier (`--verify`), no data/HTTP needed. |

## Quick start (safe, local — against the stub)

```bash
# 1) snapshot already produced by extract_snapshot.py, e.g. data/sim/snapshot.parquet

# 2) start the safe stub
python stub_server.py --port 8080 --latency-ms 5 20 &

# 3) preview (sends nothing) — payload sample + arrival shape
python run.py --data data/sim/snapshot.parquet --minutes 6 --burst-prob 0.6

# 4) fire for real at the stub (compress time 60x for a quick check)
python run.py --data data/sim/snapshot.parquet --url http://127.0.0.1:8080 \
    --minutes 6 --burst-prob 0.6 --dispatch --speed 60 --out data/sim/ledger.jsonl

# 5) analyse (with parity backtest against historical outcomes)
python analyze.py data/sim/ledger.jsonl --data data/sim/snapshot.parquet
```

Green run = all 200s, p99 < 1s, realistic decision-path mix, invalid rows skipped.

## Week-long run (supervised)

```bash
# monitors alongside (own process)
python monitors.py --interval 30 --out data/sim/health.csv \
    --serve-container prefect-serve --pg-container prefect-postgres --disk-path /app/data &

# the supervised replay (4 historical days). SLACK_WEBHOOK env -> heartbeats.
nohup python supervisor.py --data data/sim/snapshot.parquet \
    --url http://STAGING:8000 --days 4 --burst-prob 0.6 \
    --ledger-dir data/sim/ledgers > data/sim/supervisor.log 2>&1 &
```

Put `supervisor.py` under systemd (`Restart=always`) so a reboot resumes it; the
checkpoint (`data/sim/ledgers/supervisor.ckpt.json`) makes it pick up the right day.
Add `--segment-minutes 6 --speed 60` to rehearse a "day" in seconds.

## Safety

- **Never point `--url` at production.** Use the stub, then a staging serve with its
  own throwaway DB. All replays carry real `lead_ping_id`s and would pollute the
  production monitoring dataset.
- Snapshots + ledgers live under `data/` (git-ignored). They contain real lead
  attributes (PII) — keep access-controlled and purge after the test window.

## Notes on fidelity

- `created_at` is sent **verbatim** from the snapshot (no tz conversion): the model
  reads it as UTC and converts to Pacific internally, so sending the raw stored value
  reproduces the exact timing features training saw.
- Only pre-bid inputs are sent; outcome columns (`bid`,`won`,`rev`,…) are held back
  and used only by the analyzer for the parity backtest.
- The field set is the documented `BidRequest` contract; only `state` is a mandatory
  attribute, everything else is optional (the model imputes when absent).

## Next (piece 9, later)
Live tail: swap the snapshot source for a watermark poll of new `lead_pings`
(read-only, no bids) for the final realism days — requires sign-off.
