# Data-Pull & Data-Validation Audit

Scope: `src/smarthub/data_pull/*` (pull, flow, models, query_builder, windowing,
validation_*) and the persist layer `src/smarthub/core/storage.py`.
Focus: loopholes and non-production-grade behaviour. Nothing below is fixed yet —
this is a findings list with concrete remediations.

The validation *rule coverage* is genuinely strong (schema drift, ranges,
domains, cross-field integrity, missingness, batch metrics). The gaps are mostly
in **orchestration, atomicity, and gating**, not in the checks themselves.

---

## CRITICAL — broken or data-loss

### C1. The scheduled data-pull flow is broken (wrong kwarg)
`data_pull/flow.py` `fetch()` calls:

```python
fetch_leads(..., lead_type_id=lead_type_id)
```

but `data_pull/pull.py::fetch_leads` accepts `lead_type_ids` (plural). Verified
by AST:

- `fetch_leads` params: `settings, min_created_at, max_created_at, with_expected_revenue, selected_only, lead_type_ids`
- `flow.py` call kwargs: `with_expected_revenue, selected_only, lead_type_id`

Every Prefect run raises `TypeError: fetch_leads() got an unexpected keyword
argument 'lead_type_id'`, retries twice (`@task(retries=2)`), then fails and
fires the Slack failure hook. The scheduled pull has never succeeded through the
flow; only the CLI path (`smarthub-pull` → `run()`, which uses `lead_type_ids`)
works. **Fix:** rename the kwarg to `lead_type_ids=[lead_type_id]` (and pass a
list, since the plural API expects a sequence). Add a smoke test that calls the
flow with a stubbed `fetch_leads` so signature drift fails CI.

### C2. Non-atomic DuckDB upsert → data loss on mid-upsert failure
`core/storage.py::append_duckdb` does a delete-then-insert as two separately
auto-committed statements:

```python
con.execute(f'DELETE FROM "{table}" WHERE {key} IN (SELECT {key} FROM incoming)')
con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM incoming')
```

If the process dies or the INSERT raises (type mismatch, disk full) between the
two, the overlapping window's existing rows are **already deleted** and never
re-inserted — permanent loss of previously-good data. **Fix:** wrap both in
`BEGIN TRANSACTION; ... COMMIT;` (rollback on error), or use a single
`INSERT ... ON CONFLICT DO UPDATE` / `MERGE`.

### C3. Non-atomic Parquet partition rewrite → corruption on crash
`core/storage.py::append_parquet` rewrites the whole day-file in place:

```python
combined = pd.concat([pd.read_parquet(target), group])
_dedupe(combined, key).to_parquet(target, index=False)   # overwrites in place
```

A crash mid-write leaves a truncated/corrupt parquet for the entire day, and two
concurrent writers to the same partition lose each other's rows (last-writer
wins). **Fix:** write to a temp file in the same dir and `os.replace()` (atomic
rename); serialize writers per partition, or add a per-partition lock.

---

## HIGH — production robustness

### H1. Validation never gates persistence
`flow.py` computes `quality = validate(...)` but only uses it for the Slack
message; `persist(df)` then runs unconditionally, and the watermark advances.
An empty pull, a schema-drifted payload, or a critical column that is 100% null
is stored and the watermark moves past it — silently poisoning downstream
training. **Fix:** add a circuit breaker for catastrophic conditions (zero rows,
`schema_drift` missing-columns, row-count collapse vs the previous pull,
critical-field all-null). Keep the current warn-only behaviour for soft issues,
but *fail the flow before persist* on hard ones.

### H2. Watermark can permanently skip late-arriving rows
Windowing anchors on `created_at` with a small fixed overlap
(`overlap_hours` default 1.0) and advances the watermark to `max(created_at)`
pulled. Rows that are *inserted* into the warehouse after the pull but carry a
`created_at` older than `watermark − overlap` are never picked up again → silent
gaps. Overlap only rescues late *updates* to already-seen ids (via upsert), not
late *inserts*. **Fix:** watermark on an ingestion/`updated_at` column, or size
the overlap to the real max insert lag, and add a periodic wide reconciliation
pull.

### H3. Whole result loaded into memory + unbounded window
`pull.py` uses `pd.read_sql(stmt, conn)` with no `chunksize`, and `windowing`
has no max-window cap. After any outage the backfill (`default_lookback_hours`
= 168h) or a long catch-up produces one giant query/frame → OOM risk on the
box. **Fix:** cap the window (chunk large ranges into sub-windows) and/or stream
with `chunksize`, writing incrementally.

### H4. No statement/query timeout
Only `connect_timeout` is set on the Redshift connection. A hung or runaway
query blocks the task forever; `@task(retries=…)` doesn't help a hang (nothing
raises). **Fix:** set a Redshift `statement_timeout` and a Prefect task
`timeout_seconds`.

### H5. Cross-backend partial writes leave DuckDB and Parquet divergent
`save_pull` writes DuckDB then Parquet with no coordination. If one succeeds and
the other throws, the two backends disagree and there's no reconciliation or
rollback; downstream reads "prefer DuckDB, then Parquet," so which one wins is
silent. **Fix:** treat the pair transactionally (stage both, commit both, or
neither), or pick a single source of truth and derive the other.

---

## MEDIUM

- **M1. `persist` has no retries** while `fetch` does — a transient FS/S3 blip
  fails the whole flow after a successful (expensive) fetch. Add
  `@task(retries=…)` to `persist` (safe: the upsert is idempotent).
- **M2. Silent dtype coercion** (`errors="coerce"`) in `coerce_leads_dtypes`
  and throughout the rules turns bad upstream values into NaN/NaT with no count
  or telemetry. Validation then sees them as "missing" rather than "invalid,"
  masking real corruption. Log coercion counts; validate before coercing where
  feasible.
- **M3. Parquet silently drops undated rows** (`append_parquet` warns and skips
  rows with no resolvable partition date) — those rows still land in DuckDB, so
  the backends diverge and rows are lost from Parquet. Route them to a
  quarantine partition instead of dropping.
- **M4. Zero-row pull looks healthy** — reported as success with no anomaly
  alert and no volume-regression check against prior pulls.
- **M5. Fragile timezone assumption** — code assumes warehouse `created_at` is
  naive UTC (`_utc_now_naive`), yet the schema carries `pst_date`/`pst_hour`.
  A tz mismatch silently shifts every window. Assert/measure the warehouse tz.
- **M6. CLI pull has no retry/backoff** — only the Prefect task retries;
  `run()` fails outright on a transient tunnel/Redshift error.

## LOW / polish

- **L1.** `_as_datetime`/`parse_dt` use a strict `%Y-%m-%d %H:%M:%S`; a
  date-only CLI arg raises an opaque `ValueError`.
- **L2.** No guard that `max_created_at > min_created_at`; an inverted manual
  window silently returns zero rows.
- **L3.** `expected_revenue_subquery` labels `func.count()` as
  `num_selected_listings` even when `selected_only=False` (misleading count).
- **L4.** Watermark Prefect Variable has no lock; overlapping runs for the same
  lead type can interleave windows/watermark (idempotent upsert limits the blast
  radius, but it's still a race).

---

## Suggested fix order

1. **C1** (flow kwarg) — one-line fix + signature-drift test; unblocks the
   scheduled pull.
2. **C2 + C3** (atomic DuckDB txn, atomic parquet rename) — stop silent data
   loss.
3. **H1** (gate persist on catastrophic validation) — stop poisoning training.
4. **H2/H3/H4** (watermark lag, window cap/streaming, query timeout).
5. Medium/low as capacity allows.
