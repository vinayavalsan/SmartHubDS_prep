# Changelog


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

### Quality & cleanup
- Unit tests throughout (41 total: transforms, config, ORM SQL, storage);
  flake8 clean; added a `Dockerfile`.
- Reset `.env.example` to placeholders; removed all legacy/dead files
  (`prep/`, `src/monitoring/`, empty `src/utils/` stubs, empty data placeholders).

