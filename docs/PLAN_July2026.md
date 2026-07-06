# SmartHub Data Science — July 2026 Plan & Deliverables

**Team:** Nimesh & Vinaya   **Period:** 1–31 July 2026
**Status:** tentative — reconciled with Vinaya's Q3 MVP plan (Confluence).
**Starting point:** the Redshift **data pull** exists; everything else is new.

> **Scope note (reconciliation):** July builds the *platform foundations* +
> *staging environment* + a **placeholder → preliminary model** to wire up the
> prediction path. The **real, tuned model is August** — July's models exist to
> get the API/staging working end-to-end on production data (predictions logged,
> **no bids placed**).

---

## 1. July goal

Establish platform foundations and a **staging environment**, and stand up a
preliminary prediction path, so we can **generate bid predictions on production
data while keeping all outputs isolated in staging** (logged, not acted on).

MVP end-state this contributes to: extract/prepare data → train & version models
→ serve real-time bid recommendations via an API → log every decision →
monitoring dashboards → safe staging/prod environments.

## 2. Ownership areas (from the MVP plan)

| Nimesh (data eng / infra) | Vinaya (modeling / analysis) |
|---|---|
| Git repo structure, data architecture | Feature engineering |
| Infrastructure, ETL pipeline | Model training & evaluation |
| Prediction API, database design | MLflow / model tracking |
| Logging services | Monitoring dashboards |
| Staging / prod environments | Model deployment (staging) |

Both participate in design discussions and keep **documentation** current.

## 3. Deliverables at a glance

| # | Deliverable | Owner | Target (Tue) |
|---|---|---|---|
| D1 | **ETL framework** — automated pull + data **validation** (identify bad rows & missing-value patterns; don't fix yet) + accumulating storage; repo/data architecture | Nimesh | Tue 14 Jul |
| D2 | Confirmed data **semantics** (target, expected revenue, CM target) | Both | Tue 7 Jul |
| D3 | **Feature engineering + training/bid-optimization framework** (local train/test/compare) + **MLflow** tracking | Vinaya | Tue 14 Jul |
| D4 | **Prediction API** (FastAPI) + prediction **logging schema**/service + placeholder model & bid-optimization stub | Nimesh (input from V) | Tue 21 Jul |
| D5 | **Preliminary model** for the prediction infrastructure (for staging) | Vinaya | Tue 28 Jul |
| D6 | **Dashboard hosting** — host the Streamlit (training-data) dashboard, accessible to all users | Nimesh | Tue 21 Jul |
| D7 | **Staging environment** + SmartHub-platform integration approach (predict on real data, **no bids placed**); prod-env plan | Nimesh | Tue 28 Jul |
| D8 | **Documentation** + two-week sprint cadence maintained | Both | Thu 31 Jul |

---

## 4. Weekly plan (Tuesday → Tuesday · two 2-week sprints)

Reviews fall on the Tuesday DS check-ins (**7, 14, 21, 28 Jul**). Sprint 1 =
Jul 1–14, Sprint 2 = Jul 15–28, wrap 29–31.

| Sprint | Week | Dates (→ Tue review) | Focus | Owner | Deliverables |
|---|---|---|---|---|---|
| **S1** | W1 | Wed 1 → **Tue 7** | Sprint planning; **confirm semantics**; ETL framework + repo/data architecture start; scope to **auto** | Both (semantics), Nimesh (ETL/repo) | D2 |
| **S1** | W2 | Wed 8 → **Tue 14** | ETL validation (bad-row / missing patterns) + accumulating storage; feature-engineering + training/bid-opt framework scaffold + MLflow | Nimesh (ETL), Vinaya (features/training/MLflow) | D1, D3 |
| **S2** | W3 | Wed 15 → **Tue 21** | Prediction API (FastAPI) + logging schema/service + placeholder model & bid-opt stub; **host dashboards** | Nimesh (API/logging/hosting), Vinaya (bid-opt input) | D4, D6 |
| **S2** | W4 | Wed 22 → **Tue 28** | Preliminary model; **staging environment** + platform-integration approach (predict on real data, no bids); local model compare | Vinaya (model), Nimesh (staging/integration) | D5, D7 |
| — | W5 | Wed 29 → **Thu 31** | Documentation; month-end review; prep August staging deploy | Both | D8 |

Throughout: **data accumulates** (target ~3–4 weeks of clean auto data) and both
own their own tasks within each two-week sprint.

---

## 5. Infrastructure to finalise in July

| Component | What we finalise |
|---|---|
| **ETL framework** | Automated pull (scheduled, overlapping window) + **validation** (flag bad rows, catalogue missing-value patterns — fixes discussed, not auto-applied yet) + de-duped accumulating storage |
| **Orchestration** | Prefect 3 (self-hosted, Postgres-backed); `smarthub-pool` + queues (`default` pulls, `features` builds); per-lead-type watermarks; run artifacts |
| **Storage & data architecture** | DuckDB + partitioned Parquet; versioned training tables + lineage; repo structure & data architecture |
| **Prediction API** | **FastAPI** service for bid-prediction requests; placeholder model + bid-optimization algorithm; **logging schema** for bid values/details (what to log agreed) |
| **Training & tracking** | ML training + bid-optimization framework (local train/test/compare); **MLflow** for model tracking/versioning |
| **Dashboards & hosting** | Streamlit (training-data) dashboard **hosted & accessible**; second real-time pipeline-monitoring dashboard = future |
| **Environments** | **Staging** for the prediction API on real data (no bids placed); production environment plan + SmartHub-platform integration |
| **Containerisation** | Docker + Compose (`restart: unless-stopped`); `install.sh` (validation + `--down` port cleanup) |
| **Config / secrets / logging** | Validated env config; **logging services** + DB; secrets hardening (rotate, move to secret store) |
| **Quality gates** | pytest, flake8, pre-commit, CI (GitHub Actions) |

## 6. Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Source DB | Amazon Redshift (via SSH tunnel) |
| DB access | SQLAlchemy ORM + redshift-connector |
| Data handling | pandas, PyArrow |
| Storage | DuckDB + partitioned Parquet |
| Orchestration | Prefect 3 (self-hosted) |
| Metadata / logging DB | PostgreSQL |
| **Prediction API** | **FastAPI** (+ Uvicorn) |
| **Model tracking** | **MLflow** |
| Modeling (preliminary Jul; tuned Aug) | scikit-learn (LR), XGBoost / LightGBM |
| Dashboards | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Envs | local/Docker → **staging** → production |
| Quality / CI | pytest, flake8, pre-commit, GitHub Actions |

## 7. Open decisions to align on (team + Kiran/Vinaya)

These need a joint decision early — they shape the ETL, the training table, and
the model.

1. **Feature set from the pulled data** — *which columns become model features.*
   To agree together:
   - **Candidate features** (known at bid time): consumer attributes (`age`,
     `num_vehicles`, `num_drivers`, `num_auto_violations/claims/accidents`,
     `insured`, `current_carrier`, `continuous_coverage_months`, `home_owner`,
     `dui`, `credit`, `marital_status`, `gender`…), context (`state`, `zip`,
     `lead_type_id`, `campaign_id`, `account_id`, `source_type_id`,
     `traffic_tier`, `device_type`), demand signals (`total_listings`,
     `num_selected_listings`, `expected_revenue`), and time (hour, day-of-week).
   - **To decide:** which to keep vs drop; handling of **high-cardinality**
     (`zip` → region?), **zero-variance/constant** columns (e.g. `insured`,
     `dui` are single-valued in early data), **missing-value** patterns, and
     **binning** of nonlinear features (`age`).
   - **Exclude (leakage — not features):** `won` (target), `rev`, `accepted`,
     `accepted_listings`, `realized_payout`, `bidding_strategy_id`, plus PII.
2. **Target definition** — confirm `won` = "we won the lead from the partner"
   (vs `accepted`).
3. **Expected-revenue rule** — `MAX` top buyer vs **`SUM` over selected** buyers
   (exclusive vs shared leads). Currently `SUM` over `selected='true'`.
4. **CM-target source** — how CM target maps from `bidding_strategy_id`
   (not populated in current data).

## 8. Dependencies & assumptions

- **Kiran/Vinaya confirmations** (D2) gate the target, expected-revenue rule
  (MAX vs SUM), and CM-target source — needed Week 1.
- **SmartHub-platform integration** (D7) needs the platform/eng team to define
  how our staging API plugs into the live flow (predict-only, no bids).
- **Data volume** accumulates from current production bidding; model quality
  (Aug) scales with what we gather now.
- July models are **placeholder → preliminary**; the tuned model is August.
- Scope is **auto leads first**.

## 9. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Semantics unconfirmed → wrong target/revenue | Resolve W1 (D2); revenue rule as a one-line switch |
| Platform integration slips | Ship API + staging as a self-contained predict-only component; integrate when ready |
| Leakage into features | Explicit bid-time vs outcome split; time-based validation |
| Too little data for August model | Automate the pull W1 so collection starts day one |
| Scope creep (API + model + staging in one month) | Placeholder-first; two-week sprints; strict "no bids placed" boundary |


*Prepared 1 Jul 2026; reconciled with Vinaya's Q3 MVP plan. Tentative — adjust
ownership/sequencing as the team agrees.*
