# Anton — Modeling Spec

The plan for turning the pulled `lead_pings` data into Anton's bid decisions.
Read alongside [CONTEXT.md](./CONTEXT.md) (business domain) — this doc is the
spec feature engineering and training execute against.

---

## 1. Problem framing

Anton does **not** predict the bid directly. It predicts the **probability of
winning a lead at a given bid**, then chooses the bid that maximizes expected
profit within the allowed bounds.

```text
P_win = f(X, bid)                      # the model
expected_profit(bid) = P_win(X, bid) × (R − bid)
recommended_bid = argmax over bid in [floor, ceiling] of expected_profit(bid)
```

- `X` = pre-bid lead/context features (below).
- `bid` = the decision variable (a model input, swept to find the optimum).
- `R` = expected revenue if we win = top-buyer `est_payout` (see §4).
- `ceiling = R × (1 − CM_target)`, `floor` = partner minimum if one exists.

**Unit of observation:** one ping. **Target:** `won` (1/0).

---

## 2. Feature set (model inputs `X`)

All of these are known **at bid time** (the buyer-matching step runs before we
bid, so listing-derived demand signals are legitimate pre-bid features).

### Consumer / lead attributes
`age`, `num_vehicles`, `num_drivers`, `num_auto_violations`, `num_auto_claims`,
`num_auto_accidents`, `insured`, `continuous_coverage_months`,
`current_carrier`, `dui`, `sr22_required`, `home_owner`, `military_affiliation`,
`gender`, `marital_status`, `credit`, `household_income`, `pnc_bundle`.

### Context / market
`state`, `zip` (high-cardinality → bucket / roll up to region), `lead_type_id`,
`campaign_id`, `account_id` (partner), `source_type_id`, `traffic_tier`,
`device_type`.

### Demand signals (pre-bid, likely strong predictors)
`total_listings` (interested buyers = competition/demand),
`num_selected_listings`, `expected_revenue` (top-buyer `est_payout`).

### Time-derived (from `created_at`)
hour-of-day, day-of-week.

### Decision variable
`bid` — the model is a function of the bid.

---

## 3. Column classification (leakage map)

| Role | Columns |
|---|---|
| **Feature (X)** | the §2 lists above |
| **Decision variable** | `bid` |
| **Objective inputs (not predictors)** | `expected_revenue` (R), CM target (not in data yet) |
| **Target / label** | `won` |
| **EXCLUDE — post-bid outcomes (leakage)** | `rev`, `realized_payout`, `accepted` (= resold to ≥1 buyer), `accepted_listings`, `erred`, `error_reason_id`, `response_ms` |
| **Use with care** | `bidding_strategy_id` (our choice; collinear with `bid` — keep out of the win model, use for analysis only) |
| **IDs / meta (not features)** | `id`, `updated_at`, `pst_date`, `expiration_date`, `lead_created_at` |
| **PII (never pulled / never features)** | `token`, `uid`, `ip_address`, `user_agent`, `jornaya_lead_id`, `trusted_form_token`, `submission_url`, `date_of_birth` (use `age` instead) |

> Rationale: at serve time, when a ping arrives, we only know pre-bid context +
> a candidate bid + the buyer-matching results (listings, expected revenue).
> Anything that only exists *after* we bid and win must not be a feature.

### Availability timeline (the golden rule)

**Only use columns whose value is known the moment the ping arrives.** The test
for any column is: *"Would I know this value at the instant I place the bid?"*
If no, it cannot be a feature.

| When it becomes known | Columns | Use? |
|---|---|---|
| **① At ping arrival** (consumer attributes + context + buyer-match results) | `state`, `zip`, `city`, `age`, `gender`, `marital_status`, `num_vehicles`, `num_drivers`, `num_auto_violations`, `num_auto_claims`, `num_auto_accidents`, `insured`, `current_carrier`, `continuous_coverage_months`, `dui`, `sr22_required`, `home_owner`, `military_affiliation`, `credit`, `household_income`, `pnc_bundle`, `device_type`, `traffic_tier`, `lead_type_id`, `campaign_id`, `account_id`, `source_type_id`, `created_at`, **`total_listings`**, **`num_selected_listings`**, **`expected_revenue`** | ✅ **Features** |
| **② At the bid** (our decision) | `bid` | ✅ Decision variable |
| **③ Only after the outcome** (results of the bid) | `won` (**target**), `accepted`, `accepted_listings`, `rev`, `realized_payout`, `erred`, `error_reason_id`, `response_ms` | ❌ **Never a feature** (leakage) |

So features come from group ①, the model sweeps the group ② `bid`, and group ③
is used only as the label (`won`) or for after-the-fact profit analysis.

> ⚠️ Confirm with the team that `total_listings`, `num_selected_listings`, and
> `expected_revenue` are populated **at bid time** (not back-filled later). The
> flow implies buyer-matching runs before bidding, but it's worth verifying since
> they'll be strong features.

---

## 4. Open items that gate modeling

1. **`won` semantics** — *data-supported* (still worth a final nod from Kiran):
   `won = true` means the **partner gave us the lead**; `accepted = true` means we
   **resold it to ≥1 buyer**. See §6. `won` is the target.
2. **Expected-revenue aggregation** — **likely `SUM(est_payout)` over
   `selected = true` listings, NOT `MAX`.** The strategy doc says "top buyer"
   (which would be `MAX`), but that's only right for *exclusive* leads — and the
   data shows leads are mostly **non-exclusive** (`exclusive = false`,
   `accepted_listings` 2–4), i.e. sold to multiple buyers, so revenue is a sum.
   See §6. Make the aggregation configurable (sum-selected / max / sum-all),
   default to **sum-over-selected**, and validate against realized `rev` once
   real data accumulates. Pending Kiran/Vinaya confirmation.
3. **CM target source** — `b = R × (1 − CM_target)` needs the CM target, tied to
   `bidding_strategy_id` (dumb 10/25/50/75). Need a
   `bidding_strategy_id → CM_target` lookup or config value.
4. **Data volume** — ~276 rows so far. Keep the scheduled pull running over
   overlapping windows to accumulate ~3–4 weeks before training.
5. **Lead type scope** — home/commercial/life columns are all-null in the auto
   sample; start with an **auto-only** model, expand per lead type later.

---

## 5. Data-quality / EDA checklist (before feature engineering)

- Distributions & outliers (esp. `bid`, `expected_revenue`).
- Missingness per column; drop all-null / mostly-null columns per lead type.
- Cardinality (e.g. `zip` ~ thousands → bucket; `state` ~ 50 → fine).
- Win-rate balance (class imbalance handling for the target).
- Per-`account_id` / per-`lead_type_id` volume (enough rows per segment?).
- Time-zone consistency: `created_at` is **UTC**, `pst_date` is Pacific — pick
  one clock for time features and the daily Parquet buckets.
- Confirm which features actually carry signal (keep a handful; avoid noise).

### Concrete cleaning rules (confirmed from inspection — see §6)
- **Keep only real bidding decisions:** `won` is non-blank (`won <> ''`) — this
  keeps **both** `won = 'true'` (wins) **and** `won = 'false'` (losses), which the
  model needs to learn the boundary. Only blank `won` (no bid / no outcome) is
  dropped. Do **not** keep wins only.
- **Drop zero-variance features (for now):** in current data `insured`,
  `military_affiliation`, and `dui` have a single value (`false`) → no signal,
  drop them. Re-check as data grows (they may just be unpopulated today).
- **Auto-only first model:** `lead_type_id = 6`.
- **Drop test/seed rows:** private IPs (`192.168.*`, `10.*`), placeholder dates
  (e.g. expired `expiration_date` like `2023-01-01`), and `rev == bid` round
  numbers from the seed set.
- **Unreliable columns:** `expiration_date` is defaulted (`2027-06-08` on most
  rows) and `lead_created_at` is all-null → don't use.
- **Don't double-count source:** `traffic_tier` overlaps `account_id` /
  `source_type_id` — pick one, don't feed all three as independent.

---

## 6. Findings from data inspection (2026-06-25)

From a sample of `won = true` rows (early June data):

- **`won` vs `accepted`:** `won = true` = partner gave us the lead;
  `accepted = true` = we resold to ≥1 buyer. Whenever `accepted_listings = 0`,
  `accepted = false` and `rev = 0` ("won but resold to nobody" — a wash).
- **This early data is mostly money-losing:** on won+resold rows, `bid` (what we
  pay the partner) is almost always **higher** than `rev` (what buyers pay us)
  — the early over-bidding phase Kiran described. Not representative; don't judge
  profitability from it.
- **`bidding_strategy_id` is empty on every row** → the CM target is not recorded
  in the data; it must come from config / Kiran (confirms §4.3).
- **Lead-type mapping:** `lead_type_id = 6` → auto; `lead_type_id = 1` → home
  (those rows have `home_property_type` set and blank driver/vehicle counts).
- **Low feature variance in this slice:** `insured`, `credit`, `military_affiliation`,
  `num_drivers`, `num_vehicles`, `dui`, violations are nearly constant here →
  little signal yet; reinforces the need for more, more-varied data.
- **`total_listings` (pre-bid demand) vs `accepted_listings` (post-bid outcome)**
  confirmed: `accepted_listings ≤ total_listings`.

### Listings table (`lead_ping_listings`) inspection
- **Leads are mostly non-exclusive** — `exclusive` is `false` on almost all
  listings (only a couple `true`). Combined with `accepted_listings` 2–4, this
  means a ping is sold to **multiple buyers**, so expected revenue `R` is a
  **SUM across the buyers we sell to (`selected = true`)**, not `MAX`.
- **`est_payout` exactly equals `payout` on every row** in current data → no
  visible reject-discount (contradicts the "expected = reject-discounted"
  description). Almost certainly **seed/synthetic data**; can't validate the R
  rule empirically until real data accumulates.
- **`selected = true`** appears to mark the buyers we actually transact with →
  the set to sum `est_payout` over (confirm it's known at bid time).
- **`bpfm_score`** is populated on some listings (~40–75) and `0.00` on many —
  a per-buyer score of unknown meaning; pending definition.

---

## 7. Bid-strategy roadmap (from the strategy doc)

- **v1.0 — Exploratory bid** (no model): `b = R × (1 − CM_target)` plus Gaussian
  noise (σ chosen so ~95% of bids fall within ±x% of base) and `[floor, ceiling]`
  bounds. Purpose: collect data across price points.
- **v1.1 — Category win-rates** (no ML): rolling-window win rate per
  `lead_type_id × campaign_id × state`; correct the base bid toward a target win
  rate, with confidence shrinkage / hierarchical fallback for sparse categories.
  All precomputed into lookup tables (the "training" step).
- **v2.0 — ML win-probability**: train `P_win = f(X, bid)` (Logistic Regression
  baseline → XGBoost / LightGBM), then bid = argmax expected profit. Tree models
  preferred for nonlinear / threshold / segment-specific bid sensitivity.

### Training row for v2.0
`(X, bid, won_flag, R)` — one row per bidding decision, all constructable from
the store once the leakage split (§3) and §4 items are settled.

---

*Pairs with CONTEXT.md. Update §4 as questions are answered.*
