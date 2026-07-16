    # SmartHub — Data Validation Rules

The rules the data-validation layer (`smarthub/validation`) applies to each
freshly-pulled `lead_pings` batch. **Warn + report only** — every rule *detects
and reports*; nothing drops, imputes, caps, or blocks the pull. Runs on the raw
batch at pull time (both the Prefect flow and the `smarthub-pull` CLI), upstream
of any feature-engineering cleaning.

Source of truth: `src/smarthub/validation/rules.py`. This doc mirrors it.

---

## 1. Schema-level checks

| Rule | Column(s) | Condition (flag when…) | Layer |
|---|---|---|---|
| Schema drift — missing | all expected | an expected pulled column is absent | pandas |
| Schema drift — unexpected | all present | a column appears that isn't expected | pandas |
| Uniqueness | `id` | duplicate `id` values in the batch | pandera |
| Not null | `id` | `id` is null | pandera |

Expected columns = the pull's `LEADS_COLUMNS` (`data_pull/models.py`).

## 2. Numeric range checks (pandera)

Flagged when a present, coercible value falls **outside** the inclusive range.

| Column | Min | Max |
|---|---|---|
| `age` | 1 | 200 |
| `num_vehicles` | 0 | 12 |
| `num_drivers` | 0 | 6 |
| `num_auto_violations` | 0 | 20 |
| `num_auto_accidents` | 0 | 20 |
| `num_auto_claims` | 0 | 20 |
| `num_home_claims` | 0 | 20 |
| `num_dependents` | 0 | 15 |
| `continuous_coverage_months` | 0 | 600 |

## 3. Non-negative checks (pandera)

Flagged when a present value is `< 0`.

| Column | Rule |
|---|---|
| `bid` | ≥ 0 |
| `exp_rev` | ≥ 0 |
| `rev` | ≥ 0 |
| `total_listings` | ≥ 0 |
| `accepted_listings` | ≥ 0 |
| `household_income` | ≥ 0 |
| `annual_revenue` | ≥ 0 |
| `life_coverage_amount` | ≥ 0 |
| `num_employees` | ≥ 0 |
| `response_ms` | ≥ 0 |

## 4. Categorical domain checks (pandera)

Flagged when a present value isn't in the allowed set.

| Column | Allowed values |
|---|---|
| `state` | US 2-letter: 50 states + `DC`, `PR`, `GU`, `VI`, `AS`, `MP` |
| `gender` | `Male`, `Female`, `Non-binary` |
| `marital_status` | `Single`, `Married`, `Divorced`, `Widowed` |

## 5. Cross-field integrity checks (pandas)

Counts of rows violating each rule (detect-only). Runs only when the involved
columns are present.

| Rule name | Condition (flag when…) | Why it matters |
|---|---|---|
| `current_carrier_when_not_insured` | `current_carrier` populated **and** `insured` ∈ {false, f, 0, no, n} | Kiran: bad data; `current_carrier` is critical for bidding |
| `won_true_without_bid` | `won` == `true` **and** `bid` ≤ 0 | A win with no bid placed — auction-logic inconsistency |
| `erred_with_bid` | `erred` is true-ish **and** `bid` > 0 | An errored ping shouldn't have reached a bid |
| `accepted_but_won_null` | `accepted` is true-ish **and** `won` is blank | Accepted but no recorded win |
| `auto_missing_num_vehicles` | `lead_type_id` == 6 (auto) **and** `num_vehicles` null/blank | Auto lead missing a core field |
| `home_missing_property_type` | `lead_type_id` == 1 (home) **and** `home_property_type` null/blank | Home lead missing a core field |

True-ish tokens: `true`, `t`, `1`, `yes`, `y`. False-ish: `false`, `f`, `0`,
`no`, `n`.

## 6. Missing-value catalogue (pandas)

| Metric | Definition |
|---|---|
| per-column missing rate | share of rows that are null **or** blank/whitespace-only (0–1) |
| `high_missing` list | columns whose missing rate ≥ `high_missing_threshold` |

`high_missing_threshold` default `0.5`, set in
`config/smarthub.yaml (validation section)`.

## 7. Batch quality metrics (pandas)

Reported every run (headline health, not pass/fail).

| Metric | Definition | Healthy signal |
|---|---|---|
| `rows` | rows in the batch | — |
| `erred_rate` | share with `erred` true-ish | low |
| `bid_zero_rate` | share with `bid` ≤ 0 (no-bid) | context-dependent |
| `exp_rev_coverage` | share with `exp_rev` > 0 | high |
| `pst_hour_populated` | share with `pst_hour` present | 100% |
| `age_implausible_rate` | share with `age` outside 1–200 | ~0% |
| `won_false_count` | rows where `won` == `false` | **0** (warehouse never writes `false`) |
| `traffic_tier_distinct` | distinct `traffic_tier` values | stable vs baseline |

## 8. Behavior & wiring

| Aspect | Value |
|---|---|
| Mode | warn + report only — never drops/fixes/blocks |
| Runs at | data-pull, on the fetched batch (flow **and** `smarthub-pull` CLI) |
| Outputs | `data-quality-<lead_type>` Prefect artifact · "Data quality" Slack group · CLI log summary |
| Tooling | pandera (schema/range/domain) + pandas (cross-field, missing, metrics) |
| Degrades | if pandera absent: schema/range/domain skipped (warning), pandas checks still run |
| Config | `config/smarthub.yaml validation.high_missing_threshold` |

## 9. Known limitations / to revisit

| Item | Note |
|---|---|
| `household_income` numeric check | Treated as non-negative numeric, but the warehouse value is likely an income *band/string* — this can flag ~100% of rows (a rule mismatch, not bad data). Reclassify as categorical. |
| Boolean-domain checks | `BOOLISH_COLUMNS` / `BOOL_TOKENS` are defined but **not yet enforced** as a rule (only used implicitly inside cross-field checks). |
| High-missing noise | For a single-lead-type pull, other products' columns (home/commercial/life) show as high-missing though they're expected-empty. Consider scoping the catalogue per lead type. |
