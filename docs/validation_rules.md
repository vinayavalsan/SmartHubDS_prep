# SmartHub — Data Validation Rules

The rules the data-validation layer (in `smarthub/data_pull`, run right after
the pull) applies to each freshly-pulled `lead_pings` batch. **Warn + report
only** — every rule *detects and reports*; nothing drops, imputes, caps, or
blocks the pull. Runs on the raw batch at pull time (both the Prefect flow and
the `smarthub-pull` CLI), upstream of any feature-engineering cleaning.

**Registry-driven.** The per-column rules below (ranges, non-negativity,
categorical domains, `id` uniqueness) are **not** hand-listed in code any more —
they are derived from each raw field's `ValidationSpec` in the raw-field
registry, `src/smarthub/data_pull/field_registry.py`, the single source of truth
for which fields the pull extracts and how each is validated. `leads_schema()`
builds the pandera schema from that registry.

Source files: `data_pull/field_registry.py` (field + rule metadata),
`data_pull/validation_rules.py` (schema builder, cross-field, missing, metrics),
`data_pull/validation_runner.py` (orchestration + custom-rule execution),
`data_pull/validation_custom.py` (custom rules, e.g. `validate_us_state`),
`data_pull/validation_report.py` (rendering). This doc mirrors them.

---

## 1. Schema-level checks

| Rule | Column(s) | Condition (flag when…) | Layer |
|---|---|---|---|
| Schema drift — missing | all expected | an expected pulled column is absent | pandas |
| Schema drift — unexpected | all present | a column appears that isn't expected | pandas |
| Uniqueness | `id` | duplicate `id` values in the batch | pandera |
| Not null | `id` | `id` is null | pandera |

Expected columns come from the registry (`field_registry.field_names()`), which
also drives what the pull selects (`models.LEADS_COLUMNS` is derived from it).
`id` uniqueness / not-null come from that field's `ValidationSpec` (`unique=True`,
`allow_missing=False`).

## 2. Numeric range checks (pandera)

Flagged when a present, coercible value falls **outside** the inclusive range.
Ranges come from each field's `ValidationSpec` (`min_value` / `max_value`); the
schema builder emits an `in_range` check.

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

Flagged when a present value is `< 0`. These are registry fields with a
`min_value` of 0 and no `max_value`; the schema builder emits a `≥ 0` check.

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

## 4. Categorical domain checks

Flagged when a present value isn't in the allowed set.

| Column | Allowed values | Layer |
|---|---|---|
| `state` | US 2-letter: 50 states + `DC`, `PR`, `GU`, `VI`, `AS`, `MP` | custom rule |
| `gender` | `Male`, `Female`, `Non-binary` | pandera |
| `marital_status` | `Single`, `Married`, `Divorced`, `Widowed` | pandera |

`gender` / `marital_status` come from the field's `ValidationSpec.allowed_values`
(pandera `isin`). `state` needs the US-state list, expressed as a **custom rule**
(`validation_custom.validate_us_state`, referenced from `state`'s
`ValidationSpec.custom_rule`) rather than plain metadata — so it runs as pandas
and is checked **even when pandera isn't installed** (see §8).

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

The lead-type completeness rules (the last two rows) are generated from the
**feature** registry: `cross_field_checks` iterates
`feature_engineering.feature_registry.FEATURES` (over `all_lead_types()`) and
emits one `{type}_missing_{col}` rule per raw column that's mandatory for that
type — no `lead_type_id ==` branches in `validation_rules.py`.

> **Open item:** this per-type "required raw column" information now lives in
> *two* registries — the feature registry (used here) and the raw field
> registry's `lead_types` / `required`. Consolidating the boundary so it's
> defined once is pending a team decision.

True-ish tokens: `true`, `t`, `1`, `yes`, `y`. False-ish: `false`, `f`, `0`,
`no`, `n`.

## 6. Missing-value catalogue (pandas)

| Metric | Definition |
|---|---|
| per-column missing rate | share of rows that are null **or** blank/whitespace-only (0–1) |
| `high_missing` list | columns whose missing rate ≥ `high_missing_threshold` |

`high_missing_threshold` default `0.5`, set in
`config/smarthub.yaml (validation section)`.

**Scoped per lead type.** When the batch's lead type is known (the scheduled
per-type pulls pass it; an all-types manual pull doesn't), the `high_missing`
list excludes columns that don't apply to that type — so an auto pull no longer
flags empty `home_*` columns. Only the flagged `high_missing` list is scoped
(via `field_registry.columns_not_for_lead_type`); the full per-column missing
rates are unchanged.

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
| Tooling | pandera (schema/range/domain, built from the registry) + pandas (cross-field, missing, metrics, custom rules) |
| Degrades | if pandera absent: schema/range/domain skipped (warning); pandas checks **and custom rules** (e.g. `state`) still run |
| Config | `config/smarthub.yaml validation.high_missing_threshold` |

## 9. Known limitations / to revisit

| Item | Note |
|---|---|
| `household_income` numeric check | Registered as a non-negative numeric (`ValidationSpec` `min_value=0`), but the warehouse value is likely an income *band/string* — this can flag ~100% of rows (a rule mismatch, not bad data). Reclassify the field as categorical in the registry. |
| Boolean-domain checks | Boolean fields carry `kind="binary"` in the registry, but this isn't enforced as a domain check yet (would wire via a `custom_rule` or a boolean check kind). |
| High-missing noise | **Resolved.** The high-missing catalogue is now scoped per lead type (§6), so other products' columns aren't flagged on a single-type pull. |
| Two registries | Per-type required-raw columns are defined in both the feature registry (used by the completeness checks, §5) and the raw field registry (`lead_types`); consolidating is pending a team decision. |
