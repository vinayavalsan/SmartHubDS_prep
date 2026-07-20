# Anton Production Prediction Log - Schema Design

Status: proposed, ready for implementation. Reference implementation:
`src/smarthub/train_and_predict/prediction_log_schema.py`.

Scope note: this design tracks the ticket's own wording only - prediction
metadata, model information, an input feature snapshot, the final
prediction, optimizer results, candidate bid evaluations, and a structured
SHAP payload. It intentionally does not model any particular bidding
*policy* (cold-start handling, scheduled market exploration, or anything
else layered on top of the plain optimizer) and it does not include an
LLM-generated explanation field anywhere - see §2 and §9 for why.

## 1. Purpose

Anton will support two prediction endpoints (/recommend_bid and /explain_bid). This design will ensure that every prediction is permanently logged so it can be reconstructed, monitored, audited, and explained later. and then forgets how it answered them - nothing about a
prediction survives past the HTTP response. This document defines a
durable log that captures the complete lifecycle of every prediction: what
came in, which model answered, what the optimizer considered, what it
chose, and (when computed) the full SHAP breakdown behind the choice - so
that:

- **Monitoring** can track win-rate calibration, bid distribution, and
  optimizer behavior over time without re-running anything.
- **Debugging** a specific bid ("why did we bid $4.20 for lead 82931
  yesterday at 3am?") never requires the model version that served it to
  still exist, or the feature-engineering code to be unchanged since.
- **Auditing** can show a regulator or a partner exactly what data and what
  model produced a decision, on demand, indefinitely.
- **Explainability** already has the structured SHAP facts persisted, so a
  future LLM (or any other explanation UI) reads them straight out of the
  log - no recomputation, no coupling to any particular explanation
  implementation.

This closes directly against the ticket's five acceptance criteria; each is
addressed explicitly in §8.

## 2. Design principles

**Snapshot, don't reference.** Every field a reconstruction needs is copied
into the log row at prediction time - model version *and* its metrics/
lineage, the feature columns actually used, the exact post-feature-
engineering vector fed to the model. The model registry and the feature-
engineering code are both free to change or be rewritten later without
invalidating a single historical row.

**Two feature snapshots, not one.** `input_features` is the raw request as
the caller sent it (audit trail: "what did the caller ask for"). Separately,
`model_input_features` + `feature_cols` is the exact vector the
preprocessing step produced and the model actually scored (audit trail:
"what did the model see"). These two can legitimately differ - optional
features get toggled on/off per lead type, and a placeholder bid value gets
replaced by the actual chosen bid before SHAP runs - so both are logged
rather than assuming one can be derived from the other later.

**The optimizer's full candidate history is a table, not a blob.** Every
candidate bid the optimizer swept is a first-class row, with a flag for
whether it was the one selected. This is what makes "the optimizer
evaluation history for each prediction is retained" (the ticket's second
acceptance criterion) actually queryable - e.g. "what fraction of the swept
grid ends up within 10% of the selected bid's profit" - rather than
requiring a JSON-array unpack for every question.

**SHAP is structured, complete, and has no LLM anywhere in the schema.**
The stored payload covers *every* feature the model saw, not a partial
subset. There is no explanation-text column, no LLM-model column, nothing
LLM-shaped anywhere in this design - the ticket is explicit that "no
LLM-generated explanations are required" and that a future LLM can be added
"without modifying the prediction pipeline or recomputing model
explanations." Keeping this schema free of any explanation-generation
concept today is precisely what makes that true later: whatever eventually
reads `smarthub_shap_explanations` to write a sentence is a downstream
consumer, not a participant in this design.

**Numbers carry their units.** SHAP values from a tree model are commonly
computed in log-odds space, not probability space - a base value like
`-2.0` is a normal log-odds figure (`sigmoid(-2.0) ≈ 0.12`), not a
malformed probability. `shap_space` is a real column here, not a comment,
and both the margin and the probability form of the base value are stored
side by side so nothing downstream has to guess which space a number is in.

**Every table carries `lead_ping_id`.** Not just the parent - see §3.2.
The whole reason this log exists is to answer "what did Anton do for lead
X, and why", and that question is always going to start from a
`lead_ping_id`, not a `prediction_id` (nobody outside this schema knows or
cares about `prediction_id`). Denormalizing the correlation key onto every
table means every one of them answers that question on its own, with no
join required first.

## 3. Entity-relationship overview

```
smarthub_prediction_log  (1)
        │
        ├──< smarthub_optimizer_bid_evaluations   (0..N - every candidate bid evaluated)
        │
        └──< smarthub_shap_explanations           (0..1 - present iff SHAP actually ran)
```

One row in `smarthub_prediction_log` per call to `/recommend_bid` or
`/explain_bid`. `smarthub_optimizer_bid_evaluations` holds one row per
candidate bid the optimizer's grid sweep considered for that prediction.
`smarthub_shap_explanations` holds at most one row per prediction - present
only when a SHAP explanation was actually computed for it.

### 3.1 Join keys

All three tables share `(prediction_id, log_date)` as their structural join
key - `prediction_id` is the real identifier; `log_date` rides along
because of a Postgres partitioning rule (§3.2). Separately, all three
tables **also** carry `lead_ping_id` (§2, §4) - a second, business-facing
correlation key that lets you query any of the three tables directly by
"which lead was this," without going through `prediction_id` first. Use
whichever key matches the question you're asking: `prediction_id` to pull
everything about one specific API call, `lead_ping_id` to pull everything
Anton ever did for one specific lead (possibly across more than one
prediction, e.g. a retry).

### 3.2 Why `log_date` is part of every primary/foreign key

This is a Postgres mechanic, not a modeling choice, and it trips people up
the first time they hit it, so it's worth spelling out.

Postgres range-partitions a table by physically splitting it into separate
child tables (partitions), one per date range, and it enforces uniqueness
**per partition**, not across the whole table with one global index. If you
declared `PRIMARY KEY (prediction_id)` alone on a table partitioned by
`log_date`, Postgres cannot guarantee `prediction_id` is unique overall -
it would need to check every partition, not just one, and it doesn't do
that. So it refuses outright, at `CREATE TABLE` time:

```
ERROR:  unique constraint on partitioned table must include all partitioning columns
DETAIL: PRIMARY KEY constraint on table "smarthub_prediction_log" lacks column "log_date",
        which is part of the partition key.
```

The fix is simply to include the partition column in the key -
`PRIMARY KEY (prediction_id, log_date)` - which is what every table in this
design does. It's a satisfy-Postgres technicality, not a real modeling
statement: `prediction_id` is still a UUID4 generated fresh by the
application for every single prediction, so in practice it is already
globally unique on its own; `log_date` just has to ride along in the key
declaration for Postgres to accept the partitioning.

This is also why the two child tables' foreign keys reference
`(prediction_id, log_date)` rather than just `prediction_id` - a foreign
key must reference a column set that Postgres can enforce uniqueness on,
and per the rule above, that's the two-column combination, not
`prediction_id` alone. `log_date` on the child rows is simply copied from
the parent's `log_date` at insert time (see the `PredictionLogStore.
log_prediction` reference implementation) - it is never computed
independently on the child side.

## 4. `smarthub_prediction_log` (parent)

One row per call to `/recommend_bid` or `/explain_bid` - the root record
every other table hangs off of.

**Primary key:** `(prediction_id, log_date)`, composite - see §3.2 for why
`log_date` is there.

**Foreign keys:** none - this is the root table. `lead_ping_id` is a
*logical* reference to `public.lead_pings.id` (§9), not a DB-enforced FK,
since that table may live in a different database/schema.

**Check constraint:** `ck_prediction_log_endpoint` -
`endpoint IN ('recommend_bid', 'explain_bid')`.

**Indexes:**
- `pk_smarthub_prediction_log` (unique) - `(prediction_id, log_date)`
- `ix_prediction_log_lead_type_served` - `(lead_type_id, served_at)`
- `ix_prediction_log_campaign_served` - `(campaign_id, served_at)`
- `ix_prediction_log_lead_ping_id` - `(lead_ping_id)` - the lookup index for "everything Anton did for this lead" (§3.1).
- `ix_prediction_log_input_features_gin` (GIN, recommended) - `(input_features)`, for ad-hoc "which predictions had feature X = Y" queries without a full-table JSON scan.

**Partitioning:** `PARTITION BY RANGE (log_date)`, monthly (§10).

| Column | Type | Notes |
| --- | --- | --- |
| `prediction_id` | UUID | Primary key (with `log_date`, §3.2). App-generated (`uuid4()`), returned to the caller so a later `/explain_bid` call (or a support ticket) can reference the same id. |
| `served_at` | timestamptz | When this prediction was served. |
| `log_date` | date | `served_at::date` - partition key, see §3.2/§10. |
| `schema_version` | smallint | Bumped whenever this table's shape changes; lets old rows and new code coexist during a migration. |
| `request_id` | text, nullable | Caller-supplied correlation id, if the upstream system has one. |
| `lead_ping_id` | bigint, nullable | Reference to `public.lead_pings.id` - **the primary way to map a prediction (and its optimizer/SHAP children, §3.1) back to a specific lead.** Nullable because a bid can be requested before/independent of a recorded ping; see §9, this needs to be threaded through the request schema first. This is also the join key back to the realized `won` outcome for calibration monitoring. |
| `endpoint` | text | `'recommend_bid'` \| `'explain_bid'`. |
| `served_by_host` | text, nullable | Pod/instance id - ops debugging for a specific bad request. |
| `lead_type_id`, `lead_type_name` | smallint, text | Which lead-type model answered this prediction. |
| `campaign_id`, `account_id`, `source_type_id` | bigint | Business context from the request, straight through. |
| `input_features` | jsonb | The raw request payload (minus the optimizer knobs, which have their own columns below) - exactly what the caller sent. |
| `model_input_features` | jsonb | The exact row the preprocessing step produced and the model scored, **at the recommended bid**. |
| `feature_cols` | jsonb | Ordered list of columns the serving model version actually used (from its manifest) - records which optional features were on/off for this prediction. |
| `expected_revenue`, `target_cm`, `min_bid`, `bid_step` | numeric | Optimizer inputs, straight from the request. |
| `max_bid` | numeric, nullable | Derived: `expected_revenue * (1 - target_cm)`. |
| `n_candidate_bids` | integer, nullable | Size of the optimizer's grid sweep. |
| `model_version`, `model_uri` | text | Which model artifact answered this prediction. |
| `model_type`, `model_calibrated` | text, boolean | From the manifest (e.g. `lightgbm`/`logistic_regression`; isotonic calibration on/off). |
| `training_table_version` | text, nullable | From the manifest's lineage - which training-table snapshot produced this model. |
| `model_data_min_created_at`, `model_data_max_created_at` | timestamptz, nullable | From the manifest's lineage - the training data's date range. |
| `recommended_bid` | numeric, nullable | NULL means "no viable bid" (e.g. expected revenue too low to clear the target margin at or above the minimum bid). |
| `recommended_bid_predicted_win_rate`, `recommended_bid_expected_profit` | numeric, nullable | At the recommended bid. |
| `created_at` | timestamptz | Row-insert time - distinct from `served_at` only if writes are ever queued/async (§10). |

**DDL:**

```sql
CREATE TABLE smarthub_prediction_log (
    prediction_id                       UUID          NOT NULL,
    served_at                           TIMESTAMPTZ   NOT NULL DEFAULT now(),
    log_date                            DATE          NOT NULL,
    schema_version                      SMALLINT      NOT NULL DEFAULT 1,
    request_id                          TEXT,
    lead_ping_id                        BIGINT,
    endpoint                            TEXT          NOT NULL,
    served_by_host                      TEXT,

    lead_type_id                        SMALLINT      NOT NULL,
    lead_type_name                      TEXT          NOT NULL,
    campaign_id                         BIGINT        NOT NULL,
    account_id                          BIGINT,
    source_type_id                      BIGINT,

    input_features                      JSONB         NOT NULL,
    model_input_features                JSONB         NOT NULL,
    feature_cols                        JSONB         NOT NULL,

    expected_revenue                    NUMERIC(12,2) NOT NULL,
    target_cm                           NUMERIC(5,4)  NOT NULL,
    min_bid                             NUMERIC(10,2) NOT NULL,
    bid_step                            NUMERIC(10,2) NOT NULL,
    max_bid                             NUMERIC(10,2),
    n_candidate_bids                    INTEGER,

    model_version                       TEXT          NOT NULL,
    model_uri                           TEXT          NOT NULL,
    model_type                          TEXT          NOT NULL,
    model_calibrated                    BOOLEAN       NOT NULL,
    training_table_version              TEXT,
    model_data_min_created_at           TIMESTAMPTZ,
    model_data_max_created_at           TIMESTAMPTZ,

    recommended_bid                     NUMERIC(10,2),
    recommended_bid_predicted_win_rate  NUMERIC(6,5),
    recommended_bid_expected_profit     NUMERIC(12,4),
    created_at                          TIMESTAMPTZ   NOT NULL DEFAULT now(),

    CONSTRAINT pk_smarthub_prediction_log PRIMARY KEY (prediction_id, log_date),
    CONSTRAINT ck_prediction_log_endpoint
        CHECK (endpoint IN ('recommend_bid', 'explain_bid'))
) PARTITION BY RANGE (log_date);

CREATE INDEX ix_prediction_log_lead_type_served
    ON smarthub_prediction_log (lead_type_id, served_at);
CREATE INDEX ix_prediction_log_campaign_served
    ON smarthub_prediction_log (campaign_id, served_at);
CREATE INDEX ix_prediction_log_lead_ping_id
    ON smarthub_prediction_log (lead_ping_id);
CREATE INDEX ix_prediction_log_input_features_gin
    ON smarthub_prediction_log USING GIN (input_features);
```

## 5. `smarthub_optimizer_bid_evaluations` (child, 0..N)

One row per candidate bid the optimizer's grid sweep evaluated for a
prediction.

**Primary key:** `(prediction_id, log_date, sequence_no)`, composite - see
§3.2 for why `log_date` is there.

**Foreign key:** `fk_optimizer_bid_evaluations_prediction` -
`(prediction_id, log_date)` → `smarthub_prediction_log (prediction_id, log_date)`,
`ON DELETE CASCADE` (deleting a prediction row deletes its candidates with it).

**Indexes:**
- `pk_smarthub_optimizer_bid_evaluations` (unique) - `(prediction_id, log_date, sequence_no)`
- `ix_optimizer_bid_evaluations_selected` - `(prediction_id, is_selected)`, for "which bid did we actually pick" lookups without scanning every candidate.
- `ix_optimizer_bid_evaluations_lead_ping_id` - `(lead_ping_id)` - same rationale as the parent's (§3.1): pull a lead's whole candidate-bid history directly.

**Partitioning:** `PARTITION BY RANGE (log_date)`, monthly - mirrors the
parent so a query joining both tables for a given month prunes both
identically (§10).

| Column | Type | Notes |
| --- | --- | --- |
| `prediction_id`, `log_date` | UUID, date | FK to the parent (composite, see above). |
| `lead_ping_id` | bigint, nullable | Denormalized from the parent (§2, §3.1) - lets you query this table directly by lead, e.g. "show me every candidate bid Anton evaluated for lead 82931," with no join. |
| `sequence_no` | smallint | **Position of this candidate within the optimizer's sweep, ordered by ascending bid** - `0` is the lowest bid tried, the last value is the highest. See the worked example in §7 for concrete numbers. This exists so the exact grid order is reconstructable on its own terms, rather than relying on sorting the `bid` column and hoping there are no ties. |
| `bid` | numeric | |
| `predicted_win_rate` | numeric | |
| `expected_profit` | numeric | |
| `is_selected` | boolean | True on exactly one row per prediction - the candidate that became `recommended_bid`. |

**DDL:**

```sql
CREATE TABLE smarthub_optimizer_bid_evaluations (
    prediction_id       UUID          NOT NULL,
    log_date            DATE          NOT NULL,
    lead_ping_id         BIGINT,
    sequence_no          SMALLINT      NOT NULL,
    bid                  NUMERIC(10,2) NOT NULL,
    predicted_win_rate   NUMERIC(6,5)  NOT NULL,
    expected_profit      NUMERIC(12,4) NOT NULL,
    is_selected          BOOLEAN       NOT NULL DEFAULT false,

    CONSTRAINT pk_smarthub_optimizer_bid_evaluations
        PRIMARY KEY (prediction_id, log_date, sequence_no),
    CONSTRAINT fk_optimizer_bid_evaluations_prediction
        FOREIGN KEY (prediction_id, log_date)
        REFERENCES smarthub_prediction_log (prediction_id, log_date)
        ON DELETE CASCADE
) PARTITION BY RANGE (log_date);

CREATE INDEX ix_optimizer_bid_evaluations_selected
    ON smarthub_optimizer_bid_evaluations (prediction_id, is_selected);
CREATE INDEX ix_optimizer_bid_evaluations_lead_ping_id
    ON smarthub_optimizer_bid_evaluations (lead_ping_id);
```

This directly satisfies the ticket's optimizer-explainability requirement:
"record every candidate bid evaluated by the optimizer, including predicted
win probability, expected profit, and whether the bid was selected."

### 5.1 What `sequence_no` actually looks like

Say a lead has `expected_revenue = $50.00`, `target_cm = 25%` (so
`max_bid = 50.00 × (1 − 0.25) = $37.50`), `min_bid = $0.25`, and
`bid_step = $0.25`. The optimizer's grid is every bid from `$0.25` to
`$37.50` in `$0.25` increments - 150 candidates. `sequence_no` numbers them
in that same ascending order:

| sequence_no | bid | predicted_win_rate | expected_profit | is_selected |
| --- | --- | --- | --- | --- |
| 0 | 0.25 | 0.05 | 2.36 | false |
| 1 | 0.50 | 0.06 | 2.97 | false |
| … | … | … | … | … |
| 49 | 12.50 | 0.34 | 12.75 | **true** |
| … | … | … | … | … |
| 149 | 37.50 | 0.61 | 7.63 | false |

So `sequence_no = 49` is simply "the 50th bid this sweep tried, in
ascending order" - it lets you answer "what did the grid look like right
around the winning bid" or "reproduce the exact sweep for a chart" without
depending on the `bid` column being unique or sorted at query time (it
happens to be both here, but `sequence_no` doesn't rely on that holding).

## 6. `smarthub_shap_explanations` (child, 0..1)

The structured SHAP payload for the selected prediction - present only
when SHAP was actually computed for it (today, only `/explain_bid`
requests compute SHAP).

**Primary key:** `(prediction_id, log_date)` - enforces the 1:1 relationship
with the parent (at most one SHAP explanation per prediction); see §3.2
for why `log_date` is there.

**Foreign key:** `fk_shap_explanations_prediction` -
`(prediction_id, log_date)` → `smarthub_prediction_log (prediction_id, log_date)`,
`ON DELETE CASCADE`.

**Indexes:**
- `pk_smarthub_shap_explanations` (unique) - `(prediction_id, log_date)`
- `ix_shap_explanations_lead_ping_id` - `(lead_ping_id)` - pull the explanation for a lead directly, no join through `prediction_id` (§3.1).

**Partitioning:** `PARTITION BY RANGE (log_date)`, monthly - mirrors the parent (§10).

| Column | Type | Notes |
| --- | --- | --- |
| `prediction_id`, `log_date` | UUID, date | FK to the parent; also this table's primary key (1:1). |
| `lead_ping_id` | bigint, nullable | Denormalized from the parent (§2, §3.1) - "give me the explanation for lead 82931" needs no join. |
| `explained_bid` | numeric | The specific bid SHAP was computed at (the recommended bid). |
| `shap_space` | text | Always `'log_odds'` today - an explicit unit tag, not a comment, so a future reader (human or otherwise) can never silently mix it up with probability space. |
| `base_value_margin` | numeric | Raw log-odds base value, averaged across any calibration folds. |
| `base_value_probability` | numeric | `sigmoid(base_value_margin)` - the model's average predicted win rate before this lead's specific factors are applied. |
| `model_version` | text | Denormalized from the parent, for convenience when querying this table alone. |
| `top_n_surfaced` | smallint, nullable | How many of the features below were also returned in the API response for human review - the stored payload itself is never truncated (see §9). |
| `feature_contributions` | jsonb | Array covering **every** `feature_cols` entry - see JSON Schema below. |

**DDL:**

```sql
CREATE TABLE smarthub_shap_explanations (
    prediction_id            UUID           NOT NULL,
    log_date                 DATE           NOT NULL,
    lead_ping_id              BIGINT,
    explained_bid            NUMERIC(10,2)  NOT NULL,
    shap_space                TEXT           NOT NULL DEFAULT 'log_odds',
    base_value_margin         NUMERIC(10,6)  NOT NULL,
    base_value_probability    NUMERIC(6,5)   NOT NULL,
    model_version             TEXT           NOT NULL,
    top_n_surfaced            SMALLINT,
    feature_contributions     JSONB          NOT NULL,

    CONSTRAINT pk_smarthub_shap_explanations PRIMARY KEY (prediction_id, log_date),
    CONSTRAINT fk_shap_explanations_prediction
        FOREIGN KEY (prediction_id, log_date)
        REFERENCES smarthub_prediction_log (prediction_id, log_date)
        ON DELETE CASCADE
) PARTITION BY RANGE (log_date);

CREATE INDEX ix_shap_explanations_lead_ping_id
    ON smarthub_shap_explanations (lead_ping_id);
```

### SHAP payload JSON Schema (`feature_contributions`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "shap_feature_contributions",
  "type": "array",
  "minItems": 1,
  "items": {
    "type": "object",
    "required": ["feature", "value", "shap_value", "direction", "rank"],
    "additionalProperties": false,
    "properties": {
      "feature": { "type": "string" },
      "value": { "type": ["number", "string", "boolean", "null"] },
      "shap_value": {
        "type": "number",
        "description": "Log-odds contribution (see shap_space) - NOT a probability. Sign/relative magnitude are valid as-is; sum with base_value_margin and sigmoid the total for a probability."
      },
      "direction": { "type": "string", "enum": ["increased", "decreased"] },
      "rank": {
        "type": "integer",
        "minimum": 1,
        "description": "1 = largest |shap_value| across ALL feature_cols, not just the surfaced subset."
      }
    }
  }
}
```

This satisfies the ticket's model-explainability requirement directly: a
structured (JSON), LLM-independent payload with the contribution of every
feature to the predicted win probability, for the selected prediction. A
full worked example of this payload is in §7.3.

## 7. Worked example - one full prediction, sample rows for all three tables

One lead ping (`lead_ping_id = 82931`), one prediction, shown as it would
actually sit in all three tables. Same `prediction_id` and `lead_ping_id`
throughout - this is what "mapping a prediction and its explanation to a
lead ping" (§2, §3.1) looks like in practice.

### 7.1 `smarthub_prediction_log` - one row

| Column | Value |
| --- | --- |
| `prediction_id` | `a0460922-6579-4697-b8fd-263bb76a74b7` |
| `served_at` | `2026-07-19 03:00:12+00` |
| `log_date` | `2026-07-19` |
| `schema_version` | `1` |
| `request_id` | `null` |
| `lead_ping_id` | `82931` |
| `endpoint` | `explain_bid` |
| `served_by_host` | `anton-api-7d9f6c` |
| `lead_type_id` / `lead_type_name` | `6` / `auto` |
| `campaign_id` / `account_id` / `source_type_id` | `4021` / `118` / `3` |
| `input_features` | `{"state": "CA", "age": 34, "num_auto_accidents": 1, ...}` |
| `model_input_features` | `{"age": 34.0, "state": "CA", "num_auto_accidents": 1.0, "bid": 12.50, ...}` |
| `feature_cols` | `["age", "state", "num_auto_accidents", "bid", ...]` |
| `expected_revenue` / `target_cm` / `min_bid` / `bid_step` | `50.00` / `0.2500` / `0.25` / `0.25` |
| `max_bid` | `37.50` |
| `n_candidate_bids` | `150` |
| `model_version` | `v4_2026-07-09T140501Z` |
| `model_uri` | `/data/models/auto/v4_2026-07-09T140501Z.pkl` |
| `model_type` / `model_calibrated` | `lightgbm` / `true` |
| `training_table_version` | `auto_2026-07-09` |
| `model_data_min_created_at` / `model_data_max_created_at` | `2026-04-11 00:00:00+00` / `2026-07-08 23:59:00+00` |
| `recommended_bid` | `12.50` |
| `recommended_bid_predicted_win_rate` | `0.34000` |
| `recommended_bid_expected_profit` | `12.7500` |
| `created_at` | `2026-07-19 03:00:12.884+00` |

### 7.2 `smarthub_optimizer_bid_evaluations` - a slice of 150 rows

Same `prediction_id` and `lead_ping_id` on every row (first three, the
selected one, and the last one shown; the other 146 follow the same
pattern):

| sequence_no | lead_ping_id | bid | predicted_win_rate | expected_profit | is_selected |
| --- | --- | --- | --- | --- | --- |
| 0 | 82931 | 0.25 | 0.05000 | 2.3625 | false |
| 1 | 82931 | 0.50 | 0.06000 | 2.9700 | false |
| 2 | 82931 | 0.75 | 0.07000 | 3.4475 | false |
| … | … | … | … | … | … |
| 49 | 82931 | 12.50 | 0.34000 | 12.7500 | **true** |
| … | … | … | … | … | … |
| 149 | 82931 | 37.50 | 0.61000 | 7.6250 | false |

### 7.3 `smarthub_shap_explanations` - one row

| Column | Value |
| --- | --- |
| `prediction_id` | `a0460922-6579-4697-b8fd-263bb76a74b7` |
| `log_date` | `2026-07-19` |
| `lead_ping_id` | `82931` |
| `explained_bid` | `12.50` |
| `shap_space` | `log_odds` |
| `base_value_margin` | `-2.050000` |
| `base_value_probability` | `0.11400` |
| `model_version` | `v4_2026-07-09T140501Z` |
| `top_n_surfaced` | `5` |
| `feature_contributions` | see JSON below |

```json
[
  {"feature": "num_auto_accidents", "value": 1,     "shap_value": 0.84,  "direction": "increased", "rank": 1},
  {"feature": "age",                 "value": 34.0,  "shap_value": -0.31, "direction": "decreased", "rank": 2},
  {"feature": "state",               "value": "CA",  "shap_value": -0.12, "direction": "decreased", "rank": 3},
  {"feature": "num_vehicles",        "value": 2.0,   "shap_value": 0.09,  "direction": "increased", "rank": 4},
  {"feature": "insured",             "value": "Y",   "shap_value": -0.04, "direction": "decreased", "rank": 5},
  {"feature": "bid",                 "value": 12.50, "shap_value": 1.06,  "direction": "increased", "rank": 6}
]
```

(Six features shown here in full - `top_n_surfaced = 5` means only the
first five of these would have appeared in a human-facing API response;
the stored payload still has all six, per §9's second gap.)

**Reading it end to end:** lead ping `82931` came in, Anton's auto model
(`v4_2026-07-09T140501Z`) swept 150 candidate bids from `$0.25` to
`$37.50` (§7.2), picked `$12.50` as the one maximizing expected profit
(§7.1), and - because this was an `/explain_bid` call - also ran SHAP at
that specific bid (§7.3): starting from an `11.4%` population-average win
rate (`base_value_probability = 0.114`), this lead's own factors (a prior
accident, being in CA, etc.) pushed the predicted win rate up to `34%` at
the chosen bid.

## 8. Acceptance criteria - how this schema satisfies each one

**"Every prediction can be fully reconstructed from the logged data."**
`input_features` + `model_input_features` + `feature_cols` capture both what
was asked and what the model actually saw; `model_version`/`model_type`/
`model_calibrated`/lineage fields capture which model and how it was built,
independent of the registry still holding that version. Nothing needed to
answer "what happened here" lives anywhere else.

**"The optimizer evaluation history for each prediction is retained."**
`smarthub_optimizer_bid_evaluations`, one row per candidate, §5.

**"The schema stores a structured SHAP payload for the selected prediction."**
`smarthub_shap_explanations.feature_contributions`, JSON Schema in §6.

**"The schema does not depend on an LLM."** There is no explanation-text
column, no LLM-model column, and no LLM-shaped concept anywhere in this
schema - every column here is either a request/model/optimizer fact or a
structured SHAP number.

**"The schema is documented and ready for implementation."** This document
plus `prediction_log_schema.py` (SQLAlchemy Core table definitions, a
`PredictionLogStore` read/write helper, and the JSON Schema as a Python
constant) - verified end-to-end (write, read back, reconstruct all three
tables including the `lead_ping_id` lookup path, JSON-Schema-validate the
payload) against SQLite in this session; Postgres-specific behavior
(JSONB, native `uuid`) is exercised via SQLAlchemy's dialect variants and
should be re-verified against a real Postgres instance before go-live.

## 9. Two gaps worth closing, surfaced by this design (not yet acted on)

**No `lead_ping_id` on the request schema today.** The schema itself is now
fully wired for it (every table carries `lead_ping_id`, §2, §3.1), but the
live request models (`BidRequest`) don't accept one yet, so there's nothing
to populate it with at call time. Without it, this log can't be joined
back to `public.lead_pings.won` to compute realized calibration ("was
Anton's predicted win rate actually right?") - the single most important
monitoring question this log exists to answer. Recommend adding an
optional `lead_ping_id: int | None` to the request schema (and threading
it through to whatever system calls `/recommend_bid`), even though it
changes nothing about the bid decision itself.

**The SHAP-computing code path only returns a top-N subset today, not
every feature.** For this log's `feature_contributions` to genuinely cover
"every feature" (the ticket's explicit wording), the logging call site
needs the *untruncated* per-feature SHAP values, with `top_n_surfaced`
recording only how many of those are shown in an API response - a small
change wherever SHAP values are currently sliced down to a display subset,
not to the SHAP computation itself.

## 10. Scalability

**Traffic profile.** Current volume is intentionally conservative today;
this design is meant to comfortably outlive that, not just fit it. A
SHAP-computing endpoint is typically an on-demand/explanatory path rather
than the hot serving path, so its heavier per-call cost (a full candidate
sweep plus a full feature-length SHAP array) is inherently rate-limited by
being invoked less often. The plain bid-recommendation path is the hot
path and the one that needs a deliberate write-volume decision as QPS
grows - see below.

**Partitioning.** Range-partition all three tables by `log_date`, monthly
- see §3.2 for exactly why `log_date` has to be part of every key for this
to be possible at all. The payoff: retention becomes `DROP TABLE
smarthub_prediction_log_2025_01` (instant) instead of a slow, WAL-heavy
`DELETE ... WHERE log_date < ...` over hundreds of millions of rows, and
query planners prune whole months for any time-bounded query (which is
almost all monitoring/audit queries in practice).

Partitions must be created (and dropped) for all three tables together,
one month ahead of need, e.g.:

```sql
CREATE TABLE smarthub_prediction_log_2026_07
    PARTITION OF smarthub_prediction_log
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

CREATE TABLE smarthub_optimizer_bid_evaluations_2026_07
    PARTITION OF smarthub_optimizer_bid_evaluations
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

CREATE TABLE smarthub_shap_explanations_2026_07
    PARTITION OF smarthub_shap_explanations
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
```

Automate this (e.g. `pg_partman`, or a small scheduled task alongside the
existing pipeline flows) rather than creating partitions by hand - a
missing next-month partition means inserts start failing at midnight on
the 1st, not degrading gracefully.

If native partitioning feels like premature operational overhead before
volume justifies it, the fallback is a single unpartitioned table with a
BRIN index on `served_at` - BRIN indexes are extremely cheap (kilobytes,
not gigabytes) on append-mostly, naturally-time-ordered data like this, and
get you most of the time-range query performance without partition-
management tooling (and without needing `log_date` in the key at all - that
requirement only exists once you partition natively, §3.2). Migrate to
native partitioning when retention/DROP performance actually becomes the
bottleneck, not before.

**Write-volume control on the hot path.** Logging the full optimizer sweep
(100+ rows) on every single bid-recommendation call is fine at today's
volume and would not be at meaningfully higher QPS. Recommend a config
knob (matching this codebase's existing config tiering) such as
`prediction_log.optimizer_detail: full | selected_only | sampled`,
defaulting to `selected_only` for the hot path (log just the one
`is_selected=true` row - reconstructs the decision, not the full search)
and `full` for the on-demand explanatory path (already low-volume by
design, and the whole point of that endpoint is showing the search). This
keeps the schema itself unchanged either way - it's a row-count decision
at write time, not a structural one.

**Decoupling the write path from the request path.** None of this should
sit synchronously in front of the HTTP response on the hot path. The
natural evolution as QPS grows: write synchronously today (simplest, fine
at current volume) → move to a background task/queue (e.g. a lightweight
in-process async write, or a proper queue if the write volume needs true
decoupling) once the hot-path latency budget gets tight. This is
explicitly **not** designed in this document - it's a serving-path
implementation concern, not a schema concern, and shouldn't block reviewing
or implementing the tables themselves.

**Storage location.** Reuses the existing shared Postgres
(`smarthub.core.config_store`'s instance) by default - same environment-
variable pattern (`SMARTHUB_PREDICTION_LOG_DB_URL`, mirroring
`SMARTHUB_CONFIG_DB_URL`), and co-location makes the `lead_ping_id` join
against `public.lead_pings.won` (§9) a same-database query instead of
cross-database federation. Split onto a dedicated logging/analytics
instance later purely as a scaling move if this table's write volume ever
starts contending with the config store's much lighter read/write pattern.
Very old partitions (e.g. anything past a 12–24 month hot-retention window)
are natural candidates to archive to Parquet, mirroring this project's
existing year/month/day-partitioned Parquet convention already used for
raw lead pulls, rather than growing Postgres unboundedly.

## 11. Explicitly not in scope here

- Wiring `PredictionLogStore.log_prediction(...)` into any live endpoint.
- The synchronous-vs-async write-path decision (§10) - needs a QPS/latency
  budget the team hasn't set yet.
- A migration tool (Alembic vs. `MetaData.create_all`) for evolving this
  schema in a real Postgres environment - `create_all` is fine for a fresh
  environment/tests; a running production database will want real
  migrations once this ships.
- Threading `lead_ping_id` through the request schema and whatever calls
  the bid-recommendation endpoint today (§9, first gap).
- Fixing the top-N truncation so the full per-feature SHAP dict is
  available to log (§9, second gap).
- Any bidding *policy* on top of the plain optimizer (cold-start handling,
  scheduled exploration, or similar) and any LLM-based explanation
  generation - both are separate features layered on top of what this
  schema logs, not modeled by this schema itself. Should either be added
  to the live system later, the natural extension point is additive: a new
  nullable column or a new child table, not a change to what's defined
  here.

These are natural next tickets, not blockers on reviewing or approving the
schema design itself.



## 12. Presentation Layer (Natural-Language Explainability)

### Purpose

The prediction logging schema is designed to persist **structured facts**, not
human-readable explanations. The logged prediction metadata, optimizer
evaluations, and SHAP feature contributions together form the system of
record for every prediction.

A separate presentation layer can consume this structured data to generate
business-friendly explanations for non-technical users without modifying the
prediction pipeline or recomputing SHAP values.

Keeping explanation generation outside the prediction path provides several
benefits:

- Preserves the logged data as the single source of truth.
- Keeps prediction serving deterministic and fully auditable.
- Allows different explanation technologies to evolve independently.
- Avoids vendor lock-in to any specific language model implementation.

### Proposed Architecture

```
Prediction Request
        │
        ▼
 Prediction Model
        │
 ┌──────┴────────┐
 ▼               ▼
Optimizer      SHAP
        │        │
        └────┬───┘
             ▼
 Prediction Logging Schema
 (System of Record)
             │
             ▼
     Presentation Layer
             │
     ┌───────┼─────────┐
     ▼       ▼         ▼
 Dashboard  API     SLM-based Explainer
                         │
                         ▼
            Human-readable explanation
```

### Why an SLM?

The stored SHAP payload is already structured and contains the feature names,
feature values, contribution magnitudes, and prediction context. Converting
this structured information into plain English is primarily a summarization
and formatting task rather than an open-ended reasoning problem.

For this reason, a **Small Language Model (SLM)** or may be **Tiny language Model (TLM)** is expected to be sufficient
for most explainability use cases while offering:

- Lower inference latency
- Lower infrastructure cost
- Easier on-premises deployment
- Reduced operational complexity
- Better control over data privacy

Suitable SLMs include (subject to future evaluation):

- **Phi-4 Mini** (Microsoft)
- **Phi-3.5 Mini** (Microsoft)
- **Llama 3.2 3B**
- **Gemma 3 4B** (Google)
- **Qwen2.5 3B Instruct**

Suitable TLMs include (subject to future evaluation):

- **TinyLlama** 
- **Gemma 3 1B** 
- **Qwen2.5 1.5B Instruct**
- **Llama 3.2 1B Instruct** 
- **Phi-4 Mini**

The presentation layer should remain model-agnostic so that any suitable SLM,
LLM, or future explanation engine can be substituted without changing the
prediction logging schema.

### Example Flow

1. Retrieve the prediction record and associated SHAP payload.
2. Build a structured prompt from the stored data.
3. Pass the prompt to the presentation model (SLM/LLM).
4. Return a concise, human-readable explanation to the requesting user.

The generated explanation is **not** persisted as part of the prediction log.
The structured SHAP payload remains the authoritative source of
explainability, while the generated text is treated purely as a presentation
artifact.


