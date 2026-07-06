# SmartHub — Domain Context

This document captures the business context behind the code in this repo, based on
Kiran's DS Weekly walkthrough (Jun 23, 2026). It explains *what the data means* and
*what problem Anton is solving*, so the analysis in `prep/` and `src/monitoring/`
can be understood in business terms.

---

## 1. What SmartHub is

SmartHub is a **lead marketplace / reseller**. It does not generate leads itself —
it sits in the middle:

- **Upstream partners (producers)** *have* the leads and offer them to us.
- **Downstream buyers (agents)** *purchase* the leads from us.

SmartHub buys from partners and resells to buyers, keeping the spread.

**Anton** is the data-science bidding model — it decides how much SmartHub should
bid for each incoming lead. The goal is to win the most valuable leads at the best
price, programmatically and explainably (not a black box).

---

## 2. Key players

### Partners (upstream / sellers)
Companies that send us leads. ~6 total, 4 active (e.g. **Healthy Labs**,
**Launch Potato**, **Malva**). In the SmartHub portal these are called **accounts**.

- A partner has one or more **campaigns** (`campaign_id`).
- Most partners have a single campaign. **Malva** is the only one with two
  (a *home* campaign and an *auto* campaign).
- Volume is intentionally conservative right now and will grow as bigger partners
  onboard — so low sample counts today are expected to improve.

### Buyers (downstream / agents)
The agents we resell to, spread across marketplaces: **Insurance.io** (~200 accounts),
**SF Pro**, **BPIO** (1,000+ accounts). Different buyers purchase different products
(leads / calls / clicks) — mostly only one of the three. Think of leads/calls/clicks
buyers as three overlapping circles with little overlap.

---

## 3. How one lead flows (lead side)

1. **Ping** — a partner sends a ping: "here's a lead, what will you pay?" The ping
   carries non-PII attributes: state, city, vehicles, age, accident history,
   currently-insured flag, etc.
2. **Buyer match** — SmartHub instantly checks its **buyer distribution**: which
   active buyer campaigns match this lead and what they'd pay. Some buyers just have
   filters (checking them = a "ping"); others have a **ping-post** setup where we
   forward the non-PII data and they reply yes/no and sometimes a price. The backend
   ranks everyone into **listings**.
3. **Bid** — SmartHub bids back to the partner.
4. **Win** — if the partner accepts our bid, we've **won the lead**, and then try to
   actually resell it downstream.

Rough funnel from the demo: ~19,900 pings → ~76% returned a valid bid →
~5,698 leads won → ~half successfully resold.

---

## 4. Money mechanics (the important part)

**Reseller rule:** we only owe the partner money if we actually resell the lead to at
least one buyer.

- Win the lead but all buyers reject → we reject it back upstream → **no cost, no
  revenue** ("dead ping", a harmless wash).
- **Shared** lead (e.g. two buyers each at $5) where one accepts and one rejects → we
  are stuck paying the partner but only collected $5 → hurts contribution margin.
- Occasionally we lose money outright (bid $3, won, had to pay, made nothing). Early
  on bidding was too aggressive — day one lost ~$2,600. It's since been tuned down.

**Bidding strategies** are labeled `dumb 10 / 25 / 50 / 75` — these are just
**contribution-margin (CM)** targets. If a buyer would pay $10 and we use the 25%
strategy, we bid $7.50 (keeping 25%). A random **±10% variance** is applied to each
bid on purpose, so we sample many price points and learn win rate at each. Strategies
are rotated round-robin to probe the market.

**Expected revenue (key for Anton):** a *separate* table stores **expected revenue**
— what we think we'll make, **already discounted by how often buyers reject**.
Example: top buyer would pay $78 but only accepts 20% of the time → expected revenue
≈ $15.90. This is why revenue can look much larger than the bid, and why margins look
fat — the reject rate is already baked in, so you don't model it separately.

Expected revenue is **not itself the ceiling**. It is the input that, combined with
our **target CM**, sets the highest bid we'd be willing to make (see §6). See §6 for
how the actual bidding bounds are defined and discovered — this was clarified by
Vinaya and Kiran after the original draft of this doc.

> **UPDATE (1 Jul 2026 — supersedes the aggregation below):** the team decided the
> backend will **add an `expected_revenue` column to `lead_pings`**, and Anton must
> **use that value directly**. Do **not** sum listing `est_payout` ourselves — the
> backend already applies exclusivity + de-duplication (e.g. same-carrier filtering)
> that a plain `SUM` can't replicate. So the "SUM vs MAX" question is resolved: use
> the backend field. Our current listings-join is an **interim stopgap** until the
> column lands. See §10.

### ⚠️ "Payout" is overloaded — read this

The warehouse uses "payout" on the **buyer (downstream) side**, which is the opposite
of how a reseller normally says it. Confirmed against the real schema:

- On `lead_pings`: **`bid`** = what we pay the **partner** (our cost); **`rev`** = our
  realized revenue.
- On `lead_ping_listings`: **`payout`** / **`est_payout`** are **buyer-side** — money
  coming *to us* (realized / expected revenue *from that buyer*). They are **not** what
  we pay the partner.

So when this doc or the team says "payout," check the side: partner-side payout = our
`bid`; listing `payout`/`est_payout` = buyer revenue.

### Where each concept lives (concept → column)

| Concept | Table.column |
|---|---|
| Our bid to the partner (cost) | `lead_pings.bid` |
| Realized revenue | `lead_pings.rev` (≈ Σ `lead_ping_listings.payout` over accepted) |
| Expected revenue (ceiling input) | **`lead_pings.expected_revenue`** (backend field, being added) — *interim:* Σ `lead_ping_listings.est_payout` per ping |
| Won the lead (partner accepted) | `lead_pings.won` |
| Downstream accept/reject | `lead_ping_listings.post_accepted` |
| Listing selected / excluded / deduped | `lead_ping_listings.selected` / `excluded` / `de_duped` |
| CM target lever (dumb 10/25/50/75) | `lead_pings.bidding_strategy_id` |
| Exclusive vs shared lead | `lead_ping_listings.exclusive` |
| Listing counts | `lead_pings.total_listings` / `accepted_listings` |

The `smarthub.data.models` ORM mirrors these tables; `leads_with_expected_revenue_select`
aggregates `est_payout` per ping (see its docstring for the selected-only assumption).

**Resolved 1 Jul 2026 (see §10):** expected revenue → use the backend
`lead_pings.expected_revenue` field (don't aggregate listings). `bpfm_score` = a
buyer **pacing** metric (daily-quota), backend-handled, not a model input.
**Still open:** `bid_to_use`; and confirming `accepted` (resold ≥1 buyer) vs `won`
(partner accepted our bid) vs `accepted_listings`.

---

## 5. Why leads are more certain than calls

- **Leads:** when a ping arrives there's a concrete, qualified lead and we already
  know who would buy it and for how much → high certainty, high margins.
- **Calls:** the contact center may never reach the consumer (no answer, bad number,
  not actually shopping) or the consumer declines the transfer → much less certain.
- **Clicks:** essentially playing the averages. Note: in the clicks world there's no
  cost unless there's revenue; in the leads world, bids are compressed because the
  amounts are real expenses we incur whether or not we end up profiting.

---

## 6. The modeling problem (what Anton is for)

> This section was corrected after feedback from Vinaya and Kiran (Slack thread,
> Jun 24, 2026). The earlier draft framed expected revenue as "the ceiling" and gave
> Anton two either/or modes; both were wrong. The accurate framing is below.

Anton's job is to choose the **bid per ping that maximizes profit**, working within
a budgetary sandbox whose bounds it must also **discover**.

### The bounds

- **Upper bound (max bid)** is set by **expected revenue together with the target
  CM**, not by expected revenue alone. Since `CM = (revenue − bid) / revenue`, the
  most we'd pay while still hitting a target CM is:

  ```text
  max_bid = expected_revenue × (1 − target_CM)
  ```

- **Lower bound (min bid)** is a **partner-side floor**, *if one exists* — e.g.
  Launch Potato auto-rejects every bid below $10, so bidding under the floor wins
  nothing. The floor sets the *minimum* sensible bid.

- **Bounds may not exist.** The floor/ceiling/win-rate curves we looked at "may or
  may not exist for a partner / lead type / etc." Identifying *whether* bounds exist
  and *where* they sit is part of Anton's job, not a given.

### The objective: one optimization, not two modes

Profit is driven by **both win rate and CM**, which trade off against each other:

- Bid **higher** within the range → win rate ↑, but CM ↓.
- Bid **lower** → CM ↑, but win rate ↓.

So there is a **single** objective: find the bid in `[floor, ceiling]` that maximizes
overall profit — conceptually, maximize expected profit per ping:

```text
expected_profit(bid) ≈ P(win | bid) × (expected_revenue − bid)
```

Kiran's phrasing — *"optimize for CM without compromising win rate"* — is the same
idea. (The earlier "maximize win rate **or** maximize CM" framing was a false
either/or.)

### Finding the edges

Within the range we still want the **"shelves" / edges**: price points where a small
bid change barely moves win rate. If win rate at $9.50 equals win rate at $10, bid
$9.50 and keep the extra margin. Discovering where those edges live requires actively
probing the market — see §7.

---

## 7. Exploration and recency

Two requirements that the earlier draft missed entirely (raised by Kiran in the
Jun 24 thread).

### Explore around the optimum (don't just exploit it)

Anton should not only bid the current best estimate. It must bid with **deliberate
variability** — occasionally a little above and below the optimum — to gather real
data on the **shape of the market** at different price points. That probing is the
only way to learn where the edges/shelves live and keep future bids well-informed.
(This is the explore/exploit trade-off; the existing `dumb 10/25/50/75 ± 10%`
strategies probe in a fixed way, but Anton needs to probe *around its own optimum*.)

### Weight recent data; define "recent"; define the cold-start fallback

The market changes — supply and demand shift — so old data goes stale.

- Use **recent** learnings on a **rolling basis**; don't over-trust old data.
- **"Recent" must be explicitly defined** as a configurable window (e.g. a rolling
  N-day lookback), **not buried in the code**. It should be a named config value.
- When there is **no recent data** (cold start, new partner/lead type), the bidding
  pattern must be **explicitly articulated** so behavior stays organized rather than
  chaotic — e.g. a defined exploration schedule or fallback bid.

---

## 8. Attributes that matter

Significant signals (curve shape changes with each):

- **Currently insured** — insured consumers fetch higher buyer prices.
- **State** — e.g. NY vs TX differ, due to regulation and how strong our buyer
  network is in that state.
- **Lead type** (`lead_type_id`) — auto vs home behave differently.

Guidance: pick a *handful* of meaningful attributes; don't overfit on all the noisy
ones. Keep Anton explainable.

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **Ping** | A lead offer from a partner ("what will you pay?"), or the act of checking a buyer's interest. |
| **Post** | Sending the actual lead data downstream to a buyer. |
| **Partner / account** | Upstream producer who sells us leads (Healthy Labs, Launch Potato, Malva…). |
| **Buyer** | Downstream agent we resell to (on Insurance.io, SF Pro, BPIO). |
| **Campaign** | A partner's lead stream (`campaign_id`); a partner can have several. |
| **Won** | The partner accepted our bid; we now hold the lead. |
| **Accept / reject** | Whether a downstream buyer takes the lead we post. |
| **Bid** | What we offer the partner for the lead. |
| **Expected revenue** | Forecast revenue, reject-discounted. Use the backend **`lead_pings.expected_revenue`** field (being added); *interim* Σ `est_payout`. |
| **BPFM score** | A buyer **pacing** metric (progress to daily quota); >90% → excluded from the auction to protect O&O. Backend-handled. |
| **O&O vs SmartHub lead** | O&O = SmartFinancial's own campaigns (cost already realized, prioritized); SmartHub = third-party arbitrage leads. |
| **Cannibalization KPI** | Flags selling a SmartHub lead to a buyer who'd otherwise take an O&O lead (undercuts our own business). |
| **Realized / measured revenue** | What we actually made (`lead_pings.rev`); blank if the lead was never bought. |
| **Payout (⚠ overloaded)** | Partner-side: our cost = `lead_pings.bid`. Listing-side: `lead_ping_listings.payout`/`est_payout` = buyer revenue *to us*. |
| **Profit** | Realized revenue − our cost = `lead_pings.rev − lead_pings.bid` (when won). |
| **Contribution margin (CM)** | Profit ÷ revenue; also the lever in bidding strategies (dumb 10/25/50/75). |
| **Win rate** | Share of bids that win the lead, often measured across price points. |
| **Buyer distribution** | Our network of downstream buyers; its strength varies by state and lead type. |
| **Shelf / edge** | A price point where win rate barely moves — bid at the cheaper side and keep the margin. |
| **Target CM** | The contribution margin we aim to keep; with expected revenue it sets the max bid. |
| **Ceiling (max bid)** | `expected_revenue × (1 − target_CM)` — the highest bid that still hits the CM target. |
| **Floor (min bid)** | A partner-imposed minimum below which bids are auto-rejected (e.g. Launch Potato's $10); may not exist. |
| **Bounds / sandbox** | The `[floor, ceiling]` range Anton bids within; their existence and location must be discovered. |
| **Exploration (explore/exploit)** | Deliberately bidding around the optimum to learn the market's shape at other price points. |
| **Recency window** | The rolling lookback that defines "recent" data; a named config value, not hard-coded. |

---

## 10. Update — DS meeting, 1 Jul 2026

New decisions and knowledge (Nimesh was absent; captured from notes):

- **Expected revenue → a backend field.** Devs will add `lead_pings.expected_revenue`;
  Anton must **use it directly**. Don't sum listing `est_payout` — the backend already
  applies exclusivity + de-duplication (e.g. same-carrier filtering). Resolves MAX-vs-SUM.
- **`bpfm_score` = buyer pacing metric** (daily quota); >90% → excluded from auction to
  prioritise O&O. Backend concern, not a model input.
- **O&O vs SmartHub + cannibalization KPI** (see glossary) — SmartHub is arbitrage;
  don't undercut owned-and-operated leads.
- **Data quality: defaults are noise.** Providers send default values regardless of the
  real consumer (e.g. `marital_status` always "single", multi-vehicle defaults). So:
  - **treat missing data as a signal** — do *not* fill with averages;
  - build **secondary "completeness" / source-quality features** that measure how
    complete/reliable each lead and source is, instead of trusting raw fields.
- **Architecture:** at serve time Anton gets the lead via an **API ping**, not a DB
  query. **Staging is decoupled from production** — async, "fire-and-forget",
  **predict-only (no bids placed)**. **Configuration via a UI**, no hidden/hardcoded/
  script params.
- **Model objective (reaffirmed):** predict **win rate → optimise profit**; find the
  ceiling that holds CM but **bid lower when win rate is unchanged**; add a **feedback
  loop** so it converges automatically (current Anton failed to). Must be **≥ the
  current system**, maintainable, transparent. **MVP by end of Q3.**
- **Exploratory bidding** for cold-start (new sources with no history).
- **Secondary source-quality metrics:** call revenue, "unity" (SMS revenue), conversion
  rate — used to phase out low-quality partners.
