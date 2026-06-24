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

**Expected revenue & the ceiling (key for Anton):** a *separate* table stores
**expected revenue** — what we think we'll make, **already discounted by how often
buyers reject**. Example: top buyer would pay $78 but only accepts 20% of the time →
expected revenue ≈ $15.90. Anton will be handed expected revenue as a **ceiling** and
must bid under it. This is why revenue can look much larger than the bid, and why
margins look fat — the reject rate is already baked in, so you don't model it
separately.

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

Find the right **bid per ping**. Illustrated by the **Launch Potato** example:

- Launch Potato is a competitor and sets an **artificial floor** at $10 — every bid
  below $10 is auto-rejected (win rate = 0% below $10).
- The win-rate curve flattens around **$15**, suggesting competitors have an
  artificial **ceiling** there; past $15 win rate shoots up steeply.
- **Takeaway:** if the budget is $19, don't bid $19 — bid ~$15.25, win the same lead,
  and keep the extra ~$4 of margin.

So the objective is to find the **"shelves" / edges** in the win-rate-vs-price data:
price points where a small bid change causes a big win-rate jump, because that's where
our buyer distribution is beating the competition.

Two possible **modes** for Anton:
1. **Maximize win rate** — bid high enough to win, even at the cost of margin.
2. **Maximize contribution margin** — accept losing a few leads to keep margins high.

The real question is *what to bid given a ceiling* — not just always bidding the
ceiling.

---

## 7. Attributes that matter

Significant signals (curve shape changes with each):

- **Currently insured** — insured consumers fetch higher buyer prices.
- **State** — e.g. NY vs TX differ, due to regulation and how strong our buyer
  network is in that state.
- **Lead type** (`lead_type_id`) — auto vs home behave differently.

Guidance: pick a *handful* of meaningful attributes; don't overfit on all the noisy
ones. Keep Anton explainable.

---

## 8. Glossary

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
| **Expected revenue** | Forecast revenue, already discounted by buyer reject rates (a separate table). |
| **Realized / measured revenue** | What we actually made; blank if the lead was never bought. |
| **Payout** | What we pay the partner (only when we resell). |
| **Profit** | Realized revenue − payout. |
| **Contribution margin (CM)** | Profit ÷ revenue; also the lever in bidding strategies (dumb 10/25/50/75). |
| **Win rate** | Share of bids that win the lead, often measured across price points. |
| **Buyer distribution** | Our network of downstream buyers; its strength varies by state and lead type. |
| **Shelf / edge** | A price point where win rate jumps sharply — a target for smart bidding. |
