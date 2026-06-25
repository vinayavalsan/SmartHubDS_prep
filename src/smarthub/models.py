"""SQLAlchemy ORM models and query builders for the SmartHub warehouse.

Queries are expressed with the ORM / SQLAlchemy expression language rather than
raw SQL strings: this keeps column names in one place, makes bind parameters
automatic (no string interpolation), and lets the Redshift dialect generate the
final SQL.

The models mirror the real Redshift schema (``public.lead_pings`` and
``public.lead_ping_listings``). See CONTEXT.md §4 for what the columns mean.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence, Union

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Select,
    String,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
DateLike = Union[str, datetime]

# String tokens (varchar booleans) that mean "yes" in this warehouse.
TRUE_TOKEN = "true"


class Base(DeclarativeBase):
    """Declarative base for all SmartHub ORM models."""


class LeadPing(Base):
    """A row in ``public.lead_pings`` — one ping (lead offer) and our bid back.

    ``account_id`` here is the **upstream partner**; ``bid`` is what we pay the
    partner; ``rev`` is our realized revenue. Expected revenue is *not* on this
    table — it is derived from ``lead_ping_listings.est_payout`` (see below).
    """

    __tablename__ = "lead_pings"

    # --- keys / dimensions ---
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(BigInteger)  # partner
    campaign_id: Mapped[int | None] = mapped_column(BigInteger)
    lead_type_id: Mapped[int | None] = mapped_column(BigInteger)
    source_type_id: Mapped[int | None] = mapped_column(BigInteger)
    bidding_strategy_id: Mapped[int | None] = mapped_column(Integer)
    traffic_tier: Mapped[str | None] = mapped_column(String(24000))

    # --- bid / outcome ---
    total_listings: Mapped[int | None] = mapped_column(Integer)
    accepted_listings: Mapped[int | None] = mapped_column(Integer)
    bid: Mapped[float | None] = mapped_column(Numeric(12, 2))
    rev: Mapped[float | None] = mapped_column(Numeric(12, 2))
    won: Mapped[str | None] = mapped_column(String(15))
    accepted: Mapped[str | None] = mapped_column(String(15))

    # --- quality / errors ---
    response_ms: Mapped[int | None] = mapped_column(Integer)
    erred: Mapped[str | None] = mapped_column(String(15))
    error_reason_id: Mapped[int | None] = mapped_column(Integer)

    # --- consumer attributes ---
    zip: Mapped[str | None] = mapped_column(String(24000))
    city: Mapped[str | None] = mapped_column(String(24000))
    state: Mapped[str | None] = mapped_column(String(24000))
    device_type: Mapped[str | None] = mapped_column(String(24000))
    insured: Mapped[str | None] = mapped_column(String(15))
    current_carrier: Mapped[str | None] = mapped_column(String(24000))
    continuous_coverage_months: Mapped[int | None] = mapped_column(Integer)
    expiration_date: Mapped[datetime | None] = mapped_column(Date)
    military_affiliation: Mapped[str | None] = mapped_column(String(15))
    credit: Mapped[str | None] = mapped_column(String(24000))
    pnc_bundle: Mapped[str | None] = mapped_column(String(15))
    home_owner: Mapped[str | None] = mapped_column(String(15))
    gender: Mapped[str | None] = mapped_column(String(24000))
    marital_status: Mapped[str | None] = mapped_column(String(24000))
    num_drivers: Mapped[int | None] = mapped_column(Integer)
    num_vehicles: Mapped[int | None] = mapped_column(Integer)
    dui: Mapped[str | None] = mapped_column(String(15))
    sr22_required: Mapped[str | None] = mapped_column(String(15))
    age: Mapped[int | None] = mapped_column(Integer)
    num_auto_violations: Mapped[int | None] = mapped_column(Integer)
    num_auto_claims: Mapped[int | None] = mapped_column(Integer)
    num_auto_accidents: Mapped[int | None] = mapped_column(Integer)
    num_home_claims: Mapped[int | None] = mapped_column(Integer)
    home_property_type: Mapped[str | None] = mapped_column(String(24000))
    num_dependents: Mapped[int | None] = mapped_column(Integer)
    health_conditions: Mapped[str | None] = mapped_column(String(15))
    household_income: Mapped[str | None] = mapped_column(String(24000))
    life_coverage_type: Mapped[str | None] = mapped_column(String(24000))
    life_coverage_amount: Mapped[str | None] = mapped_column(String(24000))
    naics_code: Mapped[str | None] = mapped_column(String(24000))
    sic_code: Mapped[str | None] = mapped_column(String(24000))
    num_employees: Mapped[str | None] = mapped_column(String(24000))
    annual_revenue: Mapped[str | None] = mapped_column(String(24000))
    comm_required_coverages: Mapped[str | None] = mapped_column(String(32768))

    # --- PII / opaque (not pulled for modeling) ---
    token: Mapped[str | None] = mapped_column(String(32768))
    uid: Mapped[str | None] = mapped_column(String(24000))
    ip_address: Mapped[str | None] = mapped_column(String(24000))
    user_agent: Mapped[str | None] = mapped_column(String(24000))
    jornaya_lead_id: Mapped[str | None] = mapped_column(String(24000))
    trusted_form_token: Mapped[str | None] = mapped_column(String(24000))
    submission_url: Mapped[str | None] = mapped_column(String(32768))
    date_of_birth: Mapped[datetime | None] = mapped_column(Date)

    # --- time ---
    lead_created_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    pst_date: Mapped[datetime | None] = mapped_column(Date)


class LeadPingListing(Base):
    """A row in ``public.lead_ping_listings`` — one downstream buyer listing for
    a ping.

    ``account_id`` here is the **buyer** (not the partner). ``est_payout`` is the
    expected amount the buyer pays *us* (i.e. expected revenue), and ``payout``
    is the realized amount. ``post_accepted`` is the downstream accept/reject.
    """

    __tablename__ = "lead_ping_listings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    lead_ping_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("lead_pings.id"), nullable=False
    )
    bid_id: Mapped[str | None] = mapped_column(String(24000))
    network: Mapped[str | None] = mapped_column(String(24000))
    account_id: Mapped[int | None] = mapped_column(BigInteger)  # buyer
    campaign_id: Mapped[int | None] = mapped_column(BigInteger)
    exclusive: Mapped[str | None] = mapped_column(String(15))
    carrier_id: Mapped[int | None] = mapped_column(Integer)
    payout: Mapped[float | None] = mapped_column(Numeric(12, 2))  # realized rev
    est_payout: Mapped[float | None] = mapped_column(Numeric(12, 2))  # expected rev
    bpfm_score: Mapped[float | None] = mapped_column(Numeric(7, 2))
    selected: Mapped[str | None] = mapped_column(String(15))
    de_duped: Mapped[str | None] = mapped_column(String(15))
    excluded: Mapped[str | None] = mapped_column(String(15))
    posted: Mapped[str | None] = mapped_column(String(15))
    post_accepted: Mapped[str | None] = mapped_column(String(15))
    pp_ping_id: Mapped[str | None] = mapped_column(String(24000))
    pp_bid_id: Mapped[str | None] = mapped_column(String(24000))
    bid_to_use: Mapped[float | None] = mapped_column(Numeric(12, 2))
    io_ping_id: Mapped[str | None] = mapped_column(String(765))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


# Curated, non-PII column set pulled for analysis (in a stable order).
# PII / opaque columns (token, uid, ip_address, user_agent, jornaya_lead_id,
# trusted_form_token, submission_url, date_of_birth) are deliberately excluded.
LEADS_COLUMNS: Sequence = (
    LeadPing.id,
    LeadPing.created_at,
    LeadPing.account_id,
    LeadPing.campaign_id,
    LeadPing.lead_type_id,
    LeadPing.source_type_id,
    LeadPing.bidding_strategy_id,
    LeadPing.traffic_tier,
    LeadPing.total_listings,
    LeadPing.accepted_listings,
    LeadPing.bid,
    LeadPing.rev,
    LeadPing.won,
    LeadPing.accepted,
    LeadPing.erred,
    LeadPing.error_reason_id,
    LeadPing.response_ms,
    LeadPing.zip,
    LeadPing.city,
    LeadPing.state,
    LeadPing.device_type,
    LeadPing.insured,
    LeadPing.current_carrier,
    LeadPing.continuous_coverage_months,
    LeadPing.military_affiliation,
    LeadPing.credit,
    LeadPing.pnc_bundle,
    LeadPing.home_owner,
    LeadPing.gender,
    LeadPing.marital_status,
    LeadPing.num_drivers,
    LeadPing.num_vehicles,
    LeadPing.dui,
    LeadPing.sr22_required,
    LeadPing.age,
    LeadPing.num_auto_violations,
    LeadPing.num_auto_claims,
    LeadPing.num_auto_accidents,
    LeadPing.num_home_claims,
    LeadPing.home_property_type,
    LeadPing.num_dependents,
    LeadPing.health_conditions,
    LeadPing.household_income,
    LeadPing.life_coverage_type,
    LeadPing.life_coverage_amount,
    LeadPing.naics_code,
    LeadPing.sic_code,
    LeadPing.num_employees,
    LeadPing.annual_revenue,
    LeadPing.lead_created_at,
    LeadPing.expiration_date,
    LeadPing.pst_date,
)


def _as_datetime(value: DateLike) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, _DT_FORMAT)


def leads_select(min_created_at: DateLike, max_created_at: DateLike) -> Select:
    """Build the base leads query: selected columns where created_at is in range.

    Bounds may be ``datetime`` objects or ``"YYYY-MM-DD HH:MM:SS"`` strings.
    """
    lower, upper = _as_datetime(min_created_at), _as_datetime(max_created_at)
    return (
        select(*LEADS_COLUMNS)
        .where(LeadPing.created_at >= lower)
        .where(LeadPing.created_at < upper)
        .order_by(LeadPing.created_at)
    )


def expected_revenue_subquery(selected_only: bool = True):
    """Per-ping expected revenue aggregated from the listings.

    Sums ``est_payout`` (expected buyer payment = expected revenue) and
    ``payout`` (realized) per ``lead_ping_id``.

    ASSUMPTION (confirm with the team): expected revenue is the sum over
    ``selected = 'true'`` listings. Set ``selected_only=False`` to sum over all
    listings instead. See CONTEXT.md §4 open questions.
    """
    stmt = (
        select(
            LeadPingListing.lead_ping_id.label("lead_ping_id"),
            func.sum(LeadPingListing.est_payout).label("expected_revenue"),
            func.sum(LeadPingListing.payout).label("realized_payout"),
            func.count().label("num_selected_listings"),
        )
        .group_by(LeadPingListing.lead_ping_id)
    )
    if selected_only:
        stmt = stmt.where(LeadPingListing.selected == TRUE_TOKEN)
    return stmt.subquery("listing_expected_revenue")


def leads_with_expected_revenue_select(
    min_created_at: DateLike,
    max_created_at: DateLike,
    selected_only: bool = True,
) -> Select:
    """Leads query LEFT JOINed to per-ping expected revenue from the listings.

    Adds ``expected_revenue``, ``realized_payout`` and ``num_selected_listings``
    columns. Pings with no matching listings get NULLs (outer join).
    """
    lower, upper = _as_datetime(min_created_at), _as_datetime(max_created_at)
    subq = expected_revenue_subquery(selected_only)
    return (
        select(
            *LEADS_COLUMNS,
            subq.c.expected_revenue,
            subq.c.realized_payout,
            subq.c.num_selected_listings,
        )
        .outerjoin(subq, subq.c.lead_ping_id == LeadPing.id)
        .where(LeadPing.created_at >= lower)
        .where(LeadPing.created_at < upper)
        .order_by(LeadPing.created_at)
    )
