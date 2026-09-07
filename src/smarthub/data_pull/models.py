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

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Select,
    String,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from . import query_builder

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
DateLike = Union[str, datetime]
LeadTypeIds = int | Sequence[int] | None

# String tokens (varchar booleans) that mean "yes" in this warehouse.
TRUE_TOKEN = "true"


class Base(DeclarativeBase):
    """Declarative base for all SmartHub ORM models."""


class LeadPing(Base):
    """A row in ``public.lead_pings`` — one ping (lead offer) and our bid back.

    ``account_id`` here is the **upstream partner**; ``bid`` is what we pay the
    partner; ``rev`` is our realized revenue; ``exp_rev`` is our expected revenue.
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
    pst_hour: Mapped[int | None] = mapped_column(Integer)  # Pacific hour 0–23
    exp_rev: Mapped[float | None] = mapped_column(Numeric(10, 2))  # backend R


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


# Curated, non-PII column set pulled for analysis, DRIVEN BY the raw-field
# registry (field_registry.RAW_FIELD_REGISTRY) — the single source of truth for
# which raw fields the pull extracts and in what order. PII / opaque columns
# (token, uid, ip_address, user_agent, jornaya_lead_id, trusted_form_token,
# submission_url, date_of_birth) are excluded by not being registered. Each
# registered field name resolves to its ORM column here, preserving order.
LEADS_COLUMNS: Sequence = tuple(
    getattr(LeadPing, name) for name in query_builder.leads_column_names()
)


def _as_datetime(value: DateLike) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, _DT_FORMAT)


def coerce_leads_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force each ``lead_pings`` column to its ORM-declared dtype.

    Makes the on-disk schema stable regardless of nulls: a string column that
    happens to be all-null in one pull is still typed as text, so a later pull
    with real strings (e.g. ``home_property_type = 'Single Family Home'``)
    won't hit a DuckDB type-conversion error. Columns not in the model (e.g.
    the expected-revenue join outputs) are left untouched.

    Inputs
    ------
    df : pandas.DataFrame
        Pulled leads frame with raw column values.

    Returns
    -------
    pandas.DataFrame
        A copy with model columns coerced to their declared dtypes.
    """
    out = df.copy()
    for column in LeadPing.__table__.columns:
        name = column.name
        if name not in out.columns:
            continue
        col_type = column.type
        if isinstance(col_type, String):
            out[name] = out[name].astype("string")
        elif isinstance(col_type, Integer):  # covers BigInteger
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("Int64")
        elif isinstance(col_type, Numeric):
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
        elif isinstance(col_type, (Date, DateTime)):
            out[name] = pd.to_datetime(out[name], errors="coerce")

    # Canonical revenue aliases come directly from lead_pings.
    if "exp_rev" in out.columns:
        out["expected_revenue"] = pd.to_numeric(out["exp_rev"], errors="coerce")
    if "rev" in out.columns:
        out["realized_revenue"] = pd.to_numeric(out["rev"], errors="coerce").fillna(0.0)

    # Canonical realized-economics rule:
    # a cost is incurred only when the lead was sold to at least one buyer.
    #
    #     sold     = 1 when accepted_listings > 0, else 0
    #     bid_cost = sold * bid
    #     profit   = realized_revenue - bid_cost
    if {"accepted_listings", "rev", "bid"}.issubset(out.columns):
        accepted_listings = pd.to_numeric(out["accepted_listings"], errors="coerce")
        realized_revenue = pd.to_numeric(out["rev"], errors="coerce").fillna(0.0)
        bid = pd.to_numeric(out["bid"], errors="coerce")
        out["sold"] = accepted_listings.fillna(0).gt(0).astype("Int64")
        out["bid_cost"] = out["sold"].astype("float64") * bid
        out["profit"] = realized_revenue - out["bid_cost"]

    return out


def leads_select(
    min_created_at: DateLike,
    max_created_at: DateLike,
    lead_type_ids: LeadTypeIds = None,
) -> Select:
    """Build the base leads query: selected columns within a created_at range.

    Inputs
    ------
    min_created_at : str | datetime
        Inclusive lower bound for ``created_at`` (``YYYY-MM-DD HH:MM:SS``
        string or a ``datetime``).
    max_created_at : str | datetime
        Exclusive upper bound for ``created_at``.
    lead_type_ids : int | Sequence[int] | None
        Restrict to one or more lead types. ``None`` applies no lead-type filter.

    Returns
    -------
    sqlalchemy.Select
        Ordered SELECT over the curated leads columns.
    """
    lower, upper = _as_datetime(min_created_at), _as_datetime(max_created_at)
    stmt = (
        select(*LEADS_COLUMNS)
        .where(LeadPing.created_at >= lower)
        .where(LeadPing.created_at < upper)
    )
    if lead_type_ids is not None:
        ids = [lead_type_ids] if isinstance(lead_type_ids, int) else list(lead_type_ids)
        stmt = stmt.where(LeadPing.lead_type_id.in_(ids))
    return stmt.order_by(LeadPing.created_at)


def leads_with_expected_revenue_select(
    min_created_at: DateLike,
    max_created_at: DateLike,
    selected_only: bool = True,
    lead_type_ids: LeadTypeIds = None,
) -> Select:
    """Build the leads query used by the pull pipeline.

    ``expected_revenue`` is sourced from ``lead_pings.exp_rev`` during dtype
    coercion. ``selected_only`` is retained for call-site compatibility but no
    listing join is required.
    """
    del selected_only
    return leads_select(
        min_created_at=min_created_at,
        max_created_at=max_created_at,
        lead_type_ids=lead_type_ids,
    )
