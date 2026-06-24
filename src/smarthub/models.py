"""SQLAlchemy ORM models and query builders for the SmartHub warehouse.

Queries are expressed with the ORM / SQLAlchemy expression language rather than
raw SQL strings: this keeps column names in one place, makes bind parameters
automatic (no string interpolation), and lets the Redshift dialect generate the
final SQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence, Union

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    Numeric,
    Select,
    String,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
DateLike = Union[str, datetime]


class Base(DeclarativeBase):
    """Declarative base for all SmartHub ORM models."""


class LeadPing(Base):
    """A row in the ``lead_pings`` table (the raw lead offers from partners)."""

    __tablename__ = "lead_pings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    lead_type_id: Mapped[int | None] = mapped_column(Integer)
    state: Mapped[str | None] = mapped_column(String)
    zip: Mapped[str | None] = mapped_column(String)
    age: Mapped[int | None] = mapped_column(Integer)
    num_vehicles: Mapped[int | None] = mapped_column(Integer)
    num_drivers: Mapped[int | None] = mapped_column(Integer)
    insured: Mapped[str | None] = mapped_column(String)
    campaign_id: Mapped[int | None] = mapped_column(Integer)
    num_auto_violations: Mapped[int | None] = mapped_column(Integer)
    num_auto_accidents: Mapped[int | None] = mapped_column(Integer)
    continuous_coverage_months: Mapped[int | None] = mapped_column(Integer)
    home_owner: Mapped[str | None] = mapped_column(String)
    lead_created_at: Mapped[datetime | None] = mapped_column(DateTime)
    dui: Mapped[str | None] = mapped_column(String)
    won: Mapped[str | None] = mapped_column(String)
    bid: Mapped[float | None] = mapped_column(Numeric)
    rev: Mapped[float | None] = mapped_column(Numeric)


# The columns pulled for analysis, in a stable order.
LEADS_COLUMNS: Sequence = (
    LeadPing.id,
    LeadPing.created_at,
    LeadPing.lead_type_id,
    LeadPing.state,
    LeadPing.zip,
    LeadPing.age,
    LeadPing.num_vehicles,
    LeadPing.num_drivers,
    LeadPing.insured,
    LeadPing.campaign_id,
    LeadPing.num_auto_violations,
    LeadPing.num_auto_accidents,
    LeadPing.continuous_coverage_months,
    LeadPing.home_owner,
    LeadPing.lead_created_at,
    LeadPing.dui,
    LeadPing.won,
    LeadPing.bid,
    LeadPing.rev,
)


def _as_datetime(value: DateLike) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, _DT_FORMAT)


def leads_select(min_created_at: DateLike, max_created_at: DateLike) -> Select:
    """Build the leads query: selected columns where created_at is in range.

    Bounds may be ``datetime`` objects or ``"YYYY-MM-DD HH:MM:SS"`` strings.
    Returns a SQLAlchemy ``Select`` (compiled to SQL by the engine's dialect).
    """
    lower = _as_datetime(min_created_at)
    upper = _as_datetime(max_created_at)
    return (
        select(*LEADS_COLUMNS)
        .where(LeadPing.created_at >= lower)
        .where(LeadPing.created_at < upper)
        .order_by(LeadPing.created_at)
    )
