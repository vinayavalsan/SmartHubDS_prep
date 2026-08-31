"""Tests for the ORM models and query builders."""

from datetime import datetime

import pandas as pd
from sqlalchemy.dialects import postgresql

from smarthub.data_pull.models import (
    LEADS_COLUMNS,
    LeadPing,
    LeadPingListing,
    coerce_leads_dtypes,
    leads_select,
    leads_with_expected_revenue_select,
)


def _compiled_sql(stmt) -> str:
    """Compile a statement to PostgreSQL-dialect SQL (Redshift-compatible)."""
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_leads_select_targets_lead_pings_table():
    """leads_select queries lead_pings and selects the expected columns."""
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "FROM lead_pings" in sql
    for col in LEADS_COLUMNS:
        assert col.key in sql


def test_leads_select_applies_date_range():
    """leads_select applies the created_at date-range bounds."""
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "lead_pings.created_at >=" in sql
    assert "lead_pings.created_at <" in sql
    assert "2026-06-07" in sql and "2026-06-20" in sql


def test_leads_select_lead_type_filter():
    """leads_select applies the current one-or-many lead-type filter."""
    sql_all = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "lead_pings.lead_type_id IN" not in sql_all

    sql_auto = _compiled_sql(
        leads_select(
            "2026-06-07 00:00:00",
            "2026-06-20 00:00:00",
            lead_type_ids=[6],
        )
    )
    assert "lead_pings.lead_type_id IN (6)" in sql_auto

    sql_both = _compiled_sql(
        leads_select(
            "2026-06-07 00:00:00",
            "2026-06-20 00:00:00",
            lead_type_ids=[6, 1],
        )
    )
    assert "lead_pings.lead_type_id IN (6, 1)" in sql_both


def test_expected_revenue_query_lead_type_filter():
    """Expected-revenue query applies one-or-many lead-type filters."""
    sql = _compiled_sql(
        leads_with_expected_revenue_select(
            "2026-06-07 00:00:00",
            "2026-06-20 00:00:00",
            lead_type_ids=[1],
        )
    )
    assert "lead_pings.lead_type_id IN (1)" in sql
    assert "lead_pings.exp_rev" in sql
    assert "lead_ping_listings" not in sql


def test_leads_select_excludes_pii_columns():
    """leads_select never selects PII columns."""
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    for pii in ("ip_address", "user_agent", "date_of_birth", "trusted_form_token"):
        assert f"lead_pings.{pii}" not in sql


def test_leads_select_accepts_datetime_objects():
    """leads_select accepts datetime objects as bounds."""
    stmt = leads_select(datetime(2026, 6, 7), datetime(2026, 6, 20))
    assert "FROM lead_pings" in _compiled_sql(stmt)


def test_expected_revenue_query_uses_lead_pings_exp_rev():
    """Expected revenue comes from lead_pings.exp_rev with no listing join."""
    sql = _compiled_sql(
        leads_with_expected_revenue_select("2026-06-07 00:00:00", "2026-06-20 00:00:00")
    )
    assert "lead_pings.exp_rev" in sql
    assert "lead_ping_listings" not in sql
    assert "LEFT OUTER JOIN" not in sql


def test_expected_revenue_query_selected_only_is_compatibility_only():
    """selected_only no longer changes SQL because revenue is on lead_pings."""
    sql_default = _compiled_sql(
        leads_with_expected_revenue_select(
            "2026-06-07 00:00:00",
            "2026-06-20 00:00:00",
        )
    )
    sql_all = _compiled_sql(
        leads_with_expected_revenue_select(
            "2026-06-07 00:00:00",
            "2026-06-20 00:00:00",
            selected_only=False,
        )
    )
    assert sql_default == sql_all
    assert "lead_pings.exp_rev" in sql_all
    assert "lead_ping_listings" not in sql_all


def test_listing_fk_points_to_lead_pings():
    """LeadPingListing.lead_ping_id foreign-keys to LeadPing.id."""
    fks = list(LeadPingListing.__table__.c.lead_ping_id.foreign_keys)
    assert fks and fks[0].column is LeadPing.__table__.c.id


def test_coerce_leads_dtypes_stable_schema():
    """coerce_leads_dtypes assigns stable dtypes, keeping all-null as string."""
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "home_property_type": [None, None],
            "bid": ["12.00", "25.00"],
            "age": [44, None],
            "created_at": ["2026-06-20 01:00:00", "2026-06-20 02:00:00"],
        }
    )
    out = coerce_leads_dtypes(df)
    assert out["home_property_type"].dtype == "string"
    assert str(out["id"].dtype) == "Int64"
    assert out["bid"].dtype == "float64"
    assert str(out["age"].dtype) == "Int64"
    assert "datetime64" in str(out["created_at"].dtype)

    df2 = df.assign(home_property_type=["Single Family Home", "Condominium"])
    assert coerce_leads_dtypes(df2)["home_property_type"].dtype == "string"
