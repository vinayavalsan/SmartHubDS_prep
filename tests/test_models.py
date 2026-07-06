"""Tests for the ORM models and query builders."""

from datetime import datetime

from sqlalchemy.dialects import postgresql

import pandas as pd

from smarthub.data_pull.models import (
    LEADS_COLUMNS,
    LeadPing,
    LeadPingListing,
    coerce_leads_dtypes,
    leads_select,
    leads_with_expected_revenue_select,
)


def _compiled_sql(stmt) -> str:
    # Redshift speaks the PostgreSQL dialect; compiling against it confirms the
    # ORM produces valid SQL without needing a live database.
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_leads_select_targets_lead_pings_table():
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "FROM lead_pings" in sql
    for col in LEADS_COLUMNS:
        assert col.key in sql


def test_leads_select_applies_date_range():
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "lead_pings.created_at >=" in sql
    assert "lead_pings.created_at <" in sql
    assert "2026-06-07" in sql and "2026-06-20" in sql


def test_leads_select_lead_type_filter():
    sql_all = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "lead_pings.lead_type_id =" not in sql_all

    sql_auto = _compiled_sql(
        leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00", lead_type_id=6)
    )
    assert "lead_pings.lead_type_id = 6" in sql_auto


def test_expected_revenue_join_lead_type_filter():
    sql = _compiled_sql(
        leads_with_expected_revenue_select(
            "2026-06-07 00:00:00", "2026-06-20 00:00:00", lead_type_id=1
        )
    )
    assert "lead_pings.lead_type_id = 1" in sql
    assert "lead_ping_listings" in sql


def test_leads_select_excludes_pii_columns():
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    for pii in ("ip_address", "user_agent", "date_of_birth", "trusted_form_token"):
        assert f"lead_pings.{pii}" not in sql


def test_leads_select_accepts_datetime_objects():
    stmt = leads_select(datetime(2026, 6, 7), datetime(2026, 6, 20))
    assert "FROM lead_pings" in _compiled_sql(stmt)


def test_expected_revenue_join_query():
    sql = _compiled_sql(
        leads_with_expected_revenue_select("2026-06-07 00:00:00", "2026-06-20 00:00:00")
    )
    # joins the listings aggregate and exposes expected revenue
    assert "lead_ping_listings" in sql
    assert "sum(lead_ping_listings.est_payout)" in sql
    assert "expected_revenue" in sql
    assert "LEFT OUTER JOIN" in sql
    # default selected-only filter applied
    assert "lead_ping_listings.selected" in sql


def test_expected_revenue_join_all_listings():
    sql = _compiled_sql(
        leads_with_expected_revenue_select(
            "2026-06-07 00:00:00", "2026-06-20 00:00:00", selected_only=False
        )
    )
    assert "lead_ping_listings.selected" not in sql


def test_listing_fk_points_to_lead_pings():
    fks = list(LeadPingListing.__table__.c.lead_ping_id.foreign_keys)
    assert fks and fks[0].column is LeadPing.__table__.c.id


def test_coerce_leads_dtypes_stable_schema():
    # An all-null string column must still be typed as string (not numeric),
    # so a later pull with real strings won't break the DuckDB schema.
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "home_property_type": [None, None],   # all-null string col
            "bid": ["12.00", "25.00"],            # numeric-as-string
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

    # a later pull with real home strings coerces cleanly to the same dtype
    df2 = df.assign(home_property_type=["Single Family Home", "Condominium"])
    assert coerce_leads_dtypes(df2)["home_property_type"].dtype == "string"
