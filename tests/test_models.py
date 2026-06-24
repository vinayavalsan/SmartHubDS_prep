"""Tests for the ORM models and query builder."""

from datetime import datetime

from sqlalchemy.dialects import postgresql

from smarthub.models import LEADS_COLUMNS, LeadPing, leads_select


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
    # all requested columns appear in the projection
    for col in LEADS_COLUMNS:
        assert col.key in sql


def test_leads_select_applies_date_range():
    sql = _compiled_sql(leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00"))
    assert "lead_pings.created_at >=" in sql
    assert "lead_pings.created_at <" in sql
    assert "2026-06-07" in sql and "2026-06-20" in sql


def test_leads_select_accepts_datetime_objects():
    stmt = leads_select(
        datetime(2026, 6, 7, 0, 0, 0), datetime(2026, 6, 20, 0, 0, 0)
    )
    assert "FROM lead_pings" in _compiled_sql(stmt)


def test_no_raw_sql_string_in_query_builder():
    # leads_select returns a SQLAlchemy Select, not a string.
    stmt = leads_select("2026-06-07 00:00:00", "2026-06-20 00:00:00")
    assert not isinstance(stmt, str)
    assert stmt.get_final_froms()[0].name == LeadPing.__tablename__
