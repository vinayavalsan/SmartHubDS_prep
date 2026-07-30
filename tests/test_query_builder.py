"""Golden test: the registry-driven pull columns match the original pull.

The raw-field registry (field_registry) now drives which columns the pull
selects, resolved to ORM columns by models.LEADS_COLUMNS. This pins the exact
column set + order to the list the hand-maintained tuple used to produce, so the
refactor is provably behaviour-preserving.
"""

from smarthub.data_pull import field_registry as fr
from smarthub.data_pull import models
from smarthub.data_pull import query_builder as qb

# The authoritative 54-column order the pull produced before the registry
# refactor (captured from the previous models.LEADS_COLUMNS tuple).
EXPECTED_PULL_COLUMNS = [
    "id",
    "created_at",
    "account_id",
    "campaign_id",
    "lead_type_id",
    "source_type_id",
    "bidding_strategy_id",
    "traffic_tier",
    "total_listings",
    "accepted_listings",
    "bid",
    "rev",
    "won",
    "accepted",
    "erred",
    "error_reason_id",
    "response_ms",
    "zip",
    "city",
    "state",
    "device_type",
    "insured",
    "current_carrier",
    "continuous_coverage_months",
    "military_affiliation",
    "credit",
    "pnc_bundle",
    "home_owner",
    "gender",
    "marital_status",
    "num_drivers",
    "num_vehicles",
    "dui",
    "sr22_required",
    "age",
    "num_auto_violations",
    "num_auto_claims",
    "num_auto_accidents",
    "num_home_claims",
    "home_property_type",
    "num_dependents",
    "health_conditions",
    "household_income",
    "life_coverage_type",
    "life_coverage_amount",
    "naics_code",
    "sic_code",
    "num_employees",
    "annual_revenue",
    "lead_created_at",
    "expiration_date",
    "pst_date",
    "pst_hour",
    "exp_rev",
]


def test_registry_column_names_match_original_order():
    """query_builder produces exactly the original pull columns, in order."""
    assert qb.leads_column_names() == EXPECTED_PULL_COLUMNS


def test_models_leads_columns_derived_from_registry():
    """models.LEADS_COLUMNS resolves the registry names to the same ORM cols."""
    assert [c.key for c in models.LEADS_COLUMNS] == EXPECTED_PULL_COLUMNS


def test_every_pulled_name_maps_to_a_real_orm_column():
    """Every registered pulled field is a real LeadPing attribute (typo guard)."""
    for name in qb.leads_column_names():
        assert hasattr(models.LeadPing, name)


def test_no_pii_columns_are_selected():
    """PII fields are never part of the pulled column set."""
    for name in qb.leads_column_names():
        assert fr.get(name).pii is False


def test_pulled_field_specs_align_with_names():
    """pulled_field_specs stays in lockstep with the column-name list."""
    assert [s.name for s in qb.pulled_field_specs()] == qb.leads_column_names()
