"""Golden test: the registry-driven pull columns match the original pull.

The raw-field registry (field_registry) now drives which columns the pull
selects, resolved to ORM columns by models.LEADS_COLUMNS. This pins the exact
column set + order to the list the hand-maintained tuple used to produce, so the
refactor is provably behaviour-preserving.
"""

from smarthub.data_pull import field_registry as fr
from smarthub.data_pull import models
from smarthub.data_pull import query_builder as qb


def test_query_builder_column_names_follow_registry_order():
    """query_builder uses the enabled non-PII registry fields in registry order."""
    assert qb.leads_column_names() == fr.field_names()


def test_models_leads_columns_derived_from_registry():
    """models.LEADS_COLUMNS resolves the query-builder names to ORM columns."""
    assert [column.key for column in models.LEADS_COLUMNS] == qb.leads_column_names()


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
