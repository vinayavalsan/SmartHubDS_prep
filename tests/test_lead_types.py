"""Tests for the canonical lead-type registry."""

import pytest

from smarthub.core.lead_types import (
    LEAD_TYPES,
    all_lead_type_ids,
    all_lead_types,
    lead_type_id,
    lead_type_name,
)


def test_lookup_is_bidirectional():
    for name, registered_id in LEAD_TYPES.items():
        assert lead_type_id(name) == registered_id
        assert lead_type_name(registered_id) == name


def test_lookup_normalizes_names():
    assert lead_type_id(" AUTO ") == LEAD_TYPES["auto"]


def test_all_lead_types_returns_copy():
    returned = all_lead_types()
    returned["synthetic"] = 999
    assert "synthetic" not in LEAD_TYPES


def test_all_lead_type_ids_matches_registry_order():
    assert all_lead_type_ids() == tuple(LEAD_TYPES.values())


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown lead type"):
        lead_type_id("commercial")


def test_unknown_id_raises():
    with pytest.raises(ValueError, match="Unknown lead_type_id"):
        lead_type_name(999)
