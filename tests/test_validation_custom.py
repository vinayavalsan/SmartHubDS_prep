"""Tests for the registry-driven custom validation wiring (validate_us_state)."""

import pandas as pd

from smarthub.data_pull import field_registry as fr
from smarthub.data_pull import validation_custom as vc


def test_validate_us_state_flags_only_present_invalid_values():
    """Present values outside US_STATES are flagged; null/blank are ignored."""
    s = pd.Series(["CA", "ZZ", "TX", "", None, "QQ"])
    result = vc.validate_us_state(s)
    assert result.count == 2  # ZZ and QQ (blank/None ignored)
    assert set(result.examples) == {"ZZ", "QQ"}
    assert result.check == "not in allowed values"


def test_validate_us_state_clean_column_has_no_violations():
    """A fully-valid state column reports zero violations."""
    result = vc.validate_us_state(pd.Series(["CA", "NY", "TX", "DC"]))
    assert result.count == 0
    assert result.examples == []


def test_registry_wires_state_custom_rule_and_id_unique():
    """state references validate_us_state via custom_rule; id is marked unique."""
    state = fr.get("state").validation
    assert state.custom_rule is vc.validate_us_state
    assert state.allowed_values is None  # domain check delegated to the custom rule

    assert fr.get("id").validation.unique is True
