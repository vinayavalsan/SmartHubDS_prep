"""Tests for the data_pull raw-field registry (structure-only first slice)."""

from smarthub.data_pull import field_registry as fr


def test_every_key_matches_its_spec_name():
    """A field is keyed by its own ``name`` (defined exactly once)."""
    for key, spec in fr.RAW_FIELD_REGISTRY.items():
        assert key == spec.name


def test_specs_are_frozen():
    """Registry specs are immutable (frozen dataclasses)."""
    import dataclasses

    import pytest

    spec = fr.get("age")
    for obj, attr in (
        (spec, "name"),
        (spec.source, "column"),
        (spec.validation, "kind"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, "x")


def test_validation_kinds_are_known():
    """Every field declares a supported validation kind."""
    allowed = {"numeric", "categorical", "binary", "datetime"}
    for spec in fr.RAW_FIELD_REGISTRY.values():
        assert spec.validation.kind in allowed


def test_field_names_returns_enabled_only():
    """field_names lists enabled fields and matches the registry keys."""
    names = fr.field_names()
    assert "age" in names and "state" in names
    assert names == [k for k, s in fr.RAW_FIELD_REGISTRY.items() if s.enabled]


def test_fields_for_lead_type_scoping():
    """Per-lead-type scoping: auto-only fields don't appear for home."""
    auto = {s.name for s in fr.fields_for_lead_type(fr.LEAD_TYPE_AUTO)}
    home = {s.name for s in fr.fields_for_lead_type(fr.LEAD_TYPE_HOME)}
    assert "num_vehicles" in auto  # auto-only
    assert "num_vehicles" not in home
    assert "age" in auto and "age" in home  # shared


def test_get_unknown_raises():
    """get() raises KeyError for an unregistered field."""
    import pytest

    with pytest.raises(KeyError):
        fr.get("not_a_field")


def test_columns_not_for_lead_type_scoping():
    """Other products' columns are out-of-scope for a lead type; shared aren't."""
    auto_out = fr.columns_not_for_lead_type(fr.LEAD_TYPE_AUTO)
    home_out = fr.columns_not_for_lead_type(fr.LEAD_TYPE_HOME)
    # home-only columns are out of scope for auto; shared/auto columns are not.
    assert "home_property_type" in auto_out and "num_home_claims" in auto_out
    assert "num_vehicles" not in auto_out  # auto-only -> in scope for auto
    assert "age" not in auto_out  # shared -> in scope
    # symmetric for home
    assert "num_vehicles" in home_out and "home_property_type" not in home_out
    # An unmodelled lead type scopes nothing (never suppresses every field).
    assert fr.columns_not_for_lead_type(999) == set()
