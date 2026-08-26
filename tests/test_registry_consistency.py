"""Cross-registry contract tests for raw fields and model features."""

from smarthub.data_pull import field_registry
from smarthub.feature_engineering.feature_registry import FEATURES

# Raw validation kind -> model-feature kinds that are semantically compatible.
#
# A numeric warehouse field may intentionally be modeled categorically
# (for example campaign_id). Likewise a binary raw field may be modeled as a
# categorical feature. Other kind changes should be explicit and reviewed.
KIND_COMPATIBILITY = {
    "numeric_continuous": {"numeric_continuous", "categorical"},
    "numeric_discrete": {"numeric_discrete", "categorical"},
    "categorical": {"categorical"},
    "binary": {"binary", "categorical"},
    "datetime": {"datetime"},
}


def test_raw_features_exist_in_field_registry():
    """Every directly sourced model feature has a registered raw field."""
    missing = sorted(
        spec.name
        for spec in FEATURES.values()
        if spec.source == "raw" and spec.name not in field_registry.RAW_FIELD_REGISTRY
    )
    assert missing == [], f"Raw model features missing from field registry: {missing}"


def test_raw_features_use_their_matching_api_input():
    """A raw feature's API input is the same raw field it represents."""
    mismatches = {
        spec.name: spec.api_input
        for spec in FEATURES.values()
        if spec.source == "raw" and spec.api_input != spec.name
    }
    assert mismatches == {}, (
        "Raw model features should use their own field as api_input: " f"{mismatches}"
    )


def test_derived_feature_inputs_exist_in_field_registry():
    """Every derived feature's declared raw input exists in the field registry."""
    missing = {
        spec.name: spec.api_input
        for spec in FEATURES.values()
        if spec.source == "derived"
        and (
            spec.api_input is None
            or spec.api_input not in field_registry.RAW_FIELD_REGISTRY
        )
    }
    assert missing == {}, (
        "Derived features reference raw inputs missing from field registry: "
        f"{missing}"
    )


def test_derived_features_have_derivers():
    """Every derived feature has an implementation callable."""
    missing = sorted(
        spec.name
        for spec in FEATURES.values()
        if spec.source == "derived" and not callable(spec.derive)
    )
    assert missing == [], f"Derived features without a derive function: {missing}"


def test_raw_feature_kinds_are_compatible_with_validation_kinds():
    """Direct raw features use a model kind compatible with raw validation."""
    mismatches = {}

    for spec in FEATURES.values():
        if spec.source != "raw":
            continue

        raw_spec = field_registry.RAW_FIELD_REGISTRY[spec.name]
        raw_kind = raw_spec.validation.kind
        allowed_feature_kinds = KIND_COMPATIBILITY.get(raw_kind, set())

        if spec.kind not in allowed_feature_kinds:
            mismatches[spec.name] = {
                "validation_kind": raw_kind,
                "feature_kind": spec.kind,
            }

    assert mismatches == {}, (
        "Feature kinds are incompatible with field validation kinds: " f"{mismatches}"
    )


def test_feature_lead_types_are_supported_by_raw_fields():
    """Features cannot claim lead types unsupported by their raw source field."""
    mismatches = {}

    for spec in FEATURES.values():
        source_name = spec.name if spec.source == "raw" else spec.api_input
        if source_name is None:
            continue

        raw_spec = field_registry.RAW_FIELD_REGISTRY[source_name]
        unsupported = spec.lead_types - raw_spec.lead_types
        if unsupported:
            mismatches[spec.name] = {
                "source": source_name,
                "unsupported_lead_types": sorted(unsupported),
            }

    assert mismatches == {}, (
        "Feature lead-type scope exceeds its raw source field scope: " f"{mismatches}"
    )
