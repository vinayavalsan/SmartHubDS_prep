"""Centralized raw-field registry for the data-pull stage.

Single source of truth for every raw ``lead_pings`` field: how it is *extracted*
(``DataSourceSpec``) and how it should be *validated* (``ValidationSpec``),
scoped per lead type. Mirrors the ``feature_engineering.feature_registry``
pattern — one declarative entry per field, no logic in the registry itself.

The registry is now the single definition of which columns the pull selects and
in what order: ``query_builder.leads_column_names()`` reads it, and
``data_pull.models.LEADS_COLUMNS`` resolves those names to ORM columns. The old
hand-maintained ``LEADS_COLUMNS`` tuple is gone.

Wiring the *generic validation runner* to ``ValidationSpec`` (ranges, domains,
missing, presence) and moving field-specific checks like ``validate_us_state``
into ``ValidationSpec.custom_rule`` is the next slice — the validation runner
still uses ``validation_rules.py`` directly for now. Keeping the registry
metadata-only is deliberate: validation logic stays in ``validation_rules.py``
and is referenced from a field only when simple metadata can't express it.

Fields are declared explicitly in ``RAW_FIELD_REGISTRY`` so the dictionary
key, field name, lead-type scope, source, and validation metadata are visible
together at the declaration site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smarthub.core.lead_types import all_lead_types, lead_type_name

from .validation_custom import validate_us_state

# A custom validation function (e.g. ``validate_us_state``): takes a pandas
# Series and returns a ValidationResult. Concrete rules live in the leaf module
# ``validation_custom`` (imported above) and are referenced from a field's
# ``ValidationSpec.custom_rule``; this alias just types that slot.
ValidationRule = Callable[..., Any]


@dataclass(frozen=True)
class DataSourceSpec:
    """Where a raw field is read from in the warehouse."""

    table: str
    column: str
    alias: str | None = None
    join: str | None = None
    aggregation: str | None = None


@dataclass(frozen=True)
class ValidationSpec:
    """How a raw field should be validated and reported."""

    kind: str  # "numeric" | "categorical" | "binary" | "datetime"
    flag_missing: bool = False
    required: bool = True
    min_value: float | int | None = None
    max_value: float | int | None = None
    allowed_values: frozenset[Any] | None = None
    custom_rule: ValidationRule | None = None
    unique: bool = False  # values must be unique across the batch (e.g. id)


@dataclass(frozen=True)
class RawFieldSpec:
    """One raw field: its identity, the lead types that carry it, and its
    source and validation. Defined exactly once, here."""

    name: str
    lead_types: frozenset[str]
    source: DataSourceSpec
    validation: ValidationSpec
    pii: bool = False
    enabled: bool = True


# --- shared validation domains -----------------------------------------------
_GENDER_DOMAIN = frozenset({"Male", "Female", "Non-binary"})
_MARITAL_DOMAIN = frozenset({"Single", "Married", "Divorced", "Widowed"})

_ALL_LEAD_TYPES = frozenset(all_lead_types())


# --- the registry ------------------------------------------------------------
# Dictionary insertion order is the canonical pull-column order. Each raw field
# is declared explicitly so its lead-type scope, source, and validation rules
# are visible at the declaration site.
RAW_FIELD_REGISTRY: dict[str, RawFieldSpec] = {
    "id": RawFieldSpec(
        name="id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="id"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            unique=True,
        ),
        pii=False,
        enabled=True,
    ),
    "created_at": RawFieldSpec(
        name="created_at",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="created_at"),
        validation=ValidationSpec(
            kind="datetime",
            flag_missing=True,
        ),
        pii=False,
        enabled=True,
    ),
    "account_id": RawFieldSpec(
        name="account_id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="account_id"),
        validation=ValidationSpec(kind="numeric", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "campaign_id": RawFieldSpec(
        name="campaign_id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="campaign_id"),
        validation=ValidationSpec(kind="numeric", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "lead_type_id": RawFieldSpec(
        name="lead_type_id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="lead_type_id"),
        validation=ValidationSpec(kind="numeric", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "source_type_id": RawFieldSpec(
        name="source_type_id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="source_type_id"),
        validation=ValidationSpec(kind="numeric", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "bidding_strategy_id": RawFieldSpec(
        name="bidding_strategy_id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="bidding_strategy_id"),
        validation=ValidationSpec(kind="numeric", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "traffic_tier": RawFieldSpec(
        name="traffic_tier",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="traffic_tier"),
        validation=ValidationSpec(kind="categorical", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "total_listings": RawFieldSpec(
        name="total_listings",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="total_listings"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "accepted_listings": RawFieldSpec(
        name="accepted_listings",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="accepted_listings"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "bid": RawFieldSpec(
        name="bid",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="bid"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "rev": RawFieldSpec(
        name="rev",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="rev"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "won": RawFieldSpec(
        name="won",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="won"),
        validation=ValidationSpec(kind="binary", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "accepted": RawFieldSpec(
        name="accepted",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="accepted"),
        validation=ValidationSpec(kind="binary", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "erred": RawFieldSpec(
        name="erred",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="erred"),
        validation=ValidationSpec(kind="binary", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "error_reason_id": RawFieldSpec(
        name="error_reason_id",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="error_reason_id"),
        validation=ValidationSpec(kind="numeric", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "response_ms": RawFieldSpec(
        name="response_ms",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="response_ms"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "zip": RawFieldSpec(
        name="zip",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="zip"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "city": RawFieldSpec(
        name="city",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="city"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "state": RawFieldSpec(
        name="state",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="state"),
        validation=ValidationSpec(
            kind="categorical",
            flag_missing=True,
            custom_rule=validate_us_state,
        ),
        pii=False,
        enabled=True,
    ),
    "device_type": RawFieldSpec(
        name="device_type",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="device_type"),
        validation=ValidationSpec(kind="categorical", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "insured": RawFieldSpec(
        name="insured",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="insured"),
        validation=ValidationSpec(kind="binary", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "current_carrier": RawFieldSpec(
        name="current_carrier",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="current_carrier"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "continuous_coverage_months": RawFieldSpec(
        name="continuous_coverage_months",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(
            table="lead_pings",
            column="continuous_coverage_months",
        ),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
            max_value=600,
        ),
        pii=False,
        enabled=True,
    ),
    "military_affiliation": RawFieldSpec(
        name="military_affiliation",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="military_affiliation"),
        validation=ValidationSpec(kind="binary", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "credit": RawFieldSpec(
        name="credit",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="credit"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "pnc_bundle": RawFieldSpec(
        name="pnc_bundle",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="pnc_bundle"),
        validation=ValidationSpec(kind="binary", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "home_owner": RawFieldSpec(
        name="home_owner",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="home_owner"),
        validation=ValidationSpec(kind="binary", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "gender": RawFieldSpec(
        name="gender",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="gender"),
        validation=ValidationSpec(
            kind="categorical",
            flag_missing=True,
            allowed_values=_GENDER_DOMAIN,
        ),
        pii=False,
        enabled=True,
    ),
    "marital_status": RawFieldSpec(
        name="marital_status",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="marital_status"),
        validation=ValidationSpec(
            kind="categorical",
            flag_missing=True,
            allowed_values=_MARITAL_DOMAIN,
        ),
        pii=False,
        enabled=True,
    ),
    "num_drivers": RawFieldSpec(
        name="num_drivers",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="num_drivers"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
            max_value=10,
        ),
        pii=False,
        enabled=True,
    ),
    "num_vehicles": RawFieldSpec(
        name="num_vehicles",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="num_vehicles"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
            max_value=20,
        ),
        pii=False,
        enabled=True,
    ),
    "dui": RawFieldSpec(
        name="dui",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="dui"),
        validation=ValidationSpec(kind="binary", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "sr22_required": RawFieldSpec(
        name="sr22_required",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="sr22_required"),
        validation=ValidationSpec(kind="binary", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "age": RawFieldSpec(
        name="age",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="age"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=1,
            max_value=130,
        ),
        pii=False,
        enabled=True,
    ),
    "num_auto_violations": RawFieldSpec(
        name="num_auto_violations",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="num_auto_violations"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
            max_value=20,
        ),
        pii=False,
        enabled=True,
    ),
    "num_auto_claims": RawFieldSpec(
        name="num_auto_claims",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="num_auto_claims"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
            max_value=20,
        ),
        pii=False,
        enabled=True,
    ),
    "num_auto_accidents": RawFieldSpec(
        name="num_auto_accidents",
        lead_types=frozenset({"auto"}),
        source=DataSourceSpec(table="lead_pings", column="num_auto_accidents"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
            max_value=20,
        ),
        pii=False,
        enabled=True,
    ),
    "num_home_claims": RawFieldSpec(
        name="num_home_claims",
        lead_types=frozenset({"home"}),
        source=DataSourceSpec(table="lead_pings", column="num_home_claims"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=True,
            min_value=0,
            max_value=20,
        ),
        pii=False,
        enabled=True,
    ),
    "home_property_type": RawFieldSpec(
        name="home_property_type",
        lead_types=frozenset({"home"}),
        source=DataSourceSpec(table="lead_pings", column="home_property_type"),
        validation=ValidationSpec(kind="categorical", flag_missing=True),
        pii=False,
        enabled=True,
    ),
    "num_dependents": RawFieldSpec(
        name="num_dependents",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="num_dependents"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
            max_value=15,
        ),
        pii=False,
        enabled=True,
    ),
    "health_conditions": RawFieldSpec(
        name="health_conditions",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="health_conditions"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "household_income": RawFieldSpec(
        name="household_income",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="household_income"),
        validation=ValidationSpec(
            kind="categorical",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "life_coverage_type": RawFieldSpec(
        name="life_coverage_type",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="life_coverage_type"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "life_coverage_amount": RawFieldSpec(
        name="life_coverage_amount",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="life_coverage_amount"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "naics_code": RawFieldSpec(
        name="naics_code",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="naics_code"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "sic_code": RawFieldSpec(
        name="sic_code",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="sic_code"),
        validation=ValidationSpec(kind="categorical", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "num_employees": RawFieldSpec(
        name="num_employees",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="num_employees"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "annual_revenue": RawFieldSpec(
        name="annual_revenue",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="annual_revenue"),
        validation=ValidationSpec(
            kind="categorical",
            flag_missing=False,
            # min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
    "lead_created_at": RawFieldSpec(
        name="lead_created_at",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="lead_created_at"),
        validation=ValidationSpec(kind="datetime", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "expiration_date": RawFieldSpec(
        name="expiration_date",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="expiration_date"),
        validation=ValidationSpec(kind="datetime", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "pst_date": RawFieldSpec(
        name="pst_date",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="pst_date"),
        validation=ValidationSpec(kind="datetime", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "pst_hour": RawFieldSpec(
        name="pst_hour",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="pst_hour"),
        validation=ValidationSpec(kind="numeric", flag_missing=False),
        pii=False,
        enabled=True,
    ),
    "exp_rev": RawFieldSpec(
        name="exp_rev",
        lead_types=_ALL_LEAD_TYPES,
        source=DataSourceSpec(table="lead_pings", column="exp_rev"),
        validation=ValidationSpec(
            kind="numeric",
            flag_missing=False,
            min_value=0,
        ),
        pii=False,
        enabled=True,
    ),
}


def field_names() -> list[str]:
    """Return all enabled, non-PII raw-field names, in registry order."""
    return [
        name
        for name, spec in RAW_FIELD_REGISTRY.items()
        if spec.enabled and not spec.pii
    ]


def fields_for_lead_type(lead_type_name: str) -> list[RawFieldSpec]:
    """Return the enabled raw fields that apply to a given lead type.

    Inputs
    ------
    lead_type_name : str
        Registered lead-type name.

    Returns
    -------
    list[RawFieldSpec]
        The enabled specs whose ``lead_types`` include ``lead_type_name``.
    """
    return [
        spec
        for spec in RAW_FIELD_REGISTRY.values()
        if spec.enabled and lead_type_name in spec.lead_types
    ]


def get(name: str) -> RawFieldSpec:
    """Return the spec for ``name``, or raise ``KeyError`` if it is unknown."""
    return RAW_FIELD_REGISTRY[name]


def numeric_validation_bounds(
    name: str,
) -> tuple[float | int | None, float | int | None]:
    """Return configured numeric validation bounds for a raw field.

    Inputs
    ------
    name : str
        Registered raw-field name.

    Returns
    -------
    tuple[float | int | None, float | int | None]
        Minimum value followed by maximum value.

    Raises
    ------
    ValueError
        If the requested field is not configured as numeric.
    """
    spec = get(name)
    if spec.validation.kind != "numeric":
        raise ValueError(f"Field {name!r} is not configured as numeric.")
    return spec.validation.min_value, spec.validation.max_value


def columns_not_for_lead_type_id(lead_type_id: int) -> set[str]:
    """Registered raw columns that do NOT apply to a given lead type ID.

    Inputs
    ------
    lead_type_id : int
        Registered lead-type ID.

    Returns
    -------
    set[str]
        Enabled field names outside the resolved lead type's scope. Returns an
        empty set when the ID is not registered.
    """
    try:
        resolved_name = lead_type_name(lead_type_id)
    except ValueError:
        return set()
    return columns_not_for_lead_type(resolved_name)


def columns_not_for_lead_type(lead_type_name: str) -> set[str]:
    """Registered raw columns that do NOT apply to a given lead type.

    Used to scope validation reporting (e.g. the high-missing catalogue) so a
    single-lead-type pull isn't flagged for other products' columns being empty
    — an auto pull legitimately has empty ``home_*`` columns.

    Returns an empty set for an unrecognised lead type (one that appears in no
    field's ``lead_types``), so an unmodelled type never suppresses every field.

    Inputs
    ------
    lead_type_name : str
        Registered lead-type name.

    Returns
    -------
    set[str]
        Enabled field names whose ``lead_types`` exclude ``lead_type_name``
        (empty when the lead type isn't modelled in the registry).
    """
    known: set[str] = set()
    for spec in RAW_FIELD_REGISTRY.values():
        known |= set(spec.lead_types)
    if lead_type_name not in known:
        return set()
    return {
        name
        for name, spec in RAW_FIELD_REGISTRY.items()
        if spec.enabled and lead_type_name not in spec.lead_types
    }
