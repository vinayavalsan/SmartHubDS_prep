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

Fields are declared with the small ``_num`` / ``_cat`` / ``_bin`` / ``_dt``
constructors below (each returns an explicit ``RawFieldSpec``); a field that
needs anything unusual can always be written as a full ``RawFieldSpec(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .validation_custom import validate_us_state

# A custom validation function (e.g. ``validate_us_state``): takes a pandas
# Series and returns a ValidationResult. Concrete rules live in the leaf module
# ``validation_custom`` (imported above) and are referenced from a field's
# ``ValidationSpec.custom_rule``; this alias just types that slot.
ValidationRule = Callable[..., Any]

# Lead-type ids (match smarthub.core.lead_types).
LEAD_TYPE_AUTO = 6
LEAD_TYPE_HOME = 1


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
    """How a raw field should be validated (metadata-driven where possible)."""

    kind: str  # "numeric" | "categorical" | "binary" | "datetime"
    allow_missing: bool = True
    required: bool = True
    min_value: float | int | None = None
    max_value: float | int | None = None
    allowed_values: frozenset[Any] | None = None
    custom_rule: ValidationRule | None = None
    severity: str = "error"  # "error" | "warn"
    unique: bool = False  # values must be unique across the batch (e.g. id)


@dataclass(frozen=True)
class RawFieldSpec:
    """One raw field: its identity, the lead types that carry it, and its
    source and validation. Defined exactly once, here."""

    name: str
    lead_types: frozenset[int]
    source: DataSourceSpec
    validation: ValidationSpec
    pii: bool = False
    enabled: bool = True


# --- lead-type scopes + shared domains ---------------------------------------
_ALL = frozenset({LEAD_TYPE_AUTO, LEAD_TYPE_HOME})
_AUTO = frozenset({LEAD_TYPE_AUTO})
_HOME = frozenset({LEAD_TYPE_HOME})

_GENDER_DOMAIN = frozenset({"Male", "Female", "Non-binary"})
_MARITAL_DOMAIN = frozenset({"Single", "Married", "Divorced", "Widowed"})


# --- declarative field constructors (all read from lead_pings) ---------------
def _num(
    name,
    lead_types=_ALL,
    *,
    min_value=None,
    max_value=None,
    allow_missing=True,
    unique=False,
):
    return RawFieldSpec(
        name=name,
        lead_types=lead_types,
        source=DataSourceSpec(table="lead_pings", column=name),
        validation=ValidationSpec(
            kind="numeric",
            allow_missing=allow_missing,
            min_value=min_value,
            max_value=max_value,
            unique=unique,
        ),
    )


def _cat(name, lead_types=_ALL, *, allowed=None, allow_missing=True, custom_rule=None):
    return RawFieldSpec(
        name=name,
        lead_types=lead_types,
        source=DataSourceSpec(table="lead_pings", column=name),
        validation=ValidationSpec(
            kind="categorical",
            allow_missing=allow_missing,
            allowed_values=allowed,
            custom_rule=custom_rule,
        ),
    )


def _bin(name, lead_types=_ALL, *, allow_missing=True):
    return RawFieldSpec(
        name=name,
        lead_types=lead_types,
        source=DataSourceSpec(table="lead_pings", column=name),
        validation=ValidationSpec(kind="binary", allow_missing=allow_missing),
    )


def _dt(name, lead_types=_ALL, *, allow_missing=True):
    return RawFieldSpec(
        name=name,
        lead_types=lead_types,
        source=DataSourceSpec(table="lead_pings", column=name),
        validation=ValidationSpec(kind="datetime", allow_missing=allow_missing),
    )


# --- the registry ------------------------------------------------------------
# Order here defines the pull's column order (see query_builder.leads_column_names
# and models.LEADS_COLUMNS). It matches the previous hand-maintained tuple.
# Ranges/domains mirror the current validation_rules constants; the `state`
# custom_rule (validate_us_state) is attached when the runner is registry-driven.
_FIELDS: list[RawFieldSpec] = [
    _num("id", allow_missing=False, unique=True),
    _dt("created_at", allow_missing=False),
    _num("account_id"),
    _num("campaign_id"),
    _num("lead_type_id", allow_missing=False),
    _num("source_type_id"),
    _num("bidding_strategy_id"),
    _cat("traffic_tier"),
    _num("total_listings", min_value=0),
    _num("accepted_listings", min_value=0),
    _num("bid", min_value=0),
    _num("rev", min_value=0),
    _bin("won"),
    _bin("accepted"),
    _bin("erred"),
    _num("error_reason_id"),
    _num("response_ms", min_value=0),
    _cat("zip"),
    _cat("city"),
    _cat("state", allow_missing=False, custom_rule=validate_us_state),
    _cat("device_type"),
    _bin("insured"),
    _cat("current_carrier"),
    _num("continuous_coverage_months", min_value=0, max_value=600),
    _bin("military_affiliation"),
    _cat("credit"),
    _bin("pnc_bundle"),
    _bin("home_owner"),
    _cat("gender", allowed=_GENDER_DOMAIN),
    _cat("marital_status", allowed=_MARITAL_DOMAIN),
    _num("num_drivers", _AUTO, min_value=0, max_value=6),
    _num("num_vehicles", _AUTO, min_value=0, max_value=12),
    _bin("dui", _AUTO),
    _bin("sr22_required", _AUTO),
    _num("age", min_value=1, max_value=200),
    _num("num_auto_violations", _AUTO, min_value=0, max_value=20),
    _num("num_auto_claims", _AUTO, min_value=0, max_value=20),
    _num("num_auto_accidents", _AUTO, min_value=0, max_value=20),
    _num("num_home_claims", _HOME, min_value=0, max_value=20),
    _cat("home_property_type", _HOME),
    _num("num_dependents", min_value=0, max_value=15),
    _cat("health_conditions"),
    # household_income is treated as a non-negative numeric to mirror current
    # behaviour; the warehouse value may be an income band/string (a known
    # limitation to revisit — see docs/validation_rules.md).
    _num("household_income", min_value=0),
    _cat("life_coverage_type"),
    _num("life_coverage_amount", min_value=0),
    _cat("naics_code"),
    _cat("sic_code"),
    _num("num_employees", min_value=0),
    _num("annual_revenue", min_value=0),
    _dt("lead_created_at"),
    _dt("expiration_date"),
    _dt("pst_date"),
    _num("pst_hour"),
    _num("exp_rev", min_value=0),
]

RAW_FIELD_REGISTRY: dict[str, RawFieldSpec] = {spec.name: spec for spec in _FIELDS}


def field_names() -> list[str]:
    """Return all enabled, non-PII raw-field names, in registry order."""
    return [
        name
        for name, spec in RAW_FIELD_REGISTRY.items()
        if spec.enabled and not spec.pii
    ]


def fields_for_lead_type(lead_type_id: int) -> list[RawFieldSpec]:
    """Return the enabled raw fields that apply to a given lead type.

    Inputs
    ------
    lead_type_id : int
        Lead type id (e.g. 6=auto, 1=home).

    Returns
    -------
    list[RawFieldSpec]
        The enabled specs whose ``lead_types`` include ``lead_type_id``.
    """
    return [
        spec
        for spec in RAW_FIELD_REGISTRY.values()
        if spec.enabled and lead_type_id in spec.lead_types
    ]


def get(name: str) -> RawFieldSpec:
    """Return the spec for ``name``, or raise ``KeyError`` if it is unknown."""
    return RAW_FIELD_REGISTRY[name]


def columns_not_for_lead_type(lead_type_id: int) -> set[str]:
    """Registered raw columns that do NOT apply to a given lead type.

    Used to scope validation reporting (e.g. the high-missing catalogue) so a
    single-lead-type pull isn't flagged for other products' columns being empty
    — an auto pull legitimately has empty ``home_*`` columns.

    Returns an empty set for an unrecognised lead type (one that appears in no
    field's ``lead_types``), so an unmodelled type never suppresses every field.

    Inputs
    ------
    lead_type_id : int
        Lead type id (e.g. 6=auto, 1=home).

    Returns
    -------
    set[str]
        Enabled field names whose ``lead_types`` exclude ``lead_type_id``
        (empty when the lead type isn't modelled in the registry).
    """
    known: set[int] = set()
    for spec in RAW_FIELD_REGISTRY.values():
        known |= set(spec.lead_types)
    if lead_type_id not in known:
        return set()
    return {
        name
        for name, spec in RAW_FIELD_REGISTRY.items()
        if spec.enabled and lead_type_id not in spec.lead_types
    }
