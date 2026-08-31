"""Single source of truth for SmartHub model features.

Each feature is declared once in ``FEATURES``. The registry stores feature
metadata and, for derived features, the function that creates the feature.
Lead-type applicability is written directly with names such as
``frozenset({"auto", "home"})`` so new lead types do not require predefined
lead-type combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from smarthub.core import holidays
from smarthub.data_pull.field_registry import numeric_validation_bounds
from smarthub.data_pull.validation_custom import US_STATES

FeatureDeriver = Callable[[pd.DataFrame], pd.Series]

MISSING_CATEGORY = "__MISSING__"
MISSING_NUMERIC = -1


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of one model feature."""

    name: str
    kind: str  # numeric_continuous | numeric_discrete | categorical | binary
    source: str  # raw | derived
    lead_types: frozenset[str]
    enabled: bool = True
    api_input: str | None = None
    derive: FeatureDeriver | None = None
    training_include_values: frozenset[object] | None = None
    mandatory: bool = False
    missing_value: object | None = None

    def resolved_missing_value(self) -> object:
        """Return the explicit missing-value representation for the feature."""
        if self.missing_value is not None:
            return self.missing_value
        if self.kind == "categorical":
            return MISSING_CATEGORY
        return MISSING_NUMERIC


# ---------------------------------------------------------------------------
# Derived-feature functions
# ---------------------------------------------------------------------------


def _created_at_pacific(frame: pd.DataFrame) -> pd.Series:
    """Interpret ``created_at`` as UTC and convert it to Pacific time."""
    created_at = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    return created_at.dt.tz_convert("America/Los_Angeles")


def _derive_created_hour(frame: pd.DataFrame) -> pd.Series:
    """Return the Pacific hour derived from ``created_at``."""
    return _created_at_pacific(frame).dt.hour


def _derive_created_dayofweek(frame: pd.DataFrame) -> pd.Series:
    """Return the Pacific day of week derived from ``created_at``."""
    return _created_at_pacific(frame).dt.dayofweek


def _derive_is_workday(frame: pd.DataFrame) -> pd.Series:
    """Return Pacific-calendar workday status from ``created_at``."""
    created_at = _created_at_pacific(frame)

    result = pd.Series(MISSING_NUMERIC, index=frame.index, dtype="int64")
    known = created_at.notna()
    if known.any():
        result.loc[known] = (
            created_at.loc[known]
            .map(lambda value: int(holidays.is_workday(value.date())))
            .astype("int64")
        )
    return result


def _derive_is_married(frame: pd.DataFrame) -> pd.Series:
    """Return married status while preserving a missing source value."""
    marital_status = frame["marital_status"].astype("string").str.strip().str.lower()
    marital_status = marital_status.mask(marital_status == "")

    result = pd.Series(MISSING_NUMERIC, index=frame.index, dtype="int64")
    known = marital_status.notna()
    result.loc[known] = marital_status.loc[known].eq("married").astype("int64")
    return result


def _derive_multi_vehicle(frame: pd.DataFrame) -> pd.Series:
    """Return multi-vehicle status while preserving a missing source value."""
    num_vehicles = pd.to_numeric(frame["num_vehicles"], errors="coerce")

    result = pd.Series(MISSING_NUMERIC, index=frame.index, dtype="int64")
    known = num_vehicles.notna()
    result.loc[known] = num_vehicles.loc[known].gt(1).astype("int64")
    return result


def _derive_age_valid(frame: pd.DataFrame) -> pd.Series:
    """Return whether age is numeric and within the registered valid range."""
    age = pd.to_numeric(frame["age"], errors="coerce")
    min_age, max_age = numeric_validation_bounds("age")
    if min_age is None or max_age is None:
        raise ValueError("Age validation bounds must define both min and max.")

    return age.notna().mul(age.between(min_age, max_age)).astype("int64")


def _derive_age_cohort(frame: pd.DataFrame) -> pd.Series:
    """Clean age and return its categorical age cohort."""
    age = pd.to_numeric(frame["age"], errors="coerce")
    min_age, max_age = numeric_validation_bounds("age")
    if min_age is None or max_age is None:
        raise ValueError("Age validation bounds must define both min and max.")

    plausible_age = age.between(min_age, max_age)
    cleaned_age = age.where(plausible_age)

    # Keep age cleaning identical for training and serving.
    frame["age"] = cleaned_age.fillna(MISSING_NUMERIC)

    return (
        pd.cut(
            cleaned_age,
            bins=[0, 18, 25, 35, 45, 55, 65, 75, 85, 100, max_age + 1],
            labels=[
                "under_18",
                "18_24",
                "25_34",
                "35_44",
                "45_54",
                "55_64",
                "65_74",
                "75_84",
                "85_99",
                "100_plus",
            ],
            right=False,
        )
        .astype("string")
        .fillna(MISSING_CATEGORY)
    )


# ---------------------------------------------------------------------------
# Feature registry
#
# Dictionary insertion order is the canonical model-feature order. To add a
# feature, add one FeatureSpec entry. A derived feature also needs one
# _derive_* function and that function assigned to ``derive`` in its entry.
# ---------------------------------------------------------------------------

FEATURES: dict[str, FeatureSpec] = {
    # -----------------------------------------------------------------------
    # Shared features (Auto + Home)
    # -----------------------------------------------------------------------
    "bid": FeatureSpec(
        name="bid",
        kind="numeric_continuous",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="bid",
    ),
    "age": FeatureSpec(
        name="age",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="age",
    ),
    "continuous_coverage_months": FeatureSpec(
        name="continuous_coverage_months",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="continuous_coverage_months",
    ),
    "created_at": FeatureSpec(
        name="created_at",
        kind="datetime",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=False,
        mandatory=True,
        api_input="created_at",
    ),
    "created_hour": FeatureSpec(
        name="created_hour",
        kind="categorical",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="created_at",
        derive=_derive_created_hour,
    ),
    "created_dayofweek": FeatureSpec(
        name="created_dayofweek",
        kind="categorical",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="created_at",
        derive=_derive_created_dayofweek,
    ),
    "is_workday": FeatureSpec(
        name="is_workday",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="created_at",
        derive=_derive_is_workday,
    ),
    "is_married": FeatureSpec(
        name="is_married",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        enabled=False,
        api_input="marital_status",
        derive=_derive_is_married,
    ),
    "age_valid": FeatureSpec(
        name="age_valid",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="age",
        derive=_derive_age_valid,
    ),
    "age_cohort": FeatureSpec(
        name="age_cohort",
        kind="categorical",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        enabled=False,
        api_input="age",
        derive=_derive_age_cohort,
    ),
    "state": FeatureSpec(
        name="state",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="state",
        training_include_values=US_STATES,
    ),
    "gender": FeatureSpec(
        name="gender",
        kind="categorical",
        source="raw",
        enabled=True,
        lead_types=frozenset({"auto", "home"}),
        api_input="gender",
    ),
    "marital_status": FeatureSpec(
        name="marital_status",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="marital_status",
    ),
    "military_affiliation": FeatureSpec(
        name="military_affiliation",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="military_affiliation",
    ),
    "insured": FeatureSpec(
        name="insured",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="insured",
    ),
    "campaign_id": FeatureSpec(
        name="campaign_id",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="campaign_id",
        enabled=True,
        mandatory=True,
    ),
    "traffic_tier": FeatureSpec(
        name="traffic_tier",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="traffic_tier",
    ),
    "source_type_id": FeatureSpec(
        name="source_type_id",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        mandatory=True,
        api_input="source_type_id",
    ),
    "current_carrier": FeatureSpec(
        name="current_carrier",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="current_carrier",
    ),
    "home_owner": FeatureSpec(
        name="home_owner",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="home_owner",
    ),
    "device_type": FeatureSpec(
        name="device_type",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        enabled=True,
        api_input="device_type",
    ),
    # -----------------------------------------------------------------------
    # Auto-only features
    # -----------------------------------------------------------------------
    "num_vehicles": FeatureSpec(
        name="num_vehicles",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="num_vehicles",
    ),
    "num_drivers": FeatureSpec(
        name="num_drivers",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="num_drivers",
    ),
    "num_auto_violations": FeatureSpec(
        name="num_auto_violations",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="num_auto_violations",
    ),
    "num_auto_accidents": FeatureSpec(
        name="num_auto_accidents",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="num_auto_accidents",
    ),
    "multi_vehicle": FeatureSpec(
        name="multi_vehicle",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="num_vehicles",
        derive=_derive_multi_vehicle,
    ),
    "dui": FeatureSpec(
        name="dui",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="dui",
    ),
    "sr22_required": FeatureSpec(
        name="sr22_required",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto"}),
        enabled=True,
        api_input="sr22_required",
    ),
    # -----------------------------------------------------------------------
    # Home-only features
    # -----------------------------------------------------------------------
    "num_home_claims": FeatureSpec(
        name="num_home_claims",
        kind="numeric_discrete",
        source="raw",
        lead_types=frozenset({"home"}),
        enabled=True,
        api_input="num_home_claims",
    ),
    "home_property_type": FeatureSpec(
        name="home_property_type",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"home"}),
        enabled=True,
        api_input="home_property_type",
    ),
}
