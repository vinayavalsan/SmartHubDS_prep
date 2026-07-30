"""Single source of truth for SmartHub model features.

Each feature is declared once in ``FEATURES``. The registry stores feature
metadata and, for derived features, the function that creates the feature.
Lead-type applicability is written directly with names such as
``frozenset({"auto", "home"})`` so new lead types do not require predefined
lead-type combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

FeatureDeriver = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of one model feature."""

    name: str
    kind: str  # numeric | categorical | binary
    source: str  # raw | derived
    lead_types: frozenset[str]
    mandatory_for: frozenset[str] = field(default_factory=frozenset)
    api_input: str | None = None
    derive: FeatureDeriver | None = None


# ---------------------------------------------------------------------------
# Derived-feature functions
# ---------------------------------------------------------------------------


def _derive_created_hour(frame: pd.DataFrame) -> pd.Series:
    """Return the Pacific hour when available, otherwise the created-at hour."""
    if "pst_hour" in frame.columns:
        return pd.to_numeric(frame["pst_hour"], errors="coerce")

    created_at = pd.to_datetime(frame["created_at"], errors="coerce")
    return created_at.dt.hour


def _derive_created_dayofweek(frame: pd.DataFrame) -> pd.Series:
    """Return the Pacific day of week when available."""
    if "pst_date" in frame.columns:
        date_values = pd.to_datetime(frame["pst_date"], errors="coerce")
    else:
        date_values = pd.to_datetime(frame["created_at"], errors="coerce")

    return date_values.dt.dayofweek


def _derive_is_workday(frame: pd.DataFrame) -> pd.Series:
    """Return 1 for a workday and 0 for a weekend or observed holiday."""
    from smarthub.core import holidays

    if "pst_date" in frame.columns:
        date_values = pd.to_datetime(frame["pst_date"], errors="coerce")
    else:
        date_values = pd.to_datetime(frame["created_at"], errors="coerce")

    return date_values.map(
        lambda value: (int(holidays.is_workday(value.date())) if pd.notna(value) else 0)
    ).astype("int64")


def _derive_is_married(frame: pd.DataFrame) -> pd.Series:
    """Return 1 when marital status is married, otherwise 0."""
    marital_status = frame["marital_status"].astype("string").str.strip().str.lower()
    return marital_status.eq("married").fillna(False).astype("int64")


def _derive_multi_vehicle(frame: pd.DataFrame) -> pd.Series:
    """Return 1 when the lead has more than one vehicle, otherwise 0."""
    num_vehicles = pd.to_numeric(frame["num_vehicles"], errors="coerce")
    return num_vehicles.gt(1).fillna(False).astype("int64")


def _derive_age_cohort(frame: pd.DataFrame) -> pd.Series:
    """Clean age and return its categorical age cohort."""
    age = pd.to_numeric(frame["age"], errors="coerce")
    plausible_age = age.between(1, 200)
    cleaned_age = age.where(plausible_age)

    # Keep age cleaning identical for training and serving.
    frame["age"] = cleaned_age.fillna(-1)

    return pd.cut(
        cleaned_age,
        bins=[0, 18, 25, 35, 45, 55, 65, 200],
        labels=[
            "under_18",
            "18_24",
            "25_34",
            "35_44",
            "45_54",
            "55_64",
            "65_plus",
        ],
        right=False,
    ).astype("string")


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
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        mandatory_for=frozenset({"auto", "home"}),
        api_input="bid",
    ),
    "age": FeatureSpec(
        name="age",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        mandatory_for=frozenset({"auto"}),
        api_input="age",
    ),
    "continuous_coverage_months": FeatureSpec(
        name="continuous_coverage_months",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="continuous_coverage_months",
    ),
    "created_hour": FeatureSpec(
        name="created_hour",
        kind="numeric",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        api_input="created_at",
        derive=_derive_created_hour,
    ),
    "created_dayofweek": FeatureSpec(
        name="created_dayofweek",
        kind="numeric",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        api_input="created_at",
        derive=_derive_created_dayofweek,
    ),
    "is_workday": FeatureSpec(
        name="is_workday",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        api_input="created_at",
        derive=_derive_is_workday,
    ),
    "is_married": FeatureSpec(
        name="is_married",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        api_input="marital_status",
        derive=_derive_is_married,
    ),
    "age_cohort": FeatureSpec(
        name="age_cohort",
        kind="categorical",
        source="derived",
        lead_types=frozenset({"auto", "home"}),
        mandatory_for=frozenset({"auto"}),
        api_input="age",
        derive=_derive_age_cohort,
    ),
    "state": FeatureSpec(
        name="state",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="state",
    ),
    "gender": FeatureSpec(
        name="gender",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="gender",
    ),
    "marital_status": FeatureSpec(
        name="marital_status",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="marital_status",
    ),
    "military_affiliation": FeatureSpec(
        name="military_affiliation",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="military_affiliation",
    ),
    "insured": FeatureSpec(
        name="insured",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        mandatory_for=frozenset({"auto"}),
        api_input="insured",
    ),
    "campaign_id": FeatureSpec(
        name="campaign_id",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="campaign_id",
    ),
    "traffic_tier": FeatureSpec(
        name="traffic_tier",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"auto", "home"}),
        api_input="traffic_tier",
    ),
    # -----------------------------------------------------------------------
    # Auto-only features
    # -----------------------------------------------------------------------
    "num_vehicles": FeatureSpec(
        name="num_vehicles",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto"}),
        mandatory_for=frozenset({"auto"}),
        api_input="num_vehicles",
    ),
    "num_drivers": FeatureSpec(
        name="num_drivers",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto"}),
        api_input="num_drivers",
    ),
    "num_auto_violations": FeatureSpec(
        name="num_auto_violations",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto"}),
        api_input="num_auto_violations",
    ),
    "num_auto_accidents": FeatureSpec(
        name="num_auto_accidents",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"auto"}),
        mandatory_for=frozenset({"auto"}),
        api_input="num_auto_accidents",
    ),
    "multi_vehicle": FeatureSpec(
        name="multi_vehicle",
        kind="binary",
        source="derived",
        lead_types=frozenset({"auto"}),
        mandatory_for=frozenset({"auto"}),
        api_input="num_vehicles",
        derive=_derive_multi_vehicle,
    ),
    "home_owner": FeatureSpec(
        name="home_owner",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto"}),
        mandatory_for=frozenset({"auto"}),
        api_input="home_owner",
    ),
    "dui": FeatureSpec(
        name="dui",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto"}),
        mandatory_for=frozenset({"auto"}),
        api_input="dui",
    ),
    "sr22_required": FeatureSpec(
        name="sr22_required",
        kind="binary",
        source="raw",
        lead_types=frozenset({"auto"}),
        mandatory_for=frozenset({"auto"}),
        api_input="sr22_required",
    ),
    # -----------------------------------------------------------------------
    # Home-only features
    # -----------------------------------------------------------------------
    "num_home_claims": FeatureSpec(
        name="num_home_claims",
        kind="numeric",
        source="raw",
        lead_types=frozenset({"home"}),
        api_input="num_home_claims",
    ),
    "home_property_type": FeatureSpec(
        name="home_property_type",
        kind="categorical",
        source="raw",
        lead_types=frozenset({"home"}),
        api_input="home_property_type",
    ),
}
