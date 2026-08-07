"""Custom validation rules referenced from the raw-field registry.

Field-specific checks that plain ``ValidationSpec`` metadata can't express live
here as small reusable functions, referenced from a field's
``ValidationSpec.custom_rule``. This is a **leaf** module (it imports only
pandas — nothing from ``data_pull``), so the metadata-only ``field_registry``
can reference these functions directly without creating an import cycle.

Each rule takes the column's ``pandas.Series`` and returns a ``ValidationResult``
(a stable check label, the number of violating rows, and a few example values),
which ``validation_runner`` folds into the ``ValidationReport`` exactly like the
pandera-driven checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# US states + territories (2-letter). Data for validate_us_state; kept here with
# the rule that uses it rather than in validation_rules, per the registry design.
US_STATES = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "DC",
        "PR",
        "GU",
        "VI",
        "AS",
        "MP",
    }
)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a custom rule for one column.

    Inputs
    ------
    check : str
        A short, stable label for the violation (mirrors the labels the
        pandera-driven checks produce, e.g. ``"not in allowed values"``).
    count : int
        Number of rows that violate the rule.
    examples : list
        A few example offending values (for the report), capped by the rule.
    """

    check: str
    count: int
    examples: list = field(default_factory=list)


def _present_values(series: pd.Series) -> pd.Series:
    """Return the non-null, non-blank (stripped) values of ``series``."""
    stripped = series.astype("string").str.strip()
    return stripped[~(stripped.isna() | (stripped == ""))]


def validate_us_state(series: pd.Series) -> ValidationResult:
    """Flag present values that aren't a valid US state / territory code.

    Null/blank values are ignored (missingness is catalogued separately),
    matching the previous pandera ``isin(..., nullable=True)`` behaviour.

    Inputs
    ------
    series : pandas.Series
        The ``state`` column.

    Returns
    -------
    ValidationResult
        Count + examples of present values outside ``US_STATES``.
    """
    present = _present_values(series)
    invalid = present[~present.isin(US_STATES)]
    examples = invalid.dropna().unique().tolist()[:5]
    return ValidationResult(
        check="not in allowed values",
        count=int(len(invalid)),
        examples=examples,
    )
