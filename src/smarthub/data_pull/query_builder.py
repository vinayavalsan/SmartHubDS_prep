"""Registry-driven column plan for the data pull.

Turns ``field_registry.RAW_FIELD_REGISTRY`` into the ordered list of raw column
names the pull extracts, so extraction is driven by the same single source of
truth as validation. ``data_pull.models.LEADS_COLUMNS`` resolves these names to
ORM columns and assembles the actual SELECT (``leads_select`` etc.).

Only registry metadata is read here — this module imports **no** ORM, so
``models.py`` can import it to build ``LEADS_COLUMNS`` without an import cycle.
"""

from __future__ import annotations

from . import field_registry


def leads_column_names(lead_type_id: int | None = None) -> list[str]:
    """Ordered raw column names the pull selects.

    Extraction pulls the same columns for every lead type — row filtering by
    ``lead_type_id`` is a WHERE clause, not a column filter — so ``lead_type_id``
    is accepted for symmetry (and future per-type projection) but does not narrow
    the column set today. PII fields (``pii=True``) and disabled fields are
    always excluded.

    Inputs
    ------
    lead_type_id : int | None
        Present for forward-compatibility; ignored for column selection today.

    Returns
    -------
    list[str]
        The raw column names to select, in registry order.
    """
    return field_registry.field_names()


def pulled_field_specs() -> list[field_registry.RawFieldSpec]:
    """Return the ``RawFieldSpec`` for each column the pull selects, in order."""
    return [field_registry.get(name) for name in leads_column_names()]
