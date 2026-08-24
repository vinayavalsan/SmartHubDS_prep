"""Canonical SmartHub lead-type registry.

Add each lead type once to ``LEAD_TYPES``. Production code and tests resolve
IDs and names through the generic lookup functions below; no per-lead-type
constants are required.
"""

from __future__ import annotations

LEAD_TYPES: dict[str, int] = {
    "auto": 6,
    "home": 1,
}


def lead_type_id(name: str) -> int:
    """Return the registered ID for a lead-type name."""
    normalized = name.strip().lower()
    try:
        return LEAD_TYPES[normalized]
    except KeyError:
        raise ValueError(
            f"Unknown lead type {name!r}. Known lead types: {sorted(LEAD_TYPES)}"
        ) from None


def lead_type_name(lead_type_id: int) -> str:
    """Return the registered name for a lead-type ID."""
    matches = [
        name
        for name, registered_id in LEAD_TYPES.items()
        if registered_id == lead_type_id
    ]
    if matches:
        return matches[0]
    raise ValueError(
        f"Unknown lead_type_id {lead_type_id!r}. "
        f"Known IDs: {sorted(LEAD_TYPES.values())}"
    )


def all_lead_types() -> dict[str, int]:
    """Return a copy of the complete name-to-ID mapping."""
    return dict(LEAD_TYPES)


def all_lead_type_ids() -> tuple[int, ...]:
    """Return every registered lead-type ID in registry order."""
    return tuple(LEAD_TYPES.values())
