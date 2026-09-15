"""
Piece 2 — Payload builder: one snapshot row -> a faithful `/recommend_bid` body.

Standalone on purpose: it does NOT import `smarthub`, so the harness runs with
just pandas available (or anywhere). The field set below is the documented
`BidRequest` contract (docs/API_INTEGRATION.md + server/predict.BidRequest),
cross-checked against feature_engineering/feature_registry.FEATURES.

Two rules that keep the request faithful to what the model saw in training:

* POINT-IN-TIME: only pre-bid inputs are sent. Outcome columns on the row
  (`bid`, `won`, `rev`, `accepted`, ...) are never included.
* created_at VERBATIM (no timezone conversion). The model derives its timing
  features with `_created_at_pacific` = `to_datetime(created_at, utc=True)
  .tz_convert("America/Los_Angeles")` — i.e. it INTERPRETS the value as UTC.
  Training saw the raw `lead_pings.created_at`, so we must send that same raw
  value; converting it here would shift every timing feature. (The API doc's
  "Pacific" note is about how the model reads it, not how you format it.)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

# Always sent (the request is rejected 422 without these). Map: API field -> row column.
REQUIRED = {
    "expected_revenue": "expected_revenue",  # coerced from lead_pings.exp_rev
    "lead_type_id": "lead_type_id",
    "campaign_id": "campaign_id",
    "source_type_id": "source_type_id",
    "traffic_tier": "traffic_tier",
    "state": "state",                        # the only mandatory lead attribute
    "created_at": "created_at",              # sent verbatim (see module docstring)
    "lead_ping_id": "id",                    # lead_pings PK
}

# Sent only when present/non-null (the model imputes when absent). API field == column.
OPTIONAL = [
    "account_id",
    "insured", "home_owner", "dui", "sr22_required", "military_affiliation",
    "gender", "marital_status", "current_carrier", "home_property_type",
    "age", "num_vehicles", "num_drivers", "num_auto_violations",
    "num_auto_accidents", "num_home_claims", "continuous_coverage_months",
]

# Outcome / post-bid columns that must NEVER be sent (kept only for the analyzer).
OUTCOME_COLUMNS = frozenset({
    "bid", "rev", "won", "accepted", "accepted_listings", "response_ms",
    "erred", "error_reason_id", "realized_revenue", "sold", "bid_cost", "profit",
})

_DT_FMT = "%Y-%m-%d %H:%M:%S"


class PayloadError(ValueError):
    """A row is missing a required field and cannot form a valid request."""


def _is_missing(v: Any) -> bool:
    """True for None and NaN/NA (works without importing pandas/numpy)."""
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    # pandas.NA / NaT compare unequal to themselves too, but guard by type name
    tname = type(v).__name__
    if tname in ("NaTType", "NAType"):
        return True
    return False


def _clean(v: Any) -> Any:
    """Convert a numpy/pandas scalar to a JSON-native value; None if missing."""
    if _is_missing(v):
        return None
    if hasattr(v, "item"):          # numpy scalar -> python int/float/bool
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    return v


def _fmt_created_at(v: Any) -> str | None:
    """Format created_at verbatim as 'YYYY-MM-DD HH:MM:SS' (no tz conversion)."""
    if _is_missing(v):
        return None
    if isinstance(v, str):
        return v.strip()[:19]
    if isinstance(v, datetime):     # pandas.Timestamp is a datetime subclass
        return v.strftime(_DT_FMT)
    if hasattr(v, "strftime"):
        return v.strftime(_DT_FMT)
    return str(v)


def build_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Turn one snapshot row (a dict) into a `/recommend_bid` request body.

    Raises PayloadError if a required field is missing (so the caller can skip
    or count it rather than send a request that is guaranteed to 422).
    """
    body: dict[str, Any] = {}

    for api_field, col in REQUIRED.items():
        raw = row.get(col)
        val = _fmt_created_at(raw) if api_field == "created_at" else _clean(raw)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise PayloadError(f"missing required field '{api_field}' (column '{col}')")
        # ids must be ints, not floats, for a clean request
        if api_field in ("lead_type_id", "campaign_id", "source_type_id", "lead_ping_id"):
            val = int(val)
        body[api_field] = val

    for api_field in OPTIONAL:
        val = _clean(row.get(api_field))
        if val is None:
            continue
        if api_field == "account_id":
            val = int(val)
        body[api_field] = val

    return body


def required_columns() -> list[str]:
    """Columns the builder needs from the snapshot (required + optional + ids)."""
    cols = list(dict.fromkeys(list(REQUIRED.values()) + OPTIONAL))
    return cols
