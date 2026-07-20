"""Data validation for SmartHub — detect & report, never fix.

D1 deliverable: validate the raw ``lead_pings`` batch at pull time. This layer
is **warn + report only** — it flags bad rows and catalogues missing-value
patterns, but never drops, imputes, or caps anything (that stays in
``feature_engineering.build_training_table`` and is a separate decision).

Public API:
    from smarthub.validation import validate_leads, ValidationReport
"""

from .validate import RuleViolation, ValidationReport, validate_leads

__all__ = ["validate_leads", "ValidationReport", "RuleViolation"]
