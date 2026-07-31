"""Tests for the data-validation layer (detect + report, never fix)."""

import pandas as pd
import pytest

from smarthub.data_pull import validation_report as vreport
from smarthub.data_pull.validation_runner import validate_leads


def _raw():
    """Return a raw lead batch seeded with one of each validation problem."""
    return pd.DataFrame(
        {
            "id": [1, 2, 2],  # duplicate id (2)
            "created_at": pd.to_datetime(["2026-07-01"] * 3),
            "lead_type_id": [6, 6, 1],
            "bid": [5.0, 0.0, -1.0],  # negative bid on row 3
            "won": ["true", "false", ""],  # 'false' should never appear
            "erred": ["false", "true", "false"],
            "age": [40, -7648, 33],  # implausible age on row 2
            "state": ["CA", "ZZ", "TX"],  # invalid state ZZ
            "gender": ["Male", "Male", "Female"],
            "insured": ["false", "false", "false"],
            "current_carrier": ["Geico", "", "Allstate"],  # carrier + not insured
            "num_vehicles": [2, 1, None],
            "pst_hour": [None, None, None],  # 100% empty
            "exp_rev": [10.0, 0.0, 5.0],
            "traffic_tier": ["a", "a", "b"],
        }
    )


def _find(report, column, needle):
    """Return rule violations matching a column and a check-name substring."""
    return [
        v for v in report.rule_violations if v.column == column and needle in v.check
    ]


def test_flags_range_domain_and_uniqueness():
    """Range/domain/uniqueness violations are flagged on eligible rows."""
    pytest.importorskip("pandera")  # schema layer needs pandera
    raw = _raw()

    # Ordinary validation now runs only after erred and auction-ineligible rows
    # are excluded. Keep all three rows in the validation population while
    # preserving the seeded age/state/bid/id violations.
    raw["erred"] = ["false", "false", "false"]
    raw["bid"] = [5.0, 2.0, -1.0]
    raw["won"] = ["true", "", "true"]  # row 3 stays eligible despite bid < 0

    rep = validate_leads(raw)
    assert rep.schema_checked is True  # pandera installed in CI/dev
    assert rep.validated_rows == 3
    assert _find(rep, "age", "out of range")
    assert _find(rep, "bid", "negative")
    assert _find(rep, "state", "not in allowed")
    assert _find(rep, "id", "duplicate")


def test_check_labels_are_clean_and_stable():
    """Check labels are normalised and free of unstable isin/set dumps."""
    pytest.importorskip("pandera")
    rep = validate_leads(_raw())
    for v in rep.rule_violations:
        assert "{" not in v.check and "isin(" not in v.check


def test_cross_field_current_carrier_when_not_insured():
    """Ordinary cross-field checks use only training-eligible rows."""
    rep = validate_leads(_raw())

    # Row 1 is training-eligible and has carrier + insured=false.
    # Row 3 has the same raw inconsistency but is auction-ineligible, so it is
    # intentionally excluded before ordinary cross-field validation.
    assert rep.cross_field["current_carrier_when_not_insured"] == 1


def test_cross_field_won_true_without_bid():
    """Cross-field check flags won='true' rows with no bid."""
    raw = _raw()
    raw.loc[raw["id"] == 1, "won"] = "true"
    raw.loc[raw["id"] == 1, "bid"] = 0.0
    rep = validate_leads(raw)
    assert rep.cross_field["won_true_without_bid"] >= 1


def test_missing_catalogue_and_high_missing():
    """Missing-value catalogue and high-missing columns are reported."""
    rep = validate_leads(_raw(), high_missing_threshold=0.5)
    assert rep.missing["pst_hour"] == 1.0  # 100% empty
    assert "pst_hour" in rep.high_missing
    assert rep.missing["state"] == 0.0  # fully populated


def test_batch_metrics():
    """Batch metrics describe only the training-eligible population."""
    rep = validate_leads(_raw())
    m = rep.metrics

    assert rep.total_rows == 3
    assert rep.erred_rows == 1
    assert rep.auction_excluded_rows == 1
    assert rep.validated_rows == 1

    assert m["rows"] == 1
    assert m["won_false_count"] == 0
    assert m["pst_hour_populated"] == 0.0
    assert m["age_implausible_rate"] == 0.0
    assert m["exp_rev_coverage"] == 1.0


def test_schema_drift_detects_missing_expected_column():
    """Schema drift is reported when an expected column is missing."""
    raw = _raw().drop(columns=["state"])
    rep = validate_leads(raw)
    assert any("missing columns" in i and "state" in i for i in rep.schema_issues)


def test_clean_batch_passes():
    """A clean batch produces no rule or cross-field violations."""
    raw = pd.DataFrame(
        {
            "id": [1, 2],
            "created_at": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "lead_type_id": [6, 6],
            "bid": [5.0, 8.0],
            "won": ["true", ""],
            "erred": ["false", "false"],
            "age": [40, 55],
            "state": ["CA", "TX"],
            "gender": ["Male", "Female"],
            "insured": ["true", "false"],
            "current_carrier": ["Geico", ""],
            "num_vehicles": [2, 1],
            "exp_rev": [10.0, 9.0],
        }
    )
    rep = validate_leads(raw)
    assert rep.rule_violations == []
    assert rep.cross_field_hits == {}


def test_report_renderers_do_not_crash():
    """Markdown and Slack report renderers run without crashing."""
    rep = validate_leads(_raw())
    md = vreport.to_markdown(rep, "auto")
    assert "Data quality — auto" in md
    title, fields = vreport.slack_group(rep)
    assert title == "Data quality"
    assert "Status" in fields and "Cross-field flags" in fields


def test_validate_never_mutates_input():
    """validate_leads never mutates its input frame."""
    raw = _raw()
    before = raw.copy()
    validate_leads(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_high_missing_scoped_by_lead_type():
    """An empty other-product column isn't flagged high-missing for a lead type."""
    raw = _raw()
    raw["home_property_type"] = [None, None, None]  # empty on an auto pull

    # Unscoped (no lead_type_id): the empty home column is flagged high-missing.
    assert "home_property_type" in validate_leads(raw).high_missing

    # Scoped to auto: home_property_type is out-of-scope, so it's not flagged...
    auto = validate_leads(raw, lead_type_id=6)
    assert "home_property_type" not in auto.high_missing
    # ...but a shared empty column (pst_hour) is still flagged.
    assert "pst_hour" in auto.high_missing
