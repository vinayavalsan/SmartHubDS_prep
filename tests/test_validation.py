"""Tests for the data-validation layer (detect + report, never fix)."""

import pandas as pd
import pytest

from smarthub.validation import validate_leads
from smarthub.validation import report as vreport


def _raw():
    return pd.DataFrame(
        {
            "id": [1, 2, 2],                       # duplicate id (2)
            "created_at": pd.to_datetime(["2026-07-01"] * 3),
            "lead_type_id": [6, 6, 1],
            "bid": [5.0, 0.0, -1.0],               # negative bid on row 3
            "won": ["true", "false", ""],          # 'false' should never appear
            "erred": ["false", "true", "false"],
            "age": [40, -7648, 33],                # implausible age on row 2
            "state": ["CA", "ZZ", "TX"],           # invalid state ZZ
            "gender": ["Male", "Male", "Female"],
            "insured": ["false", "false", "false"],
            "current_carrier": ["Geico", "", "Allstate"],  # carrier + not insured
            "num_vehicles": [2, 1, None],
            "pst_hour": [None, None, None],        # 100% empty
            "exp_rev": [10.0, 0.0, 5.0],
            "traffic_tier": ["a", "a", "b"],
        }
    )


def _find(report, column, needle):
    return [
        v for v in report.rule_violations
        if v.column == column and needle in v.check
    ]


def test_flags_range_domain_and_uniqueness():
    pytest.importorskip("pandera")             # schema layer needs pandera
    rep = validate_leads(_raw())
    assert rep.schema_checked is True          # pandera installed in CI/dev
    assert _find(rep, "age", "out of range")
    assert _find(rep, "bid", "negative")
    assert _find(rep, "state", "not in allowed")
    assert _find(rep, "id", "duplicate")


def test_check_labels_are_clean_and_stable():
    # isin dumps the whole domain set (unstable ordering) unless normalised.
    pytest.importorskip("pandera")
    rep = validate_leads(_raw())
    for v in rep.rule_violations:
        assert "{" not in v.check and "isin(" not in v.check


def test_cross_field_current_carrier_when_not_insured():
    rep = validate_leads(_raw())
    assert rep.cross_field["current_carrier_when_not_insured"] == 2


def test_cross_field_won_true_without_bid():
    raw = _raw()
    raw.loc[raw["id"] == 1, "won"] = "true"
    raw.loc[raw["id"] == 1, "bid"] = 0.0
    rep = validate_leads(raw)
    assert rep.cross_field["won_true_without_bid"] >= 1


def test_missing_catalogue_and_high_missing():
    rep = validate_leads(_raw(), high_missing_threshold=0.5)
    assert rep.missing["pst_hour"] == 1.0        # 100% empty
    assert "pst_hour" in rep.high_missing
    assert rep.missing["state"] == 0.0           # fully populated


def test_batch_metrics():
    rep = validate_leads(_raw())
    m = rep.metrics
    assert m["rows"] == 3
    assert m["won_false_count"] == 1             # the illegal 'false'
    assert m["pst_hour_populated"] == 0.0
    assert abs(m["age_implausible_rate"] - 1 / 3) < 1e-9
    assert abs(m["exp_rev_coverage"] - 2 / 3) < 1e-9


def test_schema_drift_detects_missing_expected_column():
    raw = _raw().drop(columns=["state"])
    rep = validate_leads(raw)
    assert any("missing columns" in i and "state" in i for i in rep.schema_issues)


def test_clean_batch_passes():
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
    rep = validate_leads(_raw())
    md = vreport.to_markdown(rep, "auto")
    assert "Data quality — auto" in md
    title, fields = vreport.slack_group(rep)
    assert title == "Data quality"
    assert "Status" in fields and "Cross-field flags" in fields


def test_validate_never_mutates_input():
    raw = _raw()
    before = raw.copy()
    validate_leads(raw)
    pd.testing.assert_frame_equal(raw, before)
