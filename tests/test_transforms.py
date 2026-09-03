"""Unit tests for core and monitoring transforms."""

import numpy as np
import pandas as pd
import pytest

from smarthub.core import transforms as core_t
from smarthub.monitoring import transforms as monitoring_t


def test_safe_divide_handles_zero_and_nan():
    """safe_divide returns 0 for zero or NaN denominators."""
    num = pd.Series([10.0, 5.0, 1.0, 2.0])
    den = pd.Series([2.0, 0.0, np.nan, 4.0])
    result = monitoring_t.safe_divide(num, den)
    assert result.tolist() == [5.0, 0.0, 0.0, 0.5]


def test_contribution_margin_zero_revenue_is_zero():
    """contribution_margin is 0 when revenue is 0."""
    cm = monitoring_t.contribution_margin(pd.Series([5.0, 0.0]), pd.Series([10.0, 0.0]))
    assert cm.tolist() == [0.5, 0.0]


def test_win_rate_zero_count_is_zero():
    """win_rate is 0 when the count is 0."""
    wr = monitoring_t.win_rate(pd.Series([3, 0]), pd.Series([6, 0]))
    assert wr.tolist() == [0.5, 0.0]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", 1),
        ("True", 1),
        ("  TRUE ", 1),
        ("t", 1),
        ("1", 1),
        ("yes", 1),
        ("false", 0),
        ("False", 0),
        ("", 0),
        ("nonsense", 0),
    ],
)
def test_normalize_won_variants(raw, expected):
    """normalize_won maps truthy strings to 1 and everything else to 0."""
    out = core_t.normalize_won(pd.Series([raw]))
    assert out.iloc[0] == expected


def test_normalize_won_is_nullable_int():
    """normalize_won returns a nullable Int64 series."""
    out = core_t.normalize_won(pd.Series(["true", "false"]))
    assert str(out.dtype) == "Int64"


def _raw_leads():
    """Return a raw lead frame for the prepare/aggregate transform tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "created_at": pd.to_datetime(
                [
                    "2026-06-07 01:00:00",
                    "2026-06-07 02:00:00",
                    "2026-06-08 05:00:00",
                    "2026-06-08 06:00:00",
                ]
            ),
            "campaign_id": [10.0, 10.0, 11.0, 11.0],
            "lead_type_id": [6.0, 6.0, 6.0, 6.0],
            "state": ["NY", "  ", None, "TX"],
            "won": ["true", "false", "true", "true"],
            "accepted_listings": [1, 0, 1, 0],
            "bid": [5.0, 4.0, 0.0, 2.0],  # third row filtered out (bid <= 0)
            "rev": [10.0, 0.0, 8.0, None],
            "lead_created_at": ["x", "x", "x", "x"],  # dropped
        }
    )


def test_prepare_leads_frame_cleaning():
    """prepare_leads_frame filters, cleans, and derives lead columns."""
    out = core_t.prepare_leads_frame(_raw_leads())

    # bid <= 0 row removed
    assert len(out) == 3
    assert (out["bid"] > 0).all()

    # unused column dropped
    assert "lead_created_at" not in out.columns

    # blank/missing state -> NAvail
    assert out.loc[out["id"] == 2, "state"].iloc[0] == "NAvail"

    # derived columns use sold = accepted_listings > 0
    assert out.loc[out["id"] == 1, "sold"].iloc[0] == 1
    assert out.loc[out["id"] == 1, "bid_cost"].iloc[0] == 5.0
    assert out.loc[out["id"] == 1, "realized_profit"].iloc[0] == 5.0
    assert out.loc[out["id"] == 1, "realized_revenue"].iloc[0] == 10.0

    # won but not sold -> no bid cost; rev NaN is normalized to zero
    assert out.loc[out["id"] == 4, "sold"].iloc[0] == 0
    assert out.loc[out["id"] == 4, "bid_cost"].iloc[0] == 0.0
    assert out.loc[out["id"] == 4, "realized_profit"].iloc[0] == 0.0
    assert out.loc[out["id"] == 4, "rev"].iloc[0] == 0.0

    # id columns are nullable ints, time parts present
    assert str(out["campaign_id"].dtype) == "Int64"
    assert "created_hour" in out.columns


def test_aggregate_leads_metrics():
    """aggregate_leads produces standardized business metric columns."""
    df = core_t.prepare_leads_frame(_raw_leads())
    agg = monitoring_t.aggregate_leads(df, "state")
    assert {
        "count",
        "realized_revenue",
        "bid_cost",
        "realized_profit",
        "cm",
        "winrate",
    }.issubset(agg.columns)
    # cm is bounded and winrate within [0, 1]
    assert (agg["winrate"].between(0, 1)).all()


def test_build_metric_plot_data_winrate():
    """build_metric_plot_data returns a value column bounded to [0, 1]."""
    df = core_t.prepare_leads_frame(_raw_leads())
    plot = monitoring_t.build_metric_plot_data(df, ["state"], "winrate")
    assert "value" in plot.columns
    assert (plot["value"].between(0, 1)).all()


def _monitoring_frame():
    """Return a monitoring frame for the aggregation transform tests."""
    return pd.DataFrame(
        {
            "datetime_min": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"]
            ),
            "state": ["NY", "NY", "NY"],
            "campaign_id": [1, 1, 1],
            "realized_revenue": [100.0, 200.0, 0.0],
            "expected_revenue": [120.0, 180.0, 0.0],
            "bid_cost": [60.0, 90.0, 0.0],
            "num_opportunities": [10, 20, 0],
            "num_won": [4, 5, 0],
        }
    )


def test_add_monitoring_derived_columns():
    """add_monitoring_derived_columns computes profit/winrate and stays finite."""
    out = monitoring_t.add_monitoring_derived_columns(_monitoring_frame())
    assert out.loc[0, "realized_profit"] == 40.0
    assert out.loc[0, "winrate"] == pytest.approx(0.4)
    # zero-revenue / zero-opportunity row stays finite
    assert out.loc[2, "cm_measured"] == 0.0
    assert out.loc[2, "winrate"] == 0.0


def test_leads_to_monitoring_base():
    """leads_to_monitoring_base reshapes prepared leads into monitoring rows."""
    # prepared-leads shape (won as 0/1, bid present, expected_revenue present)
    leads = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                ["2026-06-20 01:00", "2026-06-20 02:00", "2026-06-20 03:00"]
            ),
            "state": ["NY", "NY", "CA"],
            "campaign_id": pd.array([1, 1, 2], dtype="Int64"),
            "won": pd.array([1, 0, 1], dtype="Int64"),
            "accepted_listings": pd.array([1, 0, 1], dtype="Int64"),
            "rev": [10.0, 0.0, 20.0],
            "expected_revenue": [12.0, 8.0, 25.0],
            "bid": [5.0, 7.0, 9.0],
        }
    )
    base = monitoring_t.leads_to_monitoring_base(leads)
    assert list(base["num_opportunities"]) == [1, 1, 1]
    assert list(base["num_won"]) == [1, 0, 1]
    assert base["realized_revenue"].sum() == 30.0
    assert base["expected_revenue"].sum() == 45.0
    assert base["bid_cost"].sum() == 14.0
    # feeds straight into aggregate_monitoring
    agg = monitoring_t.aggregate_monitoring(base, freq="D")
    assert agg.loc[0, "num_opportunities"] == 3
    assert agg.loc[0, "num_won"] == 2
    assert agg.loc[0, "winrate"] == pytest.approx(2 / 3)


def test_aggregate_monitoring_resamples():
    """aggregate_monitoring resamples rows to the requested frequency."""
    agg = monitoring_t.aggregate_monitoring(_monitoring_frame(), freq="D")
    assert len(agg) == 1
    assert agg.loc[0, "realized_revenue"] == 300.0
    assert agg.loc[0, "num_won"] == 9


def test_cumulative_winrate_curves():
    """cumulative_winrate_curves computes below/above winrate per threshold."""
    df = pd.DataFrame({"bid": [1.0, 2.0, 3.0, 4.0], "won": [0, 0, 1, 1]})
    curves = monitoring_t.cumulative_winrate_curves(df, bucket_size=1.0).set_index(
        "threshold"
    )
    # thresholds span 1..4
    assert list(curves.index) == [1.0, 2.0, 3.0, 4.0]
    # at x=2: below {1,2} -> 0/2; above {3,4} -> 2/2
    assert curves.loc[2.0, "winrate_below"] == 0.0
    assert curves.loc[2.0, "winrate_above"] == 1.0
    assert curves.loc[2.0, "winrate_delta"] == 1.0
    # at x=4: everything below -> 2/4; nothing above -> NaN
    assert curves.loc[4.0, "winrate_below"] == 0.5
    assert pd.isna(curves.loc[4.0, "winrate_above"])


def test_cumulative_winrate_curves_missing_cols():
    """cumulative_winrate_curves returns empty when required cols are absent."""
    assert monitoring_t.cumulative_winrate_curves(pd.DataFrame({"x": [1]})).empty


def test_funnel_counts():
    """funnel_counts tallies pings, wins, and resold listings per stage."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "won": [1, 0, 1],
            "accepted_listings": [2, 0, 0],
        }
    )
    funnel = monitoring_t.funnel_counts(df).set_index("stage")["count"]
    assert funnel["Pings"] == 3
    assert funnel["Won (partner accepted bid)"] == 2
    assert funnel["Sold"] == 1
