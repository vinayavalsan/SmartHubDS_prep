"""Unit tests for the shared metric transforms."""

import numpy as np
import pandas as pd
import pytest

from smarthub import transforms as t


def test_safe_divide_handles_zero_and_nan():
    num = pd.Series([10.0, 5.0, 1.0, 2.0])
    den = pd.Series([2.0, 0.0, np.nan, 4.0])
    result = t.safe_divide(num, den)
    assert result.tolist() == [5.0, 0.0, 0.0, 0.5]


def test_contribution_margin_zero_revenue_is_zero():
    cm = t.contribution_margin(pd.Series([5.0, 0.0]), pd.Series([10.0, 0.0]))
    assert cm.tolist() == [0.5, 0.0]


def test_win_rate_zero_count_is_zero():
    wr = t.win_rate(pd.Series([3, 0]), pd.Series([6, 0]))
    assert wr.tolist() == [0.5, 0.0]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", 1), ("True", 1), ("  TRUE ", 1), ("t", 1), ("1", 1), ("yes", 1),
        ("false", 0), ("False", 0), ("", 0), ("nonsense", 0),
    ],
)
def test_normalize_won_variants(raw, expected):
    out = t.normalize_won(pd.Series([raw]))
    assert out.iloc[0] == expected


def test_normalize_won_is_nullable_int():
    out = t.normalize_won(pd.Series(["true", "false"]))
    assert str(out.dtype) == "Int64"


def _raw_leads():
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
            "bid": [5.0, 4.0, 0.0, 2.0],  # third row filtered out (bid <= 0)
            "rev": [10.0, 0.0, 8.0, None],
            "lead_created_at": ["x", "x", "x", "x"],  # dropped
        }
    )


def test_prepare_leads_frame_cleaning():
    out = t.prepare_leads_frame(_raw_leads())

    # bid <= 0 row removed
    assert len(out) == 3
    assert (out["bid"] > 0).all()

    # unused column dropped
    assert "lead_created_at" not in out.columns

    # blank/missing state -> NAvail
    assert out.loc[out["id"] == 2, "state"].iloc[0] == "NAvail"

    # derived columns
    assert out.loc[out["id"] == 1, "payout"].iloc[0] == 5.0  # won(1)*bid(5)
    assert out.loc[out["id"] == 1, "profit"].iloc[0] == 5.0  # rev(10)-payout(5)
    # rev was NaN for id 4 -> filled 0
    assert out.loc[out["id"] == 4, "rev"].iloc[0] == 0.0

    # id columns are nullable ints, time parts present
    assert str(out["campaign_id"].dtype) == "Int64"
    assert "created_hour" in out.columns


def test_aggregate_leads_metrics():
    df = t.prepare_leads_frame(_raw_leads())
    agg = t.aggregate_leads(df, "state")
    assert {"count", "rev", "profit", "cm", "winrate"}.issubset(agg.columns)
    # cm is bounded and winrate within [0, 1]
    assert (agg["winrate"].between(0, 1)).all()


def test_build_metric_plot_data_winrate():
    df = t.prepare_leads_frame(_raw_leads())
    plot = t.build_metric_plot_data(df, ["state"], "winrate")
    assert "value" in plot.columns
    assert (plot["value"].between(0, 1)).all()


def _monitoring_frame():
    return pd.DataFrame(
        {
            "datetime_min": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"]
            ),
            "state": ["NY", "NY", "NY"],
            "campaign_id": [1, 1, 1],
            "revenue_measured": [100.0, 200.0, 0.0],
            "revenue_expected": [120.0, 180.0, 0.0],
            "payout": [60.0, 90.0, 0.0],
            "num_opportunities": [10, 20, 0],
            "num_won": [4, 5, 0],
        }
    )


def test_add_monitoring_derived_columns():
    out = t.add_monitoring_derived_columns(_monitoring_frame())
    assert out.loc[0, "profit"] == 40.0
    assert out.loc[0, "winrate"] == pytest.approx(0.4)
    # zero-revenue / zero-opportunity row stays finite
    assert out.loc[2, "cm_measured"] == 0.0
    assert out.loc[2, "winrate"] == 0.0


def test_aggregate_monitoring_resamples():
    agg = t.aggregate_monitoring(_monitoring_frame(), freq="D")
    assert len(agg) == 1
    assert agg.loc[0, "revenue_measured"] == 300.0
    assert agg.loc[0, "num_won"] == 9
