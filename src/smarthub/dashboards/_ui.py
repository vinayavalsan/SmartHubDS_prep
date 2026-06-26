"""Reusable Streamlit/Plotly helpers shared across dashboards."""

from __future__ import annotations

import pandas as pd

# All metrics the leads plots can render, with the columns each one needs.
_METRIC_REQUIRED_COLS = {
    "count": ["id"],
    "rev": ["rev"],
    "bid": ["bid"],
    "payout": ["payout"],
    "profit": ["profit"],
    "cm": ["profit", "rev"],
    "winrate": ["won", "id"],
}

# Columns that make sensible legend / grouping dimensions. A whitelist avoids
# offering high-cardinality columns (id, zip, raw timestamps) that would explode
# a chart into thousands of traces.
LEGEND_WHITELIST = [
    "state",
    "campaign_id",
    "lead_type_id",
    "account_id",
    "bidding_strategy_id",
    "insured",
    "home_owner",
    "dui",
    "num_vehicles",
    "num_drivers",
    "num_auto_violations",
    "num_auto_accidents",
    "continuous_coverage_months",
    "created_hour",
    "created_dayofweek",
    "military_affiliation",
    "gender",
    "marital_status",
    "age"
]


def get_plot_metric_options(df: pd.DataFrame) -> list[str]:
    """Return the metrics that can be computed given the columns present."""
    return [
        metric
        for metric, cols in _METRIC_REQUIRED_COLS.items()
        if all(col in df.columns for col in cols)
    ]


def get_legend_options(df: pd.DataFrame) -> list[str]:
    """Return ``["None"]`` plus the whitelisted categorical columns present."""
    present = [c for c in LEGEND_WHITELIST if c in df.columns]
    return ["None"] + sorted(present)


def get_default_option(options: list, preferred_value) -> int:
    """Index of ``preferred_value`` in ``options``, or 0 if absent."""
    return options.index(preferred_value) if preferred_value in options else 0


def style_figure(fig):
    """Apply consistent gridlines to a Plotly figure."""
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    return fig
