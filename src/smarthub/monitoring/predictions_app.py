"""Streamlit page: browse the prediction/inference log (one row per API call).

Shows the full inference flow for each `/recommend_bid` (and `/explain_bid`)
call -- model name/version, the request payload, the prediction it produced,
timestamps, and the **TAT** (turnaround time: server-side time from receiving
the request to returning the result) -- with sorting on TAT so the slowest
requests surface first. Reads the same prediction-log DB the API writes to
(`SMARTHUB_PREDICTION_LOG_DB_URL`), so nobody has to query the database by hand.

Run standalone with:
    streamlit run src/smarthub/monitoring/predictions_app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

# Columns surfaced in the table, in order. (input_features -> "payload".)
_DISPLAY_ORDER = [
    "created_at",
    "updated_at",
    "tat_seconds",
    "endpoint",
    "status",
    "model_name",
    "model_version",
    "lead_type_name",
    "decision_path",
    "recommended_bid",
    "recommended_bid_predicted_win_rate",
    "expected_revenue",
    "payload",
    "prediction_id",
    "lead_ping_id",
]


def _store():
    """Construct the prediction-log store (lazily; import kept local so the
    page loads even where the ML/serving deps aren't installed)."""
    from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore

    return PredictionLogStore()


@st.cache_data(ttl=15)
def load_predictions(limit: int, lead_type_id, status) -> pd.DataFrame:
    """Load recent prediction-log rows into a DataFrame (cached ~15s)."""
    rows = _store().recent(
        limit=limit,
        lead_type_id=lead_type_id if lead_type_id != "all" else None,
        status=status if status != "all" else None,
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Compact the request payload dict into a readable JSON string column.
    if "input_features" in df.columns:
        df["payload"] = df["input_features"].apply(
            lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
        )
    # Numeric TAT in seconds (stored as Decimal) -> float so sorting behaves.
    if "tat_seconds" in df.columns:
        df["tat_seconds"] = pd.to_numeric(df["tat_seconds"], errors="coerce")
    for c in ("created_at", "updated_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def render_inference_log(as_section: bool = False, kp: str = "il"):
    """Render the inference-log viewer (filters, TAT metrics, sortable table).

    Reused in two places: the standalone **Predictions** page (``main``) and as
    a section embedded on the **Monitoring** page. ``kp`` prefixes every widget
    key so the two instances never collide when both live in one app.
    """
    if as_section:
        st.subheader("Inference log")
    else:
        st.title("Predictions — inference log")
    st.caption(
        "One row per API call: model, request payload, prediction, timestamps, "
        "and TAT (time from request received to result returned)."
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        lead_type_id = st.selectbox(
            "Lead type", options=["all", 6, 1], format_func=str, key=f"{kp}_lt"
        )
    with c2:
        status = st.selectbox(
            "Status", options=["all", "success", "error"], key=f"{kp}_st"
        )
    with c3:
        limit = st.number_input(
            "Max rows", min_value=10, max_value=10000, value=500, step=50,
            key=f"{kp}_lim",
        )
    with c4:
        sort_choice = st.selectbox(
            "Sort by TAT",
            options=["Slowest first", "Fastest first", "Newest first"],
            key=f"{kp}_sort",
        )

    if st.button("🔄 Reload", key=f"{kp}_reload"):
        load_predictions.clear()

    try:
        df = load_predictions(int(limit), lead_type_id, status)
    except Exception as exc:  # noqa: BLE001 -- surface DB issues in the UI
        st.error(f"Could not read the prediction log: {exc}")
        st.info(
            "Is SMARTHUB_PREDICTION_LOG_DB_URL pointing at the same DB the API "
            "writes to? (Defaults to the shared Postgres in the Docker stack.)"
        )
        return

    if df.empty:
        st.info("No predictions logged yet.")
        return

    # --- Summary metrics (TAT distribution + 1s-SLA adherence), in seconds ---
    tat = df["tat_seconds"].dropna()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Requests", len(df))
    if not tat.empty:
        m2.metric("TAT p50", f"{tat.median():.3f} s")
        m3.metric("TAT p95", f"{tat.quantile(0.95):.3f} s")
        m4.metric("TAT p99", f"{tat.quantile(0.99):.3f} s")
        within = (tat <= 1.0).mean() * 100
        m5.metric("Within 1s TAT", f"{within:.1f}%")

    # Optional TAT floor filter -- e.g. "show me everything over 0.2 s".
    if not tat.empty:
        min_tat = st.slider(
            "Min TAT (seconds) to show",
            min_value=0.0,
            max_value=float(max(1.0, tat.max())),
            value=0.0,
            step=0.05,
            key=f"{kp}_mintat",
        )
        if min_tat > 0:
            df = df[df["tat_seconds"].fillna(0) >= min_tat]

    # --- Sorting ---
    if sort_choice == "Slowest first":
        df = df.sort_values("tat_seconds", ascending=False, na_position="last")
    elif sort_choice == "Fastest first":
        df = df.sort_values("tat_seconds", ascending=True, na_position="last")
    else:  # Newest first
        df = df.sort_values("created_at", ascending=False, na_position="last")

    cols = [c for c in _DISPLAY_ORDER if c in df.columns]
    st.dataframe(
        df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "tat_seconds": st.column_config.NumberColumn(
                "TAT (s)", format="%.3f"
            ),
            "created_at": st.column_config.DatetimeColumn("Created"),
            "updated_at": st.column_config.DatetimeColumn("Updated"),
            "recommended_bid": st.column_config.NumberColumn(
                "Bid", format="%.2f"
            ),
            "recommended_bid_predicted_win_rate": st.column_config.NumberColumn(
                "Win rate", format="%.3f"
            ),
        },
    )
    st.caption(
        "Tip: click any column header to re-sort, or use the 'Sort by TAT' "
        "control above. The table also shows the full request payload per row."
    )


def main():
    """Standalone Predictions page -- renders the inference-log viewer."""
    render_inference_log(as_section=False, kp="pred")


if __name__ == "__main__":
    st.set_page_config(page_title="Predictions", layout="wide")
    main()
