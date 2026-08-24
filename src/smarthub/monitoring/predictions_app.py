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
    "recommended_bid_predicted_profit",
    "expected_revenue",
    "campaign_id",
    "shap_status",
    "shap_top_factors",
    "payload",
    "prediction_id",
    "lead_ping_id",
]

# How many top SHAP factors to show inline in the table summary column.
_SHAP_SUMMARY_TOP_N = 3


def _shap_summary(explanation) -> tuple[str, str]:
    """Return ``(status, top_factors_text)`` for a stored SHAP explanation.

    ``status`` is one of ``"ready"`` / ``"pending"`` / ``"n/a"``;
    ``top_factors_text`` is a compact ``"feature ↑0.84, feature ↓0.31"`` string
    (arrows show whether the factor pushed the win-rate up or down).
    """
    if not isinstance(explanation, dict):
        return "pending", ""
    factors = explanation.get("top_factors") or explanation.get(
        "feature_contributions"
    )
    if not factors:
        # SHAP ran but produced nothing usable (e.g. unsupported model type).
        return "n/a", ""
    parts = []
    for f in factors[:_SHAP_SUMMARY_TOP_N]:
        name = f.get("feature", "?")
        val = f.get("shap", f.get("contribution", 0)) or 0
        arrow = "↑" if val >= 0 else "↓"
        parts.append(f"{name} {arrow}{abs(val):.2f}")
    return "ready", ", ".join(parts)


# Time-window options -> minutes back from now. Capped at 7 days: the log grows
# unbounded, so we never let the dashboard scan more than a week in one read.
_WINDOW_OPTIONS: dict[str, int] = {
    "Previous hour": 60,
    "Previous 6 hours": 6 * 60,
    "Previous 12 hours": 12 * 60,
    "Previous 24 hours": 24 * 60,
    "Last 2 days": 2 * 24 * 60,
    "Last 3 days": 3 * 24 * 60,
    "Last 4 days": 4 * 24 * 60,
    "Last 5 days": 5 * 24 * 60,
    "Last 6 days": 6 * 24 * 60,
    "Last 7 days": 7 * 24 * 60,
}


def _store():
    """Construct the prediction-log store (lazily; import kept local so the
    page loads even where the ML/serving deps aren't installed)."""
    from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore

    return PredictionLogStore()


@st.cache_data(ttl=15)
def load_predictions(
    limit: int,
    lead_type_id,
    status,
    minutes: int,
    campaign_id,
    lead_ping_id,
) -> pd.DataFrame:
    """Load prediction-log rows into a DataFrame (cached ~15s).

    Filters by time window (``minutes``, capped at 7 days server-side), lead
    type, status, and optionally exact ``campaign_id`` / ``lead_ping_id``.
    """
    rows = _store().filtered(
        limit=limit,
        minutes=minutes,
        lead_type_id=lead_type_id if lead_type_id != "all" else None,
        status=status if status != "all" else None,
        campaign_id=campaign_id,
        lead_ping_id=lead_ping_id,
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Compact the request payload dict into a readable JSON string column.
    if "input_features" in df.columns:
        df["payload"] = df["input_features"].apply(
            lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
        )
    # Derive inline SHAP columns from the stored explanation dict.
    if "shap_explanation" in df.columns:
        summary = df["shap_explanation"].apply(_shap_summary)
        df["shap_status"] = summary.apply(lambda t: t[0])
        df["shap_top_factors"] = summary.apply(lambda t: t[1])
    else:
        df["shap_status"] = "pending"
        df["shap_top_factors"] = ""
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
        "and TAT (time from request received to result returned). Limited to the "
        "last 7 days."
    )

    # Row 1: time window + lead type + status + sort.
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        window_label = st.selectbox(
            "Time window",
            options=list(_WINDOW_OPTIONS.keys()),
            index=3,  # default: Previous 24 hours
            key=f"{kp}_win",
        )
    with c2:
        lead_type_id = st.selectbox(
            "Lead type", options=["all", 6, 1], format_func=str, key=f"{kp}_lt"
        )
    with c3:
        status = st.selectbox(
            "Status", options=["all", "success", "error"], key=f"{kp}_st"
        )
    with c4:
        sort_choice = st.selectbox(
            "Sort by TAT",
            options=["Slowest first", "Fastest first", "Newest first"],
            key=f"{kp}_sort",
        )

    # Row 2: exact-match filters (campaign / lead ping) + max rows.
    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        campaign_raw = st.text_input(
            "Campaign ID", value="", placeholder="e.g. 40088", key=f"{kp}_camp"
        )
    with f2:
        lead_ping_raw = st.text_input(
            "Lead ping ID", value="", placeholder="e.g. 998877", key=f"{kp}_lp"
        )
    with f3:
        limit = st.number_input(
            "Max rows",
            min_value=10,
            max_value=10000,
            value=500,
            step=50,
            key=f"{kp}_lim",
        )

    def _parse_id(raw: str, label: str):
        """Parse an optional integer id filter; warn (and ignore) if invalid."""
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            st.warning(f"{label} must be a whole number — ignoring that filter.")
            return None

    campaign_id = _parse_id(campaign_raw, "Campaign ID")
    lead_ping_id = _parse_id(lead_ping_raw, "Lead ping ID")
    minutes = _WINDOW_OPTIONS[window_label]

    if st.button("🔄 Reload", key=f"{kp}_reload"):
        load_predictions.clear()

    try:
        df = load_predictions(
            int(limit), lead_type_id, status, minutes, campaign_id, lead_ping_id
        )
    except Exception as exc:  # noqa: BLE001 -- surface DB issues in the UI
        st.error(f"Could not read the prediction log: {exc}")
        st.info(
            "Is SMARTHUB_PREDICTION_LOG_DB_URL pointing at the same DB the API "
            "writes to? (Defaults to the shared Postgres in the Docker stack.)"
        )
        return

    if df.empty:
        st.info(
            f"No predictions match these filters in the {window_label.lower()}."
        )
        return

    st.caption(f"Showing {len(df):,} row(s) from the {window_label.lower()}.")

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
            "tat_seconds": st.column_config.NumberColumn("TAT (s)", format="%.3f"),
            "created_at": st.column_config.DatetimeColumn("Created"),
            "updated_at": st.column_config.DatetimeColumn("Updated"),
            "recommended_bid": st.column_config.NumberColumn("Bid", format="%.2f"),
            "recommended_bid_predicted_win_rate": st.column_config.NumberColumn(
                "Win rate", format="%.3f"
            ),
            "recommended_bid_predicted_profit": st.column_config.NumberColumn(
                "Profit", format="%.2f"
            ),
            "campaign_id": st.column_config.NumberColumn("Campaign", format="%d"),
            "shap_status": st.column_config.TextColumn("SHAP"),
            "shap_top_factors": st.column_config.TextColumn("Top SHAP factors"),
        },
    )
    st.caption(
        "Tip: click any column header to re-sort, or use the 'Sort by TAT' "
        "control above. The table also shows the full request payload per row. "
        "SHAP is backfilled asynchronously, so recent rows may read 'pending'."
    )

    _render_shap_detail(df, kp=kp)


def _render_shap_detail(df: pd.DataFrame, kp: str) -> None:
    """Expandable panel: full SHAP breakdown for one selected prediction.

    Lets you pick any ``prediction_id`` in the current result set and see every
    feature's SHAP contribution (direction + magnitude), the model's base vs.
    served win rate, and the LLM narrative if one was stored.
    """
    if "shap_explanation" not in df.columns:
        return
    ready = df[df["shap_status"] == "ready"]
    with st.expander("🔎 SHAP detail for one prediction", expanded=False):
        if ready.empty:
            st.info(
                "No SHAP-ready rows in the current view yet (explanations are "
                "backfilled asynchronously). Try widening the time window."
            )
            return
        pid = st.selectbox(
            "Prediction ID",
            options=ready["prediction_id"].tolist(),
            key=f"{kp}_shap_pid",
        )
        exp = ready.loc[ready["prediction_id"] == pid, "shap_explanation"].iloc[0]
        if not isinstance(exp, dict):
            st.warning("No structured SHAP payload on this row.")
            return

        d1, d2 = st.columns(2)
        base = exp.get("base_prediction", exp.get("base_win_rate"))
        d1.metric(
            "Base win rate", f"{base:.3f}" if isinstance(base, (int, float)) else "—"
        )
        served = exp.get("prediction")
        d2.metric(
            "Served win rate",
            f"{served:.3f}" if isinstance(served, (int, float)) else "—",
        )

        factors = exp.get("feature_contributions") or exp.get("top_factors") or []
        if factors:
            fdf = pd.DataFrame(factors)
            # Normalise the SHAP-value column name across the two payload shapes.
            if "contribution" in fdf.columns:
                fdf = fdf.rename(columns={"contribution": "shap"})
            if "shap" in fdf.columns:
                fdf["direction"] = fdf["shap"].apply(
                    lambda v: "increased" if (v or 0) >= 0 else "decreased"
                )
                fdf = fdf.reindex(
                    fdf["shap"].abs().sort_values(ascending=False).index
                )
            show = [c for c in ("feature", "value", "shap", "direction") if c in fdf]
            st.dataframe(
                fdf[show],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "shap": st.column_config.NumberColumn(
                        "SHAP (log-odds)", format="%.4f"
                    ),
                },
            )
            st.caption(
                "Contributions are in log-odds units, sorted by magnitude. They "
                "sum (with the base) to the model's uncalibrated output, so they "
                "won't exactly reconstruct the calibrated served win rate."
            )
        else:
            st.info("SHAP ran but produced no feature contributions for this row.")

        narrative = exp.get("explanation")
        if narrative:
            st.markdown(f"**Narrative:** {narrative}")


def main():
    """Standalone Predictions page -- renders the inference-log viewer."""
    render_inference_log(as_section=False, kp="pred")


if __name__ == "__main__":
    st.set_page_config(page_title="Predictions", layout="wide")
    main()
