"""Streamlit page: live Health / SLO dashboard for the bid API.

At-a-glance service health computed from the prediction log (no new infra):
request rate, TAT p50/p95/p99 vs the 1s SLA, error rate, SHAP backlog, and
freshness -- with red/green thresholds and a banner listing any active SLO
breaches (the same breaches the scheduled Slack alerter fires on).

Run standalone with:
    streamlit run src/smarthub/monitoring/health_app.py
"""

from __future__ import annotations

import streamlit as st

from smarthub.monitoring import slo


def _store():
    from smarthub.train_and_predict.prediction_log_schema import PredictionLogStore

    return PredictionLogStore()


@st.cache_data(ttl=10)
def _slis(window_minutes: int) -> dict:
    return slo.compute_slis(_store(), window_minutes=window_minutes)


def _fmt_s(v):
    return "—" if v is None else f"{v*1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def main():
    st.title("Health / SLO — bid API")
    st.caption(
        "Live service-level indicators from the prediction log. Thresholds match "
        "the Slack alerter, so what's red here is what pages you."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        window = st.selectbox(
            "Window",
            options=[5, 15, 60, 180],
            index=1,
            format_func=lambda m: f"last {m} min",
        )
    with c2:
        if st.button("🔄 Refresh"):
            _slis.clear()

    try:
        s = _slis(int(window))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the prediction log: {exc}")
        return

    thr = slo.thresholds()
    breaches = slo.evaluate_alerts(s, thr)

    # --- Alert banner ---
    if breaches:
        st.error(
            "🔴 SLO breach(es):\n\n" + "\n".join(f"- {b['message']}" for b in breaches)
        )
    elif s["requests"] == 0:
        st.info("No requests in this window yet.")
    else:
        st.success("🟢 All SLOs healthy.")

    # --- Top-line metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Requests", s["requests"], f"{s['rate_per_min'] or 0}/min")
    p99 = s["tat_p99"]
    m2.metric(
        "TAT p99",
        _fmt_s(p99),
        delta=("over SLA" if p99 and p99 > thr["tat_p99_seconds"] else "ok"),
        delta_color=("inverse" if p99 and p99 > thr["tat_p99_seconds"] else "normal"),
    )
    m3.metric(
        "Error rate",
        f"{s['error_rate_pct']:.2f}%",
        delta=("high" if s["error_rate_pct"] > thr["error_rate_pct"] else "ok"),
        delta_color=(
            "inverse" if s["error_rate_pct"] > thr["error_rate_pct"] else "normal"
        ),
    )
    m4.metric(
        "Within 1s TAT",
        "—" if s["within_1s_pct"] is None else f"{s['within_1s_pct']:.1f}%",
    )

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("TAT p50", _fmt_s(s["tat_p50"]))
    m6.metric("TAT p95", _fmt_s(s["tat_p95"]))
    m7.metric(
        "SHAP backlog",
        s["shap_backlog"],
        delta=("high" if s["shap_backlog"] > thr["shap_backlog"] else "ok"),
        delta_color=(
            "inverse" if s["shap_backlog"] > thr["shap_backlog"] else "normal"
        ),
    )
    age = s["last_ok_age_seconds"]
    m8.metric(
        "Last request",
        (
            "—"
            if age is None
            else (f"{age:.0f}s ago" if age < 120 else f"{age/60:.1f}m ago")
        ),
    )

    # --- Decision-path mix ---
    if s["decision_paths"]:
        st.subheader("Decision paths (window)")
        st.bar_chart(s["decision_paths"])

    with st.expander("Thresholds (config/smarthub.yaml → slo)"):
        st.json(thr)
    st.caption(
        "Tune thresholds in config/smarthub.yaml under `slo:` "
        "(tat_p99_seconds, error_rate_pct, shap_backlog, no_requests_minutes)."
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Health / SLO", layout="wide")
    main()
