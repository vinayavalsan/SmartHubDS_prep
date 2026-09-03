"""Single multipage Streamlit app: Leads, Monitoring, and (gated) Config.

One service / one port. Leads + Monitoring are open (read-only); the Config page
is protected by an admin password (see ``_auth``).

Run with:
    streamlit run src/smarthub/monitoring/app.py
"""

from __future__ import annotations

import streamlit as st

from smarthub.monitoring import (
    config_app,
    health_app,
    leads_app,
    performance_app,
    predictions_app,
)
from smarthub.monitoring._auth import require_password

st.set_page_config(page_title="SmartHub DS", layout="wide")


def config_page():
    """Config page, gated behind the admin password."""
    if require_password():
        config_app.main()


def main():
    """Register the pages and run the Streamlit multipage navigation."""
    pages = [
        st.Page(health_app.main, title="Health", url_path="health"),
        st.Page(leads_app.main, title="Leads", url_path="leads", default=True),
        st.Page(performance_app.main, title="Performance", url_path="performance"),
        st.Page(predictions_app.main, title="Predictions", url_path="predictions"),
        st.Page(config_page, title="Config", url_path="config"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
