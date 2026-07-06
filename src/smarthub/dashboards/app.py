"""Single multipage Streamlit app: Leads, Monitoring, and (gated) Config.

One service / one port. Leads + Monitoring are open (read-only); the Config page
is protected by an admin password (see ``_auth``).

Run with:
    streamlit run src/smarthub/dashboards/app.py
"""

from __future__ import annotations

import streamlit as st

from smarthub.dashboards import config_app, leads_app, monitoring_app
from smarthub.dashboards._auth import require_password

st.set_page_config(page_title="SmartHub DS", layout="wide")


def config_page():
    """Config page, gated behind the admin password."""
    if require_password():
        config_app.main()


def main():
    pages = [
        st.Page(leads_app.main, title="Leads", url_path="leads", default=True),
        st.Page(monitoring_app.main, title="Monitoring", url_path="monitoring"),
        st.Page(config_page, title="Config", url_path="config"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
