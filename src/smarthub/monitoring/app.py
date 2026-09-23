"""Single multipage Streamlit app: Leads, Monitoring, and (gated) Config.

One service / one port. Leads + Monitoring are open (read-only); the Config page
is protected by an admin password (see ``_auth``).

Run with:
    streamlit run src/smarthub/monitoring/app.py
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

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


def _render_app_version():
    """Show the version of the running (built/deployed) Docker image.

    Prefers ``SMARTHUB_IMAGE_TAG`` -- the image tag this container was deployed
    as (e.g. ``v0.1.3``), injected by docker-compose from ``IMAGE_TAG`` -- so the
    sidebar reflects the built image rather than possibly-stale installed
    package metadata. Falls back to the packaged version, then ``unknown``.
    """
    app_version = os.environ.get("SMARTHUB_IMAGE_TAG", "").strip()
    if not app_version:
        try:
            app_version = version("smarthub")
        except PackageNotFoundError:
            app_version = "unknown"

    st.sidebar.divider()
    st.sidebar.caption(f"Repo version: {app_version}")


def config_page():
    """Config page, gated behind the admin password."""
    if require_password():
        config_app.main()


def main():
    """Register the pages and run the Streamlit multipage navigation."""
    pages = [
        st.Page(health_app.main, title="Health", url_path="health"),
        st.Page(leads_app.main, title="Leads", url_path="leads"),
        st.Page(performance_app.main, title="Performance", url_path="performance", default=True),
        st.Page(predictions_app.main, title="Predictions", url_path="predictions"),
        st.Page(config_page, title="Config", url_path="config"),
    ]
    st.navigation(pages).run()
    _render_app_version()


if __name__ == "__main__":
    main()
