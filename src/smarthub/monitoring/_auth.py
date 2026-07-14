"""Lightweight password gate for admin-only pages (e.g. Config).

MVP-grade: a single shared password from ``CONFIG_ADMIN_PASSWORD`` (env), checked
against a Streamlit password field and remembered in the session. For stronger
auth later, use ``streamlit-authenticator`` or put the app behind a reverse proxy.
"""

from __future__ import annotations

import os

import streamlit as st


def require_password(
    env_var: str = "CONFIG_ADMIN_PASSWORD",
    session_key: str = "config_authed",
) -> bool:
    """Return True only once the correct admin password has been entered.

    If no password is configured, editing stays locked (fail closed).

    Inputs
    ------
    env_var : str
        Environment variable holding the expected password.
    session_key : str
        Session-state key used to remember a successful login.

    Returns
    -------
    bool
        Whether the user is authenticated for the current session.
    """
    expected = os.getenv(env_var)
    if not expected:
        st.warning(
            "Config is locked — set the CONFIG_ADMIN_PASSWORD environment "
            "variable to enable editing."
        )
        return False

    if st.session_state.get(session_key):
        return True

    st.subheader("Admin login")
    password = st.text_input("Admin password", type="password")
    if st.button("Unlock"):
        if password == expected:
            st.session_state[session_key] = True
            st.rerun()
        st.error("Incorrect password.")
    return False
