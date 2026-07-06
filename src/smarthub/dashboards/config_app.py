"""Streamlit page to view/edit Anton's runtime (Tier-2) config.

Renders the typed ``REGISTRY`` as a form, validates on save, writes to the
shared Postgres via ``ConfigStore``, and records who changed what. Secrets /
connection settings are intentionally NOT here — they live in the environment.

Run with:
    streamlit run src/smarthub/dashboards/config_app.py
"""

from __future__ import annotations

import streamlit as st

from smarthub.core.config_store import ConfigError, ConfigStore, ENVIRONMENTS

# set_page_config is called by the entry (app.py or the __main__ guard below).


@st.cache_resource
def get_store() -> ConfigStore:
    return ConfigStore()


def _allowed_values(item: dict) -> str:
    """Human-readable description of what values a param accepts."""
    if item["choices"]:
        return "one of: " + ", ".join(str(c) for c in item["choices"])
    if item["type"] == "bool":
        return "true / false"
    if item["type"] in ("int", "float"):
        lo, hi = item["minimum"], item["maximum"]
        if lo is not None and hi is not None:
            return f"{lo} to {hi}"
        if lo is not None:
            return f"≥ {lo}"
        if hi is not None:
            return f"≤ {hi}"
        return f"any {item['type']}"
    return "free text"


def _input_for(item: dict):
    """Render the right widget for a param (label hidden) and return the value."""
    key = item["key"]
    if item["choices"]:
        options = list(item["choices"])
        idx = options.index(item["value"]) if item["value"] in options else 0
        return st.selectbox(
            key, options=options, index=idx, label_visibility="collapsed"
        )
    if item["type"] == "bool":
        return st.checkbox(key, value=bool(item["value"]))
    if item["type"] in ("int", "float"):
        step = 1 if item["type"] == "int" else 0.05
        return st.number_input(
            key,
            value=item["value"],
            min_value=item["minimum"],
            max_value=item["maximum"],
            step=step,
            label_visibility="collapsed",
        )
    return st.text_input(
        key, value=str(item["value"]), label_visibility="collapsed"
    )


def main():
    st.title("Anton Config")
    st.caption(
        "Runtime tuning knobs (stored in Postgres, versioned). "
        "Secrets & DB connection stay in the environment, not here."
    )

    try:
        store = get_store()
    except Exception as exc:  # noqa: BLE001 - surface connection errors in the UI
        st.error(f"Could not connect to the config database: {exc}")
        st.stop()

    col1, col2 = st.columns(2)
    env = col1.selectbox("Environment", options=list(ENVIRONMENTS), index=1)
    updated_by = col2.text_input("Your name (for audit)", value="")

    items = store.resolved(env=env)

    entered: dict = {}
    with st.form("config_form"):
        for item in items:
            # name + definition
            st.markdown(f"**{item['key']}** — {item['description']}")
            # the input widget
            entered[item["key"]] = _input_for(item)
            # allowed values · default · who last changed it
            parts = [
                f"Allowed: {_allowed_values(item)}",
                f"Default: {item['default']}",
            ]
            parts.append(
                f"Last changed by {item['updated_by']} @ {item['updated_at']}"
                if item["overridden"]
                else "Currently using default"
            )
            st.caption("  ·  ".join(parts))
            st.divider()
        submitted = st.form_submit_button("Save changes")

    if submitted:
        if not updated_by.strip():
            st.warning("Please enter your name before saving.")
            st.stop()
        changed, errors = [], []
        current = {i["key"]: i["value"] for i in items}
        for key, value in entered.items():
            if value == current[key]:
                continue
            try:
                store.set(key, value, env=env, updated_by=updated_by.strip())
                changed.append(key)
            except ConfigError as exc:
                errors.append(str(exc))
        for err in errors:
            st.error(err)
        if changed:
            st.success(f"Saved ({env}): {', '.join(changed)}")
        elif not errors:
            st.info("No changes to save.")


if __name__ == "__main__":
    st.set_page_config(page_title="Anton Config", layout="centered")
    main()
