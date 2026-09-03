"""Post-training model diagnostics for SmartHub.

A standalone Streamlit app (``app.py``) for reviewing model-evaluation artifacts
after a production training run, kept deliberately separate from the regular
SmartHub monitoring dashboards (``smarthub.monitoring``). Diagnostic computations
live in ``diagnostics.py``; evaluation artifacts are loaded per MLflow ``run_id``.
"""
