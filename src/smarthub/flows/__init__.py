"""Prefect flows and tasks for SmartHub.

Kept import-light: this package's ``__init__`` does not import Prefect, so the
pure helpers (e.g. ``windowing``) can be imported and tested without Prefect
installed. The Prefect flow lives in ``data_pull_flow``.
"""
