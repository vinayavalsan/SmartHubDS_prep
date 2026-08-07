"""FastAPI application entry point for the SmartHub serving API.

The prediction and explanation routes are defined in ``server.predict`` (which
builds ``app`` and registers ``/recommend_bid`` and ``/explain_bid``). This
module is the stable import path deployments target
(``uvicorn smarthub.server.app:app``), decoupling the run command from the
internal module layout.
"""

from smarthub.server.predict import app

__all__ = ["app"]
