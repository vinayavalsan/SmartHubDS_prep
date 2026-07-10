"""Online model loading and bid prediction for SmartHub.

This module resolves serving models and exposes bid recommendation endpoints.
"""

from __future__ import annotations

import os

import pandas as pd

from . import config, optimizer, preprocessing, registry


def resolve_model_uri(lead_type_id: int = 6) -> str:
    """Resolve the model artifact used for serving.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.

    Returns
    -------
    str
        Resolved local model path or MLflow URI.

    Raises
    ------
    FileNotFoundError
        If no serving model can be resolved.
    """
    env_override = os.getenv("MODEL_URI")
    if env_override:
        return env_override

    lead_type_name = config.lead_type_name(lead_type_id)
    pinned_version = config.active_model_version()
    if pinned_version:
        return str(registry.version_path(lead_type_name, pinned_version))

    path = registry.currently_serving_model_path(lead_type_name)
    if path is None:
        raise FileNotFoundError(
            f"Nothing is currently serving lead type '{lead_type_name}'. "
            "Train and promote a model first."
        )
    return str(path)


def load_model(model_uri: str | None = None, lead_type_id: int = 6):
    """Load a local or MLflow model artifact.

    Inputs
    ------
    model_uri : str | None
        Optional local path or MLflow model URI.
    lead_type_id : int
        SmartHub lead type identifier.

    Returns
    -------
    Any
        Loaded prediction model.
    """
    uri = model_uri or resolve_model_uri(lead_type_id)
    if str(uri).endswith(".pkl"):
        import joblib

        return joblib.load(uri)

    import mlflow.sklearn

    return mlflow.sklearn.load_model(uri)


# Backward-compatible public import. The implementation belongs to optimizer.py.
optimize_bid_for_row = optimizer.optimize_bid_for_row


try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - API dependencies are optional
    _FASTAPI_AVAILABLE = False


if _FASTAPI_AVAILABLE:

    class BidRequest(BaseModel):
        """Validate optimizer controls and lead features for one API request."""

        expected_revenue: float = Field(..., gt=0)
        target_cm: float = Field(0.25, ge=0, lt=1)
        min_bid: float = Field(0.25, ge=0)
        bid_step: float = Field(0.25, gt=0)

        campaign_id: int
        account_id: int | None = None
        source_type_id: int | None = None
        lead_type_id: int = 6
        created_hour: int = Field(..., ge=0, le=23)
        created_dayofweek: int = Field(..., ge=0, le=6)

        state: str | None = None
        insured: str | None = None
        home_owner: str | None = None
        dui: str | None = None
        sr22_required: str | None = None
        military_affiliation: str | None = None
        gender: str | None = None
        marital_status: str | None = None

        num_vehicles: float | None = None
        num_drivers: float | None = None
        num_auto_violations: float | None = None
        num_auto_accidents: float | None = None
        continuous_coverage_months: float | None = None
        age: float | None = None

    app = FastAPI(title="Anton Bid Prediction API")

    @app.get("/health")
    def health(lead_type_id: int = 6):
        """Return service health and the resolved model artifact.

        Inputs
        ------
        lead_type_id : int
            SmartHub lead type identifier.

        Returns
        -------
        dict
            Service health payload.
        """
        try:
            model_uri = resolve_model_uri(lead_type_id)
        except FileNotFoundError:
            model_uri = None
        return {
            "status": "ok",
            "lead_type_id": lead_type_id,
            "model_uri": model_uri,
        }

    @app.post("/recommend_bid")
    def recommend_bid(request: BidRequest):
        """Return the expected-profit-maximizing bid for one request.

        Inputs
        ------
        request : BidRequest
            Validated bid recommendation request.

        Returns
        -------
        dict
            Recommended bid and supporting metrics.
        """
        model = load_model(lead_type_id=request.lead_type_id)
        record = request.model_dump(
            exclude={
                "expected_revenue",
                "target_cm",
                "min_bid",
                "bid_step",
            }
        )
        record["bid"] = request.min_bid
        frame = preprocessing.serving_frame(
            pd.DataFrame([record]),
            request.lead_type_id,
        )
        return optimizer.optimize_bid_for_row(
            row=frame.iloc[0],
            model=model,
            expected_revenue=request.expected_revenue,
            target_cm=request.target_cm,
            min_bid=request.min_bid,
            bid_step=request.bid_step,
        )
