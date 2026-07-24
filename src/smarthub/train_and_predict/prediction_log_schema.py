"""Prediction logging: one row per call to /recommend_bid or /explain_bid.

Single-table design (schema_version 2) per the 2026-07-22 DS weekly decision
(Vinaya + Nimesh) - see docs/PREDICTION_LOG_SCHEMA.md for the full design
writeup, worked example, and what changed from the original 3-table (v1)
design. In short: no per-candidate-bid table, no separate SHAP table - both
fold into JSON columns on a single row, and every call is logged whether it
succeeded or failed.

Uses plain SQLAlchemy Core (Table/Column, not the ORM) and stores JSON as
serialized text, matching this codebase's existing
`smarthub.core.config_store` conventions - portable across SQLite (tests)
and Postgres (production) with no dialect-specific types.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)

DEFAULT_PREDICTION_LOG_DB_URL = (
    "postgresql+psycopg2://prefect:prefect@postgres:5432/prefect"
)
SCHEMA_VERSION = (
    2  # bumped from the 3-table v1 design -- see docs/PREDICTION_LOG_SCHEMA.md
)
VALID_ENDPOINTS = ("recommend_bid", "explain_bid")
VALID_STATUSES = ("success", "error")


def prediction_log_db_url() -> str:
    """Return the prediction-log DB URL from the environment, or the default.

    Returns
    -------
    str
        ``$SMARTHUB_PREDICTION_LOG_DB_URL`` when set, else the shared
        Postgres default (mirrors ``smarthub.core.config_store``).
    """
    return os.getenv("SMARTHUB_PREDICTION_LOG_DB_URL", DEFAULT_PREDICTION_LOG_DB_URL)


_metadata = MetaData()

# JSON columns are stored as TEXT (json.dumps/json.loads at the boundary),
# not a dialect-specific JSON/JSONB type -- same portability choice
# config_store.py already makes, so this table creates identically on
# SQLite (tests) and Postgres (production).
prediction_log_table = Table(
    "smarthub_prediction_log",
    _metadata,
    Column("prediction_id", String(36), primary_key=True),
    Column("served_at", DateTime, nullable=False),
    Column("log_date", Date, nullable=False),
    Column("schema_version", SmallInteger, nullable=False, default=SCHEMA_VERSION),
    Column("request_id", String(128)),
    Column("lead_ping_id", Integer),
    Column("endpoint", String(32), nullable=False),
    Column("served_by_host", String(128)),
    Column("status", String(16), nullable=False, default="success"),
    Column("error_message", Text),
    Column("lead_type_id", SmallInteger, nullable=False),
    Column("lead_type_name", String(64), nullable=False),
    Column("campaign_id", Integer, nullable=False),
    Column("account_id", Integer),
    Column("source_type_id", Integer),
    Column("input_features", Text, nullable=False),
    Column("model_input_features", Text),
    Column("feature_cols", Text),
    Column("expected_revenue", Numeric(12, 2), nullable=False),
    Column("target_cm", Numeric(5, 4), nullable=False),
    Column("min_bid", Numeric(10, 2), nullable=False),
    Column("bid_step", Numeric(10, 2), nullable=False),
    Column("candidate_bid_generation", Text),
    Column("model_version", String(128)),
    Column("model_uri", Text),
    Column("model_type", String(64)),
    Column("model_calibrated", Boolean),
    Column("training_table_version", String(128)),
    # Text, not DateTime: copied verbatim from the manifest's `lineage` dict,
    # whose values pass through `json.dumps(..., default=str)` at save time
    # (registry.save_version) -- not guaranteed to be strict ISO 8601, so
    # storing as text avoids a parse failure turning a logging call into an
    # error. Still fully useful for audit; a downstream reader can reparse.
    Column("model_data_min_created_at", Text),
    Column("model_data_max_created_at", Text),
    Column("model_data_age_days", Integer),
    Column("decision_path", String(32)),
    Column("decision_reason", Text),
    Column("recommended_bid", Numeric(10, 2)),
    Column("recommended_bid_predicted_win_rate", Numeric(6, 5)),
    # Renamed from recommended_bid_expected_profit (2026-07-23) -- matches
    # recommended_bid_predicted_win_rate's "predicted_" naming, avoiding two
    # different words ("expected" vs. "predicted") for the same idea: a
    # model-predicted metric at the recommended bid, not a realized one.
    Column("recommended_bid_predicted_profit", Numeric(12, 4)),
    # recommended_bid_predicted_profit / expected_revenue -- added 2026-07-23.
    # Same nullability as predicted_profit (null whenever there's no
    # predicted profit to divide: cold start, no viable bid, or an error
    # before a bid was reached).
    Column("recommended_bid_predicted_cm", Numeric(6, 5)),
    Column("shap_explanation", Text),
    Column("serving_config", Text),
    Column("created_at", DateTime, nullable=False),
)


def _json_or_none(value: Any) -> str | None:
    """Serialize ``value`` to a JSON string, or ``None`` if ``value`` is ``None``."""
    return None if value is None else json.dumps(value, default=str)


def _decode_row(mapping) -> dict:
    """Decode one DB row mapping back into a plain dict with JSON columns parsed.

    Inputs
    ------
    mapping : sqlalchemy.engine.RowMapping
        A row's ``._mapping`` from a SQLAlchemy result.

    Returns
    -------
    dict
        The row as a plain dict, with ``input_features``,
        ``model_input_features``, ``feature_cols``,
        ``candidate_bid_generation``, ``shap_explanation``, and
        ``serving_config`` decoded from JSON text back to Python objects.
    """
    out = dict(mapping)
    for key in (
        "input_features",
        "model_input_features",
        "feature_cols",
        "candidate_bid_generation",
        "shap_explanation",
        "serving_config",
    ):
        if out.get(key) is not None:
            out[key] = json.loads(out[key])
    return out


class PredictionLogStore:
    """Write/read the single prediction-log table (creates it if absent)."""

    def __init__(self, url: str | None = None):
        """Open the prediction-log DB, creating the table if absent.

        Inputs
        ------
        url : str | None
            SQLAlchemy URL; defaults to ``prediction_log_db_url()`` when
            omitted.
        """
        self.engine = create_engine(url or prediction_log_db_url(), future=True)
        _metadata.create_all(self.engine)

    def log_prediction(
        self,
        *,
        endpoint: str,
        lead_type_id: int,
        lead_type_name: str,
        campaign_id: int,
        input_features: dict,
        expected_revenue: float,
        target_cm: float,
        min_bid: float,
        bid_step: float,
        status: str = "success",
        error_message: str | None = None,
        account_id: int | None = None,
        source_type_id: int | None = None,
        model_input_features: dict | None = None,
        feature_cols: list[str] | None = None,
        candidate_bid_generation: dict | None = None,
        model_version: str | None = None,
        model_uri: str | None = None,
        model_type: str | None = None,
        model_calibrated: bool | None = None,
        training_table_version: str | None = None,
        model_data_min_created_at: str | None = None,
        model_data_max_created_at: str | None = None,
        model_data_age_days: int | None = None,
        decision_path: str | None = None,
        decision_reason: str | None = None,
        recommended_bid: float | None = None,
        recommended_bid_predicted_win_rate: float | None = None,
        recommended_bid_predicted_profit: float | None = None,
        recommended_bid_predicted_cm: float | None = None,
        shap_explanation: dict | None = None,
        serving_config: dict | None = None,
        request_id: str | None = None,
        lead_ping_id: int | None = None,
        served_by_host: str | None = None,
        prediction_id: str | None = None,
        served_at: datetime | None = None,
    ) -> str:
        """Insert one prediction-log row (one row per API call, §2 of the doc).

        Inputs
        ------
        endpoint : str
            ``'recommend_bid'`` or ``'explain_bid'``.
        lead_type_id, lead_type_name : int, str
            Which lead-type model answered this prediction.
        campaign_id : int
            Business context from the request.
        input_features : dict
            The raw request payload as the caller sent it.
        expected_revenue, target_cm, min_bid, bid_step : float
            Optimizer inputs, straight from the request.
        status : str
            ``'success'`` or ``'error'`` - every call is logged either way.
        error_message : str | None
            Populated when ``status == 'error'``.
        account_id, source_type_id : int | None
            Optional business context from the request.
        model_input_features : dict | None
            The exact row the preprocessing step produced and the model
            scored. ``None`` when scoring never happened.
        feature_cols : list[str] | None
            Ordered feature columns the serving model version used.
        candidate_bid_generation : dict | None
            Summary of the candidate-bid sweep (method/bounds/count) -
            replaces v1's per-candidate table, see the module docstring.
        model_version, model_uri, model_type, model_calibrated,
        training_table_version, model_data_min_created_at,
        model_data_max_created_at, model_data_age_days :
            From the resolved model's registry manifest, or all ``None`` on
            true cold start.
        decision_path, decision_reason : str | None
            From ``predict.decide_bid``.
        recommended_bid, recommended_bid_predicted_win_rate,
        recommended_bid_predicted_profit : float | None
            From ``predict.decide_bid``. ``None`` recommended_bid means "no
            viable bid" (or an error before one was reached). Renamed from
            recommended_bid_expected_profit (2026-07-23) for consistency
            with recommended_bid_predicted_win_rate's naming.
        recommended_bid_predicted_cm : float | None
            ``recommended_bid_predicted_profit / expected_revenue`` -- added
            2026-07-23. ``None`` under the same conditions as
            recommended_bid_predicted_profit.
        shap_explanation : dict | None
            Everything ``/explain_bid`` adds beyond ``/recommend_bid``
            (``top_factors``, ``base_win_rate``, ``bid_curve``,
            ``explanation``) - replaces v1's SHAP child table.
        serving_config : dict | None
            Snapshot of the serving-policy config in effect for this
            prediction, for reproducibility.
        request_id : str | None
            Caller-supplied correlation id, if any.
        lead_ping_id : int | None
            Logical reference to ``public.lead_pings.id`` - see doc §8.
        served_by_host : str | None
            Pod/instance id; defaults to ``$HOSTNAME`` when omitted.
        prediction_id : str | None
            Explicit id to use (mainly for tests); generated (``uuid4``)
            when omitted.
        served_at : datetime | None
            Explicit timestamp to use (mainly for tests); defaults to now
            (UTC) when omitted.

        Returns
        -------
        str
            The ``prediction_id`` of the inserted row.

        Raises
        ------
        ValueError
            If ``endpoint`` or ``status`` isn't one of the valid values.
        """
        if endpoint not in VALID_ENDPOINTS:
            raise ValueError(
                f"endpoint must be one of {VALID_ENDPOINTS}, got {endpoint!r}"
            )
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")

        prediction_id = prediction_id or str(uuid.uuid4())
        served_at = served_at or datetime.now(timezone.utc)
        log_date = served_at.date() if isinstance(served_at, datetime) else served_at
        now = datetime.now(timezone.utc)

        with self.engine.begin() as conn:
            conn.execute(
                insert(prediction_log_table).values(
                    prediction_id=prediction_id,
                    served_at=served_at,
                    log_date=log_date,
                    schema_version=SCHEMA_VERSION,
                    request_id=request_id,
                    lead_ping_id=lead_ping_id,
                    endpoint=endpoint,
                    served_by_host=served_by_host or os.getenv("HOSTNAME"),
                    status=status,
                    error_message=error_message,
                    lead_type_id=lead_type_id,
                    lead_type_name=lead_type_name,
                    campaign_id=campaign_id,
                    account_id=account_id,
                    source_type_id=source_type_id,
                    input_features=_json_or_none(input_features),
                    model_input_features=_json_or_none(model_input_features),
                    feature_cols=_json_or_none(feature_cols),
                    expected_revenue=expected_revenue,
                    target_cm=target_cm,
                    min_bid=min_bid,
                    bid_step=bid_step,
                    candidate_bid_generation=_json_or_none(candidate_bid_generation),
                    model_version=model_version,
                    model_uri=model_uri,
                    model_type=model_type,
                    model_calibrated=model_calibrated,
                    training_table_version=training_table_version,
                    model_data_min_created_at=model_data_min_created_at,
                    model_data_max_created_at=model_data_max_created_at,
                    model_data_age_days=model_data_age_days,
                    decision_path=decision_path,
                    decision_reason=decision_reason,
                    recommended_bid=recommended_bid,
                    recommended_bid_predicted_win_rate=(
                        recommended_bid_predicted_win_rate
                    ),
                    recommended_bid_predicted_profit=recommended_bid_predicted_profit,
                    recommended_bid_predicted_cm=recommended_bid_predicted_cm,
                    shap_explanation=_json_or_none(shap_explanation),
                    serving_config=_json_or_none(serving_config),
                    created_at=now,
                )
            )
        return prediction_id

    def update_shap_explanation(
        self, prediction_id: str, shap_explanation: dict
    ) -> bool:
        """Attach/replace ``shap_explanation`` on an already-logged row.

        Used by ``/recommend_bid``'s background SHAP task (predict.py) --
        that route logs its row synchronously *before* computing SHAP (SHAP
        is deliberately kept off the response path, same principle as
        ``/explain_bid``'s LLM call), then fills this column in afterward
        once the background computation finishes. ``/explain_bid`` never
        needs this -- it passes ``shap_explanation`` directly to
        ``log_prediction`` since it computes SHAP before responding anyway.

        Inputs
        ------
        prediction_id : str
            The row to update (as returned by ``log_prediction``).
        shap_explanation : dict
            Same shape written by ``log_prediction``'s ``shap_explanation``
            kwarg -- here typically just ``{"top_factors", "base_win_rate"}``
            (no LLM ``explanation`` -- that stays /explain_bid-only, see
            docs/PREDICTION_LOG_SCHEMA.md).

        Returns
        -------
        bool
            Whether a row with this ``prediction_id`` was actually found and
            updated (``False`` if the id doesn't exist -- e.g. the original
            insert itself had failed).
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                update(prediction_log_table)
                .where(prediction_log_table.c.prediction_id == prediction_id)
                .values(shap_explanation=_json_or_none(shap_explanation))
            )
        return result.rowcount > 0

    def get(self, prediction_id: str) -> dict | None:
        """Return one logged prediction, JSON columns decoded, or ``None``.

        Inputs
        ------
        prediction_id : str
            The prediction to look up.

        Returns
        -------
        dict | None
            The row as a plain dict (JSON columns decoded), or ``None`` if
            no row exists for this id.
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                select(prediction_log_table).where(
                    prediction_log_table.c.prediction_id == prediction_id
                )
            ).first()
        return None if row is None else _decode_row(row._mapping)

    def recent(
        self,
        *,
        lead_type_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return the most recent logged predictions, optionally filtered.

        Inputs
        ------
        lead_type_id : int | None
            Restrict to this lead type when given.
        status : str | None
            Restrict to ``'success'`` or ``'error'`` when given.
        limit : int
            Maximum number of rows to return, most recent first.

        Returns
        -------
        list[dict]
            Matching rows, JSON columns decoded, newest first.
        """
        query = select(prediction_log_table).order_by(
            prediction_log_table.c.served_at.desc()
        )
        if lead_type_id is not None:
            query = query.where(prediction_log_table.c.lead_type_id == lead_type_id)
        if status is not None:
            query = query.where(prediction_log_table.c.status == status)
        query = query.limit(limit)
        with self.engine.begin() as conn:
            rows = conn.execute(query).all()
        return [_decode_row(r._mapping) for r in rows]
