"""Optional prediction-log pull + join for the monitoring dataset.

This is the ``--include-prediction-logs`` extension of the data pull. It reads
rows from the Postgres prediction log (``smarthub_prediction_log``) and joins
them to the raw leads/outcomes already pulled from Redshift, producing a
per-prediction **monitoring dataset** the Streamlit ``monitoring_app`` reads from
storage — so it never has to query the prediction-log DB directly.

It is a no-op unless the flag/param is set, so the default pull is unchanged.
The join key is ``smarthub_prediction_log.lead_ping_id`` == ``lead_pings.id``.
One row per ``prediction_id``; the persist step upserts on that key so a
later pull fills in outcomes that resolved after the prediction was made.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import select

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)

# Prediction-log columns retained in the monitoring dataset: the model-monitoring
# minimum (identifiers, timestamps, the recommended-bid predictions, model
# version) plus the business context the log already carries.
PREDICTION_COLUMNS = [
    "prediction_id",
    "created_at",
    "served_at",
    "lead_ping_id",
    "lead_type_id",
    "lead_type_name",
    "campaign_id",
    "account_id",
    "source_type_id",
    "expected_revenue",
    "recommended_bid",
    "recommended_bid_predicted_win_rate",
    "recommended_bid_predicted_profit",
    "recommended_bid_predicted_cm",
    "decision_path",
    "status",
    "model_name",
    "model_version",
    "model_type",
    "training_table_version",
    "tat_seconds",
]

# Raw lead/outcome columns attached on join; renamed with a ``lead_`` prefix so
# they never clash with the prediction-log fields above.
_OUTCOME_COLUMNS = ["id", "won", "rev", "exp_rev"]
_OUTCOME_RENAME = {
    "won": "lead_won",
    "rev": "lead_rev",
    "exp_rev": "lead_exp_rev",
}


def fetch_prediction_logs(
    since: datetime | str,
    until: datetime | str,
    lead_type_id: int | None = None,
    store=None,
) -> pd.DataFrame:
    """Read prediction-log rows created in ``[since, until)`` (newest last).

    Inputs
    ------
    since : datetime | str
        Inclusive lower bound on ``created_at`` (incremental watermark).
    until : datetime | str
        Exclusive upper bound on ``created_at``.
    lead_type_id : int | None
        Restrict to one lead type when given.
    store : PredictionLogStore | None
        Injectable store (tests); the default connects to the configured DB.

    Returns
    -------
    pandas.DataFrame
        Selected prediction-log columns, or an empty frame when none match.
    """
    if store is None:
        from smarthub.train_and_predict.prediction_log_schema import (
            PredictionLogStore,
        )

        store = PredictionLogStore()

    from smarthub.train_and_predict.prediction_log_schema import (
        prediction_log_table as tbl,
    )

    # Project ONLY the needed columns at the SQL level. Selecting the whole table
    # would pull the large JSON blobs (input_features, model_input_features,
    # candidate_evaluations, shap_explanation, ...) for every row into memory and
    # OOM on a wide window; the monitoring dataset needs none of them.
    selected = [tbl.c[c] for c in PREDICTION_COLUMNS if c in tbl.c]
    query = select(*selected).where(tbl.c.created_at >= since, tbl.c.created_at < until)
    if lead_type_id is not None:
        query = query.where(tbl.c.lead_type_id == lead_type_id)
    query = query.order_by(tbl.c.created_at.asc())

    with store.engine.begin() as conn:
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    if df.empty:
        logger.info("No prediction-log rows in [%s, %s).", since, until)
        return pd.DataFrame()

    logger.info("Fetched %s prediction-log rows.", f"{len(df):,}")
    return df


def build_monitoring_dataset(
    pred_df: pd.DataFrame,
    leads_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join predictions to their raw lead/outcome rows on the ping id.

    One row per prediction. Predictions without a resolvable ``lead_ping_id``
    are dropped (they can't be evaluated); predictions whose lead/outcome isn't
    available yet keep null ``lead_*`` columns and get filled on a later pull.

    Inputs
    ------
    pred_df : pandas.DataFrame
        Prediction-log rows from :func:`fetch_prediction_logs`.
    leads_df : pandas.DataFrame | None
        Raw leads/outcomes (needs at least ``id``; ``won``/``rev``/``exp_rev``
        attached when present).

    Returns
    -------
    pandas.DataFrame
        The monitoring dataset (prediction fields + ``lead_*`` outcome fields).
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    preds = pred_df.copy()
    preds["lead_ping_id"] = pd.to_numeric(preds["lead_ping_id"], errors="coerce")
    dropped = int(preds["lead_ping_id"].isna().sum())
    if dropped:
        logger.info("Dropping %s prediction(s) with no lead_ping_id.", f"{dropped:,}")
    preds = preds.dropna(subset=["lead_ping_id"]).copy()
    if preds.empty:
        return preds
    preds["lead_ping_id"] = preds["lead_ping_id"].astype("int64")

    if leads_df is not None and not leads_df.empty and "id" in leads_df.columns:
        cols = [c for c in _OUTCOME_COLUMNS if c in leads_df.columns]
        outcome = (
            leads_df[cols]
            .dropna(subset=["id"])
            .drop_duplicates(subset=["id"], keep="last")
            .rename(columns=_OUTCOME_RENAME)
        )
        outcome["id"] = pd.to_numeric(outcome["id"], errors="coerce").astype("int64")
        merged = preds.merge(outcome, left_on="lead_ping_id", right_on="id", how="left")
        merged = merged.drop(columns=["id"], errors="ignore")
    else:
        merged = preds
        for target in _OUTCOME_RENAME.values():
            merged[target] = pd.NA

    matched = int(merged["lead_won"].notna().sum()) if "lead_won" in merged else 0
    logger.info(
        "Monitoring dataset: %s prediction rows, %s matched to a lead outcome.",
        f"{len(merged):,}",
        f"{matched:,}",
    )
    return merged


def pull_and_persist_prediction_logs(
    since: datetime | str,
    until: datetime | str,
    lead_type_id: int | None = None,
    *,
    store=None,
    settings=None,
) -> dict:
    """Fetch prediction logs, join to raw outcomes, and persist the dataset.

    Shared by the CLI (``pull.run``) and the Prefect flow. The join reads only
    the leads referenced by the pulled predictions (a bounded, memory-flat
    DuckDB lookup — not a wide time-window scan), so outcomes that resolved in
    any earlier pull still get joined. Persists via ``storage.save_monitoring``
    (upsert on ``prediction_id``).

    Returns a small result dict (rows fetched / persisted); a no-op-safe empty
    result when there are no new predictions.
    """
    import pandas as pd

    from smarthub.core import storage
    from smarthub.core.config import StorageSettings

    settings = settings or StorageSettings.from_env()

    pred_df = fetch_prediction_logs(since, until, lead_type_id, store=store)
    if pred_df.empty:
        return {"prediction_rows": 0, "monitoring_rows": 0}

    # Only look up the leads the predictions actually reference.
    lead_ids = pd.to_numeric(pred_df["lead_ping_id"], errors="coerce").dropna().unique()
    try:
        leads_df = storage.read_leads_outcomes(lead_ids, settings=settings)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - persist predictions unjoined on read error
        logger.warning("Could not read lead outcomes (%s); persisting unjoined.", exc)
        leads_df = None

    monitoring = build_monitoring_dataset(pred_df, leads_df)
    result = storage.save_monitoring(monitoring, settings)
    result["prediction_rows"] = int(len(pred_df))
    logger.info("Persisted monitoring dataset: %s", result)
    return result
