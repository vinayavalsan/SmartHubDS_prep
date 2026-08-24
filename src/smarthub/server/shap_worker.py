"""Offloaded SHAP enrichment worker.

Companion to ``shap_enrichment_mode == "offload"`` (see
``train_and_predict/config.shap_enrichment_mode``). When the serving process
runs in ``offload`` mode it logs each ``/recommend_bid`` prediction with
``shap_explanation = NULL`` and does **no** SHAP work itself -- keeping the
serving CPU free for the 1 s bid TAT. This process drains those pending rows
and backfills the SHAP factor breakdown, exactly the payload the in-process
``predict._log_shap_background`` task would have written.

Design
------
- **DB-as-queue.** The prediction-log table is the queue; a row is "pending"
  when ``shap_explanation IS NULL`` and it was actually served by a model
  (``decision_path IN ('model','exploration')``). No extra broker/infra.
- **Safe concurrency.** On Postgres, rows are claimed with ``FOR UPDATE SKIP
  LOCKED`` so multiple ``shap-worker`` replicas never process the same row.
  On SQLite (local/dev/tests) the lock clause is omitted -- a single writer is
  assumed there.
- **Durable + idempotent.** A row stays ``NULL`` until its explanation is
  written, so a crash mid-batch simply leaves it for the next pass. A hard,
  non-transient failure (e.g. a non-LightGBM model, which SHAP can't explain)
  writes a small ``{"shap_error": ...}`` sentinel instead of ``NULL`` so it is
  not retried forever.
- **Short locks.** One row per transaction, so a lock is held only for that
  row's SHAP compute, not a whole batch.

Run
---
    python -m smarthub.server.shap_worker
    # env knobs:
    #   SMARTHUB_PREDICTION_LOG_DB_URL   which DB to drain (same as serving)
    #   SMARTHUB_SHAP_WORKER_POLL_SECS   idle poll interval (default 1.0)
    #   SMARTHUB_SHAP_WORKER_BATCH       max rows per wake before re-polling
    #   SMARTHUB_SHAP_WORKER_ONCE=1      drain what's pending, then exit
"""

from __future__ import annotations

import logging
import os
import signal
import time

from sqlalchemy import select, update

from smarthub.core.lead_types import lead_type_name as get_lead_type_name
from smarthub.train_and_predict import registry
from smarthub.train_and_predict.prediction_log_schema import (
    PredictionLogStore,
    _decode_row,
    _json_or_none,
    prediction_log_table,
)

logger = logging.getLogger("smarthub.server.shap_worker")

_PENDING_DECISION_PATHS = ("model", "exploration")
_SERVED_OK_STATUSES = ("success", "ok")

# Per-version model cache so we don't re-unpickle for every row. Keyed by
# (lead_type_name, model_version); the artifact for a given version never
# changes once written.
_MODEL_CACHE: dict[tuple[str, str | None], object] = {}


def _load_model_for_row(row: dict):
    """Load the exact model version that served ``row`` (fallback: current).

    Explaining with the same version that produced the bid keeps the logged
    factors faithful even if a newer model has since been promoted.
    """
    lead_type_name = row.get("lead_type_name") or get_lead_type_name(
        int(row["lead_type_id"])
    )
    version = row.get("model_version")
    cache_key = (lead_type_name, version)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import joblib

    path = None
    if version:
        path = registry.serving_model_path(lead_type_name, version)
    if path is None:
        path = registry.currently_serving_model_path(lead_type_name)
    if path is None:
        raise FileNotFoundError(
            f"No model artifact for lead_type={lead_type_name!r} "
            f"version={version!r}"
        )

    model = joblib.load(str(path))
    _MODEL_CACHE[cache_key] = model
    return model


def _compute_shap_payload(row: dict) -> dict:
    """Compute the SHAP factor payload for one logged prediction row.

    Mirrors ``predict._log_shap_background`` exactly, so ``offload`` writes an
    identical ``shap_explanation`` to what ``inprocess`` would have written.
    """
    # Imported here (not at module top) so importing this module is cheap and
    # doesn't require the `explain` extra unless the worker actually runs.
    from smarthub.server import explain as server_explain

    prediction = server_explain.PredictionOutput(
        lead_type_id=int(row["lead_type_id"]),
        model_input_features=row.get("model_input_features") or {},
        recommended_bid=row.get("recommended_bid"),
        recommended_bid_predicted_win_rate=row.get(
            "recommended_bid_predicted_win_rate"
        ),
    )
    model = _load_model_for_row(row)
    factors = server_explain.explain_from_prediction(model, prediction, with_llm=False)
    return {
        "base_prediction": factors.get("base_prediction"),
        "prediction": factors.get("prediction"),
        "feature_contributions": factors.get("feature_contributions", []),
        "top_factors": factors["top_factors"],
        "base_win_rate": factors["base_win_rate"],
    }


def _pending_select(limit: int):
    """SELECT statement for rows still awaiting a SHAP explanation."""
    return (
        select(prediction_log_table)
        .where(
            prediction_log_table.c.shap_explanation.is_(None),
            prediction_log_table.c.decision_path.in_(_PENDING_DECISION_PATHS),
            prediction_log_table.c.status.in_(_SERVED_OK_STATUSES),
            prediction_log_table.c.model_input_features.isnot(None),
        )
        .order_by(prediction_log_table.c.created_at)
        .limit(limit)
    )


def process_one(store: PredictionLogStore) -> bool:
    """Claim and enrich a single pending row. Returns False if none pending.

    The claim + backfill happen in one transaction so the row lock (on
    Postgres) is held across the SHAP compute, guaranteeing no other worker
    touches it. On a hard failure a sentinel is written so the row isn't
    retried forever.
    """
    supports_skip_locked = store.engine.dialect.name == "postgresql"
    with store.engine.begin() as conn:
        stmt = _pending_select(limit=1)
        if supports_skip_locked:
            stmt = stmt.with_for_update(skip_locked=True)
        found = conn.execute(stmt).first()
        if found is None:
            return False

        row = _decode_row(found._mapping)
        prediction_id = row["prediction_id"]
        try:
            payload = _compute_shap_payload(row)
        except Exception as exc:  # noqa: BLE001 -- enrichment is best-effort
            logger.warning(
                "SHAP enrichment failed for prediction_id=%s; writing sentinel",
                prediction_id,
                exc_info=True,
            )
            payload = {"shap_error": str(exc)}

        conn.execute(
            update(prediction_log_table)
            .where(prediction_log_table.c.prediction_id == prediction_id)
            .values(shap_explanation=_json_or_none(payload))
        )
    return True


def pending_count(store: PredictionLogStore) -> int:
    """Number of rows currently awaiting a SHAP explanation (backlog gauge)."""
    from sqlalchemy import func

    with store.engine.begin() as conn:
        return int(
            conn.execute(
                select(func.count()).select_from(
                    _pending_select(limit=1_000_000_000).subquery()
                )
            ).scalar_one()
        )


def run(
    poll_secs: float | None = None,
    batch: int | None = None,
    once: bool | None = None,
) -> None:
    """Long-running drain loop (or a single drain when ``once``)."""
    poll_secs = (
        poll_secs
        if poll_secs is not None
        else float(os.getenv("SMARTHUB_SHAP_WORKER_POLL_SECS", "1.0"))
    )
    batch = (
        batch
        if batch is not None
        else int(os.getenv("SMARTHUB_SHAP_WORKER_BATCH", "50"))
    )
    once = (
        once
        if once is not None
        else os.getenv("SMARTHUB_SHAP_WORKER_ONCE", "").strip() in {"1", "true"}
    )

    store = PredictionLogStore()
    logger.info(
        "shap-worker started (db=%s, poll=%.1fs, batch=%d, once=%s)",
        store.engine.url.render_as_string(hide_password=True),
        poll_secs,
        batch,
        once,
    )

    stopping = {"flag": False}

    def _stop(signum, _frame):
        logger.info("shap-worker received signal %s; stopping", signum)
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not stopping["flag"]:
        processed = 0
        while processed < batch and not stopping["flag"]:
            if not process_one(store):
                break
            processed += 1
        if once and processed == 0:
            break
        if processed == 0:
            time.sleep(poll_secs)
        else:
            logger.info("shap-worker enriched %d prediction(s)", processed)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("SMARTHUB_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run()


if __name__ == "__main__":
    main()
