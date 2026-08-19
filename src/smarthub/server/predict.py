"""Online model loading and bid prediction for SmartHub.

This module resolves serving models and exposes bid recommendation endpoints.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from smarthub import __version__ as _PACKAGE_VERSION
from smarthub.core.lead_types import all_lead_type_ids
from smarthub.core.lead_types import lead_type_id as get_lead_type_id
from smarthub.core.lead_types import lead_type_name as get_lead_type_name
from smarthub.train_and_predict import config, optimizer, preprocessing, registry

logger = logging.getLogger(__name__)


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/±inf) with None.

    SHAP factor values can be NaN (e.g. a numeric feature missing from the
    prepared row coerces to NaN), and Starlette's JSON encoder runs with
    ``allow_nan=False`` -- so an un-sanitized NaN anywhere in the response
    500s the request at serialization time, after the handler has returned.
    Applied at the serialization boundary so no explanation payload can break
    ``/explain_bid``.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):  # np.float32/float64/int64 -> native
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def resolve_model_uri(lead_type_id: int = get_lead_type_id("auto")) -> str:
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

    lead_type_name = get_lead_type_name(lead_type_id)
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


# In-process model cache -- /recommend_bid sits in the real-time bid path, so
# re-unpickling from disk on every single request is latency worth avoiding.
# Keyed by (resolved uri, mtime-if-local-pkl):
#   - Registry-resolved URIs are immutable versioned filenames
#     (v{n}_{timestamp}.pkl -- see registry.py), so a new promoted model
#     always resolves to a *different* uri string, which naturally busts the
#     cache with no extra bookkeeping.
#   - The mtime is a safety net for the MODEL_URI env-override case, where a
#     caller can point at (and later overwrite in place) the same local path
#     -- without it, an overwritten pinned .pkl would keep serving the old
#     in-memory model until the process restarts.
#   - Non-.pkl (MLflow) URIs have no local mtime to check, so a
#     `models:/name/Production`-style alias that starts pointing at a
#     different underlying run without the URI string changing would not
#     bust the cache. Not a concern for the .pkl-based registry path this
#     repo actually uses, but worth knowing if MLflow URIs become common.
_MODEL_CACHE: dict[tuple[str, float | None], object] = {}


def _model_cache_key(uri: str) -> tuple[str, float | None]:
    """Return the cache key for a resolved model URI (see _MODEL_CACHE)."""
    if uri.endswith(".pkl"):
        try:
            mtime: float | None = os.path.getmtime(uri)
        except OSError:
            mtime = None
        return uri, mtime
    return uri, None


def clear_model_cache() -> None:
    """Drop all cached in-memory models and manifests (tests/manual invalidation)."""
    _MODEL_CACHE.clear()
    _MANIFEST_CACHE.clear()


# Manifest cache -- keyed by (lead_type_name, version), NOT by mtime like
# _MODEL_CACHE, because a version's manifest never changes after it's written
# (registry.save_version() writes manifest_path() once, at creation, and
# nothing ever rewrites it in place -- see registry.py). So unlike the model
# cache, this one needs no invalidation logic at all: a given key's value can
# never go stale.
#
# Added because `load_model_and_manifest`'s "currently serving" tier used to
# call three separate registry helpers that each independently re-read
# current.json to re-derive the same version string -- i.e. redundant disk
# I/O on every single /recommend_bid request, on top of being uncached across
# requests. That's latency worth avoiding in the real-time bid path, same
# reasoning as _MODEL_CACHE above.
_MANIFEST_CACHE: dict[tuple[str, str], dict] = {}


def _cached_manifest(lead_type_name: str, version: str) -> dict | None:
    """Return a model version's manifest, from cache when possible.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    version : str
        Model version identifier.

    Returns
    -------
    dict | None
        The manifest, or None if this version has no manifest on disk.
    """
    key = (lead_type_name, version)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        manifest = registry.load_manifest(lead_type_name, version)
    except FileNotFoundError:
        return None
    _MANIFEST_CACHE[key] = manifest
    return manifest


def is_model_cached(lead_type_id: int = get_lead_type_id("auto")) -> bool:
    """Return whether a lead type's currently-resolved model is in memory.

    Cheap: only resolves the URI and checks the cache dict, never loads or
    touches the model file itself beyond the mtime stat `_model_cache_key`
    already needs. Used by `/health` to report status without forcing a load.
    """
    try:
        uri = resolve_model_uri(lead_type_id)
    except FileNotFoundError:
        return False
    return _model_cache_key(uri) in _MODEL_CACHE


def load_model(
    model_uri: str | None = None,
    lead_type_id: int = get_lead_type_id("auto"),
):
    """Load a local or MLflow model artifact, from cache when possible.

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
    key = _model_cache_key(uri)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    if uri.endswith(".pkl"):
        import joblib

        model = joblib.load(uri)
    else:
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(uri)

    _MODEL_CACHE[key] = model
    return model


def load_model_and_manifest(lead_type_id: int = get_lead_type_id("auto")):
    """Resolve + load the serving model and its manifest, cold-start-safe.

    Same three-tier resolution order as `resolve_model_uri` (`MODEL_URI` env
    override -> pinned `[prediction] active_model_version` -> the
    currently-serving registry pointer), but degrades to `(None, None)`
    instead of raising `FileNotFoundError` at any tier -- lets `decide_bid`
    tell true cold start (nothing ever trained/promoted) apart from a real
    error, without every caller needing its own try/except.

    Inputs
    ------
    lead_type_id : int
        SmartHub lead type identifier.

    Returns
    -------
    tuple[Any | None, dict | None]
        Loaded model (via the shared cache in `load_model`) and its
        manifest (via `_cached_manifest`), or `(None, None)` if nothing
        resolves. The `MODEL_URI` env-override tier has no registry manifest
        to pair with an arbitrary pinned URI, so it always returns `(model,
        None)` -- `decide_bid` treats a `None` manifest as "age unknown",
        not stale.

    Notes
    -----
    The "currently serving" tier reads `current.json` exactly once per call
    -- that single read IS the cache-busting check (same role as the model
    cache's own mtime check): it's how a new promotion gets picked up
    without a restart. Everything after that is resolved from the already-
    known `version` and cached, since a version's manifest/model path never
    change once written. This used to call three separate registry helpers
    that each independently re-derived the same version by re-reading
    current.json -- redundant disk I/O on every request; fixed here.
    """
    env_override = os.getenv("MODEL_URI")
    if env_override:
        return load_model(model_uri=env_override), None

    lead_type_name = get_lead_type_name(lead_type_id)

    pinned_version = config.active_model_version()
    version = pinned_version or registry.currently_serving_version(lead_type_name)
    if version is None:
        return None, None

    manifest = _cached_manifest(lead_type_name, version)
    if manifest is None:
        return None, None
    # Production-aware: resolves the local artifact, or the downloaded+cached
    # copy from production storage (S3/MinIO) when configured.
    model_path = registry.serving_model_path(lead_type_name, version)
    if model_path is None:
        return None, None
    return load_model(model_uri=str(model_path)), manifest


def _eager_load_models() -> None:
    """Warm the model cache for every configured lead type at startup.

    Best-effort: a lead type with no promoted model yet (cold start) or any
    other load failure is logged, not raised -- the API should still start
    and serve the lead types that DO have a model; a missing/broken model
    for one lead type is already handled per-request (FileNotFoundError /
    the eventual exception surfaces on that request, same as before this
    existed).
    """
    for lead_type_id in all_lead_type_ids():
        try:
            load_model(lead_type_id=lead_type_id)
        except FileNotFoundError:
            logger.warning(
                "No model to eager-load for lead_type_id=%s yet (cold start) "
                "-- will attempt again lazily on first request once one is "
                "trained and promoted.",
                lead_type_id,
            )
        except Exception:
            logger.exception(
                "Failed to eager-load model for lead_type_id=%s at startup",
                lead_type_id,
            )


# Backward-compatible public import. The implementation belongs to optimizer.py.
optimize_bid_for_row = optimizer.optimize_bid_for_row


def exploration_slot(
    created_dayofweek: int | None,
    created_hour: int | None,
    variance_pct: float | None = None,
) -> tuple[bool, int]:
    """Decide whether a lead falls in a scheduled exploration probe slot.

    Deterministic, reproducible schedule (never a per-request coin flip):
    every lead is bucketed by hour-of-week (0-167, from
    `created_dayofweek`/`created_hour`), and 1-in-`N` buckets (`N = round(1 /
    exploration_variance_pct)`) are scheduled explore slots. The same lead
    (same day-of-week + hour) always gets the same explore/exploit decision,
    so it can be recomputed and audited later.

    Inputs
    ------
    created_dayofweek : int | None
        0 (Monday) - 6 (Sunday). `None` disables exploration (can't bucket).
    created_hour : int | None
        0-23. `None` disables exploration (can't bucket).
    variance_pct : float | None
        Overrides `config.exploration_variance_pct()` (mainly for tests).
        `<= 0` disables exploration entirely.

    Returns
    -------
    tuple[bool, int]
        `(is_explore_slot, direction)` -- `direction` is `+1`/`-1` when
        `is_explore_slot` is True, else `0`.
    """
    if created_dayofweek is None or created_hour is None:
        return False, 0

    pct = config.exploration_variance_pct() if variance_pct is None else variance_pct
    if pct is None or pct <= 0:
        return False, 0

    n = round(1 / pct)
    if n <= 0:
        return False, 0

    bucket = (int(created_dayofweek) % 7) * 24 + (int(created_hour) % 24)
    if bucket % n != 0:
        return False, 0

    # Alternate direction across successive triggering buckets (0, N, 2N, ...)
    # rather than per-request randomness -- still a pure function of the
    # bucket, so it's reproducible.
    occurrence = bucket // n
    direction = 1 if occurrence % 2 == 0 else -1
    return True, direction


def _snap_to_grid(
    value: float, min_bid: float, max_bid: float, bid_step: float
) -> float:
    """Snap `value` onto the same `min_bid + k * bid_step` grid the optimizer's
    candidate bids sit on, then clip to `[min_bid, max_bid]`."""
    if bid_step is None or bid_step <= 0:
        return max(min_bid, min(value, max_bid))
    steps = round((value - min_bid) / bid_step)
    snapped = min_bid + steps * bid_step
    return max(min_bid, min(snapped, max_bid))


def _score_bid(row, model, bid: float, expected_revenue: float) -> tuple[float, float]:
    """Predicted win rate + expected profit for one row at one specific bid."""
    candidate_row = pd.DataFrame([row.to_dict()])
    candidate_row["bid"] = bid
    with optimizer.quiet_feature_name_warning():
        win_rate = float(model.predict_proba(candidate_row)[:, 1][0])
    profit = win_rate * (expected_revenue - bid)
    return win_rate, profit


def _model_data_age_days(manifest: dict | None) -> int | None:
    """Days since a model version's manifest was written, or `None` if the
    manifest is missing/malformed (age unknown, not stale)."""
    if not manifest or not manifest.get("created_at"):
        return None
    try:
        created_at = datetime.fromisoformat(manifest["created_at"])
    except (TypeError, ValueError):
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created_at).days


def decide_bid(
    row,
    model,
    manifest,
    expected_revenue: float,
    target_cm: float,
    min_bid: float,
    bid_step: float,
    created_dayofweek: int | None = None,
    created_hour: int | None = None,
    include_candidates: bool = False,
) -> dict:
    """The one bidding decision -- always one explicit, auditable path.

    Never emergent/random behavior (docs/CONTEXT.md §7): picks exactly one
    of three decision paths and says why.

    - `"model"` -- the normal case: the profit-maximizing bid from
      `model`/`manifest`. Flags `model_data_age_days` when the serving
      model's training data is older than `config.recency_window_days()`
      (informational only -- doesn't change the bid).
    - `"cold_start_fallback"` -- `model is None`: no model has ever been
      trained/promoted for this lead type. Bids a fixed, configurable
      fraction (`config.cold_start_fallback_bid_pct()`) of the way from
      `min_bid` to the CM-respecting ceiling, snapped to `bid_step`.
      Self-terminating: stops firing the moment a first model promotes.
    - `"exploration"` -- a scheduled probe (see `exploration_slot`):
      perturbs the model's optimal bid by ±`exploration_variance_pct`
      instead of bidding it as-is, to learn the market's shape at other
      price points.

    Inputs
    ------
    row : pandas.Series
        Single model-ready feature row (e.g.
        `preprocessing.serving_frame(...).iloc[0]`).
    model : Any | None
        Fitted model, or `None` for true cold start.
    manifest : dict | None
        The model's registry manifest (for `model_data_age_days`), or `None`.
    expected_revenue : float
        Expected revenue if the lead is won.
    target_cm : float
        Target contribution margin as a decimal.
    min_bid : float
        Minimum candidate bid (partner-side floor).
    bid_step : float
        Increment between candidate bids.
    created_dayofweek : int | None
        0-6, for the exploration schedule.
    created_hour : int | None
        0-23, for the exploration schedule.

    Returns
    -------
    dict
        `recommended_bid`, `recommended_bid_predicted_win_rate`,
        `recommended_bid_predicted_profit`, `max_bid`, `n_candidate_bids`,
        `decision_path`, `decision_reason`, `model_data_age_days`; plus
        `candidate_evaluations` (the full per-candidate sweep) when
        `include_candidates` is set and a model actually scored candidates.
    """
    candidate_bids, max_bid = optimizer.candidate_bids_for_revenue(
        expected_revenue, target_cm, min_bid, bid_step
    )

    if len(candidate_bids) == 0:
        # No viable bid at all -- expected revenue is too low to clear the
        # floor while still meeting the target margin. True regardless of
        # which path we'd otherwise take, so report it plainly rather than
        # picking a path that doesn't apply.
        result = optimizer.empty_result(max_bid)
        # optimizer.py's own internal naming convention
        # (recommended_bid_expected_profit, shared with the training-time
        # bulk evaluation module optimizer_evaluation.py) is intentionally
        # NOT renamed -- only this public API/logging-facing name is.
        result["recommended_bid_predicted_profit"] = result.pop(
            "recommended_bid_expected_profit"
        )
        result["decision_path"] = "cold_start_fallback" if model is None else "model"
        result["decision_reason"] = (
            "No viable bid: expected revenue is too low to bid anything at "
            "or above the minimum bid while still meeting the target margin."
        )
        result["model_data_age_days"] = _model_data_age_days(manifest)
        return result

    if model is None:
        pct = config.cold_start_fallback_bid_pct()
        raw_bid = min_bid + pct * (max_bid - min_bid)
        snapped_bid = _snap_to_grid(raw_bid, min_bid, max_bid, bid_step)
        return {
            "recommended_bid": snapped_bid,
            "recommended_bid_predicted_win_rate": None,
            "recommended_bid_predicted_profit": None,
            "max_bid": max_bid,
            "n_candidate_bids": int(len(candidate_bids)),
            "decision_path": "cold_start_fallback",
            "decision_reason": (
                "No model has ever been trained/promoted for this lead type "
                f"yet. Bidding {pct:.0%} of the way from the floor "
                f"(${min_bid:.2f}) to the CM-respecting ceiling "
                f"(${max_bid:.2f}) instead of guessing or erroring."
            ),
            "model_data_age_days": None,
        }

    result = optimizer.optimize_bid_for_row(
        row,
        model,
        expected_revenue,
        target_cm,
        min_bid,
        bid_step,
        include_candidates=include_candidates,
    )
    # See the empty_result() branch above -- same rename, same reason.
    result["recommended_bid_predicted_profit"] = result.pop(
        "recommended_bid_expected_profit"
    )

    explore, direction = exploration_slot(created_dayofweek, created_hour)
    if explore and not pd.isna(result["recommended_bid"]):
        pct = config.exploration_variance_pct()
        perturbed_raw = result["recommended_bid"] * (1 + direction * pct)
        perturbed_bid = _snap_to_grid(perturbed_raw, min_bid, max_bid, bid_step)
        win_rate, profit = _score_bid(row, model, perturbed_bid, expected_revenue)
        result["recommended_bid"] = perturbed_bid
        result["recommended_bid_predicted_win_rate"] = win_rate
        result["recommended_bid_predicted_profit"] = profit
        result["decision_path"] = "exploration"
        sign = "+" if direction > 0 else "-"
        result["decision_reason"] = (
            "Scheduled exploration probe: perturbing the profit-maximizing "
            f"bid {sign}{pct:.0%} on a defined, reproducible hour-of-week "
            "schedule to keep learning the market's shape."
        )
    else:
        result["decision_path"] = "model"
        result["decision_reason"] = (
            "Standard profit-maximizing bid from the currently-serving model."
        )

    age_days = _model_data_age_days(manifest)
    result["model_data_age_days"] = age_days
    if age_days is not None and age_days > config.recency_window_days():
        result["decision_reason"] += (
            f" Note: this model's training data is {age_days} days old, "
            f"older than the configured recency window "
            f"({config.recency_window_days()} days) -- may be due for "
            "retraining."
        )

    return result


def bid_curve_around(
    row,
    model,
    expected_revenue: float,
    min_bid: float,
    max_bid: float,
    bid_step: float,
    center_bid: float,
    n_points: int = 3,
) -> list[dict]:
    """Predicted win rate + expected profit at a few bid points bracketing
    `center_bid` -- "the shape of the market" around the chosen bid, not
    just the one winning number (docs/CONTEXT.md §7).

    Inputs
    ------
    row : pandas.Series
        Single model-ready feature row.
    model : Any | None
        Fitted model. `None` (or a NaN/missing `center_bid`/`max_bid`)
        returns an empty list -- no model to score other bids with.
    expected_revenue : float
        Expected revenue if the lead is won.
    min_bid : float
        Minimum candidate bid.
    max_bid : float
        CM-respecting ceiling.
    bid_step : float
        Increment between candidate bids.
    center_bid : float
        Bid to bracket (typically `decide_bid`'s `recommended_bid`).
    n_points : int
        Number of bid points, spanning symmetrically around `center_bid`.

    Returns
    -------
    list[dict]
        Each point: `{"bid", "predicted_win_rate", "expected_profit"}`,
        sorted ascending by bid, edge-clipped to `[min_bid, max_bid]` with
        duplicate (clipped) bids collapsed.
    """
    if model is None or pd.isna(center_bid) or pd.isna(max_bid):
        return []

    half = n_points // 2
    offsets = range(-half, n_points - half)
    bids = sorted(
        {
            max(min_bid, min(max_bid, center_bid + offset * bid_step))
            for offset in offsets
        }
    )

    frame = pd.DataFrame([row.to_dict()] * len(bids))
    frame["bid"] = bids
    with optimizer.quiet_feature_name_warning():
        win_rates = model.predict_proba(frame)[:, 1]
    bids_arr = np.array(bids, dtype=float)
    profits = win_rates * (expected_revenue - bids_arr)

    return [
        {
            "bid": float(b),
            "predicted_win_rate": float(w),
            "expected_profit": float(p),
        }
        for b, w, p in zip(bids, win_rates, profits)
    ]


try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
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
        # Optional: threads through to the prediction log (§8 of
        # docs/PREDICTION_LOG_SCHEMA.md) so a prediction can be joined back
        # to a specific lead later -- e.g. against `public.lead_pings.won`
        # for realized win-rate calibration. Never used in the bid decision
        # itself, purely a correlation key for logging.
        lead_ping_id: int | None = None
        lead_type_id: int = get_lead_type_id("auto")
        created_hour: int = Field(..., ge=0, le=23)
        created_dayofweek: int = Field(..., ge=0, le=6)

        # Response control (NOT a feature -- never affects the bid). When true,
        # the response also returns the full decision payload the prediction log
        # uses: model + input snapshot + optimizer config + a capped candidate
        # sweep (the selected bid + the first 19 by bid). Off by default so normal
        # high-QPS traffic stays lean. SHAP is never in the response (it's async);
        # fetch it later by prediction_id. See docs/PREDICTION_LOG_SCHEMA.md.
        verbose: bool = False

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

    class ExplainRequest(BaseModel):
        """Explain an already-logged prediction, by id (production mode).

        Consumes a persisted prediction rather than re-deciding the bid, so the
        explanation always corresponds to the logged bid and to the model
        version that served it. (A raw-lead / predict-then-explain local mode is
        deferred -- see the ticket's Future Work.)
        """

        prediction_id: str
        with_llm: bool = True
        with_bid_curve: bool = True

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Warm the model cache once at startup (see _eager_load_models),
        and kick off a background Ollama model-pull check.

        Local import (not top-level): `explain.py` imports `smarthub.server`
        (this module), so importing it here at call time avoids a circular
        import -- same reason `/explain_bid` imports it lazily. Both this
        check and the pull it may trigger run in a background thread (see
        `explain.ensure_model_pulled_async`) -- neither blocks startup from
        completing, nor blocks any request while a multi-minute pull runs.
        """
        _eager_load_models()
        _eager_init_db()
        _start_log_writer()  # decoupled prediction-log writer thread
        from smarthub.train_and_predict import explain

        explain.ensure_model_pulled_async()
        yield
        _stop_log_writer()  # flush remaining log rows on shutdown

    def _eager_init_db() -> None:
        """Create SmartHub's Postgres tables at startup, so they exist right
        after deploy rather than lazily on first use.

        Constructing each store runs SQLAlchemy ``create_all`` (idempotent),
        creating ``smarthub_prediction_log`` and ``smarthub_config`` /
        ``smarthub_config_history`` in the shared Postgres. Best-effort: an
        unreachable DB is logged and swallowed, never blocking startup -- same
        principle as the lazily-built prediction logger.
        """
        try:
            _prediction_log_store()  # create_all -> smarthub_prediction_log
        except Exception:  # noqa: BLE001 -- must never block startup
            logger.warning(
                "Could not eager-init the prediction-log table at startup "
                "(will retry lazily on first prediction).",
                exc_info=True,
            )
        try:
            from smarthub.core.config_store import ConfigStore

            ConfigStore()  # create_all -> smarthub_config(+_history)
        except Exception:  # noqa: BLE001 -- must never block startup
            logger.warning(
                "Could not eager-init the config-store tables at startup.",
                exc_info=True,
            )

    app = FastAPI(title="Anton Bid Prediction API", lifespan=_lifespan)

    @app.get("/health")
    def health(lead_type_id: int = get_lead_type_id("auto")):
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
            # True only if the model is currently cached in memory -- a cheap
            # check (no disk load), so a cold-start lead type with no model
            # yet correctly reports False rather than erroring.
            "model_loaded": is_model_cached(lead_type_id),
        }

    def _record_and_row(request: BidRequest):
        """Shared request -> (record dict, model-ready row) prep for both
        /recommend_bid and /explain_bid, so the two stay in lockstep."""
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
        return record, frame.iloc[0]

    # --- Prediction logging (docs/PREDICTION_LOG_SCHEMA.md) -----------------
    # One row per /recommend_bid or /explain_bid call, success or failure --
    # per the 2026-07-22 DS meeting decision. Constructed lazily (not at
    # import/startup time) so a missing/unreachable logging DB can't prevent
    # the API from starting or serving bids, and every call site wraps this
    # in a try/except that only ever logs a warning -- a logging outage must
    # never take down live bidding, same principle as SHAP not delaying a
    # prediction.
    _prediction_log_store_holder: dict = {}

    def _prediction_log_store():
        """Lazily construct and cache the shared PredictionLogStore."""
        if "store" not in _prediction_log_store_holder:
            from smarthub.train_and_predict.prediction_log_schema import (
                PredictionLogStore,
            )

            _prediction_log_store_holder["store"] = PredictionLogStore()
        return _prediction_log_store_holder["store"]

    def _manifest_log_fields(manifest: dict | None) -> dict:
        """Manifest-derived fields the prediction log wants, or all-None."""
        if not manifest:
            return {
                "model_name": None,
                "model_version": None,
                "model_uri": None,
                "model_type": None,
                "model_calibrated": None,
                "training_table_version": None,
                "model_data_min_created_at": None,
                "model_data_max_created_at": None,
            }
        lineage = manifest.get("lineage") or {}
        return {
            # Human-friendly name: the production slug (e.g. "auto_v6") when
            # promoted, else the lead-type name.
            "model_name": (
                manifest.get("production_model_version")
                or manifest.get("lead_type_name")
            ),
            "model_version": manifest.get("version"),
            "model_uri": manifest.get("model_path"),
            "model_type": lineage.get("model_type"),
            "model_calibrated": lineage.get("calibrated"),
            "training_table_version": lineage.get("training_table_version"),
            "model_data_min_created_at": lineage.get("data_min_created_at"),
            "model_data_max_created_at": lineage.get("data_max_created_at"),
        }

    def _serving_config_snapshot() -> dict:
        """Snapshot of the serving-policy config in effect right now, for
        reproducibility (Vinaya's config-versioning ask, same meeting)."""
        return {
            "exploration_variance_pct": config.exploration_variance_pct(),
            "recency_window_days": config.recency_window_days(),
            "cold_start_fallback_bid_pct": config.cold_start_fallback_bid_pct(),
        }

    # Fields copied verbatim from the log record into the verbose response, so
    # the response and the logged row can never disagree (single construction).
    _VERBOSE_RESPONSE_KEYS = (
        "served_at",
        "tat_seconds",
        "package_version",
        "lead_type_id",
        "lead_type_name",
        "campaign_id",
        "account_id",
        "source_type_id",
        "input_features",
        "model_input_features",
        "feature_cols",
        "expected_revenue",
        "target_cm",
        "min_bid",
        "bid_step",
        "candidate_bid_generation",
        "model_name",
        "model_version",
        "model_uri",
        "model_type",
        "model_calibrated",
        "training_table_version",
        "model_data_min_created_at",
        "model_data_max_created_at",
        "recommended_bid_predicted_cm",
        "serving_config",
    )

    def _response_candidate_slice(candidates, first_n: int = 19):
        """Capped candidate sweep for the verbose response.

        Returns the selected bid plus the first ``first_n`` candidates by bid
        (<=20 total), so the chosen bid is always present without shipping the
        full ~300-row sweep. The complete sweep still goes to the log.
        """
        if not candidates:
            return candidates
        head = list(candidates[:first_n])  # already bid-ascending
        if not any(c.get("selected") for c in head):
            selected = next((c for c in candidates if c.get("selected")), None)
            if selected is not None:
                head.append(selected)
        return sorted(head, key=lambda c: c.get("bid", 0.0))

    def _verbose_response(log_record: dict, full_candidates) -> dict:
        """Full decision payload for a ``verbose`` request, built from the same
        ``log_record`` that is persisted (so response == logged row), with the
        candidate sweep capped for the wire and SHAP intentionally omitted
        (it's attached asynchronously; fetch it later by ``prediction_id``)."""
        payload = {k: log_record.get(k) for k in _VERBOSE_RESPONSE_KEYS}
        served = payload.get("served_at")
        if hasattr(served, "isoformat"):
            payload["served_at"] = served.isoformat()
        payload["candidate_evaluations"] = _response_candidate_slice(full_candidates)
        return payload

    def _predicted_cm(
        predicted_profit: float | None, expected_revenue: float
    ) -> float | None:
        """Predicted contribution margin at the recommended bid --
        `recommended_bid_predicted_profit / expected_revenue`.

        `None` whenever there's no predicted profit to divide (cold start,
        "no viable bid", or an error before a bid was reached) -- never
        raises on a zero/falsy `expected_revenue` either, even though
        `BidRequest`'s own validation (`gt=0`) already rules that out for a
        real request.
        """
        if predicted_profit is None or not expected_revenue:
            return None
        return predicted_profit / expected_revenue

    def _log_prediction_safe(**kwargs) -> str | None:
        """Best-effort prediction logging -- never raises into the caller.

        Two different calling conventions, both routed through here:
        `/explain_bid` calls this synchronously (SHAP already ran inline
        there, so there's no separate latency budget to protect) and reads
        the returned `prediction_id` straight into its response. `/recommend_bid`
        instead generates its own `prediction_id` up front and schedules
        this same function as a `BackgroundTasks` job (passing that id
        explicitly via the `prediction_id` kwarg) -- so the insert this
        performs happens strictly *after* the bid response is already sent,
        never delaying it. Same failure isolation either way: any exception
        is caught and logged as a warning, never raised.

        Returns
        -------
        str | None
            The logged row's `prediction_id` (so a synchronous caller --
            `/explain_bid` -- can hand it back to the caller as a receipt --
            see docs/PREDICTION_LOG_SCHEMA.md §8), or `None` if the write
            itself failed. Ignored when this runs as a background task
            (`/recommend_bid`'s `prediction_id` was already decided and
            returned before this ever runs).
        """
        try:
            return _prediction_log_store().log_prediction(**kwargs)
        except Exception:  # noqa: BLE001 -- logging must never break serving
            logger.warning("Failed to write prediction log row", exc_info=True)
            return None

    # --- Decoupled prediction-log writer --------------------------------------
    # /recommend_bid enqueues a fully-built log row and returns immediately; a
    # dedicated daemon thread performs the Postgres INSERTs off the request
    # path. This is stronger than a Starlette BackgroundTask (which still runs
    # in the request's own worker after the response and can contend with the
    # next request under DB load) -- here the request path does no DB I/O at
    # all, so logging can never affect TAT. One queue + thread per uvicorn
    # worker process.
    _LOG_QUEUE: "queue.Queue[dict | None]" = queue.Queue(maxsize=50000)
    _LOG_WRITER = {"thread": None, "stop": threading.Event()}

    def _enqueue_log(record: dict) -> None:
        """Hand a prediction-log row to the writer thread (non-blocking).

        ``SMARTHUB_PREDICTION_LOG_SYNC=1`` forces an inline synchronous write
        instead -- used by tests so the row is visible immediately after the
        request, and available as an escape hatch for debugging. Production
        leaves it unset (async, off the request path).
        """
        if os.getenv("SMARTHUB_PREDICTION_LOG_SYNC", "").strip() in {"1", "true"}:
            _log_prediction_safe(**record)
            return
        try:
            _LOG_QUEUE.put_nowait(record)
        except queue.Full:
            # Backpressure: drop the row rather than block the request. Serving
            # correctness never depends on a log row existing.
            logger.warning(
                "prediction-log queue full -- dropping a log row (serving "
                "unaffected). Consider a faster log DB or a bigger queue."
            )

    def _log_writer_loop() -> None:
        """Drain the queue and write rows, one small batched transaction at a
        time, until stopped and drained."""
        stop = _LOG_WRITER["stop"]
        while True:
            try:
                first = _LOG_QUEUE.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            if first is None:  # shutdown sentinel
                break
            # Opportunistically batch anything already queued into one txn.
            batch = [first]
            while len(batch) < 500:
                try:
                    nxt = _LOG_QUEUE.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    stop.set()
                    break
                batch.append(nxt)
            try:
                _prediction_log_store().log_many(batch)
            except Exception:  # noqa: BLE001 -- never let the writer die
                logger.warning(
                    "prediction-log batch write failed (%d rows dropped)",
                    len(batch),
                    exc_info=True,
                )

    def _start_log_writer() -> None:
        """Start the writer thread once per process (idempotent)."""
        if _LOG_WRITER["thread"] is not None:
            return
        _LOG_WRITER["stop"].clear()
        t = threading.Thread(
            target=_log_writer_loop, name="predlog-writer", daemon=True
        )
        t.start()
        _LOG_WRITER["thread"] = t

    def _stop_log_writer() -> None:
        """Signal stop and flush remaining rows on shutdown (best-effort)."""
        t = _LOG_WRITER["thread"]
        if t is None:
            return
        _LOG_WRITER["stop"].set()
        _LOG_QUEUE.put(None)  # wake the thread
        t.join(timeout=10)

    def _log_shap_background(
        prediction_id: str | None,
        model,
        model_input_features: dict,
        lead_type_id: int,
        recommended_bid: float,
        recommended_bid_predicted_win_rate: float | None = None,
    ) -> None:
        """Best-effort: compute SHAP factors for a /recommend_bid prediction
        and attach them to its already-written log row.

        Consumes the prediction's already-prepared feature row via
        `server.explain.explain_from_prediction` (`with_llm=False`) -- the bid
        is never recomputed and no optimizer sweep is re-run; only SHAP at the
        chosen bid. Runs as a `BackgroundTasks` job (after the response is sent,
        per Starlette semantics), so it never adds latency to the bid decision.
        The LLM narrative and `bid_curve` stay /explain_bid-only.

        Silently gives up (logs a warning) on any failure -- e.g. a non-LightGBM
        model (SHAP supports `model_type='lightgbm'` only), or the logging DB
        being unreachable for the update -- since this is pure after-the-fact
        enrichment and must never surface anywhere a caller would see it.
        """
        if prediction_id is None or model is None:
            return
        try:
            from smarthub.server import explain as server_explain

            prediction = server_explain.PredictionOutput(
                lead_type_id=lead_type_id,
                model_input_features=model_input_features,
                recommended_bid=recommended_bid,
                recommended_bid_predicted_win_rate=(recommended_bid_predicted_win_rate),
            )
            factors = server_explain.explain_from_prediction(
                model, prediction, with_llm=False
            )
            _prediction_log_store().update_shap_explanation(
                prediction_id,
                {
                    "base_prediction": factors.get("base_prediction"),
                    "prediction": factors.get("prediction"),
                    "feature_contributions": factors.get("feature_contributions", []),
                    "top_factors": factors["top_factors"],
                    "base_win_rate": factors["base_win_rate"],
                },
            )
        except Exception:  # noqa: BLE001 -- best-effort enrichment only
            logger.warning(
                "Failed to compute/attach background SHAP explanation for "
                "prediction_id=%s",
                prediction_id,
                exc_info=True,
            )

    @app.post("/recommend_bid")
    def recommend_bid(request: BidRequest, background_tasks: BackgroundTasks):
        """Return the expected-profit-maximizing bid for one request.

        Runs every lead through `decide_bid` -- the same cold-start
        fallback / scheduled exploration / normal-optimizer policy
        `/explain_bid` uses, so the two always agree on what live serving
        would do for the same inputs.

        Inputs
        ------
        request : BidRequest
            Validated bid recommendation request.

        Returns
        -------
        dict
            Recommended bid, `decision_path`/`decision_reason`, supporting
            metrics, `prediction_id` -- a receipt for this prediction's log
            row -- and `lead_ping_id`, echoed back exactly as sent (`None`
            if the caller didn't supply one).

        Notes
        -----
        Nothing about prediction logging is ever on this response's
        critical path. `prediction_id` is generated here, in the request
        handler, and returned immediately -- the actual log-row insert
        (including every derived metric, e.g. `recommended_bid_predicted_cm`)
        happens entirely in a `BackgroundTasks` job that Starlette runs only
        *after* this response has been sent (see `_log_prediction_safe`).
        One consequence: `prediction_id` here is a receipt for "this is the
        id we intend to log under", not a guarantee the row exists yet -- if
        the background insert itself fails (logging DB down), it's caught
        and logged as a warning, same as any other logging failure in this
        module, with no way for the caller to know except that a later
        lookup by this id would come up empty. That's an intentional
        trade-off: the alternative is waiting for the DB write before
        responding, which is exactly the delay this design avoids.

        When a real model served this prediction (not cold start, and a
        viable bid exists), SHAP factors (`top_factors`/`base_win_rate`,
        no LLM narrative) are computed in a second background task,
        scheduled to run after the logging insert above, and attached to
        the same `prediction_id`'s log row. See `_log_shap_background`.
        """
        # TAT (turnaround time) clock: starts the moment the handler receives
        # the request, stops once the bid result is ready to return. Everything
        # after that -- the prediction-log insert and SHAP -- runs in background
        # tasks after the response is sent, so it's deliberately NOT counted.
        _t_start = time.perf_counter()
        lead_type_name = get_lead_type_name(request.lead_type_id)
        record = None
        try:
            model, manifest = load_model_and_manifest(request.lead_type_id)
            record, row = _record_and_row(request)
            result = decide_bid(
                row=row,
                model=model,
                manifest=manifest,
                expected_revenue=request.expected_revenue,
                target_cm=request.target_cm,
                min_bid=request.min_bid,
                bid_step=request.bid_step,
                created_dayofweek=request.created_dayofweek,
                created_hour=request.created_hour,
                include_candidates=True,
            )
        except Exception as exc:
            # Error path: also enqueued (off the request path) so even failure
            # logging can't block. The client gets its 500 immediately.
            _enqueue_log(
                dict(
                    endpoint="recommend_bid",
                    status="error",
                    error_message=str(exc),
                    lead_type_id=request.lead_type_id,
                    lead_type_name=lead_type_name,
                    campaign_id=request.campaign_id,
                    account_id=request.account_id,
                    source_type_id=request.source_type_id,
                    lead_ping_id=request.lead_ping_id,
                    input_features=record or request.model_dump(),
                    expected_revenue=request.expected_revenue,
                    target_cm=request.target_cm,
                    min_bid=request.min_bid,
                    bid_step=request.bid_step,
                )
            )
            raise

        # Generated here (not by the store) so it can be returned to the
        # caller right away, before the actual DB write ever happens -- see
        # the docstring's Notes section for the resulting trade-off.
        prediction_id = str(uuid.uuid4())
        result["prediction_id"] = prediction_id
        result["lead_ping_id"] = request.lead_ping_id

        # Stop the TAT clock here: the bid decision is done and the response is
        # about to be returned. Recorded in SECONDS and persisted to the log DB
        # only -- deliberately NOT returned in the response (it's an internal
        # serving metric, not something the caller needs on the bid).
        tat_seconds = round(time.perf_counter() - _t_start, 4)

        # Decoupled logging: hand the fully-built row to the in-process log
        # queue and return immediately. A dedicated writer thread does the
        # Postgres INSERT off the request path, so DB latency/contention can
        # never affect this request's TAT or the next one's (unlike a
        # BackgroundTask, which runs in this worker after the response). If the
        # queue is full the row is dropped with a warning -- serving is never
        # blocked or slowed by logging.
        served_at = datetime.now(timezone.utc)
        log_record = dict(
            endpoint="recommend_bid",
            prediction_id=prediction_id,
            served_at=served_at,
            package_version=_PACKAGE_VERSION,
            tat_seconds=tat_seconds,
            lead_type_id=request.lead_type_id,
            lead_type_name=lead_type_name,
            campaign_id=request.campaign_id,
            account_id=request.account_id,
            source_type_id=request.source_type_id,
            lead_ping_id=request.lead_ping_id,
            input_features=record,
            model_input_features=row.to_dict(),
            feature_cols=manifest.get("feature_cols") if manifest else None,
            expected_revenue=request.expected_revenue,
            target_cm=request.target_cm,
            min_bid=request.min_bid,
            bid_step=request.bid_step,
            candidate_bid_generation={
                "method": "equally_spaced",
                "min_bid": request.min_bid,
                "max_bid": result.get("max_bid"),
                "bid_step": request.bid_step,
                "n_candidates": result.get("n_candidate_bids"),
            },
            # Full optimizer sweep: every candidate bid the optimizer scored,
            # with its predicted win rate, expected profit, and whether it was
            # the selected (argmax-profit) bid. Lets a prediction's optimizer
            # decision be reconstructed without re-running the model.
            candidate_evaluations=result.get("candidate_evaluations"),
            **_manifest_log_fields(manifest),
            model_data_age_days=result.get("model_data_age_days"),
            decision_path=result.get("decision_path"),
            decision_reason=result.get("decision_reason"),
            recommended_bid=result.get("recommended_bid"),
            recommended_bid_predicted_win_rate=result.get(
                "recommended_bid_predicted_win_rate"
            ),
            recommended_bid_predicted_profit=result.get(
                "recommended_bid_predicted_profit"
            ),
            recommended_bid_predicted_cm=_predicted_cm(
                result.get("recommended_bid_predicted_profit"),
                request.expected_revenue,
            ),
            serving_config=_serving_config_snapshot(),
        )
        _enqueue_log(log_record)

        # The full candidate sweep is persisted to the log above. The default
        # response stays lean (sweep dropped); a `verbose` request instead gets
        # the full decision payload with the sweep capped (selected bid + first
        # 19 by bid). SHAP is never returned here -- fetch it later by id.
        full_candidates = result.pop("candidate_evaluations", None)
        if request.verbose:
            result.update(_verbose_response(log_record, full_candidates))

        # SHAP enrichment mode (config.shap_enrichment_mode / $SMARTHUB_SHAP_MODE):
        #   inprocess -> compute SHAP here, in a background task on this worker
        #                (legacy behaviour, kept for A/B TAT comparison).
        #   offload   -> do nothing here; the prediction row is logged with
        #                shap_explanation NULL and a separate `shap-worker`
        #                process backfills it, keeping serving CPU free.
        #   off       -> no SHAP enrichment at all.
        shap_mode = config.shap_enrichment_mode()
        if (
            shap_mode == config.SHAP_MODE_INPROCESS
            and model is not None
            and not pd.isna(result.get("recommended_bid"))
        ):
            # Scheduled after the logging insert above -- Starlette runs
            # BackgroundTasks in registration order, so this update always
            # finds its row already there (or, on a logging failure, simply
            # updates zero rows -- see update_shap_explanation).
            background_tasks.add_task(
                _log_shap_background,
                prediction_id,
                model,
                row.to_dict(),
                request.lead_type_id,
                result.get("recommended_bid"),
                result.get("recommended_bid_predicted_win_rate"),
            )

        return result

    def _prediction_output_from_log(row: dict):
        """Rebuild a ``server.explain.PredictionOutput`` from a logged row."""
        from smarthub.server import explain as server_explain

        cbg = row.get("candidate_bid_generation") or {}

        def _f(value):
            return float(value) if value is not None else None

        return server_explain.PredictionOutput(
            lead_type_id=int(row["lead_type_id"]),
            model_input_features=row.get("model_input_features") or {},
            recommended_bid=_f(row.get("recommended_bid")),
            recommended_bid_predicted_win_rate=_f(
                row.get("recommended_bid_predicted_win_rate")
            ),
            recommended_bid_predicted_profit=_f(
                row.get("recommended_bid_predicted_profit")
            ),
            decision_path=row.get("decision_path"),
            decision_reason=row.get("decision_reason"),
            expected_revenue=_f(row.get("expected_revenue")),
            min_bid=_f(row.get("min_bid")),
            max_bid=cbg.get("max_bid"),
            bid_step=_f(row.get("bid_step")),
            prediction_id=row.get("prediction_id"),
            lead_ping_id=row.get("lead_ping_id"),
        )

    @app.post("/explain_bid")
    def explain_bid_route(request: ExplainRequest):
        """Explain an already-computed prediction, by id (production mode).

        Loads the persisted prediction and runs SHAP (plus optional LLM
        narrative / nearby-bid curve) on the exact feature row it scored --
        **never re-running the bid decision** -- then persists the explanation
        back onto that same prediction-log row. So the explanation always
        corresponds to the logged bid and to the model version that served it,
        and the optimizer sweep isn't duplicated here.

        Inputs
        ------
        request : ExplainRequest
            The ``prediction_id`` to explain (+ ``with_llm`` / ``with_bid_curve``).

        Returns
        -------
        dict
            ``top_factors``, ``base_win_rate``, ``bid_curve``, ``explanation``,
            plus ``prediction_id`` and the row's ``lead_ping_id``.
        """
        from smarthub.server import explain as server_explain

        store = _prediction_log_store()
        row = store.get(request.prediction_id)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction logged for id '{request.prediction_id}'.",
            )

        prediction = _prediction_output_from_log(row)

        # Load the *exact* model version that served this prediction, so SHAP
        # reflects it even if a newer model has since been promoted. `None`
        # (cold start, or an unloadable artifact) yields the policy's own
        # explanation rather than an error.
        model = None
        model_uri = row.get("model_uri")
        if model_uri:
            try:
                model = load_model(model_uri=model_uri)
            except Exception:  # noqa: BLE001 -- artifact unloadable -> canned
                logger.warning(
                    "Could not load model '%s' to explain prediction %s; "
                    "returning the policy explanation only.",
                    model_uri,
                    request.prediction_id,
                    exc_info=True,
                )

        try:
            result = server_explain.explain_from_prediction(
                model,
                prediction,
                with_llm=request.with_llm,
                with_bid_curve=request.with_bid_curve,
            )
        except Exception as exc:  # noqa: BLE001 -- explanation must never 500
            # e.g. SHAP only supports lightgbm; degrade to a clear message
            # rather than failing the request.
            logger.warning(
                "Explanation failed for prediction %s: %s",
                request.prediction_id,
                exc,
                exc_info=True,
            )
            result = {
                "base_prediction": None,
                "prediction": None,
                "feature_contributions": [],
                "top_factors": [],
                "base_win_rate": None,
                "explanation": f"Explanation unavailable: {exc}",
            }

        # SHAP values / feature values can be NaN or ±inf; Starlette's JSON
        # encoder rejects those, so sanitize before persisting and returning.
        result = _json_safe(result)

        # Persist the explanation back onto the same row (no new log row).
        try:
            store.update_shap_explanation(
                request.prediction_id,
                {
                    "base_prediction": result.get("base_prediction"),
                    "prediction": result.get("prediction"),
                    "feature_contributions": result.get("feature_contributions"),
                    "top_factors": result.get("top_factors"),
                    "base_win_rate": result.get("base_win_rate"),
                    "bid_curve": result.get("bid_curve"),
                    "explanation": result.get("explanation"),
                },
            )
        except Exception:  # noqa: BLE001 -- logging must never break serving
            logger.warning(
                "Failed to persist explanation for prediction %s",
                request.prediction_id,
                exc_info=True,
            )

        return {
            **result,
            "prediction_id": request.prediction_id,
            "lead_ping_id": row.get("lead_ping_id"),
        }
