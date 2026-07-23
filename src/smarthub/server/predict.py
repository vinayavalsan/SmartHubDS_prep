"""Online model loading and bid prediction for SmartHub.

This module resolves serving models and exposes bid recommendation endpoints.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from smarthub.feature_engineering import features as fe
from smarthub.train_and_predict import config, optimizer, preprocessing, registry

logger = logging.getLogger(__name__)


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


def is_model_cached(lead_type_id: int = 6) -> bool:
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


def load_model(model_uri: str | None = None, lead_type_id: int = 6):
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


def load_model_and_manifest(lead_type_id: int = 6):
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

    lead_type_name = config.lead_type_name(lead_type_id)

    pinned_version = config.active_model_version()
    version = pinned_version or registry.currently_serving_version(lead_type_name)
    if version is None:
        return None, None

    manifest = _cached_manifest(lead_type_name, version)
    if manifest is None:
        return None, None
    model_path = registry.version_path(lead_type_name, version)
    if not model_path.exists():
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
    for lead_type_id in fe.LEAD_TYPES:
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
        `recommended_bid_expected_profit`, `max_bid`, `n_candidate_bids`,
        `decision_path`, `decision_reason`, `model_data_age_days`.
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
            "recommended_bid_expected_profit": None,
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
        row, model, expected_revenue, target_cm, min_bid, bid_step
    )

    explore, direction = exploration_slot(created_dayofweek, created_hour)
    if explore and not pd.isna(result["recommended_bid"]):
        pct = config.exploration_variance_pct()
        perturbed_raw = result["recommended_bid"] * (1 + direction * pct)
        perturbed_bid = _snap_to_grid(perturbed_raw, min_bid, max_bid, bid_step)
        win_rate, profit = _score_bid(row, model, perturbed_bid, expected_revenue)
        result["recommended_bid"] = perturbed_bid
        result["recommended_bid_predicted_win_rate"] = win_rate
        result["recommended_bid_expected_profit"] = profit
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
    from fastapi import BackgroundTasks, FastAPI
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
        from smarthub.train_and_predict import explain

        explain.ensure_model_pulled_async()
        yield

    app = FastAPI(title="Anton Bid Prediction API", lifespan=_lifespan)

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

    def _log_prediction_safe(**kwargs) -> str | None:
        """Best-effort prediction logging -- never raises into the caller.

        Returns
        -------
        str | None
            The logged row's `prediction_id` (so the route can hand it back
            to the caller as a receipt -- see docs/PREDICTION_LOG_SCHEMA.md
            §8), or `None` if the write itself failed.
        """
        try:
            return _prediction_log_store().log_prediction(**kwargs)
        except Exception:  # noqa: BLE001 -- logging must never break serving
            logger.warning("Failed to write prediction log row", exc_info=True)
            return None

    def _log_shap_background(
        prediction_id: str | None,
        model,
        record: dict,
        lead_type_id: int,
        recommended_bid: float,
    ) -> None:
        """Best-effort: compute SHAP factors for a /recommend_bid prediction
        and attach them to its already-written log row.

        Runs as a `BackgroundTasks` job -- scheduled by the route, but not
        actually executed until *after* the response has been sent (Starlette
        semantics) -- so this never adds latency to the bid decision itself.
        Same principle as `/explain_bid` keeping SHAP off `/recommend_bid`'s
        response path, just applied per-request instead of skipped entirely.

        Deliberately excludes the LLM narrative `explain.explain_bid()` also
        produces (`explanation`) -- that's a much heavier Ollama call, and
        running it for every single bid, even off the response path, risks
        piling up load on Ollama under concurrency. That field stays
        /explain_bid-only.

        Silently gives up (logs a warning) on any failure -- e.g. a
        non-LightGBM model (`explain.explain_row` only supports
        `model_type='lightgbm'`, see `_fitted_lgbm_estimators`), or the
        logging DB being unreachable for the update -- since this is pure
        enrichment after the fact and must never surface anywhere a caller
        would see it.
        """
        if prediction_id is None or model is None:
            return
        try:
            from smarthub.train_and_predict import explain

            explained_record = dict(record)
            explained_record["bid"] = recommended_bid
            factors = explain.explain_row(model, explained_record, lead_type_id)
            _prediction_log_store().update_shap_explanation(
                prediction_id,
                {
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
            metrics, `prediction_id` -- a receipt correlating this response
            to its prediction-log row (`None` if the logging write itself
            failed; never blocks or changes the bid decision) -- and
            `lead_ping_id`, echoed back exactly as sent (`None` if the
            caller didn't supply one).

        Notes
        -----
        When a real model served this prediction (not cold start, and a
        viable bid exists), SHAP factors (`top_factors`/`base_win_rate`,
        no LLM narrative) are computed in a background task *after* this
        response is sent and attached to the same `prediction_id`'s log row
        -- never part of this response, never adds latency here. See
        `_log_shap_background`.
        """
        lead_type_name = config.lead_type_name(request.lead_type_id)
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
            )
        except Exception as exc:
            _log_prediction_safe(
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
            raise

        prediction_id = _log_prediction_safe(
            endpoint="recommend_bid",
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
            **_manifest_log_fields(manifest),
            model_data_age_days=result.get("model_data_age_days"),
            decision_path=result.get("decision_path"),
            decision_reason=result.get("decision_reason"),
            recommended_bid=result.get("recommended_bid"),
            recommended_bid_predicted_win_rate=result.get(
                "recommended_bid_predicted_win_rate"
            ),
            recommended_bid_expected_profit=result.get(
                "recommended_bid_expected_profit"
            ),
            serving_config=_serving_config_snapshot(),
        )
        result["prediction_id"] = prediction_id
        result["lead_ping_id"] = request.lead_ping_id

        if model is not None and not pd.isna(result.get("recommended_bid")):
            background_tasks.add_task(
                _log_shap_background,
                prediction_id,
                model,
                record,
                request.lead_type_id,
                result.get("recommended_bid"),
            )

        return result

    @app.post("/explain_bid")
    def explain_bid_route(request: BidRequest):
        """Answer "why did Anton bid $X for this lead?" in plain English.

        Offline/on-demand only -- a separate, slower endpoint;
        `/recommend_bid` doesn't call it -- but runs the identical
        `decide_bid` policy, so the bid and `decision_path` always match
        what live serving would return for the same inputs.

        Inputs
        ------
        request : BidRequest
            Same shape as `/recommend_bid`.

        Returns
        -------
        dict
            Everything from `decide_bid` plus `base_win_rate`,
            `top_factors` (SHAP), `bid_curve`, `explanation` -- see
            `train_and_predict.explain.explain_bid` -- `prediction_id`, a
            receipt correlating this response to its prediction-log row
            (`None` if the logging write itself failed), and `lead_ping_id`,
            echoed back exactly as sent (`None` if the caller didn't supply
            one).
        """
        # Local import: explain.py imports this module (`smarthub.server`),
        # so importing it at module load time here would be circular.
        # Also keeps the heavier explain-only deps (shap) out of every
        # /recommend_bid request.
        from smarthub.train_and_predict import explain

        lead_type_name = config.lead_type_name(request.lead_type_id)
        record = None
        try:
            model, manifest = load_model_and_manifest(request.lead_type_id)
            record, row = _record_and_row(request)
            result = explain.explain_bid(
                model=model,
                record=record,
                lead_type_id=request.lead_type_id,
                expected_revenue=request.expected_revenue,
                manifest=manifest,
                target_cm=request.target_cm,
                min_bid=request.min_bid,
                bid_step=request.bid_step,
                created_dayofweek=request.created_dayofweek,
                created_hour=request.created_hour,
            )
        except Exception as exc:
            _log_prediction_safe(
                endpoint="explain_bid",
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
            raise

        prediction_id = _log_prediction_safe(
            endpoint="explain_bid",
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
            **_manifest_log_fields(manifest),
            model_data_age_days=result.get("model_data_age_days"),
            decision_path=result.get("decision_path"),
            decision_reason=result.get("decision_reason"),
            recommended_bid=result.get("recommended_bid"),
            recommended_bid_predicted_win_rate=result.get(
                "recommended_bid_predicted_win_rate"
            ),
            recommended_bid_expected_profit=result.get(
                "recommended_bid_expected_profit"
            ),
            shap_explanation={
                "top_factors": result.get("top_factors"),
                "base_win_rate": result.get("base_win_rate"),
                "bid_curve": result.get("bid_curve"),
                "explanation": result.get("explanation"),
            },
            serving_config=_serving_config_snapshot(),
        )
        result["prediction_id"] = prediction_id
        result["lead_ping_id"] = request.lead_ping_id
        return result
