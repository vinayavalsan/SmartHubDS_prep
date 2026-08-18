"""Versioned model registry and promotion controls for SmartHub.

Training runs and production model versions are separate concepts:
- every saved candidate receives a unique ``training_run_id``;
- only promoted models receive a sequential production version such as
  ``auto_v1``.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from smarthub.core import paths
from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)
MODEL_DIR_ROOT = paths.data_dir() / "models"
_PRODUCTION_VERSION_RE = re.compile(r"^(?P<lead>[a-z0-9_]+)_v(?P<number>\d+)$")


class RegistryError(RuntimeError):
    """Raised when a promotion/registry invariant cannot be satisfied — e.g.
    production storage is configured but unavailable, or a production version
    cannot be assigned."""


# --- production storage (promoted models) ------------------------------------
# Local training always writes to the filesystem under MODEL_DIR_ROOT. When
# production storage is configured, `promote()` also publishes the promoted
# artifact + manifest + serving pointer there, and serving prefers it. Without
# production storage, the registry operates entirely on the local filesystem.
_PRODUCTION_STORE = None
_PRODUCTION_STORE_RESOLVED = False


def _local_store():
    """Filesystem store rooted at the local model dir (always available)."""
    from smarthub.train_and_predict.model_storage import FilesystemModelStore

    return FilesystemModelStore(str(MODEL_DIR_ROOT))


def _production_store():
    """Return the production model store, or None when production is disabled.

    Production storage resolution distinguishes two cases:

    * **Disabled** (no backend configured) -> returns ``None``. Local-only dev;
      falling back to local storage is legitimate.
    * **Configured but unbuildable** (e.g. bad creds / unreachable endpoint) ->
      **raises** :class:`RegistryError`. Once production storage is the source
      of truth, a broken store must be visible, never silently downgraded to a
      possibly-stale local artifact.

    The result is cached only on success (or when disabled); a failure is not
    cached so a transient issue can recover on the next call.
    """
    global _PRODUCTION_STORE, _PRODUCTION_STORE_RESOLVED
    if _PRODUCTION_STORE_RESOLVED:
        return _PRODUCTION_STORE

    from . import config

    cfg = config.load_training_config()
    if cfg.production_storage is None:
        _PRODUCTION_STORE = None
        _PRODUCTION_STORE_RESOLVED = True
        return None
    try:
        store = cfg.production_model_store()
    except Exception as exc:  # noqa: BLE001
        raise RegistryError(
            "Production storage is configured but could not be initialised; "
            "refusing to silently fall back to local storage."
        ) from exc
    _PRODUCTION_STORE = store
    _PRODUCTION_STORE_RESOLVED = True
    return store


def reset_production_store_cache() -> None:
    """Force re-resolution of the production store (tests / config reloads)."""
    global _PRODUCTION_STORE, _PRODUCTION_STORE_RESOLVED
    _PRODUCTION_STORE = None
    _PRODUCTION_STORE_RESOLVED = False


def model_dir(lead_type_name: str) -> Path:
    return MODEL_DIR_ROOT / lead_type_name.strip().lower()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _lead_type_slug(lead_type_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", lead_type_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("lead_type_name must contain at least one letter or number.")
    return slug


def _new_training_run_id() -> str:
    return f"run_{_timestamp()}_{uuid.uuid4().hex[:8]}"


def version_path(lead_type_name: str, version: str) -> Path:
    """Return the artifact path for a training run identifier."""
    return model_dir(lead_type_name) / f"{version}.pkl"


def manifest_path(lead_type_name: str, version: str) -> Path:
    """Return the manifest path for a training run identifier."""
    return model_dir(lead_type_name) / f"{version}.json"


def load_manifest(lead_type_name: str, version: str) -> dict:
    path = manifest_path(lead_type_name, version)
    if path.exists():
        return json.loads(path.read_text())
    # Production-only model (promoted elsewhere, not trained on this box):
    # fall back to the manifest published in production storage.
    store = _production_store()
    if store is not None:
        key = f"{lead_type_name.strip().lower()}/{version}.json"
        if store.exists(key):
            return json.loads(store.read_text(key))
    raise FileNotFoundError(
        f"No manifest for lead type '{lead_type_name}' training run '{version}'."
    )


def update_manifest(lead_type_name: str, version: str, **updates) -> dict:
    """Update mutable lifecycle metadata in a saved training-run manifest."""
    manifest = load_manifest(lead_type_name, version)
    manifest.update(updates)
    manifest_path(lead_type_name, version).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    return manifest


def list_versions(lead_type_name: str) -> list[str]:
    """List saved training-run identifiers from oldest to newest."""
    folder = model_dir(lead_type_name)
    if not folder.exists():
        return []
    manifests = []
    for path in folder.glob("run_*.json"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        manifests.append((manifest.get("created_at", ""), path.stem))
    return [run_id for _, run_id in sorted(manifests)]


def list_production_versions(lead_type_name: str) -> list[str]:
    """List assigned production versions in numeric order."""
    versions = []
    folder = model_dir(lead_type_name)
    if not folder.exists():
        return versions
    for path in folder.glob("run_*.json"):
        try:
            value = json.loads(path.read_text()).get("production_model_version")
        except (OSError, json.JSONDecodeError):
            continue
        if value:
            versions.append(value)
    return sorted(set(versions), key=_production_version_number)


def _production_version_number(version: str) -> int:
    match = _PRODUCTION_VERSION_RE.match(version)
    return int(match.group("number")) if match else 0


def _assigned_version_numbers(lead_type_name: str) -> set[int]:
    """Every production version number already assigned, gathered from BOTH the
    local manifests and (when configured) production storage — so a version
    assigned on another box or a prior promotion can never be reused."""
    numbers: set[int] = set()
    folder = model_dir(lead_type_name)
    if folder.exists():
        for path in folder.glob("run_*.json"):
            try:
                value = json.loads(path.read_text()).get("production_model_version")
            except (OSError, json.JSONDecodeError):
                continue
            if value:
                numbers.add(_production_version_number(value))

    store = _production_store()  # raises if configured-but-broken (visible)
    if store is not None:
        lead = lead_type_name.strip().lower()
        # Reserved version markers are authoritative for assigned versions.
        for key in store.list(f"{lead}/versions/"):
            match = _PRODUCTION_VERSION_RE.match(Path(key).stem)
            if match:
                numbers.add(int(match.group("number")))
        # Promoted manifests also reserve their assigned version numbers.
        for key in store.list(f"{lead}/"):
            name = Path(key).name
            if name.startswith("run_") and name.endswith(".json"):
                try:
                    value = json.loads(store.read_text(key)).get(
                        "production_model_version"
                    )
                except json.JSONDecodeError:
                    continue
                if value:
                    numbers.add(_production_version_number(value))

    numbers.discard(0)
    return numbers


def _next_production_model_version(lead_type_name: str) -> str:
    """Reserve and return the next free ``<lead>_vN`` via an **atomic claim**.

    Starts from ``max(assigned) + 1`` and creates a marker with a create-if-
    absent op; on conflict (another promotion grabbed it, or it predates the
    marker scheme) it bumps and retries. This guarantees two concurrent
    promotions can never be assigned the same production version.
    """
    lead_slug = _lead_type_slug(lead_type_name)
    lead_path = lead_type_name.strip().lower()
    store = _production_store() or _local_store()
    number = max(_assigned_version_numbers(lead_type_name), default=0) + 1
    for _ in range(10_000):
        candidate = f"{lead_slug}_v{number}"
        if store.claim(f"{lead_path}/versions/{candidate}.json"):
            return candidate
        number += 1
    raise RegistryError(
        f"Could not assign a production version for '{lead_type_name}'."
    )


def save_version(
    model,
    lead_type_name: str,
    *,
    feature_cols: list[str],
    metrics: dict,
    optimizer_summary: dict | None,
    lineage: dict,
    model_params: dict,
    training_config: dict,
    promotion_mode: str,
    eligibility_status: str,
    promotion_status: str,
    promotion_decision_reason: str,
    promotion_comparison: dict | None = None,
) -> dict:
    """Save one immutable candidate artifact identified by ``training_run_id``.

    The manifest stores the resolved training configuration so the model can
    be audited and reproduced without depending on the current YAML file.
    """
    import joblib

    allowed_eligibility = {"eligible", "not_eligible", "not_evaluated"}
    allowed_promotion = {
        "promoted",
        "rejected",
        "awaiting_manual_promotion",
        "not_evaluated",
    }
    if eligibility_status not in allowed_eligibility:
        raise ValueError(f"Unsupported eligibility_status: {eligibility_status!r}")
    if promotion_status not in allowed_promotion:
        raise ValueError(f"Unsupported promotion_status: {promotion_status!r}")

    folder = model_dir(lead_type_name)
    folder.mkdir(parents=True, exist_ok=True)
    training_run_id = _new_training_run_id()
    model_file = version_path(lead_type_name, training_run_id)
    joblib.dump(model, model_file)

    manifest = {
        "training_run_id": training_run_id,
        "version": training_run_id,
        "production_model_version": None,
        "lead_type_name": lead_type_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_file),
        "feature_cols": list(feature_cols),
        "metrics": metrics,
        "optimizer_summary": optimizer_summary or {},
        "lineage": lineage,
        "model_params": model_params,
        "training_config": training_config,
        "promotion_mode": promotion_mode,
        "eligibility_status": eligibility_status,
        "promotion_status": promotion_status,
        "promotion_decision_reason": promotion_decision_reason,
        "promotion_comparison": promotion_comparison or {},
        "promotion_eligible": (
            True
            if eligibility_status == "eligible"
            else False if eligibility_status == "not_eligible" else None
        ),
        "promoted": False,
        "promoted_at": None,
        "mlflow_run_id": None,
        "mlflow_experiment_id": None,
        "mlflow_model_uri": None,
        "mlflow_registered_model_name": None,
        "mlflow_registered_model_version": None,
    }
    manifest_path(lead_type_name, training_run_id).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    return manifest


def _serving_pointer_path(lead_type_name: str) -> Path:
    return model_dir(lead_type_name) / "current.json"


def _read_serving_pointer(lead_type_name: str) -> dict | None:
    """Return the serving pointer, preferring production storage.

    When production storage is configured its ``current.json`` is the source of
    truth for what serves; otherwise the local pointer is used. So the same
    serving-read helpers work in both local-only and production deployments.
    """
    # When production storage is configured it is the source of truth: use its
    # pointer (or None = cold start). Do NOT fall back to a possibly-stale local
    # pointer. Only local-only deployments read the local pointer.
    if _production_store() is not None:  # raises if configured-but-broken
        return production_serving_pointer(lead_type_name)
    path = _serving_pointer_path(lead_type_name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def serving_model_path(lead_type_name: str, version: str) -> Path | None:
    """Local filesystem path to a version's artifact, preferring production.

    For an S3 production store this downloads+caches the object and returns the
    cache path (usable by ``joblib.load``). Falls back to the local artifact,
    or None when neither exists.
    """
    store = _production_store()  # raises if configured-but-broken (visible)
    if store is not None:
        key = f"{lead_type_name.strip().lower()}/{version}.pkl"
        # Let production errors (download/head failures) propagate — never mask
        # them by serving a possibly-stale local copy.
        if store.exists(key):
            return store.local_path(key)
        # Production is authoritative: if the promoted artifact isn't there, do
        # not serve a local copy.
        return None
    local = version_path(lead_type_name, version)
    return local if local.exists() else None


def currently_serving_version(lead_type_name: str) -> str | None:
    """Return the serving training-run identifier (legacy-compatible name)."""
    pointer = _read_serving_pointer(lead_type_name)
    if pointer is None:
        return None
    return pointer.get("training_run_id") or pointer.get("version")


def currently_serving_production_version(lead_type_name: str) -> str | None:
    pointer = _read_serving_pointer(lead_type_name)
    return pointer.get("production_model_version") if pointer else None


def currently_serving_manifest(lead_type_name: str) -> dict | None:
    version = currently_serving_version(lead_type_name)
    if version is None:
        return None
    try:
        return load_manifest(lead_type_name, version)
    except FileNotFoundError:
        return None


def currently_serving_model_path(lead_type_name: str) -> Path | None:
    version = currently_serving_version(lead_type_name)
    return serving_model_path(lead_type_name, version) if version else None


def load_currently_serving_model(lead_type_name: str):
    version = currently_serving_version(lead_type_name)
    if version is None:
        return None, None
    try:
        manifest = load_manifest(lead_type_name, version)
    except FileNotFoundError:
        return None, None
    model_file = serving_model_path(lead_type_name, version)
    if model_file is None:
        return None, None
    import joblib

    try:
        return joblib.load(model_file), manifest
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "Currently-serving model %s for %r failed to load; " "skipping comparison.",
            version,
            lead_type_name,
            exc_info=True,
        )
        return None, None


def promote(lead_type_name: str, version: str, reason: str = "") -> dict:
    """Promote an eligible training run and assign a production version once."""
    if not manifest_path(lead_type_name, version).exists():
        raise FileNotFoundError(
            f"Cannot promote unknown training run '{version}' for '{lead_type_name}'."
        )

    manifest = load_manifest(lead_type_name, version)
    if manifest.get("eligibility_status") != "eligible":
        raise ValueError(
            f"Training run '{version}' is not eligible for promotion "
            f"(status={manifest.get('eligibility_status')!r})."
        )

    previous_run_id = currently_serving_version(lead_type_name)
    previous_production_version = currently_serving_production_version(lead_type_name)
    promoted_at = datetime.now(timezone.utc).isoformat()
    production_version = manifest.get("production_model_version")
    if not production_version:
        production_version = _next_production_model_version(lead_type_name)

    promoted_fields = {
        "production_model_version": production_version,
        "promotion_status": "promoted",
        "promoted": True,
        "promoted_at": promoted_at,
        "promotion_reason": reason,
    }
    promoted_manifest = {**manifest, **promoted_fields}
    pointer = {
        "version": version,
        "training_run_id": version,
        "production_model_version": production_version,
        "promoted_at": promoted_at,
        "previous_training_run_id": previous_run_id,
        "previous_production_model_version": previous_production_version,
        "reason": reason,
    }

    # 1) Publish to production storage FIRST — artifact + manifest, then the
    #    pointer (the commit). Any failure raises here, BEFORE we mark the run
    #    promoted or switch the local pointer, so the previously-serving model
    #    stays authoritative and nothing claims this model is serving.
    store = _production_store()  # raises if configured-but-broken (visible)
    if store is not None:
        _publish_to_production(
            store, lead_type_name, version, promoted_manifest, pointer
        )

    # 2) Only after a successful publish: mark the local manifest promoted and
    #    mirror the serving pointer locally. In local-only mode (no production
    #    store), the local pointer write below is itself the commit.
    update_manifest(lead_type_name, version, **promoted_fields)
    _serving_pointer_path(lead_type_name).write_text(
        json.dumps(pointer, indent=2, ensure_ascii=False, default=str)
    )

    # 3) Best-effort audit registration; a failure here never un-promotes.
    _publish_production_mlflow(lead_type_name, version, pointer)
    return pointer


def _publish_to_production(
    store, lead_type_name: str, version: str, promoted_manifest: dict, pointer: dict
) -> None:
    """Upload the promoted artifact + manifest, then flip the production serving
    pointer (the commit). **Raises on any failure** — a promotion that can't
    fully publish to production is not a completed promotion.

    Order matters: the ``.pkl`` and ``.json`` are written before ``current.json``
    so serving never flips to a version whose artifact isn't uploaded yet.
    """
    lead = lead_type_name.strip().lower()
    store.write_bytes(
        f"{lead}/{version}.pkl",
        version_path(lead_type_name, version).read_bytes(),
    )
    store.write_text(
        f"{lead}/{version}.json",
        json.dumps(promoted_manifest, indent=2, ensure_ascii=False, default=str),
    )
    store.write_text(
        f"{lead}/current.json",
        json.dumps(pointer, indent=2, ensure_ascii=False, default=str),
    )
    logger.info(
        "Published %s to production storage (%s).",
        version,
        getattr(store, "backend", "?"),
    )


def _publish_production_mlflow(
    lead_type_name: str, version: str, pointer: dict
) -> None:
    """Register a promoted model in the production MLflow server.

    Best-effort (warns, never raises): production storage is the serving source
    of truth and has already been written by :func:`_publish_to_production`;
    MLflow here is audit/registry, so a production-MLflow outage must not fail a
    promotion. No-op when production MLflow isn't configured (checked before
    importing mlflow, so disabled setups never need it installed).
    """
    try:
        from . import config

        cfg = config.load_training_config()
        if not cfg.production_mlflow_enabled:
            return
        import joblib

        from . import mlflow_utils

        manifest = load_manifest(lead_type_name, version)
        model = joblib.load(version_path(lead_type_name, version))
        metadata = mlflow_utils.log_production_model(
            tracking_uri=cfg.mlflow_production_tracking_uri,
            experiment_name=cfg.mlflow_production_experiment_name,
            registered_model_name=(
                cfg.mlflow_production_registered_model_name
                or cfg.mlflow_registered_model_name
            ),
            run_name=f"{lead_type_name}_{pointer['production_model_version']}",
            model=model,
            model_params=manifest.get("model_params"),
            feature_cols=manifest.get("feature_cols"),
            metrics=manifest.get("metrics"),
            tags={
                "training_run_id": version,
                "lead_type_name": lead_type_name,
                "production_model_version": pointer["production_model_version"],
                "promotion_reason": pointer.get("reason", ""),
            },
        )
        update_manifest(lead_type_name, version, **metadata)
        logger.info(
            "Registered %s in production MLflow (%s v%s).",
            version,
            metadata["production_registered_model_name"],
            metadata["production_registered_model_version"],
        )
    except Exception:  # noqa: BLE001 -- audit registry; must not fail promotion
        logger.warning(
            "Failed to register %s in production MLflow.", version, exc_info=True
        )


def production_serving_pointer(lead_type_name: str) -> dict | None:
    """Return the production serving pointer, or None when unavailable."""
    store = _production_store()
    if store is None:
        return None
    key = f"{lead_type_name.strip().lower()}/current.json"
    if not store.exists(key):
        return None
    return json.loads(store.read_text(key))


def load_serving_model(lead_type_name: str):
    """Load the serving ``(model, manifest)``, preferring production storage.

    Thin alias for :func:`load_currently_serving_model`, which is now
    production-aware (it reads the production pointer/manifest/artifact when
    production storage is configured, and falls back to local otherwise).
    """
    return load_currently_serving_model(lead_type_name)


def rollback(
    lead_type_name: str,
    to_version: str | None = None,
    reason: str = "manual rollback",
) -> dict:
    """Move serving to an earlier already-promoted training run."""
    promoted = []
    for run_id in list_versions(lead_type_name):
        manifest = load_manifest(lead_type_name, run_id)
        if manifest.get("production_model_version"):
            promoted.append(
                (
                    _production_version_number(manifest["production_model_version"]),
                    run_id,
                )
            )
    promoted.sort()
    serving = currently_serving_version(lead_type_name)
    run_ids = [run_id for _, run_id in promoted]
    if to_version is None:
        if serving not in run_ids:
            raise ValueError(
                f"No currently-serving promoted model for '{lead_type_name}'."
            )
        idx = run_ids.index(serving)
        if idx == 0:
            raise ValueError(f"'{serving}' is the earliest promoted model.")
        to_version = run_ids[idx - 1]
    return promote(lead_type_name, to_version, reason=reason)


# --- Promotion policy ---------------------------------------------------------


@dataclass
class PromotionDecision:
    """Store a model-promotion decision and its comparison details."""

    promote: bool
    reason: str
    comparison: dict = field(default_factory=dict)


def decide_promotion(
    challenger_metrics: dict,
    challenger_optimizer: dict | None,
    currently_serving_metrics: dict | None,
    currently_serving_optimizer: dict | None,
    *,
    max_log_loss_regression: float,
    min_profit_ratio: float,
    max_absolute_profit_loss_tolerance: float,
    max_log_loss: float,
    min_expected_profit: float,
) -> PromotionDecision:
    """Compare challenger and serving performance and decide promotion.

    Every challenger must pass absolute log-loss and expected-profit gates.
    When a serving model exists, the challenger must also satisfy the relative
    profit and log-loss requirements.
    """
    challenger_log_loss = challenger_metrics.get("log_loss")
    challenger_profit = (challenger_optimizer or {}).get(
        "recommended_bid_total_expected_profit"
    )
    comparison = {
        "challenger_log_loss": challenger_log_loss,
        "maximum_log_loss": max_log_loss,
        "challenger_profit": challenger_profit,
        "minimum_expected_profit": min_expected_profit,
    }

    if challenger_log_loss is None:
        return PromotionDecision(
            False,
            "Challenger log loss is unavailable.",
            comparison,
        )
    if challenger_log_loss > max_log_loss:
        return PromotionDecision(
            False,
            f"Challenger log loss ({challenger_log_loss:.4f}) exceeds the "
            f"maximum allowed ({max_log_loss:.4f}).",
            comparison,
        )

    if challenger_profit is None:
        return PromotionDecision(
            False,
            "Challenger expected profit is unavailable.",
            comparison,
        )
    if challenger_profit < min_expected_profit:
        return PromotionDecision(
            False,
            f"Challenger expected profit ({challenger_profit:.2f}) is below "
            f"the required minimum ({min_expected_profit:.2f}).",
            comparison,
        )

    if currently_serving_metrics is None:
        return PromotionDecision(
            True,
            f"First model passed all absolute promotion thresholds: log loss "
            f"{challenger_log_loss:.4f} and expected profit "
            f"{challenger_profit:.2f}.",
            comparison,
        )

    serving_log_loss = currently_serving_metrics.get("log_loss")
    serving_profit = (currently_serving_optimizer or {}).get(
        "recommended_bid_total_expected_profit"
    )
    comparison.update(
        {
            "currently_serving_log_loss": serving_log_loss,
            "currently_serving_profit": serving_profit,
            "log_loss_regression": None,
        }
    )

    if serving_log_loss is None:
        return PromotionDecision(
            False,
            "Currently-serving model log loss is unavailable.",
            comparison,
        )
    if serving_profit is None:
        return PromotionDecision(
            False,
            "Currently-serving model expected profit is unavailable.",
            comparison,
        )
    if serving_profit <= 0:
        return PromotionDecision(
            False,
            f"Currently-serving expected profit ({serving_profit:.2f}) must be "
            "greater than zero for a valid relative profit comparison.",
            comparison,
        )

    log_loss_regression = challenger_log_loss - serving_log_loss
    profit_ratio = challenger_profit / serving_profit
    absolute_profit_loss = serving_profit - challenger_profit
    comparison["log_loss_regression"] = log_loss_regression
    comparison["profit_ratio"] = profit_ratio
    comparison["absolute_profit_loss"] = absolute_profit_loss
    comparison["max_absolute_profit_loss_tolerance"] = (
        max_absolute_profit_loss_tolerance
    )

    if profit_ratio < min_profit_ratio:
        return PromotionDecision(
            False,
            f"Challenger expected profit ({challenger_profit:.2f}) is below "
            f"{min_profit_ratio:.0%} of the currently-serving model's "
            f"({serving_profit:.2f}) on the same held-out rows.",
            comparison,
        )

    if absolute_profit_loss > max_absolute_profit_loss_tolerance:
        return PromotionDecision(
            False,
            f"Challenger expected profit is lower by "
            f"{absolute_profit_loss:.2f}, exceeding the allowed "
            "absolute profit-loss tolerance "
            f"of {max_absolute_profit_loss_tolerance:.2f} on the same held-out rows.",
            comparison,
        )

    if log_loss_regression > max_log_loss_regression:
        return PromotionDecision(
            False,
            f"Challenger log loss regressed by {log_loss_regression:.4f} versus "
            f"the currently-serving model (tolerance "
            f"{max_log_loss_regression:.4f}).",
            comparison,
        )

    return PromotionDecision(
        True,
        f"Challenger passed the absolute thresholds and satisfies the "
        f"{min_profit_ratio:.0%} relative profit requirement and the "
        f"{max_absolute_profit_loss_tolerance:.2f} absolute profit-loss "
        f"tolerance; log-loss regression "
        f"{log_loss_regression:.4f} is within tolerance.",
        comparison,
    )


def main(argv: list[str] | None = None) -> int:
    """Run manual model-registry operations from the command line.

    Inputs
    ------
    argv : list[str] | None
        Optional command-line argument sequence.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Manage SmartHub model promotion.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Promote a saved model version to currently serving.",
    )
    promote_parser.add_argument("--lead-type-name", required=True)
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument(
        "--reason",
        default="manual promotion",
        help="Reason recorded in the model manifest and serving pointer.",
    )

    args = parser.parse_args(argv)

    if args.command == "promote":
        from . import config, mlflow_utils

        pointer = promote(
            args.lead_type_name,
            args.version,
            reason=args.reason,
        )
        manifest = load_manifest(args.lead_type_name, args.version)
        training_config = config.load_training_config()
        experiment_name = (
            f"{training_config.mlflow_experiment_name}_{args.lead_type_name}"
        )
        mlflow_metadata = mlflow_utils.promote_training_run(
            training_run_id=args.version,
            lead_type_name=args.lead_type_name,
            production_model_version=pointer["production_model_version"],
            reason=args.reason,
            tracking_db_path=training_config.mlflow_tracking_db_path,
            artifact_root=training_config.mlflow_artifact_root,
            experiment_name=experiment_name,
            registered_model_name=training_config.mlflow_registered_model_name,
            mlflow_run_id=manifest.get("mlflow_run_id"),
        )
        update_manifest(
            args.lead_type_name,
            args.version,
            **mlflow_metadata,
        )
        logger.info(
            "Promoted %s for %s (training_run_id=%s, previous=%s, "
            "mlflow_model=%s v%s).",
            pointer["production_model_version"],
            args.lead_type_name,
            pointer["training_run_id"],
            pointer.get("previous_production_model_version"),
            mlflow_metadata["mlflow_registered_model_name"],
            mlflow_metadata["mlflow_registered_model_version"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
