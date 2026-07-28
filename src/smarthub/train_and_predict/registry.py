"""Versioned model registry and promotion controls for SmartHub.

Training runs and production model versions are separate concepts:
- every saved candidate receives a unique ``training_run_id``;
- only promoted models receive a sequential production version such as
  ``auto_v1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from smarthub.core import paths

MODEL_DIR_ROOT = paths.data_dir() / "models"
_PRODUCTION_VERSION_RE = re.compile(r"^(?P<lead>[a-z0-9_]+)_v(?P<number>\d+)$")


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
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for lead type '{lead_type_name}' training run '{version}'."
        )
    return json.loads(path.read_text())


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


def _next_production_model_version(lead_type_name: str) -> str:
    existing = [
        _production_version_number(v) for v in list_production_versions(lead_type_name)
    ]
    number = max(existing, default=0) + 1
    return f"{_lead_type_slug(lead_type_name)}_v{number}"


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
        "version": training_run_id,  # compatibility for existing serving code
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


def currently_serving_version(lead_type_name: str) -> str | None:
    """Return the serving training-run identifier (legacy-compatible name)."""
    pointer = _serving_pointer_path(lead_type_name)
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text())
    return payload.get("training_run_id") or payload.get("version")


def currently_serving_production_version(lead_type_name: str) -> str | None:
    pointer = _serving_pointer_path(lead_type_name)
    if not pointer.exists():
        return None
    return json.loads(pointer.read_text()).get("production_model_version")


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
    return version_path(lead_type_name, version) if version else None


def load_currently_serving_model(lead_type_name: str):
    version = currently_serving_version(lead_type_name)
    if version is None:
        return None, None
    try:
        manifest = load_manifest(lead_type_name, version)
    except FileNotFoundError:
        return None, None
    model_file = version_path(lead_type_name, version)
    if not model_file.exists():
        return None, None
    import joblib

    return joblib.load(model_file), manifest


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

    manifest.update(
        {
            "production_model_version": production_version,
            "promotion_status": "promoted",
            "promoted": True,
            "promoted_at": promoted_at,
            "promotion_reason": reason,
        }
    )
    update_manifest(
        lead_type_name,
        version,
        production_model_version=production_version,
        promotion_status="promoted",
        promoted=True,
        promoted_at=promoted_at,
        promotion_reason=reason,
    )

    pointer = {
        "version": version,
        "training_run_id": version,
        "production_model_version": production_version,
        "promoted_at": promoted_at,
        "previous_training_run_id": previous_run_id,
        "previous_production_model_version": previous_production_version,
        "reason": reason,
    }
    _serving_pointer_path(lead_type_name).write_text(
        json.dumps(pointer, indent=2, ensure_ascii=False, default=str)
    )
    return pointer


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
    target_cm: float,
    max_log_loss: float,
    min_expected_profit: float,
) -> PromotionDecision:
    """Compare challenger and serving performance and decide promotion.

    Every challenger must pass absolute log-loss, expected-profit, and
    contribution-margin gates. When a serving model exists, the challenger
    must also satisfy the relative profit and log-loss requirements.
    """
    challenger_log_loss = challenger_metrics.get("log_loss")
    challenger_profit = (challenger_optimizer or {}).get(
        "recommended_bid_total_expected_profit"
    )
    challenger_cm = (challenger_optimizer or {}).get("avg_recommended_bid_cm_if_won")

    comparison = {
        "challenger_log_loss": challenger_log_loss,
        "maximum_log_loss": max_log_loss,
        "challenger_profit": challenger_profit,
        "minimum_expected_profit": min_expected_profit,
        "challenger_recommended_cm": challenger_cm,
        "target_cm": target_cm,
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

    if challenger_cm is None:
        return PromotionDecision(
            False,
            "Challenger recommended contribution margin is unavailable.",
            comparison,
        )
    if challenger_cm < target_cm:
        return PromotionDecision(
            False,
            f"Challenger recommended CM ({challenger_cm:.2%}) is below the "
            f"optimizer target CM ({target_cm:.2%}).",
            comparison,
        )

    if currently_serving_metrics is None:
        return PromotionDecision(
            True,
            f"First model passed all absolute promotion thresholds: log loss "
            f"{challenger_log_loss:.4f}, expected profit "
            f"{challenger_profit:.2f}, and recommended CM "
            f"{challenger_cm:.2%}.",
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
        f"tolerance; recommended CM "
        f"is {challenger_cm:.2%}, and log-loss regression "
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

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
        logging.getLogger(__name__).info(
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
