"""Versioned model registry and promotion controls for SmartHub.

This module saves model versions, manages serving pointers, and supports rollback.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from smarthub.core import paths

# Redirectable in tests the same way smarthub.core.io.TRAINING_DIR is.
MODEL_DIR_ROOT = paths.data_dir() / "models"

_VERSION_RE = re.compile(r"^v(\d+)_")


def model_dir(lead_type_name: str) -> Path:
    """Return the model directory for a lead type.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    pathlib.Path
        Lead-type model directory.
    """
    return MODEL_DIR_ROOT / lead_type_name.strip().lower()


def _timestamp() -> str:
    """Return a sortable UTC timestamp for model version names."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _version_number(version: str) -> int:
    """Extract the numeric sequence from a model version."""
    match = _VERSION_RE.match(version)
    return int(match.group(1)) if match else 0


def list_versions(lead_type_name: str) -> list[str]:
    """List saved model versions from oldest to newest.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    list[str]
        Model version identifiers ordered oldest to newest.
    """
    folder = model_dir(lead_type_name)
    if not folder.exists():
        return []
    versions = [p.stem for p in folder.glob("v*.json") if _VERSION_RE.match(p.stem)]
    return sorted(versions, key=_version_number)


def _next_version_number(lead_type_name: str) -> int:
    """Return the next model version sequence number."""
    existing = [_version_number(v) for v in list_versions(lead_type_name)]
    return (max(existing) + 1) if existing else 1


def version_path(lead_type_name: str, version: str) -> Path:
    """Return the model artifact path for a version.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    version : str | None
        Optional training-table or model version identifier.

    Returns
    -------
    pathlib.Path
        Model artifact path.
    """
    return model_dir(lead_type_name) / f"{version}.pkl"


def manifest_path(lead_type_name: str, version: str) -> Path:
    """Return the manifest path for a model version.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    version : str | None
        Optional training-table or model version identifier.

    Returns
    -------
    pathlib.Path
        Model manifest path.
    """
    return model_dir(lead_type_name) / f"{version}.json"


def load_manifest(lead_type_name: str, version: str) -> dict:
    """Load a model-version manifest.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    version : str | None
        Optional training-table or model version identifier.

    Returns
    -------
    dict
        Parsed model manifest.

    Raises
    ------
    FileNotFoundError
        If the requested manifest does not exist.
    """
    path = manifest_path(lead_type_name, version)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for lead type '{lead_type_name}' version '{version}'."
        )
    return json.loads(path.read_text())


def save_version(
    model,
    lead_type_name: str,
    *,
    feature_cols: list[str],
    metrics: dict,
    optimizer_summary: dict | None,
    lineage: dict,
    model_params: dict,
    promotion_mode: str,
    promotion_eligible: bool | None,
    promotion_decision_reason: str,
) -> dict:
    """Save a new immutable model version without promoting it.

    Inputs
    ------
    model : Any
        Fitted model or model pipeline.
    lead_type_name : str
        Human-readable lead type name.
    feature_cols : list[str]
        Ordered model feature names.
    metrics : dict
        Model evaluation metrics.
    optimizer_summary : dict | None
        Offline optimizer summary metrics.
    lineage : dict
        Model and data lineage metadata.
    model_params : dict
        Parameters passed to the classifier.
    promotion_mode : str
        Promotion execution mode used for the training run.
    promotion_eligible : bool | None
        Whether the model passed the configured promotion policy, or ``None``
        when promotion evaluation was disabled.
    promotion_decision_reason : str
        Explanation produced by the promotion policy.

    Returns
    -------
    dict
        Manifest for the saved model version.
    """
    import joblib  # lazy: registry.py must stay importable without the `ml` extra

    folder = model_dir(lead_type_name)
    folder.mkdir(parents=True, exist_ok=True)
    number = _next_version_number(lead_type_name)
    version = f"v{number}_{_timestamp()}"

    model_file = version_path(lead_type_name, version)
    joblib.dump(model, model_file)

    manifest = {
        "version": version,
        "version_number": number,
        "lead_type_name": lead_type_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_file),
        "feature_cols": list(feature_cols),
        "metrics": metrics,
        "optimizer_summary": optimizer_summary or {},
        "lineage": lineage,
        "model_params": model_params,
        "promotion_mode": promotion_mode,
        "promotion_eligible": promotion_eligible,
        "promotion_decision_reason": promotion_decision_reason,
        "promoted": False,
        "promoted_at": None,
    }
    manifest_path(lead_type_name, version).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    return manifest


def _serving_pointer_path(lead_type_name: str) -> Path:
    """Return the currently-serving pointer path."""
    return model_dir(lead_type_name) / "current.json"


def currently_serving_version(lead_type_name: str) -> str | None:
    """Return the currently-serving model version.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    str | None
        Serving version identifier, or ``None``.
    """
    pointer = _serving_pointer_path(lead_type_name)
    if not pointer.exists():
        return None
    return json.loads(pointer.read_text()).get("version")


def currently_serving_manifest(lead_type_name: str) -> dict | None:
    """Load the currently-serving model manifest.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    dict | None
        Serving manifest, or ``None``.
    """
    version = currently_serving_version(lead_type_name)
    if version is None:
        return None
    try:
        return load_manifest(lead_type_name, version)
    except FileNotFoundError:
        return None


def currently_serving_model_path(lead_type_name: str) -> Path | None:
    """Return the currently-serving model artifact path.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    pathlib.Path | None
        Serving model path, or ``None``.
    """
    version = currently_serving_version(lead_type_name)
    return version_path(lead_type_name, version) if version else None


def load_currently_serving_model(lead_type_name: str):
    """Load the currently-serving model and manifest.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.

    Returns
    -------
    tuple[Any | None, dict | None]
        Loaded model followed by its manifest, or two ``None`` values.
    """
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

    model = joblib.load(model_file)
    return model, manifest


def promote(lead_type_name: str, version: str, reason: str = "") -> dict:
    """Promote a saved model version to serve traffic.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    version : str | None
        Optional training-table or model version identifier.
    reason : str
        Human-readable reason for the operation.

    Returns
    -------
    dict
        Updated serving-pointer metadata.

    Raises
    ------
    FileNotFoundError
        If the requested model version does not exist.
    """
    if not manifest_path(lead_type_name, version).exists():
        raise FileNotFoundError(
            f"Cannot promote unknown version '{version}' for '{lead_type_name}'."
        )

    previous = currently_serving_version(lead_type_name)
    promoted_at = datetime.now(timezone.utc).isoformat()

    manifest = load_manifest(lead_type_name, version)
    manifest["promoted"] = True
    manifest["promoted_at"] = promoted_at
    manifest["promotion_reason"] = reason
    manifest_path(lead_type_name, version).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )

    pointer = {
        "version": version,
        "promoted_at": promoted_at,
        "previous_version": previous,
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
    """Move the serving pointer to an earlier model version.

    Inputs
    ------
    lead_type_name : str
        Human-readable lead type name.
    to_version : str | None
        Optional target model version for rollback.
    reason : str
        Human-readable reason for the operation.

    Returns
    -------
    dict
        Updated serving-pointer metadata.

    Raises
    ------
    ValueError
        If no valid earlier model version is available.
    """
    versions = list_versions(lead_type_name)
    serving_version = currently_serving_version(lead_type_name)

    if to_version is None:
        if serving_version is None or serving_version not in versions:
            raise ValueError(
                f"No currently-serving model for '{lead_type_name}' to roll "
                "back from."
            )
        idx = versions.index(serving_version)
        if idx == 0:
            raise ValueError(
                f"'{serving_version}' is the earliest version for "
                f"'{lead_type_name}'; nothing to roll back to."
            )
        to_version = versions[idx - 1]

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
    min_roc_auc_regression: float = 0.01,
    min_profit_ratio: float = 0.98,
) -> PromotionDecision:
    """Compare challenger and serving metrics and decide promotion.

    Inputs
    ------
    challenger_metrics : dict
        Challenger model metrics.
    challenger_optimizer : dict | None
        Challenger optimizer metrics.
    currently_serving_metrics : dict | None
        Serving model metrics, when available.
    currently_serving_optimizer : dict | None
        Serving optimizer metrics, when available.
    min_roc_auc_regression : float
        Maximum permitted ROC AUC regression.
    min_profit_ratio : float
        Minimum challenger-to-serving profit ratio.

    Returns
    -------
    PromotionDecision
        Promotion decision and comparison details.
    """
    if currently_serving_metrics is None:
        return PromotionDecision(
            True,
            "Nothing currently serving for this lead type — promoting by default.",
        )

    challenger_auc = challenger_metrics.get("roc_auc", 0.0)
    serving_auc = currently_serving_metrics.get("roc_auc", 0.0)
    auc_drop = serving_auc - challenger_auc
    comparison = {
        "challenger_roc_auc": challenger_auc,
        "currently_serving_roc_auc": serving_auc,
        "roc_auc_drop": auc_drop,
    }
    if auc_drop > min_roc_auc_regression:
        return PromotionDecision(
            False,
            f"Challenger ROC AUC regressed by {auc_drop:.4f} versus the "
            f"currently-serving model (tolerance {min_roc_auc_regression:.4f}).",
            comparison,
        )

    serving_profit = (currently_serving_optimizer or {}).get(
        "recommended_bid_total_expected_profit"
    )
    challenger_profit = (challenger_optimizer or {}).get(
        "recommended_bid_total_expected_profit"
    )
    comparison["challenger_profit"] = challenger_profit
    comparison["currently_serving_profit"] = serving_profit

    if serving_profit is not None and challenger_profit is not None:
        if serving_profit <= 0:
            # The currently-serving model wasn't profitable on this test set;
            # any non-regressing challenger profit is an improvement or a wash.
            ok = challenger_profit >= serving_profit
        else:
            ok = challenger_profit >= serving_profit * min_profit_ratio
        comparison["profit_ratio"] = (
            (challenger_profit / serving_profit) if serving_profit else None
        )
        if not ok:
            return PromotionDecision(
                False,
                f"Challenger expected profit ({challenger_profit:.2f}) is below "
                f"{min_profit_ratio:.0%} of the currently-serving model's "
                f"({serving_profit:.2f}) on the same held-out test set.",
                comparison,
            )
        return PromotionDecision(
            True,
            f"Challenger profit {challenger_profit:.2f} >= "
            f"{min_profit_ratio:.0%} of currently-serving profit "
            f"{serving_profit:.2f} (ROC AUC drop {auc_drop:.4f} within tolerance).",
            comparison,
        )

    return PromotionDecision(
        True,
        f"ROC AUC drop {auc_drop:.4f} within tolerance "
        f"({min_roc_auc_regression:.4f}); no optimizer comparison available "
        "on one or both sides.",
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
        pointer = promote(
            args.lead_type_name,
            args.version,
            reason=args.reason,
        )
        logging.getLogger(__name__).info(
            "Promoted %s for %s (previous=%s).",
            pointer["version"],
            args.lead_type_name,
            pointer.get("previous_version"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
