"""Versioned model registry + promotion gate for Anton.

Problem this solves: ``train.py`` used to ``joblib.dump`` straight over
``data/models/anton_model_<type>.pkl`` on every run. A bad training run (a
data glitch, a schema regression) would silently overwrite a good model in
production, with no comparison, no history, and no way back.

Now every training run is saved as an **immutable, numbered, timestamped
version** — ``data/models/<lead_type>/v<N>_<UTC timestamp>.pkl`` — with a
JSON manifest of its metrics/lineage next to it
(``v<N>_<timestamp>.json``). A separate pointer file, ``current.json``,
records which version is the **currently-serving model**: the one
``predict.load_model`` actually uses to answer requests. Training a new
("challenger") model never moves that pointer by itself —
``decide_promotion`` compares the challenger against the currently-serving
model **on the same held-out test set** first (see ``train.run_training``),
and only ``promote()`` moves the pointer when the challenger is at least as
good. ``rollback()`` repoints ``current.json`` at an earlier version with no
retraining needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from smarthub.core import paths

logger = logging.getLogger("smarthub.train_and_predict.registry")

# Redirectable in tests the same way smarthub.core.io.TRAINING_DIR is.
MODEL_DIR_ROOT = paths.data_dir() / "models"

_VERSION_RE = re.compile(r"^v(\d+)_")


def model_dir(lead_type_name: str) -> Path:
    """Per-lead-type model folder: data/models/<name>/."""
    return MODEL_DIR_ROOT / lead_type_name.strip().lower()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Writes to a temp file in the same directory, then ``os.replace`` — so a
    process killed or crashing mid-write can never leave a truncated/empty
    file behind. A plain ``.write_text()`` on ``current.json`` once did
    exactly that, and the resulting empty file crashed every subsequent
    ``currently_serving_version()`` call (and, transitively, an entire
    training run) with a raw ``JSONDecodeError`` instead of degrading
    gracefully.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _timestamp() -> str:
    """UTC, filesystem-safe, lexicographically sortable timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _version_number(version: str) -> int:
    match = _VERSION_RE.match(version)
    return int(match.group(1)) if match else 0


def list_versions(lead_type_name: str) -> list[str]:
    """All saved version ids for a lead type, oldest -> newest."""
    folder = model_dir(lead_type_name)
    if not folder.exists():
        return []
    versions = [
        p.stem
        for p in folder.glob("v*.json")
        if _VERSION_RE.match(p.stem)
    ]
    return sorted(versions, key=_version_number)


def _next_version_number(lead_type_name: str) -> int:
    existing = [_version_number(v) for v in list_versions(lead_type_name)]
    return (max(existing) + 1) if existing else 1


def version_path(lead_type_name: str, version: str) -> Path:
    """Path to a version's ``.pkl`` model file."""
    return model_dir(lead_type_name) / f"{version}.pkl"


def manifest_path(lead_type_name: str, version: str) -> Path:
    """Path to a version's ``.json`` manifest file."""
    return model_dir(lead_type_name) / f"{version}.json"


def load_manifest(lead_type_name: str, version: str) -> dict:
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
) -> dict:
    """Save a new immutable model version. Does **not** promote it.

    Returns the manifest dict (includes ``version``, ``model_path``, and
    everything passed in, so callers can log/compare/promote from it).
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
        "promoted": False,
        "promoted_at": None,
    }
    _atomic_write_text(
        manifest_path(lead_type_name, version),
        json.dumps(manifest, indent=2, default=str),
    )
    return manifest


def _serving_pointer_path(lead_type_name: str) -> Path:
    return model_dir(lead_type_name) / "current.json"


def currently_serving_version(lead_type_name: str) -> str | None:
    """The version id currently promoted to serve traffic, or ``None``.

    Also ``None`` (with a logged warning, not a raise) if the pointer file
    exists but can't be parsed — e.g. left empty/truncated by a process
    killed mid-write (see ``_atomic_write_text``, added after exactly this
    took down an entire training run: a raw ``JSONDecodeError`` propagating
    out of here, through ``load_currently_serving_model``, uncaught). A
    corrupt pointer is treated the same as "nothing promoted yet" rather
    than crashing every caller.
    """
    pointer = _serving_pointer_path(lead_type_name)
    if not pointer.exists():
        return None
    try:
        return json.loads(pointer.read_text()).get("version")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Corrupt/unreadable serving pointer for '%s' (%s): %s — "
            "treating as nothing currently serving.",
            lead_type_name, pointer, exc,
        )
        return None


def currently_serving_manifest(lead_type_name: str) -> dict | None:
    version = currently_serving_version(lead_type_name)
    if version is None:
        return None
    try:
        return load_manifest(lead_type_name, version)
    except FileNotFoundError:
        return None


def currently_serving_model_path(lead_type_name: str) -> Path | None:
    """Path to the currently-promoted model file, or ``None`` if none yet."""
    version = currently_serving_version(lead_type_name)
    return version_path(lead_type_name, version) if version else None


def load_currently_serving_model(lead_type_name: str):
    """Load the currently-serving model + its manifest.

    ``(None, None)`` if nothing has been promoted for this lead type yet, or if
    the model file can't be found in this environment.

    The model file is resolved from ``version_path`` (i.e. the *current*
    ``data/models`` location), NOT the ``model_path`` recorded in the manifest —
    that stored path is absolute and environment-specific (e.g. ``/app/...`` from
    a Docker training run), so trusting it breaks a later local/other-host load.
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
    """Point the serving pointer at ``version``. Returns the pointer dict."""
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
    _atomic_write_text(
        manifest_path(lead_type_name, version),
        json.dumps(manifest, indent=2, default=str),
    )

    pointer = {
        "version": version,
        "promoted_at": promoted_at,
        "previous_version": previous,
        "reason": reason,
    }
    _atomic_write_text(
        _serving_pointer_path(lead_type_name),
        json.dumps(pointer, indent=2, default=str)
    )
    return pointer


def rollback(
    lead_type_name: str,
    to_version: str | None = None,
    reason: str = "manual rollback",
) -> dict:
    """Repoint the serving pointer at an earlier version — no retraining needed.

    Defaults to the version immediately before the one currently serving.
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
    """Decide whether the challenger should replace the currently-serving model.

    Both models must already have been scored by the caller **on the same
    held-out test set** — that's what makes this a fair comparison instead of
    comparing two numbers computed on two different data snapshots.

    Policy:
    - Nothing currently serving -> always promote (bootstrap case).
    - ROC AUC must not regress by more than ``min_roc_auc_regression``.
    - If both sides have an optimizer evaluation, the challenger's total
      expected profit on that test set must be at least ``min_profit_ratio``
      of the currently-serving model's (default: allow at most a 2% dip) —
      profit is the metric the business cares about; ROC AUC is a guardrail
      against a model that is "profitable" on this slice by being badly
      miscalibrated.
    - If an optimizer comparison isn't available on both sides, fall back to
      the ROC AUC check alone.
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
