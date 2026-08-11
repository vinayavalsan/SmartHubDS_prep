"""Build temporary model-comparison artifacts for MLflow persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)


def build_test_set_id(
    test_df: pd.DataFrame,
    *,
    training_table_version: str,
    split_settings: dict[str, Any],
) -> str:
    """Build a deterministic identifier for the exact held-out dataset."""
    row_hashes = pd.util.hash_pandas_object(
        test_df,
        index=True,
        categorize=True,
    ).to_numpy()
    digest = hashlib.sha256()
    digest.update(str(training_table_version).encode("utf-8"))
    digest.update(
        json.dumps(split_settings, sort_keys=True, default=str).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def save_comparison_artifacts(
    *,
    output_dir: str | Path,
    evaluation_df: pd.DataFrame,
    optimizer_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write the complete comparison artifact set to a temporary directory."""
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = artifact_dir / "evaluation_dataset.parquet"
    optimizer_path = artifact_dir / "optimizer_results.parquet"
    metadata_path = artifact_dir / "evaluation_metadata.json"

    evaluation_df.to_parquet(evaluation_path, index=False)
    optimizer_df.to_parquet(optimizer_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    logger.info("Prepared Model Comparison Artifacts")
    logger.info("  Evaluation dataset                    : %s", evaluation_path)
    logger.info("  Optimizer results                     : %s", optimizer_path)
    logger.info("  Metadata                              : %s", metadata_path)

    return {
        "artifact_dir": str(artifact_dir),
        "evaluation_dataset": str(evaluation_path),
        "optimizer_results": str(optimizer_path),
        "metadata": str(metadata_path),
    }
