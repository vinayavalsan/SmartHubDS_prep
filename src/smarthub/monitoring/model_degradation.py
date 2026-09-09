"""Hourly SmartHub ML win-rate degradation monitoring.

The monitor evaluates completed hourly windows for configured ML bidding
strategies. ``bidding_strategy_id`` is always a cohort key; additional cohort
features are configured as a YAML list. For each cohort it compares realized
auction wins with the model's predicted win probabilities, computes relative
win-rate bias and a Bernoulli z-score, applies persistence rules, and sends
Slack notifications only when a cohort enters or escalates degradation.

Run once (Prefect / cron friendly):
    python -m smarthub.monitoring.model_degradation

Dry-run without Slack or state changes:
    python -m smarthub.monitoring.model_degradation --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smarthub.core import io, notifications, paths, task_config
from smarthub.core.logging_utils import get_logger
from smarthub.monitoring import transforms

logger = get_logger(__name__)

_SECTION = "model_degradation"
_STATE_PATH = paths.data_dir() / "monitoring" / "model_degradation_state.json"
_TRUE_TOKENS = {"1", "true", "t", "yes", "y"}
_FALSE_TOKENS = {"0", "false", "f", "no", "n"}

_PREDICTION_MONITORING_PATH = (
    paths.data_dir()
    / "raw_datasets"
    / "monitoring_datasets"
    / "prediction_monitoring.parquet"
)


def _required_config(key: str) -> Any:
    """Return a required YAML value; monitoring policy has no code fallback."""
    missing = object()
    value = task_config.get(_SECTION, key, missing)
    if value is missing:
        raise ValueError(f"Missing required config: {_SECTION}.{key}")
    return value


def _config() -> dict[str, Any]:
    """Load and validate model-degradation task configuration."""
    strategies_raw = _required_config("ml_bidding_strategy_ids")
    strategies = sorted({int(value) for value in (strategies_raw or [])})
    cohort_raw = _required_config("cohort_features")
    cohort_features = [str(value).strip() for value in (cohort_raw or [])]
    cohort_features = [value for value in cohort_features if value]
    if "bidding_strategy_id" in cohort_features:
        raise ValueError(
            "Do not include bidding_strategy_id in model_degradation.cohort_features; "
            "it is always included automatically."
        )
    if len(cohort_features) != len(set(cohort_features)):
        raise ValueError("model_degradation.cohort_features contains duplicates.")

    cfg = {
        "enabled": bool(_required_config("enabled")),
        "ml_bidding_strategy_ids": strategies,
        "cohort_features": cohort_features,
        "window_hours": int(_required_config("window_hours")),
        "persistence_windows": int(_required_config("persistence_windows")),
        "required_bad_windows": int(_required_config("required_bad_windows")),
        "warning_winrate_bias": float(_required_config("warning_winrate_bias")),
        "warning_zscore": float(_required_config("warning_zscore")),
        "critical_winrate_bias": float(_required_config("critical_winrate_bias")),
        "critical_zscore": float(_required_config("critical_zscore")),
    }

    if cfg["window_hours"] != 1:
        raise ValueError("model_degradation.window_hours must currently be 1.")
    if cfg["persistence_windows"] < 1:
        raise ValueError("model_degradation.persistence_windows must be >= 1.")
    if not 1 <= cfg["required_bad_windows"] <= cfg["persistence_windows"]:
        raise ValueError(
            "model_degradation.required_bad_windows must be between 1 and "
            "persistence_windows."
        )
    if cfg["critical_winrate_bias"] > cfg["warning_winrate_bias"]:
        raise ValueError(
            "critical_winrate_bias must be at least as negative as "
            "warning_winrate_bias."
        )
    if cfg["critical_zscore"] > cfg["warning_zscore"]:
        raise ValueError(
            "critical_zscore must be at least as negative as warning_zscore."
        )
    return cfg


def _coerce_outcome(series: pd.Series) -> pd.Series:
    """Coerce known won/lost values to 1/0 while preserving unresolved rows."""

    def _one(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            return numeric if numeric in (0.0, 1.0) else np.nan
        token = str(value).strip().lower()
        if token in _TRUE_TOKENS:
            return 1.0
        if token in _FALSE_TOKENS:
            return 0.0
        return np.nan

    return series.map(_one).astype("float64")


def _load_prediction_monitoring() -> pd.DataFrame:
    """Load successful prediction rows, keeping the latest row per lead ping."""
    if not _PREDICTION_MONITORING_PATH.exists():
        raise io.DataNotFoundError(
            "Prediction monitoring dataset not found: " f"{_PREDICTION_MONITORING_PATH}"
        )

    frame = pd.read_parquet(_PREDICTION_MONITORING_PATH)
    if frame.empty:
        return frame

    for column in ("created_at", "served_at"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)

    if "status" in frame.columns:
        frame = frame[
            frame["status"].astype("string").str.lower().isin({"success", "ok"})
        ].copy()

    if "lead_ping_id" not in frame.columns:
        return pd.DataFrame()

    frame["lead_ping_id"] = pd.to_numeric(
        frame["lead_ping_id"], errors="coerce"
    ).astype("Int64")
    frame = frame[frame["lead_ping_id"].notna()].copy()

    sort_col = "served_at" if "served_at" in frame.columns else "created_at"
    if sort_col in frame.columns:
        frame = frame.sort_values(sort_col, ascending=False, na_position="last")

    return frame.drop_duplicates("lead_ping_id", keep="first").reset_index(drop=True)


def _attach_prediction_monitoring(
    leads_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest prediction outputs to the historical lead rows."""
    if leads_df.empty or prediction_df.empty or "id" not in leads_df.columns:
        return leads_df.copy()

    left = leads_df.copy()
    left["_lead_ping_id"] = pd.to_numeric(left["id"], errors="coerce").astype("Int64")

    right = prediction_df.copy()
    right["_lead_ping_id"] = pd.to_numeric(
        right["lead_ping_id"], errors="coerce"
    ).astype("Int64")

    keep = [
        column
        for column in (
            "_lead_ping_id",
            "prediction_id",
            "served_at",
            "model_name",
            "model_version",
            "model_type",
            "training_table_version",
            "decision_path",
            "status",
            "expected_revenue",
            "recommended_bid",
            "recommended_bid_predicted_win_rate",
            "recommended_bid_predicted_profit",
            "recommended_bid_predicted_cm",
            "tat_seconds",
        )
        if column in right.columns
    ]
    right = right[keep].rename(
        columns={"expected_revenue": "prediction_expected_revenue"}
    )

    merged = left.merge(
        right,
        on="_lead_ping_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="_lead_ping_id")
    return transforms.add_recommended_bid_metrics(merged)


def _prepare_monitoring_rows(
    frame: pd.DataFrame,
    *,
    cohort_features: list[str],
) -> pd.DataFrame:
    """Normalize joined lead/prediction rows for degradation aggregation."""
    if frame.empty:
        return frame.copy()

    required = {
        "bidding_strategy_id",
        "recommended_bid_predicted_win_rate",
        "won",
        "created_at",
        *cohort_features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        logger.warning(
            "Joined performance data is missing required degradation columns: %s.",
            missing,
        )
        return pd.DataFrame()

    out = frame.copy()
    out["_strategy_id"] = pd.to_numeric(out["bidding_strategy_id"], errors="coerce")

    for feature in cohort_features:
        out[f"_cohort_{feature}"] = out[feature]

    out["_predicted_win_rate"] = pd.to_numeric(
        out["recommended_bid_predicted_win_rate"], errors="coerce"
    )
    out["_won"] = _coerce_outcome(out["won"])
    out["_event_time"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)

    valid = (
        out["_strategy_id"].notna()
        & out["_predicted_win_rate"].between(0.0, 1.0, inclusive="both")
        & out["_won"].notna()
        & out["_event_time"].notna()
    )
    for feature in cohort_features:
        valid &= out[f"_cohort_{feature}"].notna()

    out = out.loc[valid].copy()
    if out.empty:
        return out

    out["_strategy_id"] = out["_strategy_id"].astype("int64")
    return out


def build_hourly_metrics(
    frame: pd.DataFrame,
    *,
    strategy_ids: list[int],
    cohort_features: list[str],
    as_of: pd.Timestamp,
    persistence_windows: int,
) -> pd.DataFrame:
    """Build completed hourly win-rate realization metrics by configured cohort."""
    work = _prepare_monitoring_rows(frame, cohort_features=cohort_features)
    cohort_columns = ["bidding_strategy_id", *cohort_features]
    columns = [
        "window_start",
        *cohort_columns,
        "num_opportunities",
        "predicted_wins",
        "actual_wins",
        "predicted_winrate",
        "measured_winrate",
        "winrate_bias",
        "zscore",
    ]
    if work.empty or not strategy_ids:
        return pd.DataFrame(columns=columns)

    work = work[work["_strategy_id"].isin(strategy_ids)].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    completed_end = as_of.floor("h")
    start = completed_end - pd.Timedelta(hours=persistence_windows)
    work = work[
        work["_event_time"].ge(start) & work["_event_time"].lt(completed_end)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["window_start"] = work["_event_time"].dt.floor("h")
    p = work["_predicted_win_rate"].astype("float64")
    work["_predicted_wins"] = p
    work["_bernoulli_variance"] = p * (1.0 - p)

    source_group_keys = ["window_start", "_strategy_id"] + [
        f"_cohort_{feature}" for feature in cohort_features
    ]
    rename_columns = {"_strategy_id": "bidding_strategy_id"}
    rename_columns.update(
        {f"_cohort_{feature}": feature for feature in cohort_features}
    )

    agg = (
        work.groupby(source_group_keys, observed=False, dropna=False)
        .agg(
            num_opportunities=("_won", "size"),
            predicted_wins=("_predicted_wins", "sum"),
            actual_wins=("_won", "sum"),
            win_variance=("_bernoulli_variance", "sum"),
        )
        .reset_index()
        .rename(columns=rename_columns)
    )

    n = agg["num_opportunities"].astype("float64")
    agg["predicted_winrate"] = agg["predicted_wins"] / n
    agg["measured_winrate"] = agg["actual_wins"] / n
    agg["winrate_bias"] = np.where(
        agg["predicted_winrate"].gt(0.0),
        (agg["measured_winrate"] - agg["predicted_winrate"]) / agg["predicted_winrate"],
        np.nan,
    )
    sigma = np.sqrt(agg["win_variance"])
    agg["zscore"] = np.where(
        sigma.gt(0.0),
        (agg["actual_wins"] - agg["predicted_wins"]) / sigma,
        np.nan,
    )
    return agg[columns].sort_values(["window_start", *cohort_columns])


def evaluate_degradation(
    hourly: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Return persistently degrading cohorts for the latest completed hour."""
    result_columns = list(hourly.columns) + [
        "bad_windows",
        "critical_windows",
        "observed_windows",
        "severity",
    ]
    if hourly.empty:
        return pd.DataFrame(columns=result_columns)

    work = hourly.copy()
    work["warning_bad"] = work["winrate_bias"].le(cfg["warning_winrate_bias"]) & work[
        "zscore"
    ].le(cfg["warning_zscore"])
    work["critical_bad"] = work["winrate_bias"].le(cfg["critical_winrate_bias"]) & work[
        "zscore"
    ].le(cfg["critical_zscore"])

    latest_window = as_of.floor("h") - pd.Timedelta(hours=1)
    rows: list[dict[str, Any]] = []
    keys = ["bidding_strategy_id", *cfg["cohort_features"]]
    for _, group in work.groupby(keys, observed=False, dropna=False):
        current = group[group["window_start"].eq(latest_window)]
        if current.empty:
            continue
        latest = current.iloc[-1]
        if not bool(latest["warning_bad"]):
            continue

        bad_windows = int(group["warning_bad"].sum())
        if bad_windows < int(cfg["required_bad_windows"]):
            continue

        critical_windows = int(group["critical_bad"].sum())
        record = latest[hourly.columns].to_dict()
        record.update(
            {
                "bad_windows": bad_windows,
                "critical_windows": critical_windows,
                "observed_windows": int(len(group)),
                "severity": "critical" if critical_windows else "warning",
            }
        )
        rows.append(record)

    return pd.DataFrame(rows, columns=result_columns).sort_values(
        ["severity", "winrate_bias"], ascending=[True, True]
    )


def _jsonable(value: Any) -> Any:
    """Return a stable JSON-safe scalar for cohort state."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _state_key(row: Any, cohort_features: list[str]) -> str:
    """Build a stable state key; bidding strategy is always included."""
    parts = [f"bidding_strategy_id={int(row.bidding_strategy_id)}"]
    for feature in cohort_features:
        parts.append(f"{feature}={getattr(row, feature)}")
    return "|".join(parts)


def _load_state(path: Path = _STATE_PATH) -> dict[str, dict[str, Any]]:
    """Load active degradation state; invalid/missing state starts empty."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read degradation state %s: %s", path, exc)
        return {}
    active = payload.get("active", {}) if isinstance(payload, dict) else {}
    return active if isinstance(active, dict) else {}


def _write_state(
    degraded: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    cohort_features: list[str],
    path: Path = _STATE_PATH,
) -> None:
    """Persist the currently active cohort severities for alert deduplication."""
    active: dict[str, dict[str, Any]] = {}
    for row in degraded.itertuples(index=False):
        key = _state_key(row, cohort_features)
        cohort = {"bidding_strategy_id": int(row.bidding_strategy_id)}
        cohort.update(
            {feature: _jsonable(getattr(row, feature)) for feature in cohort_features}
        )
        active[key] = {
            "severity": row.severity,
            **cohort,
            "last_seen": as_of.isoformat(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": as_of.isoformat(),
        "active": active,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _severity_rank(value: str) -> int:
    return {"warning": 1, "critical": 2}.get(value, 0)


def _alert_changes(
    degraded: pd.DataFrame,
    previous: dict[str, dict[str, Any]],
    *,
    cohort_features: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return new/escalated degradation rows and recovered prior cohorts."""
    if degraded.empty:
        current_map: dict[str, str] = {}
    else:
        current_map = {
            _state_key(row, cohort_features): row.severity
            for row in degraded.itertuples(index=False)
        }

    changed_rows = []
    for row in degraded.itertuples(index=False):
        key = _state_key(row, cohort_features)
        old = previous.get(key, {}).get("severity", "")
        if key not in previous or _severity_rank(row.severity) > _severity_rank(old):
            changed_rows.append(row._asdict())

    recovered = [
        details
        for key, details in previous.items()
        if key not in current_map and isinstance(details, dict)
    ]
    return pd.DataFrame(changed_rows, columns=degraded.columns), recovered


def _display_value(value: Any) -> str:
    """Compact cohort scalar for Slack display."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _format_degradation_table(rows: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Format degrading cohorts as a compact monospace Slack table."""
    cohort_cols = ["bidding_strategy_id", *cfg["cohort_features"]]
    labels = ["Strategy"] + [
        feature.replace("_", " ").title() for feature in cfg["cohort_features"]
    ]
    widths = []
    for column, label in zip(cohort_cols, labels):
        values = [_display_value(value) for value in rows[column].tolist()]
        widths.append(min(max([len(label), *(len(v) for v in values)]), 24))

    fixed_labels = ["N", "Pred WR", "Meas WR", "Bias", "Z", "Bad", "Status"]
    fixed_widths = [7, 8, 8, 8, 6, 5, 6]
    header_parts = [label[:width].rjust(width) for label, width in zip(labels, widths)]
    header_parts += [
        label.rjust(width) for label, width in zip(fixed_labels, fixed_widths)
    ]
    header = "  ".join(header_parts)
    lines = [header, "-" * len(header)]

    for row in rows.itertuples(index=False):
        cohort_values = [row.bidding_strategy_id] + [
            getattr(row, feature) for feature in cfg["cohort_features"]
        ]
        parts = [
            _display_value(value)[:width].rjust(width)
            for value, width in zip(cohort_values, widths)
        ]
        bad = f"{int(row.bad_windows)}/{int(cfg['persistence_windows'])}"
        status = "CRIT" if row.severity == "critical" else "WARN"
        parts += [
            f"{int(row.num_opportunities):>7}",
            f"{row.predicted_winrate:>8.2%}",
            f"{row.measured_winrate:>8.2%}",
            f"{row.winrate_bias:>8.1%}",
            f"{row.zscore:>6.2f}",
            f"{bad:>5}",
            f"{status:>6}",
        ]
        lines.append("  ".join(parts))
    return "\n".join(lines)


def _build_slack_payload(
    alerts: pd.DataFrame,
    recovered: list[dict[str, Any]],
    *,
    as_of: pd.Timestamp,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the custom Slack message for degradation/recovery state changes."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "SmartHub ML Win-Rate Degradation",
                "emoji": True,
            },
        }
    ]
    fallback = ["SmartHub ML Win-Rate Degradation"]

    if not alerts.empty:
        table = _format_degradation_table(alerts, cfg)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*Win Rate Bias — degrading cohorts*\n```" + table + "```"
                    ),
                },
            }
        )
        fallback.append(table)

    if recovered:
        lines = []
        for item in recovered:
            parts = [f"Strategy {item.get('bidding_strategy_id')}"]
            parts.extend(
                f"{feature} {item.get(feature)}" for feature in cfg["cohort_features"]
            )
            lines.append(" · ".join(parts))
        recovery_text = "\n".join(f"• {line}" for line in lines)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Recovered cohorts*\n" + recovery_text,
                },
            }
        )
        fallback.extend(lines)

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        "latest completed hour: "
                        f"`{as_of.floor('h') - pd.Timedelta(hours=1)}` "
                        f"· persistence: {cfg['required_bad_windows']}/"
                        f"{cfg['persistence_windows']} windows"
                    ),
                }
            ],
        }
    )
    return {"text": "\n".join(fallback), "blocks": blocks}


def check_once(
    *,
    lead_type_id: int,
    as_of: pd.Timestamp | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run one degradation check and return currently degraded cohorts."""
    cfg = _config()
    if not cfg["enabled"]:
        logger.info("Model degradation monitoring is disabled in task config.")
        return pd.DataFrame()
    if not cfg["ml_bidding_strategy_ids"]:
        logger.warning(
            "No model_degradation.ml_bidding_strategy_ids configured; skipping."
        )
        return pd.DataFrame()

    if as_of is not None:
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")

    lookback_hours = int(cfg["persistence_windows"]) + 2
    lookback_days = max(1, int(math.ceil(lookback_hours / 24.0)) + 1)
    logger.info(
        "Loading performance data: lookback_days=%d, lead_type_id=%d, "
        "strategies=%s, cohort_features=%s.",
        lookback_days,
        lead_type_id,
        cfg["ml_bidding_strategy_ids"],
        cfg["cohort_features"],
    )

    leads = io.load_leads_window(lookback_days)
    if leads.empty:
        logger.warning("Historical leads dataset is empty for the requested lookback.")
        return pd.DataFrame()

    if "lead_type_id" not in leads.columns:
        logger.warning("Historical leads data has no lead_type_id column; skipping.")
        return pd.DataFrame()
    lead_type_values = pd.to_numeric(leads["lead_type_id"], errors="coerce")
    leads = leads.loc[lead_type_values.eq(lead_type_id)].copy()
    logger.info(
        "Historical leads filtered to lead_type_id=%d: rows=%d.",
        lead_type_id,
        len(leads),
    )
    if leads.empty:
        logger.warning(
            "No historical lead rows found for lead_type_id=%d; skipping.",
            lead_type_id,
        )
        return pd.DataFrame()

    if "bidding_strategy_id" not in leads.columns:
        logger.warning(
            "Historical leads data has no bidding_strategy_id column; skipping."
        )
        return pd.DataFrame()
    strategy_values = pd.to_numeric(leads["bidding_strategy_id"], errors="coerce")
    leads = leads.loc[strategy_values.isin(cfg["ml_bidding_strategy_ids"])].copy()
    logger.info(
        "Historical leads filtered to ML strategies %s: rows=%d.",
        cfg["ml_bidding_strategy_ids"],
        len(leads),
    )
    if leads.empty:
        logger.warning(
            "No historical lead rows found for configured ML strategies %s; "
            "skipping.",
            cfg["ml_bidding_strategy_ids"],
        )
        return pd.DataFrame()

    timestamps = pd.to_datetime(leads.get("created_at"), errors="coerce", utc=True)
    latest_data_time = timestamps.max()
    logger.info(
        "Filtered historical-lead time range: min=%s, max=%s, " "valid_timestamps=%d.",
        timestamps.min(),
        latest_data_time,
        int(timestamps.notna().sum()),
    )
    if pd.isna(latest_data_time):
        logger.warning("Historical leads data has no valid timestamps; skipping check.")
        return pd.DataFrame()

    if as_of is None:
        as_of = latest_data_time
        logger.info(
            "Using latest historical-lead timestamp as degradation anchor: %s.",
            as_of.isoformat(),
        )
    else:
        logger.info(
            "Using explicit historical replay anchor: %s; latest lead timestamp=%s.",
            as_of.isoformat(),
            latest_data_time.isoformat(),
        )

    prediction_monitoring = _load_prediction_monitoring()
    logger.info(
        "Prediction monitoring loaded: rows=%d, columns=%d.",
        len(prediction_monitoring),
        len(prediction_monitoring.columns),
    )
    if prediction_monitoring.empty:
        logger.warning("Prediction monitoring dataset is empty; skipping.")
        return pd.DataFrame()

    performance = _attach_prediction_monitoring(leads, prediction_monitoring)
    matched_predictions = (
        int(performance["recommended_bid_predicted_win_rate"].notna().sum())
        if "recommended_bid_predicted_win_rate" in performance.columns
        else 0
    )
    logger.info(
        "Joined performance data: rows=%d, matched_predictions=%d.",
        len(performance),
        matched_predictions,
    )

    prepared = _prepare_monitoring_rows(
        performance,
        cohort_features=cfg["cohort_features"],
    )
    logger.info(
        "Joined performance rows usable for degradation: valid=%d of %d.",
        len(prepared),
        len(performance),
    )
    if prepared.empty:
        logger.warning(
            "Degradation check skipped: no joined rows have all required "
            "prediction, outcome, timestamp, and cohort fields."
        )
        return pd.DataFrame()

    hourly = build_hourly_metrics(
        performance,
        strategy_ids=cfg["ml_bidding_strategy_ids"],
        cohort_features=cfg["cohort_features"],
        as_of=as_of,
        persistence_windows=int(cfg["persistence_windows"]),
    )
    degraded = evaluate_degradation(hourly, as_of=as_of, cfg=cfg)

    logger.info(
        "Model degradation check: %d hourly cohort rows, %d active degradations.",
        len(hourly),
        len(degraded),
    )

    previous = _load_state()
    alerts, recovered = _alert_changes(
        degraded,
        previous,
        cohort_features=cfg["cohort_features"],
    )

    if dry_run:
        if degraded.empty:
            logger.info("Dry run: no persistent win-rate degradation detected.")
        else:
            logger.warning(
                "Dry run: persistent degradation detected.\n%s",
                _format_degradation_table(degraded, cfg),
            )
        return degraded

    state_changed = not alerts.empty or bool(recovered)
    delivered = True
    if state_changed:
        payload = _build_slack_payload(
            alerts,
            recovered,
            as_of=as_of,
            cfg=cfg,
        )
        delivered = notifications.notify_raw(payload)
        if not delivered:
            logger.warning(
                "Degradation state changed but Slack was not delivered; "
                "state will not advance so the next run retries."
            )

    if delivered:
        _write_state(
            degraded,
            as_of=as_of,
            cohort_features=cfg["cohort_features"],
        )
    return degraded


def _parse_as_of(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def main() -> None:
    """CLI entry point for one scheduled degradation check."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lead-type-id",
        type=int,
        required=True,
        help="lead type ID to evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute/log degradation without Slack or state changes",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "UTC timestamp for historical replay/testing. By default the monitor "
            "anchors on the latest timestamp available in the historical lead data."
        ),
    )
    args = parser.parse_args()
    check_once(
        lead_type_id=args.lead_type_id,
        as_of=_parse_as_of(args.as_of),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
