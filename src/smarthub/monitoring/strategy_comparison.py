"""Four-hour account/strategy observations; dry-run never sends or writes state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd

from smarthub.core import auction, notifications, paths, task_config, transforms
from smarthub.core.lead_types import lead_type_name
from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)
_SECTION = "strategy_comparison"


def report_bounds(as_of=None, window_hours=4):
    """Return the latest completed UTC-aligned window, end exclusive."""
    if window_hours < 1 or window_hours > 24 or 24 % window_hours:
        raise ValueError("window_hours must be a positive divisor of 24")
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    end = now.floor(f"{window_hours}h")
    return end - pd.Timedelta(hours=window_hours), end


def build_summary(frame, *, lead_type_id, ml_strategy_ids, start, end):
    """Count unique positive-bid, non-errored pings in ML-active accounts.

    Blank/non-true outcomes are losses, following normalize_won. Every strategy
    is shown separately. Average won bid is missing when there are no wins.
    """
    columns = [
        "account_id",
        "bidding_strategy_id",
        "is_ml",
        "opportunities",
        "won",
        "winrate",
        "avg_bid",
        "avg_bid_on_wins",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "id",
        "account_id",
        "bidding_strategy_id",
        "lead_type_id",
        "created_at",
        "bid",
        "won",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Strategy summary missing columns: {sorted(missing)}")
    work = frame.copy()
    for col in ("id", "account_id", "bidding_strategy_id", "lead_type_id", "bid"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce", utc=True)
    valid = (
        work["lead_type_id"].eq(lead_type_id)
        & work["created_at"].ge(start)
        & work["created_at"].lt(end)
        & work["bid"].gt(0)
        & np.isfinite(work["bid"])
        & ~auction.erred_mask(work)
    )
    work = work.loc[valid].dropna(subset=["id", "account_id", "bidding_strategy_id"])
    work = work.drop_duplicates("id", keep="last").copy()
    accounts = work.loc[work["bidding_strategy_id"].isin(ml_strategy_ids), "account_id"]
    work = work.loc[work["account_id"].isin(accounts)].copy()
    work["_won"] = transforms.normalize_won(work["won"])
    work["_won_bid"] = work["bid"].where(work["_won"].eq(1))
    summary = (
        work.groupby(["account_id", "bidding_strategy_id"])
        .agg(
            opportunities=("id", "size"),
            won=("_won", "sum"),
            avg_bid=("bid", "mean"),
            avg_bid_on_wins=("_won_bid", "mean"),
        )
        .reset_index()
    )
    summary["winrate"] = summary["won"] / summary["opportunities"]
    summary["is_ml"] = summary["bidding_strategy_id"].isin(ml_strategy_ids)
    return summary.sort_values(
        ["account_id", "is_ml", "bidding_strategy_id"], ascending=[True, False, True]
    )[columns].reset_index(drop=True)


def build_payloads(summary, *, lead_type_id, start, end, min_opportunities):
    """Build informational tables in the shared SmartHub Slack layout."""
    period = f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC (end exclusive)"
    headers = ["Account", "Strategy", "N", "Won", "Meas WR", "Avg bid", "Won bid"]
    rows = []
    for row in summary.itertuples(index=False):
        strategy = str(int(row.bidding_strategy_id)) + (" ML" if row.is_ml else "")
        if row.opportunities < min_opportunities:
            strategy += "*"
        won_bid = (
            "N/A" if pd.isna(row.avg_bid_on_wins) else f"${row.avg_bid_on_wins:.2f}"
        )
        rows.append(
            [
                str(int(row.account_id)),
                strategy,
                f"{row.opportunities:,}",
                f"{row.won:,}",
                f"{row.winrate:.1%}",
                f"${row.avg_bid:.2f}",
                won_bid,
            ]
        )
    note = (
        f"* Fewer than {min_opportunities} opportunities. "
        "Avg bid: all placed bids; Avg bid on wins: won leads only. "
        "Accounts shown only where ML placed bids."
    )
    widths = (
        [
            max(len(label), *(len(row[index]) for row in rows))
            for index, label in enumerate(headers)
        ]
        if rows
        else [len(label) for label in headers]
    )
    header = "  ".join(label.rjust(width) for label, width in zip(headers, widths))
    payloads = []
    for offset in range(0, len(rows), 20):
        lines = [header, "-" * len(header)]
        previous_account = None
        for row in rows[offset : offset + 20]:
            if previous_account is not None and row[0] != previous_account:
                lines.append("")
            lines.append(
                "  ".join(value.rjust(width) for value, width in zip(row, widths))
            )
            previous_account = row[0]
        payloads.append(
            notifications.build_table_payload(
                "strategy-comparison",
                "\n".join(lines),
                subject=f"{lead_type_name(lead_type_id)} ({lead_type_id})",
                headline="ML and non-ML bidding strategies by account",
                footer_extra=f"window: {period} · {note}",
            )
        )
    return payloads


def _load_raw(lookback):
    from smarthub.core import storage
    from smarthub.core.config import StorageSettings

    return storage.load_window_raw(StorageSettings.from_env(), lookback)


def _send(payload):
    return notifications.notify_raw(payload, category="updates")


def check_once(*, lead_type_id, as_of=None, dry_run=False, resend=False):
    """Load raw historical outcomes and send each completed-window page once."""
    if not task_config.get_bool(_SECTION, "enabled", True):
        logger.info("Strategy comparison is disabled in task config; no report sent.")
        return pd.DataFrame()
    hours = task_config.get_int(_SECTION, "window_hours", 4)
    minimum = task_config.get_int(_SECTION, "min_opportunities", 100)
    if minimum < 1:
        raise ValueError("Strategy comparison min_opportunities must be positive")
    ml_ids = task_config.get("model_degradation", "ml_bidding_strategy_ids", [])
    ml_ids = sorted({int(value) for value in ml_ids})
    if not ml_ids:
        raise ValueError("Configure model_degradation.ml_bidding_strategy_ids first")
    start, end = report_bounds(as_of, hours)
    logger.info(
        "Strategy comparison: lead_type_id=%s; window %s to %s; dry_run=%s; resend=%s.",
        lead_type_id,
        start,
        end,
        dry_run,
        resend,
    )
    # Raw storage includes non-ML rows; prediction logs are unnecessary.
    lookback = max(
        3,
        int(math.ceil((pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400))
        + 2,
    )
    frame = _load_raw(lookback)
    summary = build_summary(
        frame,
        lead_type_id=lead_type_id,
        ml_strategy_ids=ml_ids,
        start=start,
        end=end,
    )
    if summary.empty:
        logger.warning(
            "No ML-active accounts in strategy report window %s to %s", start, end
        )
        return summary
    # Require data beyond the window as a basic freshness check. This cannot
    # establish completeness of every source row; schedule after the data pull.
    timestamps = pd.to_datetime(
        frame.loc[
            pd.to_numeric(frame["lead_type_id"], errors="coerce").eq(lead_type_id),
            "created_at",
        ],
        errors="coerce",
        utc=True,
    )
    if timestamps.max() < end or pd.isna(timestamps.max()):
        raise ValueError(
            "Strategy report data is stale; run data-pull after the report window ends"
        )
    payloads = build_payloads(
        summary,
        lead_type_id=lead_type_id,
        start=start,
        end=end,
        min_opportunities=minimum,
    )
    if dry_run:
        for payload in payloads:
            logger.info("Dry run (updates):\n%s", payload["text"])
        return summary
    state_path = (
        paths.data_dir() / "monitoring" / f"strategy_comparison_{lead_type_id}.json"
    )
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    period = f"{start.isoformat()}/{end.isoformat()}"
    if state.get("period") == period and not resend:
        sent = set(state.get("sent_pages", []))
        # Freeze pages after the first attempt so a partial retry cannot shift
        # account rows between already-delivered and outstanding pages.
        payloads = state.get("payloads", payloads)
    else:
        sent = set()
    if resend:
        logger.info("Manual resend: bypassing previous delivery state for this window.")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    def save_state():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "period": period,
                    "sent_pages": sorted(sent),
                    "payloads": payloads,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(state_path)

    save_state()
    delivered = 0
    skipped = 0
    for index, payload in enumerate(payloads):
        # Page identity stays stable on retries even if late outcomes change.
        page = hashlib.sha256(f"{period}/{index}".encode()).hexdigest()
        if page in sent:
            skipped += 1
            logger.info(
                "Skipping page %d/%d: already sent for this window. "
                "Run the strategy_comparison CLI to resend manually.",
                index + 1,
                len(payloads),
            )
            continue
        logger.info(
            "Sending strategy summary page %d/%d to Slack updates.",
            index + 1,
            len(payloads),
        )
        if not _send(payload):
            raise RuntimeError(
                "Strategy summary delivery failed or Slack is not configured"
            )
        sent.add(page)
        delivered += 1
        save_state()
        logger.info(
            "Slack accepted strategy summary page %d/%d.", index + 1, len(payloads)
        )
    logger.info(
        "Strategy summary completed: %d pages sent; %d already sent.",
        delivered,
        skipped,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-type-id", type=int, required=True)
    parser.add_argument(
        "--as-of", help="UTC replay anchor; reports the latest completed window"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    check_once(
        lead_type_id=args.lead_type_id,
        as_of=args.as_of,
        dry_run=args.dry_run,
        resend=not args.dry_run,
    )


if __name__ == "__main__":
    main()
