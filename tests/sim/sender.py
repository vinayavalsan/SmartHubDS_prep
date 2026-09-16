"""
Piece 4 — Sender + capture ledger.

Uses `requests` + a thread pool (the same stack as tests/bidload.py, so it runs
in the worker container with no extra installs). Each request is SUBMITTED to
the pool at its scheduled instant and not waited on, so a burst of N arrivals
runs concurrently (open-loop). Every request — success, error, timeout — is
written as one JSON line to the ledger and rolled into `Stats`.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests


@dataclass
class Stats:
    sent: int = 0
    ok: int = 0
    by_status: Counter = field(default_factory=Counter)
    decision_path: Counter = field(default_factory=Counter)
    null_bids: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    per_second: Counter = field(default_factory=Counter)
    inflight: int = 0
    max_inflight: int = 0
    errors: Counter = field(default_factory=Counter)


class Ledger:
    """Thread-safe append-only JSONL writer, one line per request."""

    def __init__(self, path: str):
        # Truncate (not append): one run.py invocation == one fresh capture.
        # A supervisor day-retry re-runs the whole day, so appending would
        # double-write that day's ledger; "w" makes each attempt authoritative.
        self._fh = open(path, "w", buffering=1)  # line-buffered, truncating
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            self._fh.write(line)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


_KEEP = (
    "recommended_bid",
    "recommended_bid_predicted_win_rate",
    "recommended_bid_predicted_profit",
    "decision_path",
    "model_data_age_days",
    "prediction_id",
    "n_candidate_bids",
    "max_bid",
)


class Sender:
    def __init__(
        self,
        url: str,
        ledger: Ledger,
        stats: Stats,
        api_key: str | None = None,
        timeout: float = 5.0,
        workers: int = 512,
    ):
        self.url = url
        self.ledger = ledger
        self.stats = stats
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=workers, pool_maxsize=workers, max_retries=0
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def submit(self, payload: dict, sched_offset: float) -> None:
        """Non-blocking: hand the request to the pool (fires concurrently)."""
        self._pool.submit(self._send, payload, sched_offset)

    def close(self) -> None:
        self._pool.shutdown(wait=True)  # let in-flight requests finish
        self._session.close()

    def _send(self, payload: dict, sched_offset: float) -> None:
        st = self.stats
        with self._lock:
            st.inflight += 1
            st.max_inflight = max(st.max_inflight, st.inflight)
        rec: dict = {
            "ts": round(sched_offset, 3),
            "lead_ping_id": payload.get("lead_ping_id"),
            "lead_type_id": payload.get("lead_type_id"),
        }
        t0 = time.perf_counter()
        dp = None
        try:
            r = self._session.post(self.url, json=payload, timeout=self.timeout)
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            rec["status"] = r.status_code
            rec["ok"] = 200 <= r.status_code < 300
            if rec["ok"]:
                try:
                    body = r.json()
                    for k in _KEEP:
                        if k in body:
                            rec[k] = body[k]
                    dp = body.get("decision_path")
                except Exception as exc:  # noqa: BLE001
                    rec["parse_error"] = str(exc)
            else:
                rec["body"] = r.text[:300]
        except Exception as exc:  # noqa: BLE001 - timeouts, conn errors
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            rec["status"] = 0
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        # roll into stats under the lock, then write the ledger line
        with self._lock:
            st.inflight -= 1
            st.sent += 1
            st.per_second[int(sched_offset)] += 1
            st.by_status[rec["status"]] += 1
            if rec.get("ok"):
                st.ok += 1
                if dp:
                    st.decision_path[dp] += 1
                if rec.get("recommended_bid") is None and "recommended_bid" in rec:
                    st.null_bids += 1
            elif rec["status"] == 0:
                st.errors[rec.get("error", "error").split(":")[0]] += 1
            if "latency_ms" in rec and rec.get("ok"):
                st.latencies_ms.append(rec["latency_ms"])
        self.ledger.write(rec)
