#!/usr/bin/env python3
"""
Piece 5 — Stub server: a safe stand-in for /recommend_bid.

A dependency-free fake of the bid API so the whole harness (builder + real HTTP
+ ledger + analyzer) can be proven end-to-end with ZERO risk before pointing at
anything real. It mimics the real contract: 422 on a missing required field,
200 with a valid-shaped bid otherwise, and a realistic decision_path mix.

    python stub_server.py --port 8080
    # then:  python run.py --data snapshot.parquet \
    #        --url http://127.0.0.1:8080 --dispatch
"""

from __future__ import annotations

import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Server(ThreadingHTTPServer):
    # Absorb bursts: big listen backlog + daemon threads (default backlog is 5,
    # which a 100+ concurrent burst overflows into connection resets).
    daemon_threads = True
    request_queue_size = 1024


_REQUIRED = (
    "expected_revenue",
    "lead_type_id",
    "campaign_id",
    "source_type_id",
    "traffic_tier",
    "state",
    "created_at",
    "lead_ping_id",
)
_SEQ = {"n": 0}


def _decide(payload: dict) -> dict:
    """Cheap plausible bid: a fraction of expected_revenue, with occasional
    cold-start / exploration / no-bid outcomes so the analyzer sees the real mix."""
    rev = float(payload.get("expected_revenue", 0) or 0)
    target_cm = float(payload.get("target_cm", 0.25) or 0.25)
    min_bid = float(payload.get("min_bid", 0.25) or 0.25)
    max_bid = round(rev * (1 - target_cm), 2)
    roll = random.random()
    if roll < 0.03:
        path, reason = "cold_start_fallback", "No promoted model for this lead type."
        bid = round(min_bid + 0.3 * (max_bid - min_bid), 2)
        win = prof = None
    elif roll < 0.08:
        path, reason = "exploration", "Scheduled hour-of-week probe."
        bid = round(random.uniform(min_bid, max(min_bid, max_bid)), 2)
        win, prof = round(random.uniform(0.4, 0.95), 4), round(
            random.uniform(0, rev), 2
        )
    elif roll < 0.15 or max_bid < min_bid:
        path, reason = "model", "No profitable bid at/above floor."
        bid, win, prof = None, None, None  # null bid = "do not bid"
    else:
        path, reason = "model", "Standard profit-maximizing bid."
        win = round(random.uniform(0.5, 0.99), 4)
        bid = round(min_bid + win * (max_bid - min_bid), 2)
        prof = round(win * rev - bid, 2)
    _SEQ["n"] += 1
    return {
        "recommended_bid": bid,
        "recommended_bid_predicted_win_rate": win,
        "recommended_bid_predicted_profit": prof,
        "max_bid": max(max_bid, 0.0),
        "n_candidate_bids": max(1, int((max(max_bid, min_bid) - min_bid) / 0.25)),
        "decision_path": path,
        "decision_reason": reason,
        "model_data_age_days": 0,
        "prediction_id": f"stub-{_SEQ['n']:09d}",
        "lead_ping_id": payload.get("lead_ping_id"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json(200, {"status": "ok", "model_loaded": True, "stub": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/recommend_bid"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"bad json: {exc}"})
            return
        missing = [
            f for f in _REQUIRED if payload.get(f) is None or payload.get(f) == ""
        ]
        if missing:
            self._json(422, {"detail": f"missing required field(s): {missing}"})
            return
        if _LATENCY:  # optional simulated model time
            time.sleep(random.uniform(*_LATENCY))
        self._json(200, _decide(payload))


_LATENCY: tuple[float, float] | None = None


def main() -> None:
    global _LATENCY
    ap = argparse.ArgumentParser(description="Fake /recommend_bid for safe testing.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--latency-ms",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="simulate model latency, e.g. --latency-ms 5 25",
    )
    args = ap.parse_args()
    if args.latency_ms:
        _LATENCY = (args.latency_ms[0] / 1000.0, args.latency_ms[1] / 1000.0)
    srv = _Server((args.host, args.port), Handler)
    print(
        f"stub /recommend_bid listening on http://{args.host}:{args.port} "
        f"(latency={args.latency_ms})",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstub server stopped")


if __name__ == "__main__":
    main()
