"""Manual client for the SmartHub serving API (recommend + explain).

Submits a sample ``/recommend_bid`` request, then explains that prediction by id
via ``/explain_bid``. Production explanation consumes the *persisted* prediction,
so the explanation is requested by ``prediction_id`` rather than by resending the
lead. Run against a live server for interactive verification:

    python -m smarthub.server.manual_api_check
"""

import json
import time

import requests

BASE_URL = "http://127.0.0.1:8000"


sample_consumer = {
    # Optimizer inputs
    "expected_revenue": 25.00,
    "target_cm": 0.25,
    "min_bid": 0.00,
    "bid_step": 0.25,
    # Required model/context fields
    "lead_type_id": 6,
    "account_id": 67890,
    "lead_ping_id": 987654321,
    "campaign_id": 12,
    "source_type_id": 1950,
    "traffic_tier": "1-1456261",
    "created_at": "2026-08-20T21:00:00Z",
    # Consumer / lead attributes
    "state": "TX",
    "insured": "true",
    "home_owner": "false",
    "dui": "false",
    "num_vehicles": 2,
    "num_drivers": 2,
    "num_auto_violations": 0,
    "num_auto_accidents": 0,
    "continuous_coverage_months": 24,
    "military_affiliation": "false",
    "gender": "Female",
    "marital_status": "Single",
    "age": 34,
}


def main() -> None:
    """Recommend a bid, then explain that prediction by id."""
    resp = requests.post(f"{BASE_URL}/recommend_bid", json=sample_consumer, timeout=30)
    print("recommend_bid:", resp.status_code)
    resp.raise_for_status()
    prediction = resp.json()
    print(json.dumps(prediction, indent=2))

    prediction_id = prediction.get("prediction_id")
    if not prediction_id:
        return

    # Production explanation consumes the *persisted* prediction by id. The
    # server logs that prediction in a post-response background task, so a
    # back-to-back explain can beat the insert -> transient 404. A real client
    # explains seconds later and never sees this; here we just poll briefly.
    exp = None
    for _ in range(10):
        exp = requests.post(
            f"{BASE_URL}/explain_bid",
            json={"prediction_id": prediction_id},
            timeout=60,
        )
        if exp.status_code != 404:
            break
        time.sleep(0.3)

    print("explain_bid:", exp.status_code)
    try:
        print(json.dumps(exp.json(), indent=2))
    except Exception:  # noqa: BLE001 -- print raw body on non-JSON responses
        print(exp.text)


if __name__ == "__main__":
    main()
