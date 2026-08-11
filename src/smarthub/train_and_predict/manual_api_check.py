"""Manual client for the SmartHub bid prediction API.

This module submits a sample request for interactive endpoint verification.
"""

import json

import requests

from smarthub.core.logging_utils import get_logger

logger = get_logger(__name__)
URL = "http://127.0.0.1:8000/recommend_bid"


sample_consumer = {
    # Optimizer inputs
    "expected_revenue": 25.00,
    "target_cm": 0.25,
    "min_bid": 0.00,
    "bid_step": 0.25,
    # Required model/context fields
    "campaign_id": 12345,
    "lead_type_id": 6,
    "created_hour": 14,
    "created_dayofweek": 2,
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


response = requests.post(URL, json=sample_consumer, timeout=30)

logger.info("Status code: %s", response.status_code)

try:
    result = response.json()
except Exception:
    logger.exception("Response was not valid JSON: %s", response.text)
    raise

logger.info("Response body:\n%s", json.dumps(result, indent=2))

response.raise_for_status()
