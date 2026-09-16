#!/usr/bin/env python3
"""Print which promoted model each lead type resolves to.

Reads the production serving pointer (``<lead_type>/current.json`` in the S3 /
MinIO model store) to report the promoted version, its training-run id, and the
UTC time it was promoted, plus the resolved artifact path. Flags whether the
version changed since the previous run (state kept in model_state.json).

Run inside the serve container (it has the production-storage env)::

    docker exec smarthub-serve-staging python /app/data/sim/model_info.py
"""

from __future__ import annotations

import json
import os

from smarthub.core import lead_types
from smarthub.train_and_predict import registry

STATE = "/app/data/sim/model_state.json"


def _pointer(name: str) -> dict:
    """Return the production serving pointer for a lead type ({} on miss)."""
    try:
        return registry.production_serving_pointer(name) or {}
    except Exception as exc:  # noqa: BLE001 - report, never crash the run
        return {"_error": str(exc)}


def _artifact(name: str) -> str | None:
    """Return the resolved serving artifact path, or None."""
    try:
        return str(registry.currently_serving_model_path(name))
    except Exception:  # noqa: BLE001
        return None


def _load_prev() -> dict:
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    prev = _load_prev()
    cur: dict = {}
    for name in lead_types.all_lead_types():
        lt = lead_types.lead_type_id(name)
        p = _pointer(name)
        if "_error" in p:
            print(f"  [model] {name} (lead_type={lt}): ERROR {p['_error']}")
            continue
        ver = p.get("production_model_version")
        run_id = p.get("training_run_id") or p.get("version")
        promoted_at = p.get("promoted_at")
        cur[name] = {"version": ver, "run_id": run_id, "promoted_at": promoted_at}

        note = ""
        was = prev.get(name)
        if was:
            if was.get("run_id") != run_id:
                note = f"  <<< CHANGED (was version={was.get('version')})"
            else:
                note = "  (unchanged since last run)"
        print(
            f"  [model] {name} (lead_type={lt}): version={ver} "
            f"run_id={run_id} promoted_at={promoted_at}{note}"
        )
        print(f"          artifact={_artifact(name)}")

    try:
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        with open(STATE, "w") as fh:
            json.dump(cur, fh, indent=2)
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
