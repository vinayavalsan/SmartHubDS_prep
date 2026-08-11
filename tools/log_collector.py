#!/usr/bin/env python3
"""Persist the docker-compose stack's live logs to date-stamped JSON files.

Follows EVERY container in the stack and writes one **unified JSON Lines** stream
where each line is tagged with the container it came from -- so a single file
holds the whole stack and you can filter by `container`.

Two output layouts (both JSONL):
  * combined    ->  logs/combined-YYYY-MM-DD.log     (every container, one stream)
  * per-service ->  logs/<container>/YYYY-MM-DD.log  (one folder per container)

Per line, the collector:
  * pulls the container name from the compose log prefix
  * pulls docker's own timestamp (from `--timestamps`)
  * if the message body is itself JSON (your Python services with LOG_FORMAT=json),
    it is PARSED and merged -- so structured fields (level, event, error.stack,
    error.fingerprint, request_id, ...) are preserved and a `container` key added
  * otherwise (Postgres, MinIO, nginx, Ollama, Prefect server, ...) the raw text
    is WRAPPED as {"ts", "container", "msg"} so those lines are still queryable

Operational behaviour:
  * files roll at midnight automatically (filename derives from today's date)
  * anything older than --retain-days is deleted (on start + each new day)
  * follows `docker compose logs -f` and reconnects if the stream drops
  * only NEW lines are captured (`--tail 0`), so restarts don't duplicate history

Pure stdlib. Intended to run as a systemd service from the compose project dir:

  python3 tools/log_collector.py --logdir logs --retain-days 30
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import time
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_TS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")   # leading RFC3339 token


def _today() -> str:
    return _dt.date.today().isoformat()


def _prune(logdir: Path, days: int) -> None:
    """Delete combined + per-service log files older than `days`."""
    cutoff = time.time() - days * 86400
    for pattern in ("combined-*.log", "*/*.log"):
        for f in logdir.glob(pattern):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass


def _split_container(line: str):
    """Compose prefixes each line 'container  | body' -> (container, body)."""
    head, sep, rest = line.partition("|")
    if sep and len(head) <= 64:
        name = _SAFE.sub("_", head.strip()) or "unknown"
        return name, rest.lstrip()
    return "unknown", line


def _split_ts(body: str):
    """Peel docker's leading RFC3339 timestamp (from --timestamps) off the body."""
    parts = body.split(" ", 1)
    if parts and _TS.match(parts[0]):
        return parts[0], (parts[1] if len(parts) > 1 else "")
    return None, body


def _to_record(line: str) -> dict:
    """Turn one raw compose log line into a container-tagged JSON record."""
    container, body = _split_container(line)
    ts, body = _split_ts(body)
    body = body.rstrip("\n")

    # If the app already emitted structured JSON, merge it (keep its richer
    # fields) and just stamp the container. Otherwise wrap the raw text.
    if body[:1] == "{":
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                obj["container"] = container
                if ts and "ts" not in obj:
                    obj["ts"] = ts
                return obj
        except ValueError:
            pass
    return {"ts": ts, "container": container, "msg": body}


def main() -> None:
    ap = argparse.ArgumentParser(description="date-stamped JSON docker-compose log collector")
    ap.add_argument("--logdir", default="logs", help="output directory")
    ap.add_argument("--retain-days", type=int, default=30, help="delete files older than this")
    ap.add_argument("--compose-file", default="docker-compose.yaml")
    ap.add_argument("--combined-prefix", default="combined")
    ap.add_argument("--no-per-service", action="store_true",
                    help="write only the single combined file")
    ap.add_argument("--reconnect-secs", type=float, default=5.0)
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    cmd = ["docker", "compose", "-f", args.compose_file, "logs",
           "--follow", "--no-color", "--timestamps", "--tail", "0"]

    day: str | None = None
    combined = None
    svc_files: dict[str, object] = {}

    def roll(today: str) -> None:
        nonlocal day, combined, svc_files
        if combined is not None:
            combined.close()
        for fh in svc_files.values():
            fh.close()
        svc_files = {}
        combined = open(logdir / f"{args.combined_prefix}-{today}.log", "a", buffering=1)
        day = today
        _prune(logdir, args.retain_days)

    roll(_today())
    print(f">> log collector: unified JSON -> {logdir}/ "
          f"({'combined only' if args.no_per_service else 'combined + per-service'}), "
          f"retain {args.retain_days} days", flush=True)

    while True:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as exc:  # noqa: BLE001 -- docker daemon not ready yet
            print(f">> cannot start 'docker compose logs': {exc}; retrying", flush=True)
            time.sleep(args.reconnect_secs)
            continue

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if not line.strip():
                    continue
                today = _today()
                if today != day:
                    roll(today)
                record = _to_record(line)
                out = json.dumps(record, default=str, ensure_ascii=False) + "\n"
                combined.write(out)
                if not args.no_per_service:
                    svc = record.get("container", "unknown")
                    fh = svc_files.get(svc)
                    if fh is None:
                        d = logdir / svc
                        d.mkdir(parents=True, exist_ok=True)
                        fh = open(d / f"{today}.log", "a", buffering=1)
                        svc_files[svc] = fh
                    fh.write(out)
        except Exception:  # noqa: BLE001 -- stream hiccup; reconnect below
            pass
        finally:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(args.reconnect_secs)


if __name__ == "__main__":
    main()
