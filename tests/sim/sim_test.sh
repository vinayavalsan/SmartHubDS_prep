#!/usr/bin/env bash
# One control script for the SmartHub replay/load test.
#
# Runs on the EC2 HOST (needs docker + python3; monitors use host python3 stdlib).
# It spins up an ISOLATED staging serve (same image+models, throwaway DB), then
# drives the supervised replay inside prefect-worker and lets you pull a live
# per-lead report any time.
#
#   bash tests/sim/sim_test.sh start          # real-time 4-day supervised run
#   bash tests/sim/sim_test.sh report         # live report (recommended vs actual bid)
#   bash tests/sim/sim_test.sh stop           # stop the replay + monitors (leaves serve up)
#   bash tests/sim/sim_test.sh down           # stop + remove staging serve + drop DB
#   bash tests/sim/sim_test.sh up             # just bring up the staging serve
#
# Quick compressed rehearsal against the REAL serve (finishes in ~6s):
#   SPEED=60 SEGMENT_MINUTES=6 DAYS=1 bash tests/sim/sim_test.sh start
#
# Env overrides: DAYS, BURST_PROB, SPEED, SEGMENT_MINUTES, SERVE_WORKERS, SLACK_WEBHOOK

cd "$(dirname "$0")/../.." || exit 1        # -> repo root (compose paths resolve here)

COMPOSE="docker compose -f docker-compose.prefect.yml -f tests/sim/docker-compose.staging-serve.yml"
WORKER=prefect-worker
PG=prefect-postgres
STAGING=smarthub-serve-staging
URL=http://serve-staging:8000
DATA=/app/data/sim/snapshot.parquet          # path inside the worker container
LEDGERS=/app/data/sim/ledgers
DAYS=${DAYS:-4}
BURST_PROB=${BURST_PROB:-0.6}
SPEED=${SPEED:-1}                            # 1 = real time; 60 = compressed rehearsal
SEGMENT_MINUTES=${SEGMENT_MINUTES:-1440}     # minutes per "day"
export SERVE_WORKERS=${SERVE_WORKERS:-4}

log(){ echo "[$(date +%H:%M:%S)] $*"; }

ensure_db(){
  if docker exec "$PG" psql -U prefect -tAc \
       "SELECT 1 FROM pg_database WHERE datname='smarthub_staging'" | grep -q 1; then
    log "throwaway DB smarthub_staging already exists"
  else
    log "creating throwaway DB smarthub_staging"
    docker exec "$PG" psql -U prefect -c "CREATE DATABASE smarthub_staging;"
    docker exec "$PG" sh -c \
      "pg_dump -U prefect -t 'smarthub_config*' prefect | psql -U prefect -d smarthub_staging" \
      >/dev/null 2>&1 || true   # copy prod config for fidelity (best-effort)
  fi
}

wait_health(){
  log "waiting for $STAGING to load the model ..."
  for i in $(seq 1 40); do
    if docker exec "$WORKER" python -c \
        "import urllib.request,json,sys; \
r=json.load(urllib.request.urlopen('$URL/health?lead_type_id=6')); \
sys.exit(0 if r.get('model_loaded') else 1)" 2>/dev/null; then
      log "staging serve healthy (model_loaded=true)"; return 0
    fi
    sleep 5
  done
  log "ERROR: staging serve did not become healthy — check: docker logs $STAGING"; return 1
}

cmd_up(){
  ensure_db
  log "bringing up $STAGING (SERVE_WORKERS=$SERVE_WORKERS)"
  $COMPOSE up -d serve-staging || exit 1
  wait_health || exit 1
}

cmd_start(){
  cp tests/sim/*.py data/sim/ 2>/dev/null      # refresh harness copies for the mount
  cmd_up || exit 1
  mkdir -p data/sim/ledgers
  # host-side monitors (stdlib python3; samples the staging serve + postgres + disk)
  pkill -f "monitors.py" 2>/dev/null
  nohup python3 tests/sim/monitors.py --interval 30 --out data/sim/health.csv \
      --serve-container "$STAGING" --pg-container "$PG" --disk-path data/sim \
      > data/sim/monitors.log 2>&1 &
  log "monitors started -> data/sim/health.csv"
  # the supervised replay, inside the worker (where pandas/requests live)
  docker exec -e SLACK_WEBHOOK="${SLACK_WEBHOOK:-}" -d "$WORKER" sh -c \
    "cd /app/data/sim && python supervisor.py --data $DATA --url $URL \
       --days $DAYS --segment-minutes $SEGMENT_MINUTES --speed $SPEED \
       --burst-prob $BURST_PROB --ledger-dir $LEDGERS \
       > /app/data/sim/supervisor.log 2>&1"
  log "replay started: days=$DAYS speed=${SPEED}x burst_prob=$BURST_PROB -> $URL"
  echo
  echo "  watch:   tail -f data/sim/supervisor.log"
  echo "  report:  bash tests/sim/sim_test.sh report"
  echo "  stop:    bash tests/sim/sim_test.sh stop"
}

cmd_report(){
  log "building live report from all ledgers ..."
  docker exec "$WORKER" sh -c \
    "cd /app/data/sim && python analyze.py '$LEDGERS/*.jsonl' --data $DATA \
       --report-csv /app/data/sim/report.csv"
  echo
  log "full per-lead CSV on host: data/sim/report.csv"
}

cmd_stop(){
  log "stopping replay + monitors"
  # the worker image has no pkill/pgrep — scan /proc from python (always present)
  docker exec "$WORKER" python -c '
import os, signal
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    try:
        cl = open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "ignore")
    except Exception:
        continue
    if "supervisor.py" in cl or "/app/data/sim/run.py" in cl:
        try:
            os.kill(int(p), signal.SIGTERM)
        except Exception:
            pass
' 2>/dev/null
  pkill -f "tests/sim/monitors.py" 2>/dev/null   # monitors run on the host (has pkill)
  log "stopped (staging serve left running — use '\''down'\'' to remove it)"
}

cmd_down(){
  cmd_stop
  log "removing $STAGING and dropping throwaway DB"
  $COMPOSE rm -sf serve-staging
  docker exec "$PG" psql -U prefect -c "DROP DATABASE IF EXISTS smarthub_staging;"
}

case "${1:-}" in
  up)     cmd_up ;;
  start)  cmd_start ;;
  report) cmd_report ;;
  stop)   cmd_stop ;;
  down)   cmd_down ;;
  *) echo "usage: $0 {up|start|report|stop|down}"; exit 1 ;;
esac
