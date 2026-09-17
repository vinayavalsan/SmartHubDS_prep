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
# Auth is ON on the staging serve. You do NOT manage a key: the script mints a
# throwaway key into the throwaway DB each run and uses it. (Override by
# exporting API_KEY=shk_... to reuse your own.)
#
# Env overrides: DAYS, BURST_PROB, SPEED, SEGMENT_MINUTES, SERVE_WORKERS,
#                SLACK_WEBHOOK, API_KEY

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
API_KEY=${API_KEY:-}                          # bearer key; auto-minted by mint_key() if empty
# Supervisor alerts/heartbeats post to SLACK_WEBHOOK. Keep the secret OUT of
# git: set SLACK_WEBHOOK=<url> in .env (gitignored) -- the sim-supervisor reads
# it via env_file. You can also just export SLACK_WEBHOOK in your shell.
SLACK_WEBHOOK=${SLACK_WEBHOOK:-}

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

mint_key(){
  # Auto-mint a throwaway bearer key straight into the throwaway smarthub_staging
  # DB, so nothing secret is ever committed and you never handle a key by hand.
  # If you exported API_KEY yourself, we respect it and skip minting.
  if [ -n "$API_KEY" ]; then
    log "using API_KEY from environment (${API_KEY:0:12}…)"; return 0
  fi
  log "minting a throwaway API key into smarthub_staging ..."
  API_KEY=$(docker exec \
      -e SMARTHUB_PREDICTION_LOG_DB_URL="postgresql+psycopg2://prefect:prefect@postgres:5432/smarthub_staging" \
      "$WORKER" python -m smarthub.server.manage_keys create --client sim-test \
      --note "sim replay/load test (throwaway)" 2>/dev/null \
    | sed -n 's/^api_key: *//p' | head -1)
  if [ -z "$API_KEY" ]; then
    log "ERROR: could not mint an API key (is $WORKER up and smarthub importable?)"
    return 1
  fi
  log "minted key ${API_KEY:0:12}… (client=sim-test, lives only in the throwaway DB)"
}

wait_health(){
  log "waiting for $STAGING to load the model ..."
  for i in $(seq 1 40); do
    if docker exec "$WORKER" python -c \
        "import urllib.request,json,sys; \
r=json.load(urllib.request.urlopen('$URL/health?lead_type_id=6', timeout=8)); \
sys.exit(0 if r.get('model_loaded') else 1)" 2>/dev/null; then
      log "staging serve healthy (model_loaded=true)"; return 0
    fi
    sleep 5
  done
  log "ERROR: staging serve did not become healthy — check: docker logs $STAGING"; return 1
}

show_model(){
  # Report WHICH promoted model each lead type resolves to, from the production
  # serving pointer (the S3/MinIO current.json): promoted version, the UTC time
  # it was promoted, the resolved artifact, and whether it changed since the
  # last run. Runs inside the serve container (it has the production-storage env).
  cp tests/sim/model_info.py data/sim/ 2>/dev/null   # ensure helper is on the mount
  log "resolved model(s) from production store (S3/MinIO current.json):"
  docker exec "$STAGING" python /app/data/sim/model_info.py 2>/dev/null \
    || log "could not read model info (serve down, or prod storage not configured?)"
}

cmd_up(){
  ensure_db
  mint_key || exit 1        # key must exist in the DB before the serve caches it
  log "bringing up $STAGING (SERVE_WORKERS=$SERVE_WORKERS)"
  # --force-recreate: start with a clean key cache so the just-minted key is
  # loaded on the serve's first authenticated request (no 60s cache-TTL wait).
  $COMPOSE up -d --force-recreate serve-staging || exit 1
  wait_health || exit 1
  show_model               # report which model artifact is actually loaded
}

cmd_start(){
  cp tests/sim/*.py data/sim/ 2>/dev/null      # refresh harness copies for the mount
  cmd_up || exit 1
  mkdir -p data/sim/ledgers
  # Fresh run: move any prior ledgers + checkpoint aside so `report` reflects
  # ONLY this run's traffic (nothing deleted — kept under ledgers/archive/<ts>).
  if ls data/sim/ledgers/*.jsonl >/dev/null 2>&1 \
       || [ -f data/sim/ledgers/supervisor.ckpt.json ]; then
    arch="data/sim/ledgers/archive/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$arch"
    mv data/sim/ledgers/*.jsonl "$arch"/ 2>/dev/null
    mv data/sim/ledgers/supervisor.ckpt.json "$arch"/ 2>/dev/null
    log "archived previous ledgers -> $arch"
  fi
  # host-side monitors (stdlib python3; samples the staging serve + postgres + disk)
  pkill -f "monitors.py" 2>/dev/null
  nohup python3 tests/sim/monitors.py --interval 30 --out data/sim/health.csv \
      --serve-container "$STAGING" --pg-container "$PG" --pg-db smarthub_staging \
      --disk-path data/sim \
      > data/sim/monitors.log 2>&1 &
  log "monitors started -> data/sim/health.csv"
  # the supervised replay as a MANAGED container (not `docker exec -d`, which
  # dies when prefect-worker is recreated). It survives daemon/host restarts and
  # resumes from the checkpoint; config is passed via SIM_* env for the service.
  export SIM_URL="$URL" SIM_DATA="$DATA" SIM_LEDGER_DIR="$LEDGERS" \
         SIM_DAYS="$DAYS" SIM_SEGMENT_MINUTES="$SEGMENT_MINUTES" \
         SIM_SPEED="$SPEED" SIM_BURST_PROB="$BURST_PROB" \
         SIM_API_KEY="${API_KEY:-}"
  $COMPOSE up -d sim-supervisor || exit 1
  log "replay started (managed): days=$DAYS speed=${SPEED}x burst_prob=$BURST_PROB -> $URL"
  echo
  echo "  watch:   docker logs -f smarthub-sim-supervisor"
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
  $COMPOSE stop sim-supervisor 2>/dev/null        # stop the managed replay container
  pkill -f "tests/sim/monitors.py" 2>/dev/null    # monitors run on the host (has pkill)
  log "stopped (staging serve left running — use '\''down'\'' to remove it)"
}

cmd_down(){
  cmd_stop
  log "removing staging serve + supervisor and dropping throwaway DB"
  $COMPOSE rm -sf serve-staging sim-supervisor
  docker exec "$PG" psql -U prefect -c "DROP DATABASE IF EXISTS smarthub_staging;"
}

cmd_test_alert(){
  local hook="${SLACK_WEBHOOK:-}"
  if [ -z "$hook" ] && [ -f .env ]; then          # fall back to .env (gitignored)
    hook=$(sed -n 's/^SLACK_WEBHOOK=//p' .env | tail -1 | tr -d '"'"'"'"')
  fi
  if [ -z "$hook" ]; then
    log "no SLACK_WEBHOOK -- set it in .env or export it"; return 1
  fi
  local msg=":wrench: SmartHub replay: test alert from $(hostname) at \
$(date -u +%Y-%m-%dT%H:%M:%SZ) — Slack alerts are wired."
  if command -v curl >/dev/null 2>&1; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
      -H 'Content-Type: application/json' --data "{\"text\":\"$msg\"}" "$hook")
    log "posted test alert -> Slack (HTTP $code; 200 = delivered)"
  else
    log "curl not found; posting via the worker container"
    docker exec -e HOOK="$hook" -e MSG="$msg" "$WORKER" python -c \
      "import os,json,urllib.request; \
r=urllib.request.urlopen(urllib.request.Request(os.environ['HOOK'], \
data=json.dumps({'text':os.environ['MSG']}).encode(), \
headers={'Content-Type':'application/json'}), timeout=10); \
print('HTTP', r.status)"
  fi
}

case "${1:-}" in
  up)     cmd_up ;;
  start)  cmd_start ;;
  report) cmd_report ;;
  stop)   cmd_stop ;;
  down)   cmd_down ;;
  test)   cmd_test_alert ;;
  *) echo "usage: $0 {up|start|report|stop|down|test}"; exit 1 ;;
esac
