#!/usr/bin/env bash
# SmartHub installer: validate prerequisites + .env, then bring up the
# Prefect stack (Postgres + server + worker).
#
#   ./install.sh            # validate and start
#   ./install.sh --check    # validate only (no docker up)
#   ./install.sh --down     # stop the stack and free the host ports
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.prefect.yml"
LOCAL_OVERRIDE="docker-compose.local.yml"
# Host ports the stack binds (free these on --down).
HOST_PORTS=(4200 8501)
REQUIRED_VARS=(
  SSH_HOST
  SSH_USER
  SSH_PRIVATE_KEY_PATH
  REDSHIFT_HOST
  REDSHIFT_DB
  REDSHIFT_USER
  REDSHIFT_PASSWORD
)

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
info()  { printf "  %s\n" "$*"; }

fail() { red "✗ $*"; exit 1; }

free_ports() {
  if ! command -v lsof >/dev/null 2>&1; then
    info "lsof not found; skipping explicit port free."
    return
  fi
  for p in "${HOST_PORTS[@]}"; do
    pids=$(lsof -ti "tcp:${p}" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      info "Freeing port ${p} (killing: ${pids})"
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
    fi
  done
}

# --- 1. Docker available -----------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "docker is not installed / not on PATH."
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  fail "docker compose (v2) or docker-compose (v1) is required."
fi
green "✓ docker + compose found"

daemon_up() { docker info >/dev/null 2>&1; }

# --- --down: stop the stack and free the host ports --------------------------
if [[ "${1:-}" == "--down" ]]; then
  if daemon_up; then
    info "Stopping the stack ..."
    # Include the local override + prod profile so every service (incl.
    # Watchtower) is torn down regardless of which mode brought it up.
    $COMPOSE -f "$COMPOSE_FILE" -f "$LOCAL_OVERRIDE" --profile prod \
      down --remove-orphans || true
  else
    info "Docker daemon not running — containers already stopped; freeing ports only."
  fi
  free_ports
  green "Stopped; ports ${HOST_PORTS[*]} freed."
  exit 0
fi

# --- 2. .env present ---------------------------------------------------------
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    fail ".env was missing — created one from .env.example. Fill it in, then re-run."
  fi
  fail ".env is missing and no .env.example to copy from."
fi
green "✓ .env present"

# Load .env (export simple KEY=VALUE pairs).
set -a
# shellcheck disable=SC1091
source .env
set +a

# --- 3. Required variables non-empty ----------------------------------------
missing=()
for var in "${REQUIRED_VARS[@]}"; do
  [[ -n "${!var:-}" ]] || missing+=("$var")
done
if (( ${#missing[@]} )); then
  red "✗ Missing required variables in .env:"
  for v in "${missing[@]}"; do info "- $v"; done
  exit 1
fi
green "✓ required variables set"

# --- 4. SSH key file exists --------------------------------------------------
key_path="${SSH_PRIVATE_KEY_PATH/#\~/$HOME}"
[[ -f "$key_path" ]] || fail "SSH key not found at SSH_PRIVATE_KEY_PATH: $key_path"
green "✓ SSH key found ($key_path)"

# --- 5. STORAGE_BACKEND valid (if set) --------------------------------------
backend="${STORAGE_BACKEND:-both}"
case "$backend" in
  duckdb|parquet|both) green "✓ STORAGE_BACKEND=$backend" ;;
  *) fail "STORAGE_BACKEND must be duckdb|parquet|both (got '$backend')." ;;
esac

green "All prerequisites OK."

if [[ "${1:-}" == "--check" ]]; then
  info "Validation only (--check); not starting Docker."
  exit 0
fi

# --- 6. Bring the stack up ---------------------------------------------------
daemon_up || fail "Docker daemon is not running. Start Docker / Rancher Desktop and retry."
green "✓ docker daemon running"

echo
# SMARTHUB_ENV decides where images come from:
#   local (default) -> BUILD from source (no pull, no Watchtower)
#   staging/prod    -> PULL from Docker Hub + run Watchtower auto-update
SMARTHUB_ENV="${SMARTHUB_ENV:-local}"
if [[ "$SMARTHUB_ENV" == "local" ]]; then
  info "SMARTHUB_ENV=local → building images from source (no pull, no Watchtower)."
  $COMPOSE -f "$COMPOSE_FILE" -f "$LOCAL_OVERRIDE" up -d --build
else
  info "SMARTHUB_ENV=$SMARTHUB_ENV → pulling images from Docker Hub + Watchtower."
  $COMPOSE -f "$COMPOSE_FILE" --profile prod up -d --pull always
fi

echo
green "Up ($SMARTHUB_ENV)."
info "Prefect UI: http://localhost:4200   Dashboard (Leads/Monitoring/Config): http://localhost:8500"
info "Logs:  docker logs prefect-worker -f"
