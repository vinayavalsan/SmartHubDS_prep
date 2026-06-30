#!/usr/bin/env bash
# SmartHub installer: validate prerequisites + .env, then bring up the
# Prefect stack (Postgres + server + worker).
#
#   ./install.sh            # validate and start
#   ./install.sh --check    # validate only (no docker up)
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.prefect.yml"
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
echo
info "Starting the Prefect stack ($COMPOSE_FILE) ..."
$COMPOSE -f "$COMPOSE_FILE" up --build -d

echo
green "Up. Prefect UI: http://localhost:4200"
info "Logs:  docker logs prefect-worker -f"
