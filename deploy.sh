#!/usr/bin/env bash
# deploy.sh -- deploy the SmartHub stack by PULLING pre-built, versioned images
# from Docker Hub. The images are built + pushed by the version-bump CI on every
# release, so what runs here is byte-identical to what CI built and tagged -- no
# source rebuild on the box.
#
# Usage:
#   ./deploy.sh v0.1.5     # deploy a specific released version
#   ./deploy.sh 0.1.5      # the leading 'v' is optional
#   ./deploy.sh            # deploy the ':*-latest' images
#
# .env keys used (this folder):
#   IMAGE_REPO=<dockerhubuser>/smarthub   # namespace CI pushes to; REQUIRED unless
#                                         # it really is 'smarthub/smarthub'
#   SLACK_WEBHOOK_URL=...                  # deploy notification (optional)
#   DEPLOY_MODE=local|ec2 / PREFECT_UI_HOST=1.2.3.4  # override Prefect UI host
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.yaml"
HOST_NAME="$(hostname)"
APP_SERVICES="worker dashboard serve"

envval() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "\"'"; }

# ---- 1. which version to pull ---------------------------------------------
REQ="${1:-}"
if [ -n "$REQ" ]; then
  case "$REQ" in v*) TAG="$REQ" ;; *) TAG="v$REQ" ;; esac
else
  TAG="latest"
fi
export IMAGE_TAG="$TAG"

# ---- 2. which Docker Hub namespace ----------------------------------------
IMAGE_REPO="$(envval IMAGE_REPO)"; [ -z "$IMAGE_REPO" ] && IMAGE_REPO="smarthub/smarthub"
export IMAGE_REPO

# ---- Slack helper: slack <good|danger> <title> <detail> -------------------
slack() {
  local level="$1" title="$2" detail="$3" hook color when
  hook="$(envval SLACK_WEBHOOK_URL)"
  [ -z "${hook:-}" ] && return 0
  command -v curl >/dev/null 2>&1 || return 0
  case "$level" in danger) color="#e01e5a" ;; *) color="#2eb67d" ;; esac
  when="$(date -u +'%Y-%m-%d %H:%M:%SZ')"
  curl -sS -o /dev/null -X POST -H 'Content-Type: application/json' "$hook" --data @- <<JSON || true
{"attachments":[{"color":"$color","blocks":[
{"type":"section","text":{"type":"mrkdwn","text":"$title"}},
{"type":"section","fields":[
{"type":"mrkdwn","text":"*Image:*\n$IMAGE_REPO:*-$IMAGE_TAG"},
{"type":"mrkdwn","text":"*Host:*\n$HOST_NAME"},
{"type":"mrkdwn","text":"*When (UTC):*\n$when"}
]}]}]}
JSON
}
trap 'slack danger ":rotating_light: *SmartHub deploy FAILED*" "tag ${IMAGE_TAG} on ${HOST_NAME}"' ERR

# ---- 3. Prefect UI host (public IP on EC2 so a remote browser can reach it) -
detect_public_ip() {
  local t ip
  t="$(curl -s --max-time 2 -X PUT http://169.254.169.254/latest/api/token \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
  if [ -n "$t" ]; then
    ip="$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: $t" \
          http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
  else
    ip="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
  fi
  echo "$ip"
}
MODE="${DEPLOY_MODE:-auto}"
if [ "$MODE" = "auto" ]; then
  detect_public_ip | grep -qE '^[0-9]+\.[0-9]+\.' && MODE=ec2 || MODE=local
fi
if [ "$MODE" = "ec2" ]; then UI_HOST="${PREFECT_UI_HOST:-$(detect_public_ip)}"; else UI_HOST="localhost"; fi
export PREFECT_UI_API_URL="http://${UI_HOST}:4200/api"

# ---- 4. pull the released images, then start (NO build) --------------------
echo ">> deploying ${IMAGE_REPO}:*-${IMAGE_TAG}  (Prefect UI -> ${PREFECT_UI_API_URL})"
if ! docker compose -f "$COMPOSE_FILE" pull $APP_SERVICES; then
  echo "ERROR: could not pull ${IMAGE_REPO}:{worker,dashboard,serve}-${IMAGE_TAG}" >&2
  echo "       Check the tag exists and that IMAGE_REPO in .env matches the" >&2
  echo "       Docker Hub namespace CI pushes to (DOCKERHUB_USERNAME/smarthub)." >&2
  exit 1
fi
docker compose -f "$COMPOSE_FILE" up -d --no-build

# mlflow DB must exist or mlflow-ui crash-loops (idempotent).
docker exec prefect-postgres sh -c \
  "psql -U prefect -tc \"SELECT 1 FROM pg_database WHERE datname='mlflow'\" | grep -q 1 \
   || psql -U prefect -c 'CREATE DATABASE mlflow'" >/dev/null 2>&1 || true
docker restart smarthub-mlflow-ui >/dev/null 2>&1 || true

# wait for prefect-server health, then re-up so the worker's health-gate opens.
echo -n ">> waiting for prefect-server to be healthy "
for _ in $(seq 1 36); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' prefect-server 2>/dev/null || echo x)" = "healthy" ] \
    && { echo " -> healthy"; break; }
  echo -n "."; sleep 5
done
docker compose -f "$COMPOSE_FILE" up -d --no-build

echo
docker compose -f "$COMPOSE_FILE" ps
echo ">> deployed ${IMAGE_TAG}"
slack good ":rocket: *SmartHub deployed -- ${IMAGE_TAG}*" "pulled ${IMAGE_REPO}:*-${IMAGE_TAG}"
