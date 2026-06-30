#!/usr/bin/env bash
# Wire everything up at container start, then run the worker:
#   1. wait for the Prefect server API
#   2. create the work pool + queue (idempotent)
#   3. register deployments from prefect.yaml
#   4. start the worker on the pool/queue
set -euo pipefail

: "${PREFECT_API_URL:?PREFECT_API_URL must be set}"
POOL="${PREFECT_WORK_POOL:-smarthub-pool}"
# Space-separated; one worker serves all of these queues in the pool.
QUEUES="${PREFECT_WORK_QUEUES:-default features}"

cd /app

echo "Waiting for Prefect API at ${PREFECT_API_URL} ..."
until prefect work-pool ls >/dev/null 2>&1; do
  echo "  ...server not ready yet, retrying"
  sleep 3
done
echo "Prefect API is up."

# Idempotent: ignore 'already exists' errors.
prefect work-pool create "${POOL}" --type process 2>/dev/null || true

queue_args=()
for q in ${QUEUES}; do
  prefect work-queue create "${q}" --pool "${POOL}" 2>/dev/null || true
  queue_args+=(--work-queue "${q}")
done

echo "Registering deployments from prefect.yaml ..."
prefect --no-prompt deploy --all

echo "Starting worker on pool='${POOL}' queues='${QUEUES}' ..."
exec prefect worker start --pool "${POOL}" "${queue_args[@]}"
