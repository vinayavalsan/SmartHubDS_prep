# SmartHub — Production Cutover (single-switch)

Everything per-environment is driven by **one variable you already have:
`SLACK_ENV_LABEL`**. Set it to `prod` and the stack automatically uses prod-only databases
and a fresh model registry — no other env edits needed.

> Use a **lowercase** value (`staging` / `prod`). The Slack code already uppercases it for
> display, so your alerts still read `STAGING` / `PROD`, while the database/prefix names stay
> conventional lowercase.

## What `SLACK_ENV_LABEL` controls

| Thing | `staging` | `prod` |
|---|---|---|
| App/dashboard tables (prediction log, config store, API keys) | DB `smarthub_staging` | DB `smarthub_prod` |
| MLflow | DB `mlflow_staging` | DB `mlflow_prod` |
| Model registry (same S3 bucket, different prefix) | `staging-models` | `prod-models` → **versions restart at v1.0.0** |
| Slack alert label | `STAGING` | `PROD` |

The S3 **bucket** stays the same; only the **prefix** changes, which is what makes the prod
registry start fresh at `v1.0.0`. Staging data is never touched — it lives in its own
DBs/prefix.

> One-time code change (already in the repo): `docker-compose.yaml` derives the DB URLs, the
> MLflow DB, and the S3 prefix from `SLACK_ENV_LABEL`. Pull the branch onto the box, or it's
> included when you deploy the new images.

## Cutover steps (on the box, `10.0.0.215`)

```bash
cd /home/ubuntu/smarthub-anton/staging/SmartHubDS_prep

# 1. create the fresh prod databases (tables auto-create on first start)
docker compose exec postgres psql -U prefect -c "CREATE DATABASE smarthub_prod;"
docker compose exec postgres psql -U prefect -c "CREATE DATABASE mlflow_prod;"

# 2. flip the single switch in .env  (use lowercase)
#    SLACK_ENV_LABEL=prod
#    (also set SMARTHUB_API_AUTH_ENABLED=true for prod)

# 3. recreate so the new env takes effect
docker compose up -d --force-recreate

# 4. verify everything points at prod
docker compose exec serve env | grep -E 'SLACK_ENV_LABEL|PREDICTION_LOG_DB_URL|CONFIG_DB_URL|S3_PREFIX|MLFLOW_TRACKING'
#    expect: smarthub_prod, prod-models, mlflow_prod
docker compose exec postgres psql -U prefect -d smarthub_prod -c "\dt"   # fresh, empty tables
curl -i http://localhost:8000/health?lead_type_id=6
```

After the next training run promotes a model, it lands in `prod-models/<type>/` as `v1.0.0`.
Issue the prod API key with `docker compose exec serve smarthub-apikey create --client prod
--note "prod" --expires-in-days 365`, and set the prod business values on the dashboard
Config page (env = prod).

> Note: if your current staging `.env` has `SLACK_ENV_LABEL=STAGING` (uppercase), change it to
> lowercase `staging` so staging's DB name is `smarthub_staging` (not `smarthub_STAGING`).
> Create `smarthub_staging` / `mlflow_staging` the same way if you keep running staging.

## Docker image version reset (v1.0.0)

Image version is separate from `SLACK_ENV_LABEL` (it's a release number, not an environment).
To make the images you pull start at `v1.0.0`:

1. Open the release PR into the deploy branch (`smarthub.etl.pipeline`) with the **`major`**
   label → `version-bump.yml` builds & pushes `serve/worker/dashboard` at **`v1.0.0`**
   (a major bump takes `0.x.y` → `1.0.0`).
2. On the box, pull that version: `./deploy.sh 1.0.0` (sets `IMAGE_TAG=v1.0.0` and brings the
   stack up). Pin `IMAGE_TAG=v1.0.0` in `.env` if you don't want Watchtower rolling it.

## Rollback

```bash
# set SLACK_ENV_LABEL back to staging in .env, then:
docker compose up -d --force-recreate
```

Prod's data stays in `smarthub_prod` / `mlflow_prod` / `prod-models`, so flipping back and
forth is clean and non-destructive.
