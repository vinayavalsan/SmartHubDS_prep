# SmartHub Anton

Data-science toolkit for SmartHub / Anton: pull lead data from Redshift and
explore it through Streamlit dashboards. For the business domain (what the data
means and what Anton is solving), see [CONTEXT.md](./CONTEXT.md).

## Project layout

```text
.
├── src/smarthub/                 # installable package (src layout)
│   ├── config.py                 # env-driven, validated settings
│   ├── cli.py                    # argument parsing
│   ├── paths.py                  # project-root path resolution
│   ├── logging_utils.py          # logging setup
│   ├── io.py                     # data loading/saving (friendly errors)
│   ├── transforms.py             # shared metric definitions (single source of truth)
│   ├── data_pull.py              # Redshift -> parquet pull
│   └── dashboards/
│       ├── leads_app.py          # raw lead-ping dashboard
│       └── monitoring_app.py     # DS performance dashboard
├── tests/                        # pytest unit tests for transforms & config
├── data/                         # input data (etl/sample_data.csv) + leads.parquet
├── pyproject.toml                # packaging, deps, scripts, pytest config
├── requirements.txt              # runtime deps
├── Dockerfile                    # container for pull + dashboards
└── .env.example                  # copy to .env and fill in
```

> The legacy `prep/` and `src/monitoring/` scripts are superseded by
> `src/smarthub/` and can be deleted whenever convenient.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # editable install + dev tools (pytest, flake8)
```

Then create your `.env`:

```bash
cp .env.example .env        # fill in SSH + Redshift credentials
```

## Usage

### 1. Pull data

The date range is now passed on the command line (no editing source):

```bash
smarthub-pull \
    --min-created-at "2026-06-07 00:00:00" \
    --max-created-at "2026-06-20 00:00:00"
# or:  python -m smarthub.data_pull --min-created-at ... --max-created-at ...
```

This writes `data/leads.parquet` (path resolved from the project root, so it
works regardless of where you run it). Use `--output` to override.

### 2. Launch a dashboard

```bash
streamlit run src/smarthub/dashboards/leads_app.py        # lead-ping explorer
streamlit run src/smarthub/dashboards/monitoring_app.py   # DS performance
```

## Testing & linting

```bash
pytest          # unit tests for metric math and config validation
flake8          # style (max line length 88)
```

## Docker

```bash
docker build -t smarthub .
# leads dashboard (default):
docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/app/data" smarthub
# data pull:
docker run --rm --env-file .env -v "$PWD/data:/app/data" smarthub \
    smarthub-pull --min-created-at "2026-06-07 00:00:00" \
                  --max-created-at "2026-06-20 00:00:00"
```

## What the dashboards show

The dashboards visualise win rate, contribution margin, profit and revenue
across price points, time, states and campaigns — the "find the shelves"
analysis described in [CONTEXT.md](./CONTEXT.md). Metric definitions live in one
place (`transforms.py`) so the two dashboards stay consistent.

### Open item

Expected revenue lives in a **separate table**, not in the `lead_pings` table
that `data_pull.py` currently queries. To model the bid ceiling for Anton, that
table will need to be joined into the pull. See CONTEXT.md §4.
