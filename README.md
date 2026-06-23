# SmartHub Anton

## SmartHub Anton Prep
The prep folder repository contains two scripts for pulling lead data and visualizing it in a Streamlit dashboard.

### Files

```text
.
├── data_pull.py      # Pulls lead data from Redshift through an SSH tunnel
├── visualize.py      # Streamlit dashboard for analyzing the pulled data
├── leads.parquet     # Generated data file, created by data_pull.py
└── .env              # Local environment variables, not committed to git
```

### What the scripts do

#### `data_pull.py`

`data_pull.py` connects to Redshift through an SSH tunnel, runs a SQL query against `lead_pings`, loads the result into a pandas dataframe, and writes the data to:

```text
leads.parquet
```

The script uses a date range controlled by:

```python
min_created_at = "2026-06-07 00:00:00"
max_created_at = "2026-06-20 00:00:00"
```

Update these values before running the pull if you need a different date range.

#### `visualize.py`

`visualize.py` launches a Streamlit dashboard that reads:

```text
leads.parquet
```

## Setup

Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

## Environment variables

Create a local `.env` file in the repo root.

```bash
touch .env
```

Add the following variables:

```text
SSH_HOST=<ssh_host>
SSH_USER=<ssh_user>
SSH_PRIVATE_KEY_PATH=<path_to_private_key>

REDSHIFT_HOST=<redshift_host>
REDSHIFT_PORT=5439
REDSHIFT_DB=<redshift_database>
REDSHIFT_USER=<redshift_user>
REDSHIFT_PASSWORD=<redshift_password>
```


## Usage

### 1. Pull data

Run:

```bash
python data_pull.py
```

This creates or overwrites:

```text
leads.parquet
```

The dashboard expects this file to exist in the same directory as `visualize.py`.

### 2. Launch the dashboard

Run:

```bash
streamlit run visualize.py
```

Then open the local Streamlit URL printed in the terminal.

## Dashboard workflow

1. Use the sidebar to select `lead_type_id`.
2. Optionally select a `campaign_id`.
3. Select or deselect states.
4. Review the summary metrics at the top.
5. Review the aggregated table and raw data sample.
6. Use the plot sections for deeper analysis.

## Plot sections

### Plot Type 1

Use this for plotting any selected feature on the x-axis.

Inputs:

- x-axis feature
- one or more y-axis metrics
- optional legend division

Example:

```text
x-axis = num_auto_violations
y-axis = winrate
legend = state
```

### Plot Type 2 - Time Series

Use this for time-binned trend plots.

Inputs:

- frequency: `1 hr` or `1 day`
- one or more y-axis metrics
- optional legend division

The x-axis is always based on `created_at`.

### Plot Type 3 - Metric Value Series

Use this for binned plots based on dollar-value columns.

Inputs:

- x-axis metric: `profit`, `bid`, `payout`, or `revenue`
- bin width: `$0.50`, `$1`, `$2`, `$5`, or `$10`
- one or more y-axis metrics
- optional legend division

This behaves like Plot Type 2, except the x-axis uses numeric value bins instead of time bins.

## Notes

- `visualize.py` drops unused columns such as `lead_created_at` and `excluded` when loading data.
- Rows with `bid <= 0` are filtered out.
- Blank or missing `state` values are replaced with `NAvail`.
- `won` is converted from string values like `"true"` and `"false"` into `1` and `0`.
- `payout` is calculated as `won * bid`.
- `profit` is calculated as `rev - payout`.
- `cm` is calculated as `profit / rev`.
- `winrate` is calculated as `won / count` or as the mean of `won`, depending on context.


