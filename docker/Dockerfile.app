FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SMARTHUB_ROOT=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

COPY README.md ./

EXPOSE 8501

# Default: serve the leads dashboard. Override the command to run the
# monitoring dashboard or the data pull, e.g.:
#   docker run ... smarthub-pull --min-created-at "..." --max-created-at "..."
#   docker run ... streamlit run src/smarthub/dashboards/monitoring_app.py \
#       --server.address 0.0.0.0
CMD ["streamlit", "run", "src/smarthub/dashboards/app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501"]
