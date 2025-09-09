# ImmoEliza – Airflow + Streamlit (local-first)

A lean, local-first pipeline for scraping Belgian listings, cleaning & analytics in DuckDB, model training, and a Streamlit dashboard.

---

## Quick start

```bash
# 0) Clone + enter
git clone https://github.com/<you>/immoeliza-airflow.git
cd immoeliza-airflow

# 1) Python venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .    # editable install of src/immoeliza

# 3) (optional) Local env vars
cp .env.example .env
# Edit IMMO_ANALYTICS_DB, SCRAPE_* options, etc.

# 4) Streamlit dashboard
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export IMMO_ANALYTICS_DB="analytics/immoeliza.duckdb"
streamlit run webui/app.py
```

- **Left sidebar button** “Rebuild analytics (clean → upsert)” runs the full analytics cleaning and writes into DuckDB with short‑lived connections (no locks).  
- Filters = City / Property type / Postal contains; download CSV of the current filtered slice.

---

## Airflow

### Initialize & login

```bash
export AIRFLOW_HOME="$PWD/.airflow"
airflow db migrate           # or: airflow db init
airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email you@example.com
```

Start services (two terminals):

```bash
# Terminal A
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
airflow webserver --port 8080

# Terminal B
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
airflow scheduler
```

Open http://localhost:8080 and login with the credentials you just created.

> **No DAGs showing?** See **Troubleshooting** below.

### CLI helpers

```bash
# See parsed environment
airflow info

# List DAGs parsed by scheduler
airflow dags list

# List tasks in our DAG
airflow tasks list immoeliza_pipeline --tree

# Trigger now
airflow dags trigger immoeliza_pipeline
```

---

## What the DAG does

Single pipeline **`immoeliza_pipeline`** (daily by default):

1. **Scrape** apartments & houses (two tasks).  
2. **Clean for analysis** → write Parquet and **upsert → DuckDB**.  
3. **Clean for training** (guard k€ vs €, fill m² where possible, strict sanity checks).  
4. **Train regression** (dual target: `price` and `price_per_m2`, KFold(5), log-target; metrics = RMSE on original scale).  
5. **Save model + metadata.json** in `models/` with timestamps.  
6. **Notify** (log / optional hook).

Streamlit reads DuckDB for current market view and lets you **rebuild analytics** ad‑hoc without DAG.

---

## One‑liners (manual)

```bash
# Rebuild analytics parquet (+ upsert to DuckDB)
python -c "from immoeliza.cleaning.analysis_clean import run as r; print(r())"

# Build training parquet
python -c "from immoeliza.cleaning.training_clean import run as r; print(r())"

# Train a model from latest training parquet
python - <<'PY'
from immoeliza.modeling.train_regression import train
import glob
p = sorted(glob.glob('data/processed/training/*/training.parquet'))[-1]
print(train(p, target='price'))          # or: target='price_per_m2'
PY
```

---

## Troubleshooting

**No DAGs in UI**  
- Ensure the scheduler is running.  
- Ensure your DAG file is in `airflow/dags/immoeliza_pipeline_dag.py` and ends with `.py`.  
- Set env for both webserver & scheduler:
  ```bash
  export AIRFLOW_HOME="$PWD/.airflow"
  export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  ```
- Verify parse via CLI: `airflow dags list`.  
- Check import errors: `airflow tasks list immoeliza_pipeline --tree`.  
- Install requirements **in the same venv** you run Airflow from.

**DuckDB lock**  
We use short‑lived connections everywhere. If you still hit a lock, exit Streamlit (or its “Rebuild” run), then retry.

**Weird rows (postal code as price, odd €/m²)**  
Cleaning now enforces: price must have currency cues or structured “price” field; if `price == postal_code` with no currency text nearby → set `price = NaN`. Sale guard: drop `price < 50k` for sale, enforce 10 ≤ m² ≤ 1000. `/m²` computed only when both price and m² valid.

---

## Project layout

```
immoeliza-airflow/
  airflow/dags/immoeliza_pipeline_dag.py
  analytics/immoeliza.duckdb
  data/{raw,processed,...}/
  models/ (joblib + metadata.json)
  src/immoeliza/...
  webui/app.py
  .env  .env.example  requirements.txt  Makefile
```

---

## License

MIT (or your choice).
