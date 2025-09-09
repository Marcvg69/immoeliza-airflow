# Makefile – developer shortcuts
# Usage: `make help`

SHELL := /bin/bash
.ONESHELL:
.SILENT:

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
PIP    := $(BIN)/pip
STREAM := $(BIN)/streamlit
AIRF   := $(BIN)/airflow

DB ?= analytics/immoeliza.duckdb
export IMMO_ANALYTICS_DB := $(DB)
export PYTHONPATH := $(shell pwd)/src:$(PYTHONPATH)

AIRFLOW_VERSION ?= 2.9.3
PY_VER          ?= 3.11
AF_CONSTRAINTS  := https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW_VERSION)/constraints-$(PY_VER).txt

TRAIN_FILE := $(shell $(PY) - <<'PY' 2>/dev/null || true
import glob,sys; p=sorted(glob.glob('data/processed/training/*/training.parquet'))[-1] if glob.glob('data/processed/training/*/training.parquet') else ''
print(p)
PY
)

.PHONY: help
help:
	echo ""
	echo "Targets:"
	echo "  dev               Create venv + install deps (+ project editable)"
	echo "  install-airflow   Install Apache Airflow with constraints"
	echo "  analytics         Rebuild processed analytics + upsert to DuckDB"
	echo "  training          Build training parquet"
	echo "  train-price       Train (target=price)"
	echo "  train-ppm2        Train (target=price_per_m2)"
	echo "  ui                Run Streamlit dashboard"
	echo "  airflow-init      Init Airflow DB + create admin"
	echo "  airflow-web       Start Airflow webserver"
	echo "  airflow-sched     Start Airflow scheduler"
	echo "  airflow-up        Webserver + Scheduler"
	echo "  airflow-down      Kill Airflow procs"
	echo "  check-db          Quick table counts in DuckDB"
	echo "  clean             Remove caches/tmp (keeps models & analytics)"
	echo ""

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) -V >/dev/null 2>&1 || true

.PHONY: dev
dev: $(VENV)
	$(PIP) install -U pip wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-airflow
install-airflow: $(VENV)
	$(PIP) install "apache-airflow==$(AIRFLOW_VERSION)" --constraint "$(AF_CONSTRAINTS)"

.PHONY: analytics
analytics:
	$(PY) - <<'PY'
from immoeliza.cleaning.analysis_clean import run
print(run())
PY

.PHONY: training
training:
	$(PY) - <<'PY'
from immoeliza.cleaning.training_clean import run
print(run())
PY

.PHONY: train-price
train-price:
	@if [ -z "$(TRAIN_FILE)" ]; then echo "No training parquet found. Run 'make training' first."; exit 1; fi
	$(PY) - <<'PY'
from immoeliza.modeling.train import train
print(train("$(TRAIN_FILE)", target="price"))
PY

.PHONY: train-ppm2
train-ppm2:
	@if [ -z "$(TRAIN_FILE)" ]; then echo "No training parquet found. Run 'make training' first."; exit 1; fi
	$(PY) - <<'PY'
from immoeliza.modeling.train import train
print(train("$(TRAIN_FILE)", target="price_per_m2"))
PY

.PHONY: ui
ui:
	@echo "DuckDB: $(IMMO_ANALYTICS_DB)"
	$(STREAM) run webui/app.py

.PHONY: airflow-init
airflow-init:
	@[ -d airflow ] || mkdir -p airflow/dags
	AIRFLOW_HOME="$$PWD/airflow" $(AIRF) db init
	AIRFLOW_HOME="$$PWD/airflow" $(AIRF) users create --role Admin --username admin --password admin --firstname a --lastname b --email x@y.z || true
	@echo "Airflow home: $$PWD/airflow"

.PHONY: airflow-web
airflow-web:
	AIRFLOW_HOME="$$PWD/airflow" $(AIRF) webserver -p 8080

.PHONY: airflow-sched
airflow-sched:
	AIRFLOW_HOME="$$PWD/airflow" $(AIRF) scheduler

.PHONY: airflow-up
airflow-up:
	$(MAKE) -j2 airflow-web airflow-sched

.PHONY: airflow-down
airflow-down:
	pkill -f "airflow webserver" || true
	pkill -f "airflow scheduler" || true

.PHONY: check-db
check-db:
	$(PY) - <<'PY'
import duckdb,os
db=os.environ.get("IMMO_ANALYTICS_DB","analytics/immoeliza.duckdb")
con=duckdb.connect(db,read_only=True)
for t in ("listings_latest","listings_history","market_daily_summary"):
  try:
    n=con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t}: {n}")
  except Exception as e:
    print(f"{t}: missing ({e})")
con.close()
PY

.PHONY: clean
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name "*.pyc" -delete || true
	rm -rf .pytest_cache .mypy_cache || true
	echo "Cleaned."
