# Immo-Eliza Airflow

Pipeline to scrape, clean, analyse and model Belgian real estate data.

## Quick start
1. `python3 -m venv .venv && source .venv/bin/activate`
2. Install Airflow with constraints (see project notes) then: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and adjust paths if needed.
4. `airflow db init && airflow webserver & airflow scheduler &`

