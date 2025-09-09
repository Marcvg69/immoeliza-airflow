# src/immoeliza/analytics/to_duckdb.py
from __future__ import annotations

from pathlib import Path
import logging
import duckdb
import pandas as pd

logger = logging.getLogger("immoeliza.scraping")

SCHEMA_LATEST = """
CREATE TABLE IF NOT EXISTS listings_latest (
    snapshot_date DATE,
    url           TEXT,
    title         TEXT,
    city          TEXT,
    postal_code   VARCHAR,
    region        VARCHAR,
    property_type VARCHAR,
    price         DOUBLE,
    surface_m2    DOUBLE,
    bedrooms      INTEGER,
    bathrooms     INTEGER,
    year_built    INTEGER,
    energy_label  VARCHAR,
    price_per_m2  DOUBLE
)
"""

SCHEMA_HISTORY = """
CREATE TABLE IF NOT EXISTS listings_history AS
SELECT * FROM listings_latest WHERE 1=0
"""

SCHEMA_SUMMARY = """
CREATE TABLE IF NOT EXISTS market_daily_summary (
    snapshot_date DATE,
    city          TEXT,
    property_type TEXT,
    n             INT,
    median_price_per_m2 DOUBLE
)
"""

def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_LATEST)
    con.execute(SCHEMA_HISTORY)
    con.execute(SCHEMA_SUMMARY)


def upsert_listings_with_summary(df: pd.DataFrame, db_path: str | Path) -> None:
    """
    Upsert df into listings_latest (by url), append to listings_history,
    and insert/merge a daily summary into market_daily_summary.
    """
    if df.empty:
        logger.info("to_duckdb: dataframe is empty; nothing to upsert.")
        return

    db_path = str(db_path)
    con = duckdb.connect(db_path)
    try:
        _ensure_schema(con)
        con.register("df", df)

        # normalize column order
        cols = [
            "snapshot_date","url","title","city","postal_code","region",
            "property_type","price","surface_m2","bedrooms","bathrooms",
            "year_built","energy_label","price_per_m2"
        ]
        con.execute("CREATE OR REPLACE TEMP VIEW _df AS SELECT " + ",".join(cols) + " FROM df")

        # upsert latest on url
        con.execute("DELETE FROM listings_latest WHERE url IN (SELECT url FROM _df)")
        con.execute("INSERT INTO listings_latest SELECT * FROM _df")

        # append to history
        con.execute("INSERT INTO listings_history SELECT * FROM _df")

        # daily summary (median €/m²)
        con.execute("""
            INSERT INTO market_daily_summary
            SELECT
                snapshot_date,
                city,
                property_type,
                COUNT(*) AS n,
                median(price_per_m2) AS median_price_per_m2
            FROM _df
            WHERE price_per_m2 IS NOT NULL
            GROUP BY snapshot_date, city, property_type
        """)
        logger.info("to_duckdb: upserted %d rows; summary updated.", len(df))
    finally:
        con.close()
