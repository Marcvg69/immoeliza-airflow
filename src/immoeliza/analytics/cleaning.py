from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

from ..config import settings
from ..io_paths import raw_path, today_partition
from ..scraping.common import logger

Kind = Literal["apartments","houses"]

@dataclass
class Paths:
    analytics_db: Path = Path(settings.analytics_db)
    training_root: Path = Path(settings.data_root) / "training"
    snapshot_date: str = date.today().isoformat()

def _read_details(kind: Kind) -> pd.DataFrame:
    p = Path(raw_path(f"{kind}_details"))
    if not p.exists():
        raise FileNotFoundError(f"Missing {kind} details parquet at {p}")
    df = pd.read_parquet(p)
    df["kind"] = kind
    return df

_num = re.compile(r"[\d\.,]+")

def _to_int(x):
    if pd.isna(x): return np.nan
    m = _num.search(str(x))
    if not m: return np.nan
    s = m.group(0).replace('.', '').replace(',', '.')
    try: return int(float(s))
    except: return np.nan

def _to_float(x):
    if pd.isna(x): return np.nan
    m = _num.search(str(x))
    if not m: return np.nan
    s = m.group(0).replace('.', '').replace(',', '.')
    try: return float(s)
    except: return np.nan

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Best-effort column harmonization from legacy scraper outputs
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    out = pd.DataFrame()
    out["url"] = df[pick("url")] if pick("url") else np.nan
    out["title"] = df[pick("title","name","heading")] if pick("title","name","heading") else ""
    out["price"] = df[pick("price","amount")] if pick("price","amount") else np.nan
    out["city"] = df[pick("city","town","locality")] if pick("city","town","locality") else ""
    out["postal_code"] = df[pick("postalcode","postal_code","zip","zipcode")] if pick("postalcode","postal_code","zip","zipcode") else ""
    out["property_type"] = df[pick("type","property_type","category")] if pick("type","property_type","category") else ""
    out["bedrooms"] = df[pick("bedrooms","beds")] if pick("bedrooms","beds") else np.nan
    out["bathrooms"] = df[pick("bathrooms","baths")] if pick("bathrooms","baths") else np.nan
    out["surface_m2"] = df[pick("surface","area","living_area","size")] if pick("surface","area","living_area","size") else np.nan
    out["year_built"] = df[pick("year","built","construction_year")] if pick("year","built","construction_year") else np.nan
    out["energy_label"] = df[pick("epc","energy","energy_label")] if pick("epc","energy","energy_label") else ""
    # numeric coercion
    out["price"] = out["price"].apply(_to_int)
    out["bedrooms"] = out["bedrooms"].apply(_to_int)
    out["bathrooms"] = out["bathrooms"].apply(_to_int)
    out["surface_m2"] = out["surface_m2"].apply(_to_float)
    out["year_built"] = out["year_built"].apply(_to_int)
    # engineered
    out["price_per_m2"] = out["price"] / out["surface_m2"]
    out["snapshot_date"] = date.today().isoformat()
    return out

def run_for_analysis() -> str:
    p = Paths()
    a = _read_details("apartments")
    h = _read_details("houses")
    df = pd.concat([a,h], ignore_index=True)
    df_norm = normalize(df).dropna(subset=["price"]).reset_index(drop=True)

    p.analytics_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p.analytics_db))
    con.execute("""
        CREATE TABLE IF NOT EXISTS listings_history AS
        SELECT * FROM (SELECT 0 AS dummy) WHERE 1=0;
    """)
    con.execute("CREATE TABLE IF NOT EXISTS listings_latest AS SELECT * FROM listings_history WHERE 1=0;")
    # replace latest
    con.execute("DELETE FROM listings_latest;")
    con.execute("INSERT INTO listings_latest SELECT * FROM df_norm")
    # append to history with snapshot_date (already present)
    con.execute("INSERT INTO listings_history SELECT * FROM df_norm")
    # daily summary
    con.execute("""
        CREATE TABLE IF NOT EXISTS market_daily_summary AS
        SELECT * FROM (SELECT 0 AS dummy) WHERE 1=0;
    """)
    con.execute("""
        INSERT INTO market_daily_summary
        SELECT snapshot_date, city, postal_code, property_type,
               COUNT(*) AS n_listings,
               AVG(price) AS avg_price,
               MEDIAN(price) AS med_price,
               AVG(price_per_m2) AS avg_ppm2
        FROM df_norm
        GROUP BY 1,2,3,4
    """)
    con.close()
    logger.info("📚 Wrote analytics into %s", p.analytics_db)
    return str(p.analytics_db)

def run_for_training() -> str:
    """Produce a clean ML dataset with features + target price."""
    p = Paths()
    a = _read_details("apartments")
    h = _read_details("houses")
    df = pd.concat([a,h], ignore_index=True)
    df = normalize(df)
    # Keep reasonable rows for modeling
    df = df.dropna(subset=["price","surface_m2"]).query("price > 10000 and surface_m2 > 10")
    # select features
    train = df[[
        "price","surface_m2","bedrooms","bathrooms","postal_code","city","property_type","year_built","energy_label"
    ]].copy()
    train["bedrooms"] = train["bedrooms"].fillna(0)
    train["bathrooms"] = train["bathrooms"].fillna(0)
    # save partitioned by date
    out_dir = p.training_root / p.snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "training.parquet"
    train.to_parquet(out, index=False)
    logger.info("🧹 Training dataset saved -> %s (%d rows)", out, len(train))
    return str(out)
