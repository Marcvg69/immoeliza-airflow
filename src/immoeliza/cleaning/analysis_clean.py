"""
Build processed analytics parquet and upsert short-lived tables into DuckDB.

- Loads latest *_details.parquet for apartments + houses under data/raw/**.
- Normalises/cleans into a single DataFrame with canonical columns.
- Writes a daily snapshot to data/processed/analysis/YYYY-MM-DD/listings.parquet.
- Upserts into DuckDB (tables: listings_latest, listings_history, market_daily_summary),
  auto-migrating tables to include any missing columns.

Public API:
    - run() -> str
    - normalize(df) -> pd.DataFrame
    - postal_to_region(postal_code) -> Optional[str]
"""

from __future__ import annotations

import os
import re
import time
import logging
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("immoeliza.scraping")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ----------------------------- Config helpers ----------------------------- #

def _data_root() -> Path:
    return Path(os.environ.get("IMMO_DATA_ROOT", "data")).resolve()

def _analytics_db_path() -> Path:
    return Path(os.environ.get("IMMO_ANALYTICS_DB", "analytics/immoeliza.duckdb")).resolve()

def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")

def _find_latest_details_parquet(kind: str) -> Optional[Path]:
    """Find the latest {kind}_details.parquet under data/raw/**."""
    assert kind in {"apartments", "houses"}
    root = _data_root() / "raw"
    if not root.exists():
        return None
    cands = list(root.rglob(f"{kind}_details.parquet"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _processed_analytics_out() -> Path:
    out_dir = _data_root() / "processed" / "analysis" / _today_str()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "listings.parquet"


# ----------------------------- Cleaning utilities ----------------------------- #

_NUM_RE = re.compile(r"[\d\.]+")

def _sget(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Series-get: returns the first existing column, else a None-filled Series."""
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series([None] * len(df), index=df.index)

def _to_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).replace(",", "")
    m = _NUM_RE.findall(s)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    f = _to_float(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

def _clean_city(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    return s or None

def _clean_type(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().title()
    mapping = {
        "Apt": "Apartment",
        "Appartement": "Apartment",
        "Studio": "Apartment",
        "Flat": "Apartment",
        "House": "Residence",
        "Townhouse": "Residence",
        "Residence": "Residence",
        "Villa": "Villa",
        "Land": "Land",
        "Duplex": "Apartment",
    }
    return mapping.get(s, s)

def postal_to_region(postal: Optional[str | int]) -> Optional[str]:
    try:
        pc = int(str(postal)[:4])
    except Exception:
        return None
    if 1000 <= pc < 1300:
        return "Brussels"
    if 1300 <= pc < 1500 or 3000 <= pc < 4000:
        return "Flanders"
    if 2000 <= pc < 3000 or 8000 <= pc < 10000:
        return "Flanders"
    if 4000 <= pc < 8000:
        return "Wallonia"
    return None

def _infer_is_sale(row: pd.Series) -> Optional[bool]:
    url = str(row.get("url") or "")
    tx  = str(row.get("transaction") or "")
    if "/for-sale/" in url or "sale" in tx.lower():
        return True
    if "/for-rent/" in url or "rent" in tx.lower():
        return False
    return None

CANON_COLS: List[str] = [
    "snapshot_date", "url", "title",
    "city", "postal_code", "region",
    "property_type",
    "price", "surface_m2", "price_per_m2",
    "bedrooms", "bathrooms",
    "year_built", "energy_label",
    "is_sale",
]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with canonical columns and types."""
    # snapshot_date: safe whether column exists or not
    if "snapshot_date" in df.columns:
        sd = pd.to_datetime(df["snapshot_date"], errors="coerce")
        snap = sd.dt.date
    else:
        snap = pd.Series([date.today()] * len(df), index=df.index)

    url   = _sget(df, "url")
    title = _sget(df, "title")

    city = _sget(df, "city", "location", "cityName").map(_clean_city)

    postal_raw = _sget(df, "postal_code", "postalCode")
    postal = postal_raw.astype(str).str.extract(r"(\d{4})", expand=False)
    region = postal.map(postal_to_region)

    ptype = _sget(df, "property_type", "type").map(_clean_type)

    price      = _sget(df, "price", "price_eur").map(_to_float)
    surface_m2 = _sget(df, "surface_m2", "surface", "area").map(_to_float)

    bedrooms   = _sget(df, "bedrooms", "beds").map(_to_int)
    bathrooms  = _sget(df, "bathrooms", "baths").map(_to_int)
    year_built = _sget(df, "year_built", "build_year").map(_to_int)

    energy_raw = _sget(df, "energy_label", "epc")
    energy = energy_raw.astype("string").str.strip().str.upper()
    energy = energy.where(energy_raw.notna(), None)

    if "is_sale" in df.columns:
        is_sale = df["is_sale"]
    else:
        is_sale = df.apply(_infer_is_sale, axis=1) if len(df) else pd.Series([], dtype="boolean")

    # derived
    ppm = (price / surface_m2).where((price.notna()) & (surface_m2.notna()) & (surface_m2 > 0))

    out = pd.DataFrame({
        "snapshot_date": snap,
        "url": url,
        "title": title,
        "city": city,
        "postal_code": postal,
        "region": region,
        "property_type": ptype,
        "price": price,
        "surface_m2": surface_m2,
        "price_per_m2": ppm,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "energy_label": energy,
        "is_sale": is_sale,
    }, columns=CANON_COLS)

    # Cast best-effort types
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.date
    for c in ["price", "surface_m2", "price_per_m2"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["bedrooms", "bathrooms", "year_built"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out


# ----------------------------- IO + DuckDB upsert ----------------------------- #

def _load_both_details() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for kind in ("apartments", "houses"):
        p = _find_latest_details_parquet(kind)
        if p and p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame(columns=CANON_COLS)
    return pd.concat(parts, ignore_index=True, sort=False)

def _ensure_duckdb_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create or migrate target tables so they always include our canonical columns."""
    # Desired schemas
    latest_schema: Dict[str, str] = {
        "snapshot_date": "DATE",
        "url": "TEXT",
        "title": "TEXT",
        "city": "TEXT",
        "postal_code": "VARCHAR",
        "region": "VARCHAR",
        "property_type": "VARCHAR",
        "price": "DOUBLE",
        "surface_m2": "DOUBLE",
        "price_per_m2": "DOUBLE",
        "bedrooms": "INTEGER",
        "bathrooms": "INTEGER",
        "year_built": "INTEGER",
        "energy_label": "VARCHAR",
        "is_sale": "BOOLEAN",
    }
    history_schema = latest_schema.copy()
    summary_schema: Dict[str, str] = {
        "snapshot_date": "DATE",
        "city": "TEXT",
        "property_type": "TEXT",
        "n": "INT",
        "median_price_per_m2": "DOUBLE",
    }

    # Create if not exists
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS listings_latest (
            {", ".join(f"{k} {v}" for k, v in latest_schema.items())}
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS listings_history AS
        SELECT * FROM listings_latest WHERE 1=0;
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS market_daily_summary (
            {", ".join(f"{k} {v}" for k, v in summary_schema.items())}
        );
    """)

    # Migrate: add any missing columns
    def _migrate(table: str, spec: Dict[str, str]) -> None:
        cols = con.execute(f"PRAGMA table_info('{table}')").df()["name"].tolist()
        for col, sqlt in spec.items():
            if col not in cols:
                logger.info("🛠️ DuckDB: ALTER TABLE %s ADD COLUMN %s %s", table, col, sqlt)
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {sqlt}')

    _migrate("listings_latest", latest_schema)
    _migrate("listings_history", history_schema)
    _migrate("market_daily_summary", summary_schema)

def _upsert_with_retry(df: pd.DataFrame, adb: Path, retries: int = 6, wait: float = 0.6) -> None:
    last_err: Optional[BaseException] = None
    for _ in range(retries):
        try:
            adb.parent.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(str(adb))
            try:
                _ensure_duckdb_schema(con)

                # Ensure column order + types
                tmp = df[CANON_COLS].copy()
                tmp["is_sale"] = tmp["is_sale"].astype("boolean")

                con.register("df", tmp)

                # Replace today's slice in listings_latest
                con.execute("DELETE FROM listings_latest WHERE snapshot_date = ?", [_today_str()])
                con.execute(f"""
                    INSERT INTO listings_latest ({", ".join(CANON_COLS)})
                    SELECT {", ".join(CANON_COLS)} FROM df;
                """)

                # Append to history
                con.execute(f"""
                    INSERT INTO listings_history ({", ".join(CANON_COLS)})
                    SELECT {", ".join(CANON_COLS)} FROM df;
                """)

                # Recompute today's summary
                con.execute("DELETE FROM market_daily_summary WHERE snapshot_date = ?", [_today_str()])
                con.execute("""
                    INSERT INTO market_daily_summary
                    SELECT
                        snapshot_date,
                        COALESCE(city, '') as city,
                        COALESCE(property_type, '') as property_type,
                        COUNT(*) as n,
                        median(price_per_m2) as median_price_per_m2
                    FROM listings_latest
                    WHERE snapshot_date = ?
                    GROUP BY 1,2,3
                    ORDER BY 1,2,3
                """, [_today_str()])
            finally:
                con.close()
            return
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "could not set lock" in msg or "conflicting lock" in msg:
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("DuckDB upsert failed after retries") from last_err


# ----------------------------- Public entrypoint ----------------------------- #

def run() -> str:
    """Build processed analytics parquet and upsert into DuckDB. Return parquet path."""
    raw_df = _load_both_details()
    df = normalize(raw_df)

    out_parquet = _processed_analytics_out()
    df.to_parquet(out_parquet, index=False)

    _upsert_with_retry(df, _analytics_db_path())
    logger.info("📚 Wrote analytics into %s", _analytics_db_path())
    return str(out_parquet)


__all__ = ["run", "normalize", "postal_to_region"]
