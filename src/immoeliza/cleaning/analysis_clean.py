# src/immoeliza/cleaning/analysis_clean.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

LOG_PREFIX = "immoeliza.scraping"
DATA_ROOT = Path(os.getenv("IMMO_DATA_ROOT", "data"))
AN_PROC_DIR = DATA_ROOT / "processed" / "analysis"
AN_RAW_DIR = DATA_ROOT / "interim" / "analysis"
DUCKDB_PATH = Path(os.getenv("IMMO_ANALYTICS_DB", "analytics/immoeliza.duckdb"))

# --------- tiny logger ----------
def _log(msg: str) -> None:
    print(f"INFO:{LOG_PREFIX}:{msg}")

# --------- parsing helpers ----------
_num_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
def _to_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    m = _num_re.findall(s)
    if not m:
        return None
    s = m[0].replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _mid_from_range(text: str) -> float | None:
    # capture like "85–90 m²" or "85-90"
    if not isinstance(text, str):
        return None
    parts = re.split(r"[–\-to]+", text)
    nums = []
    for p in parts[:2]:
        v = _to_float(p)
        if v is not None:
            nums.append(v)
    if len(nums) == 2:
        return float(np.mean(nums))
    return None

def _infer_is_sale(url: str | None) -> bool | None:
    if not url:
        return None
    u = str(url)
    if "/for-sale/" in u:
        return True
    if "/for-rent/" in u or "/for-rent" in u:
        return False
    return None

# --------- core normalize ----------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure expected columns exist
    for c in ["price", "surface_m2", "price_per_m2", "postal_code", "url", "city",
              "property_type", "title", "bedrooms", "bathrooms", "year_built", "snapshot_date", "is_sale"]:
        if c not in out.columns:
            out[c] = pd.Series([None] * len(out))

    # snapshot_date to date
    sd = out.get("snapshot_date")
    if sd is None or sd.isna().all():
        out["snapshot_date"] = date.today()
    else:
        out["snapshot_date"] = pd.to_datetime(sd, errors="coerce").dt.date.fillna(date.today())

    # surface_m2: range mid or number
    s_candidates = ["surface_m2", "surface", "area", "area_m2", "surface (m²)", "surface_m²"]
    surf = out[s_candidates].bfill(axis=1).iloc[:, 0] if any(c in out.columns for c in s_candidates) else out["surface_m2"]
    surf = surf.apply(lambda v: _mid_from_range(v) if isinstance(v, str) and re.search(r"\d+\s*[–\-to]+\s*\d+", v) else _to_float(v))
    out["surface_m2"] = surf

    # price: first numeric token (accept comma decimal), guard against k€ mis-unit
    p_candidates = ["price", "price_eur", "price_text", "price_amount"]
    price_raw = out[p_candidates].bfill(axis=1).iloc[:, 0] if any(c in out.columns for c in p_candidates) else out["price"]
    price = price_raw.apply(_to_float)

    # if some rows look like "315" but others go to millions, keep as-is; k€->€ guard is in training_clean.
    out["price"] = price

    # bedrooms / bathrooms ints
    for col in ["bedrooms", "bathrooms", "year_built", "postal_code"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # is_sale inference from URL if missing
    if "is_sale" not in out.columns or out["is_sale"].isna().any():
        out["is_sale"] = out["is_sale"]
        out.loc[out["is_sale"].isna(), "is_sale"] = out["url"].apply(_infer_is_sale)

    # price_per_m2 (compute if possible)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppm2 = out["price"] / out["surface_m2"]
        ppm2 = ppm2.replace([np.inf, -np.inf], np.nan)
    out["price_per_m2"] = out.get("price_per_m2").where(lambda s: s.notna(), ppm2)

    # dedupe by URL
    if "url" in out.columns:
        before = len(out)
        out = out.drop_duplicates(subset=["url"], keep="first")
        _log(f"normalize(): dropped {before - len(out)} duplicate URLs")

    return out

@dataclass
class SanitySummary:
    when: str
    total_in: int
    total_out: int
    dropped_duplicates: int
    dropped_bad_price_for_sale: int
    dropped_surface_outlier: int
    dropped_ppm2_outlier: int
    dropped_null_key: int

def sanity_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, SanitySummary]:
    total_in = len(df)
    before = total_in

    # 1) required keys to be even remotely useful
    req = ["url", "price", "city", "postal_code"]
    null_mask = pd.Series(False, index=df.index)
    for c in req:
        if c in df.columns:
            null_mask |= df[c].isna()
    df1 = df[~null_mask].copy()
    dropped_null = before - len(df1); before = len(df1)

    # 2) drop blatantly wrong SALE prices (e.g., 1030€ looks like rent)
    # Keep rents; only apply guard where we know it's a sale
    sale_mask = df1["is_sale"] == True
    bad_sale_price = sale_mask & df1["price"].between(0, 10000, inclusive="both")
    df2 = df1[~bad_sale_price].copy()
    dropped_bad_sale = before - len(df2); before = len(df2)

    # 3) surface outliers (residential bounds)
    # keep None for surface if unknown; only drop if present and absurd
    surf = df2["surface_m2"]
    surf_bad = surf.notna() & ((surf < 8) | (surf > 1000))
    df3 = df2[~surf_bad].copy()
    dropped_surf = before - len(df3); before = len(df3)

    # 4) price-per-m2 sanity (if both present)
    ppm2 = df3["price_per_m2"]
    ppm2_bad = ppm2.notna() & ((ppm2 < 500) | (ppm2 > 20000))
    df4 = df3[~ppm2_bad].copy()
    dropped_ppm2 = before - len(df4); before = len(df4)

    # recompute ppm2 safely post-filter (might have lost surf/price)
    with np.errstate(divide="ignore", invalid="ignore"):
        df4["price_per_m2"] = (df4["price"] / df4["surface_m2"]).replace([np.inf, -np.inf], np.nan)

    summary = SanitySummary(
        when=datetime.now().isoformat(timespec="seconds"),
        total_in=total_in,
        total_out=len(df4),
        dropped_duplicates=0,  # handled earlier in normalize()
        dropped_bad_price_for_sale=int(dropped_bad_sale),
        dropped_surface_outlier=int(dropped_surf),
        dropped_ppm2_outlier=int(dropped_ppm2),
        dropped_null_key=int(dropped_null),
    )
    return df4, summary

# --------- DuckDB upsert ----------
def _ensure_duckdb() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("""
        CREATE TABLE IF NOT EXISTS listings_latest (
            snapshot_date DATE,
            city TEXT,
            postal_code INTEGER,
            property_type TEXT,
            price DOUBLE,
            price_per_m2 DOUBLE,
            surface_m2 DOUBLE,
            bedrooms INTEGER,
            bathrooms INTEGER,
            energy_label TEXT,
            is_sale BOOLEAN,
            url TEXT,
            title TEXT,
            year_built INTEGER
        );
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS listings_history AS
        SELECT * FROM listings_latest WHERE 1=0;
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS market_daily_summary (
            snapshot_date DATE,
            city TEXT,
            property_type TEXT,
            n INTEGER,
            median_price DOUBLE,
            median_surface_m2 DOUBLE,
            median_price_per_m2 DOUBLE
        );
        """)
    finally:
        con.close()

def upsert_duckdb(df: pd.DataFrame) -> None:
    _ensure_duckdb()
    con = duckdb.connect(DUCKDB_PATH)
    try:
        # replace latest in full (idempotent daily snapshot)
        con.execute("DELETE FROM listings_latest;")
        con.register("tmp_df", df)
        con.execute("""
            INSERT INTO listings_latest
            SELECT snapshot_date, city, postal_code, property_type, price, price_per_m2, surface_m2,
                   bedrooms, bathrooms, energy_label, is_sale, url, title, year_built
            FROM tmp_df
        """)
        # append to history
        con.execute("""
            INSERT INTO listings_history
            SELECT * FROM listings_latest
        """)
        # small summary table
        con.execute("DELETE FROM market_daily_summary WHERE snapshot_date IN (SELECT DISTINCT snapshot_date FROM listings_latest);")
        con.execute("""
            INSERT INTO market_daily_summary
            SELECT
              snapshot_date,
              city,
              property_type,
              COUNT(*) as n,
              MEDIAN(price) as median_price,
              MEDIAN(surface_m2) as median_surface_m2,
              MEDIAN(price_per_m2) as median_price_per_m2
            FROM listings_latest
            GROUP BY snapshot_date, city, property_type
        """)
    finally:
        con.close()

# --------- IO helpers ----------
def _read_compacted() -> pd.DataFrame:
    # your details_compact step writes one parquet per kind; read both if present
    parts: List[pd.DataFrame] = []
    for kind in ("apartments", "houses"):
        # latest compact parquet path convention:
        # data/processed/analysis/YYYY-MM-DD/{kind}.parquet OR interim/...
        # use a broad glob:
        for root in (AN_PROC_DIR, AN_RAW_DIR, DATA_ROOT):
            gl = sorted(root.glob(f"**/{kind}.parquet"))
            if gl:
                parts.append(pd.read_parquet(gl[-1]))
                break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# --------- main ----------
def run() -> str:
    df_raw = _read_compacted()
    if df_raw.empty:
        _log("No compacted inputs found; nothing to do.")
        return str(AN_PROC_DIR)

    _log(f"normalize(): starting with {len(df_raw)} rows")
    df_norm = normalize(df_raw)

    df_clean, summ = sanity_filter(df_norm)
    _log(f"sanity_filter(): kept {summ.total_out} / {summ.total_in} rows")

    # persist parquet (date-partition)
    day_dir = AN_PROC_DIR / str(date.today())
    day_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = day_dir / "listings.parquet"
    df_clean.to_parquet(out_parquet, index=False)
    _log(f"{out_parquet}")

    # save sanity summary json for UI footer
    (day_dir / "sanity_summary.json").write_text(json.dumps(summ.__dict__, indent=2))

    # duckdb upsert
    upsert_duckdb(df_clean)
    _log(f"📚 Wrote analytics into {DUCKDB_PATH}")

    return str(out_parquet)

if __name__ == "__main__":
    print(run())
