# src/immoeliza/cleaning/analysis_clean.py
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("immoeliza.scraping")
logging.basicConfig(level=os.environ.get("IMMO_LOGLEVEL", "INFO"))

# -------------------------------------------------------------------
# Paths & helpers
# -------------------------------------------------------------------

ANALYTICS_DB_ENV = "IMMO_ANALYTICS_DB"
DEFAULT_ANALYTICS_DB = "analytics/immoeliza.duckdb"

def _analytics_db_path() -> str:
    return os.environ.get(ANALYTICS_DB_ENV, DEFAULT_ANALYTICS_DB)

def _today_str() -> str:
    return date.today().isoformat()

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _processed_analysis_out() -> Path:
    out = Path(f"data/processed/analysis/{_today_str()}/listings.parquet")
    _ensure_parent(out)
    return out

# -------------------------------------------------------------------
# Region mapping (simple BE heuristic)
# -------------------------------------------------------------------

def postal_to_region(postal_code: Optional[str | int]) -> Optional[str]:
    if postal_code is None:
        return None
    try:
        pc = int(str(postal_code)[:4])
    except Exception:
        return None
    if 1000 <= pc <= 1299:
        return "Brussels-Capital"
    first = int(str(pc)[0])
    if first in (1, 2, 3, 8, 9):
        return "Flanders"
    if first in (4, 5, 6, 7):
        return "Wallonia"
    return None

# -------------------------------------------------------------------
# Numeric parsing helpers
# -------------------------------------------------------------------

_NUM_TOKEN = re.compile(r"(\d{1,9}(?:[ .\u00A0]\d{3})*(?:[,\.]\d{1,2})?|\d{1,6}(?:[,\.]\d{1,2})?)")
_EUR_HINT = re.compile(r"(?:^|[\s:])(?:€|eur|euro)s?", re.I)
_K_SUFFIX = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*k\b", re.I)

def _clean_number_token(tok: str) -> float | None:
    s = tok.strip().replace("\u00A0", " ").replace(" ", "")
    # if we still have both '.' and ',', assume '.' is thousands sep and ',' is decimal
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # otherwise remove thousands sep (.) and keep decimal (.)
        # and convert decimal comma to dot
        s = s.replace(".", "")
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _find_first_number(text: str) -> float | None:
    if not isinstance(text, str) or not text:
        return None
    m = _NUM_TOKEN.search(text)
    if not m:
        return None
    return _clean_number_token(m.group(1))

def _parse_money_like(text: str) -> float | None:
    """Try to parse euros from a free-text cell."""
    if not isinstance(text, str) or not text:
        return None
    t = text.strip()
    # explicit k-suffix (e.g., "254,5k")
    mk = _K_SUFFIX.search(t)
    if mk:
        fv = _clean_number_token(mk.group(1))
        if fv is not None:
            return fv * 1_000.0
    # currency hint + number
    if _EUR_HINT.search(t):
        fv = _find_first_number(t)
        return fv
    # fallback: just first number
    return _find_first_number(t)

def _round_price_if_k(series: pd.Series) -> pd.Series:
    """If series looks like 'thousands' (e.g., med=250), multiply by 1_000."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    med = s.dropna().median()
    q90 = s.dropna().quantile(0.90)
    mx = s.dropna().max()
    if (med < 2_000) and (q90 < 2_000) and (mx <= 2_000):
        logger.info("Applied price unit guard: multiplied by 1_000 (med=%.1f, q90=%.1f before scaling)", med, q90)
        return s * 1_000.0
    return s

# -------------------------------------------------------------------
# Surface extraction
# -------------------------------------------------------------------

_AREA_NAME_HINTS = [
    "surface", "area", "habitable", "woon", "bewoon", "utile", "brute",
    "net", "gross", "land", "lot", "wohn", "fläche", "superficie", "sqm", "m²", "m2",
]

_M2_PAT = re.compile(
    r"(?<!\d)(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:m2|m\u00B2|m²|sqm|sq\s*m|m\s*2)\b",
    flags=re.IGNORECASE,
)
_NUM_PAT = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)")

def _to_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("\u00A0", " ").replace("\xa0", " ").replace(",", ".")
    m = _NUM_PAT.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _search_m2_in_text(text: str) -> Optional[float]:
    if not isinstance(text, str) or not text:
        return None
    t = " ".join(text.split())
    m = _M2_PAT.search(t)
    if m:
        return _to_float(m.group(1))
    return None

def _extract_surface_from_jsonish(cell: str) -> Optional[float]:
    try:
        s = str(cell)
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(s[start:end+1])

            def scan(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        lk = str(k).lower()
                        if any(h in lk for h in ("floorsize", "floor_size", "surface", "area")):
                            if isinstance(v, dict):
                                val = v.get("value") or v.get("amount") or v.get("m2") or v.get("size")
                                fv = _to_float(val)
                                if fv:
                                    return fv
                            fv = _to_float(v)
                            if fv:
                                return fv
                    for v in o.values():
                        r = scan(v)
                        if r:
                            return r
                elif isinstance(o, list):
                    for it in o:
                        r = scan(it)
                        if r:
                            return r
                return None

            return scan(obj)
    except Exception:
        return None
    return None

def _likely_area_col(name: str) -> bool:
    ln = name.lower()
    return any(h in ln for h in _AREA_NAME_HINTS)

def _best_surface(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    base = pd.Series([np.nan] * n, index=df.index, dtype="float64")

    # 1) any column whose name hints at area/surface
    direct_candidates = []
    for col in df.columns:
        if _likely_area_col(col):
            series = pd.to_numeric(df[col].apply(_to_float), errors="coerce")
            direct_candidates.append(series)
    if direct_candidates:
        stacked = pd.concat(direct_candidates, axis=1)
        base = stacked.bfill(axis=1).iloc[:, 0]

    # 2) regex across ALL object columns (no applymap deprecation)
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        rg = pd.DataFrame(
            {c: df[c].map(_search_m2_in_text) for c in obj_cols},
            index=df.index
        )
        rg_first = rg.apply(lambda r: next((v for v in r.values if v is not None), np.nan), axis=1)
        base = base.fillna(pd.to_numeric(rg_first, errors="coerce"))

    # 3) JSON-ish blobs
    for c in obj_cols:
        col_vals = df[c]
        if not col_vals.astype(str).str.contains("{").any():
            continue
        json_guess = col_vals.map(lambda s: _extract_surface_from_jsonish(str(s)))
        base = base.fillna(pd.to_numeric(json_guess, errors="coerce"))

    return base

# -------------------------------------------------------------------
# Price extraction (robust)
# -------------------------------------------------------------------

_PRICE_NAME_HINTS = ["price", "prijs", "prix", "amount", "value", "amount_eur", "asking", "kost", "kosten", "rent", "huur"]

def _likely_price_col(name: str) -> bool:
    ln = name.lower()
    return any(h in ln for h in _PRICE_NAME_HINTS)

def _best_price(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    base = pd.Series([np.nan] * n, index=df.index, dtype="float64")

    # 1) numeric from price-like columns
    direct_candidates = []
    for col in df.columns:
        if _likely_price_col(col):
            # try text parse first, then numeric
            col_series = df[col]
            parsed = col_series.map(_parse_money_like)
            num = pd.to_numeric(col_series, errors="coerce")
            combined = parsed.fillna(num)
            direct_candidates.append(combined)
    if direct_candidates:
        stacked = pd.concat(direct_candidates, axis=1)
        base = stacked.bfill(axis=1).iloc[:, 0]

    # 2) scan title/description first
    text_first = []
    for cand in ("title", "description", "desc", "summary"):
        if cand in df.columns:
            text_first.append(df[cand].map(_parse_money_like))
    if text_first:
        stacked = pd.concat(text_first, axis=1)
        base = base.fillna(stacked.bfill(axis=1).iloc[:, 0])

    # 3) last resort: scan all object cols for money-like
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        scan_all = pd.DataFrame({c: df[c].map(_parse_money_like) for c in obj_cols}, index=df.index)
        fallback = scan_all.bfill(axis=1).iloc[:, 0]
        base = base.fillna(fallback)

    # scale if obviously k€
    base = _round_price_if_k(base)
    return pd.to_numeric(base, errors="coerce")

# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------

@dataclass
class Normalized:
    url: str
    title: Optional[str]
    city: Optional[str]
    postal_code: Optional[str]
    property_type: Optional[str]
    price: Optional[float]
    surface_m2: Optional[float]
    bedrooms: Optional[int]
    bathrooms: Optional[int]
    year_built: Optional[int]
    energy_label: Optional[str]
    is_sale: Optional[bool]
    snapshot_date: date

_KEEP_COLS = [
    "snapshot_date",
    "url", "title", "city", "postal_code", "region",
    "property_type", "price", "surface_m2",
    "bedrooms", "bathrooms", "year_built", "energy_label",
    "is_sale",
]

def _first_present_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # snapshot_date (vectorized; broadcast when missing)
    if "snapshot_date" in df.columns:
        sd = pd.to_datetime(df["snapshot_date"], errors="coerce")
        # sd can be a scalar Timestamp in some edge-cases; handle both
        if isinstance(sd, pd.Series):
            out["snapshot_date"] = sd.dt.date
        else:
            out["snapshot_date"] = pd.Series([sd.date() if hasattr(sd, "date") else date.today()] * len(df), index=df.index)
    else:
        out["snapshot_date"] = pd.Series([date.today()] * len(df), index=df.index)

    # url
    url = _first_present_series(df, ["url", "link", "href"])
    out["url"] = url.astype(str)

    # strings
    out["title"]        = _first_present_series(df, ["title", "headline"])
    out["city"]         = _first_present_series(df, ["city", "locality", "address_city"])
    out["postal_code"]  = _first_present_series(df, ["postal_code", "postcode", "zip", "zip_code"])
    out["property_type"]= _first_present_series(df, ["property_type", "type", "category"])
    out["energy_label"] = _first_present_series(df, ["energy_label", "epc"])

    # numeric-ish
    out["bedrooms"]    = pd.to_numeric(df.get("bedrooms"), errors="coerce").round().astype("Int64")
    out["bathrooms"]   = pd.to_numeric(df.get("bathrooms"), errors="coerce").round().astype("Int64")
    out["year_built"]  = pd.to_numeric(df.get("year_built"), errors="coerce").round().astype("Int64")

    # robust price + surface
    out["price"]      = _best_price(df)
    out["surface_m2"] = _best_surface(df)

    # is_sale
    if "is_sale" in df.columns:
        s = df["is_sale"].astype(str).str.lower().map({"true": True, "1": True, "false": False, "0": False})
        out["is_sale"] = s.fillna(df["is_sale"])
    else:
        u = out["url"].astype(str).str.lower()
        out["is_sale"] = u.str.contains("for-sale|te-koop|a-vendre|zu-verkaufen", regex=True)

    # region from postal
    out["postal_code"] = out["postal_code"].where(out["postal_code"].notna(), None)
    out["region"] = out["postal_code"].apply(postal_to_region)

    # casts
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["surface_m2"] = pd.to_numeric(out["surface_m2"], errors="coerce")

    # small diagnostics
    filled_s = int(out["surface_m2"].notna().sum())
    filled_p = int(out["price"].notna().sum())
    logger.info("normalize(): price filled: %d / %d | surface_m2 filled: %d / %d", filled_p, len(out), filled_s, len(out))

    return out[_KEEP_COLS]

# -------------------------------------------------------------------
# DuckDB: schema & upserts
# -------------------------------------------------------------------

_SCHEMA_LATEST = """
snapshot_date DATE,
url TEXT,
title TEXT,
city TEXT,
postal_code VARCHAR,
region VARCHAR,
property_type TEXT,
price DOUBLE,
surface_m2 DOUBLE,
bedrooms INTEGER,
bathrooms INTEGER,
year_built INTEGER,
energy_label VARCHAR,
is_sale BOOLEAN
"""

_SCHEMA_HISTORY = _SCHEMA_LATEST

_SCHEMA_SUMMARY = """
snapshot_date DATE,
city TEXT,
property_type TEXT,
n INT,
median_price DOUBLE,
median_surface_m2 DOUBLE,
median_price_per_m2 DOUBLE
"""

def _migrate_schema(con: duckdb.DuckDBPyConnection) -> None:
    # Create if missing; if extra columns already exist, we'll target insert columns explicitly.
    con.execute(f"CREATE TABLE IF NOT EXISTS listings_latest   ({_SCHEMA_LATEST})")
    con.execute(f"CREATE TABLE IF NOT EXISTS listings_history  ({_SCHEMA_HISTORY})")
    con.execute(f"CREATE TABLE IF NOT EXISTS market_daily_summary ({_SCHEMA_SUMMARY})")

_INSERT_COLS = (
    "snapshot_date, url, title, city, postal_code, region, property_type, "
    "price, surface_m2, bedrooms, bathrooms, year_built, energy_label, is_sale"
)

def _upsert_with_retry(df: pd.DataFrame, db_path: str) -> None:
    con = duckdb.connect(db_path)
    try:
        _migrate_schema(con)
        con.register("df", df)

        # refresh today's snapshot in listings_latest
        day = df["snapshot_date"].iloc[0]
        con.execute("DELETE FROM listings_latest WHERE snapshot_date = ?", [day])
        con.execute(f"""
            INSERT INTO listings_latest ({_INSERT_COLS})
            SELECT {_INSERT_COLS} FROM df
        """)

        # append to history
        con.execute(f"""
            INSERT INTO listings_history ({_INSERT_COLS})
            SELECT {_INSERT_COLS} FROM df
        """)

        # recompute daily summary (replace table atomically)
        con.execute("""
            CREATE OR REPLACE TABLE market_daily_summary AS
            WITH base AS (
                SELECT
                    snapshot_date,
                    city,
                    property_type,
                    price,
                    surface_m2,
                    CASE WHEN surface_m2 > 0 AND price > 0 THEN price / surface_m2 END AS price_per_m2
                FROM listings_history
            )
            SELECT
                snapshot_date,
                city,
                property_type,
                COUNT(*) AS n,
                median(price) AS median_price,
                median(surface_m2) AS median_surface_m2,
                median(price_per_m2) AS median_price_per_m2
            FROM base
            GROUP BY 1,2,3
        """)
    finally:
        con.close()

# -------------------------------------------------------------------
# Load sources & entry point
# -------------------------------------------------------------------

def _load_latest_compacted_details(kind: str) -> Optional[pd.DataFrame]:
    today = date.today()
    today_path = Path(f"data/raw/{today:%Y/%m/%d}/{kind}_details.parquet")
    if today_path.exists():
        return pd.read_parquet(today_path)
    base = Path("data/raw")
    if not base.exists():
        return None
    files = sorted(base.rglob(f"{kind}_details.parquet"))
    if files:
        return pd.read_parquet(files[-1])
    return None

def run() -> Optional[str]:
    dfs: List[pd.DataFrame] = []
    for kind in ("apartments", "houses"):
        raw = _load_latest_compacted_details(kind)
        if raw is None or raw.empty:
            continue
        nn = normalize(raw)
        dfs.append(nn)

    if not dfs:
        logger.warning("No compacted details found; nothing to normalize.")
        return None

    df = pd.concat(dfs, ignore_index=True)

    # helper column (not part of DuckDB tables)
    df_out = df.copy()
    df_out["price_per_m2"] = np.where(
        (df_out["price"] > 0) & (df_out["surface_m2"] > 0),
        df_out["price"] / df_out["surface_m2"],
        np.nan,
    )

    out = _processed_analysis_out()
    df_out.to_parquet(out, index=False)

    logger.info("📚 Wrote analytics into %s", Path(_analytics_db_path()).resolve())
    logger.info(str(out))

    _upsert_with_retry(df, _analytics_db_path())
    return str(out)

if __name__ == "__main__":
    print(run())
