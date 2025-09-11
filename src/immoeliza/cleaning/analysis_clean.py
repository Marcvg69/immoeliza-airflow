# analysis_clean.py
# Rebuild processed analytics: normalize fields, fill price/surface, compute price_per_m2,
# and upsert safely into DuckDB without column-count mismatches.

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import duckdb
import numpy as np
import pandas as pd

LOG_PREFIX = "immoeliza.scraping"
DATA_ROOT = Path(os.getenv("IMMO_DATA_ROOT", "data"))
ANALYTICS_DIR = DATA_ROOT / "processed" / "analysis"
DUCKDB_PATH = Path(os.getenv("IMMO_ANALYTICS_DB", "analytics/immoeliza.duckdb"))

# --------------------------------- logging --------------------------------- #
def _log(msg: str) -> None:
    print(f"INFO:{LOG_PREFIX}:{msg}")

def _warn(msg: str) -> None:
    print(f"WARNING:{LOG_PREFIX}:{msg}")

def _err(msg: str) -> None:
    print(f"ERROR:{LOG_PREFIX}:{msg}")

# ----------------------------- small utilities ----------------------------- #
def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")

def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

_num_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def _to_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.floating, np.integer, np.number)):
        return float(x)
    s = str(x)
    if not s:
        return None
    m = _num_re.search(s.replace("\xa0", " ").replace(",", "."))
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except Exception:
        return None

def _parse_m2_from_text(s: str) -> Optional[float]:
    """
    Extract surface in m² from free text. Accepts decimal comma and ranges like '85–90 m²'.
    Chooses the MIDPOINT of ranges; falls back to the single value.
    """
    if not s or not isinstance(s, str):
        return None
    s_norm = s.lower().replace("\xa0", " ")
    m2_unit = r"(?:m2|m\u00b2|sqm|sq\.? m|m²)"
    # a–b m² (hyphen, en-dash, em-dash, or "to")
    range_re = re.compile(rf"(\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:[.,]\d+)?)\s*{m2_unit}")
    m = range_re.search(s_norm)
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        return (a + b) / 2.0
    single_re = re.compile(rf"(\d+(?:[.,]\d+)?)\s*{m2_unit}")
    m = single_re.search(s_norm)
    if m:
        return float(m.group(1).replace(",", "."))
    return None

# -------------------- structured spec parsing (multilingual) -------------------- #
SURFACE_SYNONYMS = {
    # EN/FR/NL (+ a common DE one) labels found in structured tables
    "surface": True,
    "surface habitable": True, "surfacehabitable": True, "habitable surface": True,
    "living area": True, "area": True,
    "bewoonbare opp.": True, "bew. opp.": True, "opp.": True,
    "bewoonbare oppervlakte": True, "oppervlakte": True, "woonoppervlakte": True,
    "wohnfläche": True,
    "m2": True, "m²": True, "sqm": True,
}

def _norm_key(k: str) -> str:
    k = k.strip().lower()
    k = k.replace(" ", " ").replace("\xa0"," ")
    k = re.sub(r"\s+", " ", k)
    return k

def _maybe_parse_structured_surface(specs: Dict[str, str | float | int]) -> Optional[float]:
    for raw_k, v in specs.items():
        k = _norm_key(str(raw_k))
        if k in SURFACE_SYNONYMS or any(k.startswith(x) for x in ["surface", "bewoon", "opp", "woon", "area", "habitable"]):
            if isinstance(v, (int, float, np.integer, np.floating)):
                vv = float(v)
                if vv > 0:
                    return vv
            vv = _parse_m2_from_text(str(v))
            if vv:
                return vv
    return None

def _coerce_dict_like(x) -> Optional[Dict]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def fill_surface_structured(df_missing: pd.DataFrame) -> pd.Series:
    """
    Fill surface_m2 using, in order:
      1) 'specs' dict/JSON of label->value (structured tables).
      2) Any column whose *name* suggests a surface field.
      3) Free text fields (title/description/...).
    Returns a float Series aligned to df_missing.index.
    """
    out = pd.Series(index=df_missing.index, dtype="float64")

    # 1) structured 'specs'
    if "specs" in df_missing.columns:
        for idx, val in df_missing["specs"].items():
            d = _coerce_dict_like(val)
            if d:
                vv = _maybe_parse_structured_surface(d)
                if vv:
                    out.loc[idx] = vv

    # 2) structured-ish: other object columns that look like label/value cells
    if out.isna().any():
        obj_cols = [c for c in df_missing.columns if df_missing[c].dtype == "object" and c != "specs"]
        for idx in out[out.isna()].index:
            row = df_missing.loc[idx, obj_cols]
            found = None
            for c, v in row.items():
                if any(tok in _norm_key(str(c)) for tok in ["surface", "opp", "woon", "area", "habitable"]):
                    vv = _parse_m2_from_text(str(v))
                    if vv:
                        found = vv; break
                vv = _parse_m2_from_text(str(v))
                if vv:
                    found = vv; break
            if found:
                out.loc[idx] = found

    # 3) free-text fallback
    if out.isna().any():
        text_cols = [c for c in ["title","description","details_text","body","summary"] if c in df_missing.columns]
        for c in text_cols:
            need = out.isna()
            if not need.any():
                break
            parsed = df_missing.loc[need, c].astype("string", errors="ignore").map(_parse_m2_from_text)
            out.loc[need] = out.loc[need].fillna(parsed)

    return out

# -------------------------------- price filling ------------------------------- #
def robust_price_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill/clean a 'price' column using a cascade:
      - existing numeric price if present
      - else try price_text/raw/title/description
      - normalize separators (comma decimal) and detect 'k' unit
      - auto-detect k€ scale for small-looking series and convert to €
    """
    df = df.copy()
    cand_cols = [c for c in ["price","price_text","price_raw","raw_price","title","description"] if c in df.columns]
    if "price" not in df.columns:
        df["price"] = pd.NA

    scan = pd.DataFrame(index=df.index)
    for c in cand_cols:
        scan[c] = df[c]

    def _price_from_any(x):
        if isinstance(x, (int,float,np.integer,np.floating)) and not pd.isna(x):
            return float(x)
        s = str(x)
        if not s:
            return None
        s = s.replace("€"," ").replace("\xa0"," ").strip()
        km = re.search(r"(\d+(?:[.,]\d+)?)\s*k\b", s.lower())
        if km:
            return float(km.group(1).replace(",",".")) * 1000.0
        m = _num_re.search(s.replace(",","."))  # interpret comma as decimal
        if not m:
            return None
        return float(m.group())

    # take the first candidate value that parses
    fallback = scan.apply(lambda row: next((v for v in row if _price_from_any(v) is not None), None), axis=1)
    price_vals = fallback.map(_price_from_any).astype("float64")

    m = df["price"].isna()
    df.loc[m, "price"] = price_vals[m]

    # k€ → € guard
    if df["price"].notna().any():
        vals = df["price"].astype("float64")
        med = np.nanmedian(vals)
        q90 = np.nanpercentile(vals, 90)
        mx  = np.nanmax(vals)
        if mx <= 10000 and q90 >= 300 and med >= 100:
            _log(f"Price unit guard applied (k€→€). stats before: med={med}, q90={q90}, max={mx}")
            df["price"] = vals * 1000.0

    return df

# ---------------------------- duckdb upsert util --------------------------- #
def _connect_duckdb(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    return con

def duckdb_type_of(s: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(s): return "BIGINT"
    if pd.api.types.is_float_dtype(s):   return "DOUBLE"
    if pd.api.types.is_bool_dtype(s):    return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(s): return "TIMESTAMP"
    return "TEXT"

def _create_table_from_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame):
    cols = ", ".join([f'"{c}" {duckdb_type_of(df[c])}' for c in df.columns])
    con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols});')

def upsert_duckdb_safe(df: pd.DataFrame, path: Path, table: str, key_cols: Iterable[str]):
    """
    Safe upsert using DuckDB MERGE with explicit column lists.
    Avoids column-count/order mismatches.
    """
    if df.empty:
        return
    con = _connect_duckdb(path)
    try:
        # normalize object→string to keep schema stable
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")

        _create_table_from_df(con, table, df)
        con.register("tmp_df", df)

        cols = [f'"{c}"' for c in df.columns]
        on = " AND ".join([f'T.{c} = S.{c}' for c in key_cols])
        updates = ", ".join([f'{c}=S.{c}' for c in cols if c.strip('"') not in key_cols])
        cols_csv = ", ".join(cols)

        merge_sql = f"""
        MERGE INTO "{table}" AS T
        USING tmp_df AS S
        ON {on}
        WHEN MATCHED THEN UPDATE SET {updates}
        WHEN NOT MATCHED THEN INSERT ({cols_csv}) VALUES ({cols_csv});
        """
        con.execute(merge_sql)
    finally:
        try: con.close()
        except Exception: pass

# -------------------------- core normalization flow ------------------------ #
BASE_COLS = [
    "snapshot_date","url","title","city","region","postal_code","property_type",
    "price","surface_m2","price_per_m2",
    "bedrooms","bathrooms","year_built","energy_label",
    "description","specs"
]

def _latest_source() -> Optional[Path]:
    """Pick a sensible default source parquet if none given."""
    candidates: List[Path] = []
    # prefer prior analytics parquet (you often re-clean these)
    for p in sorted((DATA_ROOT / "processed" / "analysis").glob("*/*")):
        if p.name == "listings.parquet":
            candidates.append(p)
    # fallback: raw/interim
    candidates += sorted(DATA_ROOT.glob("raw/**/*.parquet"))
    candidates += sorted(DATA_ROOT.glob("interim/**/*.parquet"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = _ensure_cols(df, BASE_COLS)

    # snapshot_date → date (robust to scalar)
    sd = df["snapshot_date"]
    df["snapshot_date"] = pd.to_datetime(sd, errors="coerce")
    if df["snapshot_date"].isna().all():
        df["snapshot_date"] = pd.to_datetime(datetime.now())
    df["snapshot_date"] = df["snapshot_date"].dt.date

    # 1) robust price
    before_price_na = df["price"].isna().sum()
    df = robust_price_fill(df)
    after_price_na = df["price"].isna().sum()

    # 2) surface from structured/free text (only where missing)
    m_missing_surf = df["surface_m2"].isna()
    if m_missing_surf.any():
        filled = fill_surface_structured(df.loc[m_missing_surf])
        df.loc[m_missing_surf, "surface_m2"] = filled

    # 3) price_per_m2
    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_per_m2"] = np.where(
            pd.to_numeric(df["surface_m2"], errors="coerce") > 0,
            pd.to_numeric(df["price"], errors="coerce") / pd.to_numeric(df["surface_m2"], errors="coerce"),
            np.nan
        )

    # numeric coercions for common fields
    for c in ["bedrooms","bathrooms","year_built"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tot = len(df)
    _log(f'normalize(): price filled: {before_price_na - after_price_na} / {tot} | surface_m2 filled: {int(df["surface_m2"].notna().sum())} / {tot}')
    return df

def _compute_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["snapshot_date","city","property_type"], dropna=False)
    out = grp.agg(
        n=("url","nunique"),
        median_price=("price","median"),
        median_ppm2=("price_per_m2","median")
    ).reset_index()
    out["n"] = out["n"].astype("int64")
    return out

def _write_parquet(df: pd.DataFrame) -> Path:
    out_dir = ANALYTICS_DIR / _today_str()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "listings.parquet"
    df.to_parquet(p, index=False)
    _log(str(p))
    return p

def _migrate_duckdb_schema(con: duckdb.DuckDBPyConnection):
    # keep common text columns as TEXT to avoid type drift
    for tbl, cols_types in {
        "listings_latest": {"url":"TEXT","title":"TEXT","city":"TEXT"},
        "listings_history": {"url":"TEXT","title":"TEXT","city":"TEXT"},
        "market_daily_summary": {"city":"TEXT","property_type":"TEXT","n":"INT"},
    }.items():
        try:
            for col, typ in cols_types.items():
                con.execute(f'ALTER TABLE {tbl} ALTER COLUMN {col} SET DATA TYPE {typ}')
                _log(f"🛠️ DuckDB: ALTER TABLE {tbl} ALTER COLUMN {col} SET DATA TYPE {typ}")
        except Exception:
            pass  # tables may not exist yet

def run(source_path: str | Path | None = None) -> str:
    """
    1) Load latest source parquet (or given path)
    2) Normalize (fill price/surface, compute ppm2)
    3) Save analytics parquet (date-partitioned)
    4) Upsert listings_latest (by url) and listings_history (url+snapshot_date)
       and market_daily_summary (snapshot_date+city+property_type)
    Return the path to the analytics parquet.
    """
    # 1) load
    if source_path is None:
        p = _latest_source()
        if not p:
            raise FileNotFoundError("No source parquet found under data/ (raw/interim/processed).")
        source_path = p
    source_path = Path(source_path)
    raw = pd.read_parquet(source_path)

    # 2) normalize
    nn = normalize(raw)

    # 3) write analytics parquet
    out_p = _write_parquet(nn)

    # 4) upsert into DuckDB
    con = _connect_duckdb(DUCKDB_PATH)
    try:
        _migrate_duckdb_schema(con)
    finally:
        try: con.close()
        except Exception: pass

    # latest per URL and full history
    nn["snapshot_ts"] = pd.to_datetime(nn["snapshot_date"])
    nn_sorted = nn.sort_values(["url","snapshot_ts"])
    latest = nn_sorted.groupby("url", as_index=False).tail(1).drop(columns=["snapshot_ts"])
    history = nn.copy()

    upsert_duckdb_safe(latest, DUCKDB_PATH, "listings_latest",  key_cols=["url"])
    upsert_duckdb_safe(history, DUCKDB_PATH, "listings_history", key_cols=["url","snapshot_date"])
    summary = _compute_market_summary(nn)
    upsert_duckdb_safe(summary, DUCKDB_PATH, "market_daily_summary", key_cols=["snapshot_date","city","property_type"])

    _log(f"📚 Wrote analytics into {DUCKDB_PATH}")
    return str(out_p)

if __name__ == "__main__":
    print(run())
