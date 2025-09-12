# src/immoeliza/cleaning/analysis_clean.py
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

# ---- Paths ----
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/immoeliza/cleaning -> src -> repo
ANALYTICS_DIR = REPO_ROOT / "analytics"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
ANALYTICS_DB_PATH = ANALYTICS_DIR / "immoeliza.duckdb"
SANITY_JSON_PATH = ANALYTICS_DIR / "_sanity_summary.json"
LAST_REBUILD_PATH = ANALYTICS_DIR / "_last_rebuild.txt"

_raw_candidates: list[Path] = []
for pat in [
    REPO_ROOT / "data" / "**" / "listings*.parquet",
    REPO_ROOT / "data" / "**" / "compact*.parquet",
    REPO_ROOT / "data" / "**" / "detail*.parquet",
]:
    _raw_candidates.extend(sorted(Path(REPO_ROOT).glob(str(pat.relative_to(REPO_ROOT)))))

# ---- Helpers ----
def _con() -> duckdb.DuckDBPyConnection:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(ANALYTICS_DB_PATH))

def _write_table(df: pd.DataFrame, name: str) -> None:
    with _con() as con:
        con.register("df_src", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df_src;")
        con.unregister("df_src")

def _read_table_exists(name: str) -> bool:
    with _con() as con:
        try:
            con.execute(f"SELECT 1 FROM {name} LIMIT 1;")
            return True
        except Exception:
            return False

def _clean_price(x) -> float | None:
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[^\d.,]", "", s)
    s = s.replace(".", "")  # thousand separators
    s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return np.nan
    return v if v > 0 else np.nan

def _clean_surface(x) -> float | None:
    if pd.isna(x):
        return np.nan
    s = str(x)
    # Examples: "85", "85,5", "85 m²", "85m2", "85 - 90"
    s = s.lower().replace("m²", "").replace("m2", "")
    s = re.sub(r"[^\d,.\-]", "", s).strip()
    # ranges "80-90" -> take left
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)", s)
    if not m:
        return np.nan
    s = m.group(1).replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return np.nan
    return v if v > 0 else np.nan

def _infer_property_type(url: str | None, title: str | None) -> str | None:
    text = " ".join([str(url or ""), str(title or "")]).lower()
    for key, lab in [
        ("apartment", "Apartment"),
        ("residence", "Residence"),
        ("house", "House"),
        ("villa", "Villa"),
    ]:
        if key in text:
            return lab
    return "Other"

def _is_rent_from_url(url: str | None) -> bool | None:
    s = str(url or "").lower()
    if "/for-rent/" in s:
        return True
    if "/for-sale/" in s:
        return False
    return None

def _now_ts_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

# ---- Load latest source DF from several possible locations ----
def _load_source_df() -> pd.DataFrame:
    # Prefer DuckDB history if exists
    if _read_table_exists("listings_history"):
        with _con() as con:
            return con.execute("SELECT * FROM listings_history;").df()

    # Otherwise take the newest parquet that looks like listings
    if _raw_candidates:
        newest = sorted(_raw_candidates)[-1]
        try:
            return pd.read_parquet(newest)
        except Exception:
            pass

    # Fallback: empty
    return pd.DataFrame(columns=[
        "url","price","bedrooms","bathrooms","surface_m2","title","city","postal_code",
        "property_type","is_rent","snapshot_date"
    ])

# ---- Public API ----
def rebuild_analytics(clean: bool = True) -> dict:
    df = _load_source_df().copy()

    # Normalize expected columns
    if "surface" in df and "surface_m2" not in df:
        df["surface_m2"] = df["surface"]
    if "postal-code" in df and "postal_code" not in df:
        df["postal_code"] = df["postal-code"]
    if "snapshot_ts" in df and "snapshot_date" not in df:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_ts"]).dt.floor("D")

    # Required columns w/ defaults
    for col in ["url","title","city","property_type","postal_code"]:
        if col not in df:
            df[col] = None
    for col in ["bedrooms","bathrooms","price","surface_m2"]:
        if col not in df:
            df[col] = np.nan
    if "is_rent" not in df:
        df["is_rent"] = df["url"].map(_is_rent_from_url)
    if "snapshot_date" not in df:
        df["snapshot_date"] = pd.Timestamp(datetime.now().date())

    # Clean numeric fields
    df["price"] = df["price"].apply(_clean_price).astype("float64")
    df["surface_m2"] = df["surface_m2"].apply(_clean_surface).astype("float64")

    # Fill property_type if missing
    miss_type = df["property_type"].isna() | (df["property_type"].astype(str).str.len() == 0)
    if miss_type.any():
        df.loc[miss_type, "property_type"] = [
            _infer_property_type(u, t) for u, t in zip(df.loc[miss_type, "url"], df.loc[miss_type, "title"])
        ]

    # Fill rent flag from URL where missing
    if df["is_rent"].isna().any():
        df.loc[df["is_rent"].isna(), "is_rent"] = [
            _is_rent_from_url(u) for u in df.loc[df["is_rent"].isna(), "url"]
        ]

    # Keep schema tidy
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.floor("D")
    df["postal_code"] = df["postal_code"].astype(str).str.extract(r"(\d+)")

    # price_per_m2 computed safely
    df["price_per_m2"] = np.where(df["surface_m2"] > 0, df["price"] / df["surface_m2"], np.nan)

    # --- Sanity metrics (counts only; don’t drop yet) ---
    total_rows = len(df)
    missing_price = int(df["price"].isna().sum())
    missing_m2 = int(df["surface_m2"].isna().sum())
    bad_m2 = int(((df["surface_m2"] < 10) | (df["surface_m2"] > 1000)).sum())
    bad_eur_m2 = int(((df["price_per_m2"] < 500) | (df["price_per_m2"] > 50000)).sum())

    # Optionally drop crazy rows for downstream
    if clean:
        df = df[
            (df["price"].notna()) &
            (df["surface_m2"].notna()) &
            (df["surface_m2"] >= 10) & (df["surface_m2"] <= 1000) &
            (df["price_per_m2"].isna() | ((df["price_per_m2"] >= 500) & (df["price_per_m2"] <= 50000)))
        ].copy()

    kept_after_sanity = len(df)

    # ---- Persist to DuckDB ----
    # History = all rows
    _write_table(df, "listings_history")

    # Latest = newest snapshot only
    if not df.empty:
        latest_date = df["snapshot_date"].max()
        df_latest = df[df["snapshot_date"] == latest_date].copy()
    else:
        df_latest = df.copy()

    _write_table(df_latest, "listings_latest")

    # Daily market summary
    if not df.empty:
        grp = (
            df.groupby(["snapshot_date", "city", "property_type"], dropna=False)
              .agg(n=("url", "count"),
                   median_price=("price", "median"),
                   median_eur_m2=("price_per_m2", "median"))
              .reset_index()
        )
    else:
        grp = pd.DataFrame(columns=["snapshot_date","city","property_type","n","median_price","median_eur_m2"])
    _write_table(grp, "market_daily_summary")

    # ---- side artifacts for the UI ----
    SANITY_JSON_PATH.write_text(json.dumps({
        "total_rows": total_rows,
        "missing_price": missing_price,
        "missing_m2": missing_m2,
        "bad_m2": bad_m2,
        "bad_eur_m2": bad_eur_m2,
        "kept_after_sanity": kept_after_sanity,
    }, indent=2))
    LAST_REBUILD_PATH.write_text(_now_ts_string())

    return {
        "total_rows": total_rows,
        "missing_price": missing_price,
        "missing_m2": missing_m2,
        "bad_m2": bad_m2,
        "bad_eur_m2": bad_eur_m2,
        "kept_after_sanity": kept_after_sanity,
    }

if __name__ == "__main__":
    print(rebuild_analytics())
