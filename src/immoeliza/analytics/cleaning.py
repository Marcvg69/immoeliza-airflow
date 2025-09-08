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
from ..io_paths import raw_path
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
        raise FileNotFoundError(f"Missing {kind} details parquet at {p}. Run fetch_details + compact_details first.")
    df = pd.read_parquet(p)
    if df.empty:
        raise ValueError(f"{kind}_details.parquet is empty at {p}")
    df["kind"] = kind
    return df

# ---------------- robust numeric parsing ----------------
# Normalize thousands separators: remove spaces (incl. NBSP & thin spaces), remove dots/commas as thousands,
# keep optional decimal. For PRICE we just keep an integer (euros). For SURFACE we allow decimals.
NBSP = "\u00A0"
THIN = "\u202F"

def _strip_currency_and_space(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.replace(NBSP, " ").replace(THIN, " ")
    s = s.strip()
    # drop currency symbols and any non digit/sep
    s = re.sub(r"[^\d.,\- ]+", "", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s

def _price_to_int_eur(x):
    if pd.isna(x): return np.nan
    s = _strip_currency_and_space(str(x))
    if not s: return np.nan
    # Keep only digits, dots, commas
    s2 = re.sub(r"[^\d.,]", "", s)
    # Heuristic: if there is exactly one separator and <=2 digits after it -> treat as decimal, else treat all as thousands
    if s2.count(",") + s2.count(".") == 1:
        sep = "," if "," in s2 else "."
        left, right = s2.split(sep, 1)
        if len(right) <= 2:  # decimal
            try:
                return int(float(left + "." + right))
            except:
                pass
    # Otherwise: drop all separators, treat as integer euros
    s3 = re.sub(r"[.,]", "", s2)
    if not s3: return np.nan
    try: return int(s3)
    except: return np.nan

def _surface_to_float_m2(x):
    if pd.isna(x): return np.nan
    s = _strip_currency_and_space(str(x))
    if not s: return np.nan
    # keep digits + one decimal sep
    s2 = re.sub(r"[^\d.,]", "", s)
    # prefer comma as decimal if present
    if "," in s2 and "." in s2:
        # assume dots are thousands, comma is decimal
        s2 = s2.replace(".", "").replace(",", ".")
    else:
        s2 = s2.replace(",", ".")
    try: return float(s2)
    except: 
        # last resort: digits only
        digits = re.sub(r"\D", "", s2)
        try: return float(digits)
        except: return np.nan

_price_in_text = re.compile(r"(?:€|\bEUR\b)\s*([\d\s\u00A0\u202F\.,]+)", re.IGNORECASE)
_surface_in_text = re.compile(r"(\d+(?:[.,]\d+)?)\s*m(?:²|2)\b", re.IGNORECASE)

def _extract_price_from_text(txt: str) -> float | np.nan:
    if not isinstance(txt, str): return np.nan
    m = _price_in_text.search(txt)
    if not m: return np.nan
    return _price_to_int_eur(m.group(1))

def _extract_surface_from_text(txt: str) -> float | np.nan:
    if not isinstance(txt, str): return np.nan
    m = _surface_in_text.search(txt)
    if not m: return np.nan
    return _surface_to_float_m2(m.group(1))

# ---------------- normalization ----------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower().strip(): c for c in df.columns}
    lcols = list(colmap.keys())
    def pick_exact(*names):
        for n in names:
            if n in colmap: return colmap[n]
        return None
    def pick_contains(*keywords):
        for lc in lcols:
            if any(k in lc for k in keywords):
                return colmap[lc]
        return None

    out = pd.DataFrame()

    url_col = pick_exact("url") or pick_contains("url","link","href")
    out["url"] = df[url_col] if url_col else np.nan

    title_col = pick_exact("title","name","heading") or pick_contains("title","headline","beschrijving","description")
    desc_col  = pick_exact("description","beschrijving","omschrijving","details","info") or pick_contains("descr","beschrij","omscr","details","info")
    title = df[title_col].astype(str) if title_col else pd.Series("", index=df.index)
    desc  = df[desc_col].astype(str)  if desc_col  else pd.Series("", index=df.index)

    price_col = pick_exact("price","amount","price_eur") or pick_contains("price","prijs","rent","huur","kost","asking")
    price = df[price_col] if price_col else pd.Series(np.nan, index=df.index)
    price = price.apply(_price_to_int_eur)
    price = price.fillna(title.apply(_extract_price_from_text)).fillna(desc.apply(_extract_price_from_text))

    city = None
    for nm in ("city","town","locality","gemeente","stad","municipality","plaats"):
        if nm in colmap: city = df[colmap[nm]]; break
    if city is None:
        c2 = pick_contains("city","town","locality","gemeente","stad","municipality","plaats")
        city = df[c2] if c2 else ""
    pc_col = pick_exact("postalcode","postal_code","zip","zipcode","postcode") or pick_contains("postal","zip","post")
    postal = df[pc_col] if pc_col else ""

    tcol = pick_exact("type","property_type","category","subtype","typology") or pick_contains("type","category","subtype","typology")
    ptype = df[tcol] if tcol else ""

    bedcol = pick_exact("bedrooms","beds","slaapkamers") or pick_contains("bed","slaap")
    bathcol= pick_exact("bathrooms","baths","badkamers") or pick_contains("bath","bad")
    beds   = df[bedcol] if bedcol else np.nan
    baths  = df[bathcol] if bathcol else np.nan

    surfcol = (pick_exact("surface","area","living_area","size","habitable_surface","woonopp","oppervlakte")
               or pick_contains("surface","area","m2","m²","opper","habitable","woon"))
    surface = df[surfcol] if surfcol else pd.Series(np.nan, index=df.index)
    surface = surface.apply(_surface_to_float_m2)
    surface = surface.fillna(title.apply(_extract_surface_from_text)).fillna(desc.apply(_extract_surface_from_text))

    yearcol = pick_exact("year","built","construction_year","bouwjaar") or pick_contains("year","bouw","construction")
    ybuilt  = df[yearcol] if yearcol else np.nan
    ecol    = pick_exact("epc","energy","energy_label","epc_label","peb") or pick_contains("epc","energy","peb")
    elabel  = df[ecol] if ecol else ""

    url_series = out["url"].astype(str).fillna("")
    str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    blob = df[str_cols].astype(str).agg(" ".join, axis=1).fillna("") if str_cols else pd.Series([""]*len(df), index=df.index)
    pat_rent = r"/for-rent/|for rent|te huur|à louer|zur miete|zu vermieten"
    pat_sale = r"/for-sale/|for sale|te koop|à vendre|zum verkauf|verkauf"
    is_rent = url_series.str.contains(pat_rent, case=False, regex=True) | blob.str.contains(pat_rent, case=False, regex=True)
    is_sale = url_series.str.contains(pat_sale, case=False, regex=True) | blob.str.contains(pat_sale, case=False, regex=True)
    inferred_sale = (~is_rent) & (~is_sale) & (price >= 50_000)
    is_sale = is_sale | inferred_sale

    out["price"]        = price
    out["bedrooms"]     = pd.Series(beds).apply(lambda v: np.nan if pd.isna(v) else _price_to_int_eur(v))  # int-ish
    out["bathrooms"]    = pd.Series(baths).apply(lambda v: np.nan if pd.isna(v) else _price_to_int_eur(v))
    out["surface_m2"]   = pd.Series(surface).apply(_surface_to_float_m2)
    out["year_built"]   = pd.Series(ybuilt).apply(lambda v: np.nan if pd.isna(v) else _price_to_int_eur(v))
    out["title"]        = title
    out["city"]         = city
    out["postal_code"]  = postal
    out["property_type"]= ptype
    out["energy_label"] = elabel
    out["is_rent"]      = is_rent
    out["is_sale"]      = is_sale

    out["price_per_m2"] = out["price"] / out["surface_m2"]
    out["snapshot_date"] = date.today().isoformat()
    return out

def _load_details_both() -> pd.DataFrame:
    dfs = []
    for k in ("apartments", "houses"):
        try:
            dfs.append(_read_details(k))
        except Exception as e:
            logger.warning("Skipping %s: %s", k, e)
    if not dfs:
        raise FileNotFoundError("No details parquet available. Run fetch_details + compact_details first.")
    return pd.concat(dfs, ignore_index=True)

def run_for_analysis() -> str:
    p = Paths()
    df = _load_details_both()
    df_norm = normalize(df).dropna(subset=["price"]).reset_index(drop=True)
    if df_norm.empty:
        raise ValueError("Normalized dataset is empty (no rows with price). Check details extraction.")

    p.analytics_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p.analytics_db))
    con.register("df_norm", df_norm)
    con.execute("CREATE OR REPLACE TABLE listings_latest  AS SELECT * FROM df_norm")
    con.execute("CREATE OR REPLACE TABLE listings_history AS SELECT * FROM df_norm WHERE 1=0")
    con.execute("INSERT INTO listings_history SELECT * FROM df_norm")
    con.execute("""
        CREATE OR REPLACE TABLE market_daily_summary AS
        SELECT snapshot_date, city, postal_code, property_type,
               COUNT(*) AS n_listings,
               AVG(price) AS avg_price,
               MEDIAN(price) AS med_price,
               AVG(price_per_m2) AS avg_ppm2
        FROM df_norm
        GROUP BY 1,2,3,4
        LIMIT 0
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
    p = Paths()
    df = _load_details_both()
    df = normalize(df)

    base = df[(df["is_sale"] == True)].dropna(subset=["price","surface_m2"]).copy()
    stages = [
        ("strict",  base.query("price > 10000 and surface_m2 > 10")),
        ("relaxed", base.query("price > 1000 and surface_m2 > 10")),
        ("minimal", base),
    ]
    picked_name, picked = next(((n,c) for n,c in stages if len(c)>0), ("empty_base", base))
    logger.info("🧮 Training filter picked: %s (rows=%d) [strict=%d, relaxed=%d, minimal=%d]",
                picked_name, len(picked), len(stages[0][1]), len(stages[1][1]), len(stages[2][1]))

    train = picked[[
        "price","surface_m2","bedrooms","bathrooms",
        "postal_code","city","property_type","year_built","energy_label"
    ]].copy()
    train["bedrooms"]  = train["bedrooms"].fillna(0)
    train["bathrooms"] = train["bathrooms"].fillna(0)

    out_dir = p.training_root / p.snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "training.parquet"
    train.to_parquet(out, index=False)
    logger.info("🧹 Training dataset saved -> %s (%d rows)", out, len(train))
    return str(out)
