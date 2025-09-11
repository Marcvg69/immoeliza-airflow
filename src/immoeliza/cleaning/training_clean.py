# immoeliza/cleaning/training_clean.py
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("immoeliza.scraping")
logging.basicConfig(level=os.environ.get("IMMO_LOGLEVEL", "INFO"))

ROOT = Path(".")
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
ANALYSIS_DIR = PROCESSED / "analysis"
TRAINING_DIR = PROCESSED / "training"

# --- Surface parsing helpers (used as a last-chance fill) ---------------------

SURF_TOKENS = [
    # French
    "surface habitable", "surface", "habitable", "superficie",
    # Dutch
    "bewoonbare opp.", "bew. opp.", "bewoonbare oppervlakte", "opp.", "oppervlakte", "woonopp",
    # English-ish
    "living area", "area",
    # Misc portal variants
    "woonoppervlakte", "m²",
]

SURF_PAT = re.compile(
    r"(?P<a>\d{1,3}(?:[.,]\d{1,2})?)\s*(?:[-–—]\s*(?P<b>\d{1,3}(?:[.,]\d{1,2})?))?\s*(?:m2|m\u00B2|m²|sqm|m)",
    flags=re.IGNORECASE,
)

def _to_float(x) -> Optional[float]:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        return v if np.isfinite(v) else None
    s = str(x).strip()
    s = s.replace("\u00A0", " ").replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^\d.\-–—]", "", s)
    if not s:
        return None
    # handle ranges like 85–90
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2.0
    try:
        return float(s)
    except ValueError:
        return None

def _extract_surface_any(row: pd.Series) -> Optional[float]:
    """Best-effort extraction of surface from structured specs or free text."""
    # 1) direct numeric value in typical fields
    for k in ("surface_m2", "surface", "living_area"):
        if k in row and pd.notna(row[k]):
            v = _to_float(row[k])
            if v and v > 5:
                return v

    # 2) dict-like specs/attributes
    for maybe in ("specs", "attributes", "details"):
        if maybe in row and pd.notna(row[maybe]) and isinstance(row[maybe], dict):
            for k, val in row[maybe].items():
                k_norm = re.sub(r"[^a-z]", "", str(k).lower())
                if any(tok.replace(" ", "") in k_norm for tok in [t.lower() for t in SURF_TOKENS]):
                    v = _to_float(val)
                    if v and v > 5:
                        return v

    # 3) scan blobs
    for col in ("title", "description", "raw_html"):
        if col in row and pd.notna(row[col]):
            m = SURF_PAT.search(str(row[col]))
            if m:
                a = _to_float(m.group("a"))
                b = _to_float(m.group("b"))
                pick = b if b else a
                if pick and pick > 5:
                    return pick
    return None

# --- Sources / paths ----------------------------------------------------------

def _latest_analytics_parquet() -> Path:
    """Pick newest processed analytics parquet: data/processed/analysis/YYYY-MM-DD/listings.parquet."""
    candidates = sorted(ANALYSIS_DIR.glob("*/listings.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No analytics parquet under {ANALYSIS_DIR}/**/listings.parquet")
    return candidates[-1]

# --- Trimming config (env-driven) --------------------------------------------

@dataclass
class TrimCfg:
    mode: str = os.environ.get("IMMO_TRAIN_TRIM", "qcut")  # "qcut" or "none"
    quantiles: Tuple[float, float] = tuple(
        float(x.strip()) for x in os.environ.get("IMMO_TRAIN_Q", "0.05,0.95").split(",")
    )  # type: ignore

    def should(self) -> bool:
        return self.mode.lower() in {"qcut", "quantile", "quantiles"}

# --- Guards & builders --------------------------------------------------------

def _price_unit_guard(df: pd.DataFrame):
    """If prices look like k€ (e.g. med ~250, q90 ~500, max ~8501), multiply by 1_000."""
    if "price" not in df:
        return df, {"applied": False}
    s = pd.to_numeric(df["price"], errors="coerce")
    med = float(s.median(skipna=True) or 0)
    q90 = float(s.quantile(0.9) or 0)
    mx = float(s.max(skipna=True) or 0)
    info = {"before": {"median": med, "q90": q90, "max": mx}}
    # Heuristic: likely k€ if q90 < 10k and max < 100k
    if q90 and q90 < 10000 and mx < 100000:
        df = df.copy()
        df["price"] = (s * 1000).round().astype("Int64")
        info["applied"] = True
        log.info("Price unit guard applied (k€→€). stats before: med=%.1f, q90=%.1f, max=%.1f", med, q90, mx)
    else:
        info["applied"] = False
    return df, info

def _build_training(df: pd.DataFrame, trim: TrimCfg):
    meta = {"n_in": int(len(df))}

    # Keep essential columns
    cols = [
        "url", "title", "price", "surface_m2", "bedrooms", "bathrooms",
        "postal_code", "city", "region", "property_type", "year_built",
        "energy_label", "snapshot_date",
    ]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()

    # Keep latest per URL if snapshot present
    if "url" in df.columns and "snapshot_date" in df.columns:
        try:
            df["snapshot_ts"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            df = df.sort_values("snapshot_ts").groupby("url", as_index=False).tail(1)
        except Exception:
            pass

    # Last-chance backfill of surface
    if "surface_m2" not in df.columns:
        df["surface_m2"] = pd.NA
    miss = df["surface_m2"].isna()
    if miss.any():
        extracted = df.loc[miss].apply(_extract_surface_any, axis=1)
        df.loc[miss, "surface_m2"] = extracted
        meta["surface_filled"] = int(df["surface_m2"].notna().sum())

    # Numeric coercions
    for c in ("price", "surface_m2", "bedrooms", "bathrooms", "year_built"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sanity
    if "price" in df.columns:
        df = df[df["price"] > 0]
    if "surface_m2" in df.columns:
        df.loc[df["surface_m2"] <= 5, "surface_m2"] = pd.NA  # unrealistic tiny

    # price_per_m2 when possible
    if "price" in df.columns and "surface_m2" in df.columns:
        denom = pd.to_numeric(df["surface_m2"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            ppm2 = pd.to_numeric(df["price"], errors="coerce") / denom
        df["price_per_m2"] = ppm2.where(denom > 0)

    # Target choice for trimming only (model code will decide separately)
    target = "price_per_m2" if ("price_per_m2" in df and df["price_per_m2"].notna().any()) else "price"
    meta["trim_target"] = target

    # Env-driven trimming
    if trim.should() and target in df.columns:
        s = pd.to_numeric(df[target], errors="coerce")
        q_lo, q_hi = df[target].quantile(trim.quantiles[0]), df[target].quantile(trim.quantiles[1])
        before = len(df)
        df = df[(s >= q_lo) & (s <= q_hi)]
        meta["trim"] = {
            "mode": trim.mode,
            "q": list(trim.quantiles),
            "range": [float(q_lo), float(q_hi)],
            "dropped": before - len(df),
        }

    # Final NA policy
    if target == "price":
        df = df[df["price"].notna()]
    else:
        df = df[df["price_per_m2"].notna()]

    meta["n_out"] = int(len(df))
    return df, meta

# --- Public entrypoint --------------------------------------------------------

def run(save: bool = True) -> str:
    """Build the training dataset from the latest processed analytics parquet and save it."""
    src = _latest_analytics_parquet()
    df = pd.read_parquet(src)

    # k€ → €
    df, guard = _price_unit_guard(df)

    trim = TrimCfg()
    out, meta = _build_training(df, trim)

    # Save under data/processed/training/YYYY-MM-DD/training.parquet
    day = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = TRAINING_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training.parquet"
    if save:
        out.to_parquet(out_path, index=False)
        log.info("Training dataset saved -> %s (%d rows)", out_path.as_posix(), len(out))

    # Tiny metadata
    meta_path = out_dir / "metadata.json"
    meta_doc = {
        "source": src.as_posix(),
        "saved": out_path.as_posix(),
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "price_guard": guard,
        **meta,
    }
    meta_path.write_text(json.dumps(meta_doc, indent=2))

    return out_path.as_posix()

if __name__ == "__main__":
    print(run())
