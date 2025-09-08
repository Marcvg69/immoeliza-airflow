"""
Produce a training parquet from the processed analytics daily snapshot.

- Reads the latest data/processed/analysis/*/listings.parquet
- Filters to 'for sale' rows
- Picks strict/relaxed/minimal filter that yields >0 rows
- Ensures expected columns (incl. price_per_m2 for the model)
- Saves to data/processed/training/YYYY-MM-DD/training.parquet
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional, List

import pandas as pd


CANON_TRAIN_COLS: List[str] = [
    "price", "surface_m2", "price_per_m2",
    "bedrooms", "bathrooms",
    "postal_code", "city", "property_type", "year_built", "energy_label",
]

def _data_root() -> Path:
    return Path(os.environ.get("IMMO_DATA_ROOT", "data")).resolve()

def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")

def _latest_analytics_parquet() -> Optional[Path]:
    root = _data_root() / "processed" / "analysis"
    if not root.exists():
        return None
    cands = list(root.rglob("listings.parquet"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _out_training_path() -> Path:
    out_dir = _data_root() / "processed" / "training" / _today_str()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "training.parquet"

def _pick_filter_variant(base: pd.DataFrame) -> pd.DataFrame:
    """Try strict → relaxed → minimal; return the first non-empty."""
    stages = [
        ("strict",   base.dropna(subset=["price", "surface_m2", "city", "postal_code"]).query("price > 10000 and surface_m2 > 10")),
        ("relaxed",  base.dropna(subset=["price", "surface_m2"]).query("price > 10000 and surface_m2 > 10")),
        ("minimal",  base.dropna(subset=["price", "surface_m2"])),
    ]
    for _, df in stages:
        if len(df) > 0:
            return df
    return base.iloc[0:0]

def run() -> str:
    """Build the training parquet and return its path."""
    src = _latest_analytics_parquet()
    out = _out_training_path()

    if not src or not src.exists():
        pd.DataFrame(columns=CANON_TRAIN_COLS).to_parquet(out, index=False)
        return str(out)

    df = pd.read_parquet(src)

    # keep for-sale if available
    if "is_sale" in df.columns:
        df = df[df["is_sale"] == True]

    # make sure price_per_m2 exists/consistent
    if "price_per_m2" not in df.columns:
        df["price_per_m2"] = (df["price"] / df["surface_m2"]).where(
            (df["price"].notna()) & (df["surface_m2"].notna()) & (df["surface_m2"] > 0)
        )

    picked = _pick_filter_variant(df)

    train = picked[
        ["price","surface_m2","price_per_m2",
         "bedrooms","bathrooms","postal_code","city",
         "property_type","year_built","energy_label"]
    ].copy()

    # Coerce some types
    for c in ["bedrooms","bathrooms"]:
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0).astype(int)
    train["year_built"] = pd.to_numeric(train["year_built"], errors="coerce").astype("Int64")

    train.to_parquet(out, index=False)
    return str(out)


__all__ = ["run"]
