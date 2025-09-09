"""
Prepare training data from processed analytics parquet.

Trimming strategies (set via env):
- IMMO_TRAIN_TRIM=iqr       -> IQR fences per (region, property_type). Default.
- IMMO_TRAIN_TRIM=middle50  -> Keep Q1..Q3 only (per-bucket).
- IMMO_TRAIN_TRIM=qcut      -> Keep between IMMO_TRAIN_Q="0.05,0.95" (global).
- IMMO_TRAIN_MIN_GROUP=20   -> Min rows per bucket before per-bucket stats.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger("immoeliza.scraping")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ---------------------------
# Helpers
# ---------------------------

def _latest_processed_analysis_parquet(root: str = "data/processed/analysis") -> Path:
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"Processed analysis root not found: {root}")
    # pick latest date subdir that contains listings.parquet
    candidates = sorted(rootp.glob("*/listings.parquet"))
    if not candidates:
        # also try nested date folders
        candidates = sorted(rootp.glob("*/*/listings.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No listings.parquet under {root}")
    return candidates[-1]


_M2_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:m2|m²|sqm|m\^2)\b", re.I)

def _parse_m2(text: str | float | int | None) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    m = _M2_RE.search(s)
    if not m:
        return None
    val = m.group(1).replace(",", ".")
    try:
        return float(val)
    except Exception:
        return None


def _ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _price_unit_guard(df: pd.DataFrame, price_col: str = "price") -> None:
    """If prices look like thousands (k€), multiply by 1_000."""
    if price_col not in df.columns:
        return
    s = pd.to_numeric(df[price_col], errors="coerce")
    if s.notna().sum() == 0:
        return
    q90 = s.quantile(0.90)
    mx = s.max()
    med = s.median()
    # Heuristic: values like 840, 1200, 2500 typically mean k€
    if q90 is not None and mx is not None and q90 < 10_000 and mx < 100_000:
        LOG.info("Price unit guard applied (k€→€). stats before: med=%.1f, q90=%.1f, max=%.1f", med, q90, mx)
        df[price_col] = s * 1_000


def _compute_price_per_m2(df: pd.DataFrame) -> None:
    if "price" in df.columns and "surface_m2" in df.columns:
        p = pd.to_numeric(df["price"], errors="coerce")
        m = pd.to_numeric(df["surface_m2"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["price_per_m2"] = np.where(m > 0, p / m, np.nan)
    else:
        df["price_per_m2"] = np.nan


def _title_surface_backfill(df: pd.DataFrame) -> None:
    """Last-chance surface_m2 extraction from text columns."""
    if "surface_m2" not in df.columns:
        df["surface_m2"] = np.nan
    s = pd.to_numeric(df["surface_m2"], errors="coerce")
    missing = s.isna()
    if not missing.any():
        return
    candidates = []
    for col in ["title", "summary", "description"]:
        if col in df.columns:
            candidates.append(df[col].astype("string", copy=False))
    if not candidates:
        return
    txt = pd.concat(candidates, axis=1)
    # row-wise parse: first non-null parse wins
    parsed = txt.apply(lambda row: next((v for v in (_parse_m2(x) for x in row) if v is not None), np.nan), axis=1)
    fill_count_before = int(df["surface_m2"].notna().sum())
    df.loc[missing, "surface_m2"] = df.loc[missing, "surface_m2"].fillna(parsed[missing])
    fill_count_after = int(df["surface_m2"].notna().sum())
    gained = fill_count_after - fill_count_before
    if gained > 0:
        LOG.info("surface_m2 backfilled from text: +%d rows", gained)


def _choose_trim_strategy() -> Tuple[str, float, float, int]:
    strat = os.environ.get("IMMO_TRAIN_TRIM", "iqr").lower()
    lo, hi = 0.05, 0.95
    q = os.environ.get("IMMO_TRAIN_Q")
    if q:
        try:
            lo, hi = [float(x.strip()) for x in q.split(",")]
        except Exception:
            pass
    min_group = int(os.environ.get("IMMO_TRAIN_MIN_GROUP", "20"))
    return strat, lo, hi, min_group


def _trim_iqr_fences(s: pd.Series) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return (s >= lo) & (s <= hi)


def _trim_middle50(s: pd.Series) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return (s >= q1) & (s <= q3)


def _apply_trimming(df: pd.DataFrame) -> pd.DataFrame:
    strat, lo, hi, min_group = _choose_trim_strategy()

    # Work on price_per_m2 if available, else price
    target = "price_per_m2" if "price_per_m2" in df.columns and df["price_per_m2"].notna().any() else "price"
    s = pd.to_numeric(df[target], errors="coerce")

    kept_mask = pd.Series(True, index=df.index)

    if strat == "qcut":
        lo_v, hi_v = s.quantile(lo), s.quantile(hi)
        kept_mask &= (s >= lo_v) & (s <= hi_v)
        LOG.info("Trimming strategy=qcut kept %.0f–%.0f%% range on %s (%.0f..%.0f).",
                 lo*100, hi*100, target, lo_v or np.nan, hi_v or np.nan)

    else:
        # per-bucket trimming by (region, property_type) if present
        by = [c for c in ["region", "property_type"] if c in df.columns]
        if not by:
            by = None

        if by:
            kept = []
            for keys, g in df.groupby(by):
                s_g = pd.to_numeric(g[target], errors="coerce")
                if s_g.notna().sum() < max(min_group, 4):
                    kept.append(pd.Series(True, index=g.index))
                    continue
                if strat == "middle50":
                    m = _trim_middle50(s_g)
                else:  # "iqr" default
                    m = _trim_iqr_fences(s_g)
                kept.append(m.reindex(g.index, fill_value=True))
            kept_mask &= pd.concat(kept).reindex(df.index, fill_value=True)
            LOG.info("Trimming strategy=%s applied per-bucket %s on %s", strat, by, target)
        else:
            if s.notna().sum() >= 4:
                if strat == "middle50":
                    kept_mask &= _trim_middle50(s)
                else:
                    kept_mask &= _trim_iqr_fences(s)
                LOG.info("Trimming strategy=%s applied globally on %s", strat, target)

    before, after = len(df), int(kept_mask.sum())
    LOG.info("Trimmed rows: kept %d / %d (%.1f%%)", after, before, 100.0 * after / max(before, 1))
    return df[kept_mask].copy()


# ---------------------------
# Main
# ---------------------------

def run() -> str:
    """Build training parquet with optional outlier trimming."""
    src = _latest_processed_analysis_parquet()
    df = pd.read_parquet(src)

    # Ensure key columns exist
    for col in [
        "price", "surface_m2", "postal_code", "city", "region",
        "property_type", "bedrooms", "bathrooms", "year_built",
        "energy_label", "title", "snapshot_date", "is_sale"
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Numeric casts
    _ensure_numeric(df, ["price", "surface_m2", "bedrooms", "bathrooms", "year_built"])

    # Price unit guard (k€ → €) before any derived columns
    _price_unit_guard(df, "price")

    # Backfill surface from text if missing
    _title_surface_backfill(df)

    # Compute price_per_m2 where possible
    _compute_price_per_m2(df)

    # Keep for-sale only (if flag present; otherwise assume all are for sale in this dataset)
    if "is_sale" in df.columns and df["is_sale"].notna().any():
        df = df[df["is_sale"] == True].copy()

    # Basic quality gates (very permissive—model can handle missing via pipelines)
    # Keep rows that have price OR surface (we still allow one missing; pipelines can impute)
    price_ok = pd.to_numeric(df["price"], errors="coerce") > 30_000
    surface_ok = pd.to_numeric(df["surface_m2"], errors="coerce") >= 10
    df = df[price_ok | surface_ok].copy()

    # Outlier trimming (configurable)
    df = _apply_trimming(df)

    # Final feature set for training parquet
    feat_cols = [
        "price", "surface_m2", "price_per_m2",
        "bedrooms", "bathrooms",
        "postal_code", "city", "region",
        "property_type", "year_built", "energy_label",
    ]
    train = df[feat_cols].copy()

    # Save under processed/training/<snapshot_date>/training.parquet
    # Prefer snapshot_date if present and not null; otherwise today's date
    snap = df["snapshot_date"].dropna()
    if len(snap) and str(snap.iloc[0]):
        try:
            # snapshot_date may be string or date
            snap_date = pd.to_datetime(str(snap.iloc[0])).date()
        except Exception:
            snap_date = pd.Timestamp.today().date()
    else:
        snap_date = pd.Timestamp.today().date()

    out_dir = Path("data/processed/training") / str(snap_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "training.parquet"
    train.to_parquet(out, index=False)

    LOG.info("Training dataset saved -> %s (%d rows)", out, len(train))
    return str(out)


if __name__ == "__main__":
    print(run())
