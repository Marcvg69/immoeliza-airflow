# src/immoeliza/cleaning/sanity_checks.py
from __future__ import annotations

import logging
import math
import re
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("immoeliza.scraping")

_CURRENCY_RE = re.compile(
    r"""(?ix)
    (?P<cur>€|eur|euro|euros)\s*
    (?P<val>[\d\.\s]+(?:[,\.\s]\d{1,3})?)
    |
    (?P<val2>[\d\.\s]+(?:[,\.\s]\d{1,3})?)\s*(?P<cur2>€|eur|euro|euros)
    """
)

_K_SUFFIX_RE = re.compile(r"(?i)\b(\d{1,3}(?:[.,]\d{1,3})?)\s?k\b")  # e.g., "250 k"

def _coerce_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        if isinstance(x, (int, float, np.number)):
            v = float(x)
            return v if math.isfinite(v) else None
        s = str(x).strip()
        s = s.replace("\u00A0", " ").replace(" ", "")
        s = s.replace(",", ".")
        # remove any currency marks
        s = s.replace("€", "").replace("EUR", "").replace("eur", "").replace("Euros", "")
        return float(s)
    except Exception:
        return None

def _parse_price_from_text(text: str) -> Optional[float]:
    """Parse a price only if there is a clear currency cue (€, eur, euro, or 'k')."""
    if not text or not isinstance(text, str):
        return None

    # explicit currency surrounding a number
    m = _CURRENCY_RE.search(text)
    if m:
        raw = m.group("val") or m.group("val2")
        if raw:
            raw = raw.replace("\u00A0", " ").replace(" ", "").replace(".", "").replace(",", ".")
            try:
                v = float(raw)
                return v
            except Exception:
                pass

    # “250k”, “1.2 k”, etc. → euros
    mk = _K_SUFFIX_RE.search(text)
    if mk:
        raw = mk.group(1).replace(",", ".")
        try:
            return float(raw) * 1_000.0
        except Exception:
            return None

    return None

def _looks_like_postal_as_price(price: Optional[float], postal_code: Optional[str]) -> bool:
    """Detect classic artifact where postal code ended up in price."""
    if price is None:
        return False
    if not (1_000 <= price <= 9_999):
        return False
    if not postal_code:
        return False
    # Accept strings like "1180", "1180.0", "1180,0"
    try:
        pc = str(postal_code).strip().replace(",", ".")
        pc_num = float(pc)
        return abs(pc_num - price) < 0.5
    except Exception:
        return False

def _apply_price_unit_guard(series: pd.Series) -> pd.Series:
    """If most prices are likely in k€ (<= 10k), scale by 1000 to euros."""
    s = series.copy()
    if s.dropna().empty:
        return s
    med = s.median(skipna=True)
    q90 = s.quantile(0.90)
    mx  = s.max(skipna=True)
    # Heuristic: typical Belgian listing prices in € are > 50k.
    if med <= 2_000 and q90 <= 9_000 and mx <= 10_000:
        log.info("Price unit guard applied (k€→€). stats before: med=%s, q90=%s, max=%s", med, q90, mx)
        s = s * 1_000.0
    return s

def _clip_plausible_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Surface: drop if completely insane
    if "surface_m2" in out:
        out.loc[(out["surface_m2"] < 10) | (out["surface_m2"] > 1_000), "surface_m2"] = np.nan
    # Price: drop very small sale prices (after unit guard)
    if "price" in out:
        out.loc[out["price"] < 50_000, "price"] = np.nan
    return out

def apply(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity-check & repair a normalized listings dataframe.
    Expected columns if available:
      url, title, price, price_text, city, postal_code, surface_m2, property_type, snapshot_date
    The function is defensive: missing columns are tolerated.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # Coerce types
    if "price" in out:
        out["price"] = out["price"].apply(_coerce_float)
    if "surface_m2" in out:
        out["surface_m2"] = out["surface_m2"].apply(_coerce_float)

    # Parse price from text when missing / suspicious (but only with currency cues)
    text_cols = [c for c in ("price_text", "title", "raw_text") if c in out.columns]
    if text_cols:
        # Only fill if price is NaN or looks like postal code artifact
        def _fix_price(row):
            price = row.get("price", None)
            if _looks_like_postal_as_price(price, row.get("postal_code")):
                price = None
            if price is None:
                for c in text_cols:
                    p = _parse_price_from_text(row.get(c, ""))
                    if p is not None:
                        return p
            return price

        out["price"] = out.apply(_fix_price, axis=1)

    # Unit guard (k€ → €) if stats look like thousands
    if "price" in out:
        out["price"] = _apply_price_unit_guard(out["price"])

    # Drop obvious artifacts “price == postal_code”
    if {"price", "postal_code"} <= set(out.columns):
        mask_art = [_looks_like_postal_as_price(r["price"], r["postal_code"]) for _, r in out.iterrows()]
        mask_art = pd.Series(mask_art, index=out.index)
        if mask_art.any():
            out.loc[mask_art, "price"] = np.nan

    # Plausibility clips
    out = _clip_plausible_ranges(out)

    # Compute €/m² if both present
    if {"price", "surface_m2"} <= set(out.columns):
        denom = out["surface_m2"].replace({0: np.nan})
        out["price_per_m2"] = np.where(
            denom.notna() & out["price"].notna(), out["price"] / denom, np.nan
        )

    # Final light dedupe
    base_cols = [c for c in ("url", "snapshot_date", "city", "postal_code", "property_type") if c in out.columns]
    if base_cols:
        out = out.sort_values(base_cols + [c for c in ("price", "surface_m2") if c in out.columns])
        out = out.drop_duplicates(subset=base_cols, keep="last")

    return out
