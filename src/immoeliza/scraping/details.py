"""
Detail scraper with structured spec parsing (multilingual + robust):

- Parses both free-text and structured spec tables (table/tr, dl/dt+dd, li with "Label: Value").
- Accepts multilingual surface labels:
    "Living area", "Area", "Surface habitable", "Bewoonbare opp.", "Opp.",
    "Bew. opp.", "Habitable surface", "Woonopp.", "Woonoppervlakte", "Oppervlakte", etc.
- Handles decimal commas and ranges like "85–90 m²" (midpoint by default).
    Env IMMO_SURFACE_RANGE = "mid" | "max"  (default "mid")
- Attempts robust price extraction (€, k€, euro, "EUR 250.000", "€ 250.000", etc.)
- Pulls city/postal/type/sale from URL segments if missing from page.
- Writes CSV under:
    data/raw/YYYY/MM/DD/<kind>/_real_estate_details_<timestamp>/*.csv
- Returns CSV path.

Relies on undetected_chromedriver to avoid basic bot checks.
"""

from __future__ import annotations

import os
import re
import time
import json
import math
import glob
import html
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

# Selenium / Undetected Chrome
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

logger = logging.getLogger("immoeliza.scraping")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# -------------------------
# Helpers
# -------------------------

_SURFACE_LABELS = {
    # English
    "living area", "living-area", "area", "surface", "habitable surface", "living space",
    # French
    "surface habitable", "surf. habitable", "surface utile",
    # Dutch
    "bewoonbare opp.", "bew. opp.", "bewoonbare oppervlakte", "woonopp.", "woonoppervlakte", "oppervlakte", "opp.",
    # Other short
    "m²", "sqm", "sq m", "m2",
}

_BEDROOM_LABELS = {"bedrooms", "chambres", "slaapkamers", "kamers", "ch"}
_BATHROOM_LABELS = {"bathrooms", "sdb", "bathrooms", "badkamers", "bad", "bath", "bathroom"}
_YEAR_LABELS = {"construction year", "year built", "built", "bouwjaar", "année de construction", "année"}
_ENERGY_LABELS = {"energy label", "peb", "epc", "energy class", "klasse", "label énergie"}

_NUM_RX = re.compile(r"(?<!\d)(\d{1,3}(?:[.\s]\d{3})*|\d+)([,.]\d+)?")  # 250.000, 250 000, 250000, 14,5
_M2_RANGE_SEP = re.compile(r"\s*(?:-|–|—|to|à|tot)\s*")

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def _normalize_label(s: str) -> str:
    s = html.unescape((s or "").strip().lower())
    s = re.sub(r"\s+", " ", s)
    # remove trailing colon
    s = s[:-1] if s.endswith(":") else s
    # compact special words (opp. -> opp)
    s = s.replace("opp.", "opp").replace("m²", "m2")
    return s

def _extract_numbers(text: str) -> List[float]:
    """Extract numbers tolerant to thousand sep and decimal comma."""
    out = []
    for m in _NUM_RX.finditer(text or ""):
        whole, frac = m.groups()
        whole = whole.replace(" ", "").replace(".", "")
        if frac:
            val = float(f"{whole}.{frac[1:]}")
        else:
            val = float(whole)
        out.append(val)
    return out

def _parse_surface_value(raw: str) -> Optional[float]:
    """
    Parse a surface string into m² (float).
    Accepts:
      - "85 m²", "85,5 m²", "1.250 m²"
      - "85–90 m²" -> choose midpoint (default) or max via IMMO_SURFACE_RANGE
    """
    text = (raw or "").strip()
    if not text:
        return None

    # Split potential range first
    parts = _M2_RANGE_SEP.split(text)
    chooser = os.getenv("IMMO_SURFACE_RANGE", "mid").lower()

    def _one(val_text: str) -> Optional[float]:
        nums = _extract_numbers(val_text)
        if not nums:
            return None
        return float(nums[0])

    if len(parts) >= 2:
        a = _one(parts[0])
        b = _one(parts[1])
        if a is not None and b is not None:
            if chooser == "max":
                return float(max(a, b))
            return float((a + b) / 2.0)

    # Single value path
    v = _one(text)
    return v

def _parse_price(text: str) -> Optional[float]:
    """
    Parse EUR price tolerant to formats:
      "€ 250.000", "EUR 250,000", "250 000 €", "250k", "1.2M"
    Returns absolute euros (not k€).
    """
    if not text:
        return None
    t = text.replace("\u00a0", " ").lower()
    # multipliers
    mult = 1.0
    if "k" in t and "€" in t or "eur" in t:
        # "250k €" -> multiply at end based on trailing k/m
        pass
    if re.search(r"\b(\d+(?:[.,]\d+)?)\s*m\b", t):
        m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*m\b", t)
        val = float(m.group(1).replace(".", "").replace(",", "."))
        return float(val * 1_000_000)

    if re.search(r"\b(\d+(?:[.,]\d+)?)\s*k\b", t):
        m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*k\b", t)
        val = float(m.group(1).replace(".", "").replace(",", "."))
        return float(val * 1_000)

    nums = _extract_numbers(t)
    if not nums:
        return None

    # choose the largest plausible number on page (often the actual price)
    candidate = max(nums)
    return float(candidate)

def _from_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """
    Heuristics from immovlan URLs:
      https://immovlan.be/en/detail/apartment/for-sale/1030/schaarbeek/<id>
    -> property_type = apartment, is_sale=True, postal_code=1030, city=schaarbeek
    """
    try:
        parts = [p for p in url.split("/") if p]
        # look for 'detail/<type>/for-sale|for-rent/<postal>/<city>/...'
        if "detail" in parts:
            i = parts.index("detail")
            ptype = parts[i + 1] if i + 1 < len(parts) else None
            action = parts[i + 2] if i + 2 < len(parts) else ""
            is_sale = "sale" in (action or "")
            postal = parts[i + 3] if i + 3 < len(parts) else None
            city = parts[i + 4] if i + 4 < len(parts) else None
            return (ptype, city, postal, is_sale)
    except Exception:
        pass
    return (None, None, None, None)

def _collect_label_value_pairs(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Traverse tables/dl/li to collect "label -> value" pairs.
    We normalize labels for matching against dictionaries.
    """
    pairs: Dict[str, str] = {}

    # table rows
    for tr in soup.select("table tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) >= 2:
            label = _normalize_label(cells[0].get_text(" ", strip=True))
            value = cells[-1].get_text(" ", strip=True)
            if label and value:
                pairs[label] = value

    # definition lists
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            label = _normalize_label(dt.get_text(" ", strip=True))
            value = dd.get_text(" ", strip=True)
            if label and value:
                pairs[label] = value

    # list items with "Label: Value"
    for li in soup.find_all("li"):
        txt = li.get_text(" ", strip=True)
        if ":" in txt:
            label, value = txt.split(":", 1)
            label = _normalize_label(label)
            value = value.strip()
            if label and value:
                pairs[label] = value

    return pairs

def _pick_first(soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
    for css in selectors:
        el = soup.select_one(css)
        if el:
            return el.get_text(" ", strip=True)
    return None

# -------------------------
# Scraper
# -------------------------

def _make_driver() -> uc.Chrome:
    headless = str(os.getenv("IMMO_HEADLESS", "true")).lower() == "true"
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1200,1600")
    try:
        driver = uc.Chrome(options=opts)
    except WebDriverException as e:
        logger.warning("Falling back to non-undetected Chrome: %s", e)
        driver = uc.Chrome(options=opts)  # still uc; rarely fails twice
    driver.set_page_load_timeout(60)
    return driver

def _latest_consolidated_csv(kind: str) -> Optional[Path]:
    today = datetime.now()
    root = Path("data/raw") / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}" / kind
    # pattern of consolidate step we used earlier
    hits = sorted(root.glob("_consolidated_towns_urls_*/*.csv"))
    return hits[-1] if hits else None

def _latest_urls_parquet(kind: str) -> Optional[Path]:
    today = datetime.now()
    root = Path("data/raw") / f"{today:%Y}" / f"{today:%m}" / f"{today:%d}"
    p = root / f"{kind}_urls.parquet"
    return p if p.exists() else None

def _load_input_urls(kind: str) -> List[str]:
    csv_path = _latest_consolidated_csv(kind)
    if csv_path:
        df = pd.read_csv(csv_path)
        # Expect a column "url" or the only column
        if "url" in df.columns:
            return df["url"].dropna().astype(str).tolist()
        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().astype(str).tolist()
    pq = _latest_urls_parquet(kind)
    if pq:
        df = pd.read_parquet(pq)
        if "url" in df.columns:
            return df["url"].dropna().astype(str).tolist()
        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().astype(str).tolist()
    raise FileNotFoundError(f"No consolidated URLs found for kind={kind}. Run consolidate step first.")

def _parse_detail_html(url: str, html_text: str) -> Dict:
    soup = BeautifulSoup(html_text, "lxml")

    title = _pick_first(soup, [
        "h1", "h1 span", "header h1", "div[class*=title] h1", "meta[property='og:title']"
    ]) or ""

    # collect label/value pairs from spec sections
    pairs = _collect_label_value_pairs(soup)
    norm_keys = {k: v for k, v in pairs.items()}

    # surface
    surface_val: Optional[float] = None
    for lbl in list(norm_keys.keys()):
        norm = _normalize_label(lbl)
        if any(k in norm for k in _SURFACE_LABELS):
            surface_val = _parse_surface_value(norm_keys[lbl])
            if surface_val:
                break
    if surface_val is None:
        # last resort: search any "xx m²" in the whole page
        text = soup.get_text(" ", strip=True)
        m2_guess = _parse_surface_value(text)
        surface_val = m2_guess

    # price from obvious spots first
    price_text = _pick_first(soup, [
        "[class*=price] [class*=value]",
        "[class*=price]", "span.price", "div.price", "meta[itemprop='price']",
    ])
    price = _parse_price(price_text) if price_text else None
    if price is None:
        price = _parse_price(soup.get_text(" ", strip=True))

    # bedrooms / bathrooms / year / energy label (scan by known labels + fallbacks)
    def _first_numeric_from_labels(label_set: set[str]) -> Optional[float]:
        for k, v in norm_keys.items():
            nk = _normalize_label(k)
            if nk in label_set:
                nums = _extract_numbers(v)
                if nums:
                    return float(nums[0])
        return None

    bedrooms = _first_numeric_from_labels(_BEDROOM_LABELS)
    bathrooms = _first_numeric_from_labels(_BATHROOM_LABELS)

    year_built = None
    for k, v in norm_keys.items():
        nk = _normalize_label(k)
        if nk in _YEAR_LABELS:
            nums = _extract_numbers(v)
            if nums:
                y = int(nums[0])
                if 1800 <= y <= date.today().year + 1:
                    year_built = y
                    break

    energy_label = None
    for k, v in norm_keys.items():
        nk = _normalize_label(k)
        if nk in _ENERGY_LABELS:
            # Capture A+, A, B, C, D, E, F, G
            m = re.search(r"\b(A\+|A|B|C|D|E|F|G)\b", v.upper())
            if m:
                energy_label = m.group(1)
                break

    # infer from URL if missing
    ptype_u, city_u, postal_u, is_sale_u = _from_url(url)

    # city
    city = None
    if "city" in norm_keys:
        city = norm_keys["city"]
    if not city:
        city = city_u

    # postal code
    postal_code = None
    if "postal code" in norm_keys:
        cand = _extract_numbers(norm_keys["postal code"])
        if cand:
            postal_code = f"{int(cand[0]):04d}"
    if not postal_code and postal_u and postal_u.isdigit():
        postal_code = postal_u

    # property type
    property_type = None
    if "property type" in norm_keys:
        property_type = norm_keys["property type"]
    if not property_type:
        property_type = ptype_u

    # rent/sale
    is_sale = is_sale_u

    record = {
        "snapshot_date": _today_str(),
        "url": url,
        "title": title,
        "city": city,
        "postal_code": postal_code,
        "property_type": property_type,
        "is_sale": bool(is_sale) if is_sale is not None else None,
        "price": price,
        "surface_m2": surface_val,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "energy_label": energy_label,
    }
    return record

# -------------------------
# Public API
# -------------------------

def run(kind: str, limit: Optional[int] = None) -> str:
    """
    Scrape details pages for 'apartments' or 'houses'.

    Returns the CSV path written.
    """
    kind = (kind or "").strip().lower()
    if kind not in {"apartments", "houses"}:
        raise ValueError("kind must be 'apartments' or 'houses'")

    urls = _load_input_urls(kind)
    if limit:
        urls = urls[: int(limit)]

    logger.info("🧭 Details input: %s URLs -> kind=%s (limit=%s)", len(urls), kind, limit)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    day_root = Path("data/raw") / f"{datetime.now():%Y}" / f"{datetime.now():%m}" / f"{datetime.now():%d}" / kind
    out_dir = day_root / f"_real_estate_details_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"_real_estate_details_{ts}.csv"

    driver = _make_driver()
    rows: List[Dict] = []

    try:
        for i, url in enumerate(urls, start=1):
            try:
                driver.get(url)
            except Exception as e:
                logger.warning("Driver.get failed: %s | %s", url, e)
                continue

            # let the page settle a bit; immovlan usually renders static content
            time.sleep(1.5)
            html_text = driver.page_source

            try:
                rec = _parse_detail_html(url, html_text)
                rows.append(rec)
                logger.info("✅ [%d/%d] Extracted: %s", i, len(urls), url)
            except Exception as e:
                logger.exception("Parse failed for %s: %s", url, e)

        if not rows:
            raise RuntimeError("No detail rows were extracted; check site structure or anti-bot measures.")

        df = pd.DataFrame(rows)
        # light type normalization here (full normalization occurs in cleaning)
        for intish in ["bedrooms", "bathrooms", "year_built"]:
            if intish in df.columns:
                df[intish] = pd.to_numeric(df[intish], errors="coerce").astype("Int64")
        for fl in ["price", "surface_m2"]:
            if fl in df.columns:
                df[fl] = pd.to_numeric(df[fl], errors="coerce")

        df.to_csv(out_csv, index=False)
        logger.info("💾 Saved detailed data to: %s", out_csv)
        logger.info("✅ Details scraping done for %s", kind)
        return str(out_csv)

    finally:
        try:
            driver.quit()
        except Exception:
            pass
