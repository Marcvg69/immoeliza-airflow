from datetime import datetime
import os
import pandas as pd
from .common import logger, ScrapeConfig
from .legacy import patch_driver  # noqa: F401 (ensures webdriver is patched)
from .legacy.immovlan_url_scraper import ImmovlanUrlScraper

DEFAULT_TEMPLATE = os.getenv(
    "IMMOVLAN_APARTMENTS_BASE_URL_TEMPLATE",
    "https://www.immovlan.be/en/real-estate/for-sale/apartment?search={town}"
)

def _base_url_for_town(town: str) -> str:
    return DEFAULT_TEMPLATE.format(town=str(town).strip())

def _towns_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'town' not in df.columns:
        df.columns = [c.strip() for c in df.columns]
        # fallback: use first column as 'town'
        df.rename(columns={df.columns[0]: 'town'}, inplace=True)
    return df

def run(max_pages: int = None, headless: bool = None, delay_min: float = 1.0, delay_max: float = 2.0):
    cfg = ScrapeConfig()
    if max_pages is None: max_pages = cfg.max_pages
    if headless is None: headless = cfg.headless

    df = _towns_df(cfg.towns_csv)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("🏙️ Apartments scrape | towns=%d | headless=%s | max_pages=%s", len(df), headless, max_pages)
    out_root = os.path.join(cfg.raw_dir, datetime.now().strftime("%Y/%m/%d"), "apartments")
    os.makedirs(out_root, exist_ok=True)

    for town in df["town"].astype(str):
        base_url = _base_url_for_town(town)
        scraper = ImmovlanUrlScraper(
            base_url=base_url,
            town=town,
            max_pages=max_pages,
            delay_min=delay_min,
            delay_max=delay_max,
            run_id=run_id,
            output_dir=out_root,     # write under kind-specific folder
            headless=headless
        )
        try:
            scraper.scrape_and_save_urls()
        finally:
            try: scraper.close()
            except Exception: pass

    logger.info("✅ Apartments URL scraping done.")
    return "apartments_ok"
