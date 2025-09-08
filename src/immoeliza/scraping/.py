from datetime import datetime
import os, time, glob
import pandas as pd
from .common import logger, ScrapeConfig
from .legacy import patch_driver  # ensures webdriver is configured
from .legacy.immovlan_url_scraper import ImmovlanUrlScraper
from .html_fallback import extract_listing_urls
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

KIND = "__KIND__"
DEFAULT_TEMPLATE = os.getenv(
    f"IMMOVLAN_{KIND.upper()}_BASE_URL_TEMPLATE",
    "https://www.immovlan.be/en/real-estate/for-sale/__PATH__?search={town}"
)

def _base_url_for_town(town: str) -> str:
    return DEFAULT_TEMPLATE.format(town=str(town).strip())

def _towns_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'town' not in df.columns:
        df.columns = [c.strip() for c in df.columns]
        df.rename(columns={df.columns[0]: 'town'}, inplace=True)
    return df

def _simple_driver(headless: bool):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1366,768")
    ua = os.getenv("IMMO_SCRAPE_USER_AGENT")
    if ua: opts.add_argument(f"--user-agent={ua}")
    lang = os.getenv("IMMO_ACCEPT_LANGUAGE", "en-US,en;q=0.9")
    opts.add_argument(f"--lang={lang.split(',')[0]}")
    opts.add_experimental_option("prefs", {"intl.accept_languages": lang})
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

def _write_fallback_csv(out_dir: str, town: str, run_id: str, urls: list[str]):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"fallback_urls_{town}_{run_id}_{stamp}_records_{len(urls)}.csv")
    pd.DataFrame({"url": urls}).to_csv(path, index=False)
    logger.info("🛟 Fallback saved %d URLs -> %s", len(urls), path)
    return path

def run(max_pages: int = None, headless: bool = None, delay_min: float = 1.0, delay_max: float = 2.0):
    cfg = ScrapeConfig()
    if max_pages is None: max_pages = cfg.max_pages
    if headless is None: headless = cfg.headless

    df = _towns_df(cfg.towns_csv)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(cfg.raw_dir, datetime.now().strftime("%Y/%m/%d"), KIND)
    os.makedirs(out_root, exist_ok=True)

    logger.info("🔎 %s URL scrape | towns=%d | headless=%s | max_pages=%s", KIND.capitalize(), len(df), headless, max_pages)

    for town in df["town"].astype(str):
        base_url = _base_url_for_town(town)
        town_glob = os.path.join(out_root, f"{town}_{run_id}", "*.csv")

        # 1) Try legacy scraper (wire if available, otherwise it still runs)
        try:
            scraper = ImmovlanUrlScraper(
                base_url=base_url, town=town, max_pages=max_pages,
                delay_min=delay_min, delay_max=delay_max,
                run_id=run_id, output_dir=out_root, headless=headless
            )
            try:
                scraper.scrape_and_save_urls()
            finally:
                try: scraper.close()
                except Exception: pass
        except Exception as e:
            logger.warning("Legacy scraper failed for %s: %s", town, e)

        # 2) If legacy produced no CSVs, run HTML fallback on first page
        if not glob.glob(town_glob):
            try:
                drv = _simple_driver(headless=headless)
                url = f"{base_url}&page=1" if "?" in base_url else f"{base_url}?page=1"
                drv.get(url)
                try:
                    WebDriverWait(drv, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
                except TimeoutException:
                    pass
                urls = extract_listing_urls(drv.page_source)
                drv.quit()
                if urls:
                    _write_fallback_csv(os.path.join(out_root, f"{town}_{run_id}"), town, run_id, urls)
                else:
                    logger.warning("Fallback found 0 URLs for %s", town)
            except Exception as e:
                logger.warning("Fallback failed for %s: %s", town, e)

    logger.info("✅ %s URL scraping finished.", KIND.capitalize())
    return f"{KIND}_ok"
