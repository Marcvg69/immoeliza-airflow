import os
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from immoeliza.scraping.common import logger
# use our factory (supports IMMO_DRIVER=selenium|uc and IMMO_HEADLESS)
from .patch_driver import Chrome as ChromeFactory

BASE = "https://www.immovlan.be"

def _extract_listing_urls(html: str) -> List[str]:
    """Heuristic extraction of listing detail links from page HTML."""
    soup = BeautifulSoup(html, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#"):
            continue
        # detail-ish patterns; adjust as needed
        if any(p in href for p in ("/detail/", "/classified/", "/en/for-sale/", "/en/for-rent/")):
            full = urljoin(BASE, href)
            if any(x in full for x in ("/search", "/map", "/news", "/about", "/contact")):
                continue
            urls.add(full.split("?")[0])
    return list(urls)

@dataclass
class ImmovlanUrlScraper:
    base_url: str
    town: str
    max_pages: int = -1
    delay_min: float = 1.0
    delay_max: float = 2.0
    run_id: str = ""
    output_dir: str = "data/raw"
    headless: bool = True

    def __post_init__(self):
        # prepare output folder structure: {output_dir}/{YYYY/MM/DD}/{kind}/{town_runid}/
        # the callers pass output_dir as .../YYYY/MM/DD/{kind}
        self.town = str(self.town).strip()
        self.town_dir = Path(self.output_dir) / f"{self.town}_{self.run_id}"
        self.town_dir.mkdir(parents=True, exist_ok=True)
        self.driver = self._init_driver()

    # compatibility helpers (legacy code expected these in places)
    def _has_wire(self) -> bool:
        return hasattr(self.driver, "requests")

    def _init_driver(self):
        # our factory honors IMMO_HEADLESS/IMMO_DRIVER
        drv = ChromeFactory()
        try:
            drv.set_window_size(1366, 768)
        except Exception:
            pass
        return drv

    def _page_url(self, page: int) -> str:
        # append &page= if there's already a query, else ?page=
        if "?" in self.base_url:
            return f"{self.base_url}&page={page}"
        return f"{self.base_url}?page={page}"

    def _save_partial(self, page: int, urls: List[str]):
        p = self.town_dir / f"partial_urls_page_{page}_{self.town}_{self.run_id}.csv"
        if urls:
            import pandas as pd
            pd.DataFrame({"url": urls}).to_csv(p, index=False)
            logger.info("📝 Saved %d URLs from page %s -> %s", len(urls), page, p)
        else:
            logger.warning("No URLs found on page %s", page)

    def _write_stats(self, total_urls: int):
        txt = self.town_dir / f"stats_{self.town}_{self.run_id}.txt"
        txt.write_text(f"town={self.town}\nrun_id={self.run_id}\ntotal_urls={total_urls}\n", encoding="utf-8")
        logger.info("📊 Stats saved to %s", txt)

    def get_all_listing_urls(self, town_name: str):
        """Legacy public method name kept for compatibility (not strictly needed)."""
        return self._scrape_pages()

    def _scrape_pages(self) -> int:
        maxp = 1 if self.max_pages == 0 else (self.max_pages if self.max_pages > 0 else 50)
        total = 0
        for page in range(1, maxp + 1):
            url = self._page_url(page)
            logger.info("👉📄 Visiting page %s: %s", page, url)
            try:
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
                except TimeoutException:
                    logger.warning("Timeout waiting for anchors on page %s", page)
                urls = _extract_listing_urls(self.driver.page_source)
                self._save_partial(page, urls)
                total += len(urls)
                # basic short sleep to be polite
                time.sleep(self.delay_min)
            except Exception as e:
                logger.warning("Page %s failed: %s", page, e)
                break
        # combined rollup file (optional)
        try:
            import pandas as pd, glob
            parts = sorted(glob.glob(str(self.town_dir / f"partial_urls_page_*_{self.town}_{self.run_id}.csv")))
            if parts:
                df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
                df.drop_duplicates(subset=["url"], inplace=True)
                out = self.town_dir / f"urls_{self.town}_{self.run_id}_records_{len(df)}.csv"
                df.to_csv(out, index=False)
                logger.info("💾 Rolled up %d unique URLs -> %s", len(df), out)
                total = len(df)
        except Exception as e:
            logger.warning("Rollup step failed: %s", e)
        self._write_stats(total)
        return total

    def scrape_and_save_urls(self):
        return self._scrape_pages()

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass
