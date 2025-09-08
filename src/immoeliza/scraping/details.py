from datetime import datetime
from pathlib import Path
import pandas as pd
from .common import logger, ScrapeConfig
from ..io_paths import today_partition, raw_path
from .legacy.immovlan_details_scraper import ImmovlanDetailsScraper

def run(kind: str, limit: int | None = None):
    # load consolidated URL parquet
    urls_parquet = Path(raw_path(f"{kind}_urls"))
    if not urls_parquet.exists():
        raise FileNotFoundError(f"Missing URL parquet: {urls_parquet}")
    urls = pd.read_parquet(urls_parquet)["url"].dropna().unique().tolist()
    if limit and limit > 0:
        urls = urls[:limit]

    out_dir = today_partition() / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy details scraper expects a consolidated CSV in a *_consolidated_towns_urls_* folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tmp_dir = out_dir / f"_consolidated_towns_urls_{stamp}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = tmp_dir / f"_consolidated_towns_urls_{stamp}.csv"
    pd.DataFrame({"url": urls}).to_csv(tmp_csv, index=False)

    logger.info("🧭 Details input: %s (%d URLs)", tmp_csv, len(urls))

    scraper = ImmovlanDetailsScraper(
        output_dir=str(out_dir),
        headless=ScrapeConfig().headless,
        limit=limit or -1
    )
    try:
        scraper.scrape_and_save_properties()
    finally:
        try: scraper.close()
        except Exception: pass

    logger.info("✅ Details scraping done for %s", kind)
    return f"details_{kind}_ok"
