import logging
import os
from dataclasses import dataclass

# Basic logger used across scraping modules
logger = logging.getLogger("immoeliza.scraping")
if not logger.handlers:
    level = os.getenv("IMMO_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

@dataclass
class ScrapeConfig:
    towns_csv: str = os.getenv("IMMO_TOWNS_CSV", "data/immovlan_towns_to_scrape.csv")
    raw_dir: str = os.getenv("IMMO_DATA_ROOT", "data") + "/raw"
    headless: bool = os.getenv("IMMO_HEADLESS", "true").lower() != "false"
    max_pages: int = int(os.getenv("IMMO_MAX_PAGES", "-1"))
