from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.immovlan.be"

def extract_listing_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = set()

    # common patterns (broad on purpose)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href or href.startswith("#"):
            continue
        # heuristics for listing detail pages
        if "/detail/" in href or "/classified/" in href or "/en/for-sale/" in href or "/en/for-rent/" in href:
            full = urljoin(BASE, href)
            # ignore obvious non-detail endpoints
            if any(x in full for x in ("/search", "/map", "/news", "/about", "/contact")):
                continue
            urls.add(full.split("?")[0])
    return list(urls)
