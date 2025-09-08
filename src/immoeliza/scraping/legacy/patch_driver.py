import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def _headless_env() -> bool:
    return os.getenv("IMMO_HEADLESS", "true").lower() != "false"

def Chrome(*args, **kwargs):
    opts = kwargs.pop("options", None) or Options()
    if _headless_env():
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1366,768")
    ua = os.getenv("IMMO_SCRAPE_USER_AGENT")
    if ua:
        opts.add_argument(f"--user-agent={ua}")
    lang = os.getenv("IMMO_ACCEPT_LANGUAGE", "en-US,en;q=0.9")
    opts.add_argument(f"--lang={lang.split(',')[0]}")
    opts.add_experimental_option("prefs", {"intl.accept_languages": lang})
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)
