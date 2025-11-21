"""
scrapper/moneycontrol_scraper.py
--------------------------------
Scrapes key Indian macroeconomic and sentiment indicators from Moneycontrol.
Designed for integration into stock analyzer backend.

Requires:
 - requests
 - beautifulsoup4
 - Your project's datafetch._retry, safe_float utilities
"""

import requests
from bs4 import BeautifulSoup
import logging
import re
import time
from services.data_fetch import _retry, safe_float

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

BASE_URLS = {
    "vix": "https://www.moneycontrol.com/indian-indices/india-vix-36.html",
    "inflation": "https://www.moneycontrol.com/economy/inflation.html",
    "gdp": "https://www.moneycontrol.com/economy/gdp.html",
    "repo": "https://www.moneycontrol.com/economy/interest-rates.html",
    "crude": "https://www.moneycontrol.com/commodity/crudeoil-price.html",
    "currency": "https://www.moneycontrol.com/currency/",
    "bonds": "https://www.moneycontrol.com/stocks/bonds/india-bonds.html",
}


def _fetch_html(url: str) -> str:
    """Fetch HTML content safely with retry and graceful logging."""
    def _inner():
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.text

    try:
        return _retry(_inner, retries=2, backoff=0.5)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return ""


def _extract_first_number(text):
    """Extract first numeric value from text like '6.72%'."""
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return safe_float(m.group(0)) if m else None


def get_vix():
    """Scrape India VIX index value."""
    html = _fetch_html(BASE_URLS["vix"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        val = soup.select_one("span#lastPrice, span.last_price")
        return safe_float(_extract_first_number(val.text)) if val else None
    except Exception as e:
        logger.warning("VIX parse failed: %s", e)
        return None


def get_inflation():
    """Scrape latest CPI inflation rate (approx monthly)."""
    html = _fetch_html(BASE_URLS["inflation"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        # look for pattern like "CPI inflation at 5.4%"
        match = re.search(r"([Cc][Pp][Ii].{0,30}?(\d+(\.\d+)?)%)", text)
        if match:
            return _extract_first_number(match.group(2))
        return None
    except Exception as e:
        logger.warning("Inflation parse failed: %s", e)
        return None


def get_gdp():
    """Scrape India GDP growth rate (YoY)."""
    html = _fetch_html(BASE_URLS["gdp"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        match = re.search(r"GDP\s*growth\s*(?:at|is)?\s*([0-9.]+)%", text, re.I)
        return _extract_first_number(match.group(1)) if match else None
    except Exception as e:
        logger.warning("GDP parse failed: %s", e)
        return None


def get_interest_rate():
    """Scrape RBI repo rate."""
    html = _fetch_html(BASE_URLS["repo"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        match = re.search(r"Repo\s*Rate\s*([0-9.]+)%", text, re.I)
        return _extract_first_number(match.group(1)) if match else None
    except Exception as e:
        logger.warning("Repo rate parse failed: %s", e)
        return None


def get_crude_price():
    """Scrape current crude oil price."""
    html = _fetch_html(BASE_URLS["crude"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        val = soup.select_one("span.last_price")
        return safe_float(_extract_first_number(val.text)) if val else None
    except Exception as e:
        logger.warning("Crude price parse failed: %s", e)
        return None


def get_currency_rate():
    """Scrape USD/INR exchange rate."""
    html = _fetch_html(BASE_URLS["currency"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        val = soup.select_one("span.last_price")
        return safe_float(_extract_first_number(val.text)) if val else None
    except Exception as e:
        logger.warning("Currency rate parse failed: %s", e)
        return None


def get_bond_yield():
    """Scrape 10-year government bond yield."""
    html = _fetch_html(BASE_URLS["bonds"])
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        match = re.search(r"10\s*Year\s*Yield\s*([0-9.]+)%", text, re.I)
        return _extract_first_number(match.group(1)) if match else None
    except Exception as e:
        logger.warning("Bond yield parse failed: %s", e)
        return None
