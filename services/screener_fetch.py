# services/screener_fetch.py
import logging
import re
import requests
import math
from bs4 import BeautifulSoup
from functools import lru_cache
from services.data_fetch import _retry, safe_float, normalize_ratio

logger = logging.getLogger(__name__)

BASE_URL = "https://www.screener.in/company/{}/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/1.0; +https://example.com)"
}

@lru_cache(maxsize=128)
def fetch_screener_data(symbol: str) -> dict:
    """
    Fetches supplemental fundamental data from Screener.in and normalizes numeric fields.
    Only used as enrichment source (does not overwrite valid Yahoo data).
    """

    if not symbol:
        return {}

    result = {}
    sym = symbol.replace(".NS", "").upper()

    try:
        html = _retry(lambda: requests.get(BASE_URL.format(sym), headers=HEADERS, timeout=10).text, retries=3, backoff=1)
        soup = BeautifulSoup(html, "html.parser")

        # --- Extract all key-value pairs ---
        for li in soup.select("li.flex.flex-space-between"):
            spans = li.find_all("span")
            if len(spans) >= 2:
                key = spans[0].get_text(strip=True)
                val = spans[-1].get_text(strip=True)
                result[key] = val

        html_text = soup.get_text(" ", strip=True)

        def find_value(pattern):
            match = re.search(pattern, html_text, re.IGNORECASE)
            return match.group(1).strip() if match else None

        parsed = {
            "PEG Ratio": safe_float(result.get("PEG Ratio") or find_value(r"PEG\s*Ratio\s*([\d\.\-]+)")),
            "ROCE (%)": normalize_ratio(result.get("ROCE") or find_value(r"ROCE\s*([\d\.\-]+)%")),
            "ROIC (%)": normalize_ratio(result.get("ROIC") or find_value(r"ROIC\s*([\d\.\-]+)%")),
            "Debt to Equity": safe_float(result.get("Debt to equity") or find_value(r"Debt\s*to\s*Equity\s*([\d\.]+)")),
            "Interest Coverage": safe_float(result.get("Interest Coverage Ratio") or find_value(r"Interest\s*Coverage\s*([\d\.]+)")),
            "Promoter Holding (%)": normalize_ratio(result.get("Promoter Holding") or find_value(r"Promoter\s*Holding\s*([\d\.]+)%")),
            "Promoter Pledge (%)": normalize_ratio(result.get("Pledged") or find_value(r"Pledged\s*([\d\.]+)%")),
            "Institutional Ownership (%)": normalize_ratio(result.get("FII + DII") or find_value(r"FII\s*\+\s*DII\s*([\d\.]+)%")),
            "EPS Growth (5Y)": normalize_ratio(result.get("EPS Growth 5Years") or find_value(r"EPS\s*Growth\s*5Y\s*([\d\.]+)%")),
            "Market Cap CAGR (5Y)": normalize_ratio(result.get("Market Cap 5Years CAGR") or find_value(r"Market\s*Cap\s*5Y\s*CAGR\s*([\d\.]+)%")),
        }

        clean = {}
        for k, v in parsed.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            clean[k] = {"raw": v, "value": f"{v:.2f}%" if "Growth" in k or "(%)" in k else v, "score": 0}

        logger.info("✅ Screener parsed %d fields for %s", len(clean), sym)
        return clean

    except Exception as e:
        logger.warning("⚠️ Screener fetch failed for %s: %s", sym, e)
        return {}
