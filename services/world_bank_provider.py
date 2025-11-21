# services/data_provider.py — v4 (with macro metrics + Finnhub)
import logging
import os
import requests
from functools import lru_cache
import yfinance as yf

from world_bank_data import get_series
from .data_fetch import safe_float

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# macOS / Linux
FINNHUB_API_KEY="d43kft1r01qvk0jcbd00d43kft1r01qvk0jcbd0g"


# -------------------------------------------------------
# 2️⃣ Macro-Level Economic Indicators
# -------------------------------------------------------
@lru_cache(maxsize=1)
def get_macro_metrics() -> dict:
    """
    Return macroeconomic indicators like GDP, Inflation, Interest, Crude, etc.
    Cached for efficiency — these values don’t change often.
    """
    metrics = {}

    # --- GDP Growth ---
    try:
        gdp = get_series("NY.GDP.MKTP.KD.ZG", country="IN").iloc[-1]
        metrics["GDP Growth (%)"] = {"value": round(float(gdp), 2), "src": "WorldBank"}
    except Exception:
        metrics["GDP Growth (%)"] = {"value": "N/A", "src": "WorldBank"}

    # --- Inflation ---
    try:
        infl = get_series("FP.CPI.TOTL.ZG", country="IN").iloc[-1]
        metrics["Inflation Rate (%)"] = {"value": round(float(infl), 2), "src": "WorldBank"}
    except Exception:
        metrics["Inflation Rate (%)"] = {"value": "N/A", "src": "WorldBank"}

    # --- Crude Oil ---
    try:
        crude = yf.Ticker("CL=F").history(period="1mo")["Close"].iloc[-1]
        metrics["Crude Oil ($)"] = {"value": round(float(crude), 2), "src": "YFinance"}
    except Exception:
        metrics["Crude Oil ($)"] = {"value": "N/A", "src": "YFinance"}

    # --- USD/INR ---
    try:
        inr = yf.Ticker("USDINR=X").history(period="1mo")["Close"].iloc[-1]
        metrics["USD/INR"] = {"value": round(float(inr), 2), "src": "YFinance"}
    except Exception:
        metrics["USD/INR"] = {"value": "N/A", "src": "YFinance"}

    logger.info(f"🌍 [Macro] Metrics fetched: {metrics}")
    return metrics


# -------------------------------------------------------
# Optional helper: merge both company + macro
# -------------------------------------------------------
def get_all_data() -> dict:
    """
    Combined view: company fundamentals + macro indicators.
    Used by fundamentals.py or dashboards.
    """
    macro = get_macro_metrics()
    print(f"Merging fundamentals and macro for {macro}")

    return macro
