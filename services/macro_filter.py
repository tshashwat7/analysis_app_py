# services/macro_filter.py
import yfinance as yf
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Configurable Defaults
# ---------------------------------------------------
DEFAULT_PERIOD = "2y"
DEFAULT_SMA_LENGTH = 200
DEFAULT_INDEX = "nifty50"
DEFAULT_TICKER_MAP = {
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
    "us_sp500": "^GSPC",
    "default": "^NSEI",
}

# ---------------------------------------------------
# Lightweight Cache (to reduce redundant YF calls)
# ---------------------------------------------------
_macro_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 15 * 60  # 15 minutes


def _get_from_cache(key: str) -> Optional[Dict[str, Any]]:
    entry = _macro_cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL:
        del _macro_cache[key]
        return None
    return entry["data"]


def _set_cache(key: str, data: Dict[str, Any]):
    _macro_cache[key] = {"data": data, "ts": time.time()}


# ---------------------------------------------------
# Main Function
# ---------------------------------------------------
def check_macro_trend(
    index_name: str = DEFAULT_INDEX,
    ticker_map: Dict[str, str] = DEFAULT_TICKER_MAP,
    period: str = DEFAULT_PERIOD,
    sma_length: int = DEFAULT_SMA_LENGTH,
) -> Dict[str, Any]:
    """
    Checks the long-term trend of a specified macro index based on its SMA.

    Args:
        index_name: Friendly name (e.g., 'nifty50')
        ticker_map: Dict mapping index name to Yahoo Finance ticker
        period: History period (default '2y')
        sma_length: SMA period (default 200)

    Returns:
        dict: {
            "name": str,
            "trend": str,
            "last_close": str,
            "sma_value": float,
            "confidence": float,
            "score": int
        }
    """

    index_name = index_name.lower()
    ticker = ticker_map.get(index_name) or ticker_map.get("default", "^NSEI")
    cache_key = f"{index_name}_{ticker}_{sma_length}"

    # --- 1. Check Cache ---
    cached = _get_from_cache(cache_key)
    if cached:
        return cached

    try:
        # --- 2. Fetch Data ---
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

        if df.empty or len(df) < sma_length + 20:
            raise ValueError("Insufficient historical data")

        # --- 3. Calculate SMA ---
        df["SMA"] = df["Close"].rolling(window=sma_length).mean()
        df = df.dropna(subset=["SMA"])
        if df.empty:
            raise ValueError("Failed to calculate SMA")

        last_close = float(df["Close"].iloc[-1])
        last_sma = float(df["SMA"].iloc[-1])

        if pd.isna(last_close) or pd.isna(last_sma):
            raise ValueError("NaN values in last_close or last_sma")

        # --- 4. Determine Trend & Confidence ---
        diff_pct = ((last_close - last_sma) / last_sma) * 100

        if diff_pct > 2:
            trend = "STRONG BULLISH (Price > SMA +2%)"
            score = 10
        elif diff_pct > 0:
            trend = "MILD BULLISH (Price > SMA)"
            score = 7
        elif diff_pct < -2:
            trend = "STRONG BEARISH (Price < SMA -2%)"
            score = 0
        else:
            trend = "MILD BEARISH (Price < SMA)"
            score = 3

        confidence = round(abs(diff_pct), 2)
        result = {
            "name": index_name.upper(),
            "trend": trend,
            "last_close": f"₹{round(last_close, 2)} / 200SMA: ₹{round(last_sma, 2)}",
            "sma_value": round(last_sma, 2),
            "confidence": confidence,
            "score": score,
        }

        # --- 5. Cache the result ---
        _set_cache(cache_key, result)
        logger.info(f"✅ Macro trend for {index_name.upper()}: {trend} ({confidence:.2f}% deviation)")

        return result

    except Exception as e:
        logger.warning(f"⚠️ Macro Trend Error for {index_name.upper()} ({ticker}): {e}")
        result = {
            "name": index_name.upper(),
            "trend": f"ERROR ({type(e).__name__})",
            "last_close": "N/A",
            "sma_value": None,
            "confidence": 0.0,
            "score": 0,
        }
        _set_cache(cache_key, result)
        return result


# ---------------------------------------------------
# Manual Test
# ---------------------------------------------------
if __name__ == "__main__":
    res = check_macro_trend("nifty50")
    print(res)
