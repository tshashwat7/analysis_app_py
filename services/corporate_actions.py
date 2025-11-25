"""
Optimized corporate_actions module.

- Bulk "upcoming" mode: Uses ONLY Equitymaster (no yfinance).
- Single-stock "past" / "single" mode: Uses yfinance once and caches results per-ticker.
- JSON sidecar cache for YF results at cache/yf_actions/{TICKER}.json
- Keeps your Equitymaster caching behavior for bulk API.
- Safe, resilient, and non-blocking for index loads and quick scans.
- FIXES: Strict name matching, Schema normalization, Stale data filtering.
"""

import os
import re
import json
import time
import math
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Optional (only used in single-stock mode)
import yfinance as yf
import pandas as pd

# Helpers from your codebase
from services.data_fetch import _fmt_date, _retry, safe_float

logger = logging.getLogger("corporate_actions")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(levelname)s] corporate_actions: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -------------------------------
# Local config
# -------------------------------
CACHE_PATH = "cache/equitymaster_actions.json"
CACHE_TTL_HOURS = 24

YF_CACHE_DIR = "cache/yf_actions"
os.makedirs(YF_CACHE_DIR, exist_ok=True)

# Equitymaster endpoints for different action types
_API_ENDPOINTS = {
    "Dividend": "https://www.equitymaster.com/eqtmapi/getDividendList?indexcode=1-71",
    "Bonus": "https://www.equitymaster.com/eqtmapi/getBonusList?indexcode=1-71",
    "Split": "https://www.equitymaster.com/eqtmapi/getSplitList?indexcode=1-71",
}


# -------------------------------
# Utilities
# -------------------------------
def normalize_company_name(name: str) -> str:
    """Removes common suffixes for stricter matching."""
    if not name: return ""
    name = name.lower()
    name = name.replace(" ltd", "").replace(" limited", "").replace(" plc", "").replace(" corp", "")
    return re.sub(r"\s+", " ", name).strip()

# -------------------------------
# Equitymaster fetch (bulk upcoming) - preserved with caching
# -------------------------------
def _fetch_action(action_type: str, url: str, headers: dict) -> List[Dict[str, Any]]:
    actions = []
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if not resp.ok:
            logger.warning("Equitymaster %s fetch failed status=%s", action_type, resp.status_code)
            return actions

        data_json = resp.json()
        rows = data_json.get("aaData", []) or []
        for row in rows:
            # row typically has [company, ???, value, ex_date, ...]
            if len(row) < 4:
                continue
            name = str(row[0]).strip()
            raw_value = row[2]
            raw_ex = row[3]

            value = None
            try:
                if action_type == "Dividend":
                    m = re.search(r"([\d.]+)", str(raw_value))
                    if m:
                        value = float(m.group(1))
                else:
                    value = str(raw_value).strip()
            except Exception:
                logger.debug("Failed parse equitymaster value for %s: %s", name, raw_value)

            ex_date = None
            try:
                ex_date = datetime.strptime(raw_ex.strip(), "%d-%b-%Y").date()
            except Exception:
                logger.debug("Failed parse equitymaster ex_date for %s: %s", name, raw_ex)
                continue

            if ex_date:
                actions.append({"name": name, "type": action_type, "value": value, "ex_date": ex_date})
    except Exception as e:
        logger.warning("Equitymaster fetch failed for %s: %s", action_type, e)
    return actions


def _fetch_equitymaster_data() -> List[Dict[str, Any]]:
    """Fetch Equitymaster upcoming/bulk data with caching on disk."""
    try:
        if os.path.exists(CACHE_PATH):
            mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
            if datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS):
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                # normalize ex_date back to date
                for item in cached:
                    if isinstance(item.get("ex_date"), str):
                        try:
                            item["ex_date"] = datetime.strptime(item["ex_date"], "%Y-%m-%d").date()
                        except Exception:
                            pass
                logger.info("Loaded %d items from equitymaster cache", len(cached))
                return cached
    except Exception as e:
        logger.warning("Failed to read EM cache: %s", e)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.equitymaster.com/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    all_data = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_fetch_action, t, url, headers) for t, url in _API_ENDPOINTS.items()]
        for f in futures:
            try:
                all_data.extend(f.result() or [])
            except Exception as e:
                logger.warning("Partial Equitymaster fetch failed: %s", e)

    # persist cache
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([{"name": it["name"], "type": it["type"], "value": it["value"], "ex_date": str(it["ex_date"])}
                       for it in all_data], f, indent=2)
        logger.info("Saved %d EM records to cache", len(all_data))
    except Exception as e:
        logger.warning("Failed to write EM cache: %s", e)

    return all_data


# -------------------------------
# Bulk upcoming actions (YF-FREE)
# -------------------------------
def get_bulk_upcoming_actions(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Return upcoming corporate actions for a list of tickers using Equitymaster only.
    NO yfinance calls here. Safe for bulk/index loads.
    """
    try:
        em_data = _fetch_equitymaster_data()
    except Exception as e:
        logger.warning("Equitymaster fetch error: %s", e)
        em_data = []

    results: List[Dict[str, Any]] = []
    
    # Pre-calculate threshold for stale data (Option 5)
    cutoff_date = datetime.now().date() - timedelta(days=7)

    for ticker in tickers:
        try:
            # FIX 1: Strict Prefix Matching
            # Clean the ticker key (e.g. "ADANIENT.NS" -> "adanient")
            key = ticker.replace(".NS", "").replace(".BSE", "").lower()
            
            matches = []
            for item in em_data:
                try:
                    # FIX 1: Use normalized prefix match instead of 'in'
                    # "Adani Enterprises Ltd" -> "adani enterprises"
                    norm_name = normalize_company_name(item["name"])
                    
                    # Check 1: Does company name start with ticker key? (e.g. 'adani' starts with 'adani')
                    # Check 2: Does ticker key start with company first word? (e.g. 'adanient' starts with 'adani' - simplistic)
                    # We prioritize the User's strict logic:
                    if norm_name.startswith(key):
                        matches.append(item)
                    # Fallback for strict containment if startswith fails but key is long (e.g. TATASTEEL)
                    elif len(key) > 4 and key in norm_name:
                         matches.append(item)
                except Exception:
                    continue

            upcoming = []
            for m in matches:
                # FIX 2 & 4: Safe Date Parsing
                raw_date = m.get("ex_date")
                ex_obj = None
                
                if isinstance(raw_date, (datetime, datetime.date)):
                    ex_obj = raw_date
                elif isinstance(raw_date, str):
                    try:
                        ex_obj = datetime.strptime(raw_date, "%Y-%m-%d").date()
                    except: pass
                
                # FIX 5: Stale Data Filter
                if ex_obj and isinstance(ex_obj, (datetime, datetime.date)):
                    # Ensure we compare date to date
                    d_check = ex_obj.date() if isinstance(ex_obj, datetime) else ex_obj
                    if d_check < cutoff_date:
                        continue

                upcoming.append({
                    "type": f"Upcoming {m['type']}",
                    "value": m.get("value"),
                    "ex_date": _fmt_date(ex_obj) if ex_obj else str(raw_date)
                })

            results.append({"ticker": ticker, "actions": upcoming, "source": "equitymaster"})
        except Exception as e:
            logger.debug("Error matching EM for %s: %s", ticker, e)
            results.append({"ticker": ticker, "actions": [], "source": "equitymaster"})
    return results


# -------------------------------
# Single-stock detailed YF enrichment (cached per ticker)
# -------------------------------
def _yf_cache_path(ticker: str) -> str:
    safe_t = ticker.replace("/", "_").replace("\\", "_").upper()
    return os.path.join(YF_CACHE_DIR, f"{safe_t}.json")


def _load_yf_cache(ticker: str) -> Dict[str, Any]:
    p = _yf_cache_path(ticker)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_yf_cache(ticker: str, payload: Dict[str, Any]):
    p = _yf_cache_path(ticker)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str, indent=2)
    except Exception as e:
        logger.warning("Failed to write YF cache for %s: %s", ticker, e)


def get_single_stock_history_yf(ticker: str, lookback_days: int = 365, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetch and cache single-stock corporate actions using yfinance.
    This function is intended ONLY for /analyze detail views.
    """
    ticker = ticker.strip().upper()
    cache = _load_yf_cache(ticker)
    if cache and not force_refresh:
        # Respect TTL if present
        fetched_at = cache.get("_fetched_at")
        try:
            if fetched_at:
                dt = datetime.fromisoformat(fetched_at)
                if datetime.now() - dt < timedelta(days=7):
                    return cache
        except Exception:
            pass

    # Fetch fresh from Yahoo (retry-protected)
    try:
        def _do():
            t = yf.Ticker(ticker)
            corp_df = getattr(t, "actions", None)
            if corp_df is None:
                corp_df = pd.DataFrame()

            if corp_df is not None and not getattr(corp_df, "empty", True):
                corp_df = pd.to_datetime(corp_df.index, errors="coerce")
            
            try:
                splits = getattr(t, "splits", pd.Series(dtype="float64"))
            except Exception:
                splits = pd.Series(dtype="float64")
            try:
                dividends = getattr(t, "dividends", pd.Series(dtype="float64"))
            except Exception:
                dividends = pd.Series(dtype="float64")

            # Build events list
            events = []
            for idx, val in (dividends.items() if getattr(dividends, "any", lambda: False)() else []):
                try:
                    d = idx.date() if hasattr(idx, "date") else idx
                    # FIX 3: Schema Consistency (value instead of amount)
                    events.append({"type": "Dividend", "value": float(val), "ex_date": _fmt_date(d)})
                except Exception:
                    continue

            for idx, val in (splits.items() if getattr(splits, "any", lambda: False)() else []):
                try:
                    d = idx.date() if hasattr(idx, "date") else idx
                    # FIX 3: Schema Consistency (value instead of ratio)
                    events.append({"type": "Split", "value": str(val), "ex_date": _fmt_date(d)})
                except Exception:
                    continue

            # Sort by date descending
            events = sorted(events, key=lambda x: x.get("ex_date", ""), reverse=True)

            return {
                "ticker": ticker,
                "actions": events,  
                "_fetched_at": datetime.now().isoformat(),
                "source": "yfinance"
            }

        payload = _retry(_do, retries=2, backoff=0.5, name=f"yf_{ticker}")
        _save_yf_cache(ticker, payload)
        return payload
    except Exception as e:
        logger.warning("YF enrichment failed for %s: %s", ticker, e)
        if cache:
            return cache
        return {"ticker": ticker, "actions": [], "_fetched_at": None, "source": "yfinance_fallback"}


# -------------------------------
# Master public API
# -------------------------------
def get_corporate_actions(tickers: List[str], mode: str = "past", lookback_days: int = 365, force_refresh_yf: bool = False):
    """
    Unified API:
      - mode="upcoming" -> bulk Equitymaster-only (no YF)
      - mode="single" or "past" -> single-stock YF enrichment (cached)
    """
    mode = (mode or "past").lower()
    if mode == "upcoming":
        return get_bulk_upcoming_actions(tickers)

    if mode in ("single", "past"):
        if not tickers:
            return []
        ticker = tickers[0]
        return [get_single_stock_history_yf(ticker, lookback_days=lookback_days, force_refresh=force_refresh_yf)]

    raise ValueError("Unsupported mode for corporate actions: %s" % mode)


# -------------------------------
# CLI / debug
# -------------------------------
if __name__ == "__main__":
    # Example quick test
    print(json.dumps(get_corporate_actions(["INFY.NS"], mode="upcoming"), indent=2))
    print(json.dumps(get_corporate_actions(["INFY.NS"], mode="single"), indent=2))