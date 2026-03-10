"""
corporate_actions module.

Upcoming bulk mode:
  PRIMARY  → india-corp-actions library (NSE API, symbol-exact, no name guessing)
  FALLBACK → Equitymaster (name-based match, preserved for resilience)

Past/single mode:
  yfinance (cached per ticker at cache/yf_actions/{TICKER}.json, TTL 7d)

Summary cache:
  cache/corp_actions_summary.json (TTL 24h) — pre-built flat map for fast index loads.

Install the primary library:
  pip install india-corp-actions
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

# Primary data source: india-corp-actions library (pip install india-corp-actions)
try:
    from india_corp_actions import IndiaCorpActions as _IndiaCorpActions
    _LIB_AVAILABLE = True
except ImportError:
    _IndiaCorpActions = None
    _LIB_AVAILABLE = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning(
        "india-corp-actions library not found. "
        "Run: pip install india-corp-actions  "
        "Falling back to Equitymaster only."
    )

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(levelname)s] corporate_actions: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)


# -------------------------------
# Local config
# -------------------------------
CACHE_PATH = "cache/equitymaster_actions.json"
CACHE_TTL_HOURS = 24

YF_CACHE_DIR = "cache/yf_actions"
os.makedirs(YF_CACHE_DIR, exist_ok=True)

SUMMARY_CACHE_PATH = "cache/corp_actions_summary.json"
SUMMARY_CACHE_TTL_HOURS = 24

NSE_LIB_CACHE_PATH = "cache/nse_corp_actions_lib.json"
NSE_LIB_CACHE_TTL_HOURS = 24

# Lazy singleton — created once, reused across all bulk calls
_lib_client = None

def _get_lib_client():
    """Return a cached IndiaCorpActions client instance."""
    global _lib_client
    if _lib_client is None and _LIB_AVAILABLE:
        _lib_client = _IndiaCorpActions()
    return _lib_client

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
                # logger.info("Loaded %d items from equitymaster cache", len(cached))
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
# NSE → library fetch (symbol-exact, primary)
# -------------------------------
def _fetch_nse_via_lib(tickers: List[str]) -> Dict[str, List[Dict]]:
    """
    Use india-corp-actions library to fetch upcoming actions from NSE.
    Returns dict keyed by ticker (e.g. "INFY.NS") -> list of action dicts
    in the same schema as the rest of this module.
    """
    client = _get_lib_client()
    if client is None:
        return {}

    # Library expects bare NSE symbols ("INFY", "TCS") — strip .NS suffix
    # We'll re-key results back to original ticker format after
    bare_symbols = [t.replace(".NS", "").replace(".BSE", "").upper() for t in tickers]
    ticker_map = {t.replace(".NS", "").replace(".BSE", "").upper(): t for t in tickers}

    result_by_ticker: Dict[str, List[Dict]] = {}
    try:
        actions_data = None
        if os.path.exists(NSE_LIB_CACHE_PATH):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(NSE_LIB_CACHE_PATH))
                if datetime.now() - mtime < timedelta(hours=NSE_LIB_CACHE_TTL_HOURS):
                    with open(NSE_LIB_CACHE_PATH, "r", encoding="utf-8") as f:
                        actions_data = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to read NSE lib cache: {e}")
        
        if actions_data is None:
            # get_actions() returns List[CorporateAction] for all symbols at once
            actions = client.get_actions(
                from_date=None,   # defaults to today
                to_date=None,     # defaults to 3 months ahead
                source="NSE",
            )
            actions_data = []
            for a in actions:
                actions_data.append({
                    "symbol": a.symbol,
                    "ex_date": a.ex_date,
                    "action_type": a.action_type,
                    "details": a.details
                })
            try:
                os.makedirs(os.path.dirname(NSE_LIB_CACHE_PATH), exist_ok=True)
                with open(NSE_LIB_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(actions_data, f)
            except Exception as e:
                logger.debug(f"Failed to write NSE lib cache: {e}")

        # Group by symbol and convert to our internal schema
        cutoff = datetime.now().date() - timedelta(days=7)
        for action_dict in actions_data:
            sym = (action_dict.get("symbol") or "").upper()
            if sym not in ticker_map:
                continue
            original_ticker = ticker_map[sym]

            # Parse ex_date
            ex_obj = None
            raw_ex = action_dict.get("ex_date")
            if raw_ex:
                for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
                    try:
                        ex_obj = datetime.strptime(raw_ex.strip(), fmt).date()
                        break
                    except Exception:
                        continue
            if ex_obj is None or ex_obj < cutoff:
                continue

            action_type_str = action_dict.get("action_type") or ""
            details_str = action_dict.get("details") or ""
            
            entry = {
                "type": f"Upcoming {action_type_str.title()}",
                "value": None,
                "ex_date": ex_obj.strftime("%Y-%m-%d"),
                "details": details_str,
                "source": "NSE",
            }

            # Extract numeric dividend value from details
            if action_type_str.lower() == "dividend" and details_str:
                m = re.search(r"rs\.?\s*([\d.]+)", details_str, re.I)
                if m:
                    entry["value"] = safe_float(m.group(1))

            result_by_ticker.setdefault(original_ticker, []).append(entry)

        logger.debug("NSE lib returned actions for %d/%d tickers", len(result_by_ticker), len(tickers))
    except Exception as e:
        logger.warning("india-corp-actions library fetch failed: %s", e)

    return result_by_ticker


# -------------------------------
# Bulk upcoming actions (YF-FREE)
# -------------------------------
def get_bulk_upcoming_actions(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Return upcoming corporate actions for a list of tickers.
    PRIMARY:  india-corp-actions library → NSE API (symbol-exact, authoritative)
    FALLBACK: Equitymaster (name-based match, used only when NSE has no data)
    NO yfinance calls. Safe for bulk/index loads.
    """
    # ── PRIMARY: fetch all via library in one call ────────────────────────────
    nse_by_ticker: Dict[str, List[Dict]] = {}
    if _LIB_AVAILABLE:
        nse_by_ticker = _fetch_nse_via_lib(tickers)
    else:
        logger.warning("Library unavailable — using Equitymaster only for all tickers.")

    # ── FALLBACK pool: Equitymaster (fetched once, cached 24h) ───────────────
    em_data: List[Dict] = []
    try:
        em_data = _fetch_equitymaster_data()
    except Exception as e:
        logger.warning("Equitymaster fetch error: %s", e)

    cutoff_date = datetime.now().date() - timedelta(days=7)
    results: List[Dict[str, Any]] = []
    nse_hits, em_hits = 0, 0

    for ticker in tickers:
        try:
            # ── Use NSE result if available ───────────────────────────────────
            if ticker in nse_by_ticker and nse_by_ticker[ticker]:
                results.append({"ticker": ticker, "actions": nse_by_ticker[ticker], "source": "NSE"})
                nse_hits += 1
                continue

            # ── Equitymaster fallback (original name-match logic, preserved) ──
            key = ticker.replace(".NS", "").replace(".BSE", "").lower()
            matches = []
            for item in em_data:
                try:
                    norm_name = normalize_company_name(item["name"])
                    if norm_name.startswith(key):
                        matches.append(item)
                    elif len(key) > 4 and key in norm_name:
                        matches.append(item)
                except Exception:
                    continue

            upcoming = []
            for m in matches:
                raw_date = m.get("ex_date")
                ex_obj = None
                if isinstance(raw_date, datetime):
                    ex_obj = raw_date.date()
                elif hasattr(raw_date, "year"):
                    ex_obj = raw_date
                elif isinstance(raw_date, str):
                    try:
                        ex_obj = datetime.strptime(raw_date, "%Y-%m-%d").date()
                    except Exception:
                        pass
                if not ex_obj or ex_obj < cutoff_date:
                    continue
                upcoming.append({
                    "type": f"Upcoming {m['type']}",
                    "value": m.get("value"),
                    "ex_date": _fmt_date(ex_obj),
                    "source": "Equitymaster",
                })

            if upcoming:
                em_hits += 1
            results.append({
                "ticker": ticker,
                "actions": upcoming,
                "source": "equitymaster_fallback" if upcoming else "none",
            })

        except Exception as e:
            logger.debug("Error in bulk upcoming for %s: %s", ticker, e)
            results.append({"ticker": ticker, "actions": [], "source": "error"})

    logger.info(
        "Bulk upcoming: %d tickers | NSE lib: %d | EM fallback: %d | no data: %d",
        len(tickers), nse_hits, em_hits, len(tickers) - nse_hits - em_hits,
    )
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
            # --- Apply Lookback Filter ---
            if lookback_days > 0:
                # Calculate cutoff string "YYYY-MM-DD"
                cutoff_dt = datetime.now().date() - timedelta(days=lookback_days)
                cutoff_str = _fmt_date(cutoff_dt)
                
                # Filter events newer than cutoff
                # String comparison works for ISO dates: "2025-01-01" > "2024-01-01"
                events = [e for e in events if e.get("ex_date", "") >= cutoff_str]
            # ----------------------------------
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
# Summary cache — pre-built flat map for fast index page loads
# -------------------------------
def _action_to_display(action: Dict) -> str:
    """Convert a single action dict to a short grid cell string."""
    today = datetime.now().date()
    ex_str = action.get("ex_date", "")
    try:
        if ex_str and datetime.strptime(ex_str, "%Y-%m-%d").date() >= today:
            typ = action.get("type", "").replace("Upcoming ", "")
            val = action.get("value")
            src = action.get("source", "")
            parts = []
            if "Dividend" in typ:
                parts.append("Div")
                if val:
                    parts.append(f"Rs{val}")
            elif "Bonus" in typ:
                parts.append(f"Bonus {val}" if val else "Bonus")
            elif "Split" in typ:
                parts.append(f"Split {val}" if val else "Split")
            else:
                parts.append(typ)
            parts.append(f"(Ex:{ex_str})")
            if src:
                parts.append(f"[{src}]")
            return " ".join(parts)
    except Exception:
        pass
    return ""


def build_corp_actions_summary_cache(tickers: List[str], force: bool = False) -> Dict[str, str]:
    """
    Pre-build and persist a flat {ticker: display_string} map for all tickers.
    Called once at app startup in a background thread. TTL = SUMMARY_CACHE_TTL_HOURS.

    Args:
        tickers: All tickers across all indices you want to pre-warm.
        force:   Bypass TTL and rebuild even if cache is fresh.

    Returns:
        Dict[ticker, display_string] — only tickers with upcoming actions are included.
    """
    if not force and os.path.exists(SUMMARY_CACHE_PATH):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(SUMMARY_CACHE_PATH))
            if datetime.now() - mtime < timedelta(hours=SUMMARY_CACHE_TTL_HOURS):
                with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                logger.info("Summary cache loaded (%d tickers with actions)", len(cached))
                return cached
        except Exception as e:
            logger.warning("Summary cache read failed: %s", e)

    logger.info("Building corp actions summary for %d tickers...", len(tickers))
    bulk = get_bulk_upcoming_actions(tickers)

    summary: Dict[str, str] = {}
    for item in bulk:
        ticker = item.get("ticker", "")
        actions = item.get("actions", [])
        for action in actions:
            label = _action_to_display(action)
            if label:
                summary[ticker] = label
                break  # first valid upcoming action is enough for the grid cell

    try:
        os.makedirs(os.path.dirname(SUMMARY_CACHE_PATH), exist_ok=True)
        with open(SUMMARY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary cache saved: %d/%d tickers have upcoming actions", len(summary), len(tickers))
    except Exception as e:
        logger.warning("Failed to write summary cache: %s", e)

    return summary


def get_corp_actions_summary(tickers: List[str]) -> Dict[str, str]:
    """
    Fast read path for /corporate_action_summary endpoint.
    Reads the pre-built summary cache. Rebuilds if stale or missing.
    Returns {ticker: display_string} — only tickers with upcoming actions present.
    """
    if os.path.exists(SUMMARY_CACHE_PATH):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(SUMMARY_CACHE_PATH))
            if datetime.now() - mtime < timedelta(hours=SUMMARY_CACHE_TTL_HOURS):
                with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
                    full = json.load(f)
                return {t: full[t] for t in tickers if t in full}
        except Exception as e:
            logger.warning("Summary cache read failed, rebuilding: %s", e)

    # Cache missing or stale — build it now for these tickers
    return build_corp_actions_summary_cache(tickers)


# -------------------------------
# CLI / debug
# -------------------------------
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "INFY.NS"
    print(f"=== Upcoming actions for {ticker} ===")
    print(json.dumps(get_corporate_actions([ticker], mode="upcoming"), indent=2, default=str))
    print(f"\n=== Past actions (YF) for {ticker} ===")
    print(json.dumps(get_corporate_actions([ticker], mode="single"), indent=2, default=str))
    print(f"\n=== Library available: {_LIB_AVAILABLE} ===")