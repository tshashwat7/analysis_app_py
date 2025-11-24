"""
Consolidated production-ready services/data_fetch.py
- Smart freshness logic (Version #2)
- Robust wrap_calc implementing both decorator (fundamentals) and direct-call (indicators)
- Full helper suite (from Version #1)
- Single global cache with TTLs
- Flat indicator output with alias + desc preservation
- No duplicate or circular definitions

"""

import asyncio
import datetime
import functools
import threading
import time
import math
import os
import json
import logging
from typing import Dict, Optional, Tuple, List, Any
from functools import lru_cache, partial

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Config imports with safe fallbacks
try:
    from config.constants import TECHNICAL_METRIC_MAP, FUNDAMENTAL_ALIAS_MAP, HORIZON_FETCH_CONFIG, ENABLE_CACHE
except Exception:
    ENABLE_CACHE = True
    TECHNICAL_METRIC_MAP = {}
    FUNDAMENTAL_ALIAS_MAP = {}
    HORIZON_FETCH_CONFIG = {}

logger = logging.getLogger(__name__)
# Leave logging configuration to main application

# -----------------------------
# Global cache (single source)
# -----------------------------
GLOBAL_OHLC_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = threading.Lock()
SHORT_TTL = 15 * 60    # 15 minutes for short/intraday horizons
LONG_TTL = 6 * 60 * 60 # 6 hours for daily/long horizons

# -----------------------------
# Utility helpers
# -----------------------------

def _get_cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}_{period}_{interval}".upper()


def _is_market_open_now():
    now = datetime.datetime.now()
    # Weekends (5=Sat, 6=Sun)
    if now.weekday() >= 5:
        return False
    # Simple hour check (9 AM to 4 PM) - adjust to your timezone if necessary
    if 9 <= now.hour < 16:
        return True
    return False


def _check_freshness(df: pd.DataFrame, interval: str) -> bool:
    """
    Smart freshness check (Version #2 behaviour):
    - If cache disabled -> False (force refetch)
    - If intraday & market open -> require recent (< 30 minutes)
    - If intraday & market closed -> accept end-of-day
    - If daily/weekly & weekend -> accept (Friday EOD)
    - If daily/weekly & weekday -> allow up to 30 hours
    """
    if not ENABLE_CACHE:
        return False

    if df is None or getattr(df, "empty", True):
        return False

    last_ts = df.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()

    if getattr(last_ts, "tzinfo", None):
        last_ts = last_ts.replace(tzinfo=None)

    now = datetime.datetime.now()
    age_minutes = (now - last_ts).total_seconds() / 60

    is_intraday = interval in ["1m", "5m", "15m", "30m", "60m", "90m"]

    if is_intraday:
        if _is_market_open_now():
            if age_minutes > 30:
                logger.debug(f"[STALE] Intraday data is {age_minutes:.0f} min old.")
                return False
        # Market closed -> accept EOD intraday
        return True

    else:
        # Daily/weekly/monthly
        if now.weekday() >= 5:
            # Weekend: use last available session (Friday)
            return True
        # Weekdays: allow ~30 hours to accommodate EOD overlaps
        if age_minutes > 30 * 60:
            return False

    return True


# -----------------------------
# Retry wrapper
# -----------------------------
def _retry(fn, retries=3, backoff=1.5, name: Optional[str] = None):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                logger.debug(f"[RETRY] {name or fn.__name__} attempt {attempt+1}/{retries} failed: {e}")
                time.sleep(backoff * (2 ** attempt))
            else:
                logger.warning(f"[FAIL] {name or fn.__name__} after {retries} retries: {e}")
                raise
    raise last_exc


# -----------------------------
# Safe conversions & coercions (from Version #1)
# -----------------------------

def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace("₹", "").replace(",", "").strip()
        if s in ("", "N/A", "None", "nan"):
            return None
        return float(s)
    except Exception:
        logger.debug("Failed to convert to float: %r", v, exc_info=True)
        return None


def _to_float(v: Any, default=None) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v).replace("%", "").replace("₹", "").strip())
    except Exception:
        return default


def safe_float(v, default=None):
    if v is None:
        return default
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace("₹", "").replace("%", "").replace(",", "").strip()
        if s == "" or s.lower() in ("n/a", "nan", "none"):
            return default
        return float(s)
    except Exception:
        return default


def _safe_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return float(v)
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s == "":
                return None
            if s.lower() in ("n/a", "na", "nan", "none"):
                return None
            if s.endswith("%"):
                return float(s[:-1])
            if s.endswith("x"):
                return float(s[:-1])
            return float(s)
        return float(v)
    except Exception:
        return None


def _safe_get_raw_float(metric_entry: Any) -> Optional[float]:
    if metric_entry is None:
        return None
    if isinstance(metric_entry, dict):
        for key in ["value", "raw", "score"]:
            v = metric_entry.get(key)
            if v is not None:
                f = _safe_float(v)
                if f is not None:
                    return f
        return None
    return _safe_float(metric_entry)


def normalize_ratio(v):
    v = safe_float(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if abs(v) < 2:
        v *= 100
    return v


def safe_json(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    else:
        return obj


def _coerce_value(v: Any) -> Any:
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None

    if v is None:
        return None
    try:
        if hasattr(v, "item") and callable(v.item):
            return v.item()
    except Exception:
        pass

    if _np is not None and isinstance(v, _np.ndarray):
        if v.ndim == 0:
            return _coerce_value(v.item())
        if v.size <= 20:
            return [_coerce_value(x) for x in v.tolist()]
        return [_coerce_value(x) for x in v.flatten()[:20]]

    if _pd is not None:
        if isinstance(v, _pd.Series) or isinstance(v, _pd.Index):
            lst = v.dropna().tolist() if hasattr(v, "dropna") else v.tolist()
            return [_coerce_value(x) for x in lst[:20]]
        if isinstance(v, _pd.DataFrame):
            if v.shape[0] <= 10 and v.shape[1] <= 6:
                return {c: [_coerce_value(x) for x in v[c].dropna().tolist()] for c in v.columns}
            return v

    if isinstance(v, (int, float, str, bool)):
        return v
    return v

def safe_div(a, b, default=None):
    try:
        if b == 0 or b is None or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default

def safe_get(d: dict, key: str, default=None):
    """Safe getter that returns default for invalid, None or nan-like values."""
    try:
        v = d.get(key, default)
        if v in [None, "None", "NaN", "nan", ""]:
            return default
        return v
    except Exception:
        return default

def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    try:
        if abs(v) < 100:
            return f"{round(v, 2)}%"
        return f"{round(v, 2)}"
    except Exception:
        return str(v)


def _fmt_num(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    try:
        if abs(v) >= 1e9:
            return f"{v/1e9:.1f}B"
        if abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        return f"{round(float(v), 2)}"
    except Exception:
        return str(v)


def _fmt_date(d):
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


def _is_missing_entry(e):
    if e is None:
        return True
    if isinstance(e, dict):
        v = e.get("value")
        return v in (None, "", "N/A", "NaN", "nan")
    return e in (None, "", "N/A", "NaN", "nan")


def _val(indicators: Dict[str, Dict[str, Any]], key: str):
    item = indicators.get(key)
    if not isinstance(item, dict):
        return None
    return item.get("value")


def _clamp(v, a, b):
    try:
        return max(a, min(b, v))
    except Exception:
        return a


# -----------------------------
# Metric normalizer & wrapper
# -----------------------------
def normalize_shares(shares_value: Any) -> Optional[float]:
    """Return shares as raw integer (not crores). If small (<1e7) assume value given in crores and convert."""
    try:
        if shares_value is None: return None
        s = float(shares_value)
    except Exception:
        return None
    # heuristics: if shares less than 10 million, it's probably given in crores (e.g., 10.5 -> 10.5 cr)
    # Typical Nifty 50 companies have shares > 1 Cr.
    if 0 < s < 1e7:
        return s * 1e7
    return s

def normalize_currency_value(val: Any, unit_hint: str = None) -> Optional[float]:
    """
    Simple normalizer: if val is string with commas or suffix like 'Cr' or ' crore', parse it.
    Return numeric rupees (raw).
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", "").strip().lower()
    try:
        if not s: return None
        # handle common suffixes
        if s.endswith("cr") or "crore" in s:
            num = float(s.replace("cr", "").replace("crore", "").strip())
            return num * 1e7
        if s.endswith("mn") or "m" in s:
            num = float(s.replace("mn", "").replace("m", "").strip())
            return num * 1e6
        if s.endswith("b") or "bn" in s:
            num = float(s.replace("bn", "").replace("b", "").strip())
            return num * 1e9
        return float(s)
    except Exception:
        return None

def safe_info_normalized(key: str, info: dict):
    """Wrapper around safe_info that normalizes typical values like sharesOutstanding, marketCap, bookValue etc."""
    # Use internal safe_get or simple dict get, assuming info is a dict
    val = info.get(key)
    
    if key in ("sharesOutstanding", "shares_outstanding"):
        return normalize_shares(val)
    if key in ("marketCap", "market_cap", "enterpriseValue"):
        return normalize_currency_value(val)
    if key in ("bookValue", "book_value"):
        try:
            return float(val) if val is not None else None
        except:
            return None
    
    # Fallback to standard safe_float for other keys
    return safe_float(val)

def _normalize_single_metric(d: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    out = dict(d) if isinstance(d, dict) else {"value": d}
    if "value" in out:
        out["value"] = _coerce_value(out["value"])
    else:
        out["value"] = _coerce_value(out.get("raw"))
    try:
        s = int(round(float(out.get("score", 0))))
    except Exception:
        s = 0
    out["score"] = max(0, min(10, s))
    if not out.get("desc"):
        out["desc"] = f"{metric_name} -> {out.get('value')}"
    else:
        out["desc"] = str(out["desc"])
    return out


def _is_multi_metric_dict(raw: Any) -> bool:
    return isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values())


def _wrap_calc(arg, name: Optional[str] = None):
    """
    Unified wrapper:
    - If used as decorator: @_wrap_calc("metric_key") -> returns single metric dict (value, score, desc, alias, source)
    - If used as direct-call: _wrap_calc(fn, "group_name") -> returns flat dict of submetric -> { value, score, desc, alias, source }
    """

    # Decorator mode (fundamentals)
    if isinstance(arg, str) and name is None:
        metric_key = arg
        alias_name = FUNDAMENTAL_ALIAS_MAP.get(metric_key, metric_key)

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # keep retry local to avoid circular import issues
                try:
                    raw = _retry(lambda: fn(*args, **kwargs), name=metric_key)
                except Exception as e:
                    return {
                        "value": None,
                        "score": 0,
                        "desc": f"Error: {e}",
                        "alias": alias_name,
                        "source": "core",
                    }

                if raw is None:
                    return {
                        "value": None,
                        "score": 0,
                        "desc": f"{metric_key} -> None",
                        "alias": alias_name,
                        "source": "core",
                    }

                metric = _normalize_single_metric(raw, metric_key)
                metric["alias"] = alias_name
                metric["source"] = "core"
                return metric
            return wrapper
        return decorator

    # Direct-call mode (indicators)
    if callable(arg) and isinstance(name, str):
        fn = arg
        metric_group_alias = TECHNICAL_METRIC_MAP.get(name, name)

        try:
            raw = _retry(lambda: fn(), name=name)
        except Exception as e:
            return {
                name: {
                    "value": None,
                    "score": 0,
                    "desc": f"Error: {e}",
                    "alias": metric_group_alias,
                    "source": "technical",
                }
            }

        # raw expected to be dict of submetrics; produce flat dict
        out = {}
        if raw is None:
            return out

        for sub_key, sub_val in raw.items():
            sub_alias = TECHNICAL_METRIC_MAP.get(sub_key, sub_key)
            metric = _normalize_single_metric(sub_val if isinstance(sub_val, dict) else {"value": sub_val}, sub_key)
            metric["alias"] = sub_alias
            metric["source"] = "technical"
            out[sub_key] = metric

        return out

    raise TypeError("Invalid usage of _wrap_calc")


# -----------------------------
# Data fetchers
# -----------------------------

def safe_history(sym: str, period: str = '2y', auto_adjust: bool = True, **kwargs):
    def _fetch():
        t = yf.Ticker(sym)
        return t.history(period=period, auto_adjust=auto_adjust, **kwargs)

    try:
        df = _retry(_fetch, retries=2, backoff=0.6)
        if df is None or isinstance(df, (list, tuple)) or getattr(df, "empty", True):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=1024)
def safe_info(sym: str):
    def _fetch():
        t = yf.Ticker(sym)
        return t.info or {}
    return _retry(_fetch, retries=2, backoff=0.6)


def get_history_for_horizon(symbol: str, horizon: str = "short_term", auto_adjust: bool = True) -> pd.DataFrame:
    cfg = HORIZON_FETCH_CONFIG.get(horizon, HORIZON_FETCH_CONFIG.get("short_term", {}))
    period, interval = cfg.get("period", "3mo"), cfg.get("interval", "1d")
    key = _get_cache_key(symbol, period, interval)

    if ENABLE_CACHE:
        with CACHE_LOCK:
            entry = GLOBAL_OHLC_CACHE.get(key)
            if entry:
                ttl = SHORT_TTL if interval in ["1m", "5m", "15m"] else LONG_TTL
                is_within_ttl = (time.time() - entry["ts"]) < ttl
                if is_within_ttl and _check_freshness(entry["df"], interval):
                    return entry["df"].copy()

    try:
        df = safe_history(symbol, period=period, interval=interval, auto_adjust=auto_adjust)
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()

        if ENABLE_CACHE:
            with CACHE_LOCK:
                GLOBAL_OHLC_CACHE[key] = {"df": df.copy(), "ts": time.time(), "interval": interval}
        return df

    except Exception as e:
        logger.error(f"[{symbol}] Fetch error for {horizon}: {e}")
        return pd.DataFrame()


def get_benchmark_data(horizon: str = "short_term", benchmark_symbol: str = "^NSEI", auto_adjust: bool = True) -> pd.DataFrame:
    cfg = HORIZON_FETCH_CONFIG.get(horizon, HORIZON_FETCH_CONFIG.get("short_term", {}))
    period, interval = cfg.get("period", "3mo"), cfg.get("interval", "1d")
    # Update Cache Key to use the SYMBOL, not just the horizon
    key = _get_cache_key(benchmark_symbol, period, interval)

    with CACHE_LOCK:
        entry = GLOBAL_OHLC_CACHE.get(key)
        if entry:
            ttl = LONG_TTL
            if (time.time() - entry["ts"]) < ttl:
                if _check_freshness(entry["df"], interval):
                    return entry["df"].copy()

    try:
        df = safe_history(benchmark_symbol, period=period, interval=interval, auto_adjust=auto_adjust)
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()

        with CACHE_LOCK:
            GLOBAL_OHLC_CACHE[key] = {"df": df.copy(), "ts": time.time(), "interval": interval}
        return df
    except Exception as e:
        logger.error(f"[{benchmark_symbol}] Final benchmark fetch error: {e}")
        return pd.DataFrame()


# Minimal legacy fetch_data for scripts that expect it (uses HORIZON_FETCH_CONFIG)

def fetch_data(period: str = None, interval: str = None, auto_adjust: bool = True, horizon: str = None) -> pd.DataFrame:
    if period is None or interval is None:
        if horizon:
            cfg = HORIZON_FETCH_CONFIG.get(horizon, None)
            if cfg:
                period = period or cfg.get("period")
                interval = interval or cfg.get("interval")
        period = period or "1y"
        interval = interval or "1d"

    try:
        df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=auto_adjust)
        if getattr(df, "empty", True):
            return pd.DataFrame()
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def get_price_history(symbol: str, period: str = "2y", auto_adjust: bool = True) -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=auto_adjust)
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def parse_index_csv(csv_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    try:
        if not os.path.exists(csv_path):
            return pairs
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        symbol_col = None
        name_col = None
        for c in df.columns:
            uc = c.strip().upper()
            if uc in ("SYMBOL", "TICKER", "CODE"):
                symbol_col = c
            if uc in ("NAME", "COMPANY", "COMPANYNAME", "SHORTNAME"):
                name_col = c
        if symbol_col is None and len(df.columns) >= 1:
            symbol_col = df.columns[0]
        if name_col is None and len(df.columns) >= 2:
            name_col = df.columns[1]

        for _, row in df.iterrows():
            sym = str(row.get(symbol_col, "")).strip()
            if not sym:
                continue
            if not sym.upper().endswith(".NS"):
                sym = f"{sym}.NS"
            name = sym.replace(".NS", "")
            if name_col:
                val = str(row.get(name_col, ""))
                if val:
                    name = val.strip()
            pairs.append((sym, name))
    except Exception as e:
        logger.exception("Error parsing index CSV: %s", e)
        return []
    return pairs


def save_stocks_list(pairs: List[Tuple[str, str]], index_file: str):
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump([{"symbol": s, "name": n} for s, n in pairs], f, indent=2)
    except Exception:
        pass


# Async helper
async def run_sync_or_async(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))


# small exports for convenience
__all__ = [
    "get_history_for_horizon",
    "get_benchmark_data",
    "get_price_history",
    "safe_history",
    "safe_info",
    "fetch_data",
    "_wrap_calc",
    "safe_float",
    "safe_json",
]
