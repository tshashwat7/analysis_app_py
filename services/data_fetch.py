"""
Consolidated production-ready services/data_fetch.py
- Hybrid Architecture: Parquet (L1 Cache) + Yahoo Finance (Source)
- Smart freshness logic (Timezone-Aware Fix)
- Full helper suite (Kept from Version #1)
- Auto-saving of OHLCV data to Disk
- Memory Safety: Enforces row limits on RAM cache
"""

import asyncio
import datetime
import functools
import random
import threading
import time
import math
import os
import json
import logging
from cachetools import LRUCache
from typing import Dict, Optional, Tuple, List, Any
from functools import lru_cache, partial

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# --- NEW: Import the Parquet Engine ---
from services.data_layer import ParquetStore

# Config imports with safe fallbacks
try:
    from config.constants import TECHNICAL_METRIC_MAP, FUNDAMENTAL_ALIAS_MAP, HORIZON_FETCH_CONFIG, ENABLE_CACHE
except Exception:
    ENABLE_CACHE = True
    TECHNICAL_METRIC_MAP = {}
    FUNDAMENTAL_ALIAS_MAP = {}
    HORIZON_FETCH_CONFIG = {}

logger = logging.getLogger(__name__)

# -----------------------------
# Global cache
# -----------------------------
GLOBAL_OHLC_CACHE = LRUCache(maxsize=50)
CACHE_LOCK = threading.Lock()
SHORT_TTL = 15 * 60    
LONG_TTL = 6 * 60 * 60 

# -----------------------------
# Utility helpers
# -----------------------------

def _enforce_cache_limits(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Prevent RAM Bloat: Trims DataFrame to a maximum safe size before caching.
    """
    if df is None or df.empty:
        return df
        
    # Safe limits:
    # Intraday: ~7 days (approx 2500 rows)
    # Daily: ~10 years (approx 2500 rows)
    MAX_ROWS = 2500 
    
    if len(df) > MAX_ROWS:
        return df.tail(MAX_ROWS).copy()
    return df

def _get_cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}_{period}_{interval}".upper()

def _is_market_open_now():
    now = datetime.datetime.now()
    if now.weekday() >= 5: return False
    if 9 <= now.hour < 16: return True
    return False

def _check_freshness(df: pd.DataFrame, interval: str) -> bool:
    if not ENABLE_CACHE: return False
    if df is None or getattr(df, "empty", True): return False

    try:
        last_ts = pd.to_datetime(df.index[-1]).tz_localize(None)
        now = datetime.datetime.now()
        
        # Total minutes since the last candle
        age_minutes = (now - last_ts).total_seconds() / 60
        
        # Future protection
        if age_minutes < -10: return False 

        is_intraday = "m" in interval
        
        if is_intraday:
            # If market is OPEN (Mon-Fri, 9:15-15:30), strict check
            if _is_market_open_now():
                return age_minutes < 30 # Must be recent
            
            # If market is CLOSED, any data from the last close is fine
            # (Logic: If it's 8 PM, data from 3:30 PM is still "fresh")
            if age_minutes < (18 * 60): # Within last 18 hours
                return True
            return False
            
        else: # Daily
            # If today is Sat/Sun, anything from Friday is fine
            if now.weekday() >= 5: 
                return age_minutes < (72 * 60) # 3 days buffer
            
            # Weekday: Data must be from today or yesterday
            return age_minutes < (24 * 60)

    except Exception as e:
        return False
# -----------------------------
# Retry wrapper
# -----------------------------
def _retry(fn, retries=3, backoff=1.5, name: Optional[str] = None):
    last_exc = None
    for attempt in range(retries):
        try: return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                logger.debug(f"[RETRY] {name or fn.__name__} attempt {attempt+1}/{retries} failed: {e}")
                time.sleep(backoff * (2 ** attempt))
            else:
                logger.warning(f"[FAIL] {name or fn.__name__}: {e}")
                raise
    raise last_exc

# -----------------------------
# Safe conversions & coercions
# -----------------------------
def _num(v) -> Optional[float]:
    if v is None: return None
    try:
        if isinstance(v, (int, float)): return float(v)
        s = str(v).replace("₹", "").replace(",", "").strip()
        if s in ("", "N/A", "None", "nan"): return None
        return float(s)
    except Exception:
        logger.debug("Failed to convert to float: %r", v, exc_info=True)
        return None


def _to_float(v: Any, default=None) -> Optional[float]:
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return float(v)
        return float(str(v).replace("%", "").replace("₹", "").strip())
    except Exception: return default

def safe_float(v, default=None):
    if v is None: return default
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return default
        if isinstance(v, (int, float)): return float(v)
        s = str(v).replace("₹", "").replace(",", "").strip()
        if s.lower() in ("", "n/a", "na", "none", "nan"): return default
        if "%" in s: s = s.replace("%", "").strip()
        if s.endswith("x"): return float(s[:-1])
        return float(s)
    except Exception: return default

def _safe_float(v): return safe_float(v, default=None)

def _safe_get_raw_float(metric_entry: Any) -> Optional[float]:
    if metric_entry is None: return None
    if isinstance(metric_entry, dict):
        for key in ["value", "raw", "score"]:
            v = metric_entry.get(key)
            if v is not None:
                f = _safe_float(v)
                if f is not None: return f
    return _safe_float(metric_entry)

def normalize_ratio(v):
    v = safe_float(v)
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    if abs(v) < 2: v *= 100
    return v

def safe_json(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)): return obj.to_dict()
    elif isinstance(obj, (np.integer, np.floating)): return obj.item()
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [safe_json(v) for v in obj]
    return obj

def _coerce_value(v: Any) -> Any:
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None
    if v is None: return None
    try:
        if hasattr(v, "item") and callable(v.item): return v.item()
    except Exception: pass
    if _np is not None and isinstance(v, _np.ndarray):
        if v.ndim == 0: return _coerce_value(v.item())
        if v.size <= 20: return [_coerce_value(x) for x in v.tolist()]
        return [_coerce_value(x) for x in v.flatten()[:20]]
    if _pd is not None:
        if isinstance(v, _pd.Series) or isinstance(v, _pd.Index):
            lst = v.dropna().tolist() if hasattr(v, "dropna") else v.tolist()
            return [_coerce_value(x) for x in lst[:20]]
        if isinstance(v, _pd.DataFrame):
            if v.shape[0] <= 10 and v.shape[1] <= 6:
                return {c: [_coerce_value(x) for x in v[c].dropna().tolist()] for c in v.columns}
            return v
    if isinstance(v, (int, float, str, bool)): return v
    return v

def safe_div(a, b, default=None):
    try:
        if b == 0 or b is None or pd.isna(b): return default
        return a / b
    except Exception: return default

def safe_get(d: dict, key: str, default=None):
    try:
        v = d.get(key, default)
        if v in [None, "None", "NaN", "nan", ""]: return default
        return v
    except Exception: return default

def _fmt_pct(v: Optional[float]) -> str:
    if v is None: return "N/A"
    try: return f"{round(v, 2)}%" if abs(v) < 100 else f"{round(v, 2)}"
    except Exception: return str(v)

def _fmt_num(v: Optional[float]) -> str:
    if v is None: return "N/A"
    try:
        if abs(v) >= 1e9: return f"{v/1e9:.1f}B"
        if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
        return f"{round(float(v), 2)}"
    except Exception: return str(v)

def _fmt_date(d): return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

def _is_missing_entry(e):
    if e is None: return True
    if isinstance(e, dict): return e.get("value") in (None, "", "N/A", "NaN", "nan")
    return e in (None, "", "N/A", "NaN", "nan")

def _val(indicators: Dict[str, Dict[str, Any]], key: str):
    item = indicators.get(key)
    if not isinstance(item, dict): return None
    return item.get("value")

def _clamp(v, a, b):
    try: return max(a, min(b, v))
    except Exception: return a

def normalize_shares(shares_value):
    try:
        s = float(shares_value)
        if 0 < s < 1e7: return s * 1e7
        return s
    except: return None

def normalize_currency_value(val, unit_hint=None):
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).replace(",", "").strip().lower()
    try:
        if s.endswith("cr") or "crore" in s: return float(s.replace("cr", "").replace("crore", "").strip()) * 1e7
        if s.endswith("mn") or "m" in s: return float(s.replace("mn", "").replace("m", "").strip()) * 1e6
        if s.endswith("b") or "bn" in s: return float(s.replace("bn", "").replace("b", "").strip()) * 1e9
        return float(s)
    except: return None

def safe_info_normalized(key: str, info: dict):
    val = info.get(key)
    if key in ("sharesOutstanding", "shares_outstanding"): return normalize_shares(val)
    if key in ("marketCap", "market_cap", "enterpriseValue"): return normalize_currency_value(val)
    if key in ("bookValue", "book_value"): return safe_float(val)
    return safe_float(val)

def _normalize_single_metric(d, metric_name):
    out = dict(d) if isinstance(d, dict) else {"value": d}
    if "value" in out: out["value"] = _coerce_value(out["value"])
    else: out["value"] = _coerce_value(out.get("raw"))
    try: s = int(round(float(out.get("score", 0))))
    except: s = 0
    out["score"] = max(0, min(10, s))
    if not out.get("desc"): out["desc"] = f"{metric_name} -> {out.get('value')}"
    else: out["desc"] = str(out["desc"])
    return out

def _wrap_calc(arg, name: Optional[str] = None):
    if isinstance(arg, str) and name is None:
        metric_key = arg
        alias_name = FUNDAMENTAL_ALIAS_MAP.get(metric_key, metric_key)
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    raw = _retry(lambda: fn(*args, **kwargs), name=metric_key)
                except Exception as e:
                    return {"value": None, "score": 0, "desc": f"Error: {e}", "alias": alias_name, "source": "core"}
                if raw is None:
                    return {"value": None, "score": 0, "desc": f"{metric_key} -> None", "alias": alias_name, "source": "core"}
                metric = _normalize_single_metric(raw, metric_key)
                metric["alias"] = alias_name
                metric["source"] = "core"
                return metric
            return wrapper
        return decorator

    if callable(arg) and isinstance(name, str):
        fn = arg
        metric_group_alias = TECHNICAL_METRIC_MAP.get(name, name)
        try:
            raw = _retry(lambda: fn(), name=name)
        except Exception as e:
            return {name: {"value": None, "score": 0, "desc": f"Error: {e}", "alias": metric_group_alias, "source": "technical"}}
        out = {}
        if raw is None: return out
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
        time.sleep(random.uniform(0.5, 1.5)) 
        t = yf.Ticker(sym)
        return t.history(period=period, auto_adjust=auto_adjust, **kwargs)
    try:
        df = _retry(_fetch, retries=3, backoff=2.0)
        if df is None or getattr(df, "empty", True): return pd.DataFrame()
        return df
    except: return pd.DataFrame()

@lru_cache(maxsize=1024)
def safe_info(sym: str):
    def _fetch(): return yf.Ticker(sym).info or {}
    return _retry(_fetch, retries=2, backoff=0.6)

def get_history_for_horizon(symbol: str, horizon: str = "short_term", auto_adjust: bool = True) -> pd.DataFrame:
    cfg = HORIZON_FETCH_CONFIG.get(horizon, HORIZON_FETCH_CONFIG.get("short_term", {}))
    period, interval = cfg.get("period", "3mo"), cfg.get("interval", "1d")
    key = _get_cache_key(symbol, period, interval)
    
    ttl = SHORT_TTL if interval in ["1m", "5m", "15m"] else LONG_TTL

    # --- TIER 1: L1 RAM CACHE (Bounded LRU) ---
    if ENABLE_CACHE:
        with CACHE_LOCK:
            entry = GLOBAL_OHLC_CACHE.get(key)
            if entry:
                if (time.time() - entry["ts"]) < ttl:
                    return entry["df"].copy()
                else:
                    try: del GLOBAL_OHLC_CACHE[key]
                    except KeyError: pass

    # --- TIER 2: L2 PARQUET CACHE ---
    if ENABLE_CACHE:
        max_age_mins = ttl / 60
        lookback_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        target_days = lookback_map.get(period, 730)
        
        df_parquet = ParquetStore.load_ohlcv(symbol, interval, max_age_minutes=max_age_mins, lookback_days=target_days)
        
        # VALIDATION CHECK: Needs 'Close' column to be useful
        if df_parquet is not None and not df_parquet.empty and "Close" in df_parquet.columns:
            if _check_freshness(df_parquet, interval):
                with CACHE_LOCK:
                    # 🔥 FIX: Trim before caching
                    cached_df = _enforce_cache_limits(df_parquet, interval)
                    GLOBAL_OHLC_CACHE[key] = {"df": cached_df, "ts": time.time(), "interval": interval}
                return df_parquet

    # --- TIER 3: L3 YAHOO API ---
    try:
        df = safe_history(symbol, period=period, interval=interval, auto_adjust=auto_adjust)
        
        if df is not None and not df.empty:
            ParquetStore.save_ohlcv(symbol, df, interval)
            if ENABLE_CACHE:
                with CACHE_LOCK:
                    # 🔥 FIX: Trim before caching
                    cached_df = _enforce_cache_limits(df, interval)
                    GLOBAL_OHLC_CACHE[key] = {"df": cached_df, "ts": time.time(), "interval": interval}
        return df
    except Exception as e:
        logger.error(f"[{symbol}] Fetch error for {horizon}: {e}")
        return pd.DataFrame()


def get_benchmark_data(horizon: str = "short_term", benchmark_symbol: str = "^NSEI", auto_adjust: bool = True) -> pd.DataFrame:
    """
    Fetches Benchmark data with 3-Tier Caching (LRU RAM -> Partial Parquet -> Yahoo).
    """
    cfg = HORIZON_FETCH_CONFIG.get(horizon, HORIZON_FETCH_CONFIG.get("short_term", {}))
    period, interval = cfg.get("period", "3mo"), cfg.get("interval", "1d")
    key = _get_cache_key(benchmark_symbol, period, interval)
    
    # Benchmarks are less volatile, can use longer TTLs
    ttl = LONG_TTL 

    # --- TIER 1: L1 RAM CACHE (Bounded LRU) ---
    if ENABLE_CACHE:
        with CACHE_LOCK:
            entry = GLOBAL_OHLC_CACHE.get(key)
            if entry:
                if (time.time() - entry["ts"]) < ttl:
                    return entry["df"].copy()
                else:
                    try: del GLOBAL_OHLC_CACHE[key]
                    except KeyError: pass

    # --- TIER 2: L2 PARQUET (Disk) ---
    if ENABLE_CACHE:
        max_age_mins = 720 
        lookback_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        target_days = lookback_map.get(period, 730)
        df_parquet = ParquetStore.load_ohlcv(benchmark_symbol, interval, max_age_minutes=max_age_mins, lookback_days=target_days)
        
        # VALIDATION CHECK: Needs 'Close' column
        if df_parquet is not None and not df_parquet.empty and "Close" in df_parquet.columns:
            if _check_freshness(df_parquet, interval):
                with CACHE_LOCK:
                    cached_df = _enforce_cache_limits(df_parquet, interval)
                    GLOBAL_OHLC_CACHE[key] = {"df": cached_df, "ts": time.time(), "interval": interval}
                return df_parquet

    # --- TIER 3: L3 YAHOO API (Source) ---
    try:
        df = safe_history(benchmark_symbol, period=period, interval=interval, auto_adjust=auto_adjust)
        if df is not None and not df.empty:
            ParquetStore.save_ohlcv(benchmark_symbol, df, interval)
            if ENABLE_CACHE:
                with CACHE_LOCK:
                    # 🔥 FIX: Trim benchmark too
                    cached_df = _enforce_cache_limits(df, interval)
                    GLOBAL_OHLC_CACHE[key] = {"df": cached_df, "ts": time.time(), "interval": interval}
        return df
    except Exception as e:
        logger.error(f"[{benchmark_symbol}] Benchmark error: {e}")
        return pd.DataFrame()

def fetch_data(period: str = None, interval: str = None, auto_adjust: bool = True, horizon: str = None) -> pd.DataFrame:
    if period is None or interval is None:
        if horizon:
            cfg = HORIZON_FETCH_CONFIG.get(horizon, None)
            if cfg: period, interval = cfg.get("period"), cfg.get("interval")
        period, interval = period or "1y", interval or "1d"
    try:
        df = yf.download("^NSEI", period=period, interval=interval, progress=False, auto_adjust=auto_adjust)
        return df.sort_index() if not getattr(df, "empty", True) else pd.DataFrame()
    except: return pd.DataFrame()

def get_price_history(symbol: str, period: str = "2y", auto_adjust: bool = True) -> pd.DataFrame:
    return safe_history(symbol, period, auto_adjust)

def parse_index_csv(csv_path: str) -> List[Tuple[str, str]]:
    pairs = []
    try:
        if not os.path.exists(csv_path): return pairs
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        cols = [c.upper() for c in df.columns]
        s_idx = next((i for i,c in enumerate(cols) if c in ("SYMBOL","TICKER","CODE")), 0)
        n_idx = next((i for i,c in enumerate(cols) if c in ("NAME","COMPANY")), 1 if len(cols)>1 else None)
        for i in range(len(df)):
            row = df.iloc[i]
            sym = str(row.iloc[s_idx]).strip()
            if not sym: continue
            if not sym.upper().endswith(".NS"): sym = f"{sym}.NS"
            name = str(row.iloc[n_idx]).strip() if n_idx is not None else sym
            pairs.append((sym, name))
    except: pass
    return pairs

def save_stocks_list(pairs, index_file):
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump([{"symbol": s, "name": n} for s, n in pairs], f, indent=2)
    except: pass

async def run_sync_or_async(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func): return await func(*args, **kwargs)
    else: return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))

__all__ = [
    "get_history_for_horizon", "get_benchmark_data", "get_price_history",
    "safe_history", "safe_info", "fetch_data", "_wrap_calc", "safe_float", "safe_json",
    "safe_info_normalized"
]