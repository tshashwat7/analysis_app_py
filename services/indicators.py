# services/indicators.py
"""
Production-ready indicators orchestrator (multi-horizon, profile-driven, cached).
- Metric implementations preserved (only neutral-score corrections and Ichimoku purity applied).
- INDICATOR_METRIC_MAP maps metric_key -> {"func": callable, "horizon": "<horizon_key>|default|special"}.
- compute_indicators orchestrates metric selection from HORIZON_PROFILE_MAP and executes them
  with correct timeframe data (df_local) and local OHLC/price variables.
- Caching: per-call df_cache and bench_cache avoid redundant yfinance downloads.
"""

import logging
from typing import Dict, Any, Callable, Optional, Tuple
import warnings
import numpy as np
from math import atan, degrees
import pandas as pd
import pandas_ta as ta
import inspect

logger = logging.getLogger(__name__)

# Shared helpers from your project (assumed present)
from config.constants import CORE_TECHNICAL_SETUP_METRICS, GROWTH_WEIGHTS, MOMENTUM_WEIGHTS, QUALITY_WEIGHTS, TECHNICAL_WEIGHTS, HORIZON_PROFILE_MAP, TECHNICAL_METRIC_MAP, VALUE_WEIGHTS
from services.data_fetch import (
    get_benchmark_data,
    safe_float,
    _wrap_calc,
    get_history_for_horizon,
    _safe_get_raw_float
)

# ========================================================
# 🚀 PERFORMANCE OPTIMIZER: SAFE SLICING
# ========================================================
MIN_ROWS_FOR_ACCURACY = 400  # 200 DMA + 200 warm-up periods

def _slice_for_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slices the DataFrame to the most recent N rows to speed up pandas_ta.
    Calculations on 5000 rows vs 400 rows is ~10x faster.
    We keep 400 rows to ensure 200 DMA and recursive indicators (EMA/RSI) are accurate.
    """
    if df is None or df.empty:
        return df
    
    if len(df) > MIN_ROWS_FOR_ACCURACY:
        # Take the last N rows
        return df.iloc[-MIN_ROWS_FOR_ACCURACY:].copy()
    return df

# Defensive attempt to mark wrappers created by _wrap_calc
try:
    _wrap_calc._is_wrapped_calc = True
except Exception:
    # no-op if not possible
    pass


# -----------------------------
# Indicator metric functions
# (Unchanged calculation logic, except for neutral scoring fix and Ichimoku purity)
# -----------------------------

def compute_rsi(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Missing Close column for RSI")

        rsi_series = ta.rsi(df["Close"], length=length) # <-- The series is calculated here
        if rsi_series is None or rsi_series.empty:
            raise ValueError("RSI data unavailable")

        rsi_val = safe_float(rsi_series.iloc[-1])
        if rsi_val is None:
            raise ValueError("RSI value is NaN")

        if rsi_val < 30:
            score, zone = 10, "Oversold (Buy)"
        elif rsi_val > 70:
            score, zone = 0, "Overbought (Sell)"
        elif 45 <= rsi_val <= 65:
            score, zone = 8, "Healthy Momentum"
        else:
            score, zone = 5, "Neutral"

        return {
            "rsi": {
                "value": round(rsi_val, 2), 
                "score": score, 
                "desc": zone,
                "full_series": rsi_series.to_dict() 
            }
        }

    return _wrap_calc(_inner, "RSI")


def compute_mfi(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        mfi_series = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=length)
        if mfi_series is None or mfi_series.empty:
            raise ValueError("MFI empty")

        mfi_val = safe_float(mfi_series.iloc[-1])
        if mfi_val is None:
            raise ValueError("Invalid MFI value")

        if mfi_val < 20:
            score, desc = 10, "Oversold (Buy)"
        elif mfi_val > 80:
            score, desc = 0, "Overbought (Sell)"
        elif 45 <= mfi_val <= 65:
            score, desc = 8, "Healthy Momentum"
        else:
            score, desc = 5, "Neutral"

        return {"mfi": {"value": round(mfi_val, 2), "score": score, "desc": desc}}

    return _wrap_calc(_inner, "MFI")


def compute_short_ma_cross(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        ema5 = ta.ema(df["Close"], length=5)
        ema20 = ta.ema(df["Close"], length=20)
        if ema5 is None or ema20 is None or ema5.empty or ema20.empty:
            raise ValueError("EMA data unavailable")

        e5, e20 = safe_float(ema5.iloc[-1]), safe_float(ema20.iloc[-1])
        if e5 is None or e20 is None:
            raise ValueError("Invalid EMA values")

        if e5 > e20:
            signal, score = "Bullish", 10
        elif e5 < e20:
            signal, score = "Bearish", 0
        else:
            signal, score = "Neutral", 5

        return {"short_ma_cross": {"value": signal, "score": score}}

    return _wrap_calc(_inner, "Short MA Cross (5/20)")


def compute_adx(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        required_cols = {"High", "Low", "Close"}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            raise ValueError("Missing High/Low/Close columns for ADX")

        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=length)
        if adx_df is None or adx_df.empty:
            raise ValueError("ADX DataFrame empty")

        # FIX: Do this lookup ONCE
        adx_col = next((c for c in adx_df.columns if "adx" in c.lower()), None)
        di_plus_col = next((c for c in adx_df.columns if "dmp" in c.lower() or "di+" in c.lower()), None)
        di_minus_col = next((c for c in adx_df.columns if "dmn" in c.lower() or "di-" in c.lower()), None)

        if adx_col is None:
            raise ValueError(f"ADX column missing. Found: {list(adx_df.columns)}")

        adx_val = safe_float(adx_df[adx_col].iloc[-1])
        di_plus_val = safe_float(adx_df[di_plus_col].iloc[-1]) if di_plus_col else None
        di_minus_val = safe_float(adx_df[di_minus_col].iloc[-1]) if di_minus_col else None

        if adx_val is None:
            raise ValueError("ADX value invalid")

        if adx_val < 20:
            score, trend = 0, "Weak / Range-bound"
        elif adx_val <= 25:
            score, trend = 5, "Developing / Moderate Trend"
        else:
            score, trend = 10, "Strong Trend"

        return {
            "adx": {"value": round(adx_val, 2),"raw": adx_val, "score": score}, 
            "adx_signal": {"value": trend, "score": score}, 
            
            # 🆕 PATCH: Expose DI+ and DI- for the Trend Strength Composite
            "di_plus": {
                "value": round(di_plus_val, 2) if di_plus_val is not None else None, 
                "raw": di_plus_val, # Ensure raw value is available
                "score": 5,          # Neutral score for standalone use
                "desc": f"DI+ {di_plus_val:.2f}"
            }, 
            "di_minus": {
                "value": round(di_minus_val, 2) if di_minus_val is not None else None, 
                "raw": di_minus_val, # Ensure raw value is available
                "score": 5,          # Neutral score for standalone use
                "desc": f"DI- {di_minus_val:.2f}"
            },
            }

    return _wrap_calc(_inner, "ADX")


def compute_stochastic(df: pd.DataFrame, k_length: int = 14, d_length: int = 3, smooth_k: int = 3) -> Dict[str, Dict[str, Any]]:
    def _inner():
        required_cols = {"High", "Low", "Close"}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            raise ValueError("Missing High/Low/Close columns for Stochastic")

        stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=k_length, d=d_length, smooth_k=smooth_k)
        if stoch_df is None or stoch_df.empty or len(stoch_df) < 1:
            raise ValueError("Stochastic DataFrame empty")

        stoch_df.columns = [c.lower() for c in stoch_df.columns]
        k_col = next((c for c in stoch_df.columns if "k" in c.lower()), None)
        d_col = next((c for c in stoch_df.columns if "d" in c.lower()), None)

        if not all([k_col, d_col]):
            raise ValueError(f"Missing %K or %D columns: {stoch_df.columns}")

        stoch_k = safe_float(stoch_df[k_col].iloc[-1])
        stoch_d = safe_float(stoch_df[d_col].iloc[-1])
        if stoch_k is None or stoch_d is None:
            raise ValueError("Stochastic values invalid or NaN")

        if stoch_k < 20 and stoch_d < 20:
            zone, score = "Oversold (Buy)", 10
        elif stoch_k > 80 and stoch_d > 80:
            zone, score = "Overbought (Sell)", 0
        else:
            zone, score = "Neutral", 5

        crossover_status = "Neutral"
        crossover_score = 5

        if len(stoch_df) >= 2:
            stoch_k_prev = safe_float(stoch_df[k_col].iloc[-2])
            stoch_d_prev = safe_float(stoch_df[d_col].iloc[-2])

            if stoch_k_prev is not None and stoch_d_prev is not None:
                if stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d:
                    crossover_status = "Bullish"
                    crossover_score = 10
                elif stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d:
                    crossover_status = "Bearish"
                    crossover_score = 0

        return {
            "stoch_k": {"value": round(stoch_k, 2), "score": score},
            "stoch_d": {"value": round(stoch_d, 2), "score": score},
            "stoch_cross": {"value": crossover_status, "score": crossover_score},
        }

    return _wrap_calc(_inner, "Stochastic")


def compute_ema_crossover(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Missing Close column for EMA crossover")

        ema20 = ta.ema(df["Close"], length=20)
        ema50 = ta.ema(df["Close"], length=50)
        ema200 = ta.ema(df["Close"], length=200)

        if ema20 is None or ema50 is None or ema200 is None:
            raise ValueError("EMA calculation failed")

        e20 = safe_float(ema20.iloc[-1])
        e50 = safe_float(ema50.iloc[-1])
        e200 = safe_float(ema200.iloc[-1])

        if any(v is None for v in [e20, e50, e200]):
            raise ValueError("Invalid EMA values")

        if e20 > e50 > e200:
            trend, score = "Bullish", 10
        elif e20 < e50 < e200:
            trend, score = "Bearish", 0
        else:
            trend, score = "Neutral", 5

        return {
            "ema_20": {"value": round(e20, 2), "score": score},
            "ema_50": {"value": round(e50, 2), "score": score},
            "ema_200": {"value": round(e200, 2), "score": score},
            "ema_cross_trend": {"value": trend, "score": score},
        }

    return _wrap_calc(_inner, "EMA Crossover")

#
# NEW: compute_macd (Refactored from inline logic)
#
def compute_macd(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        close_ser = df["Close"].dropna()
        if len(close_ser) < 35: # Standard minimum for 12/26/9 MACD
            raise ValueError(f"Not enough data points ({len(close_ser)}) for MACD")

        macd_df = ta.macd(close_ser, fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            raise ValueError("Empty MACD DataFrame")

        # Standardize column names
        macd_df.columns = [c.lower() for c in macd_df.columns]
        macd_col = next((c for c in macd_df.columns if "macd_" in c), None)
        signal_col = next((c for c in macd_df.columns if "macds_" in c or "signal" in c), None)
        hist_col = next((c for c in macd_df.columns if "macdh_" in c or "hist" in c), None)

        if not all([macd_col, signal_col, hist_col]):
            raise ValueError(f"Missing expected MACD columns: {macd_df.columns}")

        macd_val = safe_float(macd_df[macd_col].iloc[-1])
        macd_signal = safe_float(macd_df[signal_col].iloc[-1])
        hist_series = macd_df[hist_col].dropna()
        macd_hist = safe_float(hist_series.iloc[-1])

        if macd_val is None or macd_signal is None or macd_hist is None:
            raise ValueError("Invalid MACD values (None/NaN)")

        # 1. Crossover signal + scoring
        if macd_val > macd_signal:
            cross, score = "Bullish", 10
        elif macd_val < macd_signal:
            cross, score = "Bearish", 0
        else:
            cross, score = "Neutral", 5

        # 2. Histogram z-score normalization
        hist_score = 5
        window_size = min(100, len(hist_series))
        
        # Calculate z-score based on all data *except* the last bar
        if window_size > 1:
            hist_window = hist_series.tail(window_size)
            hist_mean = hist_window.iloc[:-1].mean()
            hist_std = hist_window.iloc[:-1].std()
            hist_z = 0.0

            if hist_std and hist_std > 1e-9:
                hist_z = (macd_hist - hist_mean) / hist_std
                if hist_z > 1.0:
                    hist_score = 10
                elif hist_z < -1.0:
                    hist_score = 0
        else:
            hist_z = 0.0
            hist_desc = "Z-Score requires more data"

        # 3. Histogram strength (simple)
        hist_desc = f"MACD Histogram {macd_hist:.3f}"
        hist_strength = 10 if macd_hist > 0.5 else 7 if macd_hist > 0 else 3 if macd_hist > -0.5 else 0

        # Return all 4 metrics
        return {
            "macd": {"value": round(macd_val, 2), "score": score},
            "macd_cross": {"value": cross, "score": score},
            "macd_hist_z": {"value": round(hist_z, 4), "score": hist_score, "desc": "MACD Hist momentum normalized (Z-Score)"},
            "macd_histogram": {"value": round(macd_hist, 3), "score": hist_strength, "desc": hist_desc},
        }

    return _wrap_calc(_inner, "MACD")

def compute_vwap(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        required_cols = {"High", "Low", "Close", "Volume"}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            raise ValueError("Missing required columns for VWAP")

        vwap_series = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        if vwap_series is None or vwap_series.empty:
            raise ValueError("VWAP DataFrame empty")

        vwap_val = safe_float(vwap_series.iloc[-1])
        price_val = safe_float(df["Close"].iloc[-1])
        if vwap_val is None or price_val is None:
            raise ValueError("Invalid VWAP or Close values")

        if price_val > vwap_val:
            bias, score = "Bullish", 10
        elif price_val < vwap_val:
            bias, score = "Bearish", 0
        else:
            bias, score = "Neutral", 5

        return {"vwap": {"value": round(vwap_val, 2), "score": score}, "vwap_bias": {"value": bias, "score": score}}

    return _wrap_calc(_inner, "VWAP")


def compute_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Missing Close column for Bollinger Bands")

        bb_df = ta.bbands(df["Close"], length=length, std=std_dev)
        if bb_df is None or bb_df.empty:
            raise ValueError("Bollinger Bands empty")

        bb_df.columns = [c.lower() for c in bb_df.columns]
        upper_col = next((c for c in bb_df.columns if "u" in c), None)
        lower_col = next((c for c in bb_df.columns if "l" in c), None)
        mid_col = next((c for c in bb_df.columns if "m" in c), None)

        upper = safe_float(bb_df[upper_col].iloc[-1])
        lower = safe_float(bb_df[lower_col].iloc[-1])
        mid = safe_float(bb_df[mid_col].iloc[-1])
        price = safe_float(df["Close"].iloc[-1])

        if any(v is None for v in [upper, lower, mid, price]):
            raise ValueError("Invalid Bollinger Band values")

        if price < lower:
            band, score = "Oversold (Buy)", 10
        elif price > upper:
            band, score = "Overbought (Sell)", 0
        else:
            band, score = "Neutral", 5

        bb_width_val = ((upper - lower) / mid) * 100 if mid and mid != 0 else None
        bb_width_score = 0
        bb_width_desc = "Volatile"

        if bb_width_val is not None:
            if bb_width_val < 5:
                bb_width_score = 5
                bb_width_desc = f"Consolidation (Narrow, {bb_width_val:.2f}%)"
            else:
                bb_width_desc = f"Volatile (Wide, {bb_width_val:.2f}%)"
            bb_width_val = round(bb_width_val, 2)

        return {
            "bb_high": {"value": round(upper, 2), "score": 0},
            "bb_mid": {"value": round(mid, 2), "score": 0},
            "bb_low": {"value": round(lower, 2), "score": score, "desc": band},
            "bb_width": {"value": bb_width_val,"raw": bb_width_val, "score": bb_width_score, "desc": bb_width_desc},
        }

    return _wrap_calc(_inner, "Bollinger Bands")


def compute_rvol(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Volume" not in df.columns:
            raise ValueError("Missing Volume column for RVOL")

        vol_today = safe_float(df["Volume"].iloc[-1])
        avg_vol_20 = safe_float(df["Volume"].tail(20).mean())

        if avg_vol_20 is None or avg_vol_20 == 0:
            raise ValueError("Invalid 20-day avg volume")

        rvol = safe_float(vol_today / avg_vol_20)
        if rvol is None:
            raise ValueError("RVOL NaN")

        # NEW SCORING:
        # Score 10 for high volume
        # Score 0 for low volume
        # Score 5 for normal volume
        if rvol > 1.5:
            score = 10
        elif rvol < 0.8: # Penalize low volume
            score = 0
        else: # Normal volume (0.8 to 1.5)
            score = 5

        return {"rvol": {"value": round(rvol, 2), "score": score}}

    return _wrap_calc(_inner, "RVOL")


def compute_obv_divergence(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"Close", "Volume"}.issubset(df.columns):
            raise ValueError("Missing columns for OBV")

        obv = ta.obv(df["Close"], df["Volume"])
        if obv is None or obv.empty:
            raise ValueError("OBV series empty")

        price_change = safe_float(df["Close"].iloc[-1]) - safe_float(df["Close"].iloc[-5])
        obv_change = safe_float(obv.iloc[-1]) - safe_float(obv.iloc[-5])

        if price_change is None or obv_change is None:
            raise ValueError("OBV change NaN")

        if price_change * obv_change > 0:
            signal, score = "Confirming", 10
        elif price_change * obv_change < 0:
            signal, score = "Diverging", 0
        else:
            signal, score = "Neutral", 5

        return {"obv_div": {"value": signal, "score": score}}

    return _wrap_calc(_inner, "OBV Divergence")


def compute_vpt(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"Close", "Volume"}.issubset(df.columns):
            raise ValueError("Missing columns for VPT")

        vpt = ta.pvt(df["Close"], df["Volume"])
        if vpt is None or vpt.empty:
            raise ValueError("VPT series empty")

        vpt_val = safe_float(vpt.iloc[-1])
        vpt_prev = safe_float(vpt.iloc[-5]) if len(vpt) >= 5 else vpt_val

        if vpt_val is None or vpt_prev is None:
            raise ValueError("VPT values invalid")

        if vpt_val > vpt_prev:
            signal, score = "Accumulation", 10
        elif vpt_val < vpt_prev:
            signal, score = "Distribution", 0
        else:
            signal, score = "Neutral", 5

        return {"vpt": {"value": round(vpt_val, 2), "score": score, "desc": signal}}

    return _wrap_calc(_inner, "VPT")


def compute_volume_spike(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Volume" not in df.columns:
            raise ValueError("Missing Volume data")

        vol_today = safe_float(df["Volume"].iloc[-1])
        avg_vol_20 = safe_float(df["Volume"].tail(20).mean())

        if avg_vol_20 is None or avg_vol_20 == 0:
            raise ValueError("Invalid volume mean")

        spike_ratio = safe_float(vol_today / avg_vol_20)
        if spike_ratio is None:
            raise ValueError("Spike ratio invalid")

        if spike_ratio > 1.5:
            signal, score = "Strong Spike", 10
        elif spike_ratio > 1.2:
            signal, score = "Moderate Spike", 5
        else:
            # Neutral should be 5, not 0
            signal, score = "Normal", 5

        return {"vol_spike_ratio": {"value": round(spike_ratio, 2), "score": score}, "vol_spike_signal": {"value": signal, "score": score}}

    return _wrap_calc(_inner, "Volume Spike")


def compute_atr(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC for ATR")

        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        if atr_series is None or atr_series.empty:
            raise ValueError("ATR empty")

        atr_val = safe_float(atr_series.iloc[-1])
        price = safe_float(df["Close"].iloc[-1])
        if price is None or atr_val is None or price == 0:
            raise ValueError("Invalid ATR inputs")

        atr_pct_val = round(atr_val / price * 100, 2)
        if atr_pct_val is None:
            raise ValueError("ATR % invalid")

        if 1.0 <= atr_pct_val <= 3.0:
            score = 10
        elif atr_pct_val < 1.0:
            score = 7
        elif atr_pct_val <= 5.0:
            score = 5
        else:
            score = 0

        return {
            "atr_14": {"value": round(atr_val, 2), "score": score},
            "atr_pct": {
                "value": atr_pct_val,
                "score": score,
                "desc": f"{atr_pct_val:.2f}% ATR Volatility",
                }
        }

    return _wrap_calc(_inner, "ATR")


# Ichimoku: PURE function now (no internal fetching).
def compute_ichimoku(symbol: str, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _manual_ichimoku(df_local: pd.DataFrame):
        high, low = df_local["High"], df_local["Low"]
        conv = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        base = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        span_a = ((conv + base) / 2).shift(26)
        span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        return conv, base, span_a, span_b

    def _inner():
        _df_local = df
        if _df_local is None or _df_local.empty or not {"High", "Low", "Close"}.issubset(_df_local.columns):
            raise ValueError("Missing or invalid OHLC for Ichimoku")

        if len(_df_local) < 60:
            raise ValueError(f"Insufficient data for Ichimoku (need 60 bars, got {len(_df_local)})")

        span_a_val = span_b_val = tenkan_val = kijun_val = None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                ichimoku = ta.ichimoku(
                    high=_df_local["High"],
                    low=_df_local["Low"],
                    close=_df_local["Close"],
                    include_span=True
                )

            if isinstance(ichimoku, tuple):
                if len(ichimoku) >= 6:
                    conv_df, base_df, span_a_df, span_b_df, *_ = ichimoku
                    tenkan_val = safe_float(conv_df.iloc[-1])
                    kijun_val = safe_float(base_df.iloc[-1])
                    span_a_val = safe_float(span_a_df.iloc[-1])
                    span_b_val = safe_float(span_b_df.iloc[-1])
            elif isinstance(ichimoku, pd.DataFrame):
                cols = ichimoku.columns
                conv_col = next((c for c in cols if "TENKAN" in c.upper()), None)
                base_col = next((c for c in cols if "KIJUN" in c.upper()), None)
                span_a_col = next((c for c in cols if "ISA" in c.upper()), None)
                span_b_col = next((c for c in cols if "ISB" in c.upper()), None)
                if conv_col:
                    tenkan_val = safe_float(ichimoku[conv_col].iloc[-1])
                if base_col:
                    kijun_val = safe_float(ichimoku[base_col].iloc[-1])
                if span_a_col:
                    span_a_val = safe_float(ichimoku[span_a_col].iloc[-1])
                if span_b_col:
                    span_b_val = safe_float(ichimoku[span_b_col].iloc[-1])

        except Exception as e:
            logger.warning("Ichimoku Cloud (library) failed: %s", e)

        # If still missing, fallback to manual computation
        if span_a_val is None or span_b_val is None:
            try:
                conv, base, span_a_series, span_b_series = _manual_ichimoku(_df_local)
                tenkan_val = tenkan_val or safe_float(conv.iloc[-1])
                kijun_val = kijun_val or safe_float(base.iloc[-1])
                span_a_val = span_a_val or safe_float(span_a_series.iloc[-1])
                span_b_val = span_b_val or safe_float(span_b_series.iloc[-1])
            except Exception as e:
                logger.warning("Manual Ichimoku computation failed: %s", e)

        if span_a_val is None or span_b_val is None:
            raise ValueError("Ichimoku spans unavailable after manual attempt")

        price = safe_float(_df_local["Close"].iloc[-1])
        upper, lower = max(span_a_val, span_b_val), min(span_a_val, span_b_val)

        strong_bull = (price > upper) and (tenkan_val and kijun_val and tenkan_val > kijun_val)
        strong_bear = (price < lower) and (tenkan_val and kijun_val and tenkan_val < kijun_val)

        if strong_bull:
            signal, score = "Strong Bullish", 10
        elif strong_bear:
            signal, score = "Strong Bearish", 0
        elif price > upper:
            signal, score = "Mild Bullish", 7
        elif price < lower:
            signal, score = "Mild Bearish", 3
        else:
            signal, score = "Neutral", 5

        return {
            "ichi_cloud": {"value": signal, "score": score, "desc": f"Ichimoku Cloud {signal}"},
            "ichi_span_a": {"value": round(span_a_val, 2), "score": 0},
            "ichi_span_b": {"value": round(span_b_val, 2), "score": 0},
            "ichi_tenkan": {"value": round(tenkan_val, 2) if tenkan_val else "N/A", "score": 0},
            "ichi_kijun": {"value": round(kijun_val, 2) if kijun_val else "N/A", "score": 0},
        }

    return _wrap_calc(_inner, "Ichimoku")


def compute_price_action(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC data")

        high, low, close = df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
        if high == low:
            raise ValueError("Invalid range (High == Low)")

        position = (close - low) / (high - low)

        if position >= 0.75:
            score, signal = 10, "Strong Bullish Close"
        elif position >= 0.5:
            score, signal = 5, "Moderate Close"
        else:
            score, signal = 0, "Weak Close"

        return {"price_action": {"value": round(position * 100, 1), "score": score, "signal": signal}}

    return _wrap_calc(_inner, "Price Action")


#
# REVISED: compute_200dma (Horizon-Aware)
#
def compute_200dma(df, close, price, horizon: str = "short_term"):
    ma_length_and_type = None
    def _inner():
        
        # 1. Define equivalent lengths based on horizon
        # short_term (1d): 200-day MA is the standard
        # long_term (1wk): 50-week MA is the standard (~250 days)
        # multibagger (1mo): 12-month MA is the standard
        # intraday (15m): A 200-period MA is still a valid long-term intraday trend
        
        ma_length_map = {
            "intraday": 200,
            "short_term": 200,
            "long_term": 50,
            "multibagger": 12
        }
        ma_type_map = {
            "intraday": "MA",
            "short_term": "DMA",
            "long_term": "WMA",
            "multibagger": "MMA"
        }
        
        ma_length = ma_length_map.get(horizon, 200)
        ma_type = ma_type_map.get(horizon, "MA")

        # 2. Check if we have enough data
        num_bars = len(close.dropna())
        if num_bars < ma_length:
            raise ValueError(f"Insufficient data ({num_bars} bars) for {ma_length}{ma_type}")

        # 3. Calculate
        sma_series = ta.sma(close, length=ma_length)
        sma_val = safe_float(sma_series.iloc[-1])
        if sma_val is None:
            raise ValueError(f"{ma_length}{ma_type} NaN")

        pct_diff = ((price - sma_val) / sma_val) * 100

        # 4. Score
        if pct_diff > 0:
            score, desc = 10, f"Uptrend (Above {ma_length} {ma_type})"
        elif abs(pct_diff) < 3:
            score, desc = 5, f"Neutral (Near {ma_length} {ma_type})"
        else:
            score, desc = 0, f"Downtrend (Below {ma_length} {ma_type})"

        # 5. Return dynamic keys
        key_name = f"price_vs_{ma_length}{ma_type.lower()}_pct"
        ma_key_name = f"{ma_length}{ma_type.lower()}"
        ma_length_and_type = (ma_length, ma_type)
        return {
            key_name: {"value": round(pct_diff, 2), "score": score, "desc": desc},
            ma_key_name: {"value": round(sma_val, 2), "score": 0},
        }

    return _wrap_calc(_inner, f"Long-Term MA {ma_length_and_type}")

def compute_volume_trend(df, indicators):
    def _inner():
        volume = df["Volume"]
        vol_avg50 = safe_float(volume.tail(50).mean())
        vol_today = safe_float(volume.iloc[-1])
        if vol_avg50 is None or vol_today is None:
            raise ValueError("Volume NaN")

        trend = (
            "Rising"
            if vol_today > 1.2 * vol_avg50
            else "Falling" if vol_today < 0.8 * vol_avg50
            else "Neutral"
        )
        score = 10 if trend == "Rising" else 5 if trend == "Neutral" else 0
        return {
            "vol_trend": {
                "value": trend,
                "score": score,
                "desc": f"{trend} volume trend"
            }
        }

    return _wrap_calc(_inner, "Volume Trend")


def compute_volume_vs_avg20(df, indicators):
    def _inner():
        volume = df["Volume"]
        vol_avg20 = safe_float(volume.tail(20).mean())
        vol_today = safe_float(volume.iloc[-1])
        if vol_avg20 is None or vol_today is None or vol_avg20 == 0:
            raise ValueError("Invalid volume data")
        ratio = vol_today / vol_avg20
        return {"vol_vs_avg20": {"value": round(ratio, 2), "score": 5}}

    return _wrap_calc(_inner, "Volume vs Avg20")


#
# REVISED: compute_dma_cross (Horizon-Aware)
#
def compute_dma_cross(df, close, horizon: str = "short_term"):
    short_len = long_len = 0  # Default initialization
    def _inner():
        
        # 1. Define equivalent lengths
        # short_term (1d): 20-day / 50-day
        # long_term (1wk): 10-week / 40-week (common equivalents for 50/200 day)
        # multibagger (1mo): 6-month / 12-month
        # intraday (15m): 20-bar / 50-bar is a standard relative cross
        
        length_map = {
            "intraday": (20, 50),
            "short_term": (20, 50),
            "long_term": (10, 40),
            "multibagger": (6, 12)
        }
        type_map = {
            "intraday": ("MA", "MA"),
            "short_term": ("DMA", "DMA"),
            "long_term": ("WMA", "WMA"),
            "multibagger": ("MMA", "MMA")
        }
        
        short_len, long_len = length_map.get(horizon, (20, 50))
        short_type, long_type = type_map.get(horizon, ("MA", "MA"))
        
        # 2. Calculate
        dma_short = ta.sma(close, length=short_len)
        dma_long = ta.sma(close, length=long_len)

        dma_short_val = safe_float(dma_short.iloc[-1])
        dma_long_val = safe_float(dma_long.iloc[-1])

        if dma_short_val is None or dma_long_val is None:
            raise ValueError("DMA values NaN")

        # 3. Score
        if dma_short_val > dma_long_val:
            score, desc = 10, f"Bullish ({short_len} > {long_len})"
        else:
            score, desc = 0, f"Bearish ({short_len} < {long_len})"
            
        # 4. Return dynamic keys
        short_key = f"dma_{short_len}{short_type.lower()}"
        long_key = f"dma_{long_len}{long_type.lower()}"
        cross_key = f"dma_{short_len}_{long_len}_cross"

        return {
            short_key: {"value": round(dma_short_val, 2), "score": 0},
            long_key: {"value": round(dma_long_val, 2), "score": 0},
            cross_key: {"value": desc, "score": score, "desc": desc},
        }

    return _wrap_calc(_inner, f"{short_len}/{long_len} DMA Cross")

#
# NEW (from compute_ema_crossover): compute_ma_crossover (Horizon-Aware)
# Note: Renamed to be generic, as it now uses SMAs for long-term
#
def compute_ma_crossover(df: pd.DataFrame, horizon: str = "short_term"):
    ma_type = ""
    def _inner():
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Missing Close column")

        close = df["Close"]
        
        # 1. Define equivalent lengths
        # We will use the same logic from our other functions
        
        len_map = {
            "intraday": (20, 50, 200),     # 20, 50, 200 bar EMA
            "short_term": (20, 50, 200),  # 20, 50, 200 day EMA
            "long_term": (10, 40, 50),    # 10, 40, 50 week SMA (long_term uses SMA)
            "multibagger": (6, 12, 12)    # 6, 12 month SMA (12 is the longest)
        }
        type_map = {
            "intraday": "EMA",
            "short_term": "EMA",
            "long_term": "WMA",
            "multibagger": "MMA"
        }
        
        l_short, l_mid, l_long = len_map.get(horizon, (20, 50, 200))
        ma_type = type_map.get(horizon, "EMA")
        
        # 2. Calculate (Use EMA or SMA based on horizon)
        if horizon in ("long_term", "multibagger"):
            ma_short = ta.sma(close, length=l_short)
            ma_mid = ta.sma(close, length=l_mid)
            ma_long = ta.sma(close, length=l_long)
        else:
            ma_short = ta.ema(close, length=l_short)
            ma_mid = ta.ema(close, length=l_mid)
            ma_long = ta.ema(close, length=l_long)

        # 3. Get values
        v_short = safe_float(ma_short.iloc[-1])
        v_mid = safe_float(ma_mid.iloc[-1])
        v_long = safe_float(ma_long.iloc[-1])

        if any(v is None for v in [v_short, v_mid, v_long]):
            raise ValueError("Invalid MA values")

        # 4. Score
        if v_short > v_mid > v_long:
            trend, score = "Bullish", 10
        elif v_short < v_mid < v_long:
            trend, score = "Bearish", 0
        else:
            trend, score = "Neutral", 5
            
        # 5. Return dynamic keys
        key_short = f"ma_{l_short}{ma_type.lower()}"
        key_mid = f"ma_{l_mid}{ma_type.lower()}"
        key_long = f"ma_{l_long}{ma_type.lower()}"
        key_cross = "ma_cross_trend"

        return {
            key_short: {"value": round(v_short, 2), "score": score},
            key_mid: {"value": round(v_mid, 2), "score": score},
            key_long: {"value": round(v_long, 2), "score": score},
            key_cross: {"value": trend, "score": score},
        }

    return _wrap_calc(_inner, f"MA Crossover ({ma_type})")

#
# REVISED: compute_relative_strength (Horizon-Aware)
#
def compute_relative_strength(symbol, df, benchmark_df, horizon: str = "short_term") -> Dict[str, Dict[str, Any]]:
    lookback = 20  # Default initialization
    def _inner():
        if df is None or df.empty:
            raise ValueError("Stock data missing")
        if benchmark_df is None or benchmark_df.empty:
            raise ValueError("Benchmark data missing")

        # 1. Define equivalent lookbacks
        # short_term (1d): 20 days (1 month)
        # long_term (1wk): 52 weeks (1 year)
        # multibagger (1mo): 12 months (1 year)
        # intraday (15m): 20 bars (5 hours)
        
        lookback_map = {
            "intraday": 20,
            "short_term": 20,
            "long_term": 52,
            "multibagger": 12
        }
        lookback = lookback_map.get(horizon, 20)
        
        # ... (rest of the function is the same, just uses `lookback`) ...
        
        stock_close = None
        for col in ("Adj Close", "Close"):
            if col in df.columns:
                stock_close = df[col].dropna()
                break
        if stock_close is None or stock_close.empty:
            raise ValueError("Stock close series missing")

        nifty_close = None
        for col in ("Adj Close", "Close"):
            if col in benchmark_df.columns:
                nifty_close = benchmark_df[col].dropna()
                break
        if nifty_close is None or nifty_close.empty:
            raise ValueError("Benchmark close series missing")

        if len(stock_close) <= lookback or len(nifty_close) <= lookback:
            raise ValueError(f"Insufficient data (<{lookback} bars) for relative strength")

        stock_ret = (float(stock_close.iloc[-1].item()) / float(stock_close.iloc[-lookback].item()) - 1.0) * 100.0
        nifty_ret = (float(nifty_close.iloc[-1].item()) / float(nifty_close.iloc[-lookback].item()) - 1.0) * 100.0

        rel_strength = stock_ret - nifty_ret
        if rel_strength > 0:
            score, desc = 10, "Outperforming"
        elif rel_strength < 0:
            score, desc = 0, "Underperforming"
        else:
            score, desc = 5, "In-line"

        return {
            "rel_strength_nifty": {
                "value": round(rel_strength, 2),
                "score": score,
                "desc": f"{desc} vs NIFTY ({rel_strength:.2f}%) over {lookback} bars"
            }
        }

    return _wrap_calc(_inner, "Relative Strength vs NIFTY (%)")

def compute_entry_price(df: pd.DataFrame, price: float):
    def _inner():
        # NEW: Calculate bb_mid (SMA 20) directly
        sma20_series = ta.sma(df["Close"], length=20)
        if sma20_series is None or sma20_series.empty:
            raise ValueError("SMA(20) for BB Mid unavailable")

        bb_mid = safe_float(sma20_series.iloc[-1])
        if bb_mid is None:
            bb_mid = price # Fallback in case of NaN

        # Your original logic, now using `df`
        last_high = df["High"].tail(10).max() if "High" in df.columns else None
        
        # Use bb_mid if last_high isn't available
        confirm_val = round(last_high * 1.005, 2) if last_high else round(bb_mid, 2)
        
        entry_score = 10 if price > (confirm_val or 0) else 5
        
        return {"entry_confirm": {"value": confirm_val, "score": entry_score, "desc": "Approximate breakout confirmation level"}}

    return _wrap_calc(_inner, "Entry Price (Confirm)")

#
# REVISED: compute_200dma_slope (Horizon-Aware)
#
def compute_200dma_slope(df: pd.DataFrame, close: pd.Series, horizon: str = "short_term"):
    ma_length_and_type = None
    def _inner():
        # 1. Define equivalent lengths (must match compute_200dma)
        ma_length_map = {
            "intraday": 200,
            "short_term": 200,
            "long_term": 50,
            "multibagger": 12
        }
        ma_type_map = {
            "intraday": "MA",
            "short_term": "DMA",
            "long_term": "WMA",
            "multibagger": "MMA"
        }
        
        ma_length = ma_length_map.get(horizon, 200)
        ma_type = ma_type_map.get(horizon, "MA")
        
        # 2. Calculate the MA series
        sma_series = ta.sma(close, length=ma_length)
        
        # 3. Check data for slope calculation
        lookback = 10
        if sma_series.isnull().all() or len(sma_series.dropna()) < lookback:
            raise ValueError(f"Insufficient data for {ma_length}{ma_type} slope (need {lookback} periods)")
        
        y = sma_series.tail(lookback).values
        if np.isnan(y).any():
             raise ValueError("SMA values contain NaN, cannot calculate slope")

        # 4. Calculate slope
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        angle = degrees(atan(slope))

        if angle > 0.1:
            score, trend = 10, "Rising"
        elif angle < -0.1:
            score, trend = 0, "Falling"
        else:
            score, trend = 5, "Flat"
            
        key_name = f"{ma_length}{ma_type.lower()}_slope"
        ma_length_and_type = (ma_length, ma_type)
        return {key_name: {"value": trend, "score": score, "desc": f"Slope: {angle:.2f}°"}}

    return _wrap_calc(_inner, f" {ma_length_and_type} Trend (Slope)")

def compute_atr_sl(df, indicators, high, low, close, price):
    def _inner():
        atr_val = safe_float(ta.atr(high, low, close, length=14).iloc[-1])
        if atr_val is None:
            raise ValueError("ATR NaN")
        sl = price - (2 * atr_val)
        return {
            "sl_2x_atr": {"value": round(sl, 2), "score": 0},
        }

    return _wrap_calc(_inner, "Suggested SL (2xATR)")

#
# REVISED: compute_supertrend (More robust, handles NaN)
#
def compute_supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC data for SuperTrend")

        try:
            st_df = ta.supertrend(df["High"], df["Low"], df["Close"], length=length, multiplier=multiplier)
        except Exception as e:
            raise ValueError(f"SuperTrend failed: {e}")

        if st_df is None or st_df.empty:
            raise ValueError("SuperTrend DataFrame empty")

        st_df.columns = [c.lower() for c in st_df.columns]
        trend_col = next((c for c in st_df.columns if "direction" in c or "trend" in c), None)
        
        if not trend_col:
            # Fallback to the last column (which is usually the trend)
            trend_col = st_df.columns[-1]

        # --- NEW ROBUSTNESS FIX ---
        # Find the last valid (non-NaN) value in the trend series
        st_series = st_df[trend_col].dropna()
        if st_series.empty:
            raise ValueError("SuperTrend calculation resulted in all NaN values")

        # Get the last valid value
        trend_val = safe_float(st_series.iloc[-1])
        # --- END FIX ---

        if trend_val is None:
            # This check should be redundant now, but good to have
            raise ValueError("Could not find a valid SuperTrend value after NaN check")

        signal = "Bullish" if trend_val > 0 else "Bearish"
        score = 10 if signal == "Bullish" else 0

        return {
            "supertrend_signal": {"value": signal, "score": score, "desc": f"SuperTrend ({length},{multiplier}) {signal}"}
        }

    return _wrap_calc(_inner, "SuperTrend")

def compute_regression_slope(df: pd.DataFrame, length: int = 20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Missing Close for regression slope")

        if len(df) < length:
            raise ValueError(f"Insufficient data ({len(df)}) for regression slope")

        y = df["Close"].tail(length).values
        x = np.arange(length)
        slope, intercept = np.polyfit(x, y, 1)

        angle = degrees(atan(slope))
        desc = "Rising" if angle > 1 else "Falling" if angle < -1 else "Flat"
        score = 10 if angle > 2 else 5 if abs(angle) <= 2 else 0

        return {
            "reg_slope": {"value": round(angle, 2), "score": score, "desc": f"Trend: {desc} ({angle:.2f}°)"}
        }

    return _wrap_calc(_inner, "Regression Slope")


def compute_cci(df: pd.DataFrame, length: int = 20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC for CCI")

        cci_series = ta.cci(df["High"], df["Low"], df["Close"], length=length)
        if cci_series is None or cci_series.empty:
            raise ValueError("CCI empty")

        cci_val = safe_float(cci_series.iloc[-1])
        if cci_val is None:
            raise ValueError("CCI NaN")

        if cci_val < -100:
            desc, score = "Oversold (Buy)", 10
        elif cci_val > 100:
            desc, score = "Overbought (Sell)", 0
        else:
            desc, score = "Neutral", 5

        return {"cci": {"value": round(cci_val, 2), "score": score, "desc": desc}}

    return _wrap_calc(_inner, "CCI")

#
# REVISED: compute_rsi_slope
#
def compute_rsi_slope(df: pd.DataFrame,
                      rsi_series: Optional[pd.Series] = None,
                      length: int = 14,
                      lookback: int = 5) -> Dict[str, Dict[str, Any]]:
    def _inner():
        rsi_data = None
        
        # 🆕 FIX 2: Attempt to rebuild and use the injected Series
        if rsi_series and isinstance(rsi_series, dict) and rsi_series:
            try:
                # Rebuild pandas Series from dictionary (must handle index if available)
                rsi_data = pd.Series(rsi_series).sort_index()
            except Exception as e:
                logger.warning(f"Failed to rebuild RSI series from dict: {e}. Falling back to recalculation.")

        # Recalculation Fallback (Original Inefficiency, kept for robustness)
        if rsi_data is None or rsi_data.empty:
            if "Close" not in df.columns:
                raise ValueError("Missing Close for RSI slope computation")
            rsi_data = ta.rsi(df["Close"], length=length) 

        if rsi_data is None or rsi_data.empty:
            raise ValueError("RSI data unavailable for slope")
        
        # --- Start original slope logic ---
        if len(rsi_data.dropna()) < lookback:
            raise ValueError(f"Insufficient RSI data ({len(rsi_data.dropna())}) for slope (need {lookback})")

        recent = rsi_data.tail(lookback).values

        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)

        if slope > 0.5:
            score, desc = 10, "RSI accelerating (Bullish Momentum)"
        elif slope < -0.5:
            score, desc = 0, "RSI decelerating (Momentum Loss)"
        else:
            score, desc = 5, "RSI stable (Neutral Momentum)"

        return {
            "rsi_slope": {
                "value": round(slope, 2),
                "score": score,
                "desc": f"{desc} [{slope:.2f}/step]"
            }
        }

    return _wrap_calc(_inner, "RSI Slope")

def compute_bb_percent_b(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Missing Close for BB%B")

        bb_df = ta.bbands(df["Close"], length=length, std=std_dev)
        if bb_df is None or bb_df.empty:
            raise ValueError("Bollinger Bands empty")

        bb_df.columns = [c.lower() for c in bb_df.columns]
        upper_col = next((c for c in bb_df.columns if "u" in c.lower()), None)
        lower_col = next((c for c in bb_df.columns if "l" in c.lower()), None)
        upper = bb_df[upper_col].iloc[-1]
        lower = bb_df[lower_col].iloc[-1]

        if lower == upper:
            raise ValueError("Invalid BB width")
        price = safe_float(df["Close"].iloc[-1])
        bb_percent_b = (price - lower) / (upper - lower)
        score = 10 if bb_percent_b < 0.2 else 0 if bb_percent_b > 0.8 else 5

        return {"bb_percent_b": {"value": round(bb_percent_b, 3), "score": score, "desc": f"{bb_percent_b:.2f} position in band"}}

    return _wrap_calc(_inner, "Bollinger %B")


def compute_cmf(df: pd.DataFrame, length: int = 20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
            raise ValueError("Missing OHLCV for CMF")

        cmf_series = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=length)
        if cmf_series is None or cmf_series.empty:
            raise ValueError("CMF empty")

        cmf_val = safe_float(cmf_series.iloc[-1])
        if cmf_val is None:
            raise ValueError("CMF NaN")

        if cmf_val > 0.05:
            desc, score = "Accumulation (Buy Pressure)", 10
        elif cmf_val < -0.05:
            desc, score = "Distribution (Sell Pressure)", 0
        else:
            desc, score = "Neutral", 5

        return {"cmf_signal": {"value": round(cmf_val, 3), "score": score, "desc": desc}}

    return _wrap_calc(_inner, "Chaikin Money Flow")


def compute_donchian(df: pd.DataFrame, length: int = 20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC for Donchian")

        upper = df["High"].rolling(length).max().iloc[-1]
        lower = df["Low"].rolling(length).min().iloc[-1]
        price = df["Close"].iloc[-1]

        if price > upper:
            signal, score = "Breakout Above Upper", 10
        elif price < lower:
            signal, score = "Breakdown Below Lower", 0
        else:
            signal, score = "Inside Range", 5

        return {"donchian_signal": {"value": signal, "score": score, "desc": f"{signal} ({length}-period)"}}

    return _wrap_calc(_inner, "Donchian Channel")


def compute_nifty_trend_score(benchmark_df: pd.DataFrame):
    def _inner():
        try:
            if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
                raise ValueError("Missing Close in NIFTY data")

            close = benchmark_df["Close"]
            ema_50 = ta.ema(close, 50).iloc[-1]
            ema_200 = ta.ema(close, 200).iloc[-1]
            latest = close.iloc[-1]

            diff = (latest - ema_200) / ema_200 * 100
            if latest > ema_50 > ema_200:
                trend, score = "Strong Uptrend", 9
            elif latest > ema_200:
                trend, score = "Moderate Uptrend", 7
            elif latest < ema_200 and latest > ema_50:
                trend, score = "Neutral / Pullback", 5
            else:
                trend, score = "Downtrend", 2

            return {
                "nifty_trend_score": {
                    "value": round(diff, 2),
                    "score": score,
                    "desc": f"{trend} ({diff:.2f}% above 200 EMA)"
                }
            }
        except Exception as e:
            logger.warning(f"compute_nifty_trend_score failed: {e}")
            return {"nifty_trend_score": {"value": None, "score": 0, "desc": "NIFTY trend unavailable"}}
    return _wrap_calc(_inner, "NIFTY Trend Score")


def compute_gap_percent(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        if df is None or df.empty or len(df) < 2:
            raise ValueError("Insufficient daily data for gap")

        prev_close = safe_float(df["Close"].iloc[-2])
        today_open = safe_float(df["Open"].iloc[-1])

        if prev_close is None or today_open is None or prev_close == 0:
            raise ValueError("Invalid open/close for gap")

        gap_pct = ((today_open - prev_close) / prev_close) * 100

        if gap_pct > 2.0:
            score, desc = 10, "Strong Gap Up"
        elif gap_pct > 0.5:
            score, desc = 7, "Moderate Gap Up"
        elif gap_pct < -2.0:
            score, desc = 1, "Strong Gap Down"
        elif gap_pct < -0.5:
            score, desc = 3, "Moderate Gap Down"
        else:
            score, desc = 5, "No Gap"

        return {"gap_percent": {"value": round(gap_pct, 2), "score": score, "desc": desc}}

    return _wrap_calc(_inner, "Gap Percent")

#
# --- ADD THIS NEW FUNCTION TO indicators.py ---
#

def compute_psar(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculates the Parabolic SAR (PSAR) for trend direction and potential stops.
    """
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC data for Parabolic SAR")

        psar_df = ta.psar(df["High"], df["Low"], df["Close"])
        if psar_df is None or psar_df.empty:
            raise ValueError("PSAR calculation failed")

        # FIX: Dynamically find columns starting with PSARl (long) and PSARs (short)
        # pandas_ta names them like PSARl_0.02_0.2
        col_l = next((c for c in psar_df.columns if c.startswith("PSARl")), None)
        col_s = next((c for c in psar_df.columns if c.startswith("PSARs")), None)
        col_r = next((c for c in psar_df.columns if c.startswith("PSARr")), None) # Reversal

        # Get values safely
        val_l = safe_float(psar_df[col_l].iloc[-1]) if col_l else None
        val_s = safe_float(psar_df[col_s].iloc[-1]) if col_s else None
        
        # The level is whichever one is active (not NaN), or fallback
        psar_level = val_l if val_l is not None else val_s
        
        trend_val = safe_float(psar_df[col_r].iloc[-1]) if col_r else 0

        if psar_level is None:
            raise ValueError("PSAR level is NaN")

        if trend_val > 0:
            signal, score = "Bullish", 10
        else:
            signal, score = "Bearish", 0
        return {
            "psar_trend": { "value": signal, "score": score, "desc": f"PSAR Trend is {signal}" },
            "psar_level": { "value": round(psar_level, 2), "score": 0, "desc": f"PSAR Trailing Stop: {round(psar_level, 2)}" }
        }

    return _wrap_calc(_inner, "Parabolic SAR")

#
# --- ADD THIS NEW FUNCTION TO indicators.py ---
#

def compute_keltner_squeeze(df: pd.DataFrame, 
                            bb_length: int = 20, 
                            bb_std: float = 2.0,
                            kc_length: int = 20,
                            kc_scalar: float = 1.5) -> Dict[str, Dict[str, Any]]:
    """
    Calculates Keltner Channels (KC) and checks for a TTM Squeeze
    (Bollinger Bands go inside Keltner Channels).
    """
    def _inner():
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Missing OHLC data for Keltner/BB Squeeze")

        # 1. Calculate Bollinger Bands
        bb_df = ta.bbands(df["Close"], length=bb_length, std=bb_std)
        if bb_df is None:
            raise ValueError("Bollinger Bands calculation failed")
            
        bb_upper_col = next(c for c in bb_df.columns if "u" in c.lower())
        bb_lower_col = next(c for c in bb_df.columns if "l" in c.lower())
        bb_upper = safe_float(bb_df[bb_upper_col].iloc[-1])
        bb_lower = safe_float(bb_df[bb_lower_col].iloc[-1])

        # 2. Calculate Keltner Channels
        # Note: We use kc_scalar=1.5, which is a common setting for squeezes
        kc_df = ta.kc(df["High"], df["Low"], df["Close"], length=kc_length, scalar=kc_scalar)
        if kc_df is None:
            raise ValueError("Keltner Channels calculation failed")

        kc_upper_col = next(c for c in kc_df.columns if "u" in c.lower())
        kc_lower_col = next(c for c in kc_df.columns if "l" in c.lower())
        kc_upper = safe_float(kc_df[kc_upper_col].iloc[-1])
        kc_lower = safe_float(kc_df[kc_lower_col].iloc[-1])

        if any(v is None for v in [bb_upper, bb_lower, kc_upper, kc_lower]):
            raise ValueError("NaN values in BB or KC bands")

        # 3. Check for the Squeeze
        is_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        if is_squeeze:
            signal, score = "Squeeze On", 10 # 10 = High potential energy
        else:
            signal, score = "Squeeze Off", 5 # 5 = Neutral, trend is active

        return {
            "ttm_squeeze": {
                "value": signal,
                "score": score,
                "desc": f"Volatility Squeeze: {signal}"
            },
            "kc_upper": {"value": round(kc_upper, 2), "score": 0},
            "kc_lower": {"value": round(kc_lower, 2), "score": 0}
        }
        
    return _wrap_calc(_inner, "TTM Squeeze")

# In services/indicators.py (add these new functions near the other compute_... functions):

def compute_ema_slope(df: pd.DataFrame, length_s: int = 20, length_l: int = 50, lookback: int = 10) -> Dict[str, Dict[str, Any]]:
    """Calculates the angle/slope of short-term and mid-term EMAs over a recent lookback window."""
    def _inner():
        if df is None or len(df) < length_l or "Close" not in df.columns:
            raise ValueError("Insufficient data for EMA slopes")
        
        close = df["Close"]
        
        # Calculate EMAs
        ema_s_series = ta.ema(close, length=length_s)
        ema_l_series = ta.ema(close, length=length_l)
        
        if ema_s_series.empty or ema_l_series.empty:
            raise ValueError("EMA series calculation failed")
            
        y_s = ema_s_series.tail(lookback).values
        y_l = ema_l_series.tail(lookback).values
        
        if np.isnan(y_s).any() or np.isnan(y_l).any():
            raise ValueError("NaNs in EMA lookback window")

        x = np.arange(lookback)
        
        # Short Slope: calculate angle from linear regression slope
        slope_s, _ = np.polyfit(x, y_s, 1)
        angle_s = degrees(atan(slope_s))
        
        # Long Slope: calculate angle
        slope_l, _ = np.polyfit(x, y_l, 1)
        angle_l = degrees(atan(slope_l))
        
        score = 10 if angle_s > 1.5 else 5 if angle_s > 0 else 0
        desc = f"{length_s}EMA Angle: {angle_s:.2f}°"

        return {
            # 🆕 RAW INPUT FOR TREND STRENGTH COMPOSITE
            "ema_20_slope": {"value": round(angle_s, 2), "raw": angle_s, "score": score, "desc": desc},
            "ema_50_slope": {"value": round(angle_l, 2), "raw": angle_l, "score": 0, "desc": f"{length_l}EMA Angle: {angle_l:.2f}°"},
        }
    return _wrap_calc(_inner, "EMA Slopes")


def compute_true_range(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculates the latest True Range (TR) and True Range Percentage (TR%)."""
    def _inner():
        if df is None or len(df) < 2 or not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError("Insufficient data for True Range")
            
        tr_series = ta.true_range(df["High"], df["Low"], df["Close"])
        if tr_series.empty:
            raise ValueError("True Range calculation failed")
        
        tr_val = safe_float(tr_series.iloc[-1])
        price = safe_float(df["Close"].iloc[-1])
        
        if tr_val is None or price is None or price == 0:
            raise ValueError("Invalid TR or Price value")

        tr_pct = (tr_val / price) * 100
        
        # Raw TR is used for compression checks
        return {
            "true_range": {"value": round(tr_val, 2), "raw": tr_val, "score": 0, "desc": f"Raw TR: {tr_val:.2f}"},
            "true_range_pct": {"value": round(tr_pct, 2), "raw": tr_pct, "score": 5, "desc": f"TR %: {tr_pct:.2f}%"}
        }
    return _wrap_calc(_inner, "True Range")


def compute_historical_volatility(df: pd.DataFrame, periods: Tuple[int, int] = (10, 20)) -> Dict[str, Dict[str, Any]]:
    """Calculates annualized Historical Volatility (HV) for multiple short-term periods."""
    def _inner():
        if df is None or len(df) < periods[1] + 1 or "Close" not in df.columns:
            raise ValueError(f"Insufficient data for HV (need >{periods[1]} bars)")
            
        close = df["Close"].astype(float)
        
        # Calculate logarithmic returns
        log_returns = np.log(close / close.shift(1)).dropna()
        
        # Annualization factor (assuming 252 trading days/periods)
        annualization_factor = np.sqrt(252)
        output = {}
        
        for p in periods:
            if len(log_returns) >= p:
                hv_series = log_returns.tail(p)
                hv_val = hv_series.std() * annualization_factor * 100 # Convert to percent
                
                # 🆕 FIX: Implement HV Scoring Logic
                if hv_val < 20.0:
                    score_hv = 10  # Low Volatility (< 20%) -> Stable
                elif hv_val < 40.0:
                    score_hv = 5   # Medium Volatility (20-40%) -> Neutral
                else:
                    score_hv = 2   # High Volatility (> 40%) -> Risky
                
                # Output format remains the same
                output[f"hv_{p}"] = {
                    "value": round(hv_val, 2), 
                    "raw": hv_val, 
                    "score": score_hv, 
                    "desc": f"{p}D HV: {hv_val:.2f}%"
                }

        if not output:
            raise ValueError(f"HV calculation failed for periods {periods}")
            
        return output
    return _wrap_calc(_inner, "Historical Volatility")



def compute_volume_score(indicators: Dict[str, Any]) -> float:
    rvol = indicators.get("rvol", {}).get("score", 5)
    obv = indicators.get("obv_div", {}).get("score", 5)
    vpt = indicators.get("vpt", {}).get("score", 5)
    volume_spike = indicators.get("vol_spike_ratio", {}).get("score", 5)

    score = (
        (rvol * 0.25) +
        (obv * 0.35) +
        (vpt * 0.25) +
        (volume_spike * 0.15)
    )

    return round(score, 2)


def compute_technical_score(indicators: Dict[str, Dict[str, Any]],
                            weights: Optional[Dict[str, float]] = None) -> int:
    if not indicators:
        return 0

    weights = dict(weights or TECHNICAL_WEIGHTS)
    score = 0.0
    weight_sum = 0.0

    def _val(key: str):
        item = indicators.get(key)
        if not isinstance(item, dict):
            return None
        return item.get("value")

    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            return float(str(v).replace("%", "").replace("₹", "").strip())
        except Exception:
            return None

    adx_val = _val("adx")
    adx_f = _to_float(adx_val)
    if adx_f is not None:
        if adx_f > 50:
            weights.update({"macd_cross": 2.0, "price_vs_200dma_pct": 2.0, "vwap_bias": 1.2, "rsi": 0.8, "stoch_k": 0.5})
        elif adx_f > 25:
            weights.update({"macd_cross": 1.5, "price_vs_200dma_pct": 1.5, "vwap_bias": 1.0})
        elif adx_f < 20:
            weights.update({"macd_cross": 0.6, "price_vs_200dma_pct": 0.6, "vwap_bias": 0.5, "rsi": 1.2, "stoch_k": 1.2})

    def retrieve_and_contribute(key: str, indicator_key: Optional[str] = None):
        nonlocal score, weight_sum
        indicators_key = indicator_key or key
        indicator_item = indicators.get(indicators_key)
        if not indicator_item:
            return

        s = indicator_item.get("score")
        v = indicator_item.get("value")

        if s is None or (s == 0 and (v is None or str(v).upper() in ["N/A", "NONE", "NA"])):
            return

        w = weights.get(key, 1.0)
        weight_sum += 10 * w
        score += s * w

    metric_list = [
        ("rsi", None),
        ("macd_cross", None),
        ("macd_hist_z", None),
        ("price_vs_200dma_pct", None),
        ("adx", None),
        ("vwap_bias", None),
        ("vol_trend", None),
        ("rvol", None),
        ("stoch_k", None),
        ("bb_low", None),
        ("bb_width", None),
        ("entry_confirm", None),
        ("dma_20_50_cross", None),
        ("dma_200_slope", None),
        ("ichi_cloud", None),
        ("obv_div", None),
        ("atr_14", None),
        ("vol_spike_ratio", None),
        ("rel_strength_nifty", None),
        ("price_action", None),
    ]

    for key, ikey in metric_list:
        retrieve_and_contribute(key, ikey)
    final_score = round(score / weight_sum * 100, 1) if weight_sum > 0 else 0
    return max(0, min(100, final_score))

def compute_pivot_points(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Classic + Fibonacci Pivot Points (Bundle of 7 metrics).
    Uses previous candle. Includes Alias/Source metadata for UI.
    """
    def _inner():
        if df is None or len(df) < 2:
            raise ValueError("Insufficient data for Pivots")

        # Get PREVIOUS period's data
        prev = df.iloc[-2]
        high = safe_float(prev["High"])
        low = safe_float(prev["Low"])
        close = safe_float(prev["Close"])
        
        # Robust NaN check (The Tweak)
        if any(pd.isna(v) for v in [high, low, close]):
            raise ValueError("Invalid OHLC (NaN) for pivots")

        pp = (high + low + close) / 3
        range_hl = high - low

        r1 = pp + 0.382 * range_hl
        r2 = pp + 0.618 * range_hl
        r3 = pp + 1.000 * range_hl
        
        s1 = pp - 0.382 * range_hl
        s2 = pp - 0.618 * range_hl
        s3 = pp - 1.000 * range_hl

        # Returns standardized dictionary structure
        return {
            "pivot_point":  {"value": round(pp, 2), "raw": pp, "score": 0, "desc": "Pivot Point", "alias": "Pivot Point", "source": "pivot"},
            "resistance_1": {"value": round(r1, 2), "raw": r1, "score": 0, "desc": "Fib R1",      "alias": "Resistance 1", "source": "pivot"},
            "resistance_2": {"value": round(r2, 2), "raw": r2, "score": 0, "desc": "Fib R2",      "alias": "Resistance 2", "source": "pivot"},
            "resistance_3": {"value": round(r3, 2), "raw": r3, "score": 0, "desc": "Fib R3",      "alias": "Resistance 3", "source": "pivot"},
            "support_1":    {"value": round(s1, 2), "raw": s1, "score": 0, "desc": "Fib S1",      "alias": "Support 1",    "source": "pivot"},
            "support_2":    {"value": round(s2, 2), "raw": s2, "score": 0, "desc": "Fib S2",      "alias": "Support 2",    "source": "pivot"},
            "support_3":    {"value": round(s3, 2), "raw": s3, "score": 0, "desc": "Fib S3",      "alias": "Support 3",    "source": "pivot"},
        }

    return _wrap_calc(_inner, "Pivot Levels (Fib)")


# -----------------------------
# Metric registry with explicit horizon overrides
# horizon: "intraday" | "short_term" | "long_term" | "multibagger" | "default" | "special"
# -----------------------------
INDICATOR_METRIC_MAP: Dict[str, Dict[str, Any]] = {
    # Default / horizon-agnostic metrics
    "rsi": {"func": compute_rsi, "horizon": "default"},
    "mfi": {"func": compute_mfi, "horizon": "default"},
    "rsi_slope": {"func": compute_rsi_slope, "horizon": "default"},
    "price_action": {"func": compute_price_action, "horizon": "default"},
    "short_ma_cross": {"func": compute_short_ma_cross, "horizon": "default"},
    "bb_high": {"func": compute_bollinger_bands, "horizon": "default"},
    "bb_mid": {"func": compute_bollinger_bands, "horizon": "default"},
    "bb_low": {"func": compute_bollinger_bands, "horizon": "default"},
    "bb_width": {"func": compute_bollinger_bands, "horizon": "default"},
    "bb_percent_b": {"func": compute_bb_percent_b, "horizon": "default"},
    "rvol": {"func": compute_rvol, "horizon": "default"},
    "obv_div": {"func": compute_obv_divergence, "horizon": "default"},
    "vpt": {"func": compute_vpt, "horizon": "default"},
    "vol_spike_ratio": {"func": compute_volume_spike, "horizon": "default"},
    "vol_spike_signal": {"func": compute_volume_spike, "horizon": "default"},
    "vwap": {"func": compute_vwap, "horizon": "intraday"},
    "vwap_bias": {"func": compute_vwap, "horizon": "intraday"},
    "atr_14": {"func": compute_atr, "horizon": "default"},
    "atr_pct": {"func": compute_atr, "horizon": "default"},
    "cmf_signal": {"func": compute_cmf, "horizon": "default"},
    "donchian_signal": {"func": compute_donchian, "horizon": "default"},
    "entry_confirm": {"func": compute_entry_price, "horizon": "default"},
    "sl_2x_atr": {"func": compute_atr_sl, "horizon": "default"},
    "nifty_trend_score": {"func": compute_nifty_trend_score, "horizon": "long_term"},
    "gap_percent": {"func": compute_gap_percent, "horizon": "long_term"},

    # Short-term / daily metrics (should be computed on daily / short_term horizon)
    "adx": {"func": compute_adx, "horizon": "short_term"},
    "stoch_k": {"func": compute_stochastic, "horizon": "short_term"},
    "stoch_d": {"func": compute_stochastic, "horizon": "short_term"},
    "stoch_cross": {"func": compute_stochastic, "horizon": "short_term"},
    "cci": {"func": compute_cci, "horizon": "short_term"},
    "supertrend_signal": {"func": compute_supertrend, "horizon": "short_term"},
    "macd": {"func": compute_macd, "horizon": "default"},
    "macd_cross": {"func": compute_macd, "horizon": "default"},
    "macd_hist_z": {"func": compute_macd, "horizon": "default"},
    "macd_histogram": {"func": compute_macd, "horizon": "default"},

    "ma_cross_trend": {"func": compute_ma_crossover, "horizon": "default"},
    "ma_20ema": {"func": compute_ma_crossover, "horizon": "default"},
    "ma_50ema": {"func": compute_ma_crossover, "horizon": "default"},
    "ma_200ema": {"func": compute_ma_crossover, "horizon": "default"},

    # 🆕 NEW: Raw Components for Composites (Trend)
    "ema_20_slope": {"func": compute_ema_slope, "horizon": "default"},
    "ema_50_slope": {"func": compute_ema_slope, "horizon": "default"},

    # 🆕 NEW: Raw Components for Composites (Volatility)
    "true_range": {"func": compute_true_range, "horizon": "default"},
    "true_range_pct": {"func": compute_true_range, "horizon": "default"},
    
    # Run HV on daily data (short_term) by default for accuracy
    "hv_10": {"func": compute_historical_volatility, "horizon": "short_term"}, 
    "hv_20": {"func": compute_historical_volatility, "horizon": "short_term"},

    # Long-term metrics (2y / weekly / monthly frames)
    "price_vs_200dma_pct": {"func": compute_200dma, "horizon": "long_term"},
    "dma_200": {"func": compute_200dma, "horizon": "long_term"},
     "dma_200_slope": {"func": compute_200dma_slope, "horizon": "long_term"},
    "dma_20": {"func": compute_dma_cross, "horizon": "long_term"},
    "dma_50": {"func": compute_dma_cross, "horizon": "long_term"},
    "dma_20_50_cross": {"func": compute_dma_cross, "horizon": "long_term"},
    "rel_strength_nifty": {"func": compute_relative_strength, "horizon": "long_term"},
    "ichi_cloud": {"func": compute_ichimoku, "horizon": "long_term"},
    "ichi_span_a": {"func": compute_ichimoku, "horizon": "long_term"},
    "ichi_span_b": {"func": compute_ichimoku, "horizon": "long_term"},
    "ichi_tenkan": {"func": compute_ichimoku, "horizon": "long_term"},
    "ichi_kijun": {"func": compute_ichimoku, "horizon": "long_term"},
    "psar_trend": {"func": compute_psar, "horizon": "short_term"},
    "psar_level": {"func": compute_psar, "horizon": "short_term"},
    
    # Keltner Squeeze
    "ttm_squeeze": {"func": compute_keltner_squeeze, "horizon": "short_term"},
    "kc_upper": {"func": compute_keltner_squeeze, "horizon": "short_term"},
    "kc_lower": {"func": compute_keltner_squeeze, "horizon": "short_term"},

    # --- Pivot Points Bundle (Horizon: short_term = Daily) ---
    "pivot_point":  {"func": compute_pivot_points, "horizon": "short_term"},
    "resistance_1": {"func": compute_pivot_points, "horizon": "short_term"},
    "resistance_2": {"func": compute_pivot_points, "horizon": "short_term"},
    "resistance_3": {"func": compute_pivot_points, "horizon": "short_term"},
    "support_1":    {"func": compute_pivot_points, "horizon": "short_term"},
    "support_2":    {"func": compute_pivot_points, "horizon": "short_term"},
    "support_3":    {"func": compute_pivot_points, "horizon": "short_term"},
}


# -----------------------------
# ----Main orchestrator
# --- NEW: compute_indicators (Horizon-Aware) ---
#


def compute_indicators(
    symbol: str,
    df_hash: str = None, # df_hash and benchmark_hash are now unused but kept for cache key
    benchmark_hash: str = None,
    horizon: str = "short_term",
    benchmark_symbol: str = "^NSEI"
) -> Dict[str, Dict[str, Any]]:

    logger.info(f"[CACHE] computing indicators for {symbol} (horizon={horizon})")

    # 1. Get the list of metrics required for this *profile*
    profile = HORIZON_PROFILE_MAP.get(horizon, {})
    if not profile:
        logger.warning(f"[{symbol}] Profile '{horizon}' not found in HORIZON_PROFILE_MAP.")
        return {}
        
    # Get all metrics this profile cares about
    profile_metrics = list(profile.get("metrics", {}).keys())
    # Also get all metrics used in penalties
    profile_metrics.extend(list(profile.get("penalties", {}).keys()))
    # Also get all metrics from meta-scores (if they exist)
    profile_metrics.extend(list(MOMENTUM_WEIGHTS.keys()))
    profile_metrics.extend(list(VALUE_WEIGHTS.keys()))
    profile_metrics.extend(list(GROWTH_WEIGHTS.keys()))
    profile_metrics.extend(list(QUALITY_WEIGHTS.keys()))
    
    # Get unique list
    profile_metrics = list(set(profile_metrics + CORE_TECHNICAL_SETUP_METRICS))    
    indicators: Dict[str, Dict[str, Any]] = {}
    
    # 2. This cache is CRITICAL.
    # It holds dataframes (e.g., "intraday", "long_term")
    # for this *single* run, so we don't re-fetch.

    # 3. Add Price from the *profile's* horizon
    try:
        df_profile = get_history_for_horizon(symbol, horizon)
        if not df_profile.empty:
            price = float(df_profile["Close"].iloc[-1])
            indicators["Price"] = {"value": round(price, 2), "score": 0, "alias": "Price", "desc": "Current Price"}
        else:
            price = None
    except Exception:
        price = None
        
    # 4. Loop through required metrics and run them on correct data
    macd_handled = False
    
    for metric_key in profile_metrics:
        if not metric_key:
            continue
            
        meta = INDICATOR_METRIC_MAP.get(metric_key)
        if not meta:
            # This is not an indicator (e.g., a fundamental metric), skip it.
            continue
            
        fn = meta.get("func")

        # --- Handle MACD Bundle (as it's special) ---
        if metric_key in {"macd", "macd_cross", "macd_hist_z", "macd_histogram"}:
            if macd_handled:
                continue # Already ran
            macd_handled = True
            
            # MACD always runs on "short_term" (daily) data
            try:
                df_macd = get_history_for_horizon(symbol, "short_term")
                if not df_macd.empty:
                    # 🚀 FIX 1: Slice MACD data explicitly here
                    df_macd_sliced = _slice_for_speed(df_macd)
                    macd_results = compute_macd(df_macd_sliced)
                    indicators.update(macd_results)
            except Exception as e:
                logger.warning(f"[{symbol}] MACD bundle failed: {e}")
            continue # Move to next metric

        # --- Pivot Bundle Logic ---
        pivot_keys = {
            "pivot_point", "resistance_1", "resistance_2", "resistance_3",
            "support_1", "support_2", "support_3"
        }
        
        if metric_key in pivot_keys:
            # Safety Check: If already calculated, skip
            if indicators.get("_pivot_done"):
                continue

            try:
                # Default to short_term (daily) for standard pivots
                piv_horizon = meta.get("horizon", "short_term")
                df_piv = get_history_for_horizon(symbol, piv_horizon)
                
                if not df_piv.empty:
                    # Note: Pivot function needs previous day, so slicing to 400 is safe
                    piv_results = compute_pivot_points(df_piv)
                    indicators.update(piv_results)
                    indicators["_pivot_done"] = True
            except Exception as e:
                logger.warning(f"[{symbol}] Pivot bundle failed: {e}")

        if fn is None:
            continue # Skip non-callable metrics

        # --- This is the "Smart" Logic ---
        # 5. Get the *data horizon* this metric needs
        data_horizon = meta.get("horizon", "default")
        
        # 'default' means use the profile's main horizon
        if data_horizon == "default":
            data_horizon = horizon
        elif data_horizon == "special":
            # 'special' metrics (like bid_ask) handle their own data
            pass 
            
        # 6. Get the correct dataframe from the cache
        df_local = None
        if data_horizon != "special":
            try:
                df_local = get_history_for_horizon(symbol, data_horizon)
                if df_local is not None:
                    df_local = _slice_for_speed(df_local)
                elif df_local is None or df_local.empty:
                    raise ValueError(f"DF for horizon '{data_horizon}' is None")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to get DF for metric '{metric_key}' (horizon '{data_horizon}'): {e}")
                continue # Skip metric if data is bad

        # 7. Get the benchmark DF (if needed)
        # Note: We'll fetch benchmark data based on the *metric's* horizon
        benchmark_df = None
        sig = inspect.signature(fn)
        if "benchmark_df" in sig.parameters:
            try:
                benchmark_df = get_benchmark_data(data_horizon, benchmark_symbol=benchmark_symbol)
                # 🚀 FIX 3: Slicing logic for Benchmark DF (Applied here)
                if benchmark_df is not None and not benchmark_df.empty:
                    benchmark_df = _slice_for_speed(benchmark_df)
            except Exception as e:
                 logger.warning(f"[{symbol}] Failed to get Benchmark DF for metric '{metric_key}': {e}")

        # 8. Build kwargs and Run
        kwargs = {}
        try:
            # (Set price from earlier, or get it now if it failed)
            if price is None and not df_local.empty:
                 price = float(df_local["Close"].iloc[-1])
            
            # 🆕 FIX: INJECT RSI SERIES DATA FOR DEPENDENT FUNCTION (RSI SLOPE)
            if metric_key == "rsi_slope":
                rsi_entry = indicators.get("rsi")
                # Safely extract the series dictionary stored by compute_rsi
                rsi_series_data = rsi_entry.get("full_series") if rsi_entry else None
                
                if rsi_series_data:
                    # Add the series data to kwargs for the function to consume
                    kwargs["rsi_series"] = rsi_series_data
            
            # Inspect function parameters and build kwargs
            for p in sig.parameters.values():
                name = p.name
                if name == "symbol":
                    kwargs["symbol"] = symbol
                elif name == "df":
                    kwargs["df"] = df_local # <-- Pass the correct local df
                elif name == "benchmark_df":
                    kwargs["benchmark_df"] = benchmark_df
                elif name == "close":
                    kwargs["close"] = df_local["Close"]
                elif name == "high":
                    kwargs["high"] = df_local["High"]
                elif name == "low":
                    kwargs["low"] = df_local["Low"]
                elif name == "volume":
                    kwargs["volume"] = df_local["Volume"]
                elif name == "price":
                    kwargs["price"] = price
                elif name == "horizon":
                    kwargs["horizon"] = data_horizon # Pass the data horizon
            
            # Run the metric function
            result = fn(**kwargs)
            if isinstance(result, dict):
                indicators.update(result)
                
        except Exception as e:
            logger.warning(f"[{symbol}] Metric '{metric_key}' (func {fn.__name__}) failed: {e}")

    # 9. Add technical_score (from the old, simple logic)
    # Your signal_engine doesn't use this, but we'll keep it for compatibility
    try:
        raw_score = compute_technical_score(indicators)
        indicators["technical_score"] = {
            "value": raw_score, 
            "score": 0, 
            "desc": "Aggregate Technical Score",
            "alias": "Technical Score"
        }
        indicators["Horizon"] = {"value": horizon, "score": 0, "desc": "Horizon setting"}
    except Exception as e:
        logger.warning("[%s] Failed computing technical_score: %s", symbol, e)
        indicators["technical_score"] = {
            "value": -1, 
            "score": 0, 
            "desc": "Aggregate Technical Score",
            "alias": "Technical Score"
        }

    return indicators

