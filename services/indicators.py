# services/indicators.py
"""
Phase 3 Refactored (v7 - FINAL GOLD): 
- RESTORED: ema20_50Cross for Short Term.
- LOGIC FIX: 'gapPercent' now correctly scores moderate gaps (0.5% - 2.0%).
- LOGIC FIX: 'ma_trend' awards partial score (7) if Short MA > Mid MA (Developing Trend).
- ACCURACY: Uses WMA/MMA for slopes on Long Term/Multibagger horizons.
"""


from typing import Dict, Any, List, Optional
import numpy as np
from math import atan, degrees
import pandas as pd
import pandas_ta as ta
import inspect

import logging
logger = logging.getLogger(__name__)

# Shared helpers
from config.constants import (
    ADX_HORIZON_CONFIG, ATR_HORIZON_CONFIG, ATR_MULTIPLIERS, CORE_TECHNICAL_SETUP_METRICS, GROWTH_WEIGHTS, MOMENTUM_WEIGHTS, 
    QUALITY_WEIGHTS, STOCH_HORIZON_CONFIG, HORIZON_PROFILE_MAP, 
    VALUE_WEIGHTS
)
from services.data_fetch import (
    _safe_float,
    get_benchmark_data,
    safe_float,
    _wrap_calc,
    get_history_for_horizon
)
from services.analyzers.pattern_analyzer import run_pattern_analysis

PYTHON_LOOKBACK_MAP = {
    "intraday": 650,       # ~1 month (increased for EMA200 warmup)
    "short_term": 800,     # (keeps all 756 + buffer)
    "long_term": 280,      # ~5 years weekly
    "multibagger": 120     # 10 years monthly
}

INDICATOR_MIN_ROWS = {
    "rsi": 14,
    "macd": 35,  # 26 (slow) + 9 (signal)
    "ema200": 200,
    "ema50": 50,
    "ema20": 20,
    "wma50": 50,
    "wma40": 40,
    "wma10": 10,
    "mma12": 12,
    "mma24": 24,
    "mma6": 6,
    "adx": 14,
    "stoch": 14,
    "bbands": 20,
    "obv": 20,
    "atr": 14,
    "ichimoku": 60,
    "cci": 20,
    "mfi": 14,
    "vwap": 20,
    "keltner": 20
}

class _Validator:
    @staticmethod
    def require(df: pd.DataFrame, cols: List[str] = None, 
                min_rows: int = 1, indicator: str = None):
        """
        Validates DataFrame with soft-fail for insufficient data.
        
        Args:
            df: DataFrame to validate
            cols: Required column names
            min_rows: Minimum rows required (can be overridden by indicator)
            indicator: Name of indicator being calculated (for smart min_rows)
        
        Raises:
            ValueError: Only on critical failures (empty df, missing cols, < 10 rows)
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        if cols and not set(cols).issubset(df.columns):
            raise ValueError(f"Missing required columns: {cols}")
        
        # Determine actual minimum required
        required = min_rows
        if indicator and indicator in INDICATOR_MIN_ROWS:
            required = max(required, INDICATOR_MIN_ROWS[indicator])
        
        # Soft fail: Warn but continue with limited data
        if len(df) < required:
            logger.warning(
                f"{indicator or 'Calculation'}: Limited data ({len(df)} rows, "
                f"recommended {required}). Proceeding with available data."
            )
            
            # Hard fail only on critically insufficient data
            if len(df) < 10:
                raise ValueError(
                    f"Critically insufficient data for {indicator or 'calculation'}: "
                    f"{len(df)} < 10 rows"
                )
    
    @staticmethod
    def extract_last(series: Any, name: str = "Indicator") -> float:
        if series is None or (isinstance(series, (pd.Series, pd.DataFrame)) and series.empty):
            raise ValueError(f"{name} data unavailable/empty")
        
        if isinstance(series, pd.DataFrame):
            val = series.iloc[-1].iloc[0]
        else:
            val = series.iloc[-1]
            
        f_val = safe_float(val)
        if f_val is None:
            return None
        return f_val

def _slice_for_speed(df: pd.DataFrame, horizon: str = "short_term") -> pd.DataFrame:
    if df is None or df.empty: return df
    max_rows = PYTHON_LOOKBACK_MAP.get(horizon, 600)
    if len(df) > max_rows:
        return df.tail(max_rows).copy()
    return df.copy()

def _safe_last_vals(series: pd.Series, n: int = 1) -> Optional[List[float]]:
    if series is None or series.empty: return None
    s = series.dropna()
    if len(s) < n: return None
    try:
        vals = [safe_float(x) for x in s.iloc[-n:].tolist()]
        if any(v is None for v in vals): return None
        return vals
    except Exception: return None

# ========================================================
# 📊 METRIC CALCULATORS
# ========================================================

def compute_rsi(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close"],min_rows=length, indicator="rsi")
        rsi_series = ta.rsi(df["Close"], length=length)
        val = _Validator.extract_last(rsi_series, "RSI")
        if val is None: return {}

        if val < 30: score, zone = 10, "Oversold (Buy)"
        elif val > 70: score, zone = 0, "Overbought (Sell)"
        elif 45 <= val <= 65: score, zone = 8, "Healthy Momentum"
        else: score, zone = 5, "Neutral"

        slope_val = 0.0
        try:
            if len(rsi_series) >= 5:
                y = rsi_series.dropna().tail(5).values
                x = np.arange(len(y))
                slope_val = np.polyfit(x, y, 1)[0]
        except: pass

        return {
            "rsi": {"value": round(val, 2), "score": score, "desc": zone},
            "rsiSlope": {"value": round(slope_val, 2), "score": 0, "desc": f"Slope {slope_val:.2f}"}
        }
    return _wrap_calc(_inner, "RSI")

def compute_mfi(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close", "Volume"], min_rows=length, indicator="mfi")
        val = _Validator.extract_last(ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=length), "MFI")
        if val is None: return {}

        if val < 20: score, desc = 10, "Oversold (Buy)"
        elif val > 80: score, desc = 0, "Overbought (Sell)"
        elif 45 <= val <= 65: score, desc = 8, "Healthy Momentum"
        else: score, desc = 5, "Neutral"

        return {"mfi": {"value": round(val, 2), "score": score, "desc": desc}}
    return _wrap_calc(_inner, "MFI")

def compute_cci(df: pd.DataFrame, length: int = 20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=length, indicator="cci")
        val = _Validator.extract_last(ta.cci(df["High"], df["Low"], df["Close"], length=length), "CCI")
        if val is None: return {}

        if val < -100: desc, score = "Oversold (Buy)", 10
        elif val > 100: desc, score = "Overbought (Sell)", 0
        else: desc, score = "Neutral", 5

        return {"cci": {"value": round(val, 2), "score": score, "desc": desc}}
    return _wrap_calc(_inner, "CCI")

def compute_adx(df: pd.DataFrame, horizon: str = "short_term") -> Dict[str, Dict[str, Any]]:
    def _inner():
        length = ADX_HORIZON_CONFIG.get(horizon, 14)
        _Validator.require(df, ["High", "Low", "Close"], min_rows=length, indicator=f"adx_{horizon}")
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=length)
        if adx_df is None or adx_df.empty: raise ValueError("ADX empty")

        adx_col = next((c for c in adx_df.columns if "adx" in c.lower()), None)
        di_p_col = next((c for c in adx_df.columns if "dmp" in c.lower() or "di+" in c.lower()), None)
        di_m_col = next((c for c in adx_df.columns if "dmn" in c.lower() or "di-" in c.lower()), None)

        if not adx_col: raise ValueError("ADX column missing")

        adx_val = _Validator.extract_last(adx_df[adx_col], "ADX")
        if adx_val is None: return {}
        
        diPlus = safe_float(adx_df[di_p_col].iloc[-1]) if di_p_col else None
        diMinus = safe_float(adx_df[di_m_col].iloc[-1]) if di_m_col else None

        if adx_val < 20: score, trend = 0, "Weak / Range-bound"
        elif adx_val <= 25: score, trend = 5, "Developing / Moderate"
        else: score, trend = 10, "Strong Trend"

        return {
            "adx": {"value": round(adx_val, 2), 'length': length,  "raw": adx_val, "score": score, "desc": f"adx -> {adx_val:.2f}"},
            "adx_signal": {"value": trend, "score": score, "desc": f"adx_signal -> {trend}"},
            "diPlus": {"value": round(diPlus, 2) if diPlus else None, "score": 5, "desc": f"diPlus -> {diPlus:.1f}"},
            "diMinus": {"value": round(diMinus, 2) if diMinus else None, "score": 5, "desc": f"diMinus -> {diMinus:.1f}"},
        }
    return _wrap_calc(_inner, "ADX")

def compute_stochastic(df: pd.DataFrame, horizon:str = "short_term") -> Dict[str, Dict[str, Any]]:
    def _inner():
        # 1. Get dynamic settings
        cfg = STOCH_HORIZON_CONFIG.get(horizon, STOCH_HORIZON_CONFIG["short_term"])
        k_len = cfg["k"]
        d_len = cfg["d"]
        s_k   = cfg["smooth"]
        _Validator.require(df, ["High", "Low", "Close"], 
                          min_rows=max(k_len, d_len) + s_k, 
                          indicator=f"stoch_{horizon}")
        
        # 2. Calculate using dynamic settings
        stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=k_len, d=d_len, smooth_k=s_k)
        if stoch_df is None or stoch_df.empty: raise ValueError("Stoch empty")

        stoch_df.columns = [c.lower() for c in stoch_df.columns]
        k_col = next((c for c in stoch_df.columns if "k" in c), None)
        d_col = next((c for c in stoch_df.columns if "d" in c), None)

        k_val = _Validator.extract_last(stoch_df[k_col], f"Stoch %K({k_len})")
        d_val = _Validator.extract_last(stoch_df[d_col], f"Stoch %D({d_len})")
        
        if k_val is None or d_val is None: return {}

        if k_val < 20 and d_val < 20: zone, score = "Oversold (Buy)", 10
        elif k_val > 80 and d_val > 80: zone, score = "Overbought (Sell)", 0
        else: zone, score = "Neutral", 5

        cross_score, cross_status = 5, "Neutral"
        if len(stoch_df) >= 2:
            k_prev = safe_float(stoch_df[k_col].iloc[-2])
            d_prev = safe_float(stoch_df[d_col].iloc[-2])
            if k_prev is not None and d_prev is not None:
                if k_prev <= d_prev and k_val > d_val: cross_status, cross_score = "Bullish", 10
                elif k_prev >= d_prev and k_val < d_val: cross_status, cross_score = "Bearish", 0

        return {
            "stochK": {"value": round(k_val, 2), 'length': k_len, "score": score, "desc": f"Stoch({k_len})"},
            "stochD": {"value": round(d_val, 2), 'length': d_len, "score": score, "desc": f"Stoch({d_len})"},
            "stochCross": {"value": cross_status, "score": cross_score},
        }
    return _wrap_calc(_inner, "Stochastic")

def compute_macd(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close"], min_rows=35)
        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty: raise ValueError("MACD empty")

        macd_df.columns = [c.lower() for c in macd_df.columns]
        macd_col = next((c for c in macd_df.columns if "macd_" in c), None)
        sig_col = next((c for c in macd_df.columns if "macds_" in c or "signal" in c), None)
        hist_col = next((c for c in macd_df.columns if "macdh_" in c or "hist" in c), None)

        if not all([macd_col, sig_col, hist_col]): raise ValueError("Missing MACD cols")

        macd_val = _Validator.extract_last(macd_df[macd_col], "MACD")
        sig_val = _Validator.extract_last(macd_df[sig_col], "Signal")
        hist_val = _Validator.extract_last(macd_df[hist_col], "Hist")

        # We use safe_float on iloc[-2] if it exists
        prev_hist_val = 0.0
        if len(macd_df) >= 2:
            prev_hist_val = safe_float(macd_df[hist_col].iloc[-2]) or 0.0
        
        if macd_val is None or sig_val is None: return {}

        if macd_val > sig_val: cross, score = "Bullish", 10
        elif macd_val < sig_val: cross, score = "Bearish", 0
        else: cross, score = "Neutral", 5

        hist_score, hist_z = 5, 0.0
        hist_series = macd_df[hist_col].dropna()
        if len(hist_series) > 1:
            window = hist_series.tail(min(100, len(hist_series)))
            std = window.std()
            if std > 1e-9:
                hist_z = (hist_val - window.mean()) / std
                if hist_z > 1.0: hist_score = 10
                elif hist_z < -1.0: hist_score = 0

        hist_strength = 10 if hist_val > 0.5 else 7 if hist_val > 0 else 3 if hist_val > -0.5 else 0

        return {
            "macd": {"value": round(macd_val, 2), "score": score, "desc": f"macd -> {macd_val:.2f}"},
            "macdCross": {"value": cross, "score": score, "desc": f"macdCross -> {cross}"},
            "macdHistZ": {"value": round(hist_z, 4), "score": hist_score, "desc": "MACD Hist Z-Score"},
            "macdHistogram": {"value": round(hist_val, 3), "score": hist_strength, "desc": f"Hist {hist_val:.3f}"},
            "prevMacdHistogram": {"value": round(prev_hist_val, 3), "score": 0, "desc": "Prev Hist"}
        }
    return _wrap_calc(_inner, "MACD")

def compute_vwap(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close", "Volume"], min_rows=20, indicator="vwap")
        vwap_val = _Validator.extract_last(ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]), "VWAP")
        if vwap_val is None: return {}
        price = _Validator.extract_last(df["Close"], "Close")

        if price > vwap_val: bias, score = "Bullish", 10
        elif price < vwap_val: bias, score = "Bearish", 0
        else: bias, score = "Neutral", 5

        return {
            "vwap": {"value": round(vwap_val, 2), "score": score, "desc": f"vwap -> {vwap_val:.1f}"}, 
            "vwapBias": {"value": bias, "score": score, "desc": f"vwapBias -> {bias}"}
        }
    return _wrap_calc(_inner, "VWAP")

def compute_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close"], min_rows=length, indicator="bbands")
        bb = ta.bbands(df["Close"], length=length, std=std_dev)
        if bb is None: raise ValueError("BB failed")
        
        cols = {c.lower(): c for c in bb.columns}
        u_col = next((k for k in cols if "u" in k), None)
        l_col = next((k for k in cols if "l" in k), None)
        m_col = next((k for k in cols if "m" in k), None)

        upper = _Validator.extract_last(bb[cols[u_col]], "BB Upper")
        lower = _Validator.extract_last(bb[cols[l_col]], "BB Lower")
        mid = _Validator.extract_last(bb[cols[m_col]], "BB Mid")
        price = _Validator.extract_last(df["Close"], "Close")
        
        if upper is None or lower is None: return {}

        if price < lower: band, score = "Oversold (Buy)", 10
        elif price > upper: band, score = "Overbought (Sell)", 0
        else: band, score = "Neutral", 5

        bb_width_val = ((upper - lower) / mid) * 100 if mid else 0
        width_desc = f"Narrow ({bb_width_val:.2f}%)" if bb_width_val < 5 else f"Wide ({bb_width_val:.2f}%)"
        width_score = 5 if bb_width_val < 5 else 0

        pct_b = 0.5
        if upper != lower:
            pct_b = (price - lower) / (upper - lower)
        
        return {
            "bbHigh": {"value": round(upper, 2), "score": 0, "desc": f"bbHigh -> {upper:.2f}"},
            "bbMid": {"value": round(mid, 2), "score": 0, "desc": f"bbMid -> {mid:.2f}"},
            "bbLow": {"value": round(lower, 2), "score": score, "desc": band},
            "bbWidth": {"value": round(bb_width_val, 2), "raw": bb_width_val, "score": width_score, "desc": width_desc},
            "bbPercentB": {"value": round(pct_b, 3), "score": 10 if pct_b < 0.2 else 0 if pct_b > 0.8 else 5, "desc": f"bbPercentB -> {pct_b:.3f}"}
        }
    return _wrap_calc(_inner, "Bollinger Bands")


def compute_rvol(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Volume"], min_rows=30, indicator="rvol")
        vol = df["Volume"]
        today = safe_float(vol.iloc[-1])
        avg = safe_float(vol.tail(20).mean())
        avg_30days = safe_float(vol.tail(30).mean())
        if not avg: raise ValueError("Zero avg volume")
        if not avg_30days or avg_30days == 0: 
            raise ValueError("Zero average volume indicators compute_rvol")
        
        rvol = today / avg
        score = 10 if rvol > 1.5 else 0 if rvol < 0.8 else 5
        return {
            "rvol": {"value": round(rvol, 2), "score": score, "desc": f"rvol -> {rvol:.2f}"},
            "volume": {"value": round(today, 2), "score": score, "desc": f"volume today -> {today:.2f}"},
            "avg_volume_30Days": {"value": round(avg_30days, 2),"raw": avg_30days,"alias": "Avg Volume (20D)","desc": f"20-period average volume: {int(avg_30days)}"}
        }
    return _wrap_calc(_inner, "RVOL")

def compute_obv_divergence(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close", "Volume"], min_rows=20, indicator="obv")
        obv = ta.obv(df["Close"], df["Volume"])
        if obv is None or len(obv) < 5: raise ValueError("OBV data insufficient")
        
        p_chg = safe_float(df["Close"].iloc[-1]) - safe_float(df["Close"].iloc[-5])
        o_chg = safe_float(obv.iloc[-1]) - safe_float(obv.iloc[-5])
        
        if p_chg * o_chg > 0: sig, score = "Confirming", 10
        elif p_chg * o_chg < 0: sig, score = "Diverging", 0
        else: sig, score = "Neutral", 5
        
        return {"obvDiv": {"value": sig, "score": score, "desc": f"obvDiv -> {sig}"}}
    return _wrap_calc(_inner, "OBV Divergence")

def compute_vpt(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close", "Volume"], min_rows=20, indicator="vpt")
        vpt = ta.pvt(df["Close"], df["Volume"])
        curr = _Validator.extract_last(vpt, "VPT")
        prev = safe_float(vpt.iloc[-5]) if len(vpt) >= 5 else curr
        
        if curr > prev: sig, score = "Accumulation", 10
        elif curr < prev: sig, score = "Distribution", 0
        else: sig, score = "Neutral", 5
        
        return {"vpt": {"value": round(curr, 2), "score": score, "desc": sig}}
    return _wrap_calc(_inner, "VPT")

def compute_volume_spike(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Volume"],  min_rows=20, indicator="volume_spike")
        today = safe_float(df["Volume"].iloc[-1])
        avg = safe_float(df["Volume"].tail(20).mean())
        if not avg: raise ValueError("Zero avg volume")
        
        ratio = today / avg
        if ratio > 1.5: sig, score = "Strong Spike", 10
        elif ratio > 1.2: sig, score = "Moderate Spike", 5
        else: sig, score = "Normal", 5
        
        return {
            "volSpikeRatio": {"value": round(ratio, 2), "score": score, "desc": f"volSpikeRatio -> {ratio:.2f}"},
            "volSpikeSignal": {"value": sig, "score": score, "desc": f"volSpikeSignal -> {sig}"}
        }
    return _wrap_calc(_inner, "Volume Spike")
# // not needed now
def compute_atr(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"])
        atr = _Validator.extract_last(ta.atr(df["High"], df["Low"], df["Close"], length=14), "ATR")
        if atr is None: return {}
        price = _Validator.extract_last(df["Close"], "Close")
        
        pct = (atr / price) * 100
        if 1.0 <= pct <= 3.0: score = 10
        elif pct < 1.0: score = 7
        elif pct <= 5.0: score = 5
        else: score = 0
        
        return {
            "atr14": {"value": round(atr, 2), "score": score, "desc": f"atr14 -> {atr:.2f}"},
            "atrPct": {"value": round(pct, 2), "score": score, "desc": f"{pct:.2f}%"}
        }
    return _wrap_calc(_inner, "ATR")

# use dynamic one
def compute_dynamic_atr(df: pd.DataFrame, horizon: str = "short_term") -> Dict[str, Dict[str, Any]]:
    def _inner():
        length = ATR_HORIZON_CONFIG.get(horizon, 14)

        _Validator.require(df, ["High", "Low", "Close"],min_rows=length, indicator=f"atr_{horizon}")

        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=length)
        atr_val = _Validator.extract_last(atr_series, f"ATR_{length}")

        sma_series = ta.sma(df["Close"], length=length)
        sma_val = _Validator.extract_last(sma_series, f"SMA_{length}")

        if atr_val is None: return {}

        price = _Validator.extract_last(df["Close"], "Close")
        
        # Guard against zero price
        if not price: return {}
        
        pct = (atr_val / price) * 100

        # Volatility scoring
        if 1.0 <= pct <= 3.0: score = 10
        elif pct < 1.0: score = 7
        elif pct <= 5.0: score = 5
        else: score = 0

        atrSmaRatio = 0.0
        if sma_val and sma_val > 0:
            atrSmaRatio = atr_val / sma_val

        return {
            "atrDynamic": {
                "value": round(atr_val, 2),
                "length": length,               # Metadata for UI/Debug
                "score": score,
                "alias": f"ATR ({length})",     # Dynamic Label
                "desc": f"ATR({length}) = {atr_val:.2f}",
                "source": "technical"
            },
            "atrPct": {
                "value": round(pct, 2),
                "score": score,
                "desc": f"Volatility {pct:.2f}%",
                "source": "technical"
            },
            "atrSmaRatio": {
                "value": round(atrSmaRatio, 4),
                "score": 0, # Usually just a filter, not a scorer
                "desc": f"ATR/SMA Ratio: {atrSmaRatio:.4f}"
            }
        }
    return _wrap_calc(_inner, "Dynamic ATR")

def compute_dynamic_sl(df: pd.DataFrame, price: float, horizon: str = "short_term"):
    def _inner():
        length = ATR_HORIZON_CONFIG.get(horizon, 14)
        _Validator.require(df, ["High", "Low", "Close"], 
                          min_rows=length, indicator=f"sl_atr_{horizon}")
        mult_cfg = ATR_MULTIPLIERS.get(horizon, {"sl": 2.0})
        # Handle case where mult_cfg is a float or dict (safety check)
        multiplier = mult_cfg.get("sl", 2.0) if isinstance(mult_cfg, dict) else 2.0

        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=length)
        atr_val = safe_float(atr_series.iloc[-1])

        if atr_val is None or price is None: return {}

        sl_price = price - (atr_val * multiplier)
        
        # Avoid div by zero
        if not price: return {}
        
        risk_pct = ((price - sl_price) / price) * 100

        return {
            "slAtrDynamic": {
                "value": round(sl_price, 2),
                "alias": f"SL ({multiplier}x ATR{length})", # Clear Label
                "desc": f"Stop Loss using ATR({length})",
                "source": "technical"
            },
            "riskPerSharePct": {
                "value": round(risk_pct, 2),
                "desc": f"Risk {risk_pct:.1f}%",
                "source": "technical"
            }
        }
    return _wrap_calc(_inner, "Dynamic ATR Stop Loss")

def compute_supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        # 1. Keep your Validators
        _Validator.require(df, ["High", "Low", "Close"],  min_rows=length, indicator="supertrend")
        try:
            st = ta.supertrend(df["High"], df["Low"], df["Close"], length=length, multiplier=multiplier)
        except Exception as e:
            raise ValueError(f"SuperTrend lib error: {e}")
        if st is None or st.empty: raise ValueError("SuperTrend result empty")
        cols = st.columns
        # 2. Identify the Direction Column (d_col) vs The Value Column (l_col)
        # pandas-ta usually names direction col like "SUPERTd_7_3.0" and value like "SUPERT_7_3.0"
        d_col = next((c for c in cols if any(x in c.lower() for x in ("d_", "dir", "trend"))), None)
        l_col = next((c for c in cols if "super" in c.lower() and c != d_col), None)
        close = _Validator.extract_last(df["Close"], "Close")
        # 3. Robust Extraction of Direction (trend_val)
        trend_val = None
        prev_trend_val = None

        if d_col:
            trend_val = _safe_float(st[d_col].iloc[-1])
            if len(st) >= 2:
                prev_trend_val = _safe_float(st[d_col].iloc[-2])

        # 4. Robust Extraction of Level (st_level_val)
        st_level_val = None
        if l_col:
            st_level_val = _safe_float(st[l_col].iloc[-1])
            
        # Fallback Logic If direction missing, infer from level
        if trend_val is None and st_level_val is not None:
             trend_val = 1.0 if close > st_level_val else -1.0

        if trend_val == 1.0: 
            sig, score = "Bullish", 10
        else: 
            sig, score = "Bearish", 0
        
        return {
            "supertrendSignal": { "value": sig, "score": score, "desc": f"ST ({length},{multiplier}) {sig}"},
            "supertrendValue": {"value": st_level_val, "score": 0, "desc": f"Level {st_level_val}","alias": "SuperTrend Value"},
            "prevSupertrend": {"value": prev_trend_val if prev_trend_val is not None else 0, "score": 0}
        }
    return _wrap_calc(_inner, "SuperTrend")

def compute_price_action(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=2, indicator="priceAction")
        high = df["High"].iloc[-1]
        # ✅ W57 FIX: Corrected logic flip (Support must be min of Lows, Resistance must be max of Highs)
        support1 = df["Low"].rolling(window=20).min()
        resistance1 = df["High"].rolling(window=20).max()
        low = df["Low"].iloc[-1]
        close = df["Close"].iloc[-1]
        
        if high == low: 
            logger.debug("Zero range candle — skipping priceAction")
            return {}
        pos = (close - low) / (high - low)
        
        if pos >= 0.75: sig, score = "Strong Bullish Close", 10
        elif pos >= 0.5: sig, score = "Moderate Close", 5
        else: sig, score = "Weak Close", 0

        return {"priceAction": {"value": round(pos*100, 1), "score": score, "signal": sig, "desc": f"priceAction -> {round(pos*100, 1)}"}}
    return _wrap_calc(_inner, "Price Action")

def compute_price_vs_base_ma(df: pd.DataFrame, close, price, horizon: str = "short_term"):
    """Returns priceVsPrimaryTrendPct (Renamed from compute_200dma). 
    Features dynamic lookback fallback for newly listed stocks."""
    lens = {"intraday": 200, "short_term": 200, "long_term": 50, "multibagger": 24}
    types = {"intraday": "EMA", "short_term": "EMA", "long_term": "WMA", "multibagger": "MMA"}
    L_original = lens.get(horizon, 200)
    T = types.get(horizon, "EMA")
    
    def _inner():
        # Fallback logic for newly listed stocks (e.g. IPOs)
        L = L_original
        if len(df) < L:
            if horizon in ["intraday", "short_term"] and len(df) >= 50: L = 50
            elif horizon == "long_term" and len(df) >= 20: L = 20
            elif horizon == "multibagger" and len(df) >= 12: L = 12
            else: return {} # Still insufficient history, gracefully skip metric
            
        _Validator.require(df, ["Close"], min_rows=L, indicator=f"price_vs_ma_{horizon}")
        
        # Use the correct MA function matching the horizon type
        if T == "EMA":
            ma_series = ta.ema(close, length=L)
        elif T == "WMA":
            ma_series = ta.wma(close, length=L)
        else:
            ma_series = ta.sma(close, length=L)  # MMA fallback — pandas-ta has no native MMA
            
        try: sma = _Validator.extract_last(ma_series, f"{L}{T}")
        except: sma = None
        
        if not sma: return {}
        diff = ((price - sma) / sma) * 100
        score = 10 if diff > 0 else 5 if abs(diff) < 3 else 0
        desc = "Above" if diff > 0 else "Below"
        
        return {
            "priceVsPrimaryTrendPct": {"value": round(diff, 2), "score": score, "desc": f"{desc} {T}{L} ({diff:.1f}%)", "alias": f"Price vs {T}{L}"},
            f"priceVs{L_original}{T.lower()}Pct": {"value": round(diff, 2), "score": score} # Legacy
        }
    return _wrap_calc(_inner, "Price vs Trend")

def compute_ema_slope(df: pd.DataFrame, horizon: str = "short_term"):
    """
    Calculates Normalized Slopes. 
    Fixes: Scaling issue (uses %), Configurable Lookback, Zero Division Guard.
    """
    def _inner():
        # 1. Horizon Configs
        # ✅ l_s/l_l aligned to compute_dynamic_ma_trend so slope is measured on the
        # same MAs displayed as maFast/maMid — no more measuring WMA(20) slope
        # while showing WMA(10) as maFast.
        cfg = {
            "intraday":    {"type": "EMA", "l_s": 20, "l_l": 50},
            "short_term":  {"type": "EMA", "l_s": 20, "l_l": 50},
            "long_term":   {"type": "WMA", "l_s": 10, "l_l": 40},  # matches WMA(10/40/50) trend
            "multibagger": {"type": "MMA", "l_s": 6,  "l_l": 24}   # matches MMA(6/12/24) trend
        }.get(horizon, {"type": "EMA", "l_s": 20, "l_l": 50})
        
        # 2. Dynamic Lookback
        max_length = max(cfg["l_s"], cfg["l_l"])
        lookback = {"intraday": 5, "short_term": 10, "long_term": 10, "multibagger": 10}.get(horizon, 10)
        
        _Validator.require(df, ["Close"], min_rows=max_length + lookback, indicator=f"ma_slope_{horizon}")
        
        fn = ta.ema if cfg["type"] == "EMA" else ta.wma if cfg["type"] == "WMA" else ta.sma 
        prefix = "ema" if cfg["type"] == "EMA" else "wma" if cfg["type"] == "WMA" else "mma"
        close = df["Close"]
        
        def _get_normalized_angle(series, period):
            if len(series) < period: return 0.0
            y_raw = series.tail(period).values
            if np.isnan(y_raw).any(): return 0.0
            
            # ✅ FIX #2: Zero Division Guard
            if y_raw[0] == 0: return 0.0
            
            # ✅ NORMALIZE: Convert to % index (Starts at 100)
            y_norm = (y_raw / y_raw[0]) * 100.0
            
            x = np.arange(period)
            slope, _ = np.polyfit(x, y_norm, 1)
            return _safe_float(np.degrees(np.arctan(slope)))

        s_ma = fn(close, length=cfg["l_s"])
        l_ma = fn(close, length=cfg["l_l"])
        
        ang_s = _get_normalized_angle(s_ma, lookback)
        ang_l = _get_normalized_angle(l_ma, lookback)
        
        # ✅ FIX #1: Realistic Thresholds for Normalized Data
        # 20° ~= 3.6% move over 10 bars (Strong Trend)
        # 5°  ~= 0.9% move over 10 bars (Moderate Trend)
        score = 10 if ang_s > 20 else 7 if ang_s > 5 else 0
        
        return {
            "maFastSlope": {
                "value": round(ang_s, 2), "raw": ang_s, "score": score, 
                "desc": f"{ang_s:.1f}°", "alias": f"{prefix.upper()} Slope"
            },
            "maSlowSlope": {
                "value": round(ang_l, 2), "raw": ang_l, "score": 0, 
                "desc": f"{ang_l:.1f}°", "alias": f"{prefix.upper()} Slow Slope"
            },
            # Legacy keys for safety
            f"{prefix}{cfg['l_s']}Slope": {"value": round(ang_s, 2), "score": score},
            f"{prefix}{cfg['l_l']}Slope": {"value": round(ang_l, 2), "score": 0}
        }
    return _wrap_calc(_inner, "MA Slopes")

def compute_dynamic_ma_cross(df: pd.DataFrame, close: pd.Series, horizon: str = "short_term"):
    """Returns maCrossSignal"""
    cfg = {
        "intraday":    {"l": (20, 50), "t": "EMA", "p": "ema"},
        "short_term":  {"l": (20, 50), "t": "EMA", "p": "ema"},
        "long_term":   {"l": (10, 40), "t": "WMA", "p": "wma"},
        "multibagger": {"l": (6, 12),  "t": "SMA", "p": "mma"}
    }.get(horizon, {"l": (20, 50), "t": "EMA", "p": "ema"})
    
    s_len, l_len = cfg["l"]
    fn = ta.ema if cfg["t"] == "EMA" else ta.wma if cfg["t"] == "WMA" else ta.sma
    prefix = cfg["p"]

    def _inner():
        # Use s_len as minimum viable threshold — _safe_last_vals handles partial data gracefully
        _Validator.require(df, ["Close"], 
                          min_rows=s_len + 2,  # +2 for prev values; l_len is handled gracefully below
                          indicator=f"ma_cross_{horizon}")
        s_ma = fn(close, length=s_len)
        l_ma = fn(close, length=l_len)
        vals_s = _safe_last_vals(s_ma, 2)
        vals_l = _safe_last_vals(l_ma, 2)
        
        # For IPO stocks: if slow MA unavailable but fast MA is, return neutral cross
        if not vals_s:
            raise ValueError("Insufficient data for fast MA")
        if not vals_l:
            # Can't compute a cross without both MAs — emit neutral signal
            s_curr = vals_s[1]
            return {
                "maCrossSignal": {"value": 0, "score": 5, "desc": "Neutral (Limited Data)", "alias": f"{prefix.upper()} Cross"},
                f"{prefix}{s_len}{l_len}cross": {"value": 0, "score": 5}  # Legacy
            }
        
        s_curr, l_curr = vals_s[1], vals_l[1]
        s_prev, l_prev = vals_s[0], vals_l[0]
        
        val, score, desc = -1, 0, "Bearish"
        if s_curr > l_curr:
            val, desc = 1, "Bullish"
            score = 10 if (s_prev <= l_prev) else 7 
        elif (s_prev >= l_prev) and (s_curr < l_curr):
            val, desc = -1, "Bearish Cross"
            
        return {
            "maCrossSignal": {"value": val, "score": score, "desc": desc, "alias": f"{prefix.upper()} Cross"},
            f"{prefix}{s_len}{l_len}cross": {"value": val, "score": score} # Legacy
        }
    return _wrap_calc(_inner, "MA Cross")

def compute_dynamic_ma_trend(df: pd.DataFrame, horizon: str = "short_term"):
    """
    Calculates Trend Alignment using Horizon-Specific Physics.
    Returns: maFast, maMid, maSlow, maTrendSignal
    """
    cfg = {
        "intraday":    {"l": (20, 50, 200), "t": "EMA", "p": "ema"},
        "short_term":  {"l": (20, 50, 200), "t": "EMA", "p": "ema"},
        "long_term":   {"l": (10, 40, 50),  "t": "WMA", "p": "wma"},
        "multibagger": {"l": (6, 12, 24),   "t": "SMA", "p": "mma"}
    }.get(horizon, {"l": (20, 50, 200), "t": "EMA", "p": "ema"})
    
    l_s, l_m, l_l = cfg["l"]
    fn = ta.ema if cfg["t"] == "EMA" else ta.wma if cfg["t"] == "WMA" else ta.sma
    pre = cfg["p"]

    def _inner():
        # Use l_s as minimum viable threshold — per-MA try/except handles partial data gracefully
        _Validator.require(df, ["Close"], min_rows=l_s, indicator=f"ma_trend_{horizon}")
        
        try: s_val = _Validator.extract_last(fn(df["Close"], length=l_s), "MA Fast")
        except: s_val = None
        
        try: m_val = _Validator.extract_last(fn(df["Close"], length=l_m), "MA Mid")
        except: m_val = None
        
        try: l_val = _Validator.extract_last(fn(df["Close"], length=l_l), "MA Slow")
        except: l_val = None
        
        val, score, desc = 0, 5, "Neutral"
        
        if s_val and m_val:
            # We have at least fast and mid MAs
            if l_val:
                if s_val > m_val > l_val: val, score, desc = 1, 10, "Strong Uptrend"
                elif s_val < m_val < l_val: val, score, desc = -1, 0, "Strong Downtrend"
                elif s_val > m_val: val, score, desc = 0.5, 7, "Developing Uptrend"
                elif s_val < m_val: val, score, desc = -0.5, 3, "Developing Downtrend"
            else:
                # No slow MA (e.g., IPO stock), evaluate on what we have
                if s_val > m_val: val, score, desc = 0.5, 7, "Developing Uptrend (Limited Data)"
                elif s_val < m_val: val, score, desc = -0.5, 3, "Developing Downtrend (Limited Data)"
            
        return {
            # ✅ STANDARDIZED KEYS
            "maFast": {"value": round(s_val, 2) if s_val is not None else None, "score": 0, "desc": f"{pre.upper()}({l_s})", "length": l_s, "type": cfg["t"], "alias": f"{pre.upper()} Fast"},
            "maMid":  {"value": round(m_val, 2) if m_val is not None else None, "score": 0, "desc": f"{pre.upper()}({l_m})", "length": l_m, "type": cfg["t"], "alias": f"{pre.upper()} Mid"},
            "maSlow": {"value": round(l_val, 2) if l_val is not None else None, "score": 0, "desc": f"{pre.upper()}({l_l})" if l_val else "N/A", "length": l_l, "type": cfg["t"], "alias": f"{pre.upper()} Slow"},
            "maTrendSignal": {"value": val, "score": score, "desc": desc, "alias": "MA Trend Alignment"},
            
            # Legacy
            f"{pre}{l_s}": {"value": round(s_val, 2) if s_val is not None else None, "score": 0},
            f"{pre}{l_m}": {"value": round(m_val, 2) if m_val is not None else None, "score": 0},
            f"{pre}{l_l}": {"value": round(l_val, 2) if l_val is not None else None, "score": 0},
            f"{pre}{l_s}{l_m}{l_l}Trend": {"value": val, "score": score}
        }
    return _wrap_calc(_inner, f"{pre.upper()} Trend")

def compute_pivot_points(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=2)
        prev = df.iloc[-2]
        h, l, c = prev["High"], prev["Low"], prev["Close"]
        pp = (h + l + c) / 3
        rng = h - l
        
        r1, s1 = pp + 0.382 * rng, pp - 0.382 * rng
        r2, s2 = pp + 0.618 * rng, pp - 0.618 * rng
        r3, s3 = pp + 1.000 * rng, pp - 1.000 * rng
        
        return {
            "pivotPoint":  {"value": round(pp, 2), "score": 0},
            "resistance1": {"value": round(r1, 2), "score": 0},
            "resistance2": {"value": round(r2, 2), "score": 0},
            "resistance3": {"value": round(r3, 2), "score": 0},
            "support1":    {"value": round(s1, 2), "score": 0},
            "support2":    {"value": round(s2, 2), "score": 0},
            "support3":    {"value": round(s3, 2), "score": 0},
        }
    return _wrap_calc(_inner, "Pivot Levels")

def compute_keltner_squeeze(df: pd.DataFrame, bb_len=20, kc_len=20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=max(bb_len, kc_len), indicator="keltner")
        # BB
        bb = ta.bbands(df["Close"], length=bb_len, std=2.0)
        u_col = next(c for c in bb.columns if "u" in c.lower())
        l_col = next(c for c in bb.columns if "l" in c.lower())
        bb_u = _Validator.extract_last(bb[u_col])
        bb_l = _Validator.extract_last(bb[l_col])
        
        # RESTORED: SMA + TR based Keltner Channels (Legacy Match)
        mid = ta.sma(df["Close"], length=kc_len)
        tr = ta.true_range(df["High"], df["Low"], df["Close"])
        atr_sma = ta.sma(tr, length=kc_len)
        
        if mid is None or atr_sma is None: return {}
        
        kc_u_val = mid.iloc[-1] + (1.5 * atr_sma.iloc[-1])
        kc_l_val = mid.iloc[-1] - (1.5 * atr_sma.iloc[-1])
        
        if bb_u is None or kc_u_val is None: return {}

        sqz = (bb_u < kc_u_val) and (bb_l > kc_l_val)
        return {
            "ttmSqueeze": {"value": "Squeeze On" if sqz else "Off", "score": 10 if sqz else 5, "desc": f"ttmSqueeze -> {'On' if sqz else 'Off'}"},
            "kcUpper": {"value": round(kc_u_val, 2), "score": 0, "desc": f"kcUpper -> {kc_u_val:.2f}"},
            "kcLower": {"value": round(kc_l_val, 2), "score": 0, "desc": f"kcLower -> {kc_l_val:.2f}"}
        }
    return _wrap_calc(_inner, "TTM Squeeze")

def compute_ichimoku(symbol: str, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=60)
        # pandas-ta returns a tuple: (df_lines, df_span)
        # We generally access [0] for the lines
        ichi_result = ta.ichimoku(df["High"], df["Low"], df["Close"])
        
        if ichi_result is None: raise ValueError("Ichimoku calc failed")
        ichi = ichi_result[0]
        
        # Dynamic column finding (pandas-ta column names change with params)
        cols = {c: c for c in ichi.columns}
        def _get(k): 
            col_name = next((c for c in cols if k in c), None)
            return _Validator.extract_last(ichi[col_name]) if col_name else None
        
        tk = _get("TS") # Tenkan-sen (Conversion)
        kj = _get("KS") # Kijun-sen (Base)
        sa = _get("SA") # Span A
        sb = _get("SB") # Span B
        
        px = _Validator.extract_last(df["Close"])
        
        if tk is None or kj is None or sa is None or sb is None: return {}

        # Signal Logic
        bull_tk = tk > kj
        above_cloud = px > max(sa, sb)
        below_cloud = px < min(sa, sb)
        inside_cloud = not above_cloud and not below_cloud
        
        sig = "Neutral"
        score = 5
        
        if above_cloud and bull_tk: 
            sig, score = "Strong Bullish", 10
        elif below_cloud and not bull_tk: 
            sig, score = "Strong Bearish", 0
        elif above_cloud: 
            sig, score = "Mild Bullish", 8
        elif bull_tk and inside_cloud: 
            sig, score = "Neutral Bullish", 6
        elif not bull_tk and inside_cloud: 
            sig, score = "Neutral Bearish", 4
        elif below_cloud: 
            sig, score = "Mild Bearish", 3
        
        return {
            "ichiCloud": {"value": sig, "score": score, "desc": f"Cloud: {sig}"},
            "ichiSpanA": {"value": round(sa, 2), "score": 0},
            "ichiSpanB": {"value": round(sb, 2), "score": 0},
            "ichiTenkan": {"value": round(tk, 2), "score": 0, "desc": "Conversion Line"},
            "ichiKijun":  {"value": round(kj, 2), "score": 0, "desc": "Base Line"},
            "ichiChikou": {"value": round(px, 2), "score": 0, "desc": "Lagging Span"},
            "ichiChikouRef": {"value": round(df["Close"].iloc[-26], 2), "score": 0, "desc": "Ref Price"},
        }
    return _wrap_calc(_inner, "Ichimoku")

def compute_nifty_trend_score(benchmark_df: pd.DataFrame, horizon: str = "short_term"):
    def _inner():
        ema_slow_len = {"intraday": 50, "short_term": 200, "long_term": 50, "multibagger": 12}.get(horizon, 200)
        ema_fast_len = {"intraday": 20, "short_term": 50,  "long_term": 20, "multibagger": 6}.get(horizon, 50)
        
        if benchmark_df is None or benchmark_df.empty: return {"niftyTrendScore": {"value": None, "score": None}}
        _Validator.require(benchmark_df, ["Close"], min_rows=ema_slow_len, indicator="nifty_trend")
        close = benchmark_df["Close"].dropna()
        if len(close) < ema_fast_len: return {"niftyTrendScore": {"value": None, "score": None}}
        
        cur = close.iloc[-1]
        ema_fast_series = ta.ema(close, ema_fast_len)
        ema_slow_series = ta.ema(close, ema_slow_len)
        
        if ema_fast_series is None:
            logger.warning(f"NIFTY Trend: ema{ema_fast_len} is None (insufficient data)")
            return {"niftyTrendScore": {"value": None, "score": None}}
            
        ema_fast = safe_float(ema_fast_series.iloc[-1])
        ema_slow = safe_float(ema_slow_series.iloc[-1]) if ema_slow_series is not None else None
        
        if ema_slow:
            diff = (cur - ema_slow) / ema_slow * 100
            if ema_fast and cur > ema_fast > ema_slow: sig, score = "Strong Uptrend", 9
            elif cur > ema_slow: sig, score = "Moderate Uptrend", 7
            else: sig, score = "Downtrend", 2
        elif ema_fast:
            diff = (cur - ema_fast) / ema_fast * 100
            if cur > ema_fast: sig, score = "Uptrend (Weak)", 7
            else: sig, score = "Downtrend", 3
        else:
            return {"niftyTrendScore": {"value": None, "score": None}}
            
        return {"niftyTrendScore": {"value": round(diff, 2), "score": score, "desc": sig}}
    return _wrap_calc(_inner, "NIFTY Trend")

def compute_relative_strength(symbol, df, benchmark_df, horizon="short_term"):
    """
    Calculates Alpha vs Nifty.
    Horizon-Aware Lookbacks:- Intraday: 20 candles (5 hours) - Short Term: 20 days (1 month) - Long Term/Multibagger: 52 bars (1 year)
    """
    def _inner():
        lookback = {"intraday": 20, "short_term": 20, "long_term": 52,
                   "multibagger": 52}.get(horizon, 20)
        
        _Validator.require(df, ["Close"], min_rows=lookback, 
                          indicator=f"rel_strength_{horizon}")
        _Validator.require(benchmark_df, ["Close"], min_rows=lookback, 
                          indicator=f"benchmark_{horizon}")
        
        # 1. Map lookback to horizon
        lookback = {"intraday": 20, "short_term": 20, "long_term": 52,"multibagger": 52}.get(horizon, 20)
        
        if len(df) < lookback or len(benchmark_df) < lookback:
            logger.debug(f"Not enough data for Relative Strength: {len(df)} vs {lookback} required")
            return {}

        s_now, s_old = df["Close"].iloc[-1], df["Close"].iloc[-lookback]
        b_now, b_old = benchmark_df["Close"].iloc[-1], benchmark_df["Close"].iloc[-lookback]
        
        # 2. Calculate percentage returns
        s_ret = _safe_float((s_now / s_old - 1) * 100)
        b_ret = _safe_float((b_now / b_old - 1) * 100)
        rs = s_ret - b_ret # Alpha
        
        return {
            "relStrengthNifty": {"raw": round(rs, 2),"value": round(rs, 2),"alias": "Relative Strength (Alpha)","desc": f"{horizon.title()} Alpha: {rs:.1f}%"}}
    return _wrap_calc(_inner, "RS vs Nifty")

def compute_entry_price(df, price):
    def _inner():
        mid = ta.sma(df["Close"], 20).iloc[-1]
        confirm = mid * 1.005
        return {"entryConfirm": {"value": round(confirm, 2), "score": 10 if price > confirm else 5}}
    return _wrap_calc(_inner, "Entry")

def compute_gap_percent(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Open", "Close"], min_rows=2)
        prev_c = df["Close"].iloc[-2]
        curr_o = df["Open"].iloc[-1]
        gap = ((curr_o - prev_c) / prev_c) * 100
        
        # LOGIC FIX: Match Legacy Scoring (Moderate Gap = Score 7)
        if gap > 2.0: score, desc = 10, "Strong Gap Up"
        elif gap > 0.5: score, desc = 7, "Moderate Gap Up"
        elif gap < -2.0: score, desc = 0, "Strong Gap Down"
        elif gap < -0.5: score, desc = 3, "Moderate Gap Down"
        else: score, desc = 5, "No Significant Gap"

        return {"gapPercent": {"value": round(gap, 2), "score": score, "desc": f"gapPercent -> {gap:.2f}"}}
    return _wrap_calc(_inner, "Gap")

def compute_psar(df: pd.DataFrame):
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=20, indicator="psar")
        psar = ta.psar(df["High"], df["Low"], df["Close"])
        r_col = next((c for c in psar.columns if "r" in c.lower()), None)
        val_col = next((c for c in psar.columns if c != r_col), None)
        
        trend = psar[r_col].iloc[-1] 
        level = psar[val_col].iloc[-1]
        
        sig, score = ("Bullish", 10) if trend > 0 else ("Bearish", 0)
        
        if pd.isna(level): level = df["Close"].iloc[-1] 
        
        return {
            "psarTrend": {"value": sig, "score": score, "desc": f"psarTrend -> {sig}"},
            "psarLevel": {"value": round(level, 2), "score": 0, "desc": f"psarLevel -> {level:.2f}"}
        }
    return _wrap_calc(_inner, "PSAR")

def compute_true_range(df):
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], 
                          min_rows=2, indicator="true_range")
        tr = ta.true_range(df["High"], df["Low"], df["Close"]).iloc[-1]
        pct = (tr / df["Close"].iloc[-1]) * 100
        return {"trueRange": {"value": round(tr, 2), "score": 0}, "trueRangePct": {"value": round(pct, 2), "score": 5}}
    return _wrap_calc(_inner, "TR")

def compute_historical_volatility(df, periods=(10, 20)):
    def _inner():
        max_period = max(periods)
        _Validator.require(df, ["Close"], min_rows=max_period, 
                          indicator="historical_vol")
        ret = np.log(df["Close"] / df["Close"].shift(1))
        res = {}
        for p in periods:
            hv = ret.tail(p).std() * np.sqrt(252) * 100
            res[f"hv{p}"] = {"value": round(hv, 2), "score": 10 if hv < 20 else 5}
        return res
    return _wrap_calc(_inner, "HV")

def compute_short_ma_cross(df):
    def _inner():
        _Validator.require(df, ["Close"], min_rows=20, indicator="short_ma_cross")
        e5 = ta.ema(df["Close"], 5).iloc[-1]
        e20 = ta.ema(df["Close"], 20).iloc[-1]
        return {"shortMaCross": {"value": "Bull" if e5 > e20 else "Bear", "score": 10 if e5 > e20 else 0}}
    return _wrap_calc(_inner, "Short MA Cross")

def compute_vol_trend(df):
    def _inner():
        _Validator.require(df, ["Volume"], min_rows=50, indicator="vol_trend")
        v = df["Volume"]
        trend = "Rising" if v.iloc[-1] > v.tail(50).mean() * 1.2 else "Neutral"
        return {"volTrend": {"value": trend, "score": 10 if trend == "Rising" else 5}}
    return _wrap_calc(_inner, "Vol Trend")

def compute_reg_slope(df):
    def _inner():
        _Validator.require(df, ["Close"], min_rows=20, indicator="regSlope")
        y = df["Close"].tail(20).values
        x = np.arange(len(y))
        slope = degrees(atan(np.polyfit(x, y, 1)[0]))
        return {"regSlope": {"value": round(slope, 2), "score": 10 if slope > 2 else 0}}
    return _wrap_calc(_inner, "Reg Slope")

def compute_wick_rejection(df: pd.DataFrame, horizon: str = "short_term") -> Dict[str, Dict[str, Any]]:
    """
    Calculates the 'Upper Wick Ratio' to detect Bull Traps.
    Ratio = Upper Wick / Candle Body.
    > 2.0 implies the market rejected higher prices (Shooting Star-like).
    """
    def _inner():
        _Validator.require(df, ["Open", "High", "Close", "Low"], min_rows=2, indicator="wick_rejection")
        
        open_p = df["Open"].iloc[-1]
        close_p = df["Close"].iloc[-1]
        high_p = df["High"].iloc[-1]
        
        # Calculate dimensions
        body_size = abs(close_p - open_p)
        upper_wick = high_p - max(open_p, close_p)
        
        # Avoid division by zero for Dojis (tiny bodies)
        # If body is tiny, we use a small epsilon or relative to price
        denominator = max(body_size, close_p * 0.001) 
        
        ratio = upper_wick / denominator
        
        # Scoring: High Ratio = Bad (0), Low Ratio = Good (10) for breakouts
        if ratio > 3.0: sig, score = "Severe Rejection", 0
        elif ratio > 1.5: sig, score = "Weak Close", 4
        else: sig, score = "Solid Close", 10
        
        return {
            "wickRejection": {
                "value": safe_float(round(ratio, 2)), 
                "score": score, 
                "desc": f"Wick/Body Ratio: {ratio:.1f} ({sig})"
            }
        }
    return _wrap_calc(_inner, "Wick Rejection")

def compute_cmf(df: pd.DataFrame, length: int = 20):
    def _inner():
        _Validator.require(df, ["High", "Low", "Close", "Volume"],min_rows=length, indicator="cmf")
        cmf = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=length)
        val = _Validator.extract_last(cmf, "CMF")
        if val is None: return {}
        
        desc = "Bullish" if val > 0.05 else "Bearish" if val < -0.05 else "Neutral"
        score = 10 if val > 0.1 else 7 if val > 0 else 3
        return {"cmfSignal": {"value": round(val, 3), "score": score, "desc": desc}}
    return _wrap_calc(_inner, "CMF")

def compute_52w_position(df: pd.DataFrame, horizon: str = "short_term") -> Dict[str, Dict[str, Any]]:
    """
    Calculates distance from 52-week high.
    Horizon-Aware: Uses 252 days for Short-Term/Daily and 52 weeks for Long-Term/Weekly.
    """
    def _inner():
        # 1. Determine lookback based on horizon
        # 252 days for daily (Short Term), 52 weeks for weekly (Long Term)
        lookback = 52 if horizon in ["long_term", "multibagger"] else 252
        
        _Validator.require(df, ["High", "Close"], min_rows=lookback)
        
        current_price = df["Close"].iloc[-1]
        high_period = df["High"].tail(lookback).max()
        
        if not high_period: return {}
        
        # Position as % of Period High (e.g., 98.5 means 1.5% below high)
        pos_pct = (current_price / high_period) * 100
        
        return {
            "position52w": {
                "raw": round(pos_pct, 2),
                "value": round(pos_pct, 2),
                "alias": "52W Position %",
                "desc": f"{pos_pct:.1f}% of 52W High",
                "source": "technical"
            }
        }
    return _wrap_calc(_inner, "52W Position")

def compute_composite_scores(indicators: Dict, horizon: str = "short_term") -> Dict:
    """
    Compute composite scores using config-based weighted scoring.
    
    Uses COMPOSITE_SCORING_CONFIG from technical_score_config.py
    to calculate trendStrength, momentumStrength, volatilityQuality.
    """
    from config.technical_score_config import compute_all_composites
    
    try:
        composites = compute_all_composites(indicators, horizon)
        return composites
    except Exception as e:
        logger.error(f"Composite scoring failed for {horizon}: {e}")
        
        # Fallback to legacy averaging (temporary safety net)
        return _compute_composites_legacy(indicators, horizon)

def _compute_composites_legacy(indicators: Dict, horizon: str = "short_term") -> Dict:
    """
    Compute composite scores by averaging existing metric scores.
    NO custom thresholds needed - reuses scores from compute_rsi(), compute_adx(), etc.
    """
    def _avg_scores(metric_keys: list) -> float:
        """Average score from existing indicators."""
        scores = []
        for key in metric_keys:
            metric = indicators.get(key, {})
            score = metric.get("score")
            if score is not None:
                scores.append(float(score))
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    
    # Horizon-specific metric lists (matches your HORIZON_METRIC_INCLUSION)
    trend_metrics = {
        "intraday": ["maFastSlope", "maTrendSignal", "supertrendSignal"],
        "short_term": ["adx", "maFastSlope", "maTrendSignal", "supertrendSignal"],
        "long_term": ["adx", "maTrendSignal"],
        "multibagger": ["adx", "maTrendSignal", "maFastSlope"]
    }
    
    momentum_metrics = {
        "intraday": ["rsi", "rsislope", "macd", "stochK"],
        "short_term": ["rsi", "rsislope", "macd", "stochK"],
        "long_term": ["rsi", "macd"],
        "multibagger": ["rsi", "macd"]
    }
    
    volatility_metrics = {
        "intraday": ["atrPct", "bbWidth"],
        "short_term": ["atrPct", "bbWidth"],
        "long_term": ["atrPct"],
        "multibagger": ["atrPct"]  # Added atrPct to prevent 0 volQuality
    }
    
    # Get metrics for this horizon
    trend_keys = trend_metrics.get(horizon, trend_metrics["short_term"])
    momentum_keys = momentum_metrics.get(horizon, momentum_metrics["short_term"])
    volatility_keys = volatility_metrics.get(horizon, volatility_metrics["short_term"])
    
    # Compute composites
    trend_score = _avg_scores(trend_keys)
    momentum_score = _avg_scores(momentum_keys)
    volatility_score = _avg_scores(volatility_keys)
    
    return {
        "trendStrength": {
            "value": trend_score,
            "score": trend_score,
            "desc": f"Trend Composite",
            "alias": "Trend Strength",
            "source": "composite"
        },
        "momentumStrength": {
            "value": momentum_score,
            "score": momentum_score,
            "desc": f"Momentum Composite",
            "alias": "Momentum Strength",
            "source": "composite"
        },
        "volatilityQuality": {
            "value": volatility_score,
            "score": volatility_score,
            "desc": f"Volatility Composite",
            "alias": "Volatility Quality",
            "source": "composite"
        }
    }

INDICATOR_METRIC_MAP = {
    "rsi": {"func": compute_rsi, "horizon": "default"},
    "mfi": {"func": compute_mfi, "horizon": "default"},
    "adx": {"func": compute_adx, "horizon": "default"},
    "cci": {"func": compute_cci, "horizon": "short_term"},
    "stochK": {"func": compute_stochastic, "horizon": "default"},
    "macd": {"func": compute_macd, "horizon": "default"},
    "vwap": {"func": compute_vwap, "horizon": "default"},
    "bbHigh": {"func": compute_bollinger_bands, "horizon": "default"},
    "rvol": {"func": compute_rvol, "horizon": "default"},
    "obvDiv": {"func": compute_obv_divergence, "horizon": "default"},
    # "atr14": {"func": compute_atr, "horizon": "default"},
    # "sl2xAtr": {"func": compute_atr_sl, "horizon": "default"},
    "atrDynamic": {"func": compute_dynamic_atr, "horizon": "default"},
    "slAtrDynamic": {"func": compute_dynamic_sl, "horizon": "default"},

    "supertrendSignal": {"func": compute_supertrend, "horizon": "short_term"},
    "psarTrend": {"func": compute_psar, "horizon": "short_term"},
    "ichiCloud": {"func": compute_ichimoku, "horizon": "long_term"},
    "priceAction": {"func": compute_price_action, "horizon": "default"},
    "entryConfirm": {"func": compute_entry_price, "horizon": "default"},
    "niftyTrendScore": {"func": compute_nifty_trend_score, "horizon": "default"},
    "gapPercent": {"func": compute_gap_percent, "horizon": "default"},
    "relStrengthNifty": {"func": compute_relative_strength, "horizon": "default"},
    # Consolidated Trend + Components

    "maTrendSignal": {"func": compute_dynamic_ma_trend, "horizon": "default"},
    "maCrossSignal": {"func": compute_dynamic_ma_cross, "horizon": "default"},
    "priceVsPrimaryTrendPct": {"func": compute_price_vs_base_ma, "horizon": "default"},
    "maFastSlope": {"func": compute_ema_slope, "horizon": "default"},

    # "maTrendSetup": {"func": compute_dynamic_ma_trend, "horizon": "default"},
    # "maCrossSetup": {"func": compute_dynamic_ma_cross, "horizon": "short_term"},
    # "priceVsMaSlowPct": {"func": compute_200dma, "horizon": "default"},
    # "ma_slopes": {"func": compute_ema_slope, "horizon": "default"},

    "pivotPoint": {"func": compute_pivot_points, "horizon": "short_term"},
    "ttmSqueeze": {"func": compute_keltner_squeeze, "horizon": "short_term"},
    "trueRange": {"func": compute_true_range, "horizon": "default"},
    "hv10": {"func": compute_historical_volatility, "horizon": "short_term"},
    "shortMaCross": {"func": compute_short_ma_cross, "horizon": "default"},
    "volTrend": {"func": compute_vol_trend, "horizon": "default"},
    "regSlope": {"func": compute_reg_slope, "horizon": "default"},
    "volSpikeRatio": {"func": compute_volume_spike, "horizon": "default"},
    "vpt": {"func": compute_vpt, "horizon": "default"},
    "cmfSignal": {"func": compute_cmf, "horizon": "short_term"},
    "vwapBias": {"func": compute_vwap, "horizon": "default"},
    "bbpercentb": {"func": compute_bollinger_bands, "horizon": "default"},
    "wickRejection": {"func": compute_wick_rejection, "horizon": "default"},
    "position52w": {"func": compute_52w_position, "horizon": "default"}
}

def compute_indicators(
    symbol: str,
    horizon: str = "short_term",
    benchmark_symbol: str = "^NSEI",
    df_hash: str = None, 
    benchmark_hash: str = None,
) -> Dict[str, Dict[str, Any]]:

    profile = HORIZON_PROFILE_MAP.get(horizon, {})
    if not profile: return {}
    
    raw_metrics = set(profile.get("metrics", {}).keys())
    if "penalties" in profile:
        raw_metrics.update(profile["penalties"].keys())
        
    for pool in [MOMENTUM_WEIGHTS, VALUE_WEIGHTS, GROWTH_WEIGHTS, QUALITY_WEIGHTS]:
        raw_metrics.update(pool.keys())
    raw_metrics.update(CORE_TECHNICAL_SETUP_METRICS)
    
    # 🚨 RESTORED LEGACY DENSITY: Force crucial structure metrics
    # This matches the legacy "calculate everything" approach for completeness
    ALWAYS_CALC = {
        "macd", "adx", "rsi", "maFastSlope", "cmfSignal", # Changed ma_slopes -> maFastSlope
        "obvDiv", "priceVsPrimaryTrendPct", "gapPercent", "maTrendSignal", # Changed maTrendSetup -> maTrendSignal
        "supertrendSignal", "psarTrend", "ttmSqueeze", "bbWidth", 
        "bbpercentb", "volSpikeRatio", "pivotPoint","regSlope",
        "maCrossSignal", # Changed maCrossSetup -> maCrossSignal
        "wickRejection", "atrDynamic", "slAtrDynamic","ichiCloud","trueRange","hv10","stochK", "stochD","position52w","relStrengthNifty"
    }
    raw_metrics.update(ALWAYS_CALC)
    
    PRIORITY = ["rsi", "macd", "maTrendSignal", "pivotPoint"]
    ordered = [m for m in PRIORITY if m in raw_metrics] + [m for m in raw_metrics if m not in PRIORITY]
    
    required_horizons = {horizon}
    # ⚡ STITCHING: Always include intraday for live stitching if we're in a slow horizon
    if horizon in ["long_term", "multibagger"]:
        required_horizons.add("intraday")

    for m in ordered:
        h_spec = INDICATOR_METRIC_MAP.get(m, {}).get("horizon", "default")
        if h_spec != "default": 
            required_horizons.add(h_spec)
        
    dfs_cache = {}
    for h in required_horizons:
        try:
            raw = get_history_for_horizon(symbol, h)
            if raw is not None and not raw.empty:
                dfs_cache[h] = _slice_for_speed(raw, horizon=h)
        except: pass

    # Find latest live price for stitching before we start indicator logic
    live_price = None
    for h in ["intraday", "short_term", horizon]:
        if h in dfs_cache and not dfs_cache[h].empty and "Close" in dfs_cache[h].columns:
            c_vals = dfs_cache[h]["Close"].dropna()
            if not c_vals.empty:
                live_price = float(c_vals.iloc[-1])
                break

    benchmark_df = None
    try:
        raw_bench = get_benchmark_data(horizon, benchmark_symbol)
        if raw_bench is not None and not raw_bench.empty:
            benchmark_df = raw_bench.copy()
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            if benchmark_df.index.tz is None:
                benchmark_df.index = benchmark_df.index.tz_localize("UTC")
            else:
                benchmark_df.index = benchmark_df.index.tz_convert("UTC")
            benchmark_df = benchmark_df.sort_index()
    except: pass

    indicators = {}
    detectedPatterns = {}
    
    if horizon in dfs_cache:
        try:
            # 🚨 SAFETY: Drop trailing NaNs from the series before indexing
            series = dfs_cache[horizon]["Close"].dropna()

            # ⚡ SMART STITCH: If weekly/monthly data is stale, append today's live price
            if horizon in ["long_term", "multibagger"] and live_price is not None:
                if not series.empty:
                    last_price = float(series.iloc[-1])
                    if abs(live_price - last_price) > 0.01: # Use 0.01 tolerance for float
                        # Append a pseudo-candle for the current "incomplete" period
                        stitch_idx = pd.Timestamp.now(tz=series.index.tz)
                        series = pd.concat([series, pd.Series([live_price], index=[stitch_idx])])

            indicators["symbol"] = {"value": symbol, "score":0, "alias":"Ticker", "desc": "Stock symbol"}
            # 1. Current Price
            if not series.empty:
                price = float(series.iloc[-1])
                indicators["price"] = {"value": round(price, 2), "score": 0, "alias": "Price", "desc": "Current"}

            if len(series) > 1:
                prev = float(series.iloc[-2])
                indicators["prevClose"] = {"value": round(prev, 2), "score": 0, "alias": "Prev Close", "desc": "Previous Close"}

            if len(series) > 10:
                price_10_ago = float(series.iloc[-11]) # -1 is current, -11 is 10 bars ago
                indicators["price10Ago"] = {"value": round(price_10_ago, 2), "score": 0}
                
                # Calculate Percentage Slope/Change
                slope_val = ((price - price_10_ago) / price_10_ago) * 100
                indicators["priceSlope"] = {"value": round(slope_val, 2), "score": 0, "alias": "Price Slope", "desc": "10-period % change" }
            else:
                # Fallback if history is too short (fewer than 10 bars)
                indicators["priceSlope"] = {"value": 0.0, "score": 0}
                logger.debug(f"[{symbol}] Price slope skipped: insufficient history ({len(series)} bars)")

        except Exception as e:
            pass

    # Execution flags to prevent duplicates
    done_flags = set()
    
    for metric in ordered:
        meta = INDICATOR_METRIC_MAP.get(metric)
        if not meta: continue
        
        # Bundle check
        if metric in done_flags: continue
        
        # Special Bundle Handling
        if metric == "maTrendSignal":
            done_flags.update(["maFast", "maMid", "maSlow", "maTrendSignal"])
            # mma24 added: multibagger maSlow is now MMA(24), emitted as legacy key "mma24"
            done_flags.update(["ema20", "ema50", "ema200", "wma10", "wma40", "wma50", "mma6", "mma12", "mma24"])
            
        elif metric == "maFastSlope":
            done_flags.update(["maFastSlope", "maSlowSlope"])
            done_flags.update(["ema20Slope", "ema50Slope", "wma50Slope", "mma12Slope"])

        fn = meta["func"]
        h_req = meta.get("horizon", "default")
        data_h = horizon if h_req == "default" else h_req
        
        df_local = dfs_cache.get(data_h)
        if df_local is None: continue
        
        kwargs = {}
        try:
            sig = inspect.signature(fn)
            for p in sig.parameters.values():
                if p.name == "symbol": kwargs["symbol"] = symbol
                elif p.name == "df": kwargs["df"] = df_local
                elif p.name == "benchmark_df": kwargs["benchmark_df"] = benchmark_df
                elif p.name == "price": kwargs["price"] = indicators.get("price", {}).get("value")
                elif p.name == "close": kwargs["close"] = df_local["Close"] if df_local is not None else None
                elif p.name == "high": kwargs["high"] = df_local["High"] if df_local is not None else None
                elif p.name == "low": kwargs["low"] = df_local["Low"] if df_local is not None else None
                elif p.name == "horizon": kwargs["horizon"] = data_h
            
            res = fn(**kwargs)
            if res:
                indicators.update(res)
                done_flags.add(metric)

        except Exception as e:
            logger.debug(f"[{symbol}] Metric {metric} failed: {e}")

    # --- Derived Metrics Calculation ---
    try:
        # 1. DI Spread
        di_p = indicators.get("diPlus", {}).get("value")
        di_m = indicators.get("diMinus", {}).get("value")
        if di_p is not None and di_m is not None:
            indicators["diSpread"] = {"value": round(di_p - di_m, 2), "score": 0}

        # 2. HV Trend
        hv10 = indicators.get("hv10", {}).get("value")
        hv20 = indicators.get("hv20", {}).get("value")
        if hv10 and hv20:
            hv_val = "declining" if hv10 < hv20 else "rising" if hv10 > (hv20 * 1.05) else "stable"
            indicators["hvTrend"] = {"value": hv_val, "score": 0}

        # 3. True Range Consistency
        tr = indicators.get("trueRange", {}).get("value")
        atr_pct = indicators.get("atrPct", {}).get("value")
        if tr and atr_pct and atr_pct > 0:
            tr_ratio = tr / atr_pct
            indicators["trueRangeConsistency"] = {"value": round(tr_ratio, 3), "score": 0}
        
        # 4. slDistance
        current_price = indicators.get('price', {}).get('value')
        sl_price = indicators.get('slAtrDynamic', {}).get('value')

        if current_price and sl_price:
            slDistance = abs(current_price - sl_price) / current_price * 100
            indicators['slDistance'] = {
                'value': slDistance,
                'desc': f'SL Distance: {slDistance:.2f}%',
                'alias': 'Stop Loss Distance (%)',
                'source': 'technical',
                'score': 0
            }

    except Exception as e:
        logger.debug(f"Derived metrics calculation failed: {e}")
    try:
        # =========================================================
        # NEW: PATTERN INJECTION
        # =========================================================
        # We use the DF specific to the requested horizon
        # If horizon is "intraday", we use the Intraday DF.
        df_for_patterns = dfs_cache.get(horizon)
        
        if df_for_patterns is not None and not df_for_patterns.empty:
            try:
                # 1. Detect Patterns
                # This function internally calls 'detect' on all patterns 
                # AND calls 'merge_pattern_into_indicators' to update the dict.
                detectedPatterns = run_pattern_analysis(df_for_patterns, indicators, horizon=horizon)

                # indicators.update(patterns)
            except Exception as e:
                logger.error(f"[{symbol}] Pattern detection failed for {horizon}: {e}")

        # =========================================================
        indicators.update(compute_composite_scores(indicators, horizon))
        # indicators["technicalScore"] = {"value": compute_technical_score(indicators, horizon), "score": 0}
        indicators["Horizon"] = {"value": horizon, "score": 0}
    except: pass
    return indicators, detectedPatterns

# Legacy Key (Hardcoded),Dynamic Key (Replacement),Why?
"""
indicators_keys = {
    'intraday': [
        'symbol', 'price', 'prevClose', 'price10Ago', 'priceSlope', 'rsi', 'rsislope',
        'macd', 'macdCross', 'macdHistZ', 'macdHistogram', 'prevMacdHistogram',
        'maFast', 'maMid', 'maSlow', 'maTrendSignal', 'ema20', 'ema50', 'ema200',
        'ema_20_50_200_trend', 'pivotPoint', 'resistance1', 'resistance2', 'resistance3',
        'support1', 'support2', 'support3', 'vwap', 'vwapBias', 'rvol','volume','avg_volume_30Days', 'obvDiv',
        'psarTrend', 'psarLevel', 'volSpikeRatio', 'volSpikeSignal', 'hv10', 'hv20',
        'stochK', 'stochD', 'stochCross', 'bbHigh', 'bbMid', 'bbLow', 'bbWidth',
        'bbpercentb', 'gapPercent', 'wickRejection', 'supertrendSignal', 'supertrendValue',
        'prevSupertrend', 'ichiCloud', 'ichiSpanA', 'ichiSpanB', 'ichiTenkan',
        'ichiKijun', 'ttmSqueeze', 'kcUpper', 'kcLower', 'adx', 'adx_signal', 'diPlus',
        'diMinus', 'niftyTrendScore', 'atrDynamic', 'atrPct', 'atrSmaRatio',
        'slAtrDynamic', 'riskPerSharePct', 'maCrossSignal', 'ema20_50Cross',
        'maFastSlope', 'maSlowSlope', 'ema20Slope', 'ema50Slope', 'trueRange',
        'trueRangePct', 'priceVsPrimaryTrendPct', 'price_vs_200ema_pct', 'priceAction',
        'cmfSignal', 'bollingerSqueeze', 'bollinger_squeeze_intraday', 'ichimokuSignals',
        'ichimoku_signals_intraday', 'goldenCross', 'golden_cross_intraday', 'doubleTopBottom',
        'double_top_bottom_intraday', 'technicalScore', 'Horizon','diSpread','hvTrend','trueRangeConsistency','trendStrength','momentumStrength','volatilityQuality'
    ],
    'short_term': [
        'symbol', 'price', 'prevClose', 'price10Ago', 'priceSlope', 'rsi', 'rsislope',
        'macd', 'macdCross', 'macdHistZ', 'macdHistogram', 'prevMacdHistogram',
        'maFast', 'maMid', 'maSlow', 'maTrendSignal', 'ema20', 'ema50', 'ema200',
        'ema_20_50_200_trend', 'pivotPoint', 'resistance1', 'resistance2', 'resistance3',
        'support1', 'support2', 'support3', 'vwap', 'vwapBias', 'obvDiv', 'rvol','avg_volume_30Days',
        'psarTrend', 'psarLevel', 'hv10', 'hv20', 'volSpikeRatio', 'volSpikeSignal',
        'stochK', 'stochD', 'stochCross', 'bbHigh', 'bbMid', 'bbLow', 'bbWidth',
        'bbpercentb', 'gapPercent', 'wickRejection', 'supertrendSignal', 'supertrendValue',
        'prevSupertrend', 'ichiCloud', 'ichiSpanA', 'ichiSpanB', 'ichiTenkan',
        'ichiKijun', 'ttmSqueeze', 'kcUpper', 'kcLower', 'adx', 'adx_signal', 'diPlus',
        'diMinus', 'niftyTrendScore', 'atrDynamic', 'atrPct', 'atrSmaRatio',
        'slAtrDynamic', 'riskPerSharePct', 'maCrossSignal', 'ema20_50Cross',
        'maFastSlope', 'maSlowSlope', 'ema20Slope', 'ema50Slope', 'trueRange',
        'trueRangePct', 'priceVsPrimaryTrendPct', 'price_vs_200ema_pct', 'priceAction',
        'cmfSignal', 'bollingerSqueeze', 'minerviniStage2', 'ichimokuSignals',
        'technicalScore', 'Horizon'
    ],
    'long_term': [
        'symbol', 'price', 'prevClose', 'price10Ago', 'priceSlope', 'rsi', 'rsislope',
        'macd', 'macdCross', 'macdHistZ', 'macdHistogram', 'prevMacdHistogram',
        'maFast', 'maMid', 'maSlow', 'maTrendSignal', 'wma10', 'wma40', 'wma50',
        'wma_10_40_50_trend', 'pivotPoint', 'resistance1', 'resistance2', 'resistance3',
        'support1', 'support2', 'support3', 'vwap', 'vwapBias', 'rvol','avg_volume_30Days', 'obvDiv',
        'psarTrend', 'psarLevel', 'hv10', 'hv20', 'volSpikeRatio', 'volSpikeSignal',
        'relStrengthNifty', 'stochK', 'stochD', 'stochCross', 'bbHigh', 'bbMid',
        'bbLow', 'bbWidth', 'bbpercentb', 'gapPercent', 'wickRejection',
        'supertrendSignal', 'supertrendValue', 'prevSupertrend', 'ichiCloud', 'ichiSpanA',
        'ichiSpanB', 'ichiTenkan', 'ichiKijun', 'ttmSqueeze', 'kcUpper', 'kcLower',
        'adx', 'adx_signal', 'diPlus', 'diMinus', 'niftyTrendScore', 'atrDynamic',
        'atrPct', 'atrSmaRatio', 'slAtrDynamic', 'riskPerSharePct', 'maCrossSignal',
        'wma_10_40_cross', 'maFastSlope', 'maSlowSlope', 'wma_20_slope', 'wma50Slope',
        'trueRange', 'trueRangePct', 'priceVsPrimaryTrendPct', 'price_vs_50wma_pct',
        'priceAction', 'cmfSignal', 'flagPennant', 'cupHandle', 'ichimokuSignals',
        'technicalScore', 'Horizon'
    ],
    'multibagger': [
        'symbol', 'price', 'prevClose', 'price10Ago', 'priceSlope', 'rsi', 'rsislope',
        'macd', 'macdCross', 'macdHistZ', 'macdHistogram', 'prevMacdHistogram',
        'maFast', 'maMid', 'maSlow', 'maTrendSignal', 'mma6', 'mma12',
        'mma_6_12_12_trend', 'pivotPoint', 'resistance1', 'resistance2', 'resistance3',
        'support1', 'support2', 'support3', 'vwap', 'vwapBias', 'rvol', 'avg_volume_30Days', 'obvDiv',
        'psarTrend', 'psarLevel', 'hv10', 'hv20', 'volSpikeRatio', 'volSpikeSignal',
        'relStrengthNifty', 'stochK', 'stochD', 'stochCross', 'bbHigh', 'bbMid',
        'bbLow', 'bbWidth', 'bbpercentb', 'gapPercent', 'wickRejection',
        'supertrendSignal', 'supertrendValue', 'prevSupertrend', 'ichiCloud', 'ichiSpanA',
        'ichiSpanB', 'ichiTenkan', 'ichiKijun', 'ttmSqueeze', 'kcUpper', 'kcLower',
        'adx', 'adx_signal', 'diPlus', 'diMinus', 'niftyTrendScore', 'atrDynamic',
        'atrPct', 'atrSmaRatio', 'slAtrDynamic', 'riskPerSharePct', 'maCrossSignal',
        'mma_6_12_cross', 'maFastSlope', 'maSlowSlope', 'mma_20_slope', 'mma_50_slope',
        'trueRange', 'trueRangePct', 'priceVsPrimaryTrendPct', 'price_vs_12mma_pct',
        'priceAction', 'cmfSignal', 'ichimokuSignals',
        'technicalScore', 'Horizon'
    ]
}
# Generic to Legacy mapping with descriptions
generic_to_legacy = {
    # Moving Average System
    'maFastSlope': {
        'legacy_keys': ['ema20Slope', 'wma_20_slope', 'mma_20_slope'],
        'description': 'The velocity/angle of the fast moving average (short-term trend momentum)',
        'horizons': {
            'intraday': 'ema20Slope',
            'short_term': 'ema20Slope', 
            'long_term': 'wma_20_slope',
            'multibagger': 'mma_20_slope'
        }
    },
    
    'maSlowSlope': {
        'legacy_keys': ['ema50Slope', 'wma50Slope', 'mma_50_slope'],
        'description': 'The velocity/angle of the slow moving average (long-term trend momentum)',
        'horizons': {
            'intraday': 'ema50Slope',
            'short_term': 'ema50Slope',
            'long_term': 'wma50Slope',
            'multibagger': 'mma_50_slope'
        }
    },
    
    'maCrossSignal': {
        'legacy_keys': ['ema20_50Cross', 'wma_10_40_cross', 'mma_6_12_cross'],
        'description': 'Fast/Slow MA crossover signal (1=bullish, -1=bearish, 0=neutral)',
        'horizons': {
            'intraday': 'ema20_50Cross',
            'short_term': 'ema20_50Cross',
            'long_term': 'wma_10_40_cross',
            'multibagger': 'mma_6_12_cross'
        }
    },
    
    'maTrendSignal': {
        'legacy_keys': ['ema_20_50_200_trend', 'wma_10_40_50_trend', 'mma_6_12_12_trend'],
        'description': 'Overall MA trend alignment (1=strong uptrend, -1=strong downtrend)',
        'horizons': {
            'intraday': 'ema_20_50_200_trend',
            'short_term': 'ema_20_50_200_trend',
            'long_term': 'wma_10_40_50_trend',
            'multibagger': 'mma_6_12_12_trend'
        }
    },
    
    'priceVsPrimaryTrendPct': {
        'legacy_keys': ['price_vs_200ema_pct', 'price_vs_50wma_pct', 'price_vs_12mma_pct'],
        'description': 'Price distance from primary trend MA (% above/below)',
        'horizons': {
            'intraday': 'price_vs_200ema_pct',
            'short_term': 'price_vs_200ema_pct',
            'long_term': 'price_vs_50wma_pct',
            'multibagger': 'price_vs_12mma_pct'
        }
    },
    
    'maFast': {
        'legacy_keys': ['ema20', 'wma10', 'mma6'],
        'description': 'Fast moving average value',
        'horizons': {
            'intraday': 'ema20',
            'short_term': 'ema20',
            'long_term': 'wma10',
            'multibagger': 'mma6'
        }
    },
    
    'maMid': {
        'legacy_keys': ['ema50', 'wma40', 'mma12'],
        'description': 'Medium moving average value',
        'horizons': {
            'intraday': 'ema50',
            'short_term': 'ema50',
            'long_term': 'wma40',
            'multibagger': 'mma12'
        }
    },
    
    'maSlow': {
        'legacy_keys': ['ema200', 'wma50', 'mma12'],
        'description': 'Slow/primary trend moving average value',
        'horizons': {
            'intraday': 'ema200',
            'short_term': 'ema200',
            'long_term': 'wma50',
            'multibagger': 'mma24'
        }
    },
    
    # Pattern Recognition
    'bollingerSqueeze': {
        'legacy_keys': ['bollinger_squeeze_intraday', 'bollinger_squeeze_short_term'],
        'description': 'Volatility squeeze pattern (consolidation before breakout)',
        'horizons': {
            'intraday': 'bollinger_squeeze_intraday',
            'short_term': 'bollinger_squeeze_short_term'
        }
    },
    
    'ichimokuSignals': {
        'legacy_keys': ['ichimoku_signals_intraday', 'ichimoku_signals_short_term', 
                       'ichimoku_signals_long_term', 'ichimoku_signals_multibagger'],
        'description': 'Ichimoku cloud signals (price vs cloud position)',
        'horizons': {
            'intraday': 'ichimoku_signals_intraday',
            'short_term': 'ichimoku_signals_short_term',
            'long_term': 'ichimoku_signals_long_term',
            'multibagger': 'ichimoku_signals_multibagger'
        }
    },
    
    # Fundamentals
    'profitGrowth3y': {
        'legacy_keys': ['epsGrowth3y'],
        'description': '3-year profit/EPS CAGR growth rate',
        'note': 'Both represent the same metric'
    }
}

# Duplicate mappings: Keys with identical values across horizons
duplicate_mappings = {
    # Moving Average Slopes
    'maFastSlope': ['ema20Slope', 'wma_20_slope', 'mma_20_slope'],
    'maSlowSlope': ['ema50Slope', 'wma50Slope', 'mma_24_slope'],
    
    # Moving Average Crossovers
    'maCrossSignal': ['ema20_50Cross', 'wma_10_40_cross', 'mma_6_12_cross'],
    
    # Moving Average Trend Signals
    'maTrendSignal': ['ema_20_50_200_trend', 'wma_10_40_50_trend', 'mma_6_12_24_trend'],
    
    # Price vs Primary Trend
    'priceVsPrimaryTrendPct': ['price_vs_200ema_pct', 'price_vs_50wma_pct', 'price_vs_12mma_pct'],
    
    # Moving Averages (Fast)
    'maFast': ['ema20', 'wma10', 'mma6'],
    
    # Moving Averages (Mid)
    'maMid': ['ema50', 'wma40', 'mma12'],
    
    # Moving Averages (Slow)
    'maSlow': ['ema200', 'wma50', 'mma24'],  # Note: updated to mma24
    
    # Pattern duplicates (horizon-specific)
    'bollingerSqueeze': ['bollinger_squeeze_intraday', 'bollinger_squeeze_short_term'],
    'ichimokuSignals': ['ichimoku_signals_intraday', 'ichimoku_signals_short_term', 
                         'ichimoku_signals_long_term', 'ichimoku_signals_multibagger'],
    'goldenCross': ['golden_cross_intraday'],
    'doubleTopBottom': ['double_top_bottom_intraday'],
    'minerviniStage2': ['minervini_stage2_short_term'],
    'flagPennant': ['flag_pennant_long_term'],
    'cupHandle': ['cup_handle_long_term'],
    
    # Growth metrics (fundamentals)
    'profitGrowth3y': ['epsGrowth3y'],  # Both represent 3Y profit/EPS CAGR
}
# ========================================
# 🎯 CANONICAL GENERIC KEYS
# ========================================

GENERIC_KEYS = {
    # === Moving Average System ===
    'maFast': {
        'legacy': {
            'intraday': 'ema20',
            'short_term': 'ema20', 
            'long_term': 'wma10',
            'multibagger': 'mma6'
        },
        'description': 'Fast moving average value (horizon-aware)'
    },
    
    'maMid': {
        'legacy': {
            'intraday': 'ema50',
            'short_term': 'ema50',
            'long_term': 'wma40',
            'multibagger': 'mma12'
        },
        'description': 'Medium moving average value'
    },
    
    'maSlow': {
        'legacy': {
            'intraday': 'ema200',
            'short_term': 'ema200',
            'long_term': 'wma50',
            'multibagger': 'mma24'  # Updated to mma24
        },
        'description': 'Slow/primary trend moving average'
    },
    
    'maFastSlope': {
        'legacy': {
            'intraday': 'ema20Slope',
            'short_term': 'ema20Slope',
            'long_term': 'wma10Slope',  
            'multibagger': 'mma6Slope'  
        },
        'description': 'Fast MA velocity (trend momentum)'
    },
    
    'maSlowSlope': {
        'legacy': {
            'intraday': 'ema50Slope',
            'short_term': 'ema50Slope',
            'long_term': 'wma50Slope',
            'multibagger': 'mma_24_slope'
        },
        'description': 'Slow MA velocity (long-term momentum)'
    },
    
    'maCrossSignal': {
        'legacy': {
            'intraday': 'ema20_50Cross',
            'short_term': 'ema20_50Cross',
            'long_term': 'wma_10_40_cross',
            'multibagger': 'mma_6_12_cross'
        },
        'description': 'Fast/Slow MA crossover (1=bull, -1=bear)'
    },
    
    'maTrendSignal': {
        'legacy': {
            'intraday': 'ema_20_50_200_trend',
            'short_term': 'ema_20_50_200_trend',
            'long_term': 'wma_10_40_50_trend',
            'multibagger': 'mma_6_12_24_trend'
        },
        'description': 'Overall MA alignment (1=strong up, -1=strong down)'
    },
    
    'priceVsPrimaryTrendPct': {
        'legacy': {
            'intraday': 'price_vs_200ema_pct',
            'short_term': 'price_vs_200ema_pct',
            'long_term': 'price_vs_50wma_pct',
            'multibagger': 'price_vs_12mma_pct'
        },
        'description': 'Price distance from primary trend MA (% above/below)'
    },
    
    # === Volatility & Risk ===
    'atrDynamic': {
        'legacy': None,  # No legacy key, always was generic
        'description': 'Horizon-aware ATR value'
    },
    
    'slAtrDynamic': {
        'legacy': None,
        'description': 'Dynamic stop loss based on ATR'
    },
    
    # === Oscillators (Always Generic) ===
    'rsi': {'legacy': None},
    'rsislope': {'legacy': None},
    'macd': {'legacy': None},
    'macdCross': {'legacy': None},
    'macdHistogram': {'legacy': None},
    'adx': {'legacy': None},
    'stochK': {'legacy': None},
    'stochD': {'legacy': None},
    'stochCross': {'legacy': None},
    
    # === Volume ===
    'rvol': {'legacy': None},
    'volSpikeRatio': {'legacy': None},
    'volSpikeSignal': {'legacy': None},
    'obvDiv': {'legacy': None},
    
    # === Trend ===
    'supertrendSignal': {'legacy': None},
    'psarTrend': {'legacy': None},
    'ichiCloud': {'legacy': None},
    'niftyTrendScore': {'legacy': None},
}

"""
