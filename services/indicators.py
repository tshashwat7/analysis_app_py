# services/indicators.py
"""
Phase 3 Refactored (v7 - FINAL GOLD): 
- RESTORED: ema_20_50_cross for Short Term.
- LOGIC FIX: 'gap_percent' now correctly scores moderate gaps (0.5% - 2.0%).
- LOGIC FIX: 'ma_trend' awards partial score (7) if Short MA > Mid MA (Developing Trend).
- ACCURACY: Uses WMA/MMA for slopes on Long Term/Multibagger horizons.
"""

import logging
from typing import Dict, Any, Callable, List, Optional, Tuple, Union
import warnings
import numpy as np
from math import atan, degrees
import pandas as pd
import pandas_ta as ta
import inspect

logger = logging.getLogger(__name__)

# Shared helpers
from config.constants import (
    ATR_HORIZON_CONFIG, ATR_MULTIPLIERS, CORE_TECHNICAL_SETUP_METRICS, GROWTH_WEIGHTS, MOMENTUM_WEIGHTS, 
    QUALITY_WEIGHTS, TECHNICAL_WEIGHTS, HORIZON_PROFILE_MAP, 
    TECHNICAL_METRIC_MAP, VALUE_WEIGHTS
)
from services.data_fetch import (
    _safe_float,
    get_benchmark_data,
    safe_float,
    _wrap_calc,
    get_history_for_horizon,
    _safe_get_raw_float
)

PYTHON_LOOKBACK_MAP = {
    "intraday": 500,    
    "short_term": 600,  
    "long_term": 800,   
    "multibagger": 3000 
}

# ========================================================
# 🔧 INTERNAL HELPERS
# ========================================================

class _Validator:
    @staticmethod
    def require(df: pd.DataFrame, cols: List[str] = None, min_rows: int = 1):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        if cols and not set(cols).issubset(df.columns):
            raise ValueError(f"Missing required columns: {cols}")
        if len(df) < min_rows:
            raise ValueError(f"Insufficient data: {len(df)} < {min_rows}")

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
        _Validator.require(df, ["Close"])
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
            "rsi_slope": {"value": round(slope_val, 2), "score": 0, "desc": f"Slope {slope_val:.2f}"}
        }
    return _wrap_calc(_inner, "RSI")

def compute_mfi(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close", "Volume"])
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
        _Validator.require(df, ["High", "Low", "Close"])
        val = _Validator.extract_last(ta.cci(df["High"], df["Low"], df["Close"], length=length), "CCI")
        if val is None: return {}

        if val < -100: desc, score = "Oversold (Buy)", 10
        elif val > 100: desc, score = "Overbought (Sell)", 0
        else: desc, score = "Neutral", 5

        return {"cci": {"value": round(val, 2), "score": score, "desc": desc}}
    return _wrap_calc(_inner, "CCI")

def compute_adx(df: pd.DataFrame, length: int = 14) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"])
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=length)
        if adx_df is None or adx_df.empty: raise ValueError("ADX empty")

        adx_col = next((c for c in adx_df.columns if "adx" in c.lower()), None)
        di_p_col = next((c for c in adx_df.columns if "dmp" in c.lower() or "di+" in c.lower()), None)
        di_m_col = next((c for c in adx_df.columns if "dmn" in c.lower() or "di-" in c.lower()), None)

        if not adx_col: raise ValueError("ADX column missing")

        adx_val = _Validator.extract_last(adx_df[adx_col], "ADX")
        if adx_val is None: return {}
        
        di_plus = safe_float(adx_df[di_p_col].iloc[-1]) if di_p_col else None
        di_minus = safe_float(adx_df[di_m_col].iloc[-1]) if di_m_col else None

        if adx_val < 20: score, trend = 0, "Weak / Range-bound"
        elif adx_val <= 25: score, trend = 5, "Developing / Moderate"
        else: score, trend = 10, "Strong Trend"

        return {
            "adx": {"value": round(adx_val, 2), "raw": adx_val, "score": score, "desc": f"adx -> {adx_val:.2f}"},
            "adx_signal": {"value": trend, "score": score, "desc": f"adx_signal -> {trend}"},
            "di_plus": {"value": round(di_plus, 2) if di_plus else None, "score": 5, "desc": f"di_plus -> {di_plus:.1f}"},
            "di_minus": {"value": round(di_minus, 2) if di_minus else None, "score": 5, "desc": f"di_minus -> {di_minus:.1f}"},
        }
    return _wrap_calc(_inner, "ADX")

def compute_stochastic(df: pd.DataFrame, k_length: int = 14, d_length: int = 3, smooth_k: int = 3) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"])
        stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=k_length, d=d_length, smooth_k=smooth_k)
        if stoch_df is None or stoch_df.empty: raise ValueError("Stoch empty")

        stoch_df.columns = [c.lower() for c in stoch_df.columns]
        k_col = next((c for c in stoch_df.columns if "k" in c), None)
        d_col = next((c for c in stoch_df.columns if "d" in c), None)

        k_val = _Validator.extract_last(stoch_df[k_col], "Stoch %K")
        d_val = _Validator.extract_last(stoch_df[d_col], "Stoch %D")
        
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
            "stoch_k": {"value": round(k_val, 2), "score": score},
            "stoch_d": {"value": round(d_val, 2), "score": score},
            "stoch_cross": {"value": cross_status, "score": cross_score},
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
            "macd_cross": {"value": cross, "score": score, "desc": f"macd_cross -> {cross}"},
            "macd_hist_z": {"value": round(hist_z, 4), "score": hist_score, "desc": "MACD Hist Z-Score"},
            "macd_histogram": {"value": round(hist_val, 3), "score": hist_strength, "desc": f"Hist {hist_val:.3f}"},
        }
    return _wrap_calc(_inner, "MACD")

def compute_vwap(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close", "Volume"])
        vwap_val = _Validator.extract_last(ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]), "VWAP")
        if vwap_val is None: return {}
        price = _Validator.extract_last(df["Close"], "Close")

        if price > vwap_val: bias, score = "Bullish", 10
        elif price < vwap_val: bias, score = "Bearish", 0
        else: bias, score = "Neutral", 5

        return {
            "vwap": {"value": round(vwap_val, 2), "score": score, "desc": f"vwap -> {vwap_val:.1f}"}, 
            "vwap_bias": {"value": bias, "score": score, "desc": f"vwap_bias -> {bias}"}
        }
    return _wrap_calc(_inner, "VWAP")

def compute_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close"])
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
            "bb_high": {"value": round(upper, 2), "score": 0, "desc": f"bb_high -> {upper:.2f}"},
            "bb_mid": {"value": round(mid, 2), "score": 0, "desc": f"bb_mid -> {mid:.2f}"},
            "bb_low": {"value": round(lower, 2), "score": score, "desc": band},
            "bb_width": {"value": round(bb_width_val, 2), "raw": bb_width_val, "score": width_score, "desc": width_desc},
            "bb_percent_b": {"value": round(pct_b, 3), "score": 10 if pct_b < 0.2 else 0 if pct_b > 0.8 else 5, "desc": f"bb_percent_b -> {pct_b:.3f}"}
        }
    return _wrap_calc(_inner, "Bollinger Bands")

def compute_rvol(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Volume"])
        vol = df["Volume"]
        today = safe_float(vol.iloc[-1])
        avg = safe_float(vol.tail(20).mean())
        if not avg: raise ValueError("Zero avg volume")
        
        rvol = today / avg
        score = 10 if rvol > 1.5 else 0 if rvol < 0.8 else 5
        return {"rvol": {"value": round(rvol, 2), "score": score, "desc": f"rvol -> {rvol:.2f}"}}
    return _wrap_calc(_inner, "RVOL")

def compute_obv_divergence(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close", "Volume"])
        obv = ta.obv(df["Close"], df["Volume"])
        if obv is None or len(obv) < 5: raise ValueError("OBV data insufficient")
        
        p_chg = safe_float(df["Close"].iloc[-1]) - safe_float(df["Close"].iloc[-5])
        o_chg = safe_float(obv.iloc[-1]) - safe_float(obv.iloc[-5])
        
        if p_chg * o_chg > 0: sig, score = "Confirming", 10
        elif p_chg * o_chg < 0: sig, score = "Diverging", 0
        else: sig, score = "Neutral", 5
        
        return {"obv_div": {"value": sig, "score": score, "desc": f"obv_div -> {sig}"}}
    return _wrap_calc(_inner, "OBV Divergence")

def compute_vpt(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["Close", "Volume"])
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
        _Validator.require(df, ["Volume"])
        today = safe_float(df["Volume"].iloc[-1])
        avg = safe_float(df["Volume"].tail(20).mean())
        if not avg: raise ValueError("Zero avg volume")
        
        ratio = today / avg
        if ratio > 1.5: sig, score = "Strong Spike", 10
        elif ratio > 1.2: sig, score = "Moderate Spike", 5
        else: sig, score = "Normal", 5
        
        return {
            "vol_spike_ratio": {"value": round(ratio, 2), "score": score, "desc": f"vol_spike_ratio -> {ratio:.2f}"},
            "vol_spike_signal": {"value": sig, "score": score, "desc": f"vol_spike_signal -> {sig}"}
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
            "atr_14": {"value": round(atr, 2), "score": score, "desc": f"atr_14 -> {atr:.2f}"},
            "atr_pct": {"value": round(pct, 2), "score": score, "desc": f"{pct:.2f}%"}
        }
    return _wrap_calc(_inner, "ATR")

# use dynamic one
def compute_dynamic_atr(df: pd.DataFrame, horizon: str = "short_term") -> Dict[str, Dict[str, Any]]:
    def _inner():
        length = ATR_HORIZON_CONFIG.get(horizon, 14)

        _Validator.require(df, ["High", "Low", "Close"])

        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=length)
        atr_val = _Validator.extract_last(atr_series, f"ATR_{length}")

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

        return {
            "atr_dynamic": {
                "value": round(atr_val, 2),
                "length": length,               # Metadata for UI/Debug
                "score": score,
                "alias": f"ATR ({length})",     # Dynamic Label
                "desc": f"ATR({length}) = {atr_val:.2f}",
                "source": "technical"
            },
            "atr_pct": {
                "value": round(pct, 2),
                "score": score,
                "desc": f"Volatility {pct:.2f}%",
                "source": "technical"
            }
        }
    return _wrap_calc(_inner, "Dynamic ATR")

def compute_dynamic_sl(df: pd.DataFrame, price: float, horizon: str = "short_term"):
    def _inner():
        length = ATR_HORIZON_CONFIG.get(horizon, 14)
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
            "sl_atr_dynamic": {
                "value": round(sl_price, 2),
                "alias": f"SL ({multiplier}x ATR{length})", # Clear Label
                "desc": f"Stop Loss using ATR({length})",
                "source": "technical"
            },
            "risk_per_share_pct": {
                "value": round(risk_pct, 2),
                "desc": f"Risk {risk_pct:.1f}%",
                "source": "technical"
            }
        }
    return _wrap_calc(_inner, "Dynamic ATR Stop Loss")

def compute_supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> Dict[str, Dict[str, Any]]:
    def _inner():
        # 1. Keep your Validators
        _Validator.require(df, ["High", "Low", "Close"])
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
        if d_col:
            trend_val = _safe_float(st[d_col].iloc[-1])
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
            "supertrend_signal": { "value": sig, "score": score, "desc": f"ST ({length},{multiplier}) {sig}"},
            "supertrend_value": {"value": st_level_val, "score": 0, "desc": f"Level {st_level_val}","alias": "SuperTrend Value"}
        }
    return _wrap_calc(_inner, "SuperTrend")

def compute_price_action(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"])
        high = df["High"].iloc[-1]
        low = df["Low"].iloc[-1]
        close = df["Close"].iloc[-1]
        
        if high == low: raise ValueError("Zero range candle")
        pos = (close - low) / (high - low)
        
        if pos >= 0.75: sig, score = "Strong Bullish Close", 10
        elif pos >= 0.5: sig, score = "Moderate Close", 5
        else: sig, score = "Weak Close", 0
        
        return {"price_action": {"value": round(pos*100, 1), "score": score, "signal": sig, "desc": f"price_action -> {round(pos*100, 1)}"}}
    return _wrap_calc(_inner, "Price Action")

def compute_200dma(df: pd.DataFrame, close, price, horizon: str = "short_term"):
    def _inner():
        lens = {"intraday": 200, "short_term": 200, "long_term": 50, "multibagger": 12}
        types = {"intraday": "MA", "short_term": "DMA", "long_term": "WMA", "multibagger": "MMA"}
        
        L = lens.get(horizon, 200)
        T = types.get(horizon, "MA")
        
        sma = _Validator.extract_last(ta.sma(close, length=L), f"{L}{T}")
        
        diff = ((price - sma) / sma) * 100
        if diff > 0: score, desc = 10, "Uptrend"
        elif abs(diff) < 3: score, desc = 5, "Neutral"
        else: score, desc = 0, "Downtrend"
        
        return {
            f"price_vs_{L}{T.lower()}_pct": {"value": round(diff, 2), "score": score, "desc": f"{desc} ({diff:.1f}%)"},
            f"{T.lower()}_{L}": {"value": round(sma, 2), "score": 0, "desc": f"{T.lower()}_{L} -> {sma:.2f}"}
        }
    return _wrap_calc(_inner, "Long-Term MA")

def compute_ema_slope(df: pd.DataFrame, horizon: str = "short_term", lookback: int = 10):
    """
    Dynamic Slope Calculator: Switches between EMA (Short) and WMA/MMA (Long) based on horizon.
    """
    def _inner():
        cfg = {
            "intraday":    {"type": "EMA", "l_s": 20, "l_l": 50},
            "short_term":  {"type": "EMA", "l_s": 20, "l_l": 50},
            "long_term":   {"type": "WMA", "l_s": 20, "l_l": 50}, 
            "multibagger": {"type": "MMA", "l_s": 20, "l_l": 50}
        }.get(horizon, {"type": "EMA", "l_s": 20, "l_l": 50})
        
        fn = ta.ema if cfg["type"] == "EMA" else ta.sma # WMA/MMA approximated by SMA for slope stability
        prefix = "ema" if cfg["type"] == "EMA" else "wma" if cfg["type"] == "WMA" else "mma"
        
        close = df["Close"]
        
        def _get_slope(series):
            if len(series) < lookback: return 0.0
            y = series.tail(lookback).values
            if np.isnan(y).any(): return 0.0
            x = np.arange(lookback)
            slope, _ = np.polyfit(x, y, 1)
            return degrees(atan(slope))

        s_ma = fn(close, length=cfg["l_s"])
        l_ma = fn(close, length=cfg["l_l"])
        
        ang_s = _get_slope(s_ma)
        ang_l = _get_slope(l_ma)
        
        score = 10 if ang_s > 1.5 else 5 if ang_s > 0 else 0
        
        return {
            f"{prefix}_{cfg['l_s']}_slope": {"value": round(ang_s, 2), "raw": ang_s, "score": score, "desc": f"{ang_s:.1f}°"},
            f"{prefix}_{cfg['l_l']}_slope": {"value": round(ang_l, 2), "raw": ang_l, "score": 0, "desc": f"{ang_l:.1f}°"}
        }
    return _wrap_calc(_inner, "MA Slopes")

def compute_dynamic_ma_cross(df: pd.DataFrame, close: pd.Series, horizon: str = "short_term"):
    cfg = {
        "intraday":    {"l": (20, 50), "t": "EMA", "p": "ema"},
        "short_term":  {"l": (20, 50), "t": "EMA", "p": "ema"},
        "long_term":   {"l": (10, 40), "t": "SMA", "p": "wma"},
        "multibagger": {"l": (6, 12),  "t": "SMA", "p": "mma"}
    }.get(horizon, {"l": (20, 50), "t": "EMA", "p": "ema"})
    
    s_len, l_len = cfg["l"]
    fn = ta.ema if cfg["t"] == "EMA" else ta.sma
    prefix = cfg["p"]

    def _inner():
        s_ma = fn(close, length=s_len)
        l_ma = fn(close, length=l_len)
        
        vals_s = _safe_last_vals(s_ma, 2)
        vals_l = _safe_last_vals(l_ma, 2)
        
        if not vals_s or not vals_l: raise ValueError("Insufficient MA data")
        
        s_curr, l_curr = vals_s[1], vals_l[1]
        s_prev, l_prev = vals_s[0], vals_l[0]
        
        val, score, desc = -1, 0, "Bearish"
        if s_curr > l_curr:
            val, desc = 1, "Bullish"
            score = 10 if (s_prev <= l_prev) else 7 
        elif (s_prev >= l_prev) and (s_curr < l_curr):
            val, desc = -1, "Bearish Cross"
            
        return {
            f"{prefix}_{s_len}": {"value": round(s_curr, 2), "score": 0, "desc": f"{prefix}_{s_len} -> {s_curr:.2f}"},
            f"{prefix}_{l_len}": {"value": round(l_curr, 2), "score": 0, "desc": f"{prefix}_{l_len} -> {l_curr:.2f}"},
            f"{prefix}_{s_len}_{l_len}_cross": {"value": val, "score": score, "desc": desc}
        }
    return _wrap_calc(_inner, f"{prefix.upper()} Cross")

def compute_dynamic_ma_trend(df: pd.DataFrame, horizon: str = "short_term"):
    """
    Calculates 3-MA trend AND returns individual MA values to ensure database completeness.
    """
    cfg = {
        "intraday":    {"l": (20, 50, 200), "t": "EMA", "p": "ema"},
        "short_term":  {"l": (20, 50, 200), "t": "EMA", "p": "ema"},
        "long_term":   {"l": (10, 40, 50),  "t": "SMA", "p": "wma"},
        "multibagger": {"l": (6, 12, 12),   "t": "SMA", "p": "mma"}
    }.get(horizon, {"l": (20, 50, 200), "t": "EMA", "p": "ema"})
    
    l_s, l_m, l_l = cfg["l"]
    fn = ta.ema if cfg["t"] == "EMA" else ta.sma
    pre = cfg["p"]

    def _inner():
        _Validator.require(df, ["Close"])
        s_val = _Validator.extract_last(fn(df["Close"], length=l_s), "MA Fast")
        m_val = _Validator.extract_last(fn(df["Close"], length=l_m), "MA Mid")
        try: l_val = _Validator.extract_last(fn(df["Close"], length=l_l), "MA Slow")
        except: l_val = None
        
        val, score, desc = 0, 5, "Neutral"
        if l_val:
            if s_val > m_val > l_val: val, score, desc = 1, 10, "Strong Uptrend"
            elif s_val < m_val < l_val: val, score, desc = -1, 0, "Strong Downtrend"
            elif s_val > m_val > l_val: val, score, desc = 0.5, 7, "Moderate Uptrend"
            # FIX: Partial Trend Logic (Short > Mid, even if Mid < Long)
            elif s_val > m_val: val, score, desc = 0.5, 7, "Developing Uptrend"
            
        return {
            f"{pre}_{l_s}": {"value": round(s_val, 2), "score": 0, "desc": f"{pre}_{l_s} -> {s_val:.2f}"},
            f"{pre}_{l_m}": {"value": round(m_val, 2), "score": 0, "desc": f"{pre}_{l_m} -> {m_val:.2f}"},
            f"{pre}_{l_l}": {"value": round(l_val, 2) if l_val else None, "score": 0, "desc": f"{pre}_{l_l} -> {l_val:.2f}" if l_val else "N/A"},
            f"{pre}_{l_s}_{l_m}_{l_l}_trend": {"value": val, "score": score, "desc": desc}
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
            "pivot_point":  {"value": round(pp, 2), "score": 0},
            "resistance_1": {"value": round(r1, 2), "score": 0},
            "resistance_2": {"value": round(r2, 2), "score": 0},
            "resistance_3": {"value": round(r3, 2), "score": 0},
            "support_1":    {"value": round(s1, 2), "score": 0},
            "support_2":    {"value": round(s2, 2), "score": 0},
            "support_3":    {"value": round(s3, 2), "score": 0},
        }
    return _wrap_calc(_inner, "Pivot Levels")

def compute_keltner_squeeze(df: pd.DataFrame, bb_len=20, kc_len=20) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"])
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
            "ttm_squeeze": {"value": "Squeeze On" if sqz else "Off", "score": 10 if sqz else 5, "desc": f"ttm_squeeze -> {'On' if sqz else 'Off'}"},
            "kc_upper": {"value": round(kc_u_val, 2), "score": 0, "desc": f"kc_upper -> {kc_u_val:.2f}"},
            "kc_lower": {"value": round(kc_l_val, 2), "score": 0, "desc": f"kc_lower -> {kc_l_val:.2f}"}
        }
    return _wrap_calc(_inner, "TTM Squeeze")

def compute_ichimoku(symbol: str, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    def _inner():
        _Validator.require(df, ["High", "Low", "Close"], min_rows=60)
        ichi = ta.ichimoku(df["High"], df["Low"], df["Close"])[0]
        cols = {c: c for c in ichi.columns}
        def _get(k): return _Validator.extract_last(ichi[next(c for c in cols if k in c)])
        
        tk, kj = _get("TS"), _get("KS")
        sa, sb = _get("SA"), _get("SB")
        px = _Validator.extract_last(df["Close"])
        
        bull = (px > max(sa, sb)) and (tk > kj)
        bear = (px < min(sa, sb)) and (tk < kj)
        
        if bull: sig, score = "Strong Bullish", 10
        elif bear: sig, score = "Strong Bearish", 0
        elif px > max(sa, sb): sig, score = "Mild Bullish", 7
        elif px < min(sa, sb): sig, score = "Mild Bearish", 3
        else: sig, score = "Neutral", 5
        
        return {
            "ichi_cloud": {"value": sig, "score": score},
            "ichi_span_a": {"value": round(sa, 2), "score": 0},
            "ichi_span_b": {"value": round(sb, 2), "score": 0}
        }
    return _wrap_calc(_inner, "Ichimoku")

def compute_nifty_trend_score(benchmark_df: pd.DataFrame):
    def _inner():
        if benchmark_df is None or benchmark_df.empty: return {"nifty_trend_score": {"value": None, "score": None}}
        close = benchmark_df["Close"].dropna()
        if len(close) < 20: return {"nifty_trend_score": {"value": None, "score": None}}
        
        cur = close.iloc[-1]
        ema50 = safe_float(ta.ema(close, 50).iloc[-1])
        ema200 = safe_float(ta.ema(close, 200).iloc[-1])
        
        if ema200:
            diff = (cur - ema200) / ema200 * 100
            if ema50 and cur > ema50 > ema200: sig, score = "Strong Uptrend", 9
            elif cur > ema200: sig, score = "Moderate Uptrend", 7
            else: sig, score = "Downtrend", 2
        elif ema50:
            diff = (cur - ema50) / ema50 * 100
            if cur > ema50: sig, score = "Uptrend (Weak)", 7
            else: sig, score = "Downtrend", 3
        else:
            return {"nifty_trend_score": {"value": None, "score": None}}
            
        return {"nifty_trend_score": {"value": round(diff, 2), "score": score, "desc": sig}}
    return _wrap_calc(_inner, "NIFTY Trend")

def compute_relative_strength(symbol, df, benchmark_df, horizon="short_term"):
    def _inner():
        _Validator.require(df, ["Close"])
        _Validator.require(benchmark_df, ["Close"])
        
        lookback = {"intraday": 20, "long_term": 52}.get(horizon, 20)
        
        s_now, s_old = df["Close"].iloc[-1], df["Close"].iloc[-lookback]
        b_now, b_old = benchmark_df["Close"].iloc[-1], benchmark_df["Close"].iloc[-lookback]
        
        s_ret = (s_now/s_old - 1) * 100
        b_ret = (b_now/b_old - 1) * 100
        rs = s_ret - b_ret
        
        score = 10 if rs > 0 else 0
        return {"rel_strength_nifty": {"value": round(rs, 2), "score": score, "desc": f"Alpha {rs:.1f}%"}}
    return _wrap_calc(_inner, "RS vs Nifty")

def compute_entry_price(df, price):
    def _inner():
        mid = ta.sma(df["Close"], 20).iloc[-1]
        confirm = mid * 1.005
        return {"entry_confirm": {"value": round(confirm, 2), "score": 10 if price > confirm else 5}}
    return _wrap_calc(_inner, "Entry")

def compute_atr_sl(df, indicators, high, low, close, price):
    def _inner():
        atr = ta.atr(high, low, close, 14).iloc[-1]
        return {"sl_2x_atr": {"value": round(price - 2*atr, 2), "score": 0}}
    return _wrap_calc(_inner, "SL")

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

        return {"gap_percent": {"value": round(gap, 2), "score": score, "desc": f"gap_percent -> {gap:.2f}"}}
    return _wrap_calc(_inner, "Gap")

def compute_psar(df: pd.DataFrame):
    def _inner():
        psar = ta.psar(df["High"], df["Low"], df["Close"])
        r_col = next((c for c in psar.columns if "r" in c.lower()), None)
        val_col = next((c for c in psar.columns if c != r_col), None)
        
        trend = psar[r_col].iloc[-1] 
        level = psar[val_col].iloc[-1]
        
        sig, score = ("Bullish", 10) if trend > 0 else ("Bearish", 0)
        
        if pd.isna(level): level = df["Close"].iloc[-1] 
        
        return {
            "psar_trend": {"value": sig, "score": score, "desc": f"psar_trend -> {sig}"},
            "psar_level": {"value": round(level, 2), "score": 0, "desc": f"psar_level -> {level:.2f}"}
        }
    return _wrap_calc(_inner, "PSAR")

def compute_true_range(df):
    def _inner():
        tr = ta.true_range(df["High"], df["Low"], df["Close"]).iloc[-1]
        pct = (tr / df["Close"].iloc[-1]) * 100
        return {"true_range": {"value": round(tr, 2), "score": 0}, "true_range_pct": {"value": round(pct, 2), "score": 5}}
    return _wrap_calc(_inner, "TR")

def compute_historical_volatility(df, periods=(10, 20)):
    def _inner():
        ret = np.log(df["Close"] / df["Close"].shift(1))
        res = {}
        for p in periods:
            hv = ret.tail(p).std() * np.sqrt(252) * 100
            res[f"hv_{p}"] = {"value": round(hv, 2), "score": 10 if hv < 20 else 5}
        return res
    return _wrap_calc(_inner, "HV")

def compute_short_ma_cross(df):
    def _inner():
        e5 = ta.ema(df["Close"], 5).iloc[-1]
        e20 = ta.ema(df["Close"], 20).iloc[-1]
        return {"short_ma_cross": {"value": "Bull" if e5 > e20 else "Bear", "score": 10 if e5 > e20 else 0}}
    return _wrap_calc(_inner, "Short MA Cross")

def compute_vol_trend(df):
    def _inner():
        v = df["Volume"]
        trend = "Rising" if v.iloc[-1] > v.tail(50).mean() * 1.2 else "Neutral"
        return {"vol_trend": {"value": trend, "score": 10 if trend == "Rising" else 5}}
    return _wrap_calc(_inner, "Vol Trend")

def compute_reg_slope(df):
    def _inner():
        y = df["Close"].tail(20).values
        x = np.arange(len(y))
        slope = degrees(atan(np.polyfit(x, y, 1)[0]))
        return {"reg_slope": {"value": round(slope, 2), "score": 10 if slope > 2 else 0}}
    return _wrap_calc(_inner, "Reg Slope")

def compute_wick_rejection(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculates the 'Upper Wick Ratio' to detect Bull Traps.
    Ratio = Upper Wick / Candle Body.
    > 2.0 implies the market rejected higher prices (Shooting Star-like).
    """
    def _inner():
        _Validator.require(df, ["Open", "High", "Close", "Low"])
        
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
            "wick_rejection": {
                "value": safe_float(round(ratio, 2)), 
                "score": score, 
                "desc": f"Wick/Body Ratio: {ratio:.1f} ({sig})"
            }
        }
    return _wrap_calc(_inner, "Wick Rejection")

def compute_cmf(df: pd.DataFrame, length: int = 20):
    def _inner():
        _Validator.require(df, ["High", "Low", "Close", "Volume"])
        cmf = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=length)
        val = _Validator.extract_last(cmf, "CMF")
        if val is None: return {}
        
        desc = "Bullish" if val > 0.05 else "Bearish" if val < -0.05 else "Neutral"
        score = 10 if val > 0.1 else 7 if val > 0 else 3
        return {"cmf_signal": {"value": round(val, 3), "score": score, "desc": desc}}
    return _wrap_calc(_inner, "CMF")

def compute_technical_score(indicators: Dict[str, Dict[str, Any]], weights=None) -> int:
    if not indicators: return 0
    weights = dict(weights or TECHNICAL_WEIGHTS)
    score, weight_sum = 0.0, 0.0
    
    try:
        adx = indicators.get("adx", {}).get("raw", 0)
        if adx > 25: weights.update({"macd_cross": 1.5, "rsi": 0.8})
        elif adx < 20: weights.update({"rsi": 1.2, "stoch_k": 1.2})
    except: pass

    for k, w in weights.items():
        item = indicators.get(k, {})
        s = item.get("score")
        if s is not None:
            score += s * w
            weight_sum += 10 * w
            
    return round(score / weight_sum * 100, 1) if weight_sum > 0 else 0

INDICATOR_METRIC_MAP = {
    "rsi": {"func": compute_rsi, "horizon": "default"},
    "mfi": {"func": compute_mfi, "horizon": "default"},
    "adx": {"func": compute_adx, "horizon": "short_term"},
    "cci": {"func": compute_cci, "horizon": "short_term"},
    "stoch_k": {"func": compute_stochastic, "horizon": "short_term"},
    "macd": {"func": compute_macd, "horizon": "default"},
    "vwap": {"func": compute_vwap, "horizon": "intraday"},
    "bb_high": {"func": compute_bollinger_bands, "horizon": "default"},
    "rvol": {"func": compute_rvol, "horizon": "default"},
    "obv_div": {"func": compute_obv_divergence, "horizon": "default"},
    # "atr_14": {"func": compute_atr, "horizon": "default"},
    # "sl_2x_atr": {"func": compute_atr_sl, "horizon": "default"},
    "atr_dynamic": {"func": compute_dynamic_atr, "horizon": "default"},
    "sl_atr_dynamic": {"func": compute_dynamic_sl, "horizon": "default"},

    "supertrend_signal": {"func": compute_supertrend, "horizon": "short_term"},
    "psar_trend": {"func": compute_psar, "horizon": "short_term"},
    "ichi_cloud": {"func": compute_ichimoku, "horizon": "long_term"},
    "price_action": {"func": compute_price_action, "horizon": "default"},
    "entry_confirm": {"func": compute_entry_price, "horizon": "default"},
    "nifty_trend_score": {"func": compute_nifty_trend_score, "horizon": "long_term"},
    "gap_percent": {"func": compute_gap_percent, "horizon": "long_term"},
    "rel_strength_nifty": {"func": compute_relative_strength, "horizon": "long_term"},
    # Consolidated Trend + Components
    "ma_trend_setup": {"func": compute_dynamic_ma_trend, "horizon": "default"},
    "ma_cross_setup": {"func": compute_dynamic_ma_cross, "horizon": "short_term"},
    "price_vs_200dma_pct": {"func": compute_200dma, "horizon": "default"},
    "ma_slopes": {"func": compute_ema_slope, "horizon": "default"},
    "pivot_point": {"func": compute_pivot_points, "horizon": "short_term"},
    "ttm_squeeze": {"func": compute_keltner_squeeze, "horizon": "short_term"},
    "true_range": {"func": compute_true_range, "horizon": "default"},
    "hv_10": {"func": compute_historical_volatility, "horizon": "short_term"},
    "short_ma_cross": {"func": compute_short_ma_cross, "horizon": "default"},
    "vol_trend": {"func": compute_vol_trend, "horizon": "default"},
    "reg_slope": {"func": compute_reg_slope, "horizon": "default"},
    "vol_spike_ratio": {"func": compute_volume_spike, "horizon": "default"},
    "vpt": {"func": compute_vpt, "horizon": "default"},
    "cmf_signal": {"func": compute_cmf, "horizon": "short_term"},
    "vwap_bias": {"func": compute_vwap, "horizon": "intraday"},
    "bb_percent_b": {"func": compute_bollinger_bands, "horizon": "default"},
    "wick_rejection": {"func": compute_wick_rejection, "horizon": "default"},
}

def compute_indicators(
    symbol: str,
    df_hash: str = None, 
    benchmark_hash: str = None,
    horizon: str = "short_term",
    benchmark_symbol: str = "^NSEI"
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
        "macd", "adx", "rsi", "ma_slopes", "cmf_signal", 
        "obv_div", "price_vs_200dma_pct", "gap_percent", "ma_trend_setup",
        "supertrend_signal", "psar_trend", "ttm_squeeze", "bb_width", 
        "bb_percent_b", "vol_spike_ratio", "pivot_point",
        # 🚨 FIX: Explicitly add Short Term Cross back
        "ma_cross_setup" , "wick_rejection", "atr_dynamic", "sl_atr_dynamic"
    }
    raw_metrics.update(ALWAYS_CALC)
    
    PRIORITY = ["rsi", "macd", "ma_trend_setup", "pivot_point"]
    ordered = [m for m in PRIORITY if m in raw_metrics] + [m for m in raw_metrics if m not in PRIORITY]
    
    required_horizons = {horizon}
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

    benchmark_df = None
    try:
        raw_bench = get_benchmark_data("long_term", benchmark_symbol)
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
    
    if horizon in dfs_cache:
        try:
            price = float(dfs_cache[horizon]["Close"].iloc[-1])
            indicators["price"] = {"value": round(price, 2), "score": 0, "alias": "Price", "desc": "Current"}
        except: pass

    # Execution flags to prevent duplicates
    done_flags = set()
    
    for metric in ordered:
        meta = INDICATOR_METRIC_MAP.get(metric)
        if not meta: continue
        
        # Bundle check
        if metric in done_flags: continue
        
        # Special Bundle Handling
        if metric == "ma_trend_setup":
            # This handles ema_20, ema_50, ema_200 AND trend
            done_flags.update(["ema_20", "ema_50", "ema_200", "wma_10", "wma_40", "wma_50", "mma_6", "mma_12"])
            
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

    try:
        indicators["technical_score"] = {"value": compute_technical_score(indicators), "score": 0}
        indicators["Horizon"] = {"value": horizon, "score": 0}
    except: pass

    return indicators