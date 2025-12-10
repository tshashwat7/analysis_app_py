# services/tradeplan/time_estimator.py
import math
from typing import Dict, Any, Optional

# --- Local Helpers to prevent Import Errors ---
def _ensure_numeric(x, default=0.0):
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, dict): return float(x.get("value") or x.get("raw") or default)
        return float(default)
    except: return float(default)

def _get_val_local(data, key, default=None):
    if not data or key not in data: return default
    return data[key]

def _extract_score(x):
    """Safely extract numeric value from indicator dict or float."""
    if isinstance(x, dict):
        return x.get("value") or x.get("score") or x.get("raw") or 5.0
    return float(x) if x is not None else 5.0

# --- MAIN DUAL ESTIMATOR ---
def estimate_hold_time_dual(
    entry: float,
    t1: Optional[float],
    t2: Optional[float],
    atr: float,
    horizon: str,
    indicators: Dict,
    strategy_summary: Dict = None,
    multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Estimates distinct hold times for T1 (Conservation) and T2 (Extension).
    Robustly handles cases where Targets or Entry are missing.
    """
    # Safe default for confidence to prevent UI crashes
    safe_conf = {"t1": "Low", "t2": "Low"}

    # 1. Guard: Entry Validity
    if not entry or entry <= 0:
        return {
            "t1_estimate": "-", "t2_estimate": "-", 
            "note": "Invalid Entry", 
            "confidence": safe_conf
        }

    # 2. Guard: Target Validity (Fixes the Crash)
    if t1 is None or t2 is None:
        return {
            "t1_estimate": "-", "t2_estimate": "-", 
            "note": "No Targets Set", 
            "confidence": safe_conf
        }

    # 3. Calculate Distances
    try:
        t1_dist = abs(t1 - entry)
        t2_dist = abs(t2 - entry)
    except TypeError:
        return {
            "t1_estimate": "-", "t2_estimate": "-", 
            "note": "Invalid Target Data", 
            "confidence": safe_conf
        }
    
    # 4. Get Velocity Metrics
    atr_val = _ensure_numeric(atr)
    if atr_val <= 0: 
        return {
            "t1_estimate": "-", "t2_estimate": "-", 
            "note": "Invalid ATR", 
            "confidence": safe_conf
        }
    
    # Trend Speed Adjustment (The "Physics")
    ts_raw = _get_val_local(indicators, "trend_strength")
    trend_strength = _extract_score(ts_raw)
    adx = _ensure_numeric(_get_val_local(indicators, "adx"), 20.0)
    
    # Base Velocity (ATR per bar)
    # If trend is strong (>7), price moves 1.2x ATR per bar
    # If trend is weak (<3), price moves 0.6x ATR per bar
    velocity_factor = 0.8  # Base friction
    if trend_strength >= 7.0: velocity_factor = 1.2
    elif trend_strength >= 5.0: velocity_factor = 1.0
    elif trend_strength <= 3.0: velocity_factor = 0.6
    
    # Apply Strategy Multiplier (e.g. Momentum = Fast, Value = Slow)
    final_velocity = atr_val * velocity_factor * (1.0 / multiplier) # Lower multiplier = Faster

    # 5. Calculate Bars
    t1_bars = t1_dist / max(final_velocity, 0.01)
    t2_bars = t2_dist / max(final_velocity, 0.01)

    # 6. Confidence Logic
    t1_conf = "High" if trend_strength > 6 else "Medium"
    t2_conf = "Medium" if (trend_strength > 6 and adx > 25) else "Low"

    # 7. Formatter
    def fmt_bars(bars, hz):
        if hz == "intraday":
            mins = bars * 15 # Assuming 15m candles
            if mins < 60: return f"~{int(mins)}m"
            return f"~{round(mins/60, 1)}h"
        elif hz == "short_term":
            days = math.ceil(bars)
            if days <= 1: return "1-2 Days"
            if days < 5: return f"~{int(days)} Days"
            return f"~{math.ceil(days/5)} Weeks"
        else: # Long term
            weeks = math.ceil(bars / 5)
            if weeks < 4: return f"~{weeks} Wks"
            return f"~{round(weeks/4, 1)} Mths"

    return {
        "t1_estimate": fmt_bars(t1_bars, horizon),
        "t2_estimate": fmt_bars(t2_bars, horizon),
        "t1_bars": round(t1_bars, 1),
        "t2_bars": round(t2_bars, 1),
        "confidence": {"t1": t1_conf, "t2": t2_conf},
        "velocity_note": f"Vel: {velocity_factor}x ATR"
    }