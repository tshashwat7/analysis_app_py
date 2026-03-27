"""
services/patterns/golden_cross.py

Downstream meta consumers
─────────────────────────
trade_enhancer (via raw.meta path):
  • meta["age_candles"]      – check_pattern_expiration + extract_pattern_execution_metadata
  • meta["formation_time"]   – real-time age recalculation

setup_pattern_matrix PATTERN_METADATA (goldenCross / deathCross):
  • entry_rules gates reference indicator keys (maMid, maSlow, rvol) — all from indicators,
    none from meta.  No meta keys needed for entry namespace evaluation.
  • invalidation gates reference maMid, maSlow, adx, rsi, price — all from indicators.
    No meta keys needed here either.

# Target calculation handled by Stage 2 (TradeEnhancer) via RR Regime Multipliers
# No meta fields needed for structural baseline in ConfigResolver

Alias fixes applied
───────────────────
  GoldenCross  → alias = "goldenCross"   (was "goldenDeathCross" — broke matrix lookup)  ✅
  DeathCross   → alias = "deathCross"    (new split class — matrix treats them separately) ✅
  Point 2: invalidation_level computed INSIDE cross_type block                              ✅
  _guard() used                                                                             ✅
  formation_time key used (matches trade_enhancer L374/946)                                 ✅
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context
from services.patterns.horizon_constants import HORIZON_MA_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Shared detection helper
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cross(
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    horizon: str,
    expected_cross: str,   # "bullish" | "bearish"
    alias: str,
    base: "BasePattern",
) -> Dict[str, Any]:
    """
    Core MA-crossover detection logic shared by both GoldenCross and DeathCross.
    Returns the full pattern result dict.
    """
    if not base._is_horizon_supported(horizon):
        return {"found": False, "score": 0, "quality": 0, "meta": {}}
    cfg = HORIZON_MA_CONFIG.get(horizon, HORIZON_MA_CONFIG["short_term"])
    mid_len  = cfg["mid_len"]
    slow_len = cfg["slow_len"]
    ma_type  = cfg["type"]

    # Guard based on the slower MA requirements + 1 for previous value check
    empty = base._guard(df, slow_len + 1)
    if empty is not None:
        return empty

    maMid  = base._get_val(indicators, "maMid") 
    maSlow = base._get_val(indicators, "maSlow")

    if maMid is None or maSlow is None:
        return {"found": False, "score": 0, "quality": 0, "meta": {}}

    try:
        close = df["Close"]
        
        # Calculate previous values using the same MA type and periods as indicators.py
        if ma_type == "EMA":
            maMid_prev  = close.ewm(span=mid_len, adjust=False).mean().iloc[-2]
            maSlow_prev = close.ewm(span=slow_len, adjust=False).mean().iloc[-2]
        elif ma_type == "WMA":
            # Simple WMA calculation for a specific point
            def get_wma_at(series, length, index):
                weights = np.arange(1, length + 1)
                window = series.iloc[index - length + 1 : index + 1]
                return np.dot(window, weights) / weights.sum()
            
            maMid_prev  = get_wma_at(close, mid_len, len(close) - 2)
            maSlow_prev = get_wma_at(close, slow_len, len(close) - 2)
        else: # SMA
            maMid_prev  = close.rolling(mid_len).mean().iloc[-2]
            maSlow_prev = close.rolling(slow_len).mean().iloc[-2]

        if pd.isna(maMid_prev) or pd.isna(maSlow_prev):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}
    except Exception as e:
        base.log_debug(f"[{alias}] MA calc error: {e}")
        return {"found": False, "score": 0, "quality": 0, "meta": {}}

    # Detect crossover direction
    cross_type = None
    if maMid_prev < maSlow_prev and maMid > maSlow:
        cross_type = "bullish"
        desc = f"Golden Cross ({mid_len}/{slow_len} {ma_type})"
    elif maMid_prev > maSlow_prev and maMid < maSlow:
        cross_type = "bearish"
        desc = f"Death Cross ({mid_len}/{slow_len} {ma_type})"

    if cross_type != expected_cross:
        return {"found": False, "score": 0, "quality": 0, "meta": {}}

    # Invalidation at maMid - the structural level that must hold.
    invalidation_level = maMid

    cross_strength       = abs(maMid - maSlow) / maSlow if maSlow else 0
    entry_conditions_met = True   # the cross itself is the entry trigger

    result = {
        "found":   True,
        "score":   base._normalize_score(90),
        "quality": 9.0,
        "desc":    desc,
        "meta": {
            "age_candles":    1,
            "formation_time": df.index[-1].timestamp(),
            "bar_index":            len(df),
            "type":                 cross_type,
            "maMid":                round(maMid, 2),
            "maSlow":               round(maSlow, 2),
            "ma_type":              ma_type,
            "mid_len":              mid_len,
            "slow_len":             slow_len,
            "formation_timestamp":  df.index[-1].isoformat(),
            "crossover_fresh":      True,
            "cross_strength":       round(cross_strength, 4),
            "trend_confirmation":   "confirmed" if cross_strength > 0.05 else "early",
            "invalidation_level":   round(invalidation_level, 2) if invalidation_level else None,
            "horizon":              horizon,
            "pattern_strength":     "strong",
            "current_price":        round(float(df["Close"].iloc[-1]), 2),
            "velocity_tracking": {
                "can_track":            entry_conditions_met,
                "entry_conditions_met": entry_conditions_met,
                "quality_sufficient":   True,
                "cross_type":           cross_type,
                "fresh_cross":          True,
            },
            "formation_context": _build_formation_context(indicators),
        },
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GoldenCross  (alias: "goldenCross")
# ─────────────────────────────────────────────────────────────────────────────

class GoldenCross(BasePattern):
    """
    Detects 50 SMA crossing ABOVE 200 SMA (bullish trend change).
    Alias: "goldenCross" — matches SETUP_PATTERN_MATRIX and config_resolver keys.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias               = "goldenCross"
        self.horizons_supported  = ["short_term", "long_term", "multibagger"]

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        return _detect_cross(df, indicators, horizon, "bullish", self.alias, self)


# ─────────────────────────────────────────────────────────────────────────────
# DeathCross  (alias: "deathCross")
# ─────────────────────────────────────────────────────────────────────────────

class DeathCross(BasePattern):
    """
    Detects 50 SMA crossing BELOW 200 SMA (bearish trend change).
    Alias: "deathCross" — matrix uses this as a CONFLICTING signal for bullish setups.

    Note: system is currently long-only; DeathCross appears in CONFLICTING lists
    to suppress bullish trades when a death cross is active.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias               = "deathCross"
        self.horizons_supported  = ["short_term", "long_term", "multibagger"]

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        return _detect_cross(df, indicators, horizon, "bearish", self.alias, self)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy alias — kept so any import of GoldenDeathCross doesn't break
# ─────────────────────────────────────────────────────────────────────────────

class GoldenDeathCross(GoldenCross):
    """
    Deprecated: use GoldenCross or DeathCross directly.

    ⚠️  IMPORTANT: This class detects BULLISH (golden) crosses ONLY.
    It does NOT detect death crosses — it extends GoldenCross and
    always uses expected_cross="bullish".  Instantiating this class
    to detect bearish signals will silently return found=False.

    Use DeathCross for bearish (death cross) signals.
    Kept for backward-compatibility only.  alias is still "goldenCross".
    """
    pass