"""
services/patterns/double_top_bottom.py

Downstream meta consumers
─────────────────────────
trade_enhancer (via raw.meta path):
  • meta["age_candles"]         – check_pattern_expiration + extract_pattern_execution_metadata
  • meta["formation_time"]      – real-time age recalculation

setup_pattern_matrix PATTERN_METADATA invalidation namespace (bullish/bearishNecklinePattern):
  • meta["neckline"]            – breakdown gates: {"max_metric": "neckline"} / {"min_metric": "neckline"}
  • meta["peak_similarity"]     – entry gates: {"max": 0.02}
  • meta["pattern_height_pct"]  – analytics field in metadata_keys

config_resolver _calculate_pattern_targets (neckline block L3676):
  • meta["target"]              – pre-calculated pattern target price
  • meta["neckline"]            – SL anchored just above/below neckline
  • meta["type"]                – "bullish" | "bearish" branch selector

All three must be present or target calculation falls through to ATR fallback.

Alias fixes applied
───────────────────
  BullishNecklinePattern → alias = "bullishNecklinePattern"  ✅
  BearishNecklinePattern → alias = "bearishNecklinePattern"  ✅
  Point 3: duplicate peak_similarity key removed — appears exactly once per class  ✅
  Point 6: HORIZON_WINDOWS applied to lookback window and min_history guard        ✅
  _guard() used                                                                     ✅
  formation_time key used (matches trade_enhancer L374/946)                         ✅
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.horizon_constants import HORIZON_WINDOWS_BARS


# ─────────────────────────────────────────────────────────────────────────────
# Shared peak / trough detectors (pure NumPy, no SciPy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _find_peaks_numpy(arr: np.ndarray, distance: int):
    peaks = []
    if len(arr) < distance * 2:
        return peaks
    for i in range(distance, len(arr) - distance):
        window = arr[i - distance: i + distance + 1]
        if np.argmax(window) == distance:
            peaks.append(i)
    return peaks


def _find_troughs_numpy(arr: np.ndarray, distance: int):
    troughs = []
    if len(arr) < distance * 2:
        return troughs
    for i in range(distance, len(arr) - distance):
        window = arr[i - distance: i + distance + 1]
        if np.argmin(window) == distance:
            troughs.append(i)
    return troughs


# ─────────────────────────────────────────────────────────────────────────────
# BullishNecklinePattern  (Double Bottom → bullish breakout)
# ─────────────────────────────────────────────────────────────────────────────

class BullishNecklinePattern(BasePattern):
    """
    Detects Double Bottom (bullish) pattern using pure NumPy.
    Alias: "bullishNecklinePattern" — matches SETUP_PATTERN_MATRIX key.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias       = "bullishNecklinePattern"
        self.peak_window = 5

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if not self._is_horizon_supported(horizon):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}
        # Point 6 fix: horizon-aware window
        hw          = HORIZON_WINDOWS_BARS.get(horizon, HORIZON_WINDOWS_BARS["short_term"])
        window_size = hw["window"]
        min_history = hw["min_history"]

        empty = self._guard(df, min_history)
        if empty is not None:
            return empty

        if getattr(self, "coerce_numeric", False):
            df = self.ensure_numeric_df(df)

        window        = df.tail(window_size).copy()
        highs         = window["High"].values
        lows          = window["Low"].values
        closes        = window["Close"].values
        current_price = float(closes[-1])

        troughs = _find_troughs_numpy(lows, self.peak_window)
        if len(troughs) < 2:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        t2, t1 = troughs[-1], troughs[-2]
        if t2 <= t1:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        price1 = float(lows[t1])
        price2 = float(lows[t2])

        # Both troughs at similar level (within 3 %)
        if not (0.97 <= (price2 / price1) <= 1.03):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        # Neckline = peak between the two troughs
        peak_rel = int(np.argmax(highs[t1:t2]))
        neckline  = float(highs[t1 + peak_rel])

        # Breakout: price closed above neckline
        if current_price <= neckline:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        height = neckline - price1
        target = neckline + height                      # resolver reads meta["target"]
        first_point_index = t1
        entry_conditions_met = current_price > neckline

        # Point 3 fix: peak_similarity defined exactly once
        peak_similarity = abs((price2 - price1) / price1)

        # Horizon-specific invalidation (price closes back below neckline = pattern failed)
        if horizon == "intraday":
            invalidation_level = neckline * 0.998
        elif horizon == "short_term":
            invalidation_level = neckline * 0.995
        else:
            invalidation_level = neckline * 0.99
        
        # Guard against index errors if first_point_index is somehow invalid
        try:
            formation_ts = window.index[first_point_index]
        except (IndexError, KeyError):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        result = {
            "found":   True,
            "score":   self._normalize_score(80),
            "quality": 8.5,
            "desc":    "Double Bottom Breakout",
            "meta": {
                # ── Fields read by trade_enhancer ────────────────────────────
                "age_candles":    window_size - first_point_index,
                "formation_time": formation_ts.timestamp(),    # key read by enhancer L374/946
                # ── Fields read by resolver _calculate_pattern_targets L3677 ─
                "target":    round(target, 2),          # resolver: depth = abs(target - neckline)
                "neckline":  round(neckline, 2),        # resolver: SL = neckline * 0.99
                "type":      "bullish",                 # resolver: branch selector
                # ── Fields in PATTERN_METADATA invalidation namespace ─────────
                # Invalidation gates use "neckline" as max_metric/min_metric key
                # Entry gates use "peak_similarity" as a max gate
                # analytics: pattern_height_pct
                "peak_similarity":   round(peak_similarity, 4),
                "pattern_height_pct": round(((neckline - price1) / price1) * 100, 2),
                # ── Structural / UI fields ─────────────────────────────────────
                "bar_index":              len(df),
                "peak_1":                 round(price1, 2),
                "peak_2":                 round(price2, 2),
                "pattern_quality":        "strong" if peak_similarity <= 0.02 else "moderate",
                "formation_timestamp":    formation_ts.isoformat(),  # ISO for logging/UI
                "pattern_duration_candles": abs(t2 - t1),
                "invalidation_level":     round(invalidation_level, 2),
                "entry_trigger_price":    round(neckline, 2),
                "horizon":                horizon,
                "pattern_strength":       "strong",
                "current_price":          round(current_price, 2),
                "velocity_tracking": {
                    "can_track":            entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient":   True,
                    "breakout_confirmed":   True,
                },
                "formation_context": _build_formation_context(indicators),
            },
        }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BearishNecklinePattern  (Double Top → bearish breakdown)
# ─────────────────────────────────────────────────────────────────────────────

class BearishNecklinePattern(BasePattern):
    """
    Detects Double Top (bearish) pattern using pure NumPy.
    Alias: "bearishNecklinePattern" — matches SETUP_PATTERN_MATRIX key.

    Note: system is currently long-only; _calculate_pattern_targets in
    config_resolver returns None for the bearish branch (L3688).  This class
    still detects the pattern so it appears in CONFLICTING lists and can
    suppress bullish setups when a double-top breakdown is active.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias       = "bearishNecklinePattern"
        self.peak_window = 5

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if not self._is_horizon_supported(horizon):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}
        # Point 6 fix: horizon-aware window
        hw          = HORIZON_WINDOWS_BARS.get(horizon, HORIZON_WINDOWS_BARS["short_term"])
        window_size = hw["window"]
        min_history = hw["min_history"]

        empty = self._guard(df, min_history)
        if empty is not None:
            return empty

        if getattr(self, "coerce_numeric", False):
            df = self.ensure_numeric_df(df)

        window        = df.tail(window_size).copy()
        highs         = window["High"].values
        lows          = window["Low"].values
        closes        = window["Close"].values
        current_price = float(closes[-1])

        peaks = _find_peaks_numpy(highs, self.peak_window)
        if len(peaks) < 2:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        p2, p1 = peaks[-1], peaks[-2]
        if p2 <= p1:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        price1 = float(highs[p1])
        price2 = float(highs[p2])

        # Both peaks at similar level (within 3 %)
        if not (0.97 <= (price2 / price1) <= 1.03):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        # Neckline = trough between the two peaks
        trough_rel = int(np.argmin(lows[p1:p2]))
        neckline   = float(lows[p1 + trough_rel])

        # Breakdown: price closed below neckline
        if current_price >= neckline:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        height = price1 - neckline
        target = neckline - height                      # resolver reads meta["target"]
        first_point_index = p1
        entry_conditions_met = current_price < neckline

        # Point 3 fix: peak_similarity defined exactly once
        peak_similarity = abs((price2 - price1) / price1)

        # Invalidation (price reclaims neckline from below = bearish pattern failed)
        if horizon == "intraday":
            invalidation_level = neckline * 1.002
        elif horizon == "short_term":
            invalidation_level = neckline * 1.005
        else:
            invalidation_level = neckline * 1.01

        try:
            formation_ts = window.index[first_point_index]
        except (IndexError, KeyError):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        result = {
            "found":   True,
            "score":   self._normalize_score(80),
            "quality": 8.5,
            "desc":    "Double Top Breakdown",
            "meta": {
                # ── Fields read by trade_enhancer ────────────────────────────
                "age_candles":    window_size - first_point_index,
                "formation_time": formation_ts.timestamp(),    # key read by enhancer L374/946
                # ── Fields read by resolver _calculate_pattern_targets L3677 ─
                "target":    round(target, 2),          # resolver reads this even for bearish
                "neckline":  round(neckline, 2),        # resolver: returns None for bearish branch
                "type":      "bearish",                 # resolver: branch selector → returns None
                # ── Fields in PATTERN_METADATA invalidation namespace ─────────
                "peak_similarity":    round(peak_similarity, 4),
                "pattern_height_pct": round(((price1 - neckline) / neckline) * 100, 2),
                # ── Structural / UI fields ─────────────────────────────────────
                "bar_index":              len(df),
                "peak_1":                 round(price1, 2),
                "peak_2":                 round(price2, 2),
                "pattern_quality":        "strong" if peak_similarity <= 0.02 else "moderate",
                "formation_timestamp":    formation_ts.isoformat(),  # ISO for logging/UI
                "pattern_duration_candles": abs(p2 - p1),
                "invalidation_level":     round(invalidation_level, 2),
                "entry_trigger_price":    round(neckline, 2),
                "horizon":                horizon,
                "pattern_strength":       "strong",
                "current_price":          round(current_price, 2),
                "velocity_tracking": {
                    "can_track":            entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient":   True,
                    "breakdown_confirmed":  True,
                },
                "formation_context": _build_formation_context(indicators),
            },
        }
        return result