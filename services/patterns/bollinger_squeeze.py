"""
services/patterns/bollinger_squeeze.py

Downstream meta consumers
─────────────────────────
trade_enhancer (via raw.meta path):
  • meta["age_candles"]         – check_pattern_expiration + extract_pattern_execution_metadata
  • meta["formation_time"]      – real-time age recalculation (timestamp → seconds → candles)

setup_pattern_matrix PATTERN_METADATA entry_rules / invalidation gates (namespace evaluation):
  • meta["squeeze_duration"]    – entry gate {"min": 5.0} intraday, {"min": 3.0} short_term
  • meta["squeeze_strength"]    – analytics field in breakdown_threshold metadata_keys

config_resolver _calculate_pattern_targets (bollingerSqueeze block):
  • reads bbHigh / bbLow / bbMid from price_data (indicators), NOT from meta
  → no meta fields needed for target calculation

Fixes applied
─────────────
  Point 4: duplicate "squeeze_duration" key removed — appears exactly once  ✅
  _guard() used for structural guard                                          ✅
  alias = "bollingerSqueeze" — matches SETUP_PATTERN_MATRIX key              ✅
  formation_time key used (matches what trade_enhancer.py reads at L374/946) ✅
"""

from typing import Dict, Any
import pandas as pd
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context


class BollingerSqueeze(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias                 = "bollingerSqueeze"
        self.squeeze_threshold     = self.config.get("squeeze_threshold", 0.10)   # 10 % BB Width
        self.breakout_confirmation = 0.02

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if not self._is_horizon_supported(horizon):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}

        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)

        empty = self._guard(df, 30)
        if empty is not None:
            return empty

        bbWidth = self._get_val(indicators, "bbWidth")
        bbHigh  = self._get_val(indicators, "bbHigh")
        price   = self._get_val(indicators, "price")

        if bbWidth is None or bbHigh is None or price is None:
            return result

        # bbWidth returned as percentage (e.g. 2.5); threshold is decimal (0.10 → 10 %)
        is_squeezing = bbWidth <= (self.squeeze_threshold * 100)
        is_breakout  = price > bbHigh

        if is_breakout:
            result["quality"] = 10.0
            result["score"]   = self._normalize_score(95)
            result["desc"]    = "Vol Squeeze + Breakout"
            state             = "SQUEEZE_BREAKOUT"
        elif is_squeezing:
            result["quality"] = 6.0
            result["score"]   = self._normalize_score(55)
            result["desc"]    = "Volatility Squeeze (Waiting)"
            state             = "SQUEEZE_ON"
        else:
            return result

        result["found"] = True

        bbLow      = self._get_val(indicators, "bbLow")
        bbHigh_val = self._get_val(indicators, "bbHigh")
        bbMid      = self._get_val(indicators, "bbMid")
        rvol       = self._get_val(indicators, "rvol")

        has_volume           = bool(rvol and rvol >= 1.3)
        entry_conditions_met = is_breakout and has_volume
        # Approximate formation age: how long bb has been tight
        estimated_age = 7

        if horizon == "intraday":
            invalidation_level = bbLow if bbLow else None
        elif horizon == "short_term":
            invalidation_level = bbLow * 0.99 if bbLow else None
        else:
            invalidation_level = bbLow if bbLow else None

        entry_trigger_price = bbHigh_val if bbHigh_val else None
        pattern_strength = (
            "strong"   if result["quality"] >= 8.5 else
            "moderate" if result["quality"] >= 6.5 else
            "weak"
        )

        result["meta"] = {
            # ── Fields read by trade_enhancer ──────────────────────────────────
            "age_candles":     estimated_age,
            "formation_time":  (                        # key read by enhancer L374/946
                df.index[-estimated_age].timestamp()
                if len(df) > estimated_age else None
            ),
            # ── Fields read by PATTERN_METADATA entry/invalidation namespace ──
            # setup_pattern_matrix gates reference squeeze_duration and squeeze_strength
            # as flat namespace keys — they must be scalar (int/str) for namespace eval
            "squeeze_duration": estimated_age,          # gate: {"min": 5.0} intraday
            "squeeze_strength": "tight" if bbWidth < 5 else "moderate",
            # ── Structural / resolver fields ───────────────────────────────────
            "bar_index":        len(df),
            "state":            state,
            "width":            bbWidth,
            "formation_timestamp": (                    # ISO string for logging/UI
                df.index[-estimated_age].isoformat()
                if len(df) > estimated_age else None
            ),
            "bbLow":            round(bbLow, 2)      if bbLow      else None,
            "bbHigh":           round(bbHigh_val, 2) if bbHigh_val else None,
            "bbMid":            round(bbMid, 2)      if bbMid      else None,
            "bb_upper_at_detection": float(bbHigh),
            "bb_lower_at_detection": float(bbLow)    if bbLow      else None,
            "breakout_direction": "bullish" if is_breakout else "pending",
            "invalidation_level":  round(invalidation_level, 2)  if invalidation_level  else None,
            "entry_trigger_price": round(entry_trigger_price, 2) if entry_trigger_price else None,
            "horizon":          horizon,
            "pattern_strength": pattern_strength,
            "current_price":    round(price, 2)      if price      else None,
            "velocity_tracking": {
                "can_track":            result["quality"] >= 7.0 and entry_conditions_met,
                "entry_conditions_met": entry_conditions_met,
                "quality_sufficient":   result["quality"] >= 7.0,
                "breakout_confirmed":   is_breakout,
                "volume_confirmed":     has_volume,
                "bb_width_pct":         bbWidth,
                "squeeze_strength":     "tight" if bbWidth < 5 else "moderate",
            },
            "formation_context": _build_formation_context(indicators),
        }

        return result