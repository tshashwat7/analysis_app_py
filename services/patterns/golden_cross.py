import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class GoldenDeathCross(BasePattern):
    """
    Detects Golden Cross (50 SMA > 200 SMA) and Death Cross (50 SMA < 200 SMA).
    Crucial for long-term trend confirmation.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "goldenCross"
        self.horizons_supported = ["short_term", "long_term", "multibagger"]

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        # Guard: Need at least 200 rows for the 200 MA
        if df is None or len(df) < 200: 
            return result

        # 1. Determine Dynamic Keys based on Horizon
        key_50, key_200 = "ema50", "ema200"
        
        if horizon == "long_term":
            key_50, key_200 = "wma40", "wma50"
        elif horizon == "multibagger":
            key_50, key_200 = "mma6", "mma12"

        # 2. Get Current Values (From Indicators Dict)
        maMid = self._get_val(indicators, "maMid")
        maSlow = self._get_val(indicators, "maSlow")
        
        # 3. Get Previous Values (Manual Calc to avoid dependency on DF columns)
        try:
            # Quick rolling calculation for just the last 2 points
            # This ensures we are independent of DF column names
            close = df["Close"]
            # Simple window mapping for fallback calc
            w50, w200 = 50, 200
            if horizon == "long_term": w50, w200 = 40, 50
            elif horizon == "multibagger": w50, w200 = 6, 12
            
            # Calculate prev values (iloc -2)
            maMid_prev = close.rolling(w50).mean().iloc[-2]
            maSlow_prev = close.rolling(w200).mean().iloc[-2]
            
            # FIX: Check for NaNs (e.g. if history is exactly 200 but window needs more)
            if pd.isna(maMid_prev) or pd.isna(maSlow_prev):
                return result
            
        except Exception as e:
            self.log_debug(f"[GoldenCross] MA calc error: {e}")
            return result 

        if None in (maMid, maSlow, maMid_prev, maSlow_prev):
            return result

        # 4. Logic: Detect Crossovers
        cross_type = None
        # Cross strength calculation
        cross_strength = abs(maMid - maSlow) / maSlow if maSlow else 0
        # Invalidation level (maMid value)
        invalidation_level = maMid if cross_type == "bullish" else maMid
        # Pattern strength
        pattern_strength = "strong"  # Golden cross is always high quality
        
        if maMid_prev < maSlow_prev and maMid > maSlow:
            cross_type = "bullish"
            result["desc"] = "Golden Cross (Bullish Trend Change)"
        elif maMid_prev > maSlow_prev and maMid < maSlow:
            cross_type = "bearish"
            result["desc"] = "Death Cross (Bearish Trend Change)"

        # 5. Build result ONCE if cross detected
        if cross_type:
            result["found"] = True
            result["score"] = self._normalize_score(90)
            result["quality"] = 9.0
            
            # 🆕 Pattern-specific entry: fresh cross IS the entry
            entry_conditions_met = True  # Cross itself is the signal
            
            # 🆕 Build metadata ONCE
            result["meta"] = {
                "bar_index": len(df),
                "type": cross_type,
                "maMid": maMid,
                "maSlow": maSlow,
                "age_candles": 1,  # Always fresh (just detected)
                "formation_timestamp": df.index[-1].isoformat(),
                "crossover_fresh": True,
                "cross_strength": round(cross_strength, 4),
                "trend_confirmation": "confirmed" if cross_strength > 0.05 else "early",
                # Entry/Exit Levels
                "invalidation_level": round(invalidation_level, 2) if invalidation_level else None,
                # Universal Fields
                "horizon": horizon,
                "pattern_strength": pattern_strength,
                "current_price": round(df["Close"].iloc[-1], 2),
                # 🆕 Pattern-Specific Velocity Tracking
                "velocity_tracking": {
                    "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient": result["quality"] >= 7.0,
                    "cross_type": cross_type,
                    "fresh_cross": True
                },
                # 🆕 Formation Context (Generic)
                "formation_context": _build_formation_context(indicators)
            }

        return result