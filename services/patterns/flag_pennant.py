import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class FlagPennantPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "flagPennant"
        # Configurable windows
        self.pole_days = self.config.get("pole_back", 15)
        self.flag_days = self.config.get("flag_back", 5)
    
    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        required_len = self.pole_days + self.flag_days
        if df is None or len(df) < required_len: return result
        
        closes = df["Close"]
        
        # FIX 1: Dynamic Windows
        pole_start_price = closes.iloc[-self.pole_days]
        pole_end_price = closes.iloc[-self.flag_days]
        
        # Calculate Pole Return
        if pole_start_price == 0: return result
        pole_return = (pole_end_price - pole_start_price) / pole_start_price
        
        is_strong_pole = pole_return > 0.05 # 5% move
        
        # ✅ NEW: Calculate Flag Boundaries (for invalidation)
        flag_highs = df["High"].iloc[-self.flag_days:]
        flag_lows = df["Low"].iloc[-self.flag_days:]
        
        flag_low = flag_lows.min()   # Support level of flag
        flag_high = flag_highs.max() # Resistance level of flag
        
        # Calculate Flag Drift
        flag_start_price = closes.iloc[-self.flag_days]
        flag_end_price = closes.iloc[-1]
        
        if flag_start_price == 0: return result
        flag_return = (flag_end_price - flag_start_price) / flag_start_price
        
        # Flag logic: Slight drift (-3% to +1%)
        is_tight_flag = (-0.03 < flag_return < 0.01)
        
        # FIX 2: Trend Check (Flag must be in uptrend)
        # Dynamic lookup for "Fast MA" based on horizon
        trend_ma_val = self._get_val(indicators, "maFast")
        is_uptrend = False
        if trend_ma_val and closes.iloc[-1] >= trend_ma_val:
            is_uptrend = True # Price at or above fast MA = Strong flag
        
        # Volume Check
        vols = df["Volume"]
        vol_pole = vols.iloc[-(self.pole_days):-(self.flag_days)].mean()
        vol_flag = vols.iloc[-(self.flag_days):].mean()
        is_vol_drying = vol_flag < vol_pole
        
        if is_strong_pole and is_tight_flag and is_uptrend:
            result["found"] = True
            qual = 6.0
            if is_vol_drying: qual += 2.0
            
            # Breakout today?
            if closes.iloc[-1] > closes.iloc[-2] * 1.01:
                qual += 2.0
                result["desc"] = "Bull Flag Breakout"
            else:
                result["desc"] = "Bull Flag (Forming)"
                
            result["quality"] = min(qual, 10.0)
            result["score"] = self._normalize_score(qual * 10)
            # ✅ ADD PATTERN AGE TRACKING
            # Calculate formation index (pole started N bars ago)
            formation_index = len(df) - self.pole_days
            # Entry Conditions
            is_current_breakout = closes.iloc[-1] > closes.iloc[-2] * 1.01
            entry_conditions_met = is_vol_drying and is_current_breakout
            # Calculate metrics
            pole_strength = "strong" if pole_return > 0.08 else "moderate" if pole_return > 0.05 else "weak"
            flag_tightness_val = abs(flag_return)
            flag_angle = "ascending" if flag_return > 0.01 else "descending" if flag_return < -0.01 else "horizontal"
            # Invalidation level (varies by horizon)
            if horizon == "intraday": invalidation_level = flag_low * 0.998
            elif horizon == "short_term": invalidation_level = flag_low * 0.995
            else: invalidation_level = flag_low * 0.99
            # Entry trigger
            entry_trigger_price = flag_high
            # Pattern strength
            pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

            _fi = max(0, formation_index)  # ✅ P1-1 FIX: safe index fallback
            result["meta"] = {
                "age_candles": len(df) - formation_index,
                "formation_time": float(df.index[_fi].timestamp()),
                "formation_timestamp": df.index[_fi].isoformat(),
                "pole_length": self.pole_days,
                "flag_length": self.flag_days,
                "bar_index": len(df),
                "type": "bullish",
                # Entry/Exit Levels
                "flag_low": round(flag_low, 2),
                "flag_high": round(flag_high, 2),
                "pole_base": float(pole_start_price),
                
                # Analytics
                "pole_gain_pct": round(pole_return * 100, 2),
                "flag_drift_pct": round(flag_return * 100, 1),
                "flag_tightness": round(flag_tightness_val, 4),
                "flag_duration_candles": self.flag_days,

                "invalidation_level": round(invalidation_level, 2),
                "entry_trigger_price": round(entry_trigger_price, 2),
                
                # Pattern Quality Metrics
                "pole_strength": pole_strength,
                "flag_angle": flag_angle,
                
                # Universal Fields
                "horizon": horizon,
                "pattern_strength": pattern_strength,
                "current_price": round(closes.iloc[-1], 2),
                # 🆕 Pattern-Specific Velocity Tracking
                "velocity_tracking": {
                    "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient": result["quality"] >= 7.0,
                    "breakout_confirmed": is_current_breakout,
                    "volume_dry": is_vol_drying
                },
                # 🆕 Formation Context (Generic)
                "formation_context": _build_formation_context(indicators)
            }
            
        return result
