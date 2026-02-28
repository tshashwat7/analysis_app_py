from typing import Dict, Any
from unittest import result
import pandas as pd
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class BollingerSqueeze(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "bollingerSqueeze"
        # Defaults if config is empty
        self.squeeze_threshold = self.config.get("squeeze_threshold", 0.10) # 10% BB Width is tight
        self.breakout_confirmation = 0.02 # 2% above band

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        """
        Logic: 
        1. Squeeze: BB Width is very low (Volatility Compression).
        2. Breakout: Price crosses Upper Band (Momentum).
        """
        # 1. Gather Data (Safely)
        bbWidth = self._get_val(indicators, "bbWidth")
        bbHigh = self._get_val(indicators, "bbHigh")
        price = self._get_val(indicators, "price")
        raw_score = 0
        # Default Output
        result = {
            "found": False,
            "score": 0,
            "quality": 0,
            "meta": {}
        }
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)

        # Explicit None check to handle valid 0.0 values
        if bbWidth is None or bbHigh is None or price is None:
            return result

        # 2. Convert BB Width to decimal if it's percentage (e.g., 5.0 vs 0.05)
        # Your indicator engine returns bbWidth as Percentage (e.g., 2.5), so we adjust threshold
        # Assuming squeeze_threshold=0.10 means 10% width
        is_squeezing = bbWidth <= (self.squeeze_threshold * 100) 

        # 3. Detect Breakout (Price > Upper Band)
        is_breakout = price > bbHigh
        if is_breakout:
            result["quality"] = 10.0
            raw_score = 95
            result["desc"] = "Vol Squeeze + Breakout"
            state = "SQUEEZE_BREAKOUT"
        elif is_squeezing:
            result["desc"] = "Volatility Squeeze (Waiting)"
            state = "SQUEEZE_ON"
        else:
            # Neither squeezing nor breaking out — no pattern
            return result
        
        result["score"] = self._normalize_score(raw_score)
        result["found"] = True
        
        # 🆕 Pattern-specific entry conditions
        bbLow = self._get_val(indicators, "bbLow")
        bbHigh_val = self._get_val(indicators, "bbHigh")
        bbMid = self._get_val(indicators, "bbMid")
        rvol = self._get_val(indicators, "rvol")
        has_volume = rvol and rvol >= 1.3 if rvol else False
        entry_conditions_met = is_breakout and has_volume
        estimated_age = 7

        # Calculate invalidation level (varies by horizon)
        if horizon == "intraday":
            invalidation_level = bbLow if bbLow else None
        elif horizon == "short_term":
            invalidation_level = bbLow * 0.99 if bbLow else None
        else:
            invalidation_level = bbLow if bbLow else None
        # Calculate entry trigger
        entry_trigger_price = bbHigh_val if bbHigh_val else None
        # Squeeze duration (estimate from age)
        squeeze_duration = estimated_age
        # Pattern strength
        pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

        # 🆕 Build metadata ONCE at the end
        result["meta"] = {
            "bar_index": len(df),
            "state": state,
            "width": bbWidth,
            "age_candles": estimated_age,
            "formation_timestamp": df.index[-estimated_age].isoformat() if len(df) > estimated_age else None,
            "bbLow": round(bbLow, 2) if bbLow else None,
            "bbHigh": round(bbHigh_val, 2) if bbHigh_val else None,
            "bbMid": round(bbMid, 2) if bbMid else None,
            "bb_upper_at_detection": float(bbHigh),
            "bb_lower_at_detection": float(bbLow),
            
            # Analytics
            "squeeze_duration": squeeze_duration,
            "squeeze_strength": "tight" if bbWidth < 5 else "moderate",
            # Entry/Exit Levels (Critical for order management)
            "invalidation_level": round(invalidation_level, 2) if invalidation_level else None,
            "entry_trigger_price": round(entry_trigger_price, 2) if entry_trigger_price else None,
            
            # Squeeze Metrics (Used in entry conditions)
            "squeeze_duration": squeeze_duration,
            "breakout_direction": "bullish" if is_breakout else "pending",
            
            # Universal Fields
            "horizon": horizon,
            "pattern_strength": pattern_strength,
            "current_price": round(price, 2) if price else None,
            # 🆕 Pattern-Specific Velocity Tracking
            "velocity_tracking": {
                "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                "entry_conditions_met": entry_conditions_met,
                "quality_sufficient": result["quality"] >= 7.0,
                "breakout_confirmed": is_breakout,
                "volume_confirmed": has_volume,
                "bb_width_pct": bbWidth,  # Already included above, but shown for clarity
                "squeeze_strength": "tight" if bbWidth < 5 else "moderate"
            },
            # 🆕 Formation Context (Generic)
            "formation_context": _build_formation_context(indicators)
        }

        return result