import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class ThreeLineStrikePattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "threeLineStrike"

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
            
        if df is None or len(df) < 5: return result
        
        # Need last 4 candles
        c = df.iloc[-4:].copy()
        
        opens = c["Open"].values
        closes = c["Close"].values
        lows = c["Low"].values
        highs = c["High"].values
        
        # CHECK BULLISH STRIKE
        is_3_red = (closes[0] < opens[0]) and (closes[1] < opens[1]) and (closes[2] < opens[2])
        is_lower_lows = (lows[1] < lows[0]) and (lows[2] < lows[1])

        if is_3_red and is_lower_lows:
            is_strike_green = closes[3] > opens[3]
            is_engulfing = (opens[3] < closes[2]) and (closes[3] > opens[0])
            
            if is_strike_green and is_engulfing:
                result["found"] = True
                result["score"] = self._normalize_score(90)
                result["quality"] = 9.0
                result["desc"] = "Bullish 3-Line Strike"

                # ✅ NOW calculate variables (after we know pattern exists)
                strike_low = lows[3]
                strike_high = highs[3]
                strike_open = opens[3]
                strike_close = closes[3]
                pattern_low = lows.min()
                pattern_high = highs.max()
                prior_range = float(np.max(highs[:3]) - np.min(lows[:3]))
                
                # ✅ Calculate body ratio
                strike_candle_body = abs(closes[3] - opens[3]) / ((highs[3] - lows[3]) or 1)
                
                # ✅ Calculate invalidation level
                if horizon == "intraday":
                    invalidation_level = strike_low * 0.995
                elif horizon == "short_term":
                    invalidation_level = strike_low * 0.99
                else:
                    invalidation_level = strike_low * 0.98

                # Pattern strength
                pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate"
                strike_strength = "strong" if strike_candle_body >= 0.7 else "moderate" if strike_candle_body >= 0.5 else "weak"
                
                rvol = self._get_val(indicators, "rvol")
                reversal_confidence = "high" if strike_candle_body >= 0.7 and (rvol and rvol >= 1.3) else "moderate"
                
                entry_conditions_met = True

                result["meta"] = {
                    "bar_index": len(df),
                    "strike_candle_body": round(strike_candle_body, 3),
                    "invalidation_level": round(invalidation_level, 2),
                    "entry_trigger_price": round(strike_close, 2),
                    "strike_strength": strike_strength,
                    "reversal_confidence": reversal_confidence,
                    "pattern_quality": "strong" if strike_candle_body >= 0.7 else "moderate",
                    "horizon": horizon,
                    "pattern_strength": pattern_strength,
                    "current_price": round(closes[3], 2),
                    "type": "bullish",
                    "strike_low": round(strike_low, 2),
                    "strike_high": round(strike_high, 2),
                    "strike_open": round(strike_open, 2),
                    "strike_close": round(strike_close, 2),
                    "pattern_low": round(pattern_low, 2),
                    "pattern_high": round(pattern_high, 2),
                    "prior_range": round(prior_range, 2),
                    "age_candles": 1,
                    "formation_time": float(df.index[-1].timestamp()),
                    "formation_timestamp": df.index[-1].isoformat(),
                    "strike_candle_body_pct": round(strike_candle_body, 3),
                    "velocity_tracking": {
                        "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                        "entry_conditions_met": entry_conditions_met,
                        "quality_sufficient": result["quality"] >= 7.0,
                        "strike_confirmed": True,
                        "pattern_type": "bullish"
                    },
                    "formation_context": _build_formation_context(indicators)
                }
                return result

        # CHECK BEARISH STRIKE
        is_3_green = (closes[0] > opens[0]) and (closes[1] > opens[1]) and (closes[2] > opens[2])
        is_higher_highs = (highs[1] > highs[0]) and (highs[2] > highs[1])
        
        if is_3_green and is_higher_highs:
            is_strike_red = closes[3] < opens[3]
            is_engulfing = (opens[3] > closes[2]) and (closes[3] < opens[0])
            
            if is_strike_red and is_engulfing:
                result["found"] = True
                result["score"] = self._normalize_score(90)
                result["quality"] = 9.0
                result["desc"] = "Bearish 3-Line Strike"
                
                # ✅ NOW calculate variables (after we know pattern exists)
                strike_low = lows[3]
                strike_high = highs[3]
                strike_open = opens[3]
                strike_close = closes[3]
                pattern_low = lows.min()
                pattern_high = highs.max()
                prior_range = float(np.max(highs[:3]) - np.min(lows[:3]))
                
                # ✅ Calculate body ratio
                strike_candle_body = abs(closes[3] - opens[3]) / ((highs[3] - lows[3]) or 1)
                
                # ✅ Calculate invalidation level (REVERSED for bearish)
                if horizon == "intraday":
                    invalidation_level = strike_high * 1.005  # ✅ Use high for bearish
                elif horizon == "short_term":
                    invalidation_level = strike_high * 1.01
                else:
                    invalidation_level = strike_high * 1.02

                pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate"
                strike_strength = "strong" if strike_candle_body >= 0.7 else "moderate" if strike_candle_body >= 0.5 else "weak"
                
                rvol = self._get_val(indicators, "rvol")
                reversal_confidence = "high" if strike_candle_body >= 0.7 and (rvol and rvol >= 1.3) else "moderate"
                
                entry_conditions_met = True

                result["meta"] = {
                    "bar_index": len(df),
                    "strike_candle_body": round(strike_candle_body, 3),
                    "invalidation_level": round(invalidation_level, 2),
                    "entry_trigger_price": round(strike_close, 2),
                    "strike_strength": strike_strength,
                    "reversal_confidence": reversal_confidence,
                    "pattern_quality": "strong" if strike_candle_body >= 0.7 else "moderate",
                    "horizon": horizon,
                    "pattern_strength": pattern_strength,
                    "current_price": round(closes[3], 2),
                    "type": "bearish",
                    "strike_low": round(strike_low, 2),
                    "strike_high": round(strike_high, 2),
                    "strike_open": round(strike_open, 2),
                    "strike_close": round(strike_close, 2),
                    "pattern_low": round(pattern_low, 2),
                    "pattern_high": round(pattern_high, 2),
                    "prior_range": round(prior_range, 2),
                    "age_candles": 1,
                    "formation_time": float(df.index[-1].timestamp()),
                    "formation_timestamp": df.index[-1].isoformat(),
                    "strike_candle_body_pct": round(strike_candle_body, 3),
                    "velocity_tracking": {
                        "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                        "entry_conditions_met": entry_conditions_met,
                        "quality_sufficient": result["quality"] >= 7.0,
                        "strike_confirmed": True,
                        "pattern_type": "bearish"
                    },
                    "formation_context": _build_formation_context(indicators)
                }
                return result
        
        return result
