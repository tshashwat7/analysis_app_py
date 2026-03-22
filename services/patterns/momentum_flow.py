import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern

class MomentumFlowPattern(BasePattern):
    """
    Detects 'Flow' - continuous momentum moves (distribution or accumulation)
    characterized by consecutive candles in the same direction with increasing volume,
    even if they don't hit extreme Bollinger Band levels.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "momentumFlow"

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        # Need at least 4 candles to establish flow
        if df is None or len(df) < 5:
            return result
            
        # Analysis window (last 4 candles)
        window = df.iloc[-4:]
        
        # Check for Bearish Flow (Lower Highs + Lower Lows + Negative Close)
        bearish_flow = True
        for i in range(1, 4):
            if not (window.iloc[i]["High"] <= window.iloc[i-1]["High"] and 
                    window.iloc[i]["Low"] < window.iloc[i-1]["Low"] and
                    window.iloc[i]["Close"] < window.iloc[i]["Open"]):
                bearish_flow = False
                break
                
        # Check for Bullish Flow (Higher Highs + Higher Lows + Positive Close)
        bullish_flow = True
        for i in range(1, 4):
            if not (window.iloc[i]["High"] > window.iloc[i-1]["High"] and 
                    window.iloc[i]["Low"] >= window.iloc[i-1]["Low"] and
                    window.iloc[i]["Close"] > window.iloc[i]["Open"]):
                bullish_flow = False
                break
                
        if bearish_flow or bullish_flow:
            flow_type = "bearish" if bearish_flow else "bullish"
            
            # Volume velocity
            vols = window["Volume"].values
            vol_growth = vols[-1] / vols[0] if vols[0] > 0 else 1.0
            
            # Price velocity
            price_move_pct = abs(window.iloc[-1]["Close"] - window.iloc[0]["Open"]) / window.iloc[0]["Open"] * 100
            
            result["found"] = True
            
            # Quality based on volume confirmation and price velocity
            quality = 6.0
            if vol_growth > 1.2: quality += 1.5
            if price_move_pct > 2.0: quality += 1.5
            if vol_growth > 2.0: quality += 1.0 # Explosive volume
            
            result["quality"] = min(10.0, quality)
            result["score"] = self._normalize_score(quality * 10)
            result["desc"] = f"Momentum Flow ({flow_type.title()})"
            
            from services.patterns.utils import _build_formation_context
            
            current_price = window.iloc[-1]["Close"]
            if flow_type == "bullish":
                invalidation_level = window.iloc[0]["Low"] * 0.99
            else:
                invalidation_level = window.iloc[0]["High"] * 1.01

            result["meta"] = {
                "type": flow_type,
                "vol_growth": round(vol_growth, 2),
                "velocity_pct": round(price_move_pct, 2),
                "candles": 4,
                "age_candles": 1, # Fresh detection
                "formation_time": float(df.index[-1].timestamp()),
                "formation_timestamp": df.index[-1].isoformat(),
                "formation_context": _build_formation_context(indicators),
                # Standardization fixes
                "bar_index": len(df),
                "invalidation_level": round(invalidation_level, 2),
                "velocity_tracking": {
                    "can_track": quality >= 7.0,
                    "entry_conditions_met": True,
                    "quality_sufficient": quality >= 7.0,
                    "flow_confirmed": True
                },
                "pattern_strength": "strong" if quality >= 8.5 else "moderate" if quality >= 6.5 else "weak",
                "current_price": round(current_price, 2),
                "horizon": horizon
            }
            
        return result
