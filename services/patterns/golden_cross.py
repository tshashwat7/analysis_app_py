import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern

class GoldenDeathCross(BasePattern):
    """
    Detects Golden Cross (50 SMA > 200 SMA) and Death Cross (50 SMA < 200 SMA).
    Crucial for long-term trend confirmation.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "golden_cross"
        self.horizons_supported = ["short_term", "long_term", "multibagger"]

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        # Guard: Need at least 200 rows for the 200 MA
        if df is None or len(df) < 200: 
            return result

        # 1. Determine Dynamic Keys based on Horizon
        key_50, key_200 = "ema_50", "ema_200"
        
        if horizon == "long_term":
            key_50, key_200 = "wma_40", "wma_50"
        elif horizon == "multibagger":
            key_50, key_200 = "mma_6", "mma_12"

        # 2. Get Current Values (From Indicators Dict)
        ma50_curr = self._get_val(indicators, key_50)
        ma200_curr = self._get_val(indicators, key_200)
        
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
            ma50_prev = close.rolling(w50).mean().iloc[-2]
            ma200_prev = close.rolling(w200).mean().iloc[-2]
            
            # FIX: Check for NaNs (e.g. if history is exactly 200 but window needs more)
            if pd.isna(ma50_prev) or pd.isna(ma200_prev):
                return result
            
        except Exception as e:
            self.log_debug(f"[GoldenCross] MA calc error: {e}")
            return result 

        if None in (ma50_curr, ma200_curr, ma50_prev, ma200_prev):
            return result

        # 4. Logic
        # Golden Cross: Prev 50 < Prev 200 AND Curr 50 > Curr 200
        if ma50_prev < ma200_prev and ma50_curr > ma200_curr:
            result["found"] = True
            result["score"] = self._normalize_score(90) # High impact event
            result["quality"] = 9.0
            result["desc"] = "Golden Cross (Bullish Trend Change)"
            result["meta"] = {"type": "bullish", "ma50": ma50_curr, "ma200": ma200_curr}

        # Death Cross: Prev 50 > Prev 200 AND Curr 50 < Curr 200
        elif ma50_prev > ma200_prev and ma50_curr < ma200_curr:
            result["found"] = True
            result["score"] = self._normalize_score(90)
            result["quality"] = 9.0
            result["desc"] = "Death Cross (Bearish Trend Change)"
            result["meta"] = {"type": "bearish", "ma50": ma50_curr, "ma200": ma200_curr}

        return result