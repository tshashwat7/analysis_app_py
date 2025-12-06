import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

class EngulfingPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "engulfing_pattern"

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if df is None or len(df) < 3: return result
        
        # Last 2 candles
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Body sizes
        prev_open, prev_close = prev["Open"], prev["Close"]
        curr_open, curr_close = curr["Open"], curr["Close"]
        
        # Bullish Engulfing: Prev Red, Curr Green, Curr Envelopes Prev
        if prev_close < prev_open: # Prev Red
            if curr_close > curr_open: # Curr Green
                if curr_open <= prev_close and curr_close >= prev_open:
                    
                    # Volume Confirmation
                    vol_boost = curr["Volume"] > prev["Volume"] * 1.1
                    
                    result["found"] = True
                    qual = 7.0 + (1.0 if vol_boost else 0)
                    result["quality"] = qual
                    result["score"] = self._normalize_score(qual * 10)
                    result["desc"] = "Bullish Engulfing"
                    result["meta"] = {"type": "bullish", "vol_boost": vol_boost}
                    return result

        # Bearish Engulfing
        if prev_close > prev_open: # Prev Green
            if curr_close < curr_open: # Curr Red
                if curr_open >= prev_close and curr_close <= prev_open:
                    
                    vol_boost = curr["Volume"] > prev["Volume"] * 1.1
                    
                    result["found"] = True
                    qual = 7.0 + (1.0 if vol_boost else 0)
                    result["quality"] = qual
                    result["score"] = self._normalize_score(qual * 10)
                    result["desc"] = "Bearish Engulfing"
                    result["meta"] = {"type": "bearish", "vol_boost": vol_boost}
                    
        return result