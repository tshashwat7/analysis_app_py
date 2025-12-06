import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

class ThreeLineStrikePattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "three_line_strike"

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
            
        if df is None or len(df) < 5: return result
        
        # Need last 4 candles
        c = df.iloc[-4:].copy() # Index 0,1,2,3 (3 is today)
        
        # Helper for candle color
        # 1=Green, -1=Red
        opens = c["Open"].values
        closes = c["Close"].values
        lows = c["Low"].values
        highs = c["High"].values
        
        # CHECK BULLISH STRIKE (Correction of trend)
        # Pattern: 3 Bearish candles, then 1 Bullish that engulfs all 3
        
        # 1. First 3 are Red
        is_3_red = (closes[0] < opens[0]) and (closes[1] < opens[1]) and (closes[2] < opens[2])
        
        # 2. Making Lower Lows
        is_lower_lows = (lows[1] < lows[0]) and (lows[2] < lows[1])
        
        if is_3_red and is_lower_lows:
            # 3. The Strike Candle (Today) must be Green
            is_strike_green = closes[3] > opens[3]
            
            # 4. Engulfing Logic:
            # Open of Strike < Close of 3rd candle (Gaps down or opens low)
            # Close of Strike > Open of 1st candle (Engulfs the whole sequence)
            is_engulfing = (opens[3] < closes[2]) and (closes[3] > opens[0])
            
            if is_strike_green and is_engulfing:
                result["found"] = True
                result["score"] = 90 # High prob pattern
                result["quality"] = 9.0
                result["desc"] = "Bullish 3-Line Strike"
                result["meta"] = {"type": "Bullish"}
                return result

        # CHECK BEARISH STRIKE
        # Pattern: 3 Green candles, then 1 Red that engulfs all 3
        is_3_green = (closes[0] > opens[0]) and (closes[1] > opens[1]) and (closes[2] > opens[2])
        is_higher_highs = (highs[1] > highs[0]) and (highs[2] > highs[1])
        
        if is_3_green and is_higher_highs:
            is_strike_red = closes[3] < opens[3]
            # Open of Strike > Close of 3rd
            # Close of Strike < Open of 1st
            is_engulfing = (opens[3] > closes[2]) and (closes[3] < opens[0])
            
            if is_strike_red and is_engulfing:
                result["found"] = True
                result["score"] = 90
                result["quality"] = 9.0
                result["desc"] = "Bearish 3-Line Strike"
                result["meta"] = {"type": "Bearish"}
        
        return result