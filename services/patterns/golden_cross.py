import pandas as pd
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

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        # Guard: Need at least 2 rows to detect a crossover
        if df is None or len(df) < 200: 
            return result

        # 1. Determine Dynamic Keys based on Horizon
        # (Consistent with your signal_engine logic)
        key_50 = "ema_50" 
        key_200 = "ema_200"
        
        if horizon == "long_term":
            key_50, key_200 = "wma_40", "wma_50"
        elif horizon == "multibagger":
            key_50, key_200 = "mma_6", "mma_12"

        # 2. Get Current Values (From Indicators Dict)
        ma50_curr = self._get_val(indicators, key_50)
        ma200_curr = self._get_val(indicators, key_200)
        
        # 3. Get Previous Values (From DF, manual calc if needed)
        # Note: Your DF might not have 'ema_50' columns if they are calc'd in memory.
        # We need to calculate the previous MAs manually if columns missing.
        try:
            # Quick rolling calculation for just the last 2 points
            # This ensures we are independent of DF column names
            close = df["Close"]
            
            # Map horizon keys to simple windows for calculation
            win_50 = 50
            win_200 = 200
            
            # Simple fallback logic for window sizes
            if horizon == "long_term": win_50, win_200 = 40, 50
            elif horizon == "multibagger": win_50, win_200 = 6, 12
            
            # Calculate last 2 values
            # Using EWM for EMA, Rolling Mean for SMA/WMA approximation
            # (Strictly speaking WMA is weighted, but SMA is close enough for crossover detection if WMA col missing)
            
            # Previous 50
            ma50_prev = close.rolling(win_50).mean().iloc[-2]
            if pd.isna(ma50_prev) or pd.isna(ma200_prev):
                return result
            # Previous 200
            ma200_prev = close.rolling(win_200).mean().iloc[-2]
            
        except Exception:
            return result # Data too short or error

        if None in (ma50_curr, ma200_curr, ma50_prev, ma200_prev):
            return result

        # 4. Logic
        # Golden Cross: Prev 50 < Prev 200 AND Curr 50 > Curr 200
        if ma50_prev < ma200_prev and ma50_curr > ma200_curr:
            result["found"] = True
            result["score"] = self._normalize_score(75)
            result["quality"] = 8.0
            result["desc"] = "Golden Cross (Bullish)"
            result["meta"] = {"type": "bullish", "ma50": ma50_curr, "ma200": ma200_curr}

        # Death Cross: Prev 50 > Prev 200 AND Curr 50 < Curr 200
        elif ma50_prev > ma200_prev and ma50_curr < ma200_curr:
            result["found"] = True
            result["score"] = self._normalize_score(75)
            result["quality"] = 8.0
            result["desc"] = "Death Cross (Bearish)"
            result["meta"] = {"type": "bearish", "ma50": ma50_curr, "ma200": ma200_curr}

        return result