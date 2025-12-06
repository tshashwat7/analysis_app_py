from typing import Dict, Any
import pandas as pd
from services.patterns.base import BasePattern

class BollingerSqueeze(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "bollinger_squeeze"
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
        bb_width = self._get_val(indicators, "bb_width")
        bb_high = self._get_val(indicators, "bb_high")
        price = self._get_val(indicators, "price")
        
        # Default Output
        result = {
            "found": False,
            "score": 0,
            "quality": 0,
            "meta": {}
        }
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)

        # FIX 1: Explicit None check to handle valid 0.0 values
        if bb_width is None or bb_high is None or price is None:
            return result

        # 2. Convert BB Width to decimal if it's percentage (e.g., 5.0 vs 0.05)
        # Your indicator engine returns bb_width as Percentage (e.g., 2.5), so we adjust threshold
        # Assuming squeeze_threshold=0.10 means 10% width
        is_squeezing = bb_width <= (self.squeeze_threshold * 100) 

        # 3. Detect Breakout (Price > Upper Band)
        is_breakout = price > bb_high

        if is_squeezing:
            result["found"] = True
            result["quality"] = 8.0
            raw_score = 75
            result["meta"] = {"state": "SQUEEZE_ON", "width": bb_width}

            if is_breakout:
                result["quality"] = 10.0
                raw_score = 95
                result["meta"]["state"] = "SQUEEZE_BREAKOUT"
                result["desc"] = "Vol Squeeze + Breakout"
            else:
                result["desc"] = "Volatility Squeeze (Waiting)"
            
            # FIX 3: Normalize Score
            result["score"] = self._normalize_score(raw_score)

        return result

    def _get_val(self, data, key):
        """Helper to extract value from your complex indicator dicts"""
        if key not in data: return None
        item = data[key]
        
        # FIX 2: Prioritize 'value' which is always numeric
        if isinstance(item, dict):
            val = item.get("value")
            # Fallback to raw ONLY if it's numeric (e.g. legacy data)
            if val is None:
                raw = item.get("raw")
                if isinstance(raw, (int, float)):
                    return raw
            return val
            
        return item