import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

# Try importing scipy, handle failure gracefully
try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

class DoubleTopBottom(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "double_top_bottom"

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        if df is None or len(df) < 60 or find_peaks is None: return result

        # Use recent window
        window = df.tail(60)
        highs = window["High"].values
        lows = window["Low"].values
        closes = window["Close"].values
        current_price = closes[-1]

        # 1. Double Top (Bearish)
        peaks, _ = find_peaks(highs, distance=5)
        if len(peaks) >= 2:
            p2 = peaks[-1]
            p1 = peaks[-2]
            
            price1 = highs[p1]
            price2 = highs[p2]
            
            # Check Level (within 3%)
            if 0.97 <= (price2 / price1) <= 1.03:
                # Find Trough (Neckline) between peaks
                trough_idx = p1 + np.argmin(lows[p1:p2])
                neckline = lows[trough_idx]
                
                # Check Breakdown
                if current_price < neckline:
                    result["found"] = True
                    result["score"] = self._normalize_score(80)
                    result["quality"] = 8.5
                    result["desc"] = "Double Top Breakdown"
                    
                    height = price1 - neckline
                    target = neckline - height
                    result["meta"] = {
                        "type": "bearish",
                        "neckline": round(neckline, 2),
                        "target": round(target, 2)
                    }
                    return result

        # 2. Double Bottom (Bullish)
        troughs, _ = find_peaks(-lows, distance=5)
        if len(troughs) >= 2:
            t2 = troughs[-1]
            t1 = troughs[-2]
            
            price1 = lows[t1]
            price2 = lows[t2]
            
            # Check Level
            if 0.97 <= (price2 / price1) <= 1.03:
                # Find Peak (Neckline)
                peak_idx = t1 + np.argmax(highs[t1:t2])
                neckline = highs[peak_idx]
                
                # Check Breakout
                if current_price > neckline:
                    result["found"] = True
                    result["score"] = self._normalize_score(80)
                    result["quality"] = 8.5
                    result["desc"] = "Double Bottom Breakout"
                    
                    height = neckline - price1
                    target = neckline + height
                    result["meta"] = {
                        "type": "bullish",
                        "neckline": round(neckline, 2),
                        "target": round(target, 2)
                    }
                    return result

        return result