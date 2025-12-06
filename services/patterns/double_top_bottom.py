import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

class DoubleTopBottom(BasePattern):
    """
    Detects Double Tops and Bottoms using pure Numpy (No SciPy).
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "double_top_bottom"
        self.peak_window = 5 # Look 5 bars left/right for local extrema

    def _find_peaks_numpy(self, arr: np.array, distance: int):
        """Simple peak detection without SciPy"""
        peaks = []
        if len(arr) < distance * 2: return peaks
        
        for i in range(distance, len(arr) - distance):
            window = arr[i - distance : i + distance + 1]
            if np.argmax(window) == distance: # Center is highest
                peaks.append(i)
        return peaks

    def _find_troughs_numpy(self, arr: np.array, distance: int):
        """Simple trough detection without SciPy"""
        troughs = []
        if len(arr) < distance * 2: return troughs
        
        for i in range(distance, len(arr) - distance):
            window = arr[i - distance : i + distance + 1]
            if np.argmin(window) == distance: # Center is lowest
                troughs.append(i)
        return troughs

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        # Guard must match the tail window used below (60)
        if df is None or len(df) < 60: return result

        # Use recent window (last 60 bars)
        window = df.tail(60).copy()
        highs = window["High"].values
        lows = window["Low"].values
        closes = window["Close"].values
        current_price = closes[-1]

        # 1. Double Top (Bearish)
        peaks = self._find_peaks_numpy(highs, self.peak_window)
        
        if len(peaks) >= 2:
            p2 = peaks[-1] # Most recent peak
            p1 = peaks[-2] # Previous peak
            
            #Defensive check for empty slice
            if p2 <= p1: return result 
            
            price1 = highs[p1]
            price2 = highs[p2]
            
            # Check Level (within 3% tolerance)
            if 0.97 <= (price2 / price1) <= 1.03:
                # Find Trough (Neckline) between peaks
                # Slice logic: relative to window start
                trough_rel = np.argmin(lows[p1:p2])
                neckline = lows[p1 + trough_rel]
                
                # Check Breakdown (Price closed below neckline)
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
        troughs = self._find_troughs_numpy(lows, self.peak_window)
        
        if len(troughs) >= 2:
            t2 = troughs[-1]
            t1 = troughs[-2]
            
            # FIX 3: Defensive check for empty slice
            if t2 <= t1: return result 
            
            price1 = lows[t1]
            price2 = lows[t2]
            
            # Check Level
            if 0.97 <= (price2 / price1) <= 1.03:
                # Find Peak (Neckline) between troughs
                peak_rel = np.argmax(highs[t1:t2])
                neckline = highs[t1 + peak_rel]
                
                # Check Breakout (Price closed above neckline)
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