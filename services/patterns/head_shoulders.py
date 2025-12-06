import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

class HeadShouldersPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "head_shoulders"

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        
        if df is None or len(df) < 60 or find_peaks is None: return result

        # --- HEAD AND SHOULDERS (BEARISH) ---
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        
        peaks, _ = find_peaks(highs, distance=5)
        
        if len(peaks) >= 3:
            # Look at last 3 peaks
            p3, p2, p1 = peaks[-1], peaks[-2], peaks[-3]
            
            ls_val = highs[p1] # Left Shoulder
            head_val = highs[p2] # Head
            rs_val = highs[p3] # Right Shoulder
            
            # Geometry Rule: Head must be higher than shoulders
            if head_val > ls_val and head_val > rs_val:
                # Symmetry Rule: Shoulders roughly equal (15% tolerance)
                if 0.85 <= (ls_val / rs_val) <= 1.15:
                    
                    # Find Neckline (lowest point between shoulders)
                    neckline = np.min(lows[p1:p3])
                    
                    # Check Breakdown
                    if closes[-1] < neckline:
                        height = head_val - neckline
                        target = neckline - height
                        
                        result["found"] = True
                        result["score"] = self._normalize_score(85)
                        result["quality"] = 9.0
                        result["desc"] = "Head & Shoulders Breakdown"
                        result["meta"] = {
                            "type": "bearish",
                            "neckline": round(neckline, 2),
                            "target": round(target, 2)
                        }
                        return result

        # (Optional: You can implement Inverse H&S here using find_peaks(-lows))
        
        return result