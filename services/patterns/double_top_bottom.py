import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class DoubleTopBottom(BasePattern):
    """
    Detects Double Tops and Bottoms using pure Numpy (No SciPy).
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "doubleTopBottom"
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
                    
                    # ✅ FIX: Use p1 directly (it's a double top, so bearish)
                    first_point_index = p1
                    
                    entry_conditions_met = current_price < neckline

                    result["meta"] = {
                        "type": "bearish",
                        "neckline": round(neckline, 2),
                        "target": round(target, 2),
                        "age_candles": 60 - first_point_index,
                        "formation_timestamp": window.index[first_point_index].isoformat(),
                        "pattern_duration_candles": abs(p2 - p1),
                        # 🆕 Pattern-Specific Velocity Tracking
                        "velocity_tracking": {
                            "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                            "entry_conditions_met": entry_conditions_met,
                            "quality_sufficient": result["quality"] >= 7.0,
                            "breakdown_confirmed": True  # Already checked in detection logic
                        },
                        # 🆕 Formation Context (Generic)
                        "formation_context": _build_formation_context(indicators)
                    }
                    # Calculate peak similarity
                    peak_similarity = abs((price2 - price1) / price1)

                    # Invalidation level (varies by horizon)
                    if horizon == "intraday":
                        invalidation_level = neckline * 0.998
                    elif horizon == "short_term":
                        invalidation_level = neckline * 0.995
                    else:
                        invalidation_level = neckline * 0.99

                    # Entry trigger is neckline
                    entry_trigger_price = neckline

                    # Pattern strength
                    pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

                    # ADD TO META (Double Top):
                    result["meta"].update({
                        # Pattern Quality Metrics
                        "peak_similarity": round(peak_similarity, 4),
                        "pattern_quality": "strong" if peak_similarity <= 0.02 else "moderate",
                        # Raw Anchors
                        "neckline": float(neckline),
                        "peak_1": float(price1), # Useful for invalidation if price breaks above peaks (for Double Top)
                        
                        # Analytics
                        "peak_similarity": round(peak_similarity, 4),
                        "pattern_height_pct": round(((price2 - neckline) / neckline) * 100, 2),
                        # Entry/Exit Levels
                        "invalidation_level": round(invalidation_level, 2),
                        "entry_trigger_price": round(entry_trigger_price, 2),
                        
                        # Universal Fields
                        "horizon": horizon,
                        "pattern_strength": pattern_strength,
                        "current_price": round(current_price, 2)
                    })
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
                    
                    # ✅ FIX: Use t1 directly (it's a double bottom, so bullish)
                    first_point_index = t1
                    
                    entry_conditions_met = current_price > neckline

                    result["meta"] = {
                        "type": "bullish",
                        "neckline": round(neckline, 2),
                        "target": round(target, 2),
                        "age_candles": 60 - first_point_index,
                        "formation_timestamp": window.index[first_point_index].isoformat(),
                        "pattern_duration_candles": abs(t2 - t1),
                        # 🆕 Pattern-Specific Velocity Tracking
                        "velocity_tracking": {
                            "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                            "entry_conditions_met": entry_conditions_met,
                            "quality_sufficient": result["quality"] >= 7.0,
                            "breakout_confirmed": True  # Already checked in detection logic
                        },
                        # 🆕 Formation Context (Generic)
                        "formation_context": _build_formation_context(indicators)
                    }
                    # Calculate peak similarity
                    peak_similarity = abs((price2 - price1) / price1)

                    # Invalidation level (varies by horizon)
                    if horizon == "intraday":
                        invalidation_level = neckline * 0.998
                    elif horizon == "short_term":
                        invalidation_level = neckline * 0.995
                    else:
                        invalidation_level = neckline * 0.99

                    # Entry trigger is neckline
                    entry_trigger_price = neckline

                    # Pattern strength
                    pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

                    # ADD TO META (Double Top):
                    result["meta"].update({
                        "bar_index": len(df),
                        # Pattern Quality Metrics
                        "peak_similarity": round(peak_similarity, 4),
                        "pattern_quality": "strong" if peak_similarity <= 0.02 else "moderate",
                        "neckline": float(neckline),  # The valley is the neckline in a Double Top
                        "peak_1": float(price1),        # Left Peak
                        "peak_2": float(price2),        # Right Peak
                        
                        # ANALYTICS (Optional, for velocity tracking)
                        "pattern_height_pct": round(((price2 - neckline) / neckline) * 100, 2),
                        "peak_similarity": round(peak_similarity, 4),
                        # Entry/Exit Levels
                        "invalidation_level": round(invalidation_level, 2),
                        "entry_trigger_price": round(entry_trigger_price, 2),
                        
                        # Universal Fields
                        "horizon": horizon,
                        "pattern_strength": pattern_strength,
                        "current_price": round(current_price, 2)
                    })
                    return result

        return result