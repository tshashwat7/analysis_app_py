import numpy as np
import pandas as pd
from typing import Dict, Any
from services.data_fetch import _safe_float
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class CupHandlePattern(BasePattern):
    """
    Detects William O'Neil's Cup and Handle Pattern.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "cupHandle"
        # Configurable Parameters
        self.min_cup_len = self.config.get("min_cup_len", 20)
        self.max_cup_depth = self.config.get("max_cup_depth", 0.50)
        self.require_volume = self.config.get("require_volume", False)
        self.handle_len = self.config.get("handle_len", 5)

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        # Need history: Cup Length + Handle Length + Buffer
        min_history = self.min_cup_len + self.handle_len + 5
        if df is None or len(df) < min_history: return result
        
        # Working with numpy arrays for speed and positional accuracy
        window = df.tail(60).copy()
        highs = window["High"].values
        lows = window["Low"].values
        closes = window["Close"].values
        volumes = window["Volume"].values
        
        # 1. Left Rim Search (First half of window)
        search_split = 30
        rim_left_idx = np.argmax(highs[:search_split])
        rim_left_val = highs[rim_left_idx]
        if rim_left_val <= 0: return result # Avoid Div/0
        
        # 2. Bottom Search
        # Must be at least 15 bars after left rim
        cup_end_search = len(highs) - self.handle_len
        if (cup_end_search - rim_left_idx) < 15: 
            return result
            
        cup_bottom_rel = np.argmin(lows[rim_left_idx:cup_end_search])
        cup_bottom_idx = rim_left_idx + cup_bottom_rel
        cup_bottom_val = lows[cup_bottom_idx]
        
        # 3. Right Rim Search
        # Must be at least 5 bars after bottom
        if (cup_end_search - cup_bottom_idx) < 5: 
            return result
            
        right_rel = np.argmax(highs[cup_bottom_idx:cup_end_search])
        rim_right_idx = cup_bottom_idx + right_rel
        rim_right_val = highs[rim_right_idx]
        
        # --- GEOMETRY VALIDATION ---
        
        # A. Depth
        cup_depth_pct = (rim_left_val - cup_bottom_val) / rim_left_val
        if cup_depth_pct > self.max_cup_depth: 
            self.log_debug(f"Cup too deep: {cup_depth_pct:.2f}")
            return result
        if cup_depth_pct < 0.10: 
            return result
            
        # B. Rim Alignment (15% tolerance)
        if not (rim_left_val * 0.85 <= rim_right_val <= rim_left_val * 1.15):
            self.log_debug("Rims misaligned")
            return result
            
        # --- HANDLE LOGIC ---
        handle_highs = highs[-self.handle_len:]
        handle_lows = lows[-self.handle_len:]
        
        # Handle must stay in upper half of cup
        mid_point = (rim_right_val + cup_bottom_val) / 2
        if np.min(handle_lows) < mid_point:
            self.log_debug("Handle too deep")
            return result
            
        # Breakout check
        current_close = closes[-1]
        is_breakout = current_close > rim_right_val
        is_forming = (current_close < rim_right_val) and (current_close > rim_right_val * 0.90)
        
        if is_breakout or is_forming:
            result["found"] = True
            qual = 6.0
            
            # Volume Check (Median)
            vol_bonus = 0.0
            try:
                v_handle = volumes[-self.handle_len:]
                v_cup = volumes[:-self.handle_len]
                if len(v_handle) > 0 and len(v_cup) > 0:
                    med_handle = np.nanmedian(v_handle)
                    med_cup = np.nanmedian(v_cup)
                    if not np.isnan(med_handle) and med_handle < (med_cup * 0.9):
                        vol_bonus = 2.0
            except Exception as e:
                self.log_debug(f"Volume check error: {e}")
                # Don't fail the whole pattern, just skip the volume bonus
            
            if self.require_volume and vol_bonus == 0:
                return result # Strict mode fails here
                
            qual += vol_bonus
            
            if is_breakout:
                result["desc"] = "Cup & Handle Breakout"
                qual += 2.0
            else:
                result["desc"] = "Cup & Handle (Forming)"
                
            result["quality"] = min(qual, 10.0)
            result["score"] = self._normalize_score(qual * 10)
            
            entry_conditions_met = is_breakout and vol_bonus > 0

            result["meta"] = {
                "depth_pct": _safe_float(round(cup_depth_pct * 100, 1)),
                "rim_level": _safe_float(round(rim_right_val, 2)),
                "handle_low": _safe_float(round(np.min(handle_lows), 2)),
                "handle_high": _safe_float(round(np.max(handle_highs), 2)),
                "coords": {
                    "left_rim_idx": int(rim_left_idx),
                    "bottom_idx": int(cup_bottom_idx),
                    "right_rim_idx": int(rim_right_idx)
                },
                "age_candles": 60 - rim_left_idx,
                "formation_timestamp": window.index[rim_left_idx].isoformat(),
                "cup_duration_candles": rim_right_idx - rim_left_idx,
                # 🆕 Pattern-Specific Velocity Tracking
                "velocity_tracking": {
                    "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient": result["quality"] >= 7.0,
                    "breakout_confirmed": is_breakout,
                    "volume_confirmed": vol_bonus > 0
                },
                # 🆕 Formation Context (Generic)
                "formation_context": _build_formation_context(indicators)
            }

            # Calculate handle depth percentage
            handle_depth_pct = ((rim_right_val - np.min(handle_lows)) / rim_right_val) * 100 if rim_right_val > 0 else 0

            # Invalidation level (varies by horizon)
            if horizon == "short_term":
                invalidation_level = np.min(handle_lows) * 0.97
            elif horizon == "long_term":
                invalidation_level = np.min(handle_lows) * 0.95
            else:
                invalidation_level = np.min(handle_lows) * 0.97  # default

            # Entry trigger is rim level
            entry_trigger_price = rim_right_val

            # Pattern strength
            pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

            # Cup shape quality assessment
            cup_shape_quality = "U-shaped" if cup_depth_pct < 35 else "irregular"

            result["meta"].update({
                "bar_index": len(df),
                # Handle Metrics (Critical for invalidation)
                "handle_depth_pct": _safe_float(round(handle_depth_pct, 2)),
                "handle_low": float(np.min(handle_lows)), 
                "rim_level": float(rim_right_val),
                # Analytics
                "cup_depth_pct": round(cup_depth_pct, 2),
                # Entry/Exit Levels todo delete
                "invalidation_level": _safe_float(round(invalidation_level, 2)),
                "entry_trigger_price": _safe_float(round(entry_trigger_price, 2)),
                # Quality Assessments
                "cup_shape_quality": cup_shape_quality,
                "handle_quality": "tight" if handle_depth_pct < 10 else "loose",
                "volume_pattern": "decreasing" if vol_bonus > 0 else "stable",
                
                # Universal Fields
                "horizon": horizon,
                "pattern_strength": pattern_strength,
                "current_price": _safe_float(round(current_close, 2))
            })
            
        return result