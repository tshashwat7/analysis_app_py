import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context, _classify_volatility

class DarvasBoxPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "darvasBox"
        # Configurable lookbacks (default to 50 for robust trend context)
        self.lookback = self.config.get("lookback", 50) 
        self.box_length = self.config.get("box_length", 5)

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        # FIX 1: Dynamic Lookback check
        if df is None or len(df) < self.lookback:
            return result

        # Work on the tail based on config
        recent = df.tail(self.lookback).copy()
        current_close = recent["Close"].iloc[-1]
        
        # Find the highest high in the earlier part of the window (Box Top candidate)
        # We look back excluding the most recent "box_length" days to find established resistance
        look_start = -(self.box_length * 2)
        look_end = -self.box_length
        
        box_high = recent["High"].iloc[look_start:look_end].max()
        box_low = recent["Low"].iloc[look_start:look_end].min()
        
        # Valid Box Criteria:
        # 1. Price recently consolidated inside this range (Last 5 days)
        last_n_highs = recent["High"].iloc[-self.box_length:-1]
        last_n_lows = recent["Low"].iloc[-self.box_length:-1]
        
        # FIX 2: Tighter Consolidation (1% tolerance instead of 2%)
        is_consolidating = (last_n_highs.max() <= box_high * 1.01) and \
                           (last_n_lows.min() >= box_low * 0.99)
                           
        # Breakout Check
        is_breakout = current_close > box_high
        
        # FIX 3: Volume vs Recent 5-day average (Sharper signal)
        vol_current = recent["Volume"].iloc[-1]
        vol_recent_avg = recent["Volume"].iloc[-self.box_length:].mean()
        has_volume = vol_current > (vol_recent_avg * 1.5)

        if is_consolidating and is_breakout:
            result["found"] = True
            
            # Scoring
            qual = 5.0
            if has_volume: qual += 3.0
            
            # Trend context (200 MA Check)
            # We try standard keys first, then fallback
            ema200 = self._get_val(indicators, "ema200") or \
                      self._get_val(indicators, "dma200") or \
                      self._get_val(indicators, "wma50") # Weekly proxy
                      
            if ema200 and current_close > ema200:
                qual += 2.0 
            
            result["quality"] = min(qual, 10.0)
            result["score"] = self._normalize_score(qual * 10)
            formation_index = len(df) - (self.box_length * 2)  # Box started 2x box_length ago
            entry_conditions_met = has_volume and is_breakout

            result["meta"] = {
                "box_high": round(box_high, 2),
                "box_low": round(box_low, 2),
                "type": "bullish",
                "breakout_vol": bool(has_volume),
                "age_candles": len(df) - formation_index,
                "formation_time": float(df.index[formation_index].timestamp()),
                "formation_timestamp": df.index[formation_index].isoformat() if formation_index >= 0 else None,
                "box_duration_candles": self.box_length * 2,
                # 🆕 Pattern-Specific Velocity Tracking
                "velocity_tracking": {
                    "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient": result["quality"] >= 7.0,
                    "breakout_confirmed": is_breakout,
                    "volume_confirmed": has_volume
                },
                # 🆕 Formation Context (Generic)
                "formation_context": _build_formation_context(indicators)
            }
            # Calculate invalidation level (varies by horizon)
            if horizon == "intraday":
                invalidation_level = box_low * 0.998
            elif horizon == "short_term":
                invalidation_level = box_low * 0.995
            else:
                invalidation_level = box_low * 0.99

            # Calculate entry trigger (varies by horizon)
            if horizon == "intraday":
                entry_trigger_price = box_high * 1.002
            elif horizon == "short_term":
                entry_trigger_price = box_high * 1.005
            else:
                entry_trigger_price = box_high * 1.005

            # Box age for entry conditions
            box_age_candles = len(df) - formation_index

            # Pattern strength
            pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

            # ADD TO META:
            result["meta"].update({
                "bar_index": len(df),
                # Analytics
                "box_height_pct": round(((box_high - box_low) / box_low) * 100, 2),
                # Entry/Exit Levels (Critical)
                "invalidation_level": round(invalidation_level, 2),
                "entry_trigger_price": round(entry_trigger_price, 2),
                
                # Box Age (Used in entry conditions)
                "box_age_candles": box_age_candles,
                
                # Physics Parameters (from config)
                "physics": {
                    "lookback": self.lookback,
                    "box_length": self.box_length,
                    "target_ratio": 2.0,
                    "max_stop_pct": 5.0
                },
                
                # Universal Fields
                "horizon": horizon,
                "pattern_strength": pattern_strength,
                "current_price": round(current_close, 2)
            })
            result["desc"] = "Darvas Box Breakout"
            return result

        return result
