import numpy as np
import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

class DarvasBoxPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "darvas_box"
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
            ema_200 = self._get_val(indicators, "ema_200") or \
                      self._get_val(indicators, "dma_200") or \
                      self._get_val(indicators, "wma_50") # Weekly proxy
                      
            if ema_200 and current_close > ema_200:
                qual += 2.0 
            
            result["quality"] = min(qual, 10.0)
            result["score"] = self._normalize_score(qual * 10)
            result["meta"] = {
                "box_high": round(box_high, 2),
                "box_low": round(box_low, 2),
                "breakout_vol": bool(has_volume)
            }
            result["desc"] = "Darvas Box Breakout"

        return result

    def _get_val(self, data, key):
        if key not in data: return None
        item = data[key]
        if isinstance(item, dict): return item.get("value")
        return item