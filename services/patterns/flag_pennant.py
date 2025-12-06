import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern

class FlagPennantPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "flag_pennant"
        # Configurable windows
        self.pole_days = self.config.get("pole_back", 15)
        self.flag_days = self.config.get("flag_back", 5)
    
    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        required_len = self.pole_days + self.flag_days
        if df is None or len(df) < required_len: return result
        
        closes = df["Close"]
        
        # FIX 1: Dynamic Windows
        pole_start_price = closes.iloc[-self.pole_days]
        pole_end_price = closes.iloc[-self.flag_days]
        
        # Calculate Pole Return
        if pole_start_price == 0: return result
        pole_return = (pole_end_price - pole_start_price) / pole_start_price
        
        is_strong_pole = pole_return > 0.05 # 5% move
        
        # Calculate Flag Drift
        flag_start_price = closes.iloc[-self.flag_days]
        flag_end_price = closes.iloc[-1]
        
        if flag_start_price == 0: return result
        flag_return = (flag_end_price - flag_start_price) / flag_start_price
        
        # Flag logic: Slight drift (-3% to +1%)
        is_tight_flag = (-0.03 < flag_return < 0.01)
        
        # FIX 2: Trend Check (Flag must be in uptrend)
        # Dynamic lookup for "Fast MA" based on horizon
        trend_ma_val = None
        if horizon == "long_term":
            trend_ma_val = self._get_val(indicators, "wma_10")
        elif horizon == "multibagger":
            trend_ma_val = self._get_val(indicators, "mma_6")
        else:
            trend_ma_val = self._get_val(indicators, "ema_20")
            
        is_uptrend = True
        if trend_ma_val and closes.iloc[-1] < trend_ma_val:
            is_uptrend = False # Price below fast MA = Weak flag
        
        # Volume Check
        vols = df["Volume"]
        vol_pole = vols.iloc[-(self.pole_days):-(self.flag_days)].mean()
        vol_flag = vols.iloc[-(self.flag_days):].mean()
        is_vol_drying = vol_flag < vol_pole
        
        if is_strong_pole and is_tight_flag and is_uptrend:
            result["found"] = True
            qual = 6.0
            if is_vol_drying: qual += 2.0
            
            # Breakout today?
            if closes.iloc[-1] > closes.iloc[-2] * 1.01:
                qual += 2.0
                result["desc"] = "Bull Flag Breakout"
            else:
                result["desc"] = "Bull Flag (Forming)"
                
            result["quality"] = min(qual, 10.0)
            result["score"] = self._normalize_score(qual * 10)
            result["meta"] = {
                "pole_gain_pct": round(pole_return * 100, 1),
                "flag_drift_pct": round(flag_return * 100, 1)
            }
            
        return result

    def _get_val(self, data, key):
        if key not in data: return None
        item = data[key]
        if isinstance(item, dict): return item.get("value")
        return item