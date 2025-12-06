import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern

class MinerviniVCPPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "minervini_stage2"
    
    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
            
        if df is None or len(df) < 50: return result
        
        # FIX 1: Robust Horizon-Aware MA Lookup
        # We need a Mid-Term Trend (50) and Long-Term Trend (200)
        
        ma_50 = (
            self._get_val(indicators, "ema_50") or 
            self._get_val(indicators, "dma_50") or 
            self._get_val(indicators, "wma_40") or # Weekly Mid
            self._get_val(indicators, "mma_6")     # Monthly Mid (6mo)
        )
        
        ma_200 = (
            self._get_val(indicators, "ema_200") or 
            self._get_val(indicators, "dma_200") or 
            self._get_val(indicators, "wma_50") or # Weekly Slow
            self._get_val(indicators, "mma_12")    # Monthly Slow (12mo)
        )
        
        close = df["Close"].iloc[-1]
        
        # Stage 2 Criteria: Price > 50 > 200
        stage2_uptrend = False
        if ma_50 and ma_200:
            if close > ma_50 and ma_50 > ma_200:
                stage2_uptrend = True
        
        if not stage2_uptrend:
            return result 
            
        # FIX 2: ATR Volatility Gate
        # Minervini VCP requires TIGHT action. High ATR % means loose action.
        atr_pct = self._get_val(indicators, "atr_pct")
        if atr_pct and atr_pct > 3.5: # Reject if weekly volatility > 3.5%
            return result

        # 2. Volatility Contraction (VCP) Logic
        # Compare range of recent 5 days vs previous 10 days
        range_recent = (df["High"].iloc[-5:].max() - df["Low"].iloc[-5:].min()) / close
        range_prev = (df["High"].iloc[-15:-5].max() - df["Low"].iloc[-15:-5].min()) / close
        
        # Contraction: Recent range is roughly half of previous range
        is_contracting = range_recent < (range_prev * 0.7) 
        is_tight = range_recent < 0.05 
        
        if is_contracting and is_tight:
            result["found"] = True
            qual = 7.0
            
            # Dry Volume Check
            vol_recent = df["Volume"].iloc[-5:].mean()
            vol_avg = df["Volume"].iloc[-50:].mean()
            if vol_recent < vol_avg: 
                qual += 2.0
                
            result["quality"] = min(qual, 10.0)
            result["score"] = self._normalize_score(qual * 10)
            result["desc"] = "Minervini VCP (Tight)"
            result["meta"] = {
                "tightness": f"{range_recent*100:.1f}%",
                "vol_dry": (vol_recent < vol_avg)
            }
            
        return result

    def _get_val(self, data, key):
        if key not in data: return None
        item = data[key]
        if isinstance(item, dict): return item.get("value")
        return item