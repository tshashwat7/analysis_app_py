import pandas as pd
from typing import Dict, Any
from services.data_fetch import _get_val
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class MinerviniVCPPattern(BasePattern):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "minerviniStage2"
    
    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
            
        if df is None or len(df) < 50: return result
        
        # FIX 1: Robust Horizon-Aware MA Lookup
        # We need a Mid-Term Trend (50) and Long-Term Trend (200)
        
        maMid = self._get_val(indicators, "maMid")
        maSlow = self._get_val(indicators, "maSlow")        
        close = df["Close"].iloc[-1]
        
        # Stage 2 Criteria: Price > 50 > 200
        stage2_uptrend = False
        if maMid and maSlow:
            if close > maMid and maMid > maSlow:
                stage2_uptrend = True
        
        if not stage2_uptrend:
            return result 
            
        # FIX 2: ATR Volatility Gate
        # Minervini VCP requires TIGHT action. High ATR % means loose action.
        atrPct = self._get_val(indicators, "atrPct")
        if atrPct and atrPct > 3.5: # Reject if weekly volatility > 3.5%
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
            # VCP contraction started approximately 15 bars ago (last tight period)
            formation_index = len(df) - 15

            # Get additional indicators
            position52w = self._get_val(indicators, "position52w")
            maFast = maMid  # Already calculated above

            # Get actual pivot point from indicators (✅ You have this calculated!)
            pivot_point = self._get_val(indicators, "pivotPoint")

            # Invalidation level (use maFast as the key support level for VCP)
            invalidation_level = maFast * 0.95 if maFast else None

            # Stage quality assessment
            stage_quality = "stage2_confirmed" if position52w and position52w >= 85 else "transitioning" if position52w and position52w >= 70 else "weak"

            # Pattern strength
            pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

            # Contraction strength
            contraction_strength = "tight" if range_recent < 0.02 else "moderate"

            # 🆕 Pattern-specific entry conditions
            vol_recent = df["Volume"].iloc[-5:].mean()
            vol_avg = df["Volume"].iloc[-50:].mean()
            vol_dry = vol_recent < vol_avg
            entry_conditions_met = is_tight and vol_dry

            result["meta"] = {
                "bar_index": len(df),
                "tightness": f"{range_recent*100:.1f}%",
                "vol_dry": vol_dry,
                "age_candles": len(df) - formation_index,
                "formation_timestamp": df.index[formation_index].isoformat() if formation_index >= 0 else None,
                "contraction_pct": round(range_recent * 100, 2),
                # Analytics
                "volatility_quality": result.get("quality", 5.0) or _get_val(indicators, "volatilityQuality"), # or pass from indicators
                # Entry/Exit Levels
                "invalidation_level": round(invalidation_level, 2) if invalidation_level else None,
                "pivot_point": round(pivot_point, 2) if pivot_point else None,  # ✅ Actual pivot from indicators
                # Stage Metrics
                "maFast": round(maFast, 2) if maFast else None,
                "maSlow": round(maSlow, 2) if maSlow else None,
                "position52w": position52w,
                "stage_quality": stage_quality,
                "contraction_strength": contraction_strength,
                # Universal Fields
                "horizon": horizon,
                "pattern_strength": pattern_strength,
                "current_price": round(close, 2),
                # 🆕 Pattern-Specific Velocity Tracking
                "velocity_tracking": {
                    "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient": result["quality"] >= 7.0,
                    "contraction_tight": is_tight,
                    "volume_dry": vol_dry
                },
                # 🆕 Formation Context (Generic)
                "formation_context": _build_formation_context(indicators)
            }
            
        return result