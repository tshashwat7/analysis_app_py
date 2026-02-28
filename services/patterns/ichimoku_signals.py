import pandas as pd
import numpy as np
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context

class IchimokuSignals(BasePattern):
    """
    Detects Ichimoku Kumo Breakouts and TK Crosses.
    
    Assumptions:
    - Standard settings: 9, 26, 52 (Configurable).
    - Checks CURRENT cloud (Span A/B at current index), not future cloud.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias = "ichimokuSignals"
        # Configurable Windows
        self.tenkan_win = self.config.get("tenkan_window", 9)
        self.kijun_win = self.config.get("kijun_window", 26)

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        result = {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        # Guard for minimal history (Need enough for Kijun + 1 previous bar)
        if df is None or len(df) < (self.kijun_win + 2): 
            return result
        
        # 1. Fetch Current Cloud State
        # Note: These values represent the cloud at the CURRENT price candle.
        # (Real Ichimoku plots this 26 bars forward, but we check price relative to support NOW).
        span_a = self._get_val(indicators, "ichiSpanA")
        span_b = self._get_val(indicators, "ichiSpanB")
        price = self._get_val(indicators, "price")
        
        if price is None or span_a is None or span_b is None:
            return result
        
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        is_above_cloud = price > cloud_top
        is_below_cloud = price < cloud_bottom
        
        # 2. Compute TK Values (Rolling)
        # We compute manually to get the PREVIOUS value for crossover detection
        try:
            high_9 = df["High"].rolling(window=self.tenkan_win).max()
            low_9 = df["Low"].rolling(window=self.tenkan_win).min()
            tenkan_series = (high_9 + low_9) / 2
            
            high_26 = df["High"].rolling(window=self.kijun_win).max()
            low_26 = df["Low"].rolling(window=self.kijun_win).min()
            kijun_series = (high_26 + low_26) / 2
            
            t_curr = tenkan_series.iloc[-1]
            k_curr = kijun_series.iloc[-1]
            t_prev = tenkan_series.iloc[-2]
            k_prev = kijun_series.iloc[-2]
        except Exception:
            t_curr, k_curr, t_prev, k_prev = np.nan, np.nan, np.nan, np.nan

        # 3. Fallback Mechanism
        # If manual calc failed (NaNs), try fetching from indicators
        if pd.isna(t_curr): t_curr = self._get_val(indicators, "ichiTenkan")
        if pd.isna(k_curr): k_curr = self._get_val(indicators, "ichiKijun")
        
        # If we STILL don't have current values, abort
        if pd.isna(t_curr) or pd.isna(k_curr):
            return result
            
        # If we lack PREVIOUS values (e.g. indicators dict doesn't have history),
        # we assume steady state (no cross) to be safe.
        if pd.isna(t_prev): t_prev = t_curr
        if pd.isna(k_prev): k_prev = k_curr

        self.log_debug(f"TK Data: T={t_curr:.2f}, K={k_curr:.2f}, T_prev={t_prev:.2f}")

        # 4. Logic
        tk_cross_bull = (t_prev <= k_prev) and (t_curr > k_curr)
        tk_cross_bear = (t_prev >= k_prev) and (t_curr < k_curr)
        
        # Scoring
        qual = 0.0
        signal_type = "NEUTRAL"
        
        if tk_cross_bull:
            if is_above_cloud:
                qual = 9.0; signal_type = "STRONG_TK_CROSS_BULL"
                result["desc"] = "Ichimoku Strong Bull Cross"
            elif is_below_cloud:
                qual = 5.0; signal_type = "WEAK_TK_CROSS_BULL"
                result["desc"] = "Ichimoku Weak Bull Cross"
            else:
                qual = 7.0; signal_type = "NEUTRAL_TK_CROSS_BULL"
                result["desc"] = "Ichimoku Neutral Bull Cross"

        elif tk_cross_bear:
            if is_below_cloud:
                qual = 9.0; signal_type = "STRONG_TK_CROSS_BEAR"
                result["desc"] = "Ichimoku Strong Bear Cross"
            elif is_above_cloud:
                qual = 5.0; signal_type = "WEAK_TK_CROSS_BEAR"
                result["desc"] = "Ichimoku Weak Bear Cross"
            else:
                qual = 7.0; signal_type = "NEUTRAL_TK_CROSS_BEAR"
                result["desc"] = "Ichimoku Neutral Bear Cross"

        elif is_above_cloud:
            qual = 5.0; signal_type = "PRICE_ABOVE_CLOUD"
            result["desc"] = "Ichimoku Cloud Support"
        
        if qual > 0:
            result["found"] = True
            result["quality"] = qual
            # Bonus points for fresh cross
            base_score = qual * 10
            if "CROSS" in signal_type: base_score += 10
            
            result["score"] = self._normalize_score(base_score)
            # Signal is fresh if cross just happened, otherwise established
            is_fresh_cross = tk_cross_bull or tk_cross_bear
            signal_age = 1 if is_fresh_cross else 5  # Fresh cross = 1 bar old, established = 5 bars

            # Calculate cloud thickness
            cloud_thickness = abs(span_a - span_b) / price if price else 0

            # Calculate TK spread
            tenkan_kijun_spread = abs(t_curr - k_curr) / price if price else 0

            # Invalidation level (cloud bottom)
            invalidation_level = cloud_bottom

            # Entry trigger varies by signal type
            entry_trigger_price = cloud_top if is_above_cloud else t_curr

            # Pattern strength
            pattern_strength = "strong" if result["quality"] >= 8.5 else "moderate" if result["quality"] >= 6.5 else "weak"

            # Cloud color
            cloud_color = "bullish" if span_a > span_b else "bearish"

            entry_conditions_met = (tk_cross_bull or tk_cross_bear) and (is_above_cloud or is_below_cloud)

            result["meta"] = {
                "bar_index": len(df),
                "signal": signal_type,
                "cloud_top": round(cloud_top, 2),
                "cloud_bottom": round(cloud_bottom, 2),
                "fresh_cross": is_fresh_cross,
                "tenkan": round(float(t_curr), 2),
                "kijun": round(float(k_curr), 2),
                "age_candles": signal_age,
                "formation_timestamp": df.index[-signal_age].isoformat() if len(df) > signal_age else None,
                "signal_freshness": "fresh" if is_fresh_cross else "established",
                 # Ichimoku Metrics (Critical for entry conditions)
                "cloud_thickness": round(cloud_thickness, 4),
                "tenkan_kijun_spread": round(tenkan_kijun_spread, 4),
                "cloud_color": cloud_color,
                # Individual Components (for invalidation checks)
                "ichiSpanA": round(span_a, 2) if span_a else None,
                "ichiSpanB": round(span_b, 2) if span_b else None,
                "ichiTenkan": round(float(t_curr), 2),
                "ichiKijun": round(float(k_curr), 2),
                
                # Entry/Exit Levels
                "invalidation_level": round(invalidation_level, 2),
                "entry_trigger_price": round(entry_trigger_price, 2) if entry_trigger_price else None,
                
                # Quality Assessment
                "tk_cross_strength": "strong" if tenkan_kijun_spread > 0.01 else "moderate",
                "entry_quality": "high" if entry_conditions_met and result["quality"] >= 8.0 else "moderate",
                
                # Universal Fields
                "horizon": horizon,
                "pattern_strength": pattern_strength,
                "current_price": round(price, 2),
                # 🆕 Pattern-Specific Velocity Tracking
                "velocity_tracking": {
                    "can_track": result["quality"] >= 7.0 and entry_conditions_met,
                    "entry_conditions_met": entry_conditions_met,
                    "quality_sufficient": result["quality"] >= 7.0,
                    "cross_confirmed": is_fresh_cross,
                    "cloud_aligned": is_above_cloud or is_below_cloud
                },
                # 🆕 Formation Context (Generic)
                "formation_context": _build_formation_context(indicators)
            }
            
        return result