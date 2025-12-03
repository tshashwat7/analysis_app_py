# services/tradeplan/time_estimator.py

import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default zig-zag efficiency factor
ZZ_FACTOR = 4.0

# Pattern-driven multipliers — accelerates estimates when strong setups found
PATTERN_SPEED_MULTIPLIERS = {
    "darvas_box":        1.3,
    "cup_handle":        1.2,
    "flag_pennant":      1.25,
    "bollinger_squeeze": 1.4,
    "minervini_stage2":  1.5,
    "three_line_strike": 2.0,
    "default":           1.0
}


def _choose_pattern_boost(patterns: Dict[str, Any]) -> float:
    """Pick highest-relevance multiplier (only if score > 40)."""
    if not patterns:
        return 1.0

    best = 1.0
    for name, p in patterns.items():
        try:
            score = float(p.get("score", 0))
            if score > 40:
                base_mult = PATTERN_SPEED_MULTIPLIERS.get(name, 1.0)
                # score scaling: 50 → 1.0, 100 → 1.5
                scaled = 1.0 + ((min(score, 100) - 50) / 100.0) * 0.5
                candidate = base_mult * scaled
                best = max(best, candidate)
        except Exception:
            continue

    return best


def _extract_slope_factor(indicators: Dict[str, Any]) -> float:
    """
    Adjusts time estimate based on Trend Steepness.
    Source: indicators.py (Dynamic MA slopes)
    """
    if not indicators:
        return 1.0

    try:
        # Priority Chain: Try Horizon-Specific Slopes first
        # These keys come from indicators.py -> compute_ema_slope
        slope_data = (
            indicators.get("ema_20_slope")      # Intraday/Short
            or indicators.get("wma_20_slope")   # Long Term
            or indicators.get("wma_50_slope")   # Long Term Fallback
            or indicators.get("mma_20_slope")   # Multibagger
            or indicators.get("mma_12_slope")   # Multibagger Fallback
            or indicators.get("ema_slope")      # Generic Fallback
        )

        if slope_data is None: 
            return 1.0

        slope_val = 0.0
        if isinstance(slope_data, dict):
            slope_val = float(slope_data.get("value") or slope_data.get("raw") or 0)
        else:
            slope_val = float(slope_data)

        # Logic: Steeper slope = Faster target hit.
        # Cap slope impact at 50 degrees to prevents unrealistic boosts
        # Max Boost = 1.0 + (50/100) = 1.5x speed
        # Min Boost = 1.0
        
        # Using divisor 10.0 assumes slope is in degrees (0-90)
        # 10 degrees -> 1.0 + 1.0 = 2.0x speed is too fast? 
        # Let's dampen it. Use divisor 20.0
        # 20 degrees -> 1.0 + 1.0 = 2.0x 
        
        # If slope is 15 deg -> min(15, 5) = 5 -> 5/10 = 0.5 -> Boost 1.5x
        # If slope is 2 deg  -> min(2, 5) = 2 -> 2/10 = 0.2 -> Boost 1.2x
        # This clamps ANY slope > 5 degrees to a 1.5x max boost. This is SAFE.
        
        return 1.0 + min(max(abs(slope_val), 0.0), 5.0) / 10.0

    except Exception as e:
        # logger.debug(f"Slope factor error: {e}")
        return 1.0


def _extract_strategy_factor(strategies: Dict[str, Any]) -> float:
    """
    Adjusts time estimate based on Strategy Personality.
    Source: strategy_analyzer.py -> results['summary']
    """
    if not strategies:
        return 1.0

    best = strategies.get("best_strategy") or strategies.get("best")
    
    if isinstance(best, list) and best:
        best = best[0]

    if not best:
        return 1.0

    label = str(best).lower()

    # --- FAST STRATEGIES ---
    # Matches: 'day_trading'
    if "day_trading" in label:
        return 1.5
    
    # Matches: 'momentum', 'trend_following'
    # Removed 'breakout' and 'scalp' as they aren't in strategy_analyzer, 
    # but keeping generic keywords is safe if you add strategies later.
    if any(x in label for x in ["momentum", "trend"]):
        return 1.2
    
    # Matches: 'swing'
    if "swing" in label:
        return 1.1

    # --- SLOW STRATEGIES ---
    # Matches: 'value', 'position_trading'
    # Removed 'quality' and 'accumulat' as they do not exist in strategy_analyzer
    if any(x in label for x in ["value", "position"]):
        return 0.8
    
    # Matches: 'income' (dividend)
    if "income" in label:
        return 0.7

    return 1.0

# -------------------------------------------------------------------------

def estimate_hold_time(
    price_val: float,
    t1: Optional[float],
    t2: Optional[float],
    atr_val: Any, 
    horizon: str = "short_term",
    indicators: Dict[str, Any] = None,
    strategies: Dict[str, Any] = None,
) -> str:
    """
    Professional-grade time estimator (Polymorphic ATR Version).
    Calculates: (Distance to Target / ATR) * ZigZagFactor / SpeedModifiers
    """

    try:
        # ----------------------------------------------------
        # 1. VALIDATE PRICE & TARGETS
        # ----------------------------------------------------
        if not price_val or price_val <= 0:
            return "-"

        valid_targets = [t for t in (t1, t2) if t and t > 0]
        if not valid_targets:
            return "-"

        # Choose target closest to price for realistic estimate
        target_price = min(valid_targets, key=lambda x: abs(x - price_val))
        distance = abs(target_price - price_val)
        if distance <= 0:
            return "Now"

        # ----------------------------------------------------
        # 2. ROBUST ATR EXTRACTION
        # ----------------------------------------------------
        # Handle both Dict (new polymorphic) and Float (legacy)
        final_atr = 0.0
        
        if isinstance(atr_val, dict):
            final_atr = float(atr_val.get("value") or 0)
        elif isinstance(atr_val, (int, float)):
            final_atr = float(atr_val)
        
        # Fallback: Try fetching from indicators if atr_val arg failed
        if final_atr <= 0 and indicators:
            dyn = indicators.get("atr_dynamic", {})
            if isinstance(dyn, dict):
                final_atr = float(dyn.get("value") or 0)
            elif isinstance(dyn, (int, float)):
                final_atr = float(dyn)

        if final_atr <= 0:
            return "-"

        # ----------------------------------------------------
        # 3. RAW TIME ESTIMATE (in 'Bars')
        # ----------------------------------------------------
        # ZigZag Logic: Price rarely moves straight. It moves in waves.
        # We assume it takes 4x the direct ATR distance to actually reach there.
        bars = (distance / final_atr) * ZZ_FACTOR
        
        if bars <= 0:
            return "-"

        # ----------------------------------------------------
        # 4. SPEED FACTORS
        # ----------------------------------------------------
        # Patterns removed until implemented in upstream engine
        strategy_factor = _extract_strategy_factor(strategies or {})
        slope_factor    = _extract_slope_factor(indicators or {})

        total_speed = strategy_factor * slope_factor
        if total_speed <= 0: total_speed = 1.0

        effective_bars = max(1.0, bars / total_speed)

        # ----------------------------------------------------
        # 5. TIMEFRAME → HUMAN READABLE OUTPUT
        # ----------------------------------------------------

        # =============== INTRADAY (15m bars assumed) ===============
        if horizon == "intraday":
            minutes = effective_bars * 15.0
            
            if minutes < 60:
                return f"~{int(minutes)} Mins"
            if minutes < 375: # Less than 1 trading day (6.25 hrs)
                hrs = round(minutes / 60, 1)
                return f"~{hrs} Hours"
            
            sessions = math.ceil(minutes / 375.0)
            return f"~{sessions} Days"

        # =============== SHORT TERM (Daily bars) ===============
        if horizon == "short_term":
            days = math.ceil(effective_bars)
            if days < 7:
                return f"~{days} Days"
            
            weeks = math.ceil(days / 5.0)
            return f"~{weeks} Weeks"

        # =============== LONG TERM (Weekly bars) ===============
        if horizon == "long_term":
            # effective_bars here represents WEEKS because ATR is weekly
            weeks = math.ceil(effective_bars)

            if weeks < 4:
                return f"~{weeks} Weeks"

            months = math.ceil(weeks / 4.0)
            if months < 12:
                return f"~{months} Months"

            return f"~{round(months / 12.0, 1)} Years"

        # =============== MULTIBAGGER (Monthly bars) ===============
        if horizon == "multibagger":
             # effective_bars here represents MONTHS because ATR is monthly
            months = math.ceil(effective_bars)

            if months < 12:
                return f"~{months} Months"

            return f"~{round(months / 12.0, 1)} Years"

        # Default fallback (treat as Days)
        return f"~{math.ceil(effective_bars)} Days"

    except Exception as e:
        logger.warning("estimate_hold_time error: %s", e)
        return "-"  
# ========================================================================================================================

#     # services/time_estimator.py
# import math
# import logging
# from typing import Dict, Any, Optional

# logger = logging.getLogger(__name__)

# def estimate_hold_time(
#     price_val: float,
#     t1: Optional[float],
#     t2: Optional[float],
#     atr_val: float,
#     horizon: str,
#     indicators: Dict[str, Any] = None,
#     strategies: Dict[str, Any] = None
# ) -> str:
#     """
#     V3 Estimator:
#     ✔ Horizon aware
#     ✔ Strategy-aware speed factor (Momentum = Fast, Value = Slow)
#     ✔ Slope-aware (Steep trend = Faster target hit)
#     ✔ UI string output only
#     """
#     try:
#         # 1. Safety Checks
#         if not price_val or price_val <= 0 or not atr_val or atr_val <= 0:
#             return "-"

#         # Select nearest target
#         valid_t = []
#         for t in (t1, t2):
#             try:
#                 if t and float(t) > 0: valid_t.append(float(t))
#             except: pass
            
#         if not valid_t:
#             return "-"

#         target_price = min(valid_t, key=lambda x: abs(x - price_val))
#         distance = abs(target_price - price_val)
#         if distance <= 0:
#             return "Now"

#         # Baseline ATR bars (ZigZag Factor)
#         # 4.0 is standard conservative estimate for non-linear moves
#         bars = (distance / atr_val) * 4.0
        
#         if bars <= 0: return "-"

#         # ---- STRATEGY SPEED BOOST ----
#         speed = 1.0
#         if strategies:
#             # Handle both direct strategy name or summary dict keys
#             best = strategies.get("best_strategy") or strategies.get("best")
#             score = strategies.get("best_score") or strategies.get("score", 0)

#             if best in ("momentum", "trend_following"):
#                 # 60+ score = strong trend = fast reach
#                 if score > 60: speed *= 1.5
#                 elif score > 40: speed *= 1.2

#             elif best == "day_trading":
#                 speed *= 2.0  # intraday momentum

#             elif best == "value":
#                 speed *= 0.8  # slow mover (market takes time to recognize value)

#             elif best == "income":
#                 speed *= 0.7  # very slow (dividend stocks)

#         # ---- SLOPE FACTOR ----
#         slope = None
#         if indicators:
#             # Try dynamic keys first
#             raw = indicators.get("ema_20_slope") or indicators.get("ema_slope") or indicators.get("wma_50_slope")
            
#             if isinstance(raw, dict):
#                 slope = raw.get("value") or raw.get("raw") or raw.get("score")
#             else:
#                 slope = raw

#         slope_factor = 1.0
#         if slope is not None:
#             try:
#                 s_val = float(slope)
#                 # Max boost 1.5x for steep slopes (e.g. 45 degrees)
#                 slope_factor = 1.0 + min(abs(s_val), 5) / 10.0 
#             except:
#                 pass

#         # Calculate effective bars
#         effective_bars = bars / (speed * slope_factor)
#         effective_bars = max(1.0, effective_bars)

#         # ---- HORIZON MAPPING ----
#         if horizon == "intraday":
#             minutes = effective_bars * 15
#             if minutes < 300: # < 5 hours
#                 return "Intraday"
#             sessions = math.ceil(minutes / 375) # 375 mins = NSE day
#             if sessions == 1: return "1-2 Days"
#             return f"~{sessions} Days"

#         if horizon == "short_term":
#             days = math.ceil(effective_bars)
#             if days < 7: return f"~{days} Days"
#             return f"~{math.ceil(days/5)} Weeks"

#         if horizon == "long_term":
#             weeks = math.ceil(effective_bars)
#             if weeks < 4: return f"~{weeks} Weeks"
#             months = math.ceil(weeks / 4)
#             if months < 12: return f"~{months} Months"
#             return f"~{round(months/12, 1)} Years"

#         if horizon == "multibagger":
#             months = math.ceil(effective_bars)
#             if months < 12: return f"~{months} Months"
#             return f"~{round(months/12, 1)} Years"

#         return f"~{math.ceil(effective_bars)} Days"

#     except Exception as e:
#         logger.error(f"est_time_v3 error: {e}")
#         return "-"