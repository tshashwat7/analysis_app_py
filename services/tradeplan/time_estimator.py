# services/tradeplan/time_estimator.py

import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default zig-zag efficiency factor (Price doesn't move in a straight line)
ZZ_FACTOR = 4.0

# 1. Pattern Speed Factors
PATTERN_SPEED_MULTIPLIERS = {
    "three_line_strike": 2.5,   # Very fast mean reversion
    "minervini_stage2":  1.8,   # Explosive breakout
    "bollinger_squeeze": 1.5,   # Volatility expansion
    "flag_pennant":      1.4,   # Trend continuation
    "darvas_box":        1.3,   # Trend continuation
    "cup_handle":        1.2,   # Measured move (medium speed)
    "ichimoku_signals":  1.1,   # Trend confirmation
    "double_top_bottom": 0.9,   # Structure building takes time
    "golden_cross":      0.8,   # Major regime change (Long hold)
    "default":           1.0
}

def _choose_pattern_boost(patterns: Dict[str, Any]) -> float:
    """Pick highest-relevance multiplier based on found patterns."""
    if not patterns: return 1.0
    best_mult = 1.0
    for name, p in patterns.items():
        try:
            if not p.get("found"): continue
            score = float(p.get("score", 0))
            if score > 50:
                base = PATTERN_SPEED_MULTIPLIERS.get(name, 1.0)
                quality_boost = 1.1 if score > 80 else 1.0
                best_mult = max(best_mult, base * quality_boost)
        except Exception: continue
    return best_mult

def _extract_slope_factor(indicators: Dict[str, Any]) -> float:
    """Steeper trend = Faster target hit."""
    if not indicators: return 1.0
    try:
        slope_data = (
            indicators.get("ema_20_slope") or indicators.get("wma_20_slope") or 
            indicators.get("wma_50_slope") or indicators.get("mma_20_slope") or 
            indicators.get("mma_12_slope") or indicators.get("ema_slope")
        )
        if slope_data is None: return 1.0
        
        slope_val = float(slope_data.get("value") or slope_data.get("raw") or 0) if isinstance(slope_data, dict) else float(slope_data)
        return 1.0 + min(max(abs(slope_val), 0.0), 10.0) / 10.0
    except Exception: return 1.0

def _extract_strategy_factor(strategies: Dict[str, Any]) -> float:
    """Adjusts time based on Strategy Personality."""
    if not strategies: return 1.0
    best = strategies.get("best_strategy") or strategies.get("best")
    if isinstance(best, list) and best: best = best[0]
    if not best: return 1.0
    
    label = str(best).lower()
    if "day_trading" in label: return 2.0
    if "minervini" in label: return 1.5
    if "canslim" in label: return 1.4
    if any(x in label for x in ["momentum", "trend"]): return 1.2
    if "swing" in label: return 1.1
    if any(x in label for x in ["value", "position"]): return 0.8
    if "income" in label: return 0.6
    return 1.0

def estimate_hold_time(
    price_val: float,
    t1: Optional[float],
    t2: Optional[float],
    atr_val: Any, 
    horizon: str = "short_term",
    indicators: Dict[str, Any] = None,
    strategies: Dict[str, Any] = None,
    patterns: Dict[str, Any] = None
) -> str:
    """
    Calculates estimated hold time string (e.g. "~3 Weeks")
    """
    try:
        if not price_val or price_val <= 0: return "-"
        
        # Handle Polymorphic ATR
        final_atr = 0.0
        if isinstance(atr_val, dict): final_atr = float(atr_val.get("value") or 0)
        elif isinstance(atr_val, (int, float)): final_atr = float(atr_val)
        if final_atr <= 0: return "-"

        # Targets
        targets = [t for t in (t1, t2) if t and t > 0]
        if not targets: return "-"
        avg_target = sum(targets) / len(targets)
        distance = abs(avg_target - price_val)
        if distance <= 0: return "Now"

        # Calculation
        base_bars = (distance / final_atr) * ZZ_FACTOR
        
        strat_speed = _extract_strategy_factor(strategies or {})
        slope_speed = _extract_slope_factor(indicators or {})
        pat_speed   = _choose_pattern_boost(patterns or {})
        
        total_speed = min(strat_speed * slope_speed * pat_speed, 4.0)
        effective_bars = max(1.0, base_bars / total_speed)

        # Output Formatting
        if horizon == "intraday":
            minutes = effective_bars * 15.0
            if minutes < 60: return f"~{int(minutes)} Mins"
            if minutes < 375: return f"~{round(minutes/60, 1)} Hours"
            return f"~{math.ceil(minutes / 375.0)} Days"

        if horizon == "short_term":
            days = math.ceil(effective_bars)
            if days < 7: return f"~{days} Days"
            return f"~{math.ceil(days/5.0)} Weeks"

        if horizon == "long_term":
            weeks = math.ceil(effective_bars)
            if weeks < 4: return f"~{weeks} Weeks"
            months = math.ceil(weeks / 4.0)
            if months < 12: return f"~{months} Months"
            return f"~{round(months/12.0, 1)} Years"

        if horizon == "multibagger":
            months = math.ceil(effective_bars)
            if months < 12: return f"~{months} Months"
            return f"~{round(months/12.0, 1)} Years"

        return f"~{int(effective_bars)} Bars"

    except Exception as e:
        logger.warning(f"Time Est Error: {e}")
        return "-"