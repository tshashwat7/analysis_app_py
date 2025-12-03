# services/time_estimator.py

import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default zig-zag efficiency factor
ZZ_FACTOR = 4.0

# Pattern-driven speed multipliers (higher → faster reach)
PATTERN_SPEED_MULTIPLIERS = {
    "darvas_box": 1.3,
    "cup_handle": 1.2,
    "flag_pennant": 1.25,
    "bollinger_squeeze": 1.4,
    "minervini_stage2": 1.5,
    "three_line_strike": 2.0,
    # default
    "default": 1.0
}


def _choose_pattern_boost(patterns: Dict[str, Any]) -> float:
    """Pick highest-relevance pattern multiplier."""
    if not patterns:
        return 1.0
    best = 1.0
    for name, pr in patterns.items():
        try:
            score = float(pr.get("score", 0))
            if score and score > 40:  # only meaningful patterns
                mult = PATTERN_SPEED_MULTIPLIERS.get(name, PATTERN_SPEED_MULTIPLIERS["default"])
                # scale by score (50..100 -> 1.0..1.5)
                score_boost = 1.0 + ((min(score, 100) - 50) / 100.0) * 0.5
                candidate = mult * score_boost
                if candidate > best:
                    best = candidate
        except Exception:
            continue
    return max(best, 1.0)


def estimate_hold_time(
    price_val: float,
    t1: Optional[float],
    t2: Optional[float],
    atr_val: float,
    horizon: str,
    indicators: Dict[str, Any] = None,
    patterns: Dict[str, Any] = None
) -> str:
    """
    Returns string only for UI: "Intraday", "~3 Days", "~2 Weeks", "~6 Months", "~1.2 Years"
    Uses ATR, horizon mapping, pattern multipliers, and dynamic MA slope factors.
    Assumes ATR is **DAILY ATR** unless intraday ATR is explicitly available.
    """

    try:
        # --- Safety Checks ---
        if not price_val or price_val <= 0 or not atr_val or atr_val <= 0:
            return "-"

        # pick nearest valid target
        valid_targets = [t for t in (t1, t2) if t is not None and t > 0]
        if not valid_targets:
            return "-"

        target_price = min(valid_targets, key=lambda x: abs(x - price_val))
        distance = abs(target_price - price_val)
        if distance <= 0:
            return "Now"

        # --- Convert to ATR bars (in DAILY units) ---
        bars = (distance / atr_val) * ZZ_FACTOR
        if bars <= 0:
            return "-"

        # --- Pattern Boost ---
        pattern_boost = _choose_pattern_boost(patterns or {})

        # --- Dynamic Slope Factor ---
        slope_factor = 1.0
        try:
            slope = None
            if indicators:
                slope = indicators.get("ema_20_slope") \
                        or indicators.get("ema_slope") \
                        or indicators.get("wma_50_slope")

                if isinstance(slope, dict):
                    slope = slope.get("value") or slope.get("raw") or slope.get("score")

                slope = float(slope) if slope not in (None, "") else None

            if slope is not None:
                # larger slope = faster target reach
                slope_factor = 1.0 + min(max(abs(slope), 0.0), 5.0) / 10.0
        except Exception:
            slope_factor = 1.0

        # --- Apply Speed Boosts ---
        effective_bars = bars / (pattern_boost * slope_factor)
        effective_bars = max(1.0, effective_bars)   # clamp lower bound

        # =====================================================
        # HORIZON-SPECIFIC TIME CONVERSION (Correct Units)
        # =====================================================

        # ------------------------
        # 🔵 INTRADAY
        # ------------------------
        # Only valid if intraday ATR exists (e.g., ATR_15m)
        if horizon == "intraday":
            atr_15m = None
            if indicators:
                atr_15m = indicators.get("atr_15m") \
                          or indicators.get("atr_10m") \
                          or indicators.get("atr_5m")

            if atr_15m:
                # 1 bar = 15 minutes
                minutes = effective_bars * 15.0
                if minutes < 300:  # < 5 hours
                    return "Intraday"
                sessions = math.ceil(minutes / 375.0)  # NSE day
                return f"~{sessions} Days"

            return "-"   # cannot estimate intraday from daily ATR

        # ------------------------
        # 🟢 SHORT TERM (Days → Weeks)
        # ------------------------
        if horizon == "short_term":
            days = math.ceil(effective_bars)  # 1 bar = 1 day
            if days < 7:
                return f"~{days} Days"

            weeks = math.ceil(days / 5.0)
            return f"~{weeks} Weeks"

        # ------------------------
        # 🟠 LONG TERM (Days → Weeks → Months → Years)
        # ------------------------
        if horizon == "long_term":
            days = effective_bars  # ATR-days
            weeks = math.ceil(days / 5.0)  # convert properly

            if weeks < 4:
                return f"~{weeks} Weeks"

            months = math.ceil(weeks / 4.0)
            if months < 12:
                return f"~{months} Months"

            return f"~{round(months / 12.0, 1)} Years"

        # ------------------------
        # 🔴 MULTIBAGGER (Days → Months → Years)
        # ------------------------
        if horizon == "multibagger":
            days = effective_bars
            months = math.ceil(days / 22.0)  # 22 trading days ≈ 1 month

            if months < 12:
                return f"~{months} Months"

            return f"~{round(months / 12.0, 1)} Years"

        # ------------------------
        # Default Fallback
        # ------------------------
        days = math.ceil(effective_bars)
        return f"~{days} Days"

    except Exception as e:
        logger.exception("calc_est_time_v3 error: %s", e)
        return "-"
