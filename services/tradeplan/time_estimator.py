# services/tradeplan/time_estimator.py (REFACTORED v3.0)
"""
Trade Timeline Estimator - Query Extractor Edition
===================================================
✅ REFACTORED: Now uses QueryOptimizedExtractor instead of direct config access
✅ BENEFITS: 
   - Consistent config hierarchy (global → horizon → setup)
   - Better caching
   - Easier testing and maintenance

Estimates time to reach targets using velocity factors from config hierarchy.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# def estimate_target_timeline(
#     entry: float,
#     t1: float,
#     t2: float,
#     indicators: Dict[str, Any],
#     horizon: str = "short_term",
#     extractor: Optional[Any] = None  # ✅ NEW: Accept extractor
# ) -> Dict[str, Any]:
#     """
#     ✅ REFACTORED: Now uses query extractor for config access.
    
#     Estimates time to reach T1 and T2 based on trend velocity.
    
#     Args:
#         entry: Entry price
#         t1: First target price
#         t2: Second target price
#         indicators: Technical indicators (for trendStrength, atr)
#         horizon: Trading timeframe
#         extractor: QueryOptimizedExtractor instance (optional - will create if None)
    
#     Returns:
#         {
#             "t1_estimate": "5 days",
#             "t2_estimate": "12 days",
#             "velocity_factor": 1.2,
#             "trend_regime": "strong",
#             "confidence": "medium"
#         }
#     """
#     try:
#         # 1. Validate Inputs
#         atr = indicators.get("atrDynamic", {}).get("value")
        
#         if not all([entry, t1, t2, atr]):
#             return {
#                 "error": "Missing required inputs",
#                 "entry": entry,
#                 "t1": t1,
#                 "t2": t2,
#                 "atr": atr
#             }
        
#         # 2. Get Extractor (create if not provided)
#         if extractor is None:
#             from config.config_helpers import get_resolver
#             resolver = get_resolver(horizon)
#             extractor = resolver.extractor
        
#         # 3. ✅ REFACTORED: Get time estimation config via extractor
#         time_cfg = _get_time_estimation_config(extractor)
#         velocity_factors = time_cfg.get("velocity_factors", {})
#         base_friction = time_cfg.get("base_friction", 0.8)
        
#         # 4. Determine Trend Regime
#         trend_strength = indicators.get("trendStrength", {}).get("value", 5.0)
        
#         if trend_strength >= velocity_factors.get("strong_trend", {}).get("min_strength", 7.0):
#             regime = "strong"
#             velocity = velocity_factors.get("strong_trend", {}).get("factor", 1.2)
#         elif trend_strength >= velocity_factors.get("normal_trend", {}).get("min_strength", 5.0):
#             regime = "normal"
#             velocity = velocity_factors.get("normal_trend", {}).get("factor", 1.0)
#         else:
#             regime = "weak"
#             velocity = velocity_factors.get("weak_trend", {}).get("factor", 0.8)
        
#         # 5. Calculate Time Estimates
#         t1_distance = abs(t1 - entry)
#         t2_distance = abs(t2 - entry)
        
#         # Bars to reach target (velocity-adjusted)
#         t1_bars = (t1_distance / atr) * velocity * base_friction
#         t2_bars = (t2_distance / atr) * velocity * base_friction
        
#         # 6. ✅ REFACTORED: Get candle rate config via extractor
#         candle_rates = _get_candle_rate_config(extractor)
#         rate_cfg = candle_rates.get(horizon, candle_rates["short_term"])
#         bars_per_day = rate_cfg["bars_per_day"]
#         unit = rate_cfg["unit"]
        
#         t1_days = t1_bars / bars_per_day if bars_per_day > 0 else t1_bars
#         t2_days = t2_bars / bars_per_day if bars_per_day > 0 else t2_bars
        
#         # 7. Format Output
#         return {
#             "t1_estimate": _format_time(t1_days, unit),
#             "t2_estimate": _format_time(t2_days, unit),
#             "velocity_factor": velocity,
#             "trend_regime": regime,
#             "trend_strength": trend_strength,
#             "confidence": "high" if regime == "strong" else "medium" if regime == "normal" else "low",
#             "raw_bars": {"t1": round(t1_bars, 1), "t2": round(t2_bars, 1)}
#         }
    
#     except Exception as e:
#         logger.error(f"Timeline estimation failed: {e}", exc_info=True)
#         return {"error": str(e)}


# ============================================================
# ✅ NEW: Config Retrieval Helpers (Extractor-based)
# ============================================================

def _get_time_estimation_config(extractor) -> Dict[str, Any]:
    """
    ✅ NEW: Get time estimation config via extractor.
    
    Checks hierarchy:
    1. Horizon-specific time_estimation config
    2. Global time_estimation config
    3. Hardcoded defaults
    
    Args:
        extractor: QueryOptimizedExtractor instance
    
    Returns:
        Time estimation config with velocity_factors and base_friction
    """
    # Try horizon-specific first
    horizon_config = extractor.base_extractor.get("horizon_time_estimation")
    if horizon_config:
        return horizon_config
    
    # Fall back to global
    global_config = extractor.base_extractor.get("time_estimation")
    if global_config:
        return global_config
    
    # Hardcoded defaults (last resort)
    logger.warning("Using hardcoded time estimation defaults - config not found")
    return {
        "velocity_factors": {
            "strong_trend": {"min_strength": 7.0, "factor": 1.2},
            "normal_trend": {"min_strength": 5.0, "factor": 1.0},
            "weak_trend": {"min_strength": 0.0, "factor": 0.8}
        },
        "base_friction": 0.8
    }


def _get_candle_rate_config(extractor) -> Dict[str, Dict]:
    """
    ✅ NEW: Get candle rate mappings via extractor.
    
    Checks hierarchy:
    1. Horizon-specific candle_rate_mappings
    2. Global candle_rate_mappings
    3. Hardcoded defaults
    
    Args:
        extractor: QueryOptimizedExtractor instance
    
    Returns:
        Candle rate config for all horizons
    """
    # Try horizon-specific first
    horizon_config = extractor.base_extractor.get("horizon_candle_rate_mappings")
    if horizon_config:
        return horizon_config
    
    # Fall back to global
    global_config = extractor.base_extractor.get("candle_rate_mappings")
    if global_config:
        return global_config
    
    # Hardcoded defaults
    logger.warning("Using hardcoded candle rate defaults - config not found")
    return {
        "intraday": {"bars_per_day": 26, "unit": "hours"},     # 15m candles
        "short_term": {"bars_per_day": 1, "unit": "days"},     # Daily
        "long_term": {"bars_per_day": 0.2, "unit": "weeks"},   # Weekly
        "multibagger": {"bars_per_day": 0.05, "unit": "months"} # Monthly
    }


def _format_time(days: float, unit: str) -> str:
    """
    Format time estimate with appropriate unit.
    
    Args:
        days: Time in days
        unit: Target unit ("hours", "days", "weeks", "months")
    
    Returns:
        Formatted string like "5 days" or "2 weeks"
    """
    if unit == "hours":
        return f"{max(1, int(days * 24))} hours"
    elif unit == "days":
        return f"{max(1, int(days))} days"
    elif unit == "weeks":
        return f"{max(1, int(days / 7))} weeks"
    else:  # months
        return f"{max(1, int(days / 30))} months"


# ============================================================
# ✅ USAGE EXAMPLES
# ============================================================

"""
USAGE IN SIGNAL ENGINE:

# Example 1: Pass extractor explicitly (recommended)
from config.config_helpers import get_resolver

resolver = get_resolver(horizon)
extractor = resolver.extractor

stop_loss, _ = calculate_stop_loss_v5(entry, indicators, exec_ctx)
t1, t2, _ = calculate_targets_v5(entry, stop_loss, indicators, horizon)

# Pass extractor to avoid recreating it
timeline = estimate_target_timeline(
    entry, t1, t2, indicators, horizon, 
    extractor=extractor  # ✅ Reuse existing extractor
)
trade_plan["est_time"] = timeline


# Example 2: Let it create extractor (simpler but less efficient)
timeline = estimate_target_timeline(entry, t1, t2, indicators, horizon)
trade_plan["est_time"] = timeline
"""