# services/tradeplan/time_estimator.py
import math
from typing import Dict, Any, Optional
from config.constants import MASTER_CONFIG as MASTERCONFIG
import logging

logger = logging.getLogger(__name__)
# --- Local Helpers to prevent Import Errors ---
def _ensure_numeric(x, default=0.0):
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, dict): return float(x.get("value") or x.get("raw") or default)
        return float(default)
    except: return float(default)

def _get_val_local(data, key, default=None):
    if not data or key not in data: return default
    return data[key]

def _extract_score(x):
    """Safely extract numeric value from indicator dict or float."""
    if isinstance(x, dict):
        return x.get("value") or x.get("score") or x.get("raw") or 5.0
    return float(x) if x is not None else 5.0

# --- MAIN DUAL ESTIMATOR ---
def estimate_hold_time_dual(
    entry_price: float,
    targets: Dict[str, float],  # {'t1': X, 't2': Y}
    atr: float,
    horizon: str,
    indicators: Dict = None,
    multiplier: float = 1.0,
    strategy_summary: str = "unknown"
) -> Dict[str, Any]:
    """
    ✅ FIXED: Uses MASTERCONFIG velocity factors instead of hardcoded values.
    
    Reads time_estimation config from MASTERCONFIG['global']['time_estimation']:
    - velocity_factors for trend regimes
    - base_friction for all calculations
    """
    

    
    # ✅ GET VELOCITY FACTORS FROM MASTERCONFIG
    try:
        time_est_cfg = MASTERCONFIG.get('global', {}).get('time_estimation', {})
        velocity_factors = time_est_cfg.get('velocity_factors', {})
        base_friction = time_est_cfg.get('base_friction', 0.8)
        
        logger.info(f"time_estimator: Loaded velocity_factors from MASTERCONFIG")
        logger.debug(f"  velocity_factors: {velocity_factors}")
        logger.debug(f"  base_friction: {base_friction}")
    except Exception as e:
        logger.error(f"Failed to load time_estimation config: {e}")
        # Fallback defaults
        velocity_factors = {
            'strong_trend': {'min_strength': 7.0, 'factor': 1.2},
            'normal_trend': {'min_strength': 5.0, 'factor': 1.0},
            'weak_trend': {'max_strength': 5.0, 'factor': 0.8}
        }
        base_friction = 0.8
    
    # ✅ DETERMINE TREND REGIME
    trend_strength = 5.0  # Default
    if indicators:
        trend_strength = float(indicators.get('trend_strength', {}).get('value', 5.0))
    
    # Select velocity factor based on trend strength
    velocity_factor = 1.0  # Default

    if trend_strength >= velocity_factors.get('strong_trend', {}).get('min_strength', 7.0):
        velocity_factor = velocity_factors.get('strong_trend', {}).get('factor', 1.2)
        logger.info(f"  Using STRONG trend velocity factor: {velocity_factor}")
    
    elif trend_strength >= velocity_factors.get('normal_trend', {}).get('min_strength', 5.0):
        velocity_factor = velocity_factors.get('normal_trend', {}).get('factor', 1.0)
        logger.info(f"  Using NORMAL trend velocity factor: {velocity_factor}")

    elif trend_strength <= velocity_factors.get('weak_trend', {}).get('max_strength', 5.0):
        velocity_factor = velocity_factors.get('weak_trend', {}).get('factor', 0.8)
        logger.info(f"  Using WEAK trend velocity factor: {velocity_factor}")
    
    # ✅ CALCULATE ESTIMATES
    t1_target = targets.get('t1', entry_price)
    t2_target = targets.get('t2', entry_price)
    
    t1_distance = abs(t1_target - entry_price)
    t2_distance = abs(t2_target - entry_price)
    
    # ATR-based speed calculation
    bars_per_atr = max(2, atr / max(entry_price * 0.001, 0.01))
    
    # Apply velocity factor and multiplier
    t1_bars = max(1, t1_distance / max(atr, entry_price * 0.001)) * velocity_factor * multiplier * base_friction
    t2_bars = max(1, t2_distance / max(atr, entry_price * 0.001)) * velocity_factor * multiplier * base_friction
    
    # Convert to time units
    candles_per_hour = {
        'intraday': 4,      # 15m candles
        'short_term': 1,    # Daily candles
        'long_term': 0.2,   # Weekly candles
        'multibagger': 0.05 # Monthly candles
    }
    
    candles_per_unit = candles_per_hour.get(horizon, 1)
    
    t1_days = t1_bars / candles_per_unit if candles_per_unit > 0 else t1_bars
    t2_days = t2_bars / candles_per_unit if candles_per_unit > 0 else t2_bars
    
    # Format estimates
    def format_estimate(days):
        if days < 1:
            return f"{max(1, int(days * 24))} hours"
        elif days < 30:
            return f"{int(days)} days"
        elif days < 365:
            return f"{int(days / 7)} weeks"
        else:
            return f"{int(days / 365)} years"
    
    return {
        't1_estimate': format_estimate(t1_days),
        't2_estimate': format_estimate(t2_days),
        'velocity_factor_used': velocity_factor,
        'trend_strength': trend_strength,
        'base_friction': base_friction,
        'raw_t1_bars': round(t1_bars, 1),
        'raw_t2_bars': round(t2_bars, 1)
    }

