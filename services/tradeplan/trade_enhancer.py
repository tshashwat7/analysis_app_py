# services/tradeplan/trade_enhancer.py
"""
Trade Plan Pattern Enhancement Engine v2.0
==========================================

Refines trade plans using MASTER_CONFIG pattern physics, entry rules, and invalidation logic.

Features:
- 9 Pattern Detection & Enhancement
- Dynamic Entry Validation
- Config-Driven Invalidation Monitoring
- RR Regime Adjustments (ADX-based)
- Pattern Expiration Tracking
- Defensive Error Handling

Patterns Supported:
1. Darvas Box
2. Cup & Handle
3. Minervini VCP/Stage 2
4. Flag/Pennant
5. Bollinger Squeeze
6. Golden/Death Cross
7. Double Top/Bottom
8. Three-Line Strike
9. Ichimoku Signals
"""

import copy
import logging
from typing import Dict, Any, Optional, Tuple, List

from config.master_config import MASTER_CONFIG
from services.data_fetch import _safe_float
from services.patterns.pattern_state_manager import (
    get_breakdown_state,
    save_breakdown_state,
    update_breakdown_state,
    delete_breakdown_state
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG EXTRACTION (with fallbacks)
# ============================================================

try:
    PHYSICS = MASTER_CONFIG["global"]["pattern_physics"]
    ENTRY_RULES = MASTER_CONFIG["global"]["pattern_entry_rules"]
    INVALIDATION_RULES = MASTER_CONFIG["global"]["pattern_invalidation"]
except (KeyError, TypeError) as e:
    logger.error(f"Failed to load pattern configs from MASTER_CONFIG: {e}")
    PHYSICS = {}
    ENTRY_RULES = {}
    INVALIDATION_RULES = {}


# ============================================================
# HELPER: Safe Config Access
# ============================================================

def get_horizon_config(horizon: str, *keys, default=None):
    """
    Safely access nested horizon config.
    
    Example:
        get_horizon_config("intraday", "execution", "stop_loss_atr_mult", default=2.0)
    """
    try:
        cfg = MASTER_CONFIG["horizons"].get(horizon, {})
        for key in keys:
            cfg = cfg.get(key, {})
        return cfg if cfg else default
    except (KeyError, TypeError, AttributeError):
        return default


def get_global_config(*keys, default=None):
    """
    Safely access nested global config.
    
    Example:
        get_global_config("calculation_engine", "divergence_detection")
    """
    try:
        cfg = MASTER_CONFIG["global"]
        for key in keys:
            cfg = cfg.get(key, {})
        return cfg if cfg else default
    except (KeyError, TypeError, AttributeError):
        return default


# ============================================================
# HELPER: Indicator Access with Multiple Fallbacks
# ============================================================

def get_indicator_value(indicators: Dict, *keys, default=None):
    """
    Tries multiple paths to find an indicator value.
    
    Example:
        get_indicator_value(indicators, "rsi", "value", default=50)
        # Tries: indicators["rsi"]["value"], indicators["rsi"], etc.
    """
    for key in keys:
        try:
            val = indicators.get(key)
            if val is None:
                continue
            
            # Handle nested dict
            if isinstance(val, dict):
                if "value" in val:
                    return _safe_float(val["value"])
                elif "raw" in val:
                    return _safe_float(val["raw"])
                else:
                    return _safe_float(val)
            else:
                return _safe_float(val)
        except (KeyError, TypeError, AttributeError):
            continue
    
    return default


# ============================================================
# 1. RR REGIME ADJUSTMENTS (ADX-Based Target Scaling)
# ============================================================

def apply_rr_regime_adjustments(
    targets: Dict[str, float],
    indicators: Dict[str, Any],
    horizon: str = "short_term"
) -> Dict[str, float]:
    """
    Adjusts T1/T2 based on ADX trend strength using MASTER_CONFIG.
    
    Strong Trend (ADX > 40): Stretch targets (2.0x, 4.0x)
    Normal Trend (ADX 20-40): Standard targets (1.5x, 3.0x)
    Weak Trend (ADX < 20): Tighten targets (1.2x, 2.5x)
    
    Args:
        targets: Dict with entry, stop_loss, t1, t2
        indicators: Technical indicators dict
        horizon: Trading timeframe
    
    Returns:
        Updated targets dict with regime metadata
    """
    try:
        # Get RR regime config
        rr_regime = get_horizon_config(horizon, "risk_management", "rr_regime_adjustments")
        
        if not rr_regime:
            logger.debug(f"No RR regime config for {horizon}, returning original targets")
            return targets
        
        # Get ADX value (try multiple paths)
        adx_val = (
            get_indicator_value(indicators, "adx", "value") or
            get_indicator_value(indicators, "adx") or
            0
        )
        
        # Determine regime
        regime = "normal_trend"
        if adx_val >= rr_regime.get("strong_trend", {}).get("adx_min", 40):
            regime = "strong_trend"
        elif adx_val < rr_regime.get("weak_trend", {}).get("adx_max", 20):
            regime = "weak_trend"
        
        # Get multipliers
        regime_cfg = rr_regime.get(regime, {})
        t1_mult = regime_cfg.get("t1_mult", 1.5)
        t2_mult = regime_cfg.get("t2_mult", 3.0)
        
        # Apply adjustments
        entry = targets.get("entry", 0)
        stop_loss = targets.get("stop_loss", 0)
        
        if not entry or not stop_loss:
            return targets
        
        risk = abs(entry - stop_loss)
        
        # Recalculate targets
        adjusted_targets = {
            "t1": round(entry + (risk * t1_mult), 2),
            "t2": round(entry + (risk * t2_mult), 2),
            "entry": entry,
            "stop_loss": stop_loss,
            "regime": regime,
            "adx": adx_val,
            "t1_mult": t1_mult,
            "t2_mult": t2_mult
        }
        
        logger.debug(f"RR regime '{regime}' applied: ADX={adx_val:.1f}, T1={adjusted_targets['t1']}, T2={adjusted_targets['t2']}")
        
        return adjusted_targets
    
    except Exception as e:
        logger.error(f"apply_rr_regime_adjustments failed: {e}", exc_info=True)
        return targets


# ============================================================
# 2. PATTERN EXPIRATION CHECK
# ============================================================

def check_pattern_expiration(
    indicators: Dict[str, Any],
    horizon: str = "short_term"
) -> Dict[str, Any]:
    """
    Checks if momentum patterns have expired based on MASTER_CONFIG rules.
    
    Patterns that expire:
    - flag_pennant: Max 20 candles (intraday), 10 days (short_term), 8 weeks (long_term)
    - three_line_strike: Max 10 candles (intraday), 8 days (short_term), 6 weeks (long_term)
    
    Returns:
        {
            "expired": bool,
            "pattern": str,
            "reason": str,
            "action": str,
            "age_candles": int
        }
    """
    result = {
        "expired": False,
        "pattern": None,
        "reason": None,
        "action": None,
        "age_candles": None
    }
    
    try:
        # Patterns that can expire
        expiring_patterns = ["flag_pennant", "three_line_strike"]
        
        for p_key in expiring_patterns:
            p_data = indicators.get(p_key, {})
            
            if not p_data.get("found"):
                continue
            
            # Get expiration config
            inval_cfg = INVALIDATION_RULES.get(p_key, {})
            exp_cfg = inval_cfg.get("expiration", {})
            max_candles_cfg = exp_cfg.get("max_duration_candles", {})
            max_candles = max_candles_cfg.get(horizon)
            
            if not max_candles:
                continue
            
            # Get pattern age (try multiple meta keys)
            meta = p_data.get("meta", {})
            age = (
                _safe_float(meta.get("age_candles")) or 
                _safe_float(meta.get("candles_since_formation")) or 
                _safe_float(meta.get("bars_since_breakout")) or
                _safe_float(meta.get("age"))
            )
            
            # If age not tracked, skip
            if not age:
                logger.debug(f"Pattern '{p_key}' has no age tracking in meta")
                continue
            
            if age > max_candles:
                result["expired"] = True
                result["pattern"] = p_key
                result["reason"] = f"{p_key.replace('_', ' ').title()} expired ({int(age)} > {max_candles} candles)"
                result["action"] = exp_cfg.get("action_on_expire", "DOWNGRADE_TO_CONSOLIDATION")
                result["age_candles"] = int(age)
                
                logger.info(f"Pattern expiration detected: {result['reason']}")
                break
    
    except Exception as e:
        logger.error(f"check_pattern_expiration failed: {e}", exc_info=True)
    
    return result


# ============================================================
# 3. PATTERN ENTRY VALIDATION
# ============================================================

def validate_pattern_entry(
    indicators: Dict[str, Any], 
    setup_type: str, 
    horizon: str = "short_term"
) -> Dict[str, Any]:
    """
    Checks if pattern-specific entry conditions are met using MASTER_CONFIG rules.
    
    Args:
        indicators: Technical indicators dict
        setup_type: Setup classification (e.g., "PATTERN_CUP_BREAKOUT")
        horizon: Trading timeframe
    
    Returns:
        {
            "confirmed": bool,
            "wait_for": str or None,
            "confidence_adjustment": int
        }
    """
    result = {"confirmed": True, "wait_for": None, "confidence_adjustment": 0}
    
    try:
        # Map generic setup types to config keys
        pattern_key = None
        if "CUP" in setup_type.upper():
            pattern_key = "cup_handle"
        elif "DARVAS" in setup_type.upper():
            pattern_key = "darvas_box"
        elif "VCP" in setup_type.upper() or "MINERVINI" in setup_type.upper():
            pattern_key = "minervini_stage2"
        elif "SQUEEZE" in setup_type.upper():
            pattern_key = "bollinger_squeeze"
        elif "FLAG" in setup_type.upper() or "PENNANT" in setup_type.upper():
            pattern_key = "flag_pennant"
        elif "STRIKE" in setup_type.upper():
            pattern_key = "three_line_strike"

        if not pattern_key:
            return result

        # Get Rule Config for this pattern & horizon
        rules = ENTRY_RULES.get(pattern_key, {}).get("horizons", {})
        rule_set = rules.get(horizon, rules.get("short_term", {}))
        
        if not rule_set:
            logger.debug(f"No entry rules for pattern '{pattern_key}' on horizon '{horizon}'")
            return result
        
        # Get Indicator Data
        pattern_data = indicators.get(pattern_key, {})
        if not pattern_data.get("found"):
            return result
            
        meta = pattern_data.get("meta", {})
        price = get_indicator_value(indicators, "price", "close", default=None)
        rvol = get_indicator_value(indicators, "rvol", "relative_volume", default=1.0)

        # ========================================================
        # PATTERN-SPECIFIC VALIDATIONS
        # ========================================================

        # --- A. CUP & HANDLE ---
        if pattern_key == "cup_handle":
            rim = _safe_float(meta.get("rim_level"))
            rim_clearance = rule_set.get("rim_clearance", 0.99)
            rvol_min = rule_set.get("rvol_min", 1.2)
            
            if rim and price:
                if price < rim * rim_clearance:
                    result["confirmed"] = False
                    result["wait_for"] = f"Price {price:.2f} < Rim {rim:.2f} × {rim_clearance:.2%}"
                    return result
                
                if rvol < rvol_min:
                    result["confirmed"] = False
                    result["wait_for"] = f"Volume {rvol:.2f}x < Required {rvol_min}x"
                    return result

                # Volume surge bonus
                if rvol > rule_set.get("rvol_bonus_threshold", 2.0):
                    result["confidence_adjustment"] = 10
                    logger.debug(f"Cup & Handle volume surge bonus: RVOL={rvol:.2f}")

        # --- B. DARVAS BOX ---
        elif pattern_key == "darvas_box":
            box_high = _safe_float(meta.get("box_high"))
            clearance = rule_set.get("box_clearance", 1.005)
            
            if box_high and price:
                if price < box_high * clearance:
                    result["confirmed"] = False
                    result["wait_for"] = f"Wait for breakout > {box_high * clearance:.2f}"
                    return result

        # --- C. VCP / STAGE 2 ---
        elif pattern_key == "minervini_stage2":
            contraction = _safe_float(meta.get("contraction_pct"))
            max_contraction = rule_set.get("contraction_max", 1.5)
            
            if contraction and contraction > max_contraction:
                result["wait_for"] = f"VCP loose ({contraction:.1f}%), prefer < {max_contraction}%"
                result["confidence_adjustment"] = -5
                logger.debug(f"VCP contraction warning: {contraction:.1f}%")

        # --- D. BOLLINGER SQUEEZE ---
        elif pattern_key == "bollinger_squeeze":
            rsi_min = rule_set.get("rsi_min", 50)
            rsi = get_indicator_value(indicators, "rsi", default=None)
            
            if rsi and rsi < rsi_min:
                result["confirmed"] = False
                result["wait_for"] = f"RSI {rsi:.0f} < {rsi_min} (Momentum Required)"
                return result

        # --- E. FLAG/PENNANT ---
        elif pattern_key == "flag_pennant":
            pole_length = _safe_float(meta.get("pole_length"))
            min_pole = rule_set.get("pole_length_min", 5)
            
            if pole_length and pole_length < min_pole:
                result["confidence_adjustment"] = -5
                logger.debug(f"Flag pole short: {pole_length} < {min_pole}")

        # --- F. THREE-LINE STRIKE ---
        elif pattern_key == "three_line_strike":
            strike_body = _safe_float(meta.get("strike_candle_body_pct"))
            min_body = rule_set.get("strike_candle_body_min", 0.6)
            
            if strike_body and strike_body < min_body:
                result["confirmed"] = False
                result["wait_for"] = f"Strike candle weak ({strike_body:.1%} < {min_body:.1%})"
                return result

        # ========================================================
        # DIVERGENCE CHECK (Applies to All Patterns)
        # ========================================================
        divergence_cfg = get_global_config("calculation_engine", "divergence_detection")
        severity_bands = divergence_cfg.get("severity_bands") if divergence_cfg else None
        
        if severity_bands:
            # Get RSI slope (try multiple paths)
            rsi_slope = (
                get_indicator_value(indicators, "rsi_slope") or
                get_indicator_value(indicators, "rsi", "slope") or
                None
            )
            
            if rsi_slope is not None:
                # Determine severity
                if rsi_slope < -0.08:
                    severity = "severe"
                elif rsi_slope < -0.03:
                    severity = "moderate"
                else:
                    severity = "minor"
            else:
                logger.debug("RSI slope not available for divergence check")
                
                band = severity_bands.get(severity, {})
                allow_entry = band.get("allow_entry", True)
                confidence_penalty = band.get("confidence_penalty", 1.0)
                
                if not allow_entry:
                    result["confirmed"] = False
                    result["wait_for"] = f"Severe Divergence Detected (RSI Slope: {rsi_slope:.3f})"
                    logger.warning(f"Pattern entry blocked by divergence: {rsi_slope:.3f}")
                    return result
                
                # Apply confidence penalty
                if confidence_penalty < 1.0:
                    penalty_pct = int((1.0 - confidence_penalty) * 100)
                    result["confidence_adjustment"] += -penalty_pct
                    logger.debug(f"Divergence penalty applied: -{penalty_pct}%")
    
    except Exception as e:
        logger.error(f"validate_pattern_entry failed: {e}", exc_info=True)
    
    return result


# ============================================================
# 4. PATTERN INVALIDATION MONITORING
# ============================================================

def check_pattern_invalidation(
    indicators: Dict[str, Any],
    position_type: str = "LONG",
    horizon: str = "short_term"
) -> Dict[str, Any]:
    """
    Monitors active patterns for failure using MASTER_CONFIG thresholds.
    
    Features:
    - Config-driven breakdown thresholds per horizon
    - Bollinger Squeeze expansion detection
    - False breakout recovery logic
    - Golden Cross → Death Cross detection
    
    Args:
        indicators: Technical indicators dict
        position_type: "LONG" or "SHORT"
        horizon: Trading timeframe
    
    Returns:
        {
            "invalidated": bool,
            "reason": str,
            "action": str,  # "EXIT_IMMEDIATELY", "EXIT_ON_CLOSE", "TIGHTEN_STOP"
            "pattern": str,
            "details": dict
        }
    """
    result = {
        "invalidated": False,
        "reason": None,
        "action": None,
        "pattern": None,
        "details": {}
    }
    
    try:
        if position_type not in ["LONG", "SHORT"]:
            return result
        
        price = get_indicator_value(indicators, "price", "close", default=None)
        if not price:
            return result
        
        # ========================================================
        # HELPER: Safe Condition Evaluator
        # ========================================================
        def safe_eval_condition(condition_str: str, price_val: float, ref_val: float) -> bool:
            """
            Safely evaluates simple comparison conditions.
            Supports: price < ref * multiplier
            """
            try:
                if "<" in condition_str:
                    parts = condition_str.split("<")
                    right_expr = parts[1].strip()
                    
                    # Handle multiplication (e.g., "box_low * 0.995")
                    if "*" in right_expr:
                        mult_parts = right_expr.split("*")
                        multiplier = float(mult_parts[1].strip().replace(")", ""))
                        threshold = ref_val * multiplier
                    else:
                        threshold = float(right_expr)
                    
                    return price_val < threshold
                
                elif ">" in condition_str:
                    parts = condition_str.split(">")
                    right_expr = parts[1].strip()
                    
                    if "*" in right_expr:
                        mult_parts = right_expr.split("*")
                        multiplier = float(mult_parts[1].strip().replace(")", ""))
                        threshold = ref_val * multiplier
                    else:
                        threshold = float(right_expr)
                    
                    return price_val > threshold
                
                return False
            
            except Exception as e:
                logger.debug(f"safe_eval_condition failed: {e}")
                return False
        
        # ========================================================
        # HELPER: Check Pattern Breakdown
        # ========================================================
        def check_breakdown(p_key: str, ref_level_key: str) -> bool:
            """
            Checks if pattern has broken down based on config thresholds.
            Returns True if invalidated (updates result dict).
            """
            nonlocal result
            
            # Currently only supporting LONG positions
            if position_type != "LONG":
                return False
            
            # Get pattern data
            p_data = indicators.get(p_key, {})
            if not p_data.get("found"):
                return False
            
            # Get Config Thresholds for this pattern + horizon
            inval_cfg = INVALIDATION_RULES.get(p_key, {})
            thresholds = inval_cfg.get("breakdown_threshold", {}).get(horizon, {})
            
            if not thresholds:
                return False
            
            # Extract Reference Level from Pattern Meta
            meta = p_data.get("meta", {})
            ref_level = _safe_float(meta.get(ref_level_key))
            
            if not ref_level:
                return False
            
            # ========================================================
            # SPECIAL CASE: Bollinger Squeeze
            # ========================================================
            if p_key == "bollinger_squeeze":
                bb_width = get_indicator_value(indicators, "bb_width", default=None)
                bb_mid = get_indicator_value(indicators, "bb_mid", default=None)
                bb_low = get_indicator_value(indicators, "bb_low", default=None)
                
                # Check False Breakout Recovery
                false_breakout_cfg = inval_cfg.get("false_breakout_reentry", {})
                if false_breakout_cfg.get("enabled"):
                    condition_str = false_breakout_cfg.get("condition", "")
                    
                    if "price > bb_mid" in condition_str and bb_mid:
                        if price > bb_mid:
                            result["invalidated"] = False
                            result["reason"] = "Squeeze False Breakout - Recovery"
                            result["action"] = "MONITOR"
                            result["pattern"] = p_key
                            result["details"] = {
                                "price": price,
                                "bb_mid": bb_mid,
                                "recovered": True
                            }
                            logger.info(f"Squeeze false breakout recovery at {price:.2f}")
                            return False
                
                # Check Squeeze Expansion (width explosion without breakout)
                or_condition = thresholds.get("or_condition", "")
                if "bb_width" in or_condition and bb_width:
                    try:
                        threshold_width = float(or_condition.split(">")[1].strip())
                        
                        if bb_width > threshold_width:
                            action = inval_cfg.get("action", {}).get(horizon, "EXIT_ON_CLOSE")
                            result["invalidated"] = True
                            result["reason"] = f"Squeeze Expansion Without Breakout"
                            result["action"] = action
                            result["pattern"] = p_key
                            result["details"] = {
                                "bb_width": bb_width,
                                "threshold": threshold_width,
                                "expansion": True
                            }
                            logger.warning(f"Squeeze invalidated: BB Width {bb_width:.2f} > {threshold_width:.2f}")
                            return True
                    except:
                        pass
                
                # Standard breakdown check (price < bb_low)
                if bb_low:
                    ref_level = bb_low
                    ref_level_key = "bb_low"
            
            # ========================================================
            # STANDARD BREAKDOWN CHECK
            # ========================================================

            condition_str = thresholds.get("condition", "")
            duration_candles = thresholds.get("duration_candles", 1)

            if not condition_str:
                return False

            # Evaluate condition using safe parser
            is_broken = safe_eval_condition(condition_str, price, ref_level)

            if is_broken:
                # ✅ PATCH 1: DURATION CANDLE TRACKING
                if duration_candles > 1:
                    # Multi-candle confirmation required
                    symbol = indicators.get("symbol", "UNKNOWN")  # Add symbol to indicators in signal_engine
                    
                    breakdown_state = get_breakdown_state(symbol, p_key, horizon)
                    
                    if not breakdown_state:
                        # First time breakdown detected
                        save_breakdown_state(
                            symbol=symbol,
                            pattern_name=p_key,
                            horizon=horizon,
                            price=price,
                            threshold=ref_level,
                            condition=condition_str
                        )
                        logger.info(f"{p_key} breakdown started on {symbol} (1/{duration_candles})")
                        return False  # Not invalidated yet
                    
                    else:
                        # Breakdown already in progress
                        count = breakdown_state.get("candle_count", 0)
                        
                        if count >= duration_candles:
                            # Duration requirement met → Invalidate
                            logger.warning(f"{p_key} breakdown CONFIRMED on {symbol} ({count}/{duration_candles})")
                            delete_breakdown_state(symbol, p_key, horizon)  # Clean up
                            # Continue to invalidation below...
                        
                        else:
                            # Still waiting for duration
                            update_breakdown_state(symbol, p_key, horizon)
                            logger.info(f"{p_key} breakdown progressing on {symbol} ({count + 1}/{duration_candles})")
                            return False  # Not invalidated yet
                
                # Single candle confirmation (duration_candles = 1) → Immediate invalidation
                action = inval_cfg.get("action", {}).get(horizon, "EXIT_ON_CLOSE")
                result["invalidated"] = True
                result["reason"] = f"{p_key.replace('_', ' ').title()} Breakdown"
                result["action"] = action
                result["pattern"] = p_key
                result["details"] = {
                    "price": price,
                    "threshold": ref_level,
                    "condition": condition_str,
                    "duration_required": duration_candles
                }
                
                logger.warning(f"Pattern invalidation: {p_key} at {price:.2f} vs {ref_level:.2f}")
                return True

            return False

        
        # ========================================================
        # CHECK ALL PATTERNS USING CONFIG
        # ========================================================
        pattern_refs = {
            "darvas_box": "box_low",
            "cup_handle": "handle_low",
            "flag_pennant": "flag_low",
            "minervini_stage2": "pivot_point",
            "bollinger_squeeze": "bb_low",
            "three_line_strike": "entry",
            "ichimoku_signals": "cloud_bottom",
            "double_top_bottom": "neckline"
        }
        
        for pattern_name, ref_key in pattern_refs.items():
            if check_breakdown(pattern_name, ref_key):
                return result
        
        # ========================================================
        # SPECIAL CHECK: Golden Cross → Death Cross
        # ========================================================
        golden_cross = indicators.get("golden_cross", {})
        if golden_cross.get("found"):
            # Try multiple MA key paths
            ema_50 = (
                get_indicator_value(indicators, "ema_50") or
                get_indicator_value(indicators, "ma_fast") or
                get_indicator_value(indicators, "mafast")
            )
            ema_200 = (
                get_indicator_value(indicators, "ema_200") or
                get_indicator_value(indicators, "ma_slow") or
                get_indicator_value(indicators, "maslow")
            )
            
            if ema_50 and ema_200 and ema_50 < ema_200:
                inval_cfg = INVALIDATION_RULES.get("golden_cross", {})
                action = inval_cfg.get("action", {}).get(horizon, "EXIT_ON_CLOSE")
                
                result["invalidated"] = True
                result["reason"] = "Death Cross (EMA 50 < EMA 200)"
                result["action"] = action
                result["pattern"] = "golden_cross"
                result["details"] = {
                    "ema_50": ema_50,
                    "ema_200": ema_200,
                    "reversal": True
                }
                
                logger.warning(f"Death Cross detected: EMA50={ema_50:.2f} < EMA200={ema_200:.2f}")
                return result
    
    except Exception as e:
        logger.error(f"check_pattern_invalidation failed: {e}", exc_info=True)
    # ✅ PATCH 1: CLEANUP STALE BREAKDOWN STATES
    # If pattern is NOT broken anymore, clean up state
    symbol = indicators.get("symbol", "UNKNOWN")
    for pattern_name in pattern_refs.keys():
        p_data = indicators.get(pattern_name, {})
        if p_data.get("found"):
            # Check if breakdown state exists but pattern is NOT invalidated
            state = get_breakdown_state(symbol, pattern_name, horizon)
            if state and not result["invalidated"]:
                # Pattern recovered → delete state
                delete_breakdown_state(symbol, pattern_name, horizon)
                logger.info(f"{pattern_name} recovered on {symbol} - state cleared")

    return result


# ============================================================
# 5. MAIN: ENHANCE PLAN WITH PATTERNS
# ============================================================

def enhance_plan_with_patterns(
    plan: Dict[str, Any],
    indicators: Dict[str, Any],
    horizon: str = "short_term"
) -> Dict[str, Any]:
    """
    Refines Trade Plan using MASTER_CONFIG pattern physics, entry rules, and invalidation logic.
    
    Process:
    1. Find best pattern (score > 60)
    2. Validate entry conditions
    3. Apply pattern-specific targets/stops
    4. Check for invalidation
    5. Apply RR regime adjustments
    6. Check for pattern expiration
    
    Args:
        plan: Trade plan dict from signal_engine
        indicators: Technical indicators dict
        horizon: Trading timeframe
    
    Returns:
        Enhanced plan dict with pattern metadata
    """
    def meta_num(key): 
        return _safe_float(meta.get(key))
    
    try:
        # ========================================================
        # 0. DEFENSIVE SETUP
        # ========================================================
        out = copy.deepcopy(plan) if plan is not None else {}
        out.setdefault("targets", {})
        out.setdefault("execution_hints", {})
        out.setdefault("analytics", {})

        entry = _safe_float(out.get("entry"))
        current_sl = _safe_float(out.get("stop_loss"))

        # Extract ATR (try multiple paths)
        atr_val = (
            get_indicator_value(indicators, "atr_dynamic") or
            get_indicator_value(indicators, "atr_14") or
            get_indicator_value(indicators, "atr") or
            get_indicator_value(indicators, "atr14")
        )

        if not atr_val or atr_val <= 0:
            out["execution_hints"]["note"] = "ATR missing or invalid - skipping pattern enhancement"
            logger.warning("ATR missing, cannot enhance plan with patterns")
            return out
        
        # ========================================================
        # 1. FIND BEST PATTERN (Score > 60)
        # ========================================================
        pattern_keys = [
            "darvas_box", "cup_handle", "bollinger_squeeze", 
            "flag_pennant", "minervini_stage2", "three_line_strike",
            "ichimoku_signals", "golden_cross", "double_top_bottom"
        ]

        valid_patterns = []
        for k in pattern_keys:
            p = indicators.get(k)
            if p and isinstance(p, dict) and p.get("found", False):
                # Try score at top level, then in meta
                score = (
                    _safe_float(p.get("score")) or
                    _safe_float(p.get("meta", {}).get("score"))
                )
                
                if score and score > 60:
                    valid_patterns.append((k, p, score))
                    logger.debug(f"Valid pattern found: {k} (score={score:.0f})")

        if not valid_patterns:
            logger.debug("No high-quality patterns found (score > 60)")
            return out

        # Sort by score (highest first)
        valid_patterns.sort(key=lambda x: x[2], reverse=True)
        best_name, best_pat, best_score = valid_patterns[0]
        meta = best_pat.get("meta", {}) or {}

        logger.info(f"Best pattern selected: {best_name} (score={best_score:.0f})")

        # ========================================================
        # 2. VALIDATE ENTRY CONDITIONS
        # ========================================================
        setup_type = out.get("setup_type", "")
        entry_validation = validate_pattern_entry(indicators, setup_type, horizon)
        
        if not entry_validation["confirmed"]:
            out["signal"] = "WAIT_PATTERN_ENTRY"
            out["reason"] = entry_validation["wait_for"]
            out["execution_hints"]["pattern_entry_wait"] = {
            "reason": entry_validation["wait_for"],
            "pattern": best_name
            }
            logger.info(f"Entry validation failed: {entry_validation['wait_for']}")
            return out
        if entry_validation["confidence_adjustment"] != 0:
            current_conf = out.get("setup_confidence", 50)
            new_conf = min(100, max(0, current_conf + entry_validation["confidence_adjustment"]))
            out["setup_confidence"] = new_conf
            logger.debug(f"Confidence adjusted: {current_conf} → {new_conf}")

        # ========================================================
        # 3. APPLY PATTERN PHYSICS (Targets & Stops)
        # ========================================================
        physics = PHYSICS.get(best_name, PHYSICS.get("default", {}))
        target_ratio = physics.get("target_ratio", 1.0)

        depth = None

        # --- DARVAS BOX ---
        if best_name == "darvas_box":
            box_high = meta_num("box_high")
            box_low = meta_num("box_low")
            if box_high and box_low:
                depth = box_high - box_low
                out["stop_loss"] = round(box_low * 0.995, 2)
                logger.debug(f"Darvas Box: depth={depth:.2f}, SL={out['stop_loss']}")

        # --- CUP & HANDLE ---
        elif best_name == "cup_handle":
            rim = meta_num("rim_level")
            depth_pct = meta_num("depth_pct")
            if rim and depth_pct:
                depth = rim * (depth_pct / 100.0)
                logger.debug(f"Cup & Handle: rim={rim:.2f}, depth={depth:.2f}")

        # --- FLAG / PENNANT ---
        elif best_name == "flag_pennant":
            pole_pct = meta_num("pole_gain_pct")
            if pole_pct and entry:
                depth = entry * (pole_pct / 100.0)
                logger.debug(f"Flag/Pennant: pole_gain={pole_pct:.1f}%, depth={depth:.2f}")

        # --- DOUBLE TOP / BOTTOM ---
        elif best_name == "double_top_bottom":
            target = meta_num("target")
            neckline = meta_num("neckline")
            if target and neckline:
                depth = abs(neckline - target)
                out["targets"]["t1"] = round(target, 2)
                logger.debug(f"Double Top/Bottom: T1={target:.2f}, neckline={neckline:.2f}")

        # --- GENERIC FALLBACK ---
        else:
            depth = meta_num("depth") or meta_num("height")

        # Apply target if depth resolved
        if depth and entry:
            t1 = round(entry + (depth * target_ratio), 2)
            t2 = round(entry + (depth * target_ratio * 2), 2)
            out["targets"]["t1"] = t1
            out["targets"]["t2"] = t2
            logger.debug(f"Pattern targets set: T1={t1}, T2={t2} (depth={depth:.2f}, ratio={target_ratio})")
            
        # ============================================================
        # NON-GEOMETRY PATTERN ROLES
        # ============================================================
        if best_name in ("bollinger_squeeze", "three_line_strike"):
            out["execution_hints"]["pattern_role"] = "momentum_confirmation"

        elif best_name in ("minervini_stage2",):
            out["execution_hints"]["pattern_role"] = "trend_continuation"

        elif best_name in ("ichimoku_signals", "golden_cross"):
            out["execution_hints"]["pattern_role"] = "regime_confirmation"

        # ========================================================
        # 4. STOP LOSS LOGIC (Pattern > ATR)
        # ========================================================
        new_sl = _safe_float(out.get("stop_loss"))
        is_short = (
            out.get("signal", "").startswith("SHORT") or 
            (entry and current_sl and current_sl > entry)
        )

        # If no pattern SL set, apply ATR-based SL
        if not new_sl and entry:
            sl_mult = get_horizon_config(horizon, "execution", "stop_loss_atr_mult", default=2.0)
            
            if is_short:
                new_sl = round(entry + (atr_val * sl_mult), 2)
            else:
                new_sl = round(entry - (atr_val * sl_mult), 2)
            
            out["stop_loss"] = new_sl
            logger.debug(f"ATR-based SL applied: {new_sl} (mult={sl_mult})")

        # Final directional sanity check
        new_sl = _safe_float(out.get("stop_loss"))
        if entry and new_sl:
            if is_short:
                # SHORT → SL must be ABOVE entry
                if new_sl <= entry:
                    out["stop_loss"] = current_sl if (current_sl and current_sl > entry) else None
                    logger.warning(f"SL sanity check failed (SHORT): {new_sl} <= {entry}")
            else:
                # LONG → SL must be BELOW entry
                if new_sl >= entry:
                    out["stop_loss"] = current_sl if (current_sl and current_sl < entry) else None
                    logger.warning(f"SL sanity check failed (LONG): {new_sl} >= {entry}")

        # ========================================================
        # 5. ANALYTICS & METADATA
        # ========================================================
        # ✅ PATCH 4: ENHANCED ANALYTICS
        out["analytics"]["pattern_driver"] = best_name
        out["analytics"]["pattern_score"] = best_score

        # 🆕 ADD PATTERN ROLE
        pattern_role = out["execution_hints"].get("pattern_role", "geometry")
        out["analytics"]["pattern_role"] = pattern_role

        # 🆕 ADD PATTERN METADATA FOR ANALYSIS
        out["analytics"]["pattern_meta"] = {
            "age_candles": meta.get("age_candles"),
            "depth": depth,
            "target_ratio": target_ratio,
            "physics_applied": bool(depth),
            "pattern_quality": best_pat.get("quality", 0),
            "formation_timestamp": meta.get("formation_timestamp")
        }

        # 🆕 ADD EXECUTION QUALITY SCORE
        execution_quality = 0
        if depth and depth > 0:
            execution_quality += 30  # Has geometry
        if out.get("stop_loss"):
            execution_quality += 20  # Has stop
        if out["targets"].get("t1"):
            execution_quality += 25  # Has target
        if best_score > 80:
            execution_quality += 25  # High pattern score

        out["analytics"]["execution_quality"] = execution_quality

        
        # Check invalidation
        pos_type = "SHORT" if is_short else "LONG"
        invalidation = check_pattern_invalidation(indicators, pos_type, horizon)
        if invalidation["invalidated"]:
            out["execution_hints"]["pattern_invalidation"] = invalidation
            logger.warning(f"Pattern invalidation detected: {invalidation['reason']}")

        # ========================================================
        # 6. APPLY RR REGIME ADJUSTMENTS (ADX-based)
        # ========================================================
        if out["targets"].get("t1") and out["targets"].get("t2"):
            regime_input = {
                "entry": entry,
                "stop_loss": out.get("stop_loss"),
                "t1": out["targets"]["t1"],
                "t2": out["targets"]["t2"]
            }
            
            adjusted = apply_rr_regime_adjustments(regime_input, indicators, horizon)
            
            if adjusted.get("regime"):
                out["targets"]["t1"] = adjusted["t1"]
                out["targets"]["t2"] = adjusted["t2"]
                out["analytics"]["rr_regime"] = adjusted["regime"]
                out["analytics"]["adx"] = adjusted["adx"]
                out["execution_hints"]["regime_note"] = (
                    f"ADX {adjusted['adx']:.0f} → {adjusted['regime'].upper()} "
                    f"(T1: {adjusted['t1_mult']}x, T2: {adjusted['t2_mult']}x)"
                )
                logger.info(f"RR regime applied: {adjusted['regime']} (ADX={adjusted['adx']:.0f})")

        # ========================================================
        # 7. CHECK PATTERN EXPIRATION
        # ========================================================
        expiration = check_pattern_expiration(indicators, horizon)
        if expiration["expired"]:
            out["execution_hints"]["pattern_expiration"] = expiration
            
            # Reduce confidence by 20%
            current_conf = out.get("setup_confidence", 50)
            out["setup_confidence"] = max(0, current_conf - 20)
            
            logger.warning(f"Pattern expired: {expiration['reason']}")

        logger.info(f"Plan enhancement complete: Pattern={best_name}, Signal={out.get('signal')}")

    except Exception as e:
        logger.error(f"enhance_plan_with_patterns failed: {e}", exc_info=True)
        out["execution_hints"]["enhancement_error"] = str(e)

    return out