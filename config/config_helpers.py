# services/config_helpers.py/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""
Business Logic Layer for Signal Engine
Bridges ConfigResolver (Data) and SignalEngine (Execution).
Uses the new consolidated 'get_signal_context' for efficiency.
"""

# services/config_helpers.py
"""
Fixed Config Helpers - Business Logic Layer
Now uses correct paths from ConfigResolver with proper horizon-first lookup.
"""

import logging
from typing import Dict, Any, Tuple, List
from config.config_resolver import get_config
from services.data_fetch import _get_val

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER: SAFE DATA EXTRACTION
# ==============================================================================


def _rule_matches(raw_val: float, op: str, target: float) -> bool:
    """Evaluate simple operator logic."""
    if op == "<": return raw_val < target
    if op == ">": return raw_val > target
    if op == "<=": return raw_val <= target
    if op == ">=": return raw_val >= target
    if op == "==": return raw_val == target
    return False


def _evaluate_condition(condition: str, indicators: Dict, setup_type: str, fundamentals: Dict = None) -> bool:
    """
    Parse string conditions like "rvol >= 2.5" or "roe > 15".
    Checks 'indicators' first, then 'fundamentals'.
    """
    try:
        parts = condition.split(" ")
        
        # Handle "setup_type in ['A', 'B']"
        if "setup_type" in condition:
            clean_setup = setup_type.upper()
            if "in" in parts:
                return clean_setup in condition 
            if "==" in parts:
                target = parts[-1].strip("'").strip('"')
                return clean_setup == target
            if "!=" in parts:
                target = parts[-1].strip("'").strip('"')
                return clean_setup != target
        
        # Handle numeric metric comparison
        if len(parts) >= 3:
            metric, op, target_str = parts[0], parts[1], parts[2]
            
            if metric == "setup_type":
                return False 
            
            # Try Indicators first
            val = _get_val(indicators, metric, None)
            
            # Then try Fundamentals
            if val is None and fundamentals:
                fund_val = fundamentals.get(metric)
                
                # If it's a dict with 'raw' key, use raw numeric value
                if isinstance(fund_val, dict):
                    val = fund_val.get('raw', None)
                else:
                    val = _get_val(fundamentals, metric, None)
            
            if val is None:
                return False


            target = float(target_str)
            return _rule_matches(val, op, target)
    except Exception:
        return False
    return False


# ==============================================================================
# 1. PENALTY LOGIC (FIXED)
# ==============================================================================

def apply_horizon_penalties(
    base_confidence: int,
    horizon: str,
    indicators: Dict,
    setup_type: str,
    fundamentals: Dict = None
) -> Dict[str, Any]:
    """
    Applies TWO types of penalties with FIXED path resolution:
    1. Simple Horizon Penalties (from scoring.penalties)
    2. Complex Setup Penalties (from setup_confidence.penalties)
    """
    ctx = get_config(horizon).get_signal_context()
    
    current_conf = base_confidence
    penalty_details = {}
    log_messages = []

    # --- A. Apply Simple Horizon Penalties ---
    simple_penalties = ctx.get("scoring", {}).get("penalties", {})
    
    for metric, rule in simple_penalties.items():
        val = _get_val(indicators, metric, None)
        if val is None and fundamentals:
            val = _get_val(fundamentals, metric, None)
        if val is None:
            continue

        op = rule.get("op") or rule.get("operator")
        limit = rule.get("val") or rule.get("value")
        penalty = rule.get("pen") or rule.get("penalty")

        if _rule_matches(val, op, limit):
            deduction = int(penalty * 100) if penalty < 1.0 else int(penalty)
            deduction = min(deduction, current_conf)
            
            if deduction > 0:
                current_conf -= deduction
                penalty_details[metric] = (deduction, f"{metric} {val:.2f} {op} {limit}")
                log_messages.append(f"Penalty: {metric} -> -{deduction}")

    # --- B. Apply Complex Setup Penalties ---
    # FIXED: Now correctly gets from ctx["setup_confidence"]["penalties"]
    complex_penalties = ctx.get("setup_confidence", {}).get("penalties", {})
    
    for name, rule in complex_penalties.items():
        condition = rule.get("condition", "")
        amount = rule.get("amount", 0)
        
        if _evaluate_condition(condition, indicators, setup_type, fundamentals):
            deduction = int(amount)
            deduction = min(deduction, current_conf)
            
            if deduction > 0:
                current_conf -= deduction
                reason = rule.get("reason", name)
                penalty_details[name] = (deduction, reason)
                log_messages.append(f"Setup Penalty: {name} -> -{deduction}")

    return {
        "adjusted_confidence": current_conf,
        "penalty_details": penalty_details,
        "log": "; ".join(log_messages) if log_messages else "No penalties applied"
    }


# ==============================================================================
# 2. ENHANCEMENT LOGIC (FIXED)
# ==============================================================================

def apply_horizon_enhancements(
    base_confidence: int,
    horizon: str,
    indicators: Dict,
    setup_type: str,
    fundamentals: Dict = None
) -> Dict[str, Any]:
    """
    Applies horizon-specific boosts/enhancements.
    FIXED: Now correctly gets enhancements from ctx.
    """
    ctx = get_config(horizon).get_signal_context()
    enhancements_config = ctx.get("enhancements", {})
    
    current_conf = base_confidence
    enhancement_details = {}
    log_messages = []

    for name, rule in enhancements_config.items():
        condition = rule.get("condition", "")
        boost_amount = rule.get("amount", 0)
        max_boost = rule.get("max_boost", boost_amount)
        
        if _evaluate_condition(condition, indicators, setup_type, fundamentals):
            actual_boost = min(int(boost_amount), int(max_boost))
            current_conf += actual_boost
            enhancement_details[name] = (actual_boost, rule.get("reason", "Enhancement"))
            log_messages.append(f"Boost: {name} -> +{actual_boost}")

    return {
        "adjusted_confidence": min(current_conf, 100),
        "enhancement_details": enhancement_details,
        "log": "; ".join(log_messages) if log_messages else "No enhancements"
    }


# ==============================================================================
# 3. GATE VALIDATION (FIXED)
# ==============================================================================
def validate_horizon_entry_gates(
    horizon: str,
    indicators: Dict,
    fundamentals: Dict = None,
    confidence: float = None,
    setup_type: str = None
) -> Dict[str, Any]:
    """
    Validates hard gates with setup-specific overrides AND resolver fallbacks.
    
    ✅ ENHANCED: Now uses ConfigResolver methods as intelligent fallbacks:
    - calculate_dynamic_confidence_floor() for ADX-based confidence
    - should_trade_volatility() for unified volatility checks
    
    Args:
        horizon: Trading horizon
        indicators: Technical indicators dict
        fundamentals: Fundamental data (optional)
        confidence: Optional confidence score
        setup_type: Optional setup type for gate overrides
    
    Returns:
        Dict with passed status, failures, gates log
    """
    config = get_config(horizon, indicators=indicators, fundamentals=fundamentals)
    ctx = config.get_signal_context()
    gates_cfg = ctx.get("gates", {})
    curr_adx = None
    curr_trend = None
    
    # ✅ Get setup-specific overrides
    overrides = {}
    if setup_type:
        all_overrides = gates_cfg.get("setup_gate_overrides", {})
        overrides = all_overrides.get(setup_type, {})
        if overrides:
            applied = []
            for gate_name, gate_value in overrides.items():
                if gate_value is None:
                    applied.append(f"{gate_name}=SKIP")
                else:
                    applied.append(f"{gate_name}={gate_value}")
            logger.info(f"[{horizon}] Setup {setup_type} overrides: {', '.join(applied)}")
    
    failures = []
    gates_log = {}

    # 1. Confidence Gate (if confidence provided)
    if confidence is not None:
        # Check for explicit setup-specific override FIRST
        if setup_type and "confidence_min" in overrides:
            conf_min = overrides["confidence_min"]
            if conf_min is None:
                # Skip gate entirely for accumulation plays
                gates_log["confidence"] = "SKIPPED (Accumulation Setup)"
                logger.info(f"[{horizon}] Confidence gate skipped for {setup_type}")
            elif confidence < conf_min:
                status = "FAIL"
                failures.append(f"Confidence {confidence:.1f} < Override {conf_min}")
                gates_log["confidence"] = status
            else:
                gates_log["confidence"] = "PASS"
        else:
            # Use dynamic floor as fallback
            adx_val = _get_val(indicators, "adx", 0)
            required_floor = config.calculate_dynamic_confidence_floor(adx_val, setup_type)
            
            if confidence < required_floor:
                status = "FAIL"
                failures.append(f"Confidence {confidence:.1f}% < Dynamic Floor {required_floor:.1f}%")
            else:
                status = "PASS"
            gates_log["confidence"] = status

    # 2. Trend Strength Gate (with override support)
    min_trend = overrides.get("min_trend_strength", gates_cfg.get("min_trend_strength", 0))
    if min_trend is None:
        # ✅ None means skip this gate completely
        gates_log["trend_strength"] = "SKIPPED (Accumulation Setup)"
        logger.info(f"[{horizon}] Trend gate skipped for {setup_type}")
    elif min_trend > 0:
        curr_adx = _get_val(indicators, "adx", 0)
        curr_trend = _get_val(indicators, "trend_strength", 0)
        if curr_trend < min_trend:
            status = "FAIL"
            failures.append(f"Trend {curr_trend:.1f} < {min_trend}")
        else:
            status = "PASS"
        gates_log["trend_strength"] = status
    else:
        gates_log["trend_strength"] = "SKIPPED"

    # 3. ADX Gate (with override support)
    adx_min = overrides.get("adx_min", gates_cfg.get("adx_min", 0))
    
    if adx_min is None:
        gates_log["adx"] = "SKIPPED (Accumulation Setup)"
        logger.info(f"[{horizon}] ADX gate skipped for {setup_type}")
    elif adx_min > 0:
        curr_adx = _get_val(indicators, "adx", 0)
        if curr_adx < adx_min:
            status = "FAIL"
            failures.append(f"ADX {curr_adx:.1f} < {adx_min}")
        else:
            status = "PASS"
        gates_log["adx"] = status
    else:
        gates_log["adx"] = "SKIPPED"

    # ==========================================
    # 4. VOLATILITY - ENHANCED VERSION (replaces Steps 4 & 5)
    # ==========================================
    # Check for volatility_quality_min override first
    if setup_type and "volatility_quality_min" in overrides:
        vol_qual_min = overrides["volatility_quality_min"]
        if vol_qual_min is None:
            gates_log["volatility_regime"] = "SKIPPED (Accumulation Setup)"
            logger.info(f"[{horizon}] Volatility gate skipped for {setup_type}")
        else:
            # Manual quality check with override
            vol_qual = _get_val(indicators, "volatility_quality", 0)
            if vol_qual < vol_qual_min:
                failures.append(f"Vol Quality {vol_qual:.1f} < Override {vol_qual_min}")
                gates_log["volatility_regime"] = "FAIL"
            else:
                gates_log["volatility_regime"] = "PASS"
    else:
        # Use unified resolver method as fallback
        can_trade_vol, vol_reason = config.should_trade_volatility(indicators, setup_type)
        
        if not can_trade_vol:
            status = "FAIL"
            failures.append(f"volatility: {vol_reason}")
        else:
            status = "PASS"
        gates_log["volatility_regime"] = status

    horizon_entry_gate_validated = {
        "passed": len(failures) == 0,
        "failures": failures,
        "gates": gates_log,
        "overrides_applied": bool(overrides),
        "setup_type": setup_type,
        "log": "Gates Passed" if not failures else f"Gates Failed: {', '.join(failures)}",
        "_debug": {
            "gate_values": {
                "adx": {
                    "actual": curr_adx,
                    "threshold": adx_min,
                    "source": "override" if "adx_min" in overrides else "horizon_default"
                },
                "trend": {
                    "actual": curr_trend,
                    "threshold": min_trend,
                    "source": "override" if "min_trend_strength" in overrides else "horizon_default"
                }
            }
        }
    }
    logger.debug(f"validate_horizon_entry_gates in helpers log: {horizon_entry_gate_validated}")

    return horizon_entry_gate_validated
# def validate_horizon_entry_gates(
#     horizon: str,
#     indicators: Dict,
#     confidence: float = None  # Now optional, added back for compatibility
# ) -> Dict[str, Any]:
#     """
#     Validates hard gates (ADX, Trend Strength, Volatility Bands).
#     FIXED: Now correctly handles all gate types including confidence_min.
#     """
#     ctx = get_config(horizon).get_signal_context()
#     gates_cfg = ctx.get("gates", {})
    
#     failures = []
#     gates_log = {}

#     # 1. Confidence Gate (if confidence provided)
#     if confidence is not None:
#         conf_min = gates_cfg.get("confidence_min", 0)
#         if conf_min > 0 and confidence < conf_min:
#             status = "FAIL"
#             failures.append(f"Confidence {confidence:.1f} < {conf_min}")
#         else:
#             status = "PASS"
#         gates_log["confidence"] = status

#     # 2. Trend Strength Gate
#     min_trend = gates_cfg.get("min_trend_strength", 0)
#     curr_trend = _get_val(indicators, "trend_strength", 0)
    
#     if min_trend and curr_trend < min_trend:
#         status = "FAIL"
#         failures.append(f"Trend {curr_trend:.1f} < {min_trend}")
#     else:
#         status = "PASS"
#     gates_log["trend_strength"] = status

#     # 3. ADX Gate
#     adx_min = gates_cfg.get("adx_min", 0)
#     curr_adx = _get_val(indicators, "adx", 0)
#     if adx_min and curr_adx < adx_min:
#         status = "FAIL"
#         failures.append(f"ADX {curr_adx:.1f} < {adx_min}")
#     else:
#         status = "PASS"
#     gates_log["adx"] = status

#     # 4. Volatility Quality Gate
#     vol_qual_min = gates_cfg.get("volatility_quality_min", 0)
#     vol_qual = _get_val(indicators, "volatility_quality", 0)
#     if vol_qual_min and vol_qual < vol_qual_min:
#         status = "FAIL"
#         failures.append(f"Vol Quality {vol_qual:.1f} < {vol_qual_min}")
#     else:
#         status = "PASS"
#     gates_log["volatility_quality"] = status

#     # 5. Volatility Bands (Safety) - FIXED: Now uses normalized dict format
#     bands = gates_cfg.get("volatility_bands", {})
#     atr_pct = _get_val(indicators, "atr_pct", 0)
    
#     if bands:
#         if atr_pct < bands.get("min", 0):
#             status = "FAIL"
#             failures.append(f"Vol too low ({atr_pct:.2f} < {bands.get('min')})")
#         elif atr_pct > bands.get("max", 99):
#             status = "FAIL"
#             failures.append(f"Vol too high ({atr_pct:.2f} > {bands.get('max')})")
#         else:
#             status = "PASS"
#         gates_log["volatility_band"] = status

#     return {
#         "passed": len(failures) == 0,
#         "failures": failures,
#         "gates": gates_log,
#         "log": "Gates Passed" if not failures else f"Gates Failed: {', '.join(failures)}"
#     }


# ==============================================================================
# 4. SETUP CLASSIFICATION (FIXED)
# ==============================================================================

def classify_setup(
    indicators: Dict,
    fundamentals: Dict = None,
    horizon: str = "short_term"
) -> Tuple[str, int]:
    """
    Setup classification using FIXED path resolution.
    Now correctly gets rules from calculation_engine.
    """
    config = get_config(horizon)
    rules = config.get_setup_rules()  # FIXED: Now uses correct path
    candidates = []
    
    logger.debug(f"[{horizon}] Classifying setup from these rules set {rules}")
    
    for setup_name, rule_cfg in rules.items():
        if config.evaluate_setup_condition(setup_name, indicators, fundamentals):
            priority = rule_cfg.get("priority", 50)
            candidates.append((setup_name, priority))
            logger.debug(f"  ✓ {setup_name} matched (priority={priority})")
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_setup, best_priority = candidates[0]
        logger.info(f"[{horizon}] Setup classified: {best_setup} (priority={best_priority})")
        return (best_setup, best_priority)
    
    logger.warning(f"[{horizon}] No setup matched, defaulting to GENERIC")
    return ("GENERIC", 0)


# ==============================================================================
# 5. POSITION SIZING (FIXED)
# ==============================================================================

def calculate_position_size(
    horizon: str,
    indicators: Dict,
    confidence: float,
    setup_type: str
) -> float:
    """
    Config-driven position sizing with smart inheritance via resolver methods.
    Formula: base_risk × conf_factor × setup_mult × vol_mult × strategy_mult
    Args:
        horizon: Trading horizon (intraday, short_term, long_term, multibagger)
        indicators: Technical indicators (needs volatility_quality)
        confidence: Final confidence score (0-100)
        setup_type: Setup classification (e.g., VALUE_TURNAROUND, MOMENTUM_BREAKOUT)
    Returns:
        Position size as fraction of capital (e.g., 0.0102 = 1.02%)
    """
    config = get_config(horizon)
    
    # ✅ METHOD 1: Use get_position_sizing_config() for all parameters
    ps_config = config.get_position_sizing_config()
    
    base_risk = ps_config["base_risk_pct"]
    max_pos = ps_config["max_position_pct"]
    
    logger.debug( f"[{horizon}] Position sizing config: " f"base_risk={base_risk:.4f}, max_pos={max_pos:.4f}")
    # ✅ METHOD 2: Use get_setup_multiplier()
    setup_mult = config.get_setup_multiplier(setup_type)
    
    # ✅ METHOD 3: Use get_volatility_multiplier()
    vol_qual = _get_val(indicators, "volatility_quality", 5.0)
    vol_mult = config.get_volatility_multiplier(vol_qual)
    
    # Calculate confidence factor
    conf_factor = confidence / 100.0
    
    # Calculate base position (before strategy multiplier)
    base_position = base_risk * conf_factor * setup_mult * vol_mult
    
    logger.debug(
        f"[{horizon}] Base position: {base_position:.4f} "
        f"(base_risk={base_risk:.4f} × conf={conf_factor:.2f} × "
        f"setup={setup_mult:.2f} × vol={vol_mult:.2f})"
    )
    
    # ✅ METHOD 4: Use get_strategy_sizing_multiplier() (already exists)
    strategy_mult = config.get_strategy_sizing_multiplier(setup_type)
    
    position = base_position * strategy_mult
    
    logger.info(
        f"[{horizon}] After strategy multiplier ({strategy_mult:.2f}): "
        f"{base_position:.4f} → {position:.4f}"
    )
    
    # Cap at max position
    final_position = min(position, max_pos)
    
    if final_position < position:
        logger.warning(
            f"[{horizon}] Position capped: {position:.4f} → {final_position:.4f} "
            f"(max={max_pos:.4f})"
        )
    
    # Final summary
    logger.info(
        f"[{horizon}] Position size: {final_position:.4f} "
        f"({final_position*100:.2f}% of capital) | "
        f"base={base_risk:.4f}, conf={conf_factor:.2f}, "
        f"setup={setup_mult:.2f}, vol={vol_mult:.2f}, "
        f"strategy={strategy_mult:.2f}"
    )
    return round(final_position, 4)


# ==============================================================================
# 6. TARGET CALCULATION WITH RESISTANCE (FIXED)
# ==============================================================================

def calculate_targets_with_resistance(
    horizon: str,
    entry: float,
    stop_loss: float,
    indicators: Dict,
    resistance_levels: List[float] = None
) -> Tuple[float, float, Dict]:
    """
    Target calculation with resistance cushioning.
    FIXED: Now uses correct path for RR multipliers.
    """
    config = get_config(horizon)
    
    # Calculate risk
    risk = abs(entry - stop_loss)
    
    # Get ADX-based R:R multipliers (FIXED: now uses correct method)
    adx = _get_val(indicators, "adx", 20.0)
    rr_mults = config.get_rr_multipliers(adx)
    
    t1_mult = rr_mults.get("t1_mult", 1.5)
    t2_mult = rr_mults.get("t2_mult", 3.0)
    
    # Calculate raw targets
    raw_t1 = entry + (risk * t1_mult)
    raw_t2 = entry + (risk * t2_mult)
    
    logger.debug(
        f"[{horizon}] Raw targets: T1={raw_t1:.2f} ({t1_mult}R), "
        f"T2={raw_t2:.2f} ({t2_mult}R)"
    )
    
    # Get resistance adjustment parameters (global)
    cushion = config.get("targets.resistance_cushion", 0.96)
    min_dist_pct = config.get("targets.min_distance_pct", 0.005)
    
    adjustments = []
    final_t1 = raw_t1
    final_t2 = raw_t2
    
    # Apply resistance adjustments
    if resistance_levels:
        resistance_levels = sorted(resistance_levels)
        
        # Adjust T1
        for res_level in resistance_levels:
            if entry < res_level <= raw_t1:
                dist_to_res = (res_level - entry) / entry
                if dist_to_res < min_dist_pct:
                    logger.warning(f"  T1 too close to resistance {res_level:.2f}")
                    continue
                
                cushioned_t1 = res_level * cushion
                if cushioned_t1 < final_t1:
                    final_t1 = cushioned_t1
                    adjustments.append(f"T1 cushioned to {final_t1:.2f} (R={res_level:.2f})")
                    logger.info(f"  T1 adjusted for resistance: {final_t1:.2f}")
                break
        
        # Adjust T2
        for res_level in resistance_levels:
            if entry < res_level <= raw_t2:
                dist_to_res = (res_level - entry) / entry
                if dist_to_res < min_dist_pct:
                    continue
                
                cushioned_t2 = res_level * cushion
                if cushioned_t2 < final_t2:
                    final_t2 = cushioned_t2
                    adjustments.append(f"T2 cushioned to {final_t2:.2f} (R={res_level:.2f})")
                    logger.info(f"  T2 adjusted for resistance: {final_t2:.2f}")
                break
    
    # Ensure T1 < T2
    if final_t1 >= final_t2:
        logger.warning(f"  T1 >= T2 after adjustments, resetting T2")
        final_t2 = raw_t2
    
    metadata = {
        "raw_t1": raw_t1,
        "raw_t2": raw_t2,
        "final_t1": final_t1,
        "final_t2": final_t2,
        "t1_mult": t1_mult,
        "t2_mult": t2_mult,
        "adx": adx,
        "adjustments": adjustments,
        "resistance_cushion": cushion,
        "min_distance_pct": min_dist_pct
    }
    
    logger.info(
        f"[{horizon}] Final targets: T1={final_t1:.2f}, T2={final_t2:.2f} "
        f"({len(adjustments)} adjustments)"
    )
    
    return (round(final_t1, 2), round(final_t2, 2), metadata)


# ==============================================================================
# 7. MA HELPERS (FIXED)
# ==============================================================================

def get_ma_keys_config(horizon: str) -> Dict[str, str]:
    """
    Returns MA keys for the horizon.
    FIXED: Now correctly gets from signal context.
    """
    ctx = get_config(horizon).get_signal_context()
    
    ma_cfg = ctx.get("moving_averages", {})
    ma_type = ma_cfg.get("type", "EMA").lower()
    
    fast = ma_cfg.get("fast", 20)
    mid = ma_cfg.get("mid", 50)
    slow = ma_cfg.get("slow", 200)
    
    return {
        "fast": f"{ma_type}_{fast}",
        "mid": f"{ma_type}_{mid}",
        "slow": f"{ma_type}_{slow}"
    }


# ==============================================================================
# 8. ENHANCED CONDITION EVALUATOR (FIXED)
# ==============================================================================
def _evaluate_condition_enhanced(
    condition: str,
    indicators: Dict,
    setup_type: str,
    horizon: str,
    fundamentals: Dict = None,
    pattern_indicators: Dict = None
) -> bool:
    """
    Enhanced condition evaluator with pattern support.
    FIXED: Now correctly evaluates complex conditions.
    """
    if not condition:
        return False
    
    try:
        # Count detected patterns
        pattern_count = 0
        if pattern_indicators:
            pattern_keys = [
                "bollinger_squeeze", "cup_handle", "darvas_box",
                "flag_pennant", "minervini_stage2", "golden_cross",
                "double_top_bottom", "three_line_strike", "ichimoku_signals"
            ]
            for key in pattern_keys:
                pat = pattern_indicators.get(key, {})
                if isinstance(pat, dict) and pat.get("found"):
                    pattern_count += 1
        
        # Build evaluation context
        eval_context = {
            # Technical Composites
            "volatility_quality": _get_val(indicators, "volatility_quality"),
            "trend_strength": _get_val(indicators, "trend_strength"),
            "momentum_strength": _get_val(indicators, "momentum_strength"),
            
            # Technical Indicators
            "rsi": _get_val(indicators, "rsi"),
            "macd_hist": _get_val(indicators, "macd_hist"),
            "adx": _get_val(indicators, "adx"),
            "atr_pct": _get_val(indicators, "atr_pct"),
            "bb_width": _get_val(indicators, "bb_width"),
            "rvol": _get_val(indicators, "rvol"),
            "price_vs_primary_trend_pct": _get_val(indicators, "price_vs_primary_trend_pct"),
            "rel_strength_nifty": _get_val(indicators, "rel_strength_nifty"),
            
            # Pattern Data
            "pattern_count": pattern_count,
            
            # Setup Context
            "setup_type": setup_type or "UNKNOWN",
            "horizon": horizon
        }
        
        # Add fundamentals
        if fundamentals:
            # Handle nested fundamental structure
            for key in ['roe', 'roce', 'roic', 'de_ratio', 'pe_ratio', 'peg_ratio', 
                       'market_cap', 'eps_growth_5y', 'revenue_growth_5y', 'earnings_stability']:
                fund_val = fundamentals.get(key)
                
                # If it's a dict with 'raw' key, use raw numeric value
                if isinstance(fund_val, dict):
                    eval_context[key] = fund_val.get('raw', 0) or 0
                else:
                    # Use _get_val for safe conversion
                    eval_context[key] = _get_val(fundamentals, key, 0)
        
        # Safely evaluate
        result = eval(condition, {"__builtins__": {}}, eval_context)
        return bool(result)
    
    except Exception as e:
        logger.warning(f"[{horizon}] Error evaluating condition '{condition}': {e}")
        return False

def validate_strategy_and_fundamentals(
    horizon: str,
    setuptype: str,
    fundamentals: Dict[str, Any] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate setup against strategy preferences and fundamental filters.
    
    This consolidates:
    1. Strategy blocked setups check
    2. Fundamental filter application
    
    Args:
        horizon: Trading horizon
        setuptype: Setup classification
        fundamentals: Fundamental data (optional)
    
    Returns:
        (passed: bool, reason: str, metadata: dict)
    
    Example:
        >>> # Intraday blocks deep value plays
        >>> passed, reason, meta = validate_strategy_and_fundamentals(
        ...     "intraday", "DEEP_VALUE_PLAY", {}
        ... )
        >>> passed
        False
        >>> reason
        "Setup DEEP_VALUE_PLAY blocked by intraday strategy preferences"
        
        >>> # Multibagger requires high ROE
        >>> passed, reason, meta = validate_strategy_and_fundamentals(
        ...     "multibagger", "VALUE_TURNAROUND", {"roe": 10, "roce": 12}
        ... )
        >>> passed
        False
        >>> "ROE" in reason
        True
    """
    config = get_config(horizon)
    metadata = {
        "strategy_allowed": False,
        "fundamental_filter": {"passed": True, "failures": []}
    }
    
    # Check 1: Strategy blocked setups
    if not config.is_setup_allowed(setuptype):
        reason = f"Setup {setuptype} blocked by {horizon} strategy preferences"
        logger.warning(f"{horizon}: {reason}")
        return (False, reason, metadata)
    
    metadata["strategy_allowed"] = True
    logger.debug(f"{horizon}: Setup {setuptype} allowed by strategy ✓")
    
    # Check 2: Fundamental filters - NOW REQUIRED for long_term/multibagger
    if fundamentals:
        fund_pass, fund_failures = config.apply_fundamental_filters(fundamentals)
        metadata["fundamental_filter"] = {
            "passed": fund_pass,
            "failures": fund_failures
        }
        
        if not fund_pass:
            reason = f"Fundamental filters failed: {', '.join(fund_failures)}"
            logger.warning(f"{horizon}: {reason}")
            return (False, reason, metadata)
    elif horizon in ["long_term", "multibagger"]:
        return False, f"Fundamental data required for {horizon} horizon", metadata
        
    logger.info(f"{horizon}: Fundamental filters passed ✓")
    
    return (True, "Strategy and fundamental checks passed", metadata)

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

logger.info("🛠️ Config Helpers Module (FIXED) initialized")
logger.info("✅ Now uses correct path resolution with horizon-first lookup")