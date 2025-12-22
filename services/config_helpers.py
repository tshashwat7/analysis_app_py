# services/config_helpers.py
"""
Config Helper Module - Complete Structure

This file contains:
1. Safe config getters
2. Penalty appliers
3. Enhancement appliers
4. ✅ _evaluate_condition_enhanced (LIVES HERE, NOT IN MASTER_CONFIG)
"""

import logging
from typing import Dict, Any, Optional, Tuple
from config.constants import (
    MASTER_CONFIG, ConfigGuard,
    # Old constants (fallbacks)
    VOL_BANDS, ATR_MULTIPLIERS, RSI_SLOPE_THRESH, MACD_MOMENTUM_THRESH,
    TREND_THRESH, HORIZON_PROFILE_MAP, VALUE_WEIGHTS, GROWTH_WEIGHTS,
    QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, STOCH_FAST, STOCH_SLOW, STOCH_THRESHOLDS
)

logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: Initialize MASTER_CONFIG with Safety Wrapper
# ============================================================================
try:
    MASTER = ConfigGuard(MASTER_CONFIG)
    CONFIG_LOADED_FROM_MASTER = True
    logger.info("✅ MASTER_CONFIG loaded via ConfigGuard")
except Exception as e:
    CONFIG_LOADED_FROM_MASTER = False
    logger.error(f"❌ Failed to load MASTER_CONFIG: {e}")
    MASTER = None

# ============================================================================
# STEP 2: Fallback Configuration Map
# ============================================================================
# Maps old constant names to their values
# Used when MASTER_CONFIG key is missing

FALLBACK_CONFIG = {
    "volatility_bands": VOL_BANDS,
    "atr_multipliers": ATR_MULTIPLIERS,
    "rsi_slope_thresholds": RSI_SLOPE_THRESH,
    "macd_thresholds": MACD_MOMENTUM_THRESH,
    "trend_thresholds": TREND_THRESH,
    "fundamental_weights": {
        "value": VALUE_WEIGHTS,
        "growth": GROWTH_WEIGHTS,
        "quality": QUALITY_WEIGHTS,
        "momentum": MOMENTUM_WEIGHTS,
    },
    "stoch_fast": STOCH_FAST,
    "stoch_slow": STOCH_SLOW,
    "stoch_thresholds": STOCH_THRESHOLDS,
}

logger.info(f"📦 Fallback config map prepared with {len(FALLBACK_CONFIG)} keys")

# ============================================================================
# STEP 3: Safe Getters (Always Return Valid Data)
# ============================================================================

def get_horizon_config(horizon: str) -> Dict[str, Any]:
    """
    Get horizon-specific configuration from MASTER_CONFIG.
    
    Returns: ConfigGuard-wrapped dict (safe access)
    Fallback: Empty dict (won't crash)
    
    Usage:
        cfg = get_horizon_config("short_term")
        atr_mult = safe_get(cfg, ["indicators", "atr_period"], default=14)
    """
    if not CONFIG_LOADED_FROM_MASTER or MASTER is None:
        logger.debug(f"[{horizon}] MASTER_CONFIG unavailable, returning empty")
        return {}
    
    try:
        horizon_cfg = MASTER["horizons"][horizon]
        logger.debug(f"[{horizon}] Loaded from MASTER_CONFIG")
        return horizon_cfg
    except KeyError as e:
        logger.warning(f"[{horizon}] Key missing from MASTER_CONFIG: {e}")
        return {}


def safe_get(config_obj: Any, keys: list, default: Any = None) -> Any:
    """
    Safely traverse nested config dict.
    
    Example:
        value = safe_get(cfg, ["volatility", "scoring_thresholds", "high"], default=8.0)
    
    Returns: Value if found, else default (never crashes)
    """
    if config_obj is None:
        return default
    
    current = config_obj
    path = " -> ".join(keys)
    
    try:
        for key in keys:
            if isinstance(current, ConfigGuard):
                current = current[key]  # ConfigGuard handles missing keys safely
            elif isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    logger.debug(f"Config path [{path}] not found, using default")
                    return default
            else:
                logger.debug(f"Config path [{path}] - intermediate value not dict, using default")
                return default
        
        logger.debug(f"Config path [{path}] = {current}")
        return current
    
    except Exception as e:
        logger.debug(f"Config path [{path}] failed: {e}, using default")
        return default


def get_config_with_fallback(config_key: str, horizon: str = None, default: Any = None) -> Tuple[Any, str]:
    """
    Get config value trying MASTER first, then fallback.
    
    Returns: (value, source)
    source: "MASTER_CONFIG" | "FALLBACK_CONSTANTS" | "DEFAULT"
    
    Example:
        atr_mult, source = get_config_with_fallback("atr_multiplier", horizon="short_term")
        logger.info(f"Using {source} for ATR multiplier")
    """
    
    # Try MASTER_CONFIG first
    if CONFIG_LOADED_FROM_MASTER and MASTER is not None:
        try:
            if horizon:
                cfg = MASTER["horizons"][horizon]
                val = safe_get(cfg, [config_key])
                if val is not None:
                    logger.debug(f"[{horizon}] {config_key} from MASTER_CONFIG = {val}")
                    return val, "MASTER_CONFIG"
            else:
                # Global config
                val = safe_get(MASTER, ["global", config_key])
                if val is not None:
                    logger.debug(f"{config_key} from MASTER_CONFIG = {val}")
                    return val, "MASTER_CONFIG"
        except Exception as e:
            logger.debug(f"Failed to get {config_key} from MASTER: {e}")
    
    # Fall back to old constants
    if config_key in FALLBACK_CONFIG:
        val = FALLBACK_CONFIG[config_key]
        logger.info(f"[{horizon or 'GLOBAL'}] {config_key} from FALLBACK_CONSTANTS (old)")
        return val, "FALLBACK_CONSTANTS"
    
    # Use default
    logger.warning(f"[{horizon or 'GLOBAL'}] {config_key} not found anywhere, using default: {default}")
    return default, "DEFAULT"


# ============================================================================
# STEP 4: Validation Helpers
# ============================================================================

def validate_config_completeness() -> Dict[str, Any]:
    """
    Check if MASTER_CONFIG has all required keys.
    
    Returns: {
        "complete": bool,
        "missing_keys": [list],
        "status": "OK" | "PARTIAL" | "MISSING_MASTER"
    }
    
    Usage:
        status = validate_config_completeness()
        if not status["complete"]:
            logger.warning(f"Config incomplete: {status['missing_keys']}")
    """
    
    if not CONFIG_LOADED_FROM_MASTER or MASTER is None:
        return {
            "complete": False,
            "status": "MISSING_MASTER",
            "note": "MASTER_CONFIG not loaded, using FALLBACK_CONSTANTS only"
        }
    
    required_keys = [
        "global",
        "horizons",
        "global.calculation_engine",
        "global.pattern_physics",
        "global.pattern_entry_rules",
        "global.pattern_invalidation",
    ]
    
    missing = []
    for key in required_keys:
        parts = key.split(".")
        try:
            val = safe_get(MASTER, parts)
            if val is None:
                missing.append(key)
        except:
            missing.append(key)
    
    return {
        "complete": len(missing) == 0,
        "missing_keys": missing,
        "status": "OK" if len(missing) == 0 else "PARTIAL",
        "note": f"Using MASTER_CONFIG with {len(missing)} gaps (will fallback)"
    }


def log_config_status():
    """
    Log which configuration system is active.
    
    Call this at startup to confirm configuration.
    """
    status = validate_config_completeness()
    logger.info("="*60)
    logger.info("CONFIG STATUS AT STARTUP:")
    logger.info(f"  MASTER_CONFIG Available: {CONFIG_LOADED_FROM_MASTER}")
    logger.info(f"  Configuration Status: {status['status']}")
    if status.get('missing_keys'):
        logger.info(f"  Missing Keys: {status['missing_keys']}")
    logger.info(f"  Fallback System: ACTIVE (using old constants as backup)")
    logger.info("="*60)

# ============================================================================
# PENALTY & ENHANCEMENT APPLIERS
# ============================================================================

def apply_horizon_penalties(
    base_confidence: float,
    horizon: str,
    indicators: Dict[str, Any],
    setup_type: str = None,
    fundamentals: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Apply horizon-specific penalties to confidence score.
    
    Reads from MASTER_CONFIG["horizons"][horizon]["penalties"]
    
    Returns: {
        "adjusted_confidence": float,
        "penalties_applied": [list of penalty names],
        "penalty_details": {name: (amount, reason), ...},
        "log": str
    }
    """
    
    cfg = get_horizon_config(horizon)
    penalties_dict = safe_get(cfg, ["scoring", "penalties"], default={})
    
    adjusted = base_confidence
    penalties_applied = []
    penalty_details = {}
    log_lines = [f"[{horizon}] PENALTIES: base={base_confidence:.1f}"]
    
    if not penalties_dict:
        log_lines.append("  (none configured)")
        logger.debug("\n".join(log_lines))
        return {
            "adjusted_confidence": base_confidence,
            "penalties_applied": [],
            "penalty_details": {},
            "log": "\n".join(log_lines)
        }
    
    for penalty_name, penalty_cfg in penalties_dict.items():
        try:
            condition = safe_get(penalty_cfg, ["condition"])
            amount = safe_get(penalty_cfg, ["amount"], default=0)
            reason = safe_get(penalty_cfg, ["reason"], default="Unspecified")
            
            if not condition or amount == 0:
                continue
            
            # Evaluate condition
            if _evaluate_condition_enhanced(condition, indicators, setup_type, horizon, fundamentals):
                adjusted -= amount
                penalties_applied.append(penalty_name)
                penalty_details[penalty_name] = (amount, reason)
                log_lines.append(f"  -{amount:.1f} {penalty_name}: {reason}")
                logger.debug(f"[{horizon}] Applied penalty: {penalty_name} (-{amount:.1f})")
        
        except Exception as e:
            logger.warning(f"[{horizon}] Error in penalty {penalty_name}: {e}")
    
    adjusted = max(0, adjusted)  # Never go below 0
    log_lines.append(f"  → final={adjusted:.1f}")
    
    return {
        "adjusted_confidence": round(adjusted, 2),
        "penalties_applied": penalties_applied,
        "penalty_details": penalty_details,
        "log": "\n".join(log_lines)
    }


def apply_horizon_enhancements(
    base_confidence: float,
    horizon: str,
    indicators: Dict[str, Any],
    setup_type: str = None,
    fundamentals: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Apply horizon-specific enhancements to confidence score.
    
    ✅ Reads from: MASTER_CONFIG["horizons"][horizon]["enhancements"]
    
    Returns: {
        "adjusted_confidence": float,
        "enhancements_applied": [list],
        "enhancement_details": {name: (amount, reason), ...},
        "log": str
    }
    """
    
    cfg = get_horizon_config(horizon)
    enhancements_dict = safe_get(cfg, ["enhancements"], default={})
    
    adjusted = base_confidence
    enhancements_applied = []
    enhancement_details = {}
    log_lines = [f"[{horizon}] ENHANCEMENTS: base={base_confidence:.1f}"]
    
    if not enhancements_dict:
        log_lines.append("  (none configured)")
        logger.debug("\n".join(log_lines))
        return {
            "adjusted_confidence": base_confidence,
            "enhancements_applied": [],
            "enhancement_details": {},
            "log": "\n".join(log_lines)
        }
    
    for enh_name, enh_cfg in enhancements_dict.items():
        try:
            condition = safe_get(enh_cfg, ["condition"])
            amount = safe_get(enh_cfg, ["amount"], default=0)
            reason = safe_get(enh_cfg, ["reason"], default="Unspecified")
            max_boost = safe_get(enh_cfg, ["max_boost"], default=15.0)
            
            if not condition or amount == 0:
                continue
            
            # ✅ Evaluate condition using helper below
            if _evaluate_condition_enhanced(condition, indicators, setup_type, horizon, fundamentals,pattern_indicators=indicators):
                actual_boost = min(amount, max_boost)
                adjusted += actual_boost
                enhancements_applied.append(enh_name)
                enhancement_details[enh_name] = (actual_boost, reason)
                log_lines.append(f"  +{actual_boost:.1f} {enh_name}: {reason}")
                logger.debug(f"[{horizon}] Applied enhancement: {enh_name} (+{actual_boost:.1f})")
        
        except Exception as e:
            logger.warning(f"[{horizon}] Error in enhancement {enh_name}: {e}")
    
    adjusted = min(99, adjusted)
    log_lines.append(f"  → final={adjusted:.1f}")
    
    return {
        "adjusted_confidence": round(adjusted, 2),
        "enhancements_applied": enhancements_applied,
        "enhancement_details": enhancement_details,
        "log": "\n".join(log_lines)
    }


# ============================================================================
# SECTION 3: Condition Evaluator (✅ LIVES HERE)
# ============================================================================
def _evaluate_condition_enhanced(
    condition_str: str,
    indicators: Dict[str, Any],
    setup_type: str,
    horizon: str,
    fundamentals: Dict[str, Any] = None,
    pattern_indicators: Dict[str, Any] = None
) -> bool:
    """
    ✅ THIS IS THE ONLY COPY - Lives in config_helpers.py
    
    Safely evaluate enhancement/penalty conditions.
    
    Supported Conditions:
    - Technical: volatility_quality >= 7.0
    - Patterns: pattern_count >= 2
    - Setup: setup_type == 'MOMENTUM_BREAKOUT'
    - Fundamentals: roe >= 20 and roce >= 25
    - Volume: rvol >= 2.0
    - Complex: trend_strength >= 7.0 and momentum_strength >= 7.0
    """
    
    if not condition_str:
        return False
    
    try:
        # ✅ Count detected patterns
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
        
        # ✅ Build evaluation context
        eval_context = {
            # Technical Composites
            "volatility_quality": _safe_get_value(indicators, "volatility_quality"),
            "trend_strength": _safe_get_value(indicators, "trend_strength"),
            "momentum_strength": _safe_get_value(indicators, "momentum_strength"),
            
            # Technical Indicators
            "rsi": _safe_get_value(indicators, "rsi"),
            "macd_hist": _safe_get_value(indicators, "macd_hist"),
            "adx": _safe_get_value(indicators, "adx"),
            "atr_pct": _safe_get_value(indicators, "atr_pct"),
            "bb_width": _safe_get_value(indicators, "bb_width"),
            "rvol": _safe_get_value(indicators, "rvol"),
            "price_vs_primary_trend_pct": _safe_get_value(indicators, "price_vs_primary_trend_pct"),
            
            # Relative Strength
            "rel_strength_nifty": _safe_get_value(indicators, "rel_strength_nifty"),
            
            # Pattern Data
            "pattern_count": pattern_count,
            
            # Setup Context
            "setup_type": setup_type or "UNKNOWN",
            "horizon": horizon
        }
        
        # ✅ Add fundamentals if available
        if fundamentals:
            eval_context.update({
                "roe": _safe_get_value(fundamentals, "roe"),
                "roce": _safe_get_value(fundamentals, "roce"),
                "roic": _safe_get_value(fundamentals, "roic"),
                "de_ratio": _safe_get_value(fundamentals, "de_ratio"),
                "pe_ratio": _safe_get_value(fundamentals, "pe_ratio"),
                "peg_ratio": _safe_get_value(fundamentals, "peg_ratio"),
                "market_cap": _safe_get_value(fundamentals, "market_cap"),
                "quarterly_growth": _safe_get_value(fundamentals, "quarterly_growth"),
                "eps_growth_5y": _safe_get_value(fundamentals, "eps_growth_5y"),
                "revenue_growth_5y": _safe_get_value(fundamentals, "revenue_growth_5y"),
                "market_cap_cagr": _safe_get_value(fundamentals, "market_cap_cagr"),
                "institutional_ownership": _safe_get_value(fundamentals, "institutional_ownership"),
                "promoter_holding": _safe_get_value(fundamentals, "promoter_holding"),
                "ocf_vs_profit": _safe_get_value(fundamentals, "ocf_vs_profit")
            })
        
        # ✅ Safely evaluate
        result = eval(condition_str, {"__builtins__": {}}, eval_context)
        return bool(result)
    
    except Exception as e:
        logger.warning(f"[{horizon}] Error evaluating condition '{condition_str}': {e}")
        return False


def _safe_get_value(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Extract numeric value from nested dict safely.
    
    Handles:
    - Missing keys → default
    - Dict with 'value' key → extract value
    - Dict with 'raw' key → extract raw
    - Direct numeric → return as-is
    """
    if not data or key not in data:
        return default
    
    val = data[key]
    
    # Direct numeric
    if isinstance(val, (int, float)):
        return float(val)
    
    # Nested dict
    if isinstance(val, dict):
        return float(val.get("value") or val.get("raw") or default)
    
    # Try casting
    try:
        return float(val)
    except:
        return default


def validate_horizon_entry_gates(
    confidence: float,
    horizon: str,
    indicators: Dict[str, Any],
    fundamentals: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate entry gates for a specific horizon.
    
    Returns: {
        "passed": bool,
        "gates": {thresholds},
        "violations": [list of failed gates],
        "log": str
    }
    """
    
    cfg = get_horizon_config(horizon)
    gates_dict = safe_get(cfg, ["gates"], default={})
    
    passed = True
    violations = []
    log_lines = [f"[{horizon}] ENTRY GATES:"]
    
    if not gates_dict:
        log_lines.append("  (none configured)")
        return {
            "passed": True,
            "gates": {},
            "violations": [],
            "log": "\n".join(log_lines)
        }
    
    # Check each gate
    gates_info = {}
    
    # Confidence gate
    conf_min = safe_get(gates_dict, ["confidence_min"], default=0)
    gates_info["confidence_min"] = conf_min
    if conf_min > 0 and confidence < conf_min:
        violations.append(f"confidence {confidence:.1f} < {conf_min:.1f}")
        passed = False
        log_lines.append(f"  ❌ Confidence: {confidence:.1f} < {conf_min:.1f}")
    else:
        log_lines.append(f"  ✅ Confidence: {confidence:.1f} >= {conf_min:.1f}")
    
    # ADX gate
    adx_min = safe_get(gates_dict, ["adx_min"], default=0)
    gates_info["adx_min"] = adx_min
    adx_val = _safe_get_value(indicators, "adx")
    if adx_min > 0 and adx_val < adx_min:
        violations.append(f"adx {adx_val:.1f} < {adx_min:.1f}")
        passed = False
        log_lines.append(f"  ❌ ADX: {adx_val:.1f} < {adx_min:.1f}")
    else:
        log_lines.append(f"  ✅ ADX: {adx_val:.1f} >= {adx_min:.1f}")
    
    # Volatility quality gate
    vol_qual_min = safe_get(gates_dict, ["volatility_quality_min"], default=0)
    gates_info["volatility_quality_min"] = vol_qual_min
    vol_qual = _safe_get_value(indicators, "volatility_quality")
    if vol_qual_min > 0 and vol_qual < vol_qual_min:
        violations.append(f"vol_quality {vol_qual:.1f} < {vol_qual_min:.1f}")
        passed = False
        log_lines.append(f"  ❌ Volatility Quality: {vol_qual:.1f} < {vol_qual_min:.1f}")
    else:
        log_lines.append(f"  ✅ Volatility Quality: {vol_qual:.1f} >= {vol_qual_min:.1f}")
    
    # R:R ratio gate (if applicable)
    rr_min = safe_get(gates_dict, ["rr_ratio_min"], default=0)
    gates_info["rr_ratio_min"] = rr_min
    # Note: RR ratio comes from trade plan, not indicators, so we just store the gate
    
    if passed:
        log_lines.append("  ✅ ALL GATES PASSED")
    else:
        log_lines.append(f"  ❌ GATES FAILED: {', '.join(violations)}")
    
    return {
        "passed": passed,
        "gates": gates_info,
        "violations": violations,
        "log": "\n".join(log_lines)
    }

# ============================================================================
# STEP 5: Horizon Lists (Helper for iterating)
# ============================================================================
def get_ma_keys_config(horizon: str) -> Dict[str, str]:
    """Get MA keys for a specific horizon."""
    
    default = {
        "fast": "ema20", 
        "mid": "ema50", 
        "slow": "ema200"
    }
    
    try:
        cfg = get_horizon_config(horizon)
        ma_keys = safe_get(cfg, ["moving_averages", "keys"], default=default)
    except (KeyError, AttributeError):
        logger.warning(f"MA keys not found for {horizon}, using default")
        ma_keys = default
    
    # Validate structure
    if isinstance(ma_keys, list) and len(ma_keys) >= 3:
        ma_keys = {"fast": ma_keys[0], "mid": ma_keys[1], "slow": ma_keys[2]}
    elif not isinstance(ma_keys, dict):
        ma_keys = default
    
    # Ensure all keys exist
    for key in ["fast", "mid", "slow"]:
        if key not in ma_keys:
            ma_keys = default
            break
    
    return ma_keys

HORIZONS = ["intraday", "short_term", "long_term", "multibagger"]

def get_all_horizons() -> list:
    """Returns list of all supported horizons"""
    return HORIZONS.copy()


# ============================================================================
# Call at module load
# ============================================================================

log_config_status()
logger.info("🛠️ Config Helpers Module initialized successfully")

# services/config_helpers.py/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""
Business Logic Layer for Signal Engine
Bridges ConfigResolver (Data) and SignalEngine (Execution).
Uses the new consolidated 'get_signal_context' for efficiency.
"""

import logging
from typing import Dict, Any, Tuple, List
from config.config_resolver import get_config

logger = logging.getLogger(__name__)

# ==============================================================================
# HELPER: SAFE DATA EXTRACTION
# ==============================================================================
def _get_val(data: Dict, key: str, default: float = 0.0) -> float:
    """Safe float extraction from diverse indicator formats."""
    if not data or key not in data:
        return default
    val = data[key]
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        # Handle dicts like {"value": 10, "score": 8}
        v = val.get("value") or val.get("raw") or val.get("score")
        return float(v) if v is not None else default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

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
    Parses string conditions like "rvol >= 2.5" or "roe > 15".
    Checks 'indicators' first, then 'fundamentals'.
    """
    try:
        # Simple splitter: "metric op value"
        parts = condition.split(" ")
        
        # Handle "setup_type in ['A', 'B']"
        if "setup_type" in condition:
            if "in" in parts:
                return setup_type in condition 
            if "==" in parts:
                target = parts[-1].strip("'").strip('"')
                return setup_type == target
            if "!=" in parts:
                target = parts[-1].strip("'").strip('"')
                return setup_type != target
                
        # Handle numeric metric comparison: "adx < 25" or "roe > 15"
        if len(parts) >= 3:
            metric, op, target_str = parts[0], parts[1], parts[2]
            
            if metric == "setup_type": return False 
            
            # 1. Try Indicators
            val = _get_val(indicators, metric, None)
            
            # 2. Try Fundamentals (if not in indicators)
            if val is None and fundamentals:
                val = _get_val(fundamentals, metric, None)
            
            # If still missing, we cannot evaluate -> False
            if val is None:
                return False

            target = float(target_str)
            return _rule_matches(val, op, target)
            
    except Exception:
        return False
    return False

# ==============================================================================
# 1. PENALTY LOGIC
# ==============================================================================
def apply_horizon_penalties(
    base_confidence: int,
    horizon: str,
    indicators: Dict,
    setup_type: str,
    fundamentals: Dict = None
) -> Dict[str, Any]:
    """
    Applies TWO types of penalties:
    1. Simple Horizon Penalties (from scoring.penalties) -> Apply to ALL setups
    2. Complex Setup Penalties (from setup_confidence.penalties) -> Apply conditionally
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
        if val is None: continue

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
    complex_penalties = ctx.get("setup_confidence", {}).get("penalties", {})
    
    for name, rule in complex_penalties.items():
        condition = rule.get("condition", "")
        amount = rule.get("amount", 0)
        
        # Pass fundamentals here too
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
# 2. ENHANCEMENT LOGIC
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
    Now correctly uses 'fundamentals' for conditions like 'roe > 20'.
    """
    ctx = get_config(horizon).get_signal_context()
    enhancements_config = ctx.get("enhancements", {})
    
    current_conf = base_confidence
    enhancement_details = {}
    log_messages = []

    for name, rule in enhancements_config.items():
        condition = rule.get("condition", "")
        boost_amount = rule.get("amount", 0)
        
        # ✅ FIXED: Now passing fundamentals to evaluation
        if _evaluate_condition(condition, indicators, setup_type, fundamentals):
            current_conf += int(boost_amount)
            enhancement_details[name] = (boost_amount, rule.get("reason", "Enhancement"))
            log_messages.append(f"Boost: {name} -> +{boost_amount}")

    return {
        "adjusted_confidence": min(current_conf, 100), # Cap at 100
        "enhancement_details": enhancement_details,
        "log": "; ".join(log_messages) if log_messages else "No enhancements"
    }

# ==============================================================================
# 3. GATE VALIDATION
# ==============================================================================
def validate_horizon_entry_gates(
    horizon: str,
    indicators: Dict
) -> Dict[str, Any]:
    """
    Validates hard gates (ADX, Trend Strength, Volatility Bands).
    ✅ FIXED: Removed unused 'confidence' and 'fundamentals' args.
    """
    ctx = get_config(horizon).get_signal_context()
    gates_cfg = ctx.get("gates", {})
    
    failures = []
    gates_log = {}

    # 1. Trend Strength Gate
    min_trend = gates_cfg.get("min_trend_strength", 0)
    curr_trend = _get_val(indicators, "trend_strength", 0)
    
    if curr_trend < min_trend:
        status = "FAIL"
        failures.append(f"Trend {curr_trend:.1f} < {min_trend}")
    else:
        status = "PASS"
    gates_log["trend_strength"] = status

    # 2. ADX Gate
    min_adx = gates_cfg.get("adx_min", 0)
    curr_adx = _get_val(indicators, "adx", 0)
    if curr_adx < min_adx:
        status = "FAIL"
        failures.append(f"ADX {curr_adx:.1f} < {min_adx}")
    else:
        status = "PASS"
    gates_log["adx"] = status

    # 3. Volatility Bands (Safety)
    bands = gates_cfg.get("volatility_bands", {})
    atr_pct = _get_val(indicators, "atr_pct", 0)
    if bands:
        if atr_pct < bands.get("min", 0):
            status = "FAIL"
            failures.append(f"Vol too low ({atr_pct} < {bands['min']})")
        elif atr_pct > bands.get("max", 99):
            status = "FAIL"
            failures.append(f"Vol too high ({atr_pct} > {bands['max']})")
        else:
            status = "PASS"
        gates_log["volatility_band"] = status

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "gates": gates_log,
        "log": f"Gates Passed" if not failures else f"Gates Failed: {', '.join(failures)}"
    }

# ==============================================================================
# 4. SETUP CLASSIFICATION
# ==============================================================================

def classify_setup(
    indicators: Dict,
    fundamentals: Dict = None,
    horizon: str = "short_term"
) -> Tuple[str, int]:
    """
    Business logic for setup classification using config-driven rules.
    
    Evaluates all setup rules from config and returns the highest priority match.
    
    Args:
        horizon: Trading horizon (intraday, short_term, etc.)
        indicators: Technical indicators dict
        fundamentals: Optional fundamental data dict
    
    Returns:
        Tuple[setup_type, priority_score]
        Example: ("MOMENTUM_BREAKOUT", 85)
    
    Example:
        >>> setup, priority = classify_setup("short_term", indicators, fundamentals)
        >>> if setup == "MOMENTUM_BREAKOUT":
        ...     # Apply breakout-specific logic
    """
    config = get_config(horizon)
    rules = config.get_setup_rules()
    candidates = []
    
    logger.debug(f"[{horizon}] Classifying setup from {len(rules)} rule sets")
    
    for setup_name, rule_cfg in rules.items():
        # Check if this setup's conditions are met
        if config.evaluate_setup_condition(setup_name, indicators, fundamentals):
            priority = rule_cfg.get("priority", 50)
            candidates.append((setup_name, priority))
            logger.debug(f"  ✓ {setup_name} matched (priority={priority})")
    
    # Return highest priority match
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_setup, best_priority = candidates[0]
        logger.info(f"[{horizon}] Setup classified: {best_setup} (priority={best_priority})")
        return (best_setup, best_priority)
    
    logger.warning(f"[{horizon}] No setup matched, defaulting to GENERIC")
    return ("GENERIC", 0)


# ==============================================================================
# 5. POSITION SIZING
# ==============================================================================

def calculate_position_size(
    horizon: str,
    indicators: Dict,
    confidence: float,
    setup_type: str
) -> float:
    """
    Config-driven position sizing with volatility and setup adjustments.
    
    Formula:
        base_risk × confidence_factor × setup_mult × volatility_mult
        (capped at max_position_pct)
    
    Args:
        horizon: Trading horizon
        indicators: Technical indicators (needs volatility_quality)
        confidence: Signal confidence (0-100)
        setup_type: Setup classification (affects multiplier)
    
    Returns:
        Position size as fraction of capital (e.g., 0.02 = 2%)
    
    Example:
        >>> size = calculate_position_size("short_term", indicators, 75, "MOMENTUM_BREAKOUT")
        >>> capital = 100000
        >>> position_value = capital * size  # e.g., 2000 INR
    """
    config = get_config(horizon)
    
    # Get base parameters
    base_risk = config.get("global.position_sizing.base_risk_pct", 0.01)
    max_pos = config.get("risk_management.max_position_pct", 0.02)
    
    # Setup multiplier
    setup_multipliers = config.get("global.position_sizing.global_setup_multipliers", {})
    setup_mult = setup_multipliers.get(setup_type, 1.0)
    
    # Volatility adjustment
    vol_qual = _get_val(indicators, "volatility_quality", 5.0)
    vol_adjustments = config.get("global.position_sizing.volatility_adjustments", {})
    
    vol_mult = 1.0
    for regime, cfg in vol_adjustments.items():
        # Check if volatility quality falls in this regime
        if "min" in cfg and "max" in cfg:
            if cfg["min"] <= vol_qual <= cfg["max"]:
                vol_mult = cfg.get("multiplier", 1.0)
                logger.debug(f"  Vol regime: {regime} (qual={vol_qual:.1f}, mult={vol_mult})")
                break
        elif "min" in cfg and vol_qual >= cfg["min"]:
            vol_mult = cfg.get("multiplier", 1.0)
            logger.debug(f"  Vol regime: {regime} (qual={vol_qual:.1f}, mult={vol_mult})")
            break
        elif "max" in cfg and vol_qual <= cfg["max"]:
            vol_mult = cfg.get("multiplier", 1.0)
            logger.debug(f"  Vol regime: {regime} (qual={vol_qual:.1f}, mult={vol_mult})")
            break
    
    # Confidence factor (scale 0-100 to 0-1)
    conf_factor = confidence / 100.0
    
    # Calculate position size
    position = base_risk * conf_factor * setup_mult * vol_mult
    
    # Cap at max
    final_position = min(position, max_pos)
    
    logger.info(
        f"[{horizon}] Position size: {final_position:.4f} "
        f"(base={base_risk}, conf={conf_factor:.2f}, "
        f"setup={setup_mult}, vol={vol_mult})"
    )
    
    return round(final_position, 4)


# ==============================================================================
# 6. TARGET CALCULATION WITH RESISTANCE
# ==============================================================================

def calculate_targets_with_resistance(
    horizon: str,
    entry: float,
    stop_loss: float,
    indicators: Dict,
    resistance_levels: List[float] = None
) -> Tuple[float, float, Dict]:
    """
    Config-driven target calculation with resistance cushioning.
    
    Calculates T1 and T2 targets using:
    - ADX-based R:R multipliers from config
    - Resistance level adjustments with configurable cushions
    - Minimum distance requirements
    
    Args:
        horizon: Trading horizon
        entry: Entry price
        stop_loss: Stop loss price
        indicators: Technical indicators (needs adx)
        resistance_levels: Optional list of resistance prices to respect
    
    Returns:
        Tuple[target_1, target_2, metadata_dict]
        
    Example:
        >>> t1, t2, meta = calculate_targets_with_resistance(
        ...     "short_term", 100.0, 98.0, indicators, [105.0, 110.0]
        ... )
        >>> print(f"T1: {t1}, T2: {t2}, Adjustments: {meta['adjustments']}")
    """
    config = get_config(horizon)
    
    # Calculate risk
    risk = abs(entry - stop_loss)
    
    # Get ADX-based R:R multipliers
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
    
    # Get resistance adjustment parameters
    cushion = config.get("global.targets.resistance_cushion", 0.96)
    min_dist_pct = config.get("global.targets.min_distance_pct", 0.005)
    
    adjustments = []
    final_t1 = raw_t1
    final_t2 = raw_t2
    
    # Apply resistance adjustments
    if resistance_levels:
        resistance_levels = sorted(resistance_levels)
        
        # Adjust T1 if near resistance
        for res_level in resistance_levels:
            if entry < res_level <= raw_t1:
                # Check if too close to resistance
                dist_to_res = (res_level - entry) / entry
                if dist_to_res < min_dist_pct:
                    logger.warning(
                        f"  T1 too close to resistance {res_level:.2f}, "
                        f"distance={dist_to_res:.3%}"
                    )
                    continue
                
                # Apply cushion
                cushioned_t1 = res_level * cushion
                if cushioned_t1 < final_t1:
                    final_t1 = cushioned_t1
                    adjustments.append(f"T1 cushioned from {raw_t1:.2f} to {final_t1:.2f} (R={res_level:.2f})")
                    logger.info(f"  T1 adjusted for resistance: {final_t1:.2f}")
                break
        
        # Adjust T2 if near resistance
        for res_level in resistance_levels:
            if entry < res_level <= raw_t2:
                dist_to_res = (res_level - entry) / entry
                if dist_to_res < min_dist_pct:
                    continue
                
                cushioned_t2 = res_level * cushion
                if cushioned_t2 < final_t2:
                    final_t2 = cushioned_t2
                    adjustments.append(f"T2 cushioned from {raw_t2:.2f} to {final_t2:.2f} (R={res_level:.2f})")
                    logger.info(f"  T2 adjusted for resistance: {final_t2:.2f}")
                break
    
    # Ensure T1 < T2
    if final_t1 >= final_t2:
        logger.warning(f"  T1 >= T2 after adjustments, resetting T2 to raw value")
        final_t2 = raw_t2
    
    # Build metadata
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
# MODULE INITIALIZATION
# ==============================================================================

logger.info("🛠️ Config Helpers Module (Business Logic) initialized")
logger.info("✅ Integrated with config.config_resolver (V2 System)")


# ==============================================================================
# 7. MA HELPERS
# ==============================================================================
def get_ma_keys_config(horizon: str) -> Dict[str, str]:
    """
    Returns the Moving Average keys (e.g., 'ema_20', 'wma_50') 
    configured for this horizon.
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

def _evaluate_condition_enhanced(
    condition: str,
    indicators: Dict,
    setup_type: str,
    horizon: str,
    fundamentals: Dict = None,
    pattern_indicators: Dict = None
) -> bool:
    """
    Enhanced condition evaluator supporting complex expressions.
    
    Supports:
    - Simple comparisons: "rsi >= 60"
    - Setup type checks: "setup_type == 'MOMENTUM_BREAKOUT'"
    - Pattern checks: "pattern_count >= 2"
    - Fundamental checks: "roe >= 20"
    - Complex logic: "trend_strength >= 7.0 and momentum_strength >= 7.0"
    
    Args:
        condition: Condition string to evaluate
        indicators: Technical indicators
        setup_type: Current setup type
        horizon: Trading horizon
        fundamentals: Optional fundamental data
        pattern_indicators: Optional pattern-specific indicators
    
    Returns:
        True if condition met, False otherwise
    
    Examples:
        >>> _evaluate_condition_enhanced(
        ...     "setup_type == 'MOMENTUM_BREAKOUT' and rvol >= 3.0",
        ...     indicators, "MOMENTUM_BREAKOUT", "short_term"
        ... )
        True
    """
    try:
        # Handle setup_type checks
        if "setup_type" in condition:
            # Replace setup_type with actual value
            condition = condition.replace("setup_type", f"'{setup_type}'")
        
        # Handle pattern_count (from pattern_indicators)
        if "pattern_count" in condition and pattern_indicators:
            pattern_keys = [
                "darvas_box", "cup_handle", "minervini_stage2",
                "flag_pennant", "bollinger_squeeze", "three_line_strike",
                "golden_cross", "double_top_bottom"
            ]
            
            active_patterns = sum(
                1 for pk in pattern_keys
                if (pattern_indicators.get(pk) or {}).get("found")
                and (pattern_indicators.get(pk) or {}).get("score", 0) > 70
            )
            
            condition = condition.replace("pattern_count", str(active_patterns))
        
        # Build evaluation context
        eval_context = {}
        
        # Add all indicator values
        for key, val in indicators.items():
            if isinstance(val, dict):
                eval_context[key] = val.get("value") or val.get("raw") or val.get("score") or 0
            elif isinstance(val, (int, float)):
                eval_context[key] = val
        
        # Add fundamental values
        if fundamentals:
            for key, val in fundamentals.items():
                if isinstance(val, dict):
                    eval_context[key] = val.get("value") or val.get("raw") or 0
                elif isinstance(val, (int, float)):
                    eval_context[key] = val
        
        # Evaluate condition
        # Note: Using eval() with controlled context
        # Only safe because conditions come from config, not user input
        result = eval(condition, {"__builtins__": {}}, eval_context)
        
        return bool(result)
        
    except Exception as e:
        logger.debug(f"Condition evaluation failed '{condition}': {e}")
        return False