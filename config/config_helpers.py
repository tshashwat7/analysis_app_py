# config/config_helpers.py
"""
Business Logic Bridge v3.0 - Refactored for QueryOptimizedExtractor
====================================================================
ARCHITECTURE UPDATE:
✅ Now uses refactored_resolver_Copy.py (v6.0) with QueryOptimizedExtractor
✅ Maintains same public API (no breaking changes)
✅ All config access goes through extractor internally

DESIGN PATTERN (Unchanged):
1. Build evaluation context ONCE (expensive)
2. Access context fields MANY times (cheap)
3. Build execution context ONCE when needed
4. No duplicate calculations

Version: 3.0 - Extractor Integration
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, time

# ✅ NEW: Import refactored resolver
from config.config_resolver import ConfigResolver, create_resolver

from services.data_fetch import _get_val
from config.config_helpers.logger_config import (
    METRICS,
    validate_required_keys,
    log_failures
)
from services.patterns.pattern_velocity_tracking import get_pattern_velocity_stats

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CENTRALIZED RESOLVER FACTORY (Updated for v6.0)
# ==============================================================================
_resolver_cache: Dict[str, ConfigResolver] = {}

def get_resolver(horizon: str, use_cache: bool = True) -> ConfigResolver:
    """
    ✅ UPDATED: Now returns refactored ConfigResolver v6.0
    
    Cached resolver factory (30x speedup maintained).
    
    Args:
        horizon: Trading timeframe
        use_cache: Whether to use cached instance
    
    Returns:
        ConfigResolver v6.0 instance with QueryOptimizedExtractor
    """
    from config.master_config import MASTER_CONFIG
    
    if use_cache and horizon in _resolver_cache:
        logger.debug(f"[{horizon}] ♻️ Resolver cache HIT")
        return _resolver_cache[horizon]
    
    logger.debug(f"[{horizon}] 🏭 Resolver cache MISS (Instantiating new)")
    # ✅ NEW: Uses refactored resolver
    resolver = create_resolver(MASTER_CONFIG, horizon)
    
    if use_cache:
        _resolver_cache[horizon] = resolver
    
    return resolver


def clear_resolver_cache():
    """Clear cached resolvers"""
    global _resolver_cache
    _resolver_cache.clear()
    logger.info("✅ Resolver cache cleared")


# ==============================================================================
# 2. EXTRACTION HELPERS (Unchanged - Pure helpers)
# ==============================================================================

@log_failures(return_on_error={}, critical=False)
def _extract_patterns(indicators: Dict, horizon: str) -> Dict:
    """
    Extract patterns with passive error isolation.
    
    LOGGING BEHAVIOR:
    - Logs pattern extraction failures to METRICS
    - Continues processing other patterns on failure
    - No console spam (DEBUG level only)
    """
    try:
        from config.setup_pattern_matrix_config import PATTERN_INDICATOR_MAPPINGS
        
        detected = {}
        patterns_failed = 0
        
        for pattern_name, horizon_map in PATTERN_INDICATOR_MAPPINGS.items():
            try:
                if not pattern_name:
                    continue
                
                p_obj = indicators.get(pattern_name, {})
                if not isinstance(p_obj, dict):
                    patterns_failed += 1
                    METRICS.log_missing_key("pattern_extraction", pattern_name, pattern_name)
                    continue
                
                raw_data = p_obj.get("raw", p_obj)
                if raw_data.get("found", False):
                    if "confidence" not in raw_data:
                        raw_data["confidence"] = 50
                    detected[pattern_name] = p_obj
            
            except Exception as e:
                patterns_failed += 1
                METRICS.log_failed_method(f"extract_pattern_{pattern_name}", e)
                continue
        
        if detected:
            logger.debug(f"[{horizon}] Detected {len(detected)} patterns")
        
        return detected
        
    except ImportError as e:
        logger.warning(f"Pattern matrix unavailable: {e}")
        return {}

def _extract_price_data(indicators: Dict, fundamentals: Dict) -> Dict:
    """Extract ALL price action data"""
    return {
        "price":  _get_val(indicators, "price", 0),
        "close": _get_val(indicators, "prev_close", 0),
        "volume": _get_val(indicators, "volume", 0),
        "position52w": _get_val(fundamentals, "position52w", 50),
        "high52w": _get_val(fundamentals, "high52w", 0),
        "bbHigh": _get_val(indicators, "bbHigh", 0),
        "bbMid": _get_val(indicators, "bbMid", 0),
        "bbLow": _get_val(indicators, "bbLow", 0),
        "avgVolume": _get_val(indicators, "avg_volume_30Days", 0),
        "rvol": _get_val(indicators, "rvol", 1.0),
        "resistance1": _get_val(indicators, "resistance1", 0),
        "resistance2": _get_val(indicators, "resistance2", 0),
        "support1": _get_val(indicators, "support1", 0),
        "support2": _get_val(indicators, "support2", 0),
        "atrDynamic": _get_val(indicators, "atrDynamic", 0)
    }

def flatten_market_data_mixed(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested dicts while PRESERVING string values for semantic comparison.
    
    Priority:
    1. Numeric raw/value → extract as float
    2. String raw/value → KEEP as string (for condition evaluation)
    3. Fallback to score only if raw/value missing
    
    Returns mixed dict: {str: float | str}
    """
    flattened = {}
    
    for key, val in data.items():
        if val is None:
            continue
            
        # Already primitive - use directly
        if isinstance(val, (int, float, str, bool)):
            flattened[key] = val
            continue
        
        # Nested dict - extract with type preservation
        if isinstance(val, dict):
            # Try 'raw' first
            raw = val.get('raw')
            if raw is not None and not _is_nested_dict(raw):
                if isinstance(raw, str):
                    # Keep string values for semantic comparison
                    flattened[key] = raw
                    continue
                else:
                    try:
                        flattened[key] = float(raw)
                        continue
                    except (ValueError, TypeError):
                        pass
            
            # Try 'value' second
            value = val.get('value')
            if value is not None and not _is_nested_dict(value):
                if isinstance(value, str):
                    # KEEP string values (don't fall back to score)
                    flattened[key] = value
                    continue
                else:
                    try:
                        flattened[key] = float(value)
                        continue
                    except (ValueError, TypeError):
                        pass
            
            # Fallback to 'score' (only for metrics without raw/value)
            score = val.get('score')
            if score is not None:
                try:
                    flattened[key] = float(score)
                except (ValueError, TypeError):
                    pass
    
    return flattened

def _is_nested_dict(value: Any) -> bool:
    """Check if value is a dict with nested structure (breakdown, meta, etc.)"""
    return isinstance(value, dict) and any(
        k in value for k in ['breakdown', 'meta', 'raw', 'value', 'score']
    )


# ==============================================================================
# 3. CONTEXT BUILDERS (✅ Updated to use v6.0 resolver)
# ==============================================================================
@log_failures(return_on_error={}, critical=False)
def build_evaluation_context_v5(
    ticker: str,
    indicators: Dict,
    fundamentals: Dict,
    horizon: str,
    patterns: Dict = None
) -> Dict[str, Any]:
    """
    ✅ UPDATED: Build evaluation context using refactored resolver v6.0
    
    This is the ONLY place where context is built.
    All other functions just ACCESS this context.
    
    LOGGING BEHAVIOR:
    - Sets METRICS context for this stock
    - Validates minimum viable data
    - Logs ERRORS only for critical missing data
    - Returns partial context on error (not crash)
    
    Returns:
        Complete evaluation context with:
        - setup classification
        - confidence calculation
        - gate validation
        - strategy recommendations
        - All resolver calculations
    
    Usage in signal_engine.py:
        # Build ONCE
        eval_ctx = build_evaluation_context_v5(
            ticker, indicators, fundamentals, horizon
        )
        
        # Then access multiple times (no recalculation)
        setup = get_setup_from_context(eval_ctx)
        conf = get_confidence_from_context(eval_ctx)
        gates = check_gates_from_context(eval_ctx, conf)
    """
    try:
        # Set metrics context for all subsequent logs
        METRICS.set_current_symbol(ticker)
        METRICS.reset()
        
        # Validate minimum viable data
        if fundamentals is None:
            logger.error(f"[{ticker}] ❌ Fundamentals is None (data fetch failure)")
            METRICS.log_missing_key("build_context", "fundamentals", "data_fetch_failure")
        
        if indicators is None or len(indicators) < 5:
            logger.error(
                f"[{ticker}] ❌ Insufficient indicators: "
                f"{len(indicators) if indicators else 0} keys (need at least 5)"
            )
            return {
                "error": "Insufficient indicators",
                "setup": {"type": "GENERIC", "priority": 0},
                "confidence": {"clamped": 50},
                "_meta": {"ticker": ticker, "horizon": horizon, "data_quality": "POOR"}
            }
        
        # ✅ UPDATED: Pass through specific pre-computed patterns if given, else extract from indicators (fallback)
        resolver = get_resolver(horizon)
        detected_patterns = patterns if patterns else _extract_patterns(indicators, horizon)
        
        price_data = _extract_price_data(indicators, fundamentals)
        clean_indicators = flatten_market_data_mixed(indicators or {})
        clean_fundamentals = flatten_market_data_mixed(fundamentals or {})
        clean_pricedata = flatten_market_data_mixed(price_data)
        # ✅ UPDATED: Build context using v6.0 resolver
        # (Internally uses extractor for all config access)
        eval_ctx = resolver.build_evaluation_context_only(
            symbol=ticker,
            fundamentals=clean_fundamentals or {},
            indicators=clean_indicators,
            price_data=clean_pricedata,
            detected_patterns=detected_patterns
        )
        logger.info(f"[{ticker}][{horizon}] ✅ Context has {len(eval_ctx)} keys: {list(eval_ctx.keys())}")

        if "error" in eval_ctx:
            logger.error(f"[{ticker}][{horizon}] ⚠️ FALLBACK CONTEXT! Error: {eval_ctx['error']}")

        
        # Log summary at end
        summary = METRICS.get_summary()
        if summary["total_issues"] > 0:
            # Build details from issue categories
            details = []
            if summary.get("missing_keys"):
                details.append(f"missing_keys={list(summary['missing_keys'].keys())}")
            if summary.get("failed_methods"):
                details.append(f"failed_methods={list(summary['failed_methods'].keys())}")
            if summary.get("none_returns"):
                details.append(f"none_returns={list(summary['none_returns'].keys())}")
            if summary.get("validation_failures"):
                details.append(f"validation_failures={list(summary['validation_failures'].keys())}")
            gates = summary.get("gates", {})
            failed_gates = [g for g, v in gates.items() if v.get("failed", 0) > 0]
            if failed_gates:
                details.append(f"failed_gates={failed_gates}")
            detail_str = " | ".join(details) if details else "unknown"
            logger.warning(
                f"[{ticker}] ⚠️ EVALUATION SUMMARY: "
                f"{summary['total_issues']} issues detected — {detail_str}"
            )
        else:
            logger.info(f"[{ticker}] ✅ EVALUATION CLEAN: No issues")
        
        # Add metadata for helpers
        if "_meta" not in eval_ctx:
            eval_ctx["_meta"] = {}
        
        eval_ctx["_meta"].update({
            "ticker": ticker,
            "horizon": horizon,
            "timestamp": datetime.now().isoformat(),
            "patterns_detected": list(detected_patterns.keys()),
            "data_completeness": {
                "fundamentals": len(fundamentals) if fundamentals else 0,
                "indicators": len(indicators) if indicators else 0,
                "patterns": len(detected_patterns)
            },
            "resolver_version": "6.0",
            "extractor_available": True
        })
        
        return eval_ctx
        
    except Exception as e:
        logger.error(f"[{ticker}] ❌ Context build failed: {e}", exc_info=True)
        METRICS.log_failed_method("build_evaluation_context_v5", e, ticker)
        
        return {
            "error": str(e),
            "setup": {"type": "GENERIC", "priority": 0},
            "confidence": {"clamped": 50},
            "_meta": {
                "ticker": ticker,
                "horizon": horizon,
                "data_quality": "ERROR"
            }
        }


@log_failures(return_on_error={}, critical=False)
def build_execution_context_v5(
    eval_ctx: Dict,
    capital: float
) -> Dict[str, Any]:
    """
    ✅ UPDATED: Build execution context using refactored resolver v6.0
    
    Takes existing evaluation context and adds execution details.
    
    Args:
        eval_ctx: Output from build_evaluation_context_v5()
        capital: Available capital
    
    Returns:
        Complete execution context with:
        - position sizing
        - risk model
        - order model
        - entry permission
    
    LOGGING BEHAVIOR:
    - Validates eval_ctx has required fields
    - Logs ERROR only for critical missing data
    - Returns fallback context on error
    """
    try:
        ticker = eval_ctx.get("_meta", {}).get("ticker", "UNKNOWN")
        horizon = eval_ctx.get("_meta", {}).get("horizon", "short_term")
        
        # Validate eval_ctx has minimum required fields
        required_sections = ["setup", "confidence", "strategy"]
        if not validate_required_keys(eval_ctx, required_sections, f"{ticker}_eval_ctx"):
            logger.error(
                f"[{ticker}] ❌ Evaluation context missing required sections: "
                f"{[s for s in required_sections if s not in eval_ctx]}"
            )
            return {
                "error": "Invalid evaluation context",
                "position_size": 0.01,
                "can_trade": False
            }
        
        # ✅ NEW: Get refactored resolver with extractor
        resolver = get_resolver(horizon)
        
        # ✅ UPDATED: Build execution context using v6.0 resolver
        full_ctx = resolver.build_execution_context_from_evaluation(
            evaluation_ctx=eval_ctx,
            capital=capital
        )
        
        return full_ctx
        
    except Exception as e:
        ticker = eval_ctx.get("_meta", {}).get("ticker", "UNKNOWN")
        logger.error(f"[{ticker}] ❌ Execution context build failed: {e}", exc_info=True)
        METRICS.log_failed_method("build_execution_context_v5", e, ticker)
        
        return {
            "error": str(e),
            "position_size": 0.01,
            "can_trade": False
        }


# ==============================================================================
# 4. CONTEXT ACCESSORS (Unchanged - Lightweight data access)
# ==============================================================================

def get_setup_from_context(eval_ctx: Dict) -> Tuple[str, int, Dict]:
    """
    ✅ Extract setup from EXISTING context
    NO recalculation - just data access
    
    Returns: (setup_type, priority, metadata)
    """
    setup_info = eval_ctx.get("setup", {})
    best = setup_info.get("best", setup_info)
    
    metadata = {
        "reasoning": setup_info.get("reasoning", ""),
        "priority": best.get("priority", setup_info.get("priority", 0)),
        "confidence_floor": setup_info.get("confidence_floor", 50),
        "require_fundamentals": setup_info.get("require_fundamentals", False),
        "patterns_primary": setup_info.get("patterns_primary", []),
        "patterns_detected": eval_ctx.get("_meta", {}).get("patterns_detected", []),
        "blocked": False,
        "horizon": eval_ctx.get("_meta", {}).get("horizon")
    }
    # Check if blocked
    # (Removed: _apply_setup_preferences no longer emits 'blocked')
    
    return (
        setup_info.get("type", "GENERIC"),
        best.get("priority", setup_info.get("priority", 0)),
        metadata
    )


def get_confidence_from_context(eval_ctx: Dict) -> Tuple[int, Dict]:
    """
    ✅ Extract confidence from existing context
    NO recalculation - just data access
    
    Returns:
        (final_confidence, metadata)
    """
    confidence_info = eval_ctx.get("confidence", {})
    
    adjustments = confidence_info.get("adjustments", {})
    
    # Build breakdown
    breakdown_parts = [f"Base: {confidence_info.get('base', 50)}%"]
    
    # Handle both old and new adjustment formats
    if isinstance(adjustments, dict):
        if "breakdown" in adjustments:
            # New format (list of strings)
            breakdown_parts.extend(adjustments["breakdown"])
        else:
            # Old format (dict of name: value)
            for name, value in adjustments.items():
                sign = "+" if value >= 0 else ""
                breakdown_parts.append(f"{name.replace('_', ' ').title()}: {sign}{value}%")
    
    breakdown_parts.append(f"Final: {confidence_info.get('clamped', 50)}%")
    
    metadata = {
        "base": confidence_info.get("base", 50),
        "adjustments": adjustments,
        "final": confidence_info.get("final", 50),
        "clamped": confidence_info.get("clamped", 50),
        "breakdown": " → ".join(breakdown_parts)
    }
    
    return (metadata["clamped"], metadata)


def check_gates_from_context(eval_ctx: Dict, confidence: float) -> Dict[str, Any]:
    """
    ✅ CORRECT: Check gates using EXISTING context
    NO recalculation - just validation
    
    Returns:
        {
            "passed": bool,
            "failed_gates": [...],
            "gate_details": {...}
        }
    """
    result = {
        "passed": True,
        "failed_gates": [],
        "gate_details": {}
    }
    
    # Gate 1: Structural Gates
    structural_gates = eval_ctx.get("structural_gates", {})
    result["gate_details"]["structural_gates"] = structural_gates
    
    if not structural_gates.get("overall", {}).get("passed", False):
        result["passed"] = False
        failures = structural_gates.get("overall", {}).get("failed_gates", [])
        for failure in failures:
            result["failed_gates"].append(f"structural: {failure}")
    
    # Gate 2: Execution Rules
    execution_rules = eval_ctx.get("execution_rules", {})
    if not execution_rules.get("overall", {}).get("passed", False):
        result["passed"] = False
        failures = execution_rules.get("overall", {}).get("failed_rules", [])
        for failure in failures:
            result["failed_gates"].append(f"execution: {failure}")
    
    # Gate 3: Opportunity Gates
    opportunity_gates = eval_ctx.get("opportunity_gates", {})
    if not opportunity_gates.get("overall", {}).get("passed", False):
        result["passed"] = False
        failures = opportunity_gates.get("overall", {}).get("failed_gates", [])
        for failure in failures:
            result["failed_gates"].append(f"opportunity: {failure}")
    # Gate 4: Setup Preferences
    # (Removed: blocking by setup preference is no longer intended)
    
    # Gate 5: Confidence Floor
    conf_info = eval_ctx.get("confidence", {})
    dynamic_floor = conf_info.get("floor", conf_info.get("min_tradeable_threshold", 50))
    
    if confidence < dynamic_floor:
        result["passed"] = False
        result["failed_gates"].append(
            f"confidence: {confidence:.1f}% < floor {dynamic_floor:.1f}%"
        )
    
    # Build summary
    if result["passed"]:
        result["summary"] = "All gates passed"
    else:
        result["summary"] = f"Failed: {', '.join(result['failed_gates'][:3])}"
    
    return result


def get_strategy_from_context(eval_ctx: Dict) -> Dict[str, Any]:
    """
    ✅ CORRECT: Extract strategy from EXISTING context
    NO recalculation - just data access
    """
    strategy_info = eval_ctx.get("strategy", {})
    
    return {
        "primary_strategy": strategy_info.get("primary", "generic"),
        "fit_score": strategy_info.get("fit_score", 0),
        "description": strategy_info.get("description", ""),
        "horizon_multiplier": strategy_info.get("horizon_multiplier", 1.0)
    }


# ==============================================================================
# 7. ✅ NEW: Strategy Analysis Helper (Using Extractor)
# ==============================================================================
  

