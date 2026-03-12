# services/tradeplan/trade_enhancer.py (REFACTORED v5.0)
"""
Pattern-Based Trade Enhancement Engine - Unified Utilities Edition
===================================================================
✅ REFACTORED v5.0: Now uses shared utilities from config_helpers.py

CHANGES FROM v4.0:
- Removed duplicate pattern metadata extraction
- Removed duplicate timeline calculation
- Now uses extract_pattern_execution_metadata()
- Now uses calculate_pattern_timeline()
- Cleaner, DRY code

Post-processing layer for real-time pattern monitoring.
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Static Analysis (Config Resolver)                │
│  - Loads setup rules from SETUP_PATTERN_MATRIX              │
│  - Detects patterns (Darvas Box, Cup & Handle, etc.)       │
│  - Returns: "Pattern found 20 candles ago"                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Real-Time Validation (Trade Enhancer) ✅          │
│  - Checks if patterns are STILL VALID                       │
│  - Monitors for pattern breakdown                           │
│  - Extracts RR regime metadata                              │
│  - Returns: "Pattern expired" OR "Pattern invalidated"      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Target Calculation (generate_trade_plan)          │
│  - Uses RR multipliers from enhancer                        │
│  - Calculates T1/T2 targets                                 │
│  - Returns: Final trade plan                                │
└─────────────────────────────────────────────────────────────┘
"""

import logging
from typing import Dict, Any, Optional, Tuple


from config.config_utility.logger_config import log_failures
from config.config_utility.market_utils import get_current_utc
from services.data_fetch import _get_val, safe_float
from services.patterns.pattern_state_manager import (
    get_breakdown_state,
    save_breakdown_state,
    update_breakdown_state,
    delete_breakdown_state
)

# ✅ NEW v5.0: Import shared utilities
from config.config_helpers import get_resolver
from services.patterns.pattern_velocity_tracking import get_pattern_velocity_stats
from services.patterns.utils import _classify_volatility
logger = logging.getLogger(__name__)
# ============================================================
# EXECUTION POLICY CONSTANTS (Config-Driven)
# ============================================================

MIN_EXECUTION_RR_GATE = {
    'intraday': 1.5,
    'short_term': 1.5,
    'long_term': 2.0,
    'multibagger': 2.5
}

VOLATILITY_BUFFER_FACTORS = {
    'low': 0.0,
    'normal': 0.25,
    'high': 0.5,
    'extreme': 1.0
}

MIN_SL_ATR_MULTIPLES = {
    'intraday': 2.0,
    'short_term': 2.0,
    'long_term': 2.5,
    'multibagger': 3.0
}

TARGET_ADJUSTMENT_FACTORS = {
    'low': 0.85,
    'normal': 1.0,
    'high': 1.15,
    'extreme': 1.3
}

BASE_SPREAD_PCT = {
    'intraday': 0.0015,
    'short_term': 0.001,
    'long_term': 0.0008,
    'multibagger': 0.0005
}

def adjust_targets_for_market_conditions(
    risk_data: Dict[str, Any],
    indicators: Dict[str, Any],
    fundamentals: Dict[str, Any],
    horizon: str
) -> Dict[str, Any]:
    """
    STATELESS market adjustment of resolver's pattern-based targets.
    
    ✅ Direction-agnostic (auto-detects LONG/SHORT from SL position)
    ✅ Horizon-agnostic (works for any timeframe)
    ✅ No DB writes (pure function)
    ✅ Preserves pattern entry (not overridden)
    
    INPUT (from resolver's exec_ctx["risk"]):
    - entry_price: Pattern entry trigger
    - stop_loss: Pattern invalidation level  
    - targets: [t1, t2] from pattern geometry
    - rrRatio: Pattern-structural RR
    
    OUTPUT (market-adjusted):
    - execution_sl: Volatility-adjusted SL
    - execution_t1/t2: Market-adjusted targets
    - execution_rr_t1/t2: Actual RR values
    """
    try:
        # === Extract Base Values ===
        pattern_entry = safe_float(risk_data.get("entry_price"), 0)
        pattern_sl = safe_float(risk_data.get("stop_loss"), 0)
        pattern_targets = risk_data.get("targets", [])
        structural_rr = safe_float(risk_data.get("rrRatio"), 0)
        
        if not all([pattern_entry, pattern_sl, pattern_targets]):
            return {"adjusted": False, "reason": "Missing base values"}
        
        pattern_t1 = safe_float(pattern_targets[0]) if len(pattern_targets) > 0 else None
        pattern_t2 = safe_float(pattern_targets[1]) if len(pattern_targets) > 1 else None
        
        if not pattern_t1:
            return {"adjusted": False, "reason": "No T1 target"}
        
        # === Auto-Detect Direction ===
        if pattern_sl < pattern_entry:
            direction = "LONG"
        elif pattern_sl > pattern_entry:
            direction = "SHORT"
        else:
            return {"adjusted": False, "reason": "Invalid SL (equals entry)"}
        
        # === Market Conditions ===
        # ✅ FIX: indicators store values as {"value": x, ...} dicts.
        # indicators.get("price") returns the dict, not the float.
        # safe_float on a dict yields None, so current_price silently fell back
        # to pattern_entry — making all volatility adjustments anchored to the
        # wrong price.  Use _get_val() which handles both dict and scalar.
        current_price = _get_val(indicators, "price") or safe_float(indicators.get("price"), pattern_entry)
        if not current_price:
            current_price = pattern_entry
        current_atr = _get_val(indicators, "atrDynamic") or safe_float(indicators.get("atrDynamic"), 0)
        if not current_atr:
            current_atr = 0
        atr_pct = _get_val(indicators, "atrPct") or safe_float(indicators.get("atrPct"), 2.0)
        if not atr_pct:
            atr_pct = 2.0
        volatility_regime = _classify_volatility(atr_pct)
        
        # === Volatility-Adjusted SL ===
        vol_buffer = VOLATILITY_BUFFER_FACTORS.get(volatility_regime, 0.25)
        min_sl_mult = MIN_SL_ATR_MULTIPLES.get(horizon, 2.0)
        min_sl_distance = current_atr * (min_sl_mult + vol_buffer)
        
        if direction == "LONG":
            volatility_sl = current_price - min_sl_distance
            execution_sl = min(pattern_sl, volatility_sl)  # Wider stop
            sl_buffer = pattern_sl - execution_sl
        else:  # SHORT
            volatility_sl = current_price + min_sl_distance
            execution_sl = max(pattern_sl, volatility_sl)  # Wider stop
            sl_buffer = execution_sl - pattern_sl
        
        # === Market-Adjusted Targets (Preserve Geometry) ===
        target_factor = TARGET_ADJUSTMENT_FACTORS.get(volatility_regime, 1.0)
        
        t1_distance = abs(pattern_t1 - pattern_entry)
        if direction == "LONG":
            execution_t1 = pattern_entry + (t1_distance * target_factor)
        else:
            execution_t1 = pattern_entry - (t1_distance * target_factor)
        
        execution_t2 = None
        if pattern_t2:
            t2_distance = abs(pattern_t2 - pattern_entry)
            if direction == "LONG":
                execution_t2 = pattern_entry + (t2_distance * target_factor)
            else:
                execution_t2 = pattern_entry - (t2_distance * target_factor)
        
        # === Spread Cost ===
        base_spread = BASE_SPREAD_PCT.get(horizon, 0.001)
        spread_multiplier = 2.0 if atr_pct > 5.0 else (1.5 if atr_pct > 3.0 else 1.0)
        market_cap_data = fundamentals.get("marketCap", {})
        mc_raw = market_cap_data.get("raw", 0) if isinstance(market_cap_data, dict) else (market_cap_data or 0)
        market_cap_crores = mc_raw / 10000000
        
        spread_cost = calculate_adaptive_spread_cost(
            current_price=current_price,
            rvol=_get_val(indicators, "rvol", 1.0),
            market_cap=market_cap_crores,
            horizon=horizon
        )
        # current_price * base_spread * spread_multiplier
        
        # === Execution RR ===
        if direction == "LONG":
            effective_risk = current_price - execution_sl + spread_cost
            effective_reward_t1 = execution_t1 - current_price - spread_cost
            effective_reward_t2 = (execution_t2 - current_price - spread_cost) if execution_t2 else None
        else:  # SHORT
            effective_risk = execution_sl - current_price + spread_cost
            effective_reward_t1 = current_price - execution_t1 - spread_cost
            effective_reward_t2 = (current_price - execution_t2 - spread_cost) if execution_t2 else None
        
        rr_t1 = effective_reward_t1 / effective_risk if (effective_risk > 0 and effective_reward_t1 > 0) else 0
        rr_t2 = effective_reward_t2 / effective_risk if (effective_risk > 0 and effective_reward_t2 and effective_reward_t2 > 0) else None
        
        # ✅ GUARD: Re-verify RR after spread (Bug 3)
        if effective_reward_t1 <= 0:
            return {"adjusted": False, "reason": "T1 target at or behind current price after spread"}
        return {
            "adjusted": True,
            "direction": direction,
            
            # Source (preserved)
            "pattern_entry": pattern_entry,
            "pattern_sl": pattern_sl,
            "pattern_t1": pattern_t1,
            "pattern_t2": pattern_t2,
            "structural_rr": structural_rr,
            
            # Execution (adjusted)
            "execution_entry": pattern_entry,  # NOT overridden
            "expected_fill_price": current_price,
            "execution_sl": execution_sl,
            "execution_t1": execution_t1,
            "execution_t2": execution_t2,
            "execution_rr_t1": round(rr_t1, 2),
            "execution_rr_t2": round(rr_t2, 2) if rr_t2 else None,
            
            # Metadata
            "volatility_regime": volatility_regime,
            "sl_buffer_added": round(sl_buffer, 2),
            "target_factor": target_factor,
            "spread_cost": round(spread_cost, 2),
            "effective_risk": round(effective_risk, 2)
        }
        
    except Exception as e:
        logger.error(f"Market adjustment failed: {e}", exc_info=True)
        return {"adjusted": False, "error": str(e)}

# ============================================================
# 1. RR REGIME ADJUSTMENTS (Returns Multipliers)
# ============================================================

def get_rr_regime_multipliers(
    eval_ctx: Dict[str, Any],  
    horizon: str = "short_term",
    extractor: Optional[Any] = None
) -> Tuple[float, float, str]:
    """
    ✅ REFACTORED v5.1: Reuse resolver's trend regime.
    """
    try:
        if extractor is None:
            resolver = get_resolver(horizon)
            extractor = resolver.extractor
        
        # ✅ Reuse resolver's trend calculation
        trend = eval_ctx.get("trend", {})
        regime_raw = trend.get("regime", "normal")  # "strong", "normal", "weak"
        
        # Map to RR config keys
        regime_map = {
            "strong": "strong_trend",
            "normal": "normal_trend",
            "weak": "weak_trend"
        }
        regime = regime_map.get(regime_raw, "normal_trend")
        
        # Get multipliers
        rr_cfg = _get_rr_regime_config(extractor)
        regime_cfg = rr_cfg.get(regime, {})
        
        t1_mult = regime_cfg.get("t1_mult", 1.5)
        t2_mult = regime_cfg.get("t2_mult", 3.0)

        adx_val = trend.get('adx')
        adx_str = f"{adx_val:.1f}" if adx_val is not None else "N/A"

        logger.info(
            f"RR regime '{regime}': ADX={adx_str}, "
            f"T1 mult={t1_mult}, T2 mult={t2_mult}"
        )
        
        return (t1_mult, t2_mult, regime)
    
    except Exception as e:
        logger.error(f"get_rr_regime_multipliers failed: {e}")
        return (1.5, 3.0, "normal_trend")


# ============================================================
# 2. PATTERN EXPIRATION CHECK
# ============================================================

def check_pattern_expiration(
    detected_patterns: Dict[str, Dict],
    horizon: str = "short_term",
    extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    ✅ POST-ENTRY: Pattern Lifecycle Monitoring (ENHANCER)
        
        ROLE: Watchdog (Active Trade)
        - Monitors expiration (max_duration_candles).
        - Checks invalidation conditions relative to entry price.
        
        ❌ DOES NOT: Validate setup eligibility (Resolver job).
    """
    result = {
        "expired": False,
        "pattern": None,
        "reason": None,
        "age_candles": None,
        "warnings": []
    }
    
    if extractor is None:
        resolver = get_resolver(horizon)
        extractor = resolver.extractor
    
    HORIZON_SECONDS = {
        "intraday": 900,
        "short_term": 86400,
        "long_term": 604800,
        "multibagger": 2592000
    }
    candle_sec = HORIZON_SECONDS.get(horizon, 86400)
    
    try:
        for pattern_name, pattern_data in detected_patterns.items():
            if not pattern_data.get("found"):
                continue
            
            # ✅ Get pattern context
            pattern_ctx = extractor.get_pattern_context(pattern_name)
            if not pattern_ctx:
                continue
            
            # ✅ USE DIRECTLY - no extraction!
            inval_cfg = pattern_ctx.invalidation  # may be {} — that's fine for expiry
            
            # ✅ FIX: Use typical_duration.max (Bug 2)
            # typical_duration.max in candle units IS the horizon-aware threshold. 
            # 20 candles = 20 of current horizon units (15min/1d/1wk/etc).
            typical = pattern_ctx.typical_duration or {}
            max_candles = typical.get("max")
            
            if not max_candles:
                continue
            
            # ✅ FIXED: Extract metadata from correct nesting
            raw = pattern_data.get("raw", {})
            meta = raw.get("meta", {})
            
            cached_age = meta.get("age_candles", 0)
            formation_ts = meta.get("formation_time")
            
            # ✅ Real-time age calculation (FIXED timezone)
            if formation_ts:
                seconds_since = get_current_utc().timestamp() - formation_ts
                real_time_age = int(seconds_since / candle_sec)
                current_age = max(cached_age, real_time_age)
                
                logger.debug(
                    f"{pattern_name}: cached={cached_age}, real_time={real_time_age}, using={current_age}"
                )
            else:
                current_age = cached_age
            
            # Warning threshold
            warning_threshold = int(max_candles * 0.8)
            if current_age > warning_threshold:
                result["warnings"].append({
                    "pattern": pattern_name,
                    "age": current_age,
                    "max_age": max_candles,
                    "remaining": max_candles - current_age,
                    "severity": "critical" if current_age > max_candles else "warning"
                })
            
            if current_age > max_candles:
                result.update({
                    "expired": True,
                    "pattern": pattern_name,
                    "reason": f"{pattern_name} expired ({current_age} > {max_candles} candles)",
                    "age_candles": current_age
                })
                logger.warning(f"⏰ Pattern expired: {result['reason']}")
                break
    
    except Exception as e:
        logger.error(f"check_pattern_expiration failed: {e}", exc_info=True)
    
    return result


# ============================================================
# 3. PATTERN INVALIDATION MONITORING
# ============================================================

def check_pattern_invalidation(
    detected_patterns: Dict[str, Dict],
    indicators: Dict[str, Any],
    symbol: str,
    position_type: str = "LONG",
    horizon: str = "short_term",
    extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    ✅ FINAL FIX: Trust extractor completely - use config AS-IS.
    
    Extractor returns FLAT, READY-TO-USE invalidation config.
    No extraction, no parsing, just USE IT.
    """
    result = {
        "invalidated": False,
        "reason": None,
        "action": None,
        "pattern": None
    }
    
    if position_type != "LONG":
        # SHORT invalidation not implemented — only LONG positions are monitored.
        # If SHORT trading support is added, add SHORT-specific conditions here.
        return result
    
    if extractor is None:
        resolver = get_resolver(horizon)
        extractor = resolver.extractor
    
    # ✅ Get current price (safe extraction)
    current_price = _get_val(indicators , "price")
    if not current_price:
        return result
    
    for pattern_name, pattern_data in detected_patterns.items():
        if not pattern_data.get("found"):
            continue
        
        # ✅ Get pattern context from extractor
        pattern_ctx = extractor.get_pattern_context(pattern_name)
        if not pattern_ctx or not pattern_ctx.invalidation:
            continue
        
        # ✅ USE CONFIG DIRECTLY - extractor already resolved everything!
        inval_cfg = pattern_ctx.invalidation
        
        # ✅ Extract values directly from top level (no nesting!)
        gates = inval_cfg.get("gates", {})
        logic = inval_cfg.get("_logic", "AND")
        
        if not gates:
            logger.debug(f"[{symbol}] No invalidation gates for {pattern_name}")
            continue
        
        # ✅ Extract pattern metadata (FIXED: raw.meta nesting)
        meta = pattern_data.get("raw", {}).get("meta", {})
        
        # ✅ Build namespace for condition evaluation
        namespace = _build_invalidation_namespace(meta, indicators)
        
        # ✅ Evaluate gates       
        is_broken, gate_results = extractor.evaluate_invalidation_gates(gates, namespace)
        
        logger.info(
            f"[{symbol}][{pattern_name}] Breakdown check: {is_broken} logic={logic})"
        )
        
        if not is_broken:
            # Pattern recovered - clean up tracking state
            state = get_breakdown_state(symbol, pattern_name, horizon)
            if state:
                delete_breakdown_state(symbol, pattern_name, horizon)
                logger.info(f"✅ [{symbol}] {pattern_name} recovered")
            continue
        
        # ✅ Multi-candle confirmation logic
        triggered_durations = [res["duration"] for res in gate_results if res["triggered"]]
        duration_candles = max(triggered_durations) if triggered_durations else 1
        needs_duration = duration_candles > 1
        
        if needs_duration:
            # Track breakdown progress in DB
            state = get_breakdown_state(symbol, pattern_name, horizon)
            
            if not state:
                # Start tracking
                # ✅ FIX: Prioritized pivot level lookup (Issue 4)
                pivot_level = (
                    namespace.get("pivot_point")
                    or namespace.get("box_low")
                    or namespace.get("handle_low")
                    or current_price
                )
                save_breakdown_state(
                    symbol, pattern_name, horizon,
                    current_price,
                    pivot_level,
                    str(gates)
                )
                logger.info(
                    f"⚠️ [{symbol}] {pattern_name} breakdown started (1/{duration_candles})"
                )
                return result
            
            else:
                count = state.get("candle_count", 0)
                
                if count >= duration_candles - 1:  # Fixed: was count >= duration_candles (off-by-one)
                    # Breakdown confirmed!
                    delete_breakdown_state(symbol, pattern_name, horizon)
                    
                    # ✅ Get action from config (also at top level!)
                    action = inval_cfg.get("action", "EXIT_ON_CLOSE")
                    
                    result["invalidated"] = True
                    result["reason"] = f"{pattern_name} breakdown confirmed ({count} candles)"
                    result["action"] = action
                    result["pattern"] = pattern_name
                    
                    logger.warning(
                        f"❌ [{symbol}] {pattern_name} INVALIDATED after {count} candles"
                    )
                    return result
                
                else:
                    # Continue tracking
                    update_breakdown_state(symbol, pattern_name, horizon)
                    logger.info(
                        f"⏳ [{symbol}] {pattern_name} breakdown progressing "
                        f"({count + 1}/{duration_candles})"
                    )
                    return result
        
        else:
            # Immediate invalidation (no duration needed)
            action = inval_cfg.get("action", "EXIT_ON_CLOSE")
            result["invalidated"] = True
            result["reason"] = f"{pattern_name} breakdown"
            result["action"] = action
            result["pattern"] = pattern_name
            
            logger.warning(f"❌ [{symbol}] {pattern_name} INVALIDATED (immediate)")
            return result
    
    return result
# ============================================================
# 4. MAIN: ENHANCE EXECUTION CONTEXT
# ============================================================

def enhance_execution_context(
    eval_ctx: Dict[str, Any],
    exec_ctx: Dict[str, Any],
    indicators: Dict[str, Any],
    symbol: str,
    horizon: str = "short_term",
    extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    ✅ Idempotency guard + ordering dependency documented.

    Post-processes execution context with real-time pattern checks.

    CHANGES FROM loder versions
    - ✅ Added idempotency guard — prevents double-penalty on retry/re-evaluation
    - ✅ Explicit ordering note — must run before finalize_trade_decision
    - ✅ Isolated confidence mutation into helper for clarity

    Args:
        eval_ctx: Evaluation context (contains detected patterns).
                  ⚠️ This dict is mutated directly (confidence.clamped,
                  confidence.pattern_expiry_penalty, confidence.adjustments).
                  Caller must NOT pass a shared reference if immutability
                  is required downstream.
        exec_ctx: Execution context from resolver
        indicators: Technical indicators
        symbol: Stock symbol
        horizon: Trading timeframe
        extractor: QueryOptimizedExtractor instance (optional)

    Returns:
        Enhanced execution context with:
        - Pattern warnings (expiration)
        - Pattern invalidation flags
        - RR regime metadata (multipliers for target adjustment)
        - Pattern timeline (unified calculation)

    ⚠️ ORDERING REQUIREMENT:
        Must be called BEFORE finalize_trade_decision().
        finalize_trade_decision reads eval_ctx["confidence"]["clamped"]
        which this function may reduce on pattern expiry.
    """
    try:
        # Get extractor if not provided
        if extractor is None:
            resolver = get_resolver(horizon)
            extractor = resolver.extractor

        detected_patterns = eval_ctx.get("patterns", {})

        if not detected_patterns:
            logger.debug(f"[{symbol}] No patterns to enhance")
            return exec_ctx

        # ===================================================================
        # ✅ v5.0: Use shared pattern metadata extractor
        # ===================================================================
        pattern_meta = extract_pattern_execution_metadata(
            eval_ctx, horizon, extractor
        )

        if pattern_meta.get("available"):
            exec_ctx["pattern_meta"] = pattern_meta
            logger.debug(
                f"[{symbol}] Pattern metadata: {pattern_meta['pattern']} "
                f"(quality={pattern_meta['quality']}, age={pattern_meta['age_candles']})"
            )

        # ===================================================================
        # ✅ v5.0: Use shared timeline calculator
        # ===================================================================
        if pattern_meta.get("available"):
            risk = exec_ctx.get("risk", {})
            trend = eval_ctx.get("trend", {})

            timeline = calculate_pattern_timeline(
                pattern_meta, risk, trend, horizon
            )

            if timeline.get("available"):
                exec_ctx["timeline"] = timeline
                logger.info(
                    f"[{symbol}] Timeline: T1={timeline['t1_estimate']}, "
                    f"T2={timeline.get('t2_estimate', 'N/A')} "
                    f"(confidence={timeline['confidence']})"
                )

        # ===================================================================
        # 1. Check Pattern Expiration
        # ===================================================================
        expiration = check_pattern_expiration(detected_patterns, horizon, extractor)
        if expiration["expired"]:
            exec_ctx["pattern_warnings"] = exec_ctx.get("pattern_warnings", [])
            exec_ctx["pattern_warnings"].append({
                "type": "expiration",
                "pattern": expiration["pattern"],
                "reason": expiration["reason"]
            })

            # Idempotency guard — skip if penalty already applied.
            # Prevents confidence being reduced twice if this function is called
            # more than once on the same eval_ctx (e.g. retry or batch re-eval).
            confidence_ctx = eval_ctx.get("confidence", {})
            if confidence_ctx.get("pattern_expiry_penalty") is not None:
                logger.debug(
                    f"[{symbol}] Pattern expiry penalty already applied "
                    f"({confidence_ctx['pattern_expiry_penalty']}%), skipping"
                )
            else:
                # ✅ Snapshot current confidence BEFORE any mutation
                current_conf = confidence_ctx.get("clamped", 50)
                penalty = -20
                new_conf = max(0, current_conf + penalty)

                # Apply penalty — eval_ctx is mutated intentionally here.
                # finalize_trade_decision reads this value downstream.
                eval_ctx["confidence"]["clamped"] = new_conf
                eval_ctx["confidence"]["pattern_expiry_penalty"] = penalty

                # ✅ Audit trail — append to adjustments breakdown.
                # Re-read adjustments AFTER mutation so we operate on the
                # live dict, not a stale snapshot captured before the write.
                adjustments = eval_ctx["confidence"].get("adjustments", {})
                breakdown = adjustments.get("breakdown", [])
                breakdown.append(
                    f"Pattern expired ({expiration['pattern']}): {penalty}%"
                )
                adjustments["breakdown"] = breakdown
                eval_ctx["confidence"]["adjustments"] = adjustments

                logger.warning(
                    f"[{symbol}] Pattern expired: {expiration['reason']} "
                    f"(confidence {current_conf} → {new_conf}, penalty={penalty}%)"
                )

        # ===================================================================
        # 2. Check Pattern Invalidation
        # ===================================================================
        trend_dir = eval_ctx.get("trend", {}).get("classification", {}).get("direction", "bullish")
        position_type = "SHORT" if trend_dir == "bearish" else "LONG"

        invalidation = check_pattern_invalidation(
            detected_patterns,
            indicators,
            symbol,
            position_type=position_type,
            horizon=horizon,
            extractor=extractor
        )

        if invalidation["invalidated"]:
            exec_ctx["pattern_invalidation"] = invalidation

            # Mark execution as blocked
            if "can_execute" not in exec_ctx:
                exec_ctx["can_execute"] = {"can_execute": False, "failures": []}

            exec_ctx["can_execute"]["can_execute"] = False
            exec_ctx["can_execute"]["failures"].append(
                f"Pattern invalidation: {invalidation['reason']}"
            )

            logger.error(
                f"[{symbol}] ❌ Pattern invalidated: {invalidation['reason']} "
                f"(action={invalidation['action']})"
            )

        # ===================================================================
        # 3. Get RR Regime Multipliers
        # ===================================================================
        t1_mult, t2_mult, regime = get_rr_regime_multipliers(
            eval_ctx, horizon, extractor
        )

        exec_ctx["rr_regime"] = {
            "regime": regime,
            "t1_multiplier": t1_mult,
            "t2_multiplier": t2_mult,
            "adx": (
                indicators.get("adx", {}).get("value")
                if isinstance(indicators.get("adx"), dict)
                else indicators.get("adx")
            )
        }

        logger.info(
            f"[{symbol}] ✅ Execution context enhanced | "
            f"Pattern={pattern_meta.get('pattern', 'N/A')} | "
            f"RR regime={regime} | "
            f"Timeline={'✅' if exec_ctx.get('timeline', {}).get('available') else '❌'}"
        )

        # ===================================================================
        # 4. Market-Adaptive RR Adjustment
        # ===================================================================
        risk = exec_ctx.get("risk", {})
        if risk:
            market_adjusted = adjust_targets_for_market_conditions(
                risk_data=risk,
                indicators=indicators,
                fundamentals=eval_ctx.get("fundamentals", {}),
                horizon=horizon
            )

            if market_adjusted.get("adjusted"):
                exec_ctx["market_adjusted_targets"] = market_adjusted

        return exec_ctx

    except Exception as e:
        logger.error(
            f"[{symbol}] enhance_execution_context failed: {e}", exc_info=True
        )
        return exec_ctx
    
def calculate_adaptive_spread_cost(
    current_price: float,
    rvol: float,
    market_cap: float,
    horizon: str
) -> float:
    """
    Calculate spread cost based on stock liquidity and horizon.
    
    Args:
        current_price: Current stock price
        rvol: Relative volume (actual vs average)
        market_cap: Market capitalization in crores
        horizon: Trading timeframe
    
    Returns:
        Spread cost in rupees
    """
    # Base spread by market cap
    if market_cap > 50000:  # Large cap (₹50,000+ crore)
        base_spread_pct = 0.05
    elif market_cap > 10000:  # Mid cap (₹10,000+ crore)
        base_spread_pct = 0.15
    else:  # Small cap
        base_spread_pct = 0.25
    
    # Adjust for volume surge
    if rvol >= 3.0:
        volume_factor = 0.7  # 30% discount for high volume
    elif rvol >= 1.5:
        volume_factor = 1.0  # No adjustment
    else:
        volume_factor = 1.3  # 30% premium for low volume
    
    # Adjust for horizon
    horizon_factors = {
        "intraday": 1.0,      # Full spread for quick trades
        "short_term": 0.8,    # Slightly better for swing
        "long_term": 0.6,     # Much better for position
        "multibagger": 0.5    # Best for long holds
    }
    horizon_factor = horizon_factors.get(horizon, 1.0)
    
    # Calculate final spread
    final_spread_pct = base_spread_pct * volume_factor * horizon_factor
    spread_cost = current_price * final_spread_pct / 100
    
    return spread_cost

@log_failures(return_on_error={}, critical=False)
def extract_pattern_execution_metadata(
    eval_ctx: Dict[str, Any],
    horizon: str,
    extractor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    ✅ REFACTORED: Focuses on timeline/physics, not invalidation.
    
    Invalidation logic is now handled by trade_enhancer.py using conditions.
    """
    try:
        patterns = eval_ctx.get("patterns", {})
        
        if not patterns:
            logger.debug(f"[{horizon}] No patterns detected")
            return {"available": False}
        
        if extractor is None:
            resolver = get_resolver(horizon)
            extractor = resolver.extractor
        
        # Find best pattern WITH CONTEXT
        best_pattern = None
        best_quality = -1
        
        for pattern_name, pattern_data in patterns.items():
            if not pattern_data.get("found"):
                continue
            
            # ✅ FIX: Skip patterns without structural context
            if not extractor.get_pattern_context(pattern_name):
                logger.warning(
                    f"[{horizon}] Pattern '{pattern_name}' has no context in extractor"
                )
                continue
            
            raw = pattern_data.get("raw", {})
            quality = raw.get("quality", 0)
            
            if quality > best_quality:
                best_pattern = pattern_name
                best_quality = quality
        
        if not best_pattern:
            logger.debug(f"[{horizon}] No valid patterns found")
            return {"available": False}
        
        pattern_data = patterns[best_pattern]
        raw = pattern_data.get("raw", {})
        meta = raw.get("meta", {})
        
        # Get pattern config
        pattern_ctx = extractor.get_pattern_context(best_pattern)
        if not pattern_ctx:
            logger.warning(
                f"[{horizon}] Pattern '{best_pattern}' has no context in extractor"
            )
            return {"available": False}
        
        # ✅ Extract invalidation timeline (for monitoring)
        max_duration = None
        if pattern_ctx.invalidation:
            exp_cfg = pattern_ctx.invalidation.get("expiration", {})
            if exp_cfg.get("enabled"):
                max_duration = exp_cfg.get("max_duration_candles")

                # Handle horizon-specific durations
                if isinstance(max_duration, dict):
                    max_duration = max_duration.get(horizon)

        # ✅ Return focused metadata (timeline + physics)
        return {
            "available": True,
            "pattern": best_pattern,
            
            # Physics
            "duration_multiplier": pattern_ctx.physics.get("duration_multiplier", 1.0),
            "target_ratio": pattern_ctx.physics.get("target_ratio", 1.0),
            
            # Reliability Treat failure_rate and min_quality as optional future enrichment
            "failure_rate": getattr(pattern_ctx, "failure_rate", None),
            "min_quality": getattr(pattern_ctx, "min_quality", None),
            
            # Timeline
            "invalidation_timeline": max_duration,
            "age_candles": meta.get("age_candles", 0),
            "formation_timestamp": meta.get("formation_time"),
            
            # Quality
            "quality": best_quality,
            
            # ❌ REMOVED: reference_level (not needed - conditions handle it)
            # ✅ KEEP: Full config for enhancer to use
            "physics": pattern_ctx.physics,
            "invalidation": pattern_ctx.invalidation  # ← Enhancer uses this
        }
    
    except Exception as e:
        logger.error(f"Pattern metadata extraction failed: {e}", exc_info=True)
        return {"available": False, "error": str(e)}


# ==============================================================================
# 7. ✅ NEW: PATTERN TIMELINE CALCULATION (Shared Utility)
# ==============================================================================

@log_failures(return_on_error={}, critical=False)
def calculate_pattern_timeline(
    pattern_meta: Dict[str, Any],
    risk: Dict[str, Any],
    trend: Dict[str, Any],
    horizon: str
) -> Dict[str, Any]:
    """
    ✅ UNIFIED: Calculate pattern-based timeline estimation.
    """
    try:
        if not pattern_meta.get("available"):
            return {"available": False, "reason": "No pattern metadata"}
        
        if not risk or not risk.get("targets"):
            return {"available": False, "reason": "No targets in risk model"}
        
        # ===================================================================
        # STEP 1: Extract key values
        # ===================================================================
        entry = risk["entry_price"]
        targets = risk["targets"]
        atr = risk.get("atr", 0)
        if atr <= 0:
            return {"available": False, "reason": "ATR not available in risk model"}
        
        # ✅ Initialize duration_mult with default from pattern metadata
        config_duration_mult = pattern_meta.get("duration_multiplier", 1.0)
        duration_mult = config_duration_mult  # ✅ Default value
        data_source = "config_physics"  # ✅ Default source
        
        # 🆕 ADD: Query historical velocity
        pattern_name = pattern_meta.get("pattern")
        trend_regime = trend.get("regime", "normal")
        
        historical_stats = None
        if pattern_name:
            historical_stats = get_pattern_velocity_stats(
                pattern_name=pattern_name,
                horizon=horizon,
                trend_regime=trend_regime,
                min_samples=10
            )
        
        # ✅ FIX: Calculate bars_per_unit BEFORE using it
        bars_per_unit = {
            "intraday": 26,      # 26 bars = 1 hour (15min candles)
            "short_term": 1,     # 1 bar = 1 day
            "long_term": 1,      # 1 bar = 1 week  (was 0.2 → 5× inflation)
            "multibagger": 1     # 1 bar = 1 month (was 0.05 → 20× inflation)
        }.get(horizon, 1)
        
        # 🆕 MODIFY: Adjust duration multiplier with historical data
        if historical_stats and historical_stats.get("sample_size", 0) >= 10:
            actual_median_days = historical_stats.get("median_days_to_t1", 0)
            
            if actual_median_days and actual_median_days > 0:
                # ✅ FIX: Estimate what config physics would predict in DAYS,
                # not in bars, using the correct horizon conversion factor.
                # Old code: config_duration_mult * 5 hardcoded "5 days/unit" for all
                # horizons — wildly wrong for intraday (unit=hours) or multibagger
                # (unit=months).  Use the actual bars_per_unit to convert:
                #   bars/unit → units → calendar days (approximate)
                horizon_to_days = {
                    "intraday": 1/26,      # 1 bar = 15min; 26 bars ≈ 1 trading hour ≈ 0.04 days
                    "short_term": 1.0,     # 1 bar = 1 day
                    "long_term": 5.0,      # 1 bar ≈ 1 week (5 trading days)
                    "multibagger": 20.0    # 1 bar ≈ 1 month (20 trading days)
                }
                days_per_bar = horizon_to_days.get(horizon, 1.0)
                # A "unit" is bars_per_unit bars, so 1 unit = bars_per_unit * days_per_bar days
                days_per_unit = bars_per_unit * days_per_bar
                # config predicts: config_duration_mult units → that many * days_per_unit days
                # (rough: T1 is ~1 unit away for most patterns before mult scaling)
                config_estimate_days = config_duration_mult * days_per_unit
                
                if config_estimate_days > 0:
                    learned_multiplier = actual_median_days / config_estimate_days
                    
                    # Blend historical data with config (70% historical, 30% config)
                    duration_mult = (
                        learned_multiplier * 0.7 +
                        config_duration_mult * 0.3
                    )
                    data_source = f"historical (n={historical_stats['sample_size']})"
                    
                    logger.debug(
                        f"Using learned multiplier: {duration_mult:.2f} "
                        f"(historical={learned_multiplier:.2f}, config={config_duration_mult:.2f})"
                    )
        
        # ===================================================================
        # STEP 2: Determine regime factor
        # ===================================================================
        regime = trend.get("regime", "normal")
        regime_factor = {
            "strong": 1.3,   # Faster movement
            "normal": 1.0,
            "weak": 0.7      # Slower movement
        }.get(regime, 1.0)
        
        # ===================================================================
        # STEP 3: Calculate unit names (already have bars_per_unit above)
        # ===================================================================
        unit_name = {
            "intraday": "days",    # ✅ Correction: 26 bars = 1 day in Indian market context (Issue 5)
            "short_term": "days",
            "long_term": "weeks",
            "multibagger": "months"
        }.get(horizon, "days")
        
        # ===================================================================
        # STEP 4: Calculate raw bar counts
        # ===================================================================
        distances = [abs(t - entry) for t in targets]
        
        # T1 timeline
        t1_bars = (distances[0] / atr) * duration_mult / regime_factor
        
        # T2 timeline (if available)
        t2_bars = None
        if len(distances) > 1:
            t2_bars = (distances[1] / atr) * duration_mult / regime_factor
        
        # ===================================================================
        # STEP 5: Convert to readable units
        # ===================================================================
        t1_units = max(1, int(t1_bars / bars_per_unit))
        t1_estimate = f"{t1_units} {unit_name}"
        
        t2_estimate = None
        if t2_bars:
            t2_units = max(1, int(t2_bars / bars_per_unit))
            t2_estimate = f"{t2_units} {unit_name}"
        
        # ===================================================================
        # STEP 6: Determine confidence level
        # ===================================================================
        failure_rate = pattern_meta.get("failure_rate", 0.5)
        if failure_rate is None:
            failure_rate = 0.5  # Default if unknown
            logger.debug("Failure rate unknown, defaulting to 0.5")
        if failure_rate < 0.3:
            confidence = "high"
        elif failure_rate < 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # ===================================================================
        # STEP 7: Add invalidation warning
        # ===================================================================
        invalidation_warning = None
        invalidation_timeline = pattern_meta.get("invalidation_timeline")
        age_candles = pattern_meta.get("age_candles", 0)
        
        if invalidation_timeline:
            remaining = invalidation_timeline - age_candles
            if remaining > 0:
                remaining_units = max(1, int(remaining / bars_per_unit))
                invalidation_warning = (
                    f"Pattern expires in {remaining} candles "
                    f"({remaining_units} {unit_name})"
                )
        
        # ===================================================================
        # STEP 8: Build result
        # ===================================================================
        return {
            "available": True,
            "t1_estimate": t1_estimate,
            "t2_estimate": t2_estimate,
            "confidence": confidence,
            "pattern_age": age_candles,
            "invalidation_warning": invalidation_warning,
            "data_source": data_source,  # 🆕 Track what drove the estimate
            "raw": {
                "t1_bars": t1_bars,
                "t2_bars": t2_bars,
                "regime": regime,
                "historical_stats": historical_stats,  # ✅ Move to raw (Issue 6)
                "regime_factor": regime_factor,
                "duration_multiplier": duration_mult  # ✅ Add for transparency
            }
        }
        
    except Exception as e:
        logger.error(f"Timeline calculation failed: {e}", exc_info=True)
        return {"available": False, "error": str(e)}

# ============================================================
# INTERNAL HELPERS (Config Retrieval)
# ============================================================
def _build_invalidation_namespace(meta: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a flattened namespace for condition evaluation.
    Merges Pattern Meta (static physics) + Indicators (real-time price).
    """
    namespace = {}
    
    # 1. Flatten Pattern Metadata (e.g., box_low, handle_low)
    for k, v in meta.items():
        if isinstance(v, (int, float, str, bool)):
            namespace[k] = v
            
    # 2. Flatten Real-Time Indicators (e.g., price, rsi, bbWidth)
    for k, v in indicators.items():
        # Handle dict format: {'value': 950.0, ...}
        if isinstance(v, dict):
            val = v.get("value")
            if val is not None:
                namespace[k] = val
                
            # Also support 'raw' if 'value' is missing
            elif "raw" in v:
                namespace[k] = v["raw"]
        
        # Handle direct values
        elif isinstance(v, (int, float, str, bool)):
            namespace[k] = v

    # 3. Add derived math helpers if needed (optional)
    if "price" in namespace and "box_low" in namespace:
        namespace["dist_to_stop"] = (namespace["price"] - namespace["box_low"]) / namespace["price"]

    return namespace

def _get_rr_regime_config(extractor) -> Dict[str, Any]:
    """Get RR regime adjustments via extractor."""
    risk_mgmt = extractor.get_risk_management_config()
    return risk_mgmt.get("rr_regime_adjustments", {})

def validate_execution_rr(
    exec_ctx: Dict[str, Any], 
    extractor: Any
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Canonical RR validator.
    Reads thresholds ONLY from risk management config via extractor.
    """
    # --- Get market-adjusted RR values ---
    market = exec_ctx.get("market_adjusted_targets", {})

    # ✅ FALLBACK: If market targets unavailable, use structural risk model
    if not market or not market.get("adjusted"):
        logger.warning(
            "[validate_execution_rr] ⚠️ Market-adjusted targets unavailable, "
            "falling back to structural risk model"
        )
        risk = exec_ctx.get("risk", {})
        structural_rr = risk.get("rrRatio") or 0

        rr_gates = extractor.get_rr_gates()
        min_structural = rr_gates.get("min_structural", 2.0)

        # Check structural RR against minimum
        if structural_rr >= min_structural:
            return True, "Structural RR OK (no market adjustment)", "structural"
        else:
            return False, (
                f"RR insufficient (structural={structural_rr:.2f}, "
                f"min={min_structural})"
            ), None

    # --- Proceed with market-adjusted values ---
    rrt1 = market.get("execution_rr_t1") or 0  # Fixed: handles both missing key AND None value
    rrt2 = market.get("execution_rr_t2") or 0  # Fixed: key exists with None, .get default won't help
    structural_rr = market.get("structural_rr") or 0

    # --- Get risk config via passed extractor ---
    rr_gates = extractor.get_rr_gates()

    min_t1 = rr_gates.get("min_t1", 1.5)
    min_t2 = rr_gates.get("min_t2", 2.0)
    min_structural = rr_gates.get("min_structural", 2.0)
    execution_floor = rr_gates.get("execution_floor", 1.0)

    # --- Absolute Floor Check ---
    if rrt1 < execution_floor:
         return False, f"RR T1 ({rrt1:.2f}) below hard floor ({execution_floor})", None

    # --- Decision Logic ---
    if rrt1 >= min_t1:
        return True, None, "T1"

    if rrt2 >= min_t2 and structural_rr >= min_structural:
        return True, "Using T2 target due to low T1 RR", "T2"

    return False, (
        f"Execution RR too low | "
        f"T1={rrt1:.2f} (<{min_t1}) | "
        f"T2={rrt2:.2f} (<{min_t2}) | "
        f"Structural={structural_rr:.2f} (<{min_structural})"
    ), None