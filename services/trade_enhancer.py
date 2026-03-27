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
import copy
import hashlib
import json
from typing import Dict, Any, Optional, Tuple


from config.config_utility.logger_config import log_failures
from config.config_utility.market_utils import get_current_utc
from services.data_fetch import _get_val, safe_float
from services.patterns.horizon_constants import HORIZON_WINDOWS_SECONDS as HORIZON_WINDOWS  # ✅ W46

# ✅ NEW v5.0: Import shared utilities
from config.config_helpers import get_resolver
from services.patterns.pattern_velocity_tracking import get_pattern_velocity_stats
from services.patterns.utils import _classify_volatility
from services.patterns.pattern_state_manager import (
    get_breakdown_state,
    save_breakdown_state,
    update_breakdown_state,
    delete_breakdown_state
)
logger = logging.getLogger(__name__)
def adjust_targets_for_market_conditions(
    risk_data: Dict[str, Any],
    indicators: Dict[str, Any],
    fundamentals: Dict[str, Any],
    horizon: str,
    extractor: Optional[Any] = None,
    exec_ctx: Optional[Dict[str, Any]] = None
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
        
        # ✅ P2-4 GUARD: Avoid LONG misclassification for missing SL
        if not pattern_sl:
            return {"adjusted": False, "reason": "Missing SL value in risk data"}
        
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
        # === Fetch Config-Driven Policy Constants ===
        # Replaces module-level hardcodes (W37)
        if extractor is None:
            resolver = get_resolver(horizon=horizon)
            extractor = resolver.extractor
        
        risk_management_cfg = extractor.get_risk_management_config()
        
        # Pull directly from config with simple defaults mirroring master_config
        vol_buffer_factors = risk_management_cfg.get("volatility_buffer_factors", {})
        min_sl_atr = risk_management_cfg.get("min_sl_atr_multiples", {})
        targ_adj_factors = risk_management_cfg.get("target_adjustment_factors", {})
        
        # Resolve NameError for volatility_regime
        volatility_regime = _classify_volatility(atr_pct)
        
        min_rr_by_trend = risk_management_cfg.get("min_rr_by_trend", {})
        
        # === Volatility-Adjusted SL ===
        vol_buffer = vol_buffer_factors.get(volatility_regime, 0.25)
        min_sl_mult = min_sl_atr.get(horizon, 2.0)
        min_sl_distance = current_atr * (min_sl_mult + vol_buffer)
        
        if direction == "LONG":
            volatility_sl = current_price - min_sl_distance
            execution_sl = min(pattern_sl, volatility_sl)  # Wider stop
            sl_buffer = pattern_sl - execution_sl
        else:  # SHORT
            volatility_sl = current_price + min_sl_distance
            execution_sl = max(pattern_sl, volatility_sl)  # Wider stop
            sl_buffer = execution_sl - pattern_sl
        
        # === Target Selection: Prioritize Structural vs Market ===
        target_factor = targ_adj_factors.get(volatility_regime, 1.0)
        
        # ✅ Fix 9A.4: Apply RR Regime Multipliers (t1_mult/t2_mult)
        if exec_ctx:
            regime_cfg = exec_ctx.get("rr_regime", {})
            t1_regime_mult = regime_cfg.get("t1_multiplier", 1.0) or 1.0
            t2_regime_mult = regime_cfg.get("t2_multiplier", 1.0) or 1.0
        else:
            t1_regime_mult = t2_regime_mult = 1.0
        
        # Calculate RR for structural targets first
        if direction == "LONG":
            struct_risk = current_price - execution_sl
            struct_reward = pattern_t1 - current_price
        else:
            struct_risk = execution_sl - current_price
            struct_reward = current_price - pattern_t1
            
        struct_rr = struct_reward / struct_risk if struct_risk > 0 else 0
        
        # TREND-BASED RR RELAXATION (Dynamic Gate)
        trend_strength = _get_val(indicators, "trendStrength", 5.0)
        effective_min_rr = 1.5
        for level in sorted(min_rr_by_trend.values(), key=lambda x: x["strength"], reverse=True):
            if trend_strength >= level["strength"]:
                effective_min_rr = level["min_rr"]
                break
        
        # Optimization: If structural RR passes relaxed gate, don't stretch target
        if struct_rr >= effective_min_rr:
            execution_t1 = pattern_t1
            execution_t2 = pattern_t2
            target_source = "structural_priority"
        else:
            # Stretch targets based on market volatility, anchored to current price.
            t1_distance = abs(pattern_t1 - current_price)
            if direction == "LONG":
                execution_t1 = current_price + (t1_distance * target_factor * t1_regime_mult)
            else:
                execution_t1 = current_price - (t1_distance * target_factor * t1_regime_mult)
            
            execution_t2 = None
            if pattern_t2:
                t2_distance = abs(pattern_t2 - current_price)
                if direction == "LONG":
                    execution_t2 = current_price + (t2_distance * target_factor * t2_regime_mult)
                else:
                    execution_t2 = current_price - (t2_distance * target_factor * t2_regime_mult)
            target_source = "market_adjusted"
        
        # Spread Cost
        spread_multiplier = 2.0 if atr_pct > 5.0 else (1.5 if atr_pct > 3.0 else 1.0)
        market_cap_data = fundamentals.get("marketCap", {})
        mc_raw = market_cap_data.get("raw", 0) if isinstance(market_cap_data, dict) else (market_cap_data or 0)
        market_cap_crores = mc_raw / 10000000
        spread_cost = calculate_adaptive_spread_cost(
            current_price=current_price,
            rvol=_get_val(indicators, "rvol", 1.0),
            market_cap=market_cap_crores,
            horizon=horizon,
            extractor=extractor
        )
        
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
            return {
                "adjusted": False, 
                "reason": "T1 target at or behind current price after spread",
                "is_hard_blocked": True # Spread is a hard block
            }
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

        # ── P3b: Guard against missing regime config ───────────────────────
        # If the regime key is absent from rr_cfg (e.g. horizon override
        # omitted it), derive sensible defaults from ADX so we never
        # silently return the weakest possible multipliers.
        adx_val = trend.get('adx') or 0.0
        if not regime_cfg:
            if adx_val >= 35:
                regime_cfg = {"t1_mult": 2.0, "t2_mult": 4.0}
            elif adx_val >= 20:
                regime_cfg = {"t1_mult": 1.5, "t2_mult": 3.0}
            else:
                regime_cfg = {"t1_mult": 1.2, "t2_mult": 2.5}
            logger.warning(
                f"RR regime '{regime}' not found in config — "
                f"using ADX-derived fallback (ADX={adx_val:.1f})"
            )
        # ── End P3b ────────────────────────────────────────────────────────

        t1_mult = regime_cfg.get("t1_mult", 1.5)
        t2_mult = regime_cfg.get("t2_mult", 3.0)

        # ✅ PATCH: Log tag for fallback vs config
        tag = "[FALLBACK]" if not rr_cfg.get(regime) else "[CONFIG]"
        adx_str = f"{adx_val:.1f}" if adx_val else "N/A"

        logger.info(
            f"{tag} RR regime '{regime}': ADX={adx_str}, "
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
    
    # ✅ W46: Use centralized window definitions
    candle_sec = HORIZON_WINDOWS.get(horizon, 86400)
    
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
                # W38 FIX: formation_ts may be a Unix float (normal case) or an
                # ISO 8601 string injected by some detectors.  Normalise to float
                # before doing arithmetic to prevent a TypeError crash.
                if isinstance(formation_ts, str):
                    try:
                        from datetime import datetime as _dt
                        formation_ts = _dt.fromisoformat(formation_ts).timestamp()
                    except Exception:
                        logger.warning(
                            f"{pattern_name}: Could not parse formation_ts ISO string "
                            f"'{formation_ts}' — skipping real-time age calculation"
                        )
                        formation_ts = None
                if formation_ts is not None:
                    seconds_since = get_current_utc().timestamp() - formation_ts
                    real_time_age = int(seconds_since / candle_sec)
                    current_age = max(cached_age, real_time_age)

                    logger.debug(
                        f"{pattern_name}: cached={cached_age}, real_time={real_time_age}, using={current_age}"
                    )
                else:
                    current_age = cached_age
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
    
    if position_type not in ["LONG", "SHORT"]:
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
        
        # ✅ W39 FIX: SHORT trades are CONFIRMED by breakdowns, not invalidated.
        # Only invalidate if is_broken AND it's a LONG trade.
        if is_broken and position_type == "SHORT":
            logger.info(f"✅ [{symbol}] {pattern_name} breakdown treated as CONFIRMATION for SHORT")
            continue

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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        # ===================================================================
        # ✅ NEW: Context Hash Verification (Risk 2)
        # ===================================================================
        try:
            stored_hash = eval_ctx.get("meta", {}).get("context_hash")
            if stored_hash:
                hash_payload = {
                    "indicators": indicators,
                    "fundamentals": eval_ctx.get("fundamentals", {})
                }
                current_hash = hashlib.md5(json.dumps(hash_payload, sort_keys=True, default=str).encode()).hexdigest()
                
                if current_hash != stored_hash:
                    # ✅ P2-2 FIX: Threshold check to filter price noise
                    _stored_px = _get_val(eval_ctx.get("indicators", {}), "price") or 0
                    _curr_px = _get_val(indicators, "price") or 0
                    _px_moved = abs(_curr_px - _stored_px) / _stored_px > 0.002 if _stored_px > 0 else True
                    
                    if _px_moved:
                        exec_ctx["stale_context"] = True
                        exec_ctx["warning"] = "Market moved significantly since evaluation"
                        logger.warning(f"[{symbol}] Stale context (hash mismatch & >0.2% price move)")
                    else:
                        logger.debug(f"[{symbol}] Context hash mismatch but price stable (<0.2% change).")
                else:
                    logger.debug(f"[{symbol}] Context hash verified.")
        except Exception as e:
            logger.error(f"[{symbol}] Context hash verification failed: {e}")

        # ✅ B5 FIX: Stop mutating eval_ctx (Discovery Purity)
        # eval_ctx = copy.deepcopy(eval_ctx) # Removed: do not mutate discovery data
        
        detected_patterns = eval_ctx.get("patterns", {})

        # ✅ S16 FIX: GENERIC setups still need market adaptation and the
        # direction conflict gate.  Only skip the pattern-specific phases
        # (expiration, invalidation, RR regime multipliers, pattern timeline).
        # Flag so the branches below can detect the short-circuit.
        setup_type = eval_ctx.get("setup", {}).get("type", "GENERIC")
        _is_generic = setup_type == "GENERIC"
        if _is_generic:
            logger.debug(f"[{symbol}] GENERIC setup — skipping pattern-specific enhancement phases")

        if not detected_patterns and not _is_generic:
            logger.debug(f"[{symbol}] No patterns to enhance")
            return exec_ctx, eval_ctx

        # ===================================================================
        # Pattern-specific phases — skipped for GENERIC setups (S16 fix).
        # Market adaptation (phase 4) and direction conflict gate run for all.
        # ===================================================================
        if not _is_generic:
            # v5.0: Use shared pattern metadata extractor
            pattern_meta = extract_pattern_execution_metadata(
                eval_ctx, horizon, extractor
            )

            if pattern_meta.get("available"):
                exec_ctx["pattern_meta"] = pattern_meta
                logger.debug(
                    f"[{symbol}] Pattern metadata: {pattern_meta['pattern']} "
                    f"(quality={pattern_meta['quality']}, age={pattern_meta['age_candles']})"
                )

            # v5.0: Use shared timeline calculator
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

            # 1. Check Pattern Expiration
            expiration = check_pattern_expiration(detected_patterns, horizon, extractor)
            if expiration["expired"]:
                exec_ctx["pattern_warnings"] = exec_ctx.get("pattern_warnings", [])
                exec_ctx["pattern_warnings"].append({
                    "type": "expiration",
                    "pattern": expiration["pattern"],
                    "reason": expiration["reason"]
                })

                # C21 FIX: idempotency guard — prevent the -20 penalty from
                # stacking on successive re-evaluations of the same context.
                _adj = exec_ctx.setdefault("confidence_adjustments", {
                    "total_penalty": 0,
                    "breakdown": []
                })
                if not _adj.get("expiry_applied"):
                    risk_cfg = extractor.get_risk_management_config()
                    penalty = risk_cfg.get("expiry_penalty", -20)
                    _adj["total_penalty"] += penalty
                    _adj["breakdown"].append(
                        f"Pattern expired ({expiration['pattern']}): {penalty}%"
                    )
                    _adj["expiry_applied"] = True  # C21: mark so second call is a no-op

                    logger.warning(
                        f"[{symbol}] Pattern expired: {expiration['reason']} (penalty={penalty}%)"
                    )
                else:
                    logger.debug(
                        f"[{symbol}] Expiry penalty already applied — skipping to prevent stacking"
                    )

            # 2. Check Pattern Invalidation
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
                if not exec_ctx.get("_invalidation_applied"):  # P1-2 idempotency guard
                    exec_ctx["pattern_invalidation"] = invalidation
                    exec_ctx.setdefault("can_execute", {"can_execute": False, "failures": []})
                    exec_ctx["can_execute"]["can_execute"] = False
                    exec_ctx["can_execute"]["is_hard_blocked"] = True
                    exec_ctx["can_execute"]["failures"].append(
                        f"Pattern invalidation: {invalidation['reason']}"
                    )
                    exec_ctx["_invalidation_applied"] = True

                logger.error(
                    f"[{symbol}] ❌ Pattern invalidated: {invalidation['reason']} "
                    f"(action={invalidation['action']})"
                )

            # 3. Get RR Regime Multipliers
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
                risk_data=exec_ctx["risk"],
                indicators=indicators,
                fundamentals=eval_ctx.get("fundamentals", {}),
                horizon=horizon,
                extractor=extractor,
                exec_ctx=exec_ctx  # ✅ Pass exec_ctx for RR Regime multipliers
            )

            if market_adjusted.get("adjusted"):
                exec_ctx["market_adjusted_targets"] = market_adjusted
            
            # ✅ BRIDGE: Propagate hard blocks (like spread failure) to can_execute
            if market_adjusted.get("is_hard_blocked"):
                exec_ctx.setdefault("can_execute", {})["can_execute"] = False
                exec_ctx["can_execute"]["is_hard_blocked"] = True
                if market_adjusted.get("reason"):
                    exec_ctx["can_execute"].setdefault("failures", []).append(
                        f"Market Adjustment: {market_adjusted['reason']}"
                    )


        # ===================================================================
        # C19 FIX: Direction Conflict Reconciliation — runs UNCONDITIONALLY
        # (was previously inside `if risk:` — missed when risk was empty)
        # ===================================================================
        # Bridge Trend vocabulary (BULLISH/BEARISH) with Execution vocabulary (LONG/SHORT)
        eval_direction = eval_ctx.get("trend", {}).get("classification", {}).get("direction", "neutral").upper()
        # Source execution direction from market_adjusted_targets or risk
        _mat = exec_ctx.get("market_adjusted_targets", {})
        execution_direction = _mat.get("direction", exec_ctx.get("risk", {}).get("direction", "neutral")).upper()

        DIRECTION_NORM = {
            "BULLISH": "LONG",
            "BEARISH": "SHORT",
            "NEUTRAL": "NEUTRAL",
            "LONG": "LONG",    # Forward compatibility
            "SHORT": "SHORT"
        }
        eval_dir_norm = DIRECTION_NORM.get(eval_direction, "UNKNOWN")
        
        # ✅ P1-3 FIX: Handle NEUTRAL conflicts explicitly
        if eval_dir_norm == "UNKNOWN":
             logger.error(f"[{symbol}] Unrecognized eval direction: {eval_direction}")

        if (
            eval_dir_norm != execution_direction
            and execution_direction != "NEUTRAL"
            and eval_dir_norm != "NEUTRAL"
            and eval_dir_norm != "UNKNOWN"
        ):
            exec_ctx["direction_conflict"] = True
            exec_ctx.setdefault("can_execute", {})
            exec_ctx["can_execute"]["can_execute"] = False
            exec_ctx["can_execute"].setdefault("failures", []).append(
                f"Direction Conflict: Trend={eval_direction} vs Execution={execution_direction}"
            )

            logger.warning(
                f"[{symbol}] DIRECTION CONFLICT | "
                f"Eval: {eval_direction} ({eval_dir_norm}) vs Execution: {execution_direction} | BLOCKING"
            )

        return exec_ctx, eval_ctx

    except Exception as e:
        logger.error(
            f"[{symbol}] enhance_execution_context failed: {e}", exc_info=True
        )
        return exec_ctx, eval_ctx
def calculate_adaptive_spread_cost(
    current_price: float,
    rvol: float,
    market_cap: float,
    horizon: str,
    extractor: Optional[Any] = None
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
    # === Config-Driven Thresholds (W37) ===
    if extractor is None:
        from config.config_helpers import get_resolver
        resolver = get_resolver(horizon)
        extractor = resolver.extractor
    
    risk_mgmt = extractor.get_risk_management_config()
    spread_cfg = risk_mgmt.get("spread_adjustment", {})
    
    # Base spread by market cap
    # Config format: {"market_cap_brackets": {"large_cap": {"min": 100000, "spread_pct": 0.001}, ...}}
    brackets = spread_cfg.get("market_cap_brackets", {})
    large_cap = brackets.get("large_cap", {"min": 100000, "spread_pct": 0.001})
    mid_cap = brackets.get("mid_cap", {"min": 10000, "max": 100000, "spread_pct": 0.002})
    small_cap = brackets.get("small_cap", {"max": 10000, "spread_pct": 0.005})

    if market_cap >= large_cap.get("min", 100000):
        base_spread_pct = large_cap.get("spread_pct", 0.001)
    elif market_cap >= mid_cap.get("min", 10000):
        base_spread_pct = mid_cap.get("spread_pct", 0.002)
    else:
        base_spread_pct = small_cap.get("spread_pct", 0.005)
    
    # Adjust for volume surge (fallback to defaults if missing in config)
    vol_cfg = risk_mgmt.get("adaptive_spread", {}).get("volume_adjustments", {})
    high_vol = vol_cfg.get("high", {"threshold": 3.0, "factor": 0.7})
    norm_vol = vol_cfg.get("normal", {"threshold": 1.5, "factor": 1.0})
    low_vol_factor = vol_cfg.get("low", {"factor": 1.3})["factor"]

    if rvol >= high_vol["threshold"]:
        volume_factor = high_vol["factor"]
    elif rvol >= norm_vol["threshold"]:
        volume_factor = norm_vol["factor"]
    else:
        volume_factor = low_vol_factor
    
    # Adjust for horizon
    horizon_factors = spread_cfg.get("horizon_factors", {
        "intraday": 1.0,
        "short_term": 0.8,
        "long_term": 0.6
    })
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
            # ✅ PATCH B: Check multiple levels for quality
            quality = raw.get("quality") or pattern_data.get("quality") or 0
            quality = float(quality)
            
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

def calculate_pattern_timeline(
    pattern_meta: Dict[str, Any],
    risk: Dict[str, Any],
    trend: Dict[str, Any],
    horizon: str,
    exec_ctx: Dict[str, Any] = None
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
        
        # ✅ Fix 11: NSE session = 9:15–15:30 = 375 min = exactly 25 × 15-min candles.
        # Old value was 26, creating a 390-min (6.5h) phantom day.
        # ✅ Fix 5.5-1: Corrected comment to reflect 1 trading day (25-26 bars of 15m)
        bars_per_unit = {
            "intraday": 25,   # ~1 trading day (15min candles, 09:15-15:30 IST)
            "short_term": 1, 
            "long_term": 1,
            "multibagger": 4  # ✅ Fix 9B.4: 4 weekly bars = ~1 month
        }.get(horizon, 1)
        
        # 🆕 MODIFY: Adjust duration multiplier with historical data
        if historical_stats and historical_stats.get("sample_size", 0) >= 10:
            learned_multiplier = historical_stats["avg_duration_mult"]
            n = historical_stats["sample_size"]
            
            # ✅ Fix 5.5-2: Continuous scaling formula for historical weight
            # Ramps from 30% (at n=10) to 90% (at n=100)
            hist_weight = min(0.90, 0.3 + 0.6 * (n - 10) / 90)
            duration_mult = (learned_multiplier * hist_weight + config_duration_mult * (1 - hist_weight))
            
            logger.info(
                f"[{exec_ctx.get('symbol', 'UNKNOWN')}] Applied historical velocity (n={n}, "
                f"weight={hist_weight:.2f}): {duration_mult:.2f}"
            )
            data_source = f"historical (n={historical_stats['sample_size']})"

            actual_median_days = historical_stats.get("median_days_to_t1", 0)
            
            if actual_median_days and actual_median_days > 0:
                # ✅ FIX: Estimate what config physics would predict in DAYS,
                # not in bars, using the correct horizon conversion factor.
                # Old code: config_duration_mult * 5 hardcoded "5 days/unit" for all
                # horizons — wildly wrong for intraday (unit=hours) or long-term/multibagger
                # (unit=months).  Use the actual bars_per_unit to convert:
                #   bars/unit → units → calendar days (approximate)
                horizon_to_days = {
                    "intraday": 1/25,      # 1 bar = 15min; 25 bars = 1 NSE trading day
                    "short_term": 1.0,     # 1 bar = 1 day
                    "long_term": 5.0      # 1 bar ≈ 1 week (5 trading days)
                }
                days_per_bar = horizon_to_days.get(horizon, 1.0)
                # A "unit" is bars_per_unit bars, so 1 unit = bars_per_unit * days_per_bar days
                days_per_unit = bars_per_unit * days_per_bar
                # config predicts: config_duration_mult units → that many * days_per_unit days
                # (rough: T1 is ~1 unit away for most patterns before mult scaling)
                config_estimate_days = config_duration_mult * days_per_unit
                
                if config_estimate_days > 0:
                    learned_multiplier = actual_median_days / config_estimate_days
                    # W44 FIX: Clamp learned_multiplier to prevent outlier math drift.
                    # Without this, a single anomalous trade (e.g. T1 hit in 200 days
                    # on a pattern estimated at 1 day) would produce a 200x multiplier
                    # that inflates all subsequent timeline estimates.
                    learned_multiplier = max(0.1, min(10.0, learned_multiplier))

                    # Blend historical data with config (70% historical, 30% config)
                    duration_mult = (
                        learned_multiplier * 0.7 +
                        config_duration_mult * 0.3
                    )
                    # W44 FIX: Clamp the blended result too so extreme config
                    # values cannot escape the safety envelope.
                    duration_mult = max(0.1, min(10.0, duration_mult))
                    data_source = f"historical (n={historical_stats['sample_size']})"

                    logger.debug(
                        f"Using learned multiplier: {duration_mult:.2f} "
                        f"(historical={learned_multiplier:.2f}, config={config_duration_mult:.2f})"
                    )
        
        # ===================================================================
        # STEP 2: Determine regime factor
        # ===================================================================
        # ✅ W13 FIX: Use multipliers from exec_ctx if available (calculated via get_rr_regime_multipliers)
        rr_regime = exec_ctx.get("rr_regime", {})
        regime = rr_regime.get("regime") or trend.get("regime", "normal")
        
        # Use T1 multiplier as the base duration factor for timeline estimates
        regime_factor = rr_regime.get("t1_multiplier")
        
        if regime_factor is None:
            # Fallback if exec_ctx doesn't have it
            regime_factor = {
                "strong": 1.3,
                "normal": 1.0,
                "weak": 0.7
            }.get(regime, 1.0)
        
        # ===================================================================
        # STEP 3: Calculate unit names (already have bars_per_unit above)
        # ===================================================================
        unit_name = {
            "intraday": "trading days",
            "short_term": "trading days",
            "long_term": "weeks",
            "multibagger": "months"  # ✅ Fix 9B.4: Multibagger uses months
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
        
        # ✅ Fix 9B.5: Cap t1_units by pattern expiry
        invalidation_timeline = pattern_meta.get("invalidation_timeline")
        age_candles = pattern_meta.get("age_candles", 0)
        
        original_t1_units = t1_units
        if invalidation_timeline:
            remaining = invalidation_timeline - age_candles
            remaining_units = max(1, int(remaining / bars_per_unit))
            t1_units = min(t1_units, remaining_units)
        
        t1_estimate = f"{t1_units} {unit_name}"
        if t1_units < original_t1_units:
             t1_estimate += " (pattern near expiry)"
        
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
    eval_ctx: Dict[str, Any], # Accept eval_ctx for trend access
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

    # --- Decision Logic with Trend-Based Relaxation ---
    trend_strength = _get_val(eval_ctx.get("indicators", {}), "trendStrength", 5.0)
    
    # Calculate effective min threshold
    effective_min_t1 = min_t1
    risk_mgmt_cfg = extractor.get_risk_management_config()
    min_rr_by_trend = risk_mgmt_cfg.get("min_rr_by_trend", {})
    if not min_rr_by_trend:
        # Emergency fallback if config is missing keys
        min_rr_by_trend = {
            "explosive": {"strength": 8.5, "min_rr": 1.0},
            "strong": {"strength": 6.5, "min_rr": 1.3}, 
            "normal": {"strength": 4.5, "min_rr": 1.5},
            "weak": {"strength": 0.0, "min_rr": 2.0}
        }

    for level in sorted(min_rr_by_trend.values(), key=lambda x: x["strength"], reverse=True):
        if trend_strength >= level["strength"]:
            effective_min_t1 = min(min_t1, level["min_rr"]) # Never harder than config
            break

    if rrt1 < execution_floor:
         return False, f"RR T1 ({rrt1:.2f}) below hard floor ({execution_floor})", None

    if rrt1 >= effective_min_t1:
        msg = f"T1 OK (Relaxed to {effective_min_t1:.1f} by trend)" if effective_min_t1 < min_t1 else None
        return True, msg, "T1"

    if rrt2 >= min_t2 and structural_rr >= min_structural:
        return True, "Using T2 target due to low T1 RR", "T2"

    return False, (
        f"Execution RR too low | "
        f"T1={rrt1:.2f} (<{min_t1}) | "
        f"T2={rrt2:.2f} (<{min_t2}) | "
        f"Structural={structural_rr:.2f} (<{min_structural})"
    ), None
