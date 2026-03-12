# services/signal_engine.py (REFACTORED v14.0)
"""
Signal Engine - Query Extractor Edition
========================================
✅ REFACTORED: Now uses QueryOptimizedExtractor for all config access

CHANGES FROM v13.0:
- All config access goes through extractor
- Extractor passed between functions (no recreation)
- Horizon pillar weights via extractor
- RR thresholds via extractor (execution params)
- Cleaner, more maintainable code

RESPONSIBILITIES (After Refactor):
1. Profile Scoring (VALUE, GROWTH, QUALITY, MOMENTUM)
2. Meta-Scoring (aggregate profile fitness)
3. Trade Plan Orchestration (delegates to resolver + extractor)

Author: Quantitative Trading System
Version: 14.0 - Query Extractor Integration
"""
from typing import Dict, Any, Optional, List
import traceback
import logging
from datetime import datetime

from config.constants import VALUE_WEIGHTS, GROWTH_WEIGHTS, QUALITY_WEIGHTS, MOMENTUM_WEIGHTS

from config.config_utility.logger_config import METRICS, track_performance, log_failures

# ============================================================================
# ✅ v14.0 IMPORTS - Query Extractor Architecture
# ============================================================================
from config.config_helpers import (
    # Resolver Factory
    get_resolver,
    build_evaluation_context_v5,
    build_execution_context_v5,
    
    # Context Accessors
    get_setup_from_context,
    get_confidence_from_context,
    check_gates_from_context,
    get_strategy_from_context
    
    # Execution Helpers
    # calculate_stop_loss_v5,
    # calculate_targets_v5,
    # get_position_size_from_context,
    
    # Pattern Extraction,
)

# Legacy Imports
from config.fundamental_score_config import extract_normalized_score
from services.patterns.pattern_velocity_tracking import classify_volatility

# Import pattern enhancer (post-processing)
from services.trade_enhancer import enhance_execution_context, validate_execution_rr
from services.summaries import build_enhanced_summaries



logger = logging.getLogger(__name__)


# ============================================================================
# 1. PROFILE SCORING (Domain Logic - Stays in Signal Engine)
# ============================================================================

def calculate_structural_eligibility(
    tech_score: float,
    fund_score: float,
    hybrid_score: float,
    horizon: str,
    extractor: Optional[Any] = None  # ✅ NEW: Accept extractor
) -> float:
    """
    ✅ REFACTORED: Now uses query extractor for horizon pillar weights.
    
    Step 6b: Blends the three structural pillars based on horizon DNA.
    
    Args:
        tech_score: Technical pillar score (0-10)
        fund_score: Fundamental pillar score (0-10)
        hybrid_score: Hybrid pillar score (0-10)
        horizon: Trading timeframe
        extractor: QueryOptimizedExtractor instance (optional)
    
    Returns:
        Weighted eligibility score (0-10)
    """
    try:
        # Get extractor if not provided
        if extractor is None:
            resolver = get_resolver(horizon)
            extractor = resolver.extractor
        
        # ✅ REFACTORED: Get weights via extractor
        weights = _get_horizon_pillar_weights(extractor)
        
        # Validate inputs
        if tech_score is None or fund_score is None or hybrid_score is None:
            logger.warning(
                f"[{horizon}] ⚠️ STRUCTURAL ELIGIBILITY: Missing pillar scores | "
                f"Tech={tech_score}, Fund={fund_score}, Hybrid={hybrid_score}"
            )
        
        # Calculate weighted eligibility
        eligibility_score = (
            (tech_score * weights['tech']) + 
            (fund_score * weights['fund']) + 
            (hybrid_score * weights['hybrid'])
        )
        
        logger.debug(
            f"[{horizon}] 📊 STRUCTURAL ELIGIBILITY: {eligibility_score:.2f} | "
            f"Components: T={tech_score:.1f}({weights['tech']:.2f}) + "
            f"F={fund_score:.1f}({weights['fund']:.2f}) + "
            f"H={hybrid_score:.1f}({weights['hybrid']:.2f})"
        )
        
        return round(eligibility_score, 2)
        
    except Exception as e:
        logger.error(
            f"[{horizon}] ❌ STRUCTURAL ELIGIBILITY FAILED | "
            f"Tech={tech_score}, Fund={fund_score}, Hybrid={hybrid_score} | "
            f"Error: {type(e).__name__}: {e}",
            exc_info=True
        )
        return 0.0


@log_failures(return_on_error={}, critical=False)
def compute_opportunity_score(
    ticker: str,
    horizon: str,
    eligibility_base_score: float,
    indicators: Dict,
    fundamentals: Dict,
    eval_ctx: Dict = None,
    extractor: Optional[Any] = None  # ✅ NEW: Accept extractor
) -> Dict[str, Any]:
    """
    ✅ REFACTORED: Now accepts and reuses extractor instance.
    
    Opportunity scoring - layers real-time technical setups on top of 
    Structural Eligibility.
    
    Args:
        ticker: Stock symbol
        horizon: Trading timeframe
        eligibility_base_score: Base eligibility from structural pillars
        indicators: Technical indicators
        fundamentals: Fundamental data
        eval_ctx: Optional pre-built evaluation context
        extractor: QueryOptimizedExtractor instance (optional)
    
    Returns:
        Opportunity scoring results with trade context
    # NOTE: Confidence is calculated by resolver's _calculate_confidence()
#       following the CONFIDENCE_CALCULATION_PIPELINE:
#       1. Setup baseline floor (40-60 depending on setup)
#       2. Horizon base adjustment (-10 for intraday, 0 for long_term)
#       3. ADX confidence bands
#       4. Volume modifiers (surge/drought/climax)
#       5. Universal adjustments (divergence, trend strength)
#       6. Horizon conditional adjustments (penalties/bonuses)
#       7. Execution rules
#       8. Clamp to horizon range
#
#       See: confidence_config.py for full configuration
    """
    start_time = datetime.now().timestamp()
    
    try:
        logger.info(
            f"[{ticker}][{horizon}] 🎯 OPPORTUNITY SCORING START | "
            f"Base Eligibility: {eligibility_base_score:.2f}"
        )
        
        # Get extractor if not provided
        if extractor is None:
            resolver = get_resolver(horizon)
            extractor = resolver.extractor
        
        # 1. Build Evaluation Context (if not cached)
        if eval_ctx is None:
            ctx_start = datetime.now().timestamp()
            eval_ctx = build_evaluation_context_v5(
                ticker=ticker,
                indicators=indicators,
                fundamentals=fundamentals,
                horizon=horizon
            )
            ctx_elapsed = datetime.now().timestamp() - ctx_start
            logger.debug(
                f"[{ticker}][{horizon}] ⏱️ Context built in {ctx_elapsed*1000:.1f}ms"
            )
        else:
            logger.debug(f"[{ticker}][{horizon}] ♻️ Using cached eval_ctx")

        # 2. Extract Setup & Confidence
        setup_type, priority, setup_meta = get_setup_from_context(eval_ctx)
        confidence, conf_meta = get_confidence_from_context(eval_ctx)
        strategy = get_strategy_from_context(eval_ctx)
        
        logger.debug(
            f"[{ticker}][{horizon}] 📋 EXTRACTED CONTEXT | "
            f"Setup={setup_type}(priority={priority}) | "
            f"Strategy={strategy['primary_strategy']}(fit={strategy['fit_score']:.1f}) | "
            f"Confidence={confidence:.1f}%"
        )

        # 3. Validate Entry Gates
        gate_result = check_gates_from_context(eval_ctx, confidence)
        if not gate_result["passed"]:
            logger.warning(
                f"[{ticker}][{horizon}] ⛔ GATES FAILED | "
                f"Failures: {gate_result.get('failed_gates', [])} | "
                f"Summary: {gate_result.get('summary')}"
            )
        else:
            logger.debug(f"[{ticker}][{horizon}] ✅ All gates passed")
        
        # 4. Calculate Opportunity Bonus (0-5 scale)
        opp_bonus = 0.0
        bonus_breakdown = {}
        
        if gate_result["passed"]:
            # Bonus A: Pattern/Setup priority
            priority_bonus = priority / 100.0
            opp_bonus += priority_bonus
            bonus_breakdown["priority"] = priority_bonus
            
            # Bonus B: Strategy Fit
            fit_bonus = strategy["fit_score"] / 100.0
            opp_bonus += fit_bonus
            bonus_breakdown["strategy_fit"] = fit_bonus
            
            # Bonus C: Conviction (Excess confidence over 50 floor)
            setup_type = eval_ctx["setup"]["type"]
            baseline_floor = extractor.get_setup_baseline_floor(setup_type)
            conviction_bonus = max(0, (confidence - baseline_floor) / 50.0)
            opp_bonus += conviction_bonus
            bonus_breakdown["conviction"] = conviction_bonus
            bonus_breakdown["baseline_floor"] = baseline_floor  # Track for debugging
            
            logger.debug(
                f"[{ticker}][{horizon}] 💎 OPPORTUNITY BONUS: {opp_bonus:.2f} | "
                f"Breakdown: Priority={priority_bonus:.2f} + "
                f"Fit={fit_bonus:.2f} + Conviction={conviction_bonus:.2f}"
            )
        else:
            logger.debug(f"[{ticker}][{horizon}] ⚠️ No bonus (gates failed)")

        # 5. Final Decision Score
        # Mathematical Refactor: Weighting instead of raw addition (Score > 10 fix)
        # Eligibility is out of 10. Bonus translates up to ~3.0.
        # Rule: 70% Base Eligibility + 30% Normalized Bonus Score.
        
        # Normalize bonus to a 10-point scale (Max practical bonus is ~3.0)
        max_practical_bonus = 3.0
        normalized_bonus = min(10.0, (opp_bonus / max_practical_bonus) * 10.0)
        
        final_score = (eligibility_base_score * 0.70) + (normalized_bonus * 0.30)
        
        # Absolute safety ceiling (though math theoretically prevents it)
        final_score = min(10.0, final_score)
        
        elapsed = datetime.now().timestamp() - start_time
        
        logger.info(
            f"[{ticker}][{horizon}] ✅ OPPORTUNITY SCORING COMPLETE | "
            f"Final={final_score:.2f} (Base[70%]={eligibility_base_score:.2f} + "
            f"Bonus[30%]={normalized_bonus:.2f}) | "
            f"Time={elapsed*1000:.1f}ms"
        )
        
        METRICS.log_performance("compute_opportunity_score", elapsed)

        return {
            "horizon": horizon,
            "final_decision_score": round(final_score, 2),
            "eligibility_base": eligibility_base_score,
            "opportunity_bonus": round(opp_bonus, 2),
            "trade_context": {
                "setup": setup_type,
                "strategy": strategy["primary_strategy"],
                "confidence": confidence,
                "patterns": setup_meta.get("patterns_detected", []),
                "gate_passed": gate_result["passed"],
                "block_reason": gate_result.get("summary")
            },
            "eval_ctx": eval_ctx
        }
        
    except Exception as e:
        logger.error(
            f"[{ticker}][{horizon}] ❌ OPPORTUNITY SCORING FAILED | "
            f"Base Score={eligibility_base_score:.2f} | "
            f"Error: {type(e).__name__}: {e}",
            exc_info=True
        )
        return {
            "horizon": horizon,
            "final_decision_score": eligibility_base_score,
            "eligibility_base": eligibility_base_score,
            "opportunity_bonus": 0.0,
            "trade_context": {
                "setup": "GENERIC",
                "strategy": "generic",
                "confidence": 50,
                "patterns": [],
                "gate_passed": False,
                "block_reason": f"Error: {str(e)}"
            },
            "eval_ctx": None,
            "error": str(e)
        }

def _create_fallback_profile(
    horizon: str,
    error_reason: str,
    eligibility_base: float = 0.0
) -> Dict[str, Any]:
    """
    ✅ NEW: Create fallback profile for failed horizon processing.
    
    Ensures compute_all_profiles always returns profiles dict with entries
    for all attempted horizons, even if processing failed.
    
    Args:
        horizon: Trading timeframe that failed
        error_reason: Human-readable failure description
        eligibility_base: Partial eligibility score if calculated
    
    Returns:
        Minimal valid profile structure
    """
    return {
        "structural_eligibility": {
            "score": eligibility_base,
            "components": {
                "technical": 0,
                "fundamental": 0,
                "hybrid": 0
            },
            "weights": {"tech": 0.33, "fund": 0.33, "hybrid": 0.34},
            "hybrid_breakdown": {}
        },
        "opportunity": {
            "bonus": 0.0,
            "trade_context": {
                "setup": "GENERIC",
                "strategy": "generic",
                "confidence": 50,
                "patterns": [],
                "gate_passed": False,
                "block_reason": error_reason
            }
        },
        "final_decision_score": eligibility_base,
        "final_score": eligibility_base,
        "category": "HOLD",
        "can_trade": False,
        "eval_ctx": None,
        "metric_details": [],
        "applied_penalties": [],
        "architecture": "v14.1-fallback",
        "timestamp": datetime.now().timestamp(),
        "status": "FAILED",
        "error": error_reason
    }

@log_failures(return_on_error={}, critical=False)
def compute_all_profiles(
    ticker: str,
    fundamentals: Dict,
    indicators_by_horizon: Dict,
    patterns_by_horizon: Dict = None,
    requested_horizons: List[str] = None  # ✅ NEW: Filter which horizons to score
) -> Dict:
    """
    ✅ REFACTORED: Optionally filters which horizons to compute.
    
    Args:
        ticker: Stock symbol
        fundamentals: Fundamental data
        indicators_by_horizon: Dict of horizon → indicators
        hybrid_pillar_multi_horizon: Unused legacy param
        requested_horizons: Optional list of horizons to score
                           If None, scores all available in indicators_by_horizon
                           If ["multibagger"], only scores multibagger
    
    Performance:
        - Full mode: 4 horizons × 200ms = 800ms
        - Single mode: 1 horizon × 200ms = 200ms (4x faster!)
    """
    start_time = datetime.now().timestamp()
    
    # FILTER: Only score requested horizons
    if requested_horizons:
        horizons_to_score = {
            h: inds 
            for h, inds in indicators_by_horizon.items() 
            if h in requested_horizons
        }
        logger.info(
            f"[{ticker}] 🎯 TARGETED PROFILE SCORING | "
            f"Requested: {requested_horizons} | "
            f"Available: {list(indicators_by_horizon.keys())}"
        )
    else:
        horizons_to_score = indicators_by_horizon
        logger.info( f"[{ticker}] 🔄 FULL PROFILE SCORING | " f"Horizons: {list(indicators_by_horizon.keys())}")
    
    profiles = {}
    failed_horizons = []
    
    try:
        # Iterate only filtered horizons
        for horizon, indicators in horizons_to_score.items():
            horizon_start = datetime.now().timestamp()
            try:
                resolver = get_resolver(horizon)
                extractor = resolver.extractor
                # --------------------------------------------------
                # 1️⃣ BUILD EVALUATION CONTEXT (RESILIENT)
                # --------------------------------------------------
                try:
                    with track_performance(f"build_eval_ctx_{horizon}"):
                        patterns = (patterns_by_horizon or {}).get(horizon, {})
                        eval_ctx = build_evaluation_context_v5( ticker, indicators, fundamentals, horizon, patterns=patterns)
                    if not eval_ctx:
                        eval_ctx = {}

                    # normalize instead of abort
                    eval_ctx.setdefault("scoring", {})
                    eval_ctx.setdefault("setup", {})

                except Exception as e:
                    logger.error(
                        f"[{ticker}][{horizon}] ❌ EVAL_CTX BUILD FAILED: {e}", exc_info=True
                    )
                    failed_horizons.append({"horizon": horizon, "stage": "eval_ctx", "error": str(e)})
                    profiles[horizon] = _create_fallback_profile(horizon, f"Eval context failed: {e}")
                    continue
                # --------------------------------------------------
                # 2️⃣ EXTRACT SCORES (DEGRADABLE)
                # --------------------------------------------------
                scoring = eval_ctx.get("scoring", {})

                technical = scoring.get("technical", {})
                fundamental = scoring.get("fundamental", {})
                hybrid = scoring.get("hybrid", {})

                tech_score = technical.get("score", 0.0)
                fund_score = fundamental.get("score", 0.0)
                hybrid_score = hybrid.get("score", 0.0)

                tech_penalties = technical.get("penalties", [])

                logger.debug(
                    f"[{ticker}][{horizon}] 📈 PILLAR SCORES | "
                    f"Technical={tech_score:.1f} | "
                    f"Fundamental={fund_score:.1f} | "
                    f"Hybrid={hybrid_score:.1f}"
                )

                # --------------------------------------------------
                # 3️⃣ STRUCTURAL ELIGIBILITY
                # --------------------------------------------------
                try:
                    eligibility_score = calculate_structural_eligibility(
                        tech_score,
                        fund_score,
                        hybrid_score,
                        horizon,
                        extractor=extractor,
                    )
                except Exception as e:
                    logger.error(
                        f"[{ticker}][{horizon}] ❌ ELIGIBILITY CALC FAILED: {e}",
                        exc_info=True
                    )
                    failed_horizons.append(
                        {"horizon": horizon, "stage": "eligibility", "error": str(e)}
                    )

                    profiles[horizon] = _create_fallback_profile(
                        horizon, f"Eligibility calc failed: {e}"
                    )
                    continue

                # --------------------------------------------------
                # 4️⃣ OPPORTUNITY SCORING
                # --------------------------------------------------
                try:
                    opportunity = compute_opportunity_score(
                        ticker,
                        horizon,
                        eligibility_score,
                        indicators,
                        fundamentals,
                        eval_ctx,
                        extractor=extractor,
                    )
                except Exception as e:
                    logger.error(
                        f"[{ticker}][{horizon}] ❌ OPPORTUNITY SCORING FAILED: {e}",
                        exc_info=True
                    )
                    failed_horizons.append(
                        {"horizon": horizon, "stage": "opportunity", "error": str(e)}
                    )

                    profiles[horizon] = _create_fallback_profile(
                        horizon,
                        f"Opportunity scoring failed: {e}",
                        eligibility_base=eligibility_score,
                    )
                    continue

                # --------------------------------------------------
                # 5️⃣ FINAL ASSEMBLY
                # --------------------------------------------------
                final_score = opportunity["final_decision_score"]

                # ── PROFILE SIGNAL: Pure score-based stock quality rating ──────
                # Independent of patterns, gates, RR, or execution context.
                # Answers: "Is this stock worth tracking/accumulating?"
                if final_score >= 8.5:
                    profile_signal = "STRONG"
                elif final_score >= 7.0:
                    profile_signal = "MODERATE"
                elif final_score >= 5.5:
                    profile_signal = "WEAK"
                else:
                    profile_signal = "AVOID"

                # Legacy compat: map profile_signal → category
                _category_legacy_map = {
                    "STRONG": "STRONG",
                    "MODERATE": "WATCH",
                    "WEAK": "HOLD",
                    "AVOID": "AVOID"
                }
                category = _category_legacy_map[profile_signal]

                # Get weights via extractor
                weights = _get_horizon_pillar_weights(extractor)

                profiles[horizon] = {
                    "structural_eligibility": {
                        "score": eligibility_score,
                        "components": {
                            "technical": tech_score,
                            "fundamental": fund_score,
                            "hybrid": hybrid_score,
                        },
                        "weights": weights,
                        "hybrid_breakdown": hybrid.get("breakdown"),
                    },
                    "opportunity": {
                        "bonus": opportunity.get("opportunity_bonus"),
                        "trade_context": opportunity.get("trade_context"),
                    },
                    "final_decision_score": final_score,
                    "final_score": final_score,
                    "profile_signal": profile_signal,
                    "category": category,
                    "can_trade": False,  # Always False at profile level; set by execution plan
                    "strategy": eval_ctx.get("strategy", {}),
                    "eval_ctx": eval_ctx,
                    "metric_details": scoring.get("metric_details", []),
                    "applied_penalties": tech_penalties,
                    "architecture": "v15.0-signal-separation",
                    "timestamp": datetime.now().timestamp(),
                    "status": "SUCCESS",
                }

                horizon_elapsed = datetime.now().timestamp() - horizon_start

                logger.info(
                    f"[{ticker}][{horizon}] ✅ PROFILE COMPLETE | "
                    f"Eligibility: {eligibility_score:.2f} | "
                    f"Final: {final_score:.2f} | "
                    f"Can Trade: {'✅' if profiles[horizon]['can_trade'] else '❌'} | "
                    f"Time: {horizon_elapsed*1000:.1f}ms"
                )

                METRICS.log_performance(f"profile_{horizon}", horizon_elapsed)

            except Exception as e:
                logger.error(
                    f"[{ticker}][{horizon}] ❌ HORIZON PROCESSING FAILED: {e}",
                    exc_info=True,
                )
                failed_horizons.append(
                    {"horizon": horizon, "stage": "unknown", "error": str(e)}
                )

                profiles[horizon] = _create_fallback_profile(
                    horizon, f"Unexpected error: {e}"
                )

        # --------------------------------------------------
        # 6️⃣ BEST HORIZON SELECTION
        # --------------------------------------------------
        successful_profiles = {
            h: p for h, p in profiles.items() if p.get("status") == "SUCCESS"
        }

        if successful_profiles:
            best_horizon = max(
                successful_profiles,
                key=lambda h: successful_profiles[h]["final_decision_score"],
            )
            best_score = successful_profiles[best_horizon]["final_decision_score"]
        else:
            best_horizon = None
            best_score = 0

        total_elapsed = datetime.now().timestamp() - start_time

        logger.info(
            f"[{ticker}] 🏆 PROFILE COMPUTATION COMPLETE | "
            f"Best: {best_horizon} (score={best_score:.2f}) | "
            f"Computed: {len(profiles)} horizons | "
            f"Time: {total_elapsed*1000:.1f}ms"
        )
        
        return {
            "ticker": ticker,
            "best_fit": best_horizon,
            "best_score": best_score,
            "profiles": profiles,
            "successful_count": len(successful_profiles),
            "failed_horizons": failed_horizons,
            "note": "Structural eligibility (3 pillars) + Opportunity timing",
        }

    except Exception as e:
        logger.error(f"[{ticker}] ❌ CATASTROPHIC FAILURE: {e}")
        return {
            "ticker": ticker,
            "best_fit": None,
            "best_score": 0,
            "profiles": profiles,
            "successful_count": 0,
            "failed_horizons": [
                {"horizon": "ALL", "stage": "catastrophic", "error": str(e)}
            ],
            "error": str(e),
            "note": "Catastrophic failure - see logs",
            "metric_details": []
        }
    
# ============================================================================
# 2. TRADE PLAN GENERATION (Clean Orchestrator with Extractor)
# ============================================================================
def finalize_trade_decision(plan: dict, eval_ctx: dict, exec_ctx: dict, extractor=None) -> None:
    """
    SETUP SIGNAL: Execution-based trade decision.

    Answers: "Can I trade this stock RIGHT NOW, given current market structure?"

    Signal vocabulary:
      BUY     — Valid entry. Setup matched. Pattern confirmed. RR passes.
      SELL    — Valid short. (bearish setups only)
      WATCH   — Strong stock but NO entry structure right now.
      HOLD    — Active position management signal. (not entry)
      BLOCKED — Setup + pattern present but execution gate failed.

    Four layers, evaluated in strict order. No layer can override a higher layer.
    """
    logger = logging.getLogger(__name__)

    # ── Extract core inputs ──────────────────────────────────────────────
    confidence = plan.get("final_confidence", 0)
    direction  = plan.get("metadata", {}).get("direction", "neutral")

    if extractor:
        try:
            min_tradeable = extractor.get_min_tradeable_confidence()
        except Exception:
            min_tradeable = 60
        try:
            hco = extractor.get_high_confidence_override()
            high_conf_threshold = hco.get("threshold", 80) if isinstance(hco, dict) else float(hco)
        except Exception:
            high_conf_threshold = 80
    else:
        min_tradeable = 60
        high_conf_threshold = 80

    can_execute     = exec_ctx.get("can_execute", {})
    execution_blocked = not can_execute.get("can_execute", True)
    failures        = can_execute.get("failures", [])

    market          = exec_ctx.get("market_adjusted_targets", {})
    rr_t1           = market.get("execution_rr_t1", 0)
    rr_t2           = market.get("execution_rr_t2", 0)
    rr_source       = market.get("rr_source", "") or eval_ctx.get("risk_candidates", {}).get("rr_source", "")
    target_source   = market.get("target_source", "") or market.get("source", "")

    setup_type      = eval_ctx.get("setup", {}).get("type", "GENERIC")

    # ── Setup intent classification ──────────────────────────────────────
    if "BREAKOUT" in setup_type or "MOMENTUM" in setup_type:
        signal_intent = "BREAKOUT"
    elif "TREND" in setup_type or "PULLBACK" in setup_type:
        signal_intent = "TREND_FOLLOWING"
    elif "ACCUMULATION" in setup_type or "VALUE" in setup_type:
        signal_intent = "ACCUMULATION"
    else:
        signal_intent = "GENERIC"

    # ── Resolve primary pattern presence ─────────────────────────────────
    primary_found = bool(
        eval_ctx.get("pattern_validation", {})
        .get("by_setup", {})
        .get(setup_type, {})
        .get("primary_found", [])
    )

    # ── ATR fallback detection ───────────────────────────────────────────
    is_atr_fallback = (
        "atr_fallback" in str(rr_source) or
        "atr_fallback" in str(target_source) or
        rr_source == "generic_atr"
    )

    # ════════════════════════════════════════════════════════════════════
    # LAYER 0: STRUCTURE GATE  (highest priority — no overrides)
    # "Does a tradeable entry structure exist?"
    # ════════════════════════════════════════════════════════════════════
    if setup_type == "GENERIC" and not primary_found:
        plan.update({
            "trade_signal":      "WATCH",
            "setup_signal":      "WATCH",
            "signal":            signal_intent,
            "status":            "NO_PATTERN",
            "reason":            (
                f"No primary pattern detected for {setup_type} setup. "
                f"Strong profile but no active entry structure. "
                f"Monitor for pattern formation."
            ),
            "execution_blocked": True,
            "can_trade":         False,
        })
        plan.setdefault("block_gates", []).append(
            "STRUCTURE_GATE: GENERIC fallback with no primary pattern cannot generate BUY"
        )
        logger.info(
            f"[{plan.get('symbol')}] \U0001F441  WATCH — No pattern. "
            f"Setup={setup_type}, Confidence={confidence}%."
        )
        plan["signal_context"] = {
            "direction":        direction,
            "confidence":       confidence,
            "setup_type":       setup_type,
            "primary_found":    primary_found,
            "signal_intent":    signal_intent,
            "rr_t1":            rr_t1,
            "rr_t2":            rr_t2,
            "execution_status": "no_pattern",
        }
        return

    # ════════════════════════════════════════════════════════════════════
    # LAYER 1: EXECUTION GATE
    # "Setup exists, pattern exists, but execution is gated"
    # ════════════════════════════════════════════════════════════════════
    if execution_blocked:
        base_signal = "BLOCKED"
        base_status = "BLOCKED"
    elif confidence < min_tradeable:
        base_signal = "HOLD"
        base_status = "LOW_CONFIDENCE"
    else:
        if direction == "bullish":
            base_signal = "BUY"
            base_status = "READY"
        elif direction == "bearish":
            base_signal = "SELL"
            base_status = "READY"
        else:
            base_signal = "HOLD"
            base_status = "NEUTRAL"

    # ════════════════════════════════════════════════════════════════════
    # LAYER 2: STRUCTURAL RESCUES  (only from structural targets, never ATR)
    # ════════════════════════════════════════════════════════════════════
    final_signal = base_signal
    final_status = base_status
    reason = f"Decision based on {direction} direction"

    is_rr_failure = any("RR" in f for f in failures)

    if base_status == "BLOCKED" and is_rr_failure and not is_atr_fallback:
        # Rescue A: High confidence + near-passing T1
        if confidence >= high_conf_threshold and rr_t1 >= 1.2:
            final_signal = "BUY" if direction == "bullish" else base_signal
            final_status = "READY"
            reason = (
                f"High confidence override: {confidence}% >= {high_conf_threshold}% "
                f"with structural T1 RR {rr_t1:.2f}"
            )
            plan.setdefault("boost_reasons", []).append({
                "reason": "High confidence structural override",
                "source": rr_source
            })
            logger.info(
                f"[{plan.get('symbol')}] \u2705 High-conf rescue: "
                f"conf={confidence}%, RR_T1={rr_t1:.2f}"
            )

        # Rescue B: Structural T2 asymmetry
        elif rr_t2 >= 2.5:
            final_signal = "BUY" if direction == "bullish" else base_signal
            final_status = "READY"
            reason = f"Structural T2 asymmetry override (RR {rr_t2:.2f}, source={rr_source})"
            plan.setdefault("boost_reasons", []).append({
                "reason": f"T2 asymmetry override (structural, RR={rr_t2:.2f})",
                "source": rr_source
            })
            logger.info(
                f"[{plan.get('symbol')}] \u2705 T2 structural rescue: "
                f"RR_T2={rr_t2:.2f} from {rr_source}"
            )

    elif base_status == "BLOCKED" and is_rr_failure and is_atr_fallback:
        # ATR fallback targets — no rescue allowed
        final_signal = "BLOCKED"
        final_status = "BLOCKED"
        reason = (
            f"RR gate failed (T1={rr_t1:.2f}) and targets are ATR-based — "
            f"no structural level to justify rescue."
        )
        logger.debug(
            f"[{plan.get('symbol')}] \u26d4 ATR rescue suppressed: "
            f"RR_T1={rr_t1:.2f}, RR_T2={rr_t2:.2f} (source={rr_source})"
        )

    # ════════════════════════════════════════════════════════════════════
    # LAYER 3: FINALIZE
    # ════════════════════════════════════════════════════════════════════
    plan.update({
        "trade_signal":  final_signal,
        "setup_signal":  final_signal,
        "signal":        signal_intent,
        "status":        final_status,
        "reason":        reason,
    })

    plan["signal_context"] = {
        "direction":        direction,
        "confidence":       confidence,
        "setup_type":       setup_type,
        "primary_found":    primary_found,
        "signal_intent":    signal_intent,
        "rr_t1":            rr_t1,
        "rr_t2":            rr_t2,
        "rr_source":        rr_source,
        "is_atr_fallback":  is_atr_fallback,
        "execution_status": final_status.lower(),
    }

def apply_rr_validation(exec_ctx: Dict[str, Any], extractor) -> None:
    """
    Final RR gate after enhancer.
    Mutates exec_ctx["can_execute"].
    """ 
    can_exec = exec_ctx.get("can_execute") or {"can_execute": True, "failures": [], "checks": {}}
    
    # ✅ Pass self.extractor explicitly
    ok, reason, recommended = validate_execution_rr(exec_ctx, extractor)
    
    if not ok:
        can_exec["can_execute"] = False
        can_exec["failures"].append(reason or "Execution RR check failed")
    else:
        can_exec.setdefault("checks", {})["rr"] = {
            "passed": True,
            "recommended_target": recommended,
            "reason": reason,
        }
    exec_ctx["can_execute"] = can_exec

def generate_trade_plan(
    symbol: str,
    winner_profile: Dict = None,
    indicators: Dict = {},
    fundamentals: Dict = {},
    horizon: str = "short_term",
    macro_trend_status: str = "N/A",
    capital: float = 100000
) -> Dict[str, Any]:

    METRICS.set_current_symbol(symbol)
    resolver = get_resolver(horizon)
    extractor = resolver.extractor

    plan = {
        "symbol": symbol,
        "horizon": horizon,
        "status": "PENDING",
        "trade_signal": "HOLD",
        "signal": "NA_CALC",
        "reason": "Initializing...",
        
        # ============================================
        # SETUP & SCORES (✅ ALL NEW)
        # ============================================
        "setup_type": None,  # Will populate from eval_ctx
        "profile_score": winner_profile.get("final_score", 0.0) if winner_profile else 0.0,
        
        # ============================================
        # CONFIDENCE (✅ ENHANCED)
        # ============================================
        "base_confidence": 0,  # Will populate
        "final_confidence": 0,  # Will populate
        "setup_confidence": 0,  # Will populate (same as base)
        "adjusted_confidence": 0,  # Will populate (pre-clamp)
        "confidence_history": [],
        
        # ============================================
        # EXECUTION VALUES
        # ============================================
        "entry": None,
        "stop_loss": None,
        "targets": {"t1": None, "t2": None},
        "position_size": 0,
        "rr_ratio": 0,
        
        # ============================================
        # AUDIT TRAIL
        # ============================================
        "penalties_applied": [],
        "boost_reasons": [],
        
        # ============================================
        # GATE STATUS (✅ ALL NEW)
        # ============================================
        "gates_passed": True,  # Will populate
        "block_reason": None,  # Will populate
        "block_gates": [],  # Will populate
        
        # ============================================
        # METADATA & ANALYTICS (✅ ENHANCED)
        # ============================================
        "metadata": {"macro_trend": macro_trend_status},
        "metric_details": {},
        "execution_hints": {},  # ✅ NEW
        "analytics": {},  # ✅ NEW
        "debug": {},  # ✅ NEW
        
        # ============================================
        # PRESENTATION
        # ============================================
        "est_time_str": "NA",
        "narratives": {},
        
        # ============================================
        # EXECUTION FLAGS
        # ============================================
        "execution_blocked": False,
        "can_trade": True
    }

    try:
        # ======================================================
        # STAGE 1: Evaluation Context
        # ======================================================
        if winner_profile and "eval_ctx" in winner_profile:
            eval_ctx = winner_profile["eval_ctx"]
        else:
            eval_ctx = build_evaluation_context_v5(
                symbol, indicators, fundamentals, horizon
            )

        # ✅ POPULATE SETUP & CONFIDENCE FIELDS IMMEDIATELY
        confidence_data = eval_ctx["confidence"]
        plan["setup_type"] = eval_ctx["setup"]["type"]
        plan["base_confidence"] = confidence_data["base"]
        plan["final_confidence"] = confidence_data["clamped"]
        plan["setup_confidence"] = confidence_data["base"]  # Same as base
        plan["adjusted_confidence"] = confidence_data.get("final", confidence_data["clamped"])

        # ✅ NEW: Add detailed confidence breakdown for debugging
        plan["confidence_breakdown"] = {
            "base_floor": confidence_data.get("base", 0),
            "horizon_adjustment": confidence_data.get("horizon_adjustment", 0),
            "total_adjustments": confidence_data.get("adjustments", {}).get("total", 0),
            "divergence_multiplier": confidence_data.get("divergence_multiplier", 1.0),
            "final_unclamped": confidence_data.get("final", 0),
            "clamped": confidence_data.get("clamped", 0),
            "clamp_range": confidence_data.get("clamp_range", [30, 95]),
            "calculation_method": confidence_data.get("calculation_method", "unknown")
        }

        # ======================================================
        # STAGE 2: Execution Context
        # ======================================================
        exec_ctx = build_execution_context_v5(eval_ctx, capital)
        exec_ctx = enhance_execution_context(
            eval_ctx, exec_ctx, indicators, symbol, horizon,
            extractor=extractor
        )

        apply_rr_validation(exec_ctx, extractor=extractor)  # Mutates exec_ctx["can_execute"]

        # ✅ POPULATE ESTIMATED TIME
        if "timeline" in exec_ctx and exec_ctx["timeline"].get("available"):
            plan["est_time_str"] = exec_ctx["timeline"].get("t1_estimate", "NA")
        else:
            plan["est_time_str"] = "NA"


        # ✅ POPULATE GATE STATUS
        can_execute = exec_ctx.get("can_execute", {})
        execution_blocked = not can_execute.get("can_execute", True)
        
        plan["gates_passed"] = not execution_blocked
        plan["execution_blocked"] = execution_blocked
        plan["can_trade"] = not execution_blocked
        
        if execution_blocked:
            failures = can_execute.get("failures", ["Execution blocked"])
            plan["block_reason"] = failures[0]  # ✅ PRIMARY BLOCK REASON
            plan["block_gates"] = failures  # ✅ ALL FAILED GATES
            
            plan.update({
                "status": "BLOCKED",
                "reason": failures[0],
                "execution_failures": failures,
            })
            
            logger.warning(f"[{symbol}] ⚠️ Execution blocked: {failures[0]}")
        else:
            plan.update({
                "execution_blocked": False,
                "can_trade": True
            })
        # ======================================================
        # STAGE 3: Risk & Order (FINALIZED VALUES)
        # ======================================================
        risk = exec_ctx["risk"]
        market_adjusted = exec_ctx.get("market_adjusted_targets", {})
        
        # ✅ FIX Issue 6: Set direction from eval_ctx BEFORE branches
        # so finalize_trade_decision always has a valid direction
        trend_info = eval_ctx.get("trend", {})
        trend_classification = trend_info.get("classification", {})
        plan["metadata"]["direction"] = trend_classification.get("direction", "neutral")
        
        if market_adjusted.get("adjusted"):
            plan["entry"] = market_adjusted["execution_entry"]
            plan["stop_loss"] = market_adjusted["execution_sl"]
            plan["targets"]["t1"] = market_adjusted["execution_t1"]
            plan["targets"]["t2"] = market_adjusted["execution_t2"]
            plan["rr_ratio"] = market_adjusted["execution_rr_t2"] or market_adjusted["execution_rr_t1"]
            plan["position_size"] = risk.get("quantity", 0) 
            
            plan["metadata"]["pattern_rr"] = market_adjusted["structural_rr"]
            plan["metadata"]["execution_rr_t1"] = market_adjusted["execution_rr_t1"]
            plan["metadata"]["execution_rr_t2"] = market_adjusted["execution_rr_t2"]
            plan["metadata"]["volatility_regime"] = market_adjusted["volatility_regime"]
            plan["metadata"]["direction"] = market_adjusted["direction"]
            plan["metadata"]["rr_source"] = risk.get("rr_source")  
            plan["metadata"]["atr_multiple"] = risk.get("atr_multiple")  
            plan["metadata"]["market_adjusted"] = True
        else:
            if risk is None:
                raise RuntimeError("exec_ctx missing 'risk' section")
            else:
                plan["entry"] = risk["entry_price"]
                plan["stop_loss"] = risk["stop_loss"]
                targets = risk.get("targets", [])
                plan["targets"]["t1"] = targets[0] if targets else 0
                plan["targets"]["t2"] = targets[1] if len(targets) > 1 else 0
                plan["rr_ratio"] = risk.get("rrRatio", 0)
                plan["position_size"] = risk.get("quantity", 0)
                plan["metadata"]["market_adjusted"] = False
                plan["metadata"]["direction"] = risk.get("direction", exec_ctx.get("direction", "neutral"))

        # Common metadata (regardless of adjustment)
        plan["metadata"]["order_model"] = exec_ctx.get("order_model") 

        # ======================================================
        # STAGE 4: Macro Adjustment (PRESENTATION ONLY)
        # ======================================================
        if macro_trend_status and "downtrend" in macro_trend_status.lower():
            original_qty = plan["position_size"]
            plan["position_size"] = int(original_qty * 0.7)
            plan["boost_reasons"].append({
                "reason": "Macro downtrend",
                "change": "-30% position"
            })
            plan["metadata"]["original_position_size"] = original_qty

        # ======================================================
        # STAGE 5: Confidence & Audit Trail
        # ======================================================
        plan["confidence_history"] = [
            {
                "step": "base_floor", 
                "value": eval_ctx["confidence"]["base"],
                "source": "setup_baseline"
            },
            {
                "step": "after_adjustments",
                "value": eval_ctx["confidence"].get("final", eval_ctx["confidence"]["clamped"]),
                "source": "universal_modifiers"
            },
            {
                "step": "final_clamped", 
                "value": eval_ctx["confidence"]["clamped"],
                "source": "clamp_range_applied"
            }
        ]

        # Parse penalties and boosts
        adjustments = eval_ctx["confidence"].get("adjustments", {})
        breakdown = adjustments.get("breakdown", [])

        plan["penalties_applied"] = []
        plan["boost_reasons"] = plan.get("boost_reasons", [])

        for entry in breakdown:
            text = str(entry).lower()
            if any(x in text for x in ["-", "penalty", "warning", "violation", "×0."]):
                plan["penalties_applied"].append({"reason": entry})
            elif any(x in text for x in ["+", "bonus", "boost"]):
                plan["boost_reasons"].append({"reason": entry})

        plan["metric_details"] = eval_ctx["scoring"]["metric_details"]

        # ✅ POPULATE EXECUTION HINTS (NEW)
        entry_permission = exec_ctx.get("entry_permission", {})
        plan["execution_hints"] = {
            "checks": entry_permission.get("checks", {}),
            "warnings": entry_permission.get("warnings", []),
            "confidence": entry_permission.get("confidence", 0),
            "strategy": entry_permission.get("strategy", "unknown"),
            "pattern_status": entry_permission.get("pattern_status", {})
        }

        # ✅ POPULATE ANALYTICS (NEW)
        plan["analytics"] = {
            "strategy_fit": eval_ctx["strategy"]["fit_score"],
            "strategy_weighted": eval_ctx["strategy"]["weighted_score"],
            "pattern_count": len(eval_ctx.get("patterns", {})),
            "hybrid_score": eval_ctx["scoring"]["hybrid"]["score"],
            "technical_score": eval_ctx["scoring"]["technical"]["score"],
            "fundamental_score": eval_ctx["scoring"]["fundamental"]["score"],
            "setup_fit": eval_ctx["setup"]["best"]["fit_score"],
            "setup_composite": eval_ctx["setup"]["best"]["composite_score"],
            "rr_regime": exec_ctx.get("rr_regime", {}).get("regime", "unknown"),
            "volatility_regime": exec_ctx.get("market_adjusted_targets", {}).get("volatility_regime", "unknown")
        }

        # ✅ POPULATE DEBUG INFO (NEW - Optional)
        if logger.level == logging.DEBUG:
            plan["debug"] = {
                "eval_ctx_keys": list(eval_ctx.keys()),
                "exec_ctx_keys": list(exec_ctx.keys()),
                "confidence_calculation": eval_ctx["confidence"].get("adjustments", {}),
                "gate_details": {
                    "structural": eval_ctx.get("structural_gates", {}).get("summary", {}),
                    "opportunity": eval_ctx.get("opportunity_gates", {}).get("overall", {})
                },
                "execution_checks": exec_ctx.get("can_execute", {}).get("checks", {}),
                "position_sizing": exec_ctx.get("position_sizing", {})
            }

        # ======================================================
        # STAGE 7: FINAL DECISION
        # ======================================================
        finalize_trade_decision(
            plan=plan,
            eval_ctx=eval_ctx,
            exec_ctx=exec_ctx,
            extractor=extractor
        )
        # ======================================================
        # STAGE 8: 🆕 GENERATE ENHANCED NARRATIVES
        # ======================================================
        try:          
            # Get opportunity result from winner_profile
            if winner_profile:
                opportunity_result = {
                    "final_decision_score": winner_profile.get("final_score", 0),
                    "opportunity_bonus": winner_profile.get("opportunity", {}).get("bonus", 0),
                    "trade_context": winner_profile.get("opportunity", {}).get("trade_context", {})
                }
                eligibility_score = winner_profile.get("structural_eligibility", {}).get("score", 0)
            else:
                # Fallback if no winner_profile
                opportunity_result = None
                eligibility_score = 0
            
            # Generate all narratives
            narratives = build_enhanced_summaries(
                eval_ctx=eval_ctx,
                exec_ctx=exec_ctx,
                opportunity_result=opportunity_result,
                eligibility_score=eligibility_score,
                ticker=symbol,
                horizon=horizon
            )
            
            # Attach narratives to plan
            plan["narratives"] = narratives
            
            logger.info(
                f"[{symbol}] ✅ Generated {len(narratives)} narrative sections"
            )
            
        except Exception as e:
            logger.error(
                f"[{symbol}] ❌ Narrative generation failed: {e}",
                exc_info=True
            )
            plan["narratives"] = {}
        return plan

    except Exception as e:
        logger.error(f"[{symbol}] ❌ CRITICAL: Trade plan failed: {e}", exc_info=True)
        METRICS.log_failed_method("generate_trade_plan", e)
        
        return {
            "symbol": symbol,
            "horizon": horizon,
            "timestamp": datetime.now().timestamp(),
            "status": "ERROR",
            "reason": str(e),
            "metadata": {"error_trace": traceback.format_exc()}
        }
# ============================================================================
# 3. META-SCORING (For UI - Simple Aggregation)
# ============================================================================

def _calculate_profile_meta_score(
    weights: Dict,
    fundamentals: Dict,
    indicators: Dict,
    horizon: str,
    profile_name: str
) -> float:
    """
    ✅ REFACTORED: Generic profile scoring using normalized extraction.
    
    Args:
        weights: Metric weights (e.g., VALUE_WEIGHTS)
        fundamentals: Fundamental data dict
        indicators: Technical indicators dict
        horizon: Trading timeframe
        profile_name: Profile identifier (for logging)
    
    Returns:
        Weighted profile score (0-10)
    """
    total_weight = 0.0
    weighted_sum = 0.0
    missing_metrics = []
    
    for metric, weight_spec in weights.items():
        # 1. Get weight (handle both dict and scalar)
        if isinstance(weight_spec, dict):
            weight = weight_spec.get("weight", 0)
        else:
            weight = weight_spec
        
        if weight == 0:
            continue
        
        # 2. Find metric data (check fundamentals first, then indicators)
        metric_data = fundamentals.get(metric) or indicators.get(metric)
        
        if metric_data is None:
            missing_metrics.append(metric)
            continue
        
        # 3. Extract normalized score (0-10)
        score = extract_normalized_score(metric_data, metric, horizon)
        
        # 4. Apply directionality (if specified)
        if isinstance(weight_spec, dict):
            direction = weight_spec.get("direction", "normal")
            if direction == "invert":
                score = 10 - score
        
        # 5. Accumulate weighted score
        weighted_sum += score * weight
        total_weight += weight
        
        logger.debug(
            f"[{profile_name}] {metric}: raw_score={score:.2f}, "
            f"weight={weight:.3f}, contribution={score*weight:.2f}"
        )
    
    # 6. Calculate final score
    if total_weight == 0:
        logger.warning(
            f"[{profile_name}] No valid metrics found! "
            f"Missing: {missing_metrics}"
        )
        return 0.0
    
    final_score = weighted_sum / total_weight
    
    logger.info(
        f"[{profile_name}] Final Score: {final_score:.2f} "
        f"(from {len(weights) - len(missing_metrics)}/{len(weights)} metrics)"
    )
    
    return round(final_score, 2)

def score_value_profile(fundamentals: Dict, horizon: str = "short_term") -> float:
    """Calculate VALUE profile meta-score."""
    return _calculate_profile_meta_score(
        VALUE_WEIGHTS, fundamentals, {}, horizon, "VALUE"
    )


def score_growth_profile(fundamentals: Dict, horizon: str = "short_term") -> float:
    """Calculate GROWTH profile meta-score."""
    return _calculate_profile_meta_score(
        GROWTH_WEIGHTS, fundamentals, {}, horizon, "GROWTH"
    )


def score_quality_profile(fundamentals: Dict, horizon: str = "short_term") -> float:
    """Calculate QUALITY profile meta-score."""
    return _calculate_profile_meta_score(
        QUALITY_WEIGHTS, fundamentals, {}, horizon, "QUALITY"
    )


def score_momentum_profile(
    fundamentals: Dict,
    indicators: Dict,
    horizon: str = "short_term"
) -> float:
    """Calculate MOMENTUM profile meta-score."""
    return _calculate_profile_meta_score(
        MOMENTUM_WEIGHTS, fundamentals, indicators, horizon, "MOMENTUM"
    )


# ============================================================================
# 4. ✅ NEW: Config Retrieval Helpers (Extractor-based)f
# ============================================================================

def _get_horizon_pillar_weights(extractor) -> Dict[str, float]:
    """
    ✅ NEW: Get horizon pillar weights via extractor.
    
    Checks hierarchy:
    1. Horizon-specific weights (from horizon config)
    2. Global HORIZON_PILLAR_WEIGHTS (from master_config)
    3. Hardcoded defaults
    
    Args:
        extractor: QueryOptimizedExtractor instance
    
    Returns:
        Pillar weights dict with keys: tech, fund, hybrid
    """
    return extractor.get_horizon_pillar_weights()

def _extract_formation_context(
    indicators: Dict[str, Any],
    eval_ctx: Dict[str, Any]
) -> Dict[str, Any]:
    """
    ✅  uses eval_ctx["trend"] instead of parsing strings.
    
    Extracts formation context for velocity tracking.
    """
    # Safe extraction helper
    def _get_val(key):
        val = indicators.get(key, {})
        if isinstance(val, dict):
            return val.get("value")
        return val
    
    atr_pct = _get_val("atrPct")
    
    # ✅ FIX: Use pre-calculated trend from resolver
    trend = eval_ctx.get("trend", {})
    trend_regime = trend.get("regime", "normal")  # "strong", "normal", "weak"
    
    return {
        "adx": _get_val("adx"),
        "trend_strength": _get_val("trendStrength"),
        "volatility_regime": classify_volatility(atr_pct),
        "trend_regime": trend_regime,  # ✅ Direct from eval_ctx
        "trend_metadata": {
            "adx": trend.get("adx"),
            "slope": trend.get("slope"),
            "source": "resolver._build_trend_context"
        }
    }