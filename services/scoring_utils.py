import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

def get_horizon_pillar_weights(extractor) -> Dict[str, float]:
    """
    Get horizon pillar weights via extractor.
    
    Args:
        extractor: QueryOptimizedExtractor instance
    
    Returns:
        Pillar weights dict with keys: tech, fund, hybrid
    """
    return extractor.get_horizon_pillar_weights()

def calculate_structural_eligibility(
    tech_score: float,
    fund_score: float,
    hybrid_score: float,
    horizon: str,
    extractor: Any
) -> float:
    """
    Blends the three structural pillars based on horizon DNA.
    
    Args:
        tech_score: Technical pillar score (0-10)
        fund_score: Fundamental pillar score (0-10)
        hybrid_score: Hybrid pillar score (0-10)
        horizon: Trading timeframe
        extractor: QueryOptimizedExtractor instance
    
    Returns:
        Weighted eligibility score (0-10)
    """
    try:
        # Get weights via extractor
        weights = get_horizon_pillar_weights(extractor)
        
        # Weight Redistribution for missing pillars
        scores = {
            'tech': tech_score,
            'fund': fund_score,
            'hybrid': hybrid_score
        }
        
        active_weights = {k: v for k, v in weights.items() if k in scores and scores[k] is not None}
        
        if not active_weights:
            logger.warning(f"[{horizon}] No active pillars for structural eligibility!")
            return 0.0
            
        total_active_weight = sum(active_weights.values())
        
        if total_active_weight <= 0:
            logger.warning(f"[{horizon}] Total active weight is zero!")
            return max(tech_score or 0.0, fund_score or 0.0, hybrid_score or 0.0)
            
        eligibility_score = 0.0
        for k, w in active_weights.items():
            # Redistribute weight proportionally to active pillars
            redistributed_weight = w / total_active_weight
            val = scores.get(k, 0.0) or 0.0
            eligibility_score += val * redistributed_weight
        
        return round(eligibility_score, 2)
        
    except Exception as e:
        logger.error(
            f"[{horizon}] ❌ STRUCTURAL ELIGIBILITY FAILED | Error: {e}",
            exc_info=True
        )
        return 0.0

def compute_opportunity_score_logic(
    ticker: str,
    horizon: str,
    eligibility_base_score: float,
    confidence: float,
    priority: float,
    strategy_fit_score: float,
    baseline_floor: float,
) -> float:
    """
    Pure math logic for opportunity scoring bonus.
    
    Returns:
        Opportunity bonus (0-5 scale)
    """
    opp_bonus = 0.0
    
    # Bonus A: Pattern/Setup priority
    priority_bonus = priority / 100.0
    opp_bonus += priority_bonus
    
    # Bonus B: Strategy Fit
    fit_bonus = strategy_fit_score / 100.0
    opp_bonus += fit_bonus
    
    # Bonus C: Conviction (Excess confidence over baseline floor)
    # Use 50 as safe fallback for bonus calc if floor is None
    floor_val = baseline_floor if baseline_floor is not None else 50.0
    conviction_bonus = max(0, (confidence - floor_val) / 50.0)
    opp_bonus += conviction_bonus
    
    return opp_bonus

def compute_opportunity_score_full(
    ticker: str,
    horizon: str,
    eligibility_base_score: float,
    indicators: Dict[str, Any],
    fundamentals: Dict[str, Any],
    eval_ctx: Dict[str, Any],
    extractor: Any
) -> Dict[str, Any]:
    """
    Full opportunity scoring logic, decoupled from signal_engine.
    Extracted from signal_engine.compute_opportunity_score.
    """
    conf_info  = eval_ctx.get("confidence", {})
    confidence = conf_info.get("clamped", 0.0)
    
    # 1. Baseline floor from setup
    setup_info     = eval_ctx.get("setup", {})
    setup_type     = setup_info.get("type", "GENERIC")
    baseline_floor = extractor.get_setup_baseline_floor(setup_type)
    
    # 2. Priority and strategy fit
    setup_priority = extractor.get_setup_priority(setup_type)
    strategy_fit   = eval_ctx.get("strategy", {}).get("fit_score", 0.0)
    
    # 3. Calculate bonus via shared logic
    opp_bonus = compute_opportunity_score_logic(
        ticker                 = ticker,
        horizon                = horizon,
        eligibility_base_score = eligibility_base_score,
        confidence             = confidence,
        priority               = setup_priority,
        strategy_fit_score     = strategy_fit,
        baseline_floor         = baseline_floor
    )
    
    # 4. Final calculation
    final_score = calculate_final_decision_score(eligibility_base_score, opp_bonus)
    
    return {
        "final_decision_score": final_score,
        "bonus_points":         round(opp_bonus, 2),
        "priority_used":        setup_priority,
        "baseline_floor_used":  baseline_floor,
        "fit_used":             strategy_fit
    }

def calculate_final_decision_score(
    eligibility_base_score: float,
    opp_bonus: float
) -> float:
    """
    Calculate final decision score from base eligibility and opportunity bonus.
    
    Args:
        eligibility_base_score: 0-10 scale
        opp_bonus: ~0-3 scale
        
    Returns:
        Final score (0-10 scale)
    """
    # Normalize bonus to a 10-point scale (Max practical bonus is ~3.0)
    max_practical_bonus = 3.0
    normalized_bonus = min(10.0, (opp_bonus / max_practical_bonus) * 10.0)
    
    final_score = (eligibility_base_score * 0.70) + (normalized_bonus * 0.30)
    return round(min(10.0, final_score), 2)
