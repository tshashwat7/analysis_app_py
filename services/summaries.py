# services/summaries.py (ENHANCED v2.0)
"""
Enhanced Summaries - Context-Driven Narrative Generation
========================================================
✅ REFACTORED: Now uses eval_ctx structure from resolver
✅ NEW: Confidence breakdown narratives
✅ NEW: Gate validation explanations
✅ NEW: Opportunity scoring details
✅ NEW: Trade plan narratives

Author: Quantitative Trading System
Version: 2.0 - Context-Driven Enhancement
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from services.data_fetch import _format_metric_name
from config.setup_pattern_matrix_config import PATTERN_INDICATOR_MAPPINGS
import logging
logger = logging.getLogger(__name__)


# ==========================================
# 🧠 KNOWLEDGE BASE (Educational Content)
# ==========================================

PATTERN_LIBRARY = {
    "cupHandle": {
        "what": "A bullish continuation pattern resembling a tea cup.",
        "implies": "The stock has consolidated gains and shaken out weak hands in the 'handle'. A breakout implies the prior uptrend is resuming with renewed energy."
    },
    "minerviniStage2": {
        "what": "Volatility Contraction Pattern (VCP) defined by Mark Minervini.",
        "implies": "Supply is drying up as institutions accumulate. Successive contractions in volatility indicate the stock is primed for an explosive breakout."
    },
    "darvasBox": {
        "what": "A momentum strategy tracking stocks making new highs in a 'box' range.",
        "implies": "The stock is in a strong uptrend, stepping up like a staircase. A breakout from the current box signals a new leg of momentum."
    },
    "bollingerSqueeze": {
        "what": "A period of extremely low volatility where Bollinger Bands narrow.",
        "implies": "The 'calm before the storm'. Energy is building up, and a violent expansion in price (breakout) is imminent."
    },
    "flagPennant": {
        "what": "A brief pause or consolidation in a strong vertical trend.",
        "implies": "The market is taking a breath before continuing the sprint. A breakout confirms the next leg up."
    },
    "goldenCross": {
        "what": "The 50-day Moving Average crosses above the 200-day Moving Average.",
        "implies": "A major long-term trend shift from bearish to bullish. Often signals the start of a sustained bull market."
    },
    "threeLineStrike": {
        "what": "A sharp 4-candle reversal pattern.",
        "implies": "Trapped traders are forced to cover positions, creating a powerful snap-back reversal in the opposite direction."
    },
    "doubleTopBottom": {
        "what": "Price tests a key level twice and reverses (W or M shape).",
        "implies": "A strong rejection of a price level. Double Bottom (W) indicates a support floor; Double Top (M) indicates a resistance ceiling."
    },
    "ichimokuSignals": {
        "what": "A comprehensive system showing support, resistance, and trend.",
        "implies": "Price is interacting with the 'Cloud'. A breakout above the cloud signals clear skies (uptrend) ahead."
    }
}

STRATEGY_LIBRARY = {
    "swing_trading": "Capitalizes on short-term price swings (3-10 days). We look for oversold dips in uptrends or mean-reversion setups.",
    "day_trading": "Focuses on intraday volatility and liquidity. We look for explosive volume and range expansion for quick profits.",
    "trend_following": "The 'Big Money' approach. We ignore small fluctuations and ride the major Moving Averages (50/200) for months.",
    "momentum": "Buying strength. We look for stocks hitting new highs with high Relative Strength (RSI) and volume surges.",
    "value": "Buying $1 for $0.50. We look for low P/E, P/B, and strong fundamentals that the market has undervalued.",
    "garp": "Growth at Reasonable Price. We combine strong earnings growth with reasonable valuations (PEG < 1.5).",
    "reversal_trading": "Bottom fishing in quality stocks. We look for oversold conditions with bullish divergence signals.",
}

SETUP_EXPLANATIONS = {
    "MOMENTUM_BREAKOUT": "Price breaking above resistance with strong volume, indicating institutional buying",
    "MOMENTUM_BREAKDOWN": "Price breaking below support with volume, signaling potential downtrend",
    "VOLATILITY_SQUEEZE": "Volatility compression preceding explosive move, like a coiled spring",
    "QUALITY_ACCUMULATION": "Quality stock in consolidation, institutions slowly accumulating position",
    "DEEP_VALUE_PLAY": "Fundamentally strong stock trading at significant discount to intrinsic value",
    "VALUE_TURNAROUND": "Improving fundamentals in previously beaten-down quality company",
    "TREND_PULLBACK": "Healthy pullback in established uptrend, offering entry opportunity",
    "DEEP_PULLBACK": "Deeper correction in strong trend, testing key support levels",
    "TREND_FOLLOWING": "Riding established trend with moving average support",
    "PATTERN_DARVAS_BREAKOUT": "Darvas box breakout - price stepping up to new highs",
    "PATTERN_VCP_BREAKOUT": "VCP pattern complete - volatility contraction resolved with breakout",
    "PATTERN_CUP_BREAKOUT": "Cup and handle pattern breakout - continuation of uptrend",
    "PATTERN_FLAG_BREAKOUT": "Bull flag breakout - brief consolidation in strong uptrend",
    "PATTERN_GOLDEN_CROSS": "Golden Cross signal - 50-day MA crossing above 200-day MA",
    "REVERSAL_MACD_CROSS_UP": "MACD bullish crossover - momentum shifting positive",
    "REVERSAL_RSI_SWING_UP": "RSI swing from oversold - potential bottom formation",
    "REVERSAL_ST_FLIP_UP": "Supertrend flip to bullish - trend reversal signal",
}

# ==========================================
# 🆕 CORE NARRATIVE GENERATORS
# ==========================================

def generate_confidence_narrative(eval_ctx: Dict[str, Any], horizon: str) -> str:
    """
    ✅ NEW: Generate human-readable confidence calculation explanation.
    
    Args:
        eval_ctx: Evaluation context from resolver
        horizon: Trading timeframe
    
    Returns:
        Narrative explanation of confidence calculation
    """
    conf = eval_ctx.get("confidence", {})
    setup_type = eval_ctx.get("setup", {}).get("type", "GENERIC")
    
    base = conf.get("base", 50)
    final = conf.get("final", 50)
    clamped = conf.get("clamped", 50)
    clamp_range = conf.get("clamp_range", [30, 95])
    
    adjustments = conf.get("adjustments", {})
    breakdown = adjustments.get("breakdown", [])
    
    # Start narrative
    narrative = f"**Confidence Score: {clamped}%** ({horizon})\n\n"
    
    # Explain starting point
    setup_name = _format_setup_name(setup_type)
    narrative += f"Started at **{base}%** ({setup_name} baseline for {horizon}).\n\n"
    
    # Categorize adjustments
    positive = []
    negative = []
    multipliers = []
    
    for entry in breakdown:
        if isinstance(entry, str):
            entry_lower = entry.lower()
            
            # Detect multipliers (divergence)
            if '×' in entry or 'multiplier' in entry_lower:
                multipliers.append(entry)
            # Detect positive adjustments
            elif '+' in entry or 'bonus' in entry_lower or 'boost' in entry_lower:
                positive.append(entry)
            # Detect negative adjustments
            elif '-' in entry or 'penalty' in entry_lower or 'warning' in entry_lower:
                negative.append(entry)
    
    # Explain positive factors
    if positive:
        narrative += "**✅ Boosting Factors:**\n"
        for p in positive[:5]:  # Top 5 boosters
            narrative += f"  • {_humanize_adjustment(p)}\n"
        narrative += "\n"
    
    # Explain negative factors
    if negative:
        narrative += "**⚠️ Risk Factors:**\n"
        for n in negative[:5]:  # Top 5 risks
            narrative += f"  • {_humanize_adjustment(n)}\n"
        narrative += "\n"
    
    # Explain multipliers (divergence)
    if multipliers:
        narrative += "**⚡ Divergence Applied:**\n"
        for m in multipliers:
            narrative += f"  • {_humanize_adjustment(m)}\n"
        narrative += "\n"
    
    # Explain clamping
    if clamped != final:
        if clamped == clamp_range[1]:
            narrative += f"Raw confidence ({final}%) exceeded {horizon} maximum, clamped to **{clamped}%**.\n"
        elif clamped == clamp_range[0]:
            narrative += f"Raw confidence ({final}%) below {horizon} minimum, raised to **{clamped}%**.\n"
    
    return narrative.strip()


def generate_gate_validation_narrative(eval_ctx: Dict[str, Any]) -> str:
    """
    ✅ NEW: Generate human-readable gate validation explanation.
    
    Args:
        eval_ctx: Evaluation context from resolver
    
    Returns:
        Narrative explanation of which gates passed/failed
    """
    gates = eval_ctx.get("entry_gates", {})
    
    if not gates:
        return "No entry gate information available."
    
    passed = gates.get("passed", False)
    results = gates.get("results", {})
    
    # Start narrative
    if passed:
        narrative = "**✅ All Entry Gates Passed**\n\n"
    else:
        narrative = "**⛔ Entry Gates Failed**\n\n"
    
    # Group results by category
    structural = results.get("structural", {})
    execution = results.get("execution_rules", {})
    opportunity = results.get("opportunity", {})
    
    # Explain structural gates
    if structural:
        narrative += "**Structural Gates:**\n"
        for gate_name, gate_result in structural.items():
            status = gate_result.get("passed", False)
            actual = gate_result.get("actual")
            required = gate_result.get("required", {})
            
            icon = "✅" if status else "❌"
            narrative += f"  {icon} {_format_gate_name(gate_name)}: "
            
            if actual is not None:
                narrative += f"{actual:.2f}"
                
                if required.get("min") is not None:
                    narrative += f" (required: ≥{required['min']})"
                elif required.get("max") is not None:
                    narrative += f" (required: ≤{required['max']})"
            else:
                narrative += "N/A"
            
            narrative += "\n"
        narrative += "\n"
    
    # Explain opportunity gates
    if opportunity:
        narrative += "**Opportunity Gates:**\n"
        for gate_name, gate_result in opportunity.items():
            status = gate_result.get("passed", False)
            actual = gate_result.get("actual")
            required = gate_result.get("required", {})
            
            icon = "✅" if status else "❌"
            narrative += f"  {icon} {_format_gate_name(gate_name)}: "
            
            if actual is not None:
                if gate_name == "confidence":
                    narrative += f"{actual:.0f}%"
                else:
                    narrative += f"{actual:.2f}"
                
                if required.get("min") is not None:
                    narrative += f" (required: ≥{required['min']})"
            else:
                narrative += "N/A"
            
            narrative += "\n"
        narrative += "\n"
    
    # Explain execution rules
    if execution:
        exec_summary = execution.get("summary", {})
        warnings = exec_summary.get("warnings", [])
        violations = exec_summary.get("violations", [])
        
        if warnings or violations:
            narrative += "**Execution Warnings:**\n"
            for w in warnings:
                narrative += f"  ⚠️ {_format_gate_name(w)}\n"
            for v in violations:
                narrative += f"  🚫 {_format_gate_name(v)}\n"
    
    return narrative.strip()


def generate_opportunity_narrative(
    eligibility_score: float,
    opportunity_result: Dict[str, Any],
    horizon: str
) -> str:
    """
    ✅ NEW: Generate human-readable opportunity scoring explanation.
    
    Args:
        eligibility_score: Base structural eligibility score
        opportunity_result: Result from compute_opportunity_score
        horizon: Trading timeframe
    
    Returns:
        Narrative explanation of opportunity scoring
    """
    final_score = opportunity_result.get("final_decision_score", 0)
    bonus = opportunity_result.get("opportunity_bonus", 0)
    trade_ctx = opportunity_result.get("trade_context", {})
    
    gate_passed = trade_ctx.get("gate_passed", False)
    setup = trade_ctx.get("setup", "GENERIC")
    strategy = trade_ctx.get("strategy", "generic")
    confidence = trade_ctx.get("confidence", 50)
    
    # Start narrative
    narrative = f"**Opportunity Score: {final_score:.1f}/10** ({horizon})\n\n"
    
    # Explain structural base
    narrative += f"**Base Structural Eligibility: {eligibility_score:.1f}/10**\n"
    narrative += "This combines Technical (70%), Fundamental (0%), and Hybrid (30%) pillars "
    narrative += f"weighted for {horizon} trading.\n\n"
    
    # Explain opportunity bonus (or lack thereof)
    if gate_passed:
        narrative += f"**Opportunity Bonus: +{bonus:.1f}**\n"
        narrative += f"Setup: {_format_setup_name(setup)}\n"
        narrative += f"Strategy: {strategy.replace('_', ' ').title()}\n"
        narrative += f"Confidence: {confidence}%\n\n"
        
        narrative += "✅ All entry gates passed - this is a tradeable opportunity.\n"
    else:
        narrative += "**No Opportunity Bonus** (entry gates not met)\n\n"
        block_reason = trade_ctx.get("block_reason", "Unknown")
        narrative += f"⛔ **Blocked:** {block_reason}\n"
        narrative += "Monitor this stock - conditions may improve.\n"
    
    return narrative.strip()


def generate_trade_plan_narrative(exec_ctx: Dict[str, Any], ticker: str) -> str:
    """
    ✅ NEW: Generate human-readable trade plan explanation.
    
    Args:
        exec_ctx: Execution context from resolver
        ticker: Stock symbol
    
    Returns:
        Narrative explanation of trade plan
    """
    risk = exec_ctx.get("risk", {})
    timeline = exec_ctx.get("timeline", {})
    pattern_meta = exec_ctx.get("pattern_meta", {})
    
    entry = risk.get("entry_price")
    sl = risk.get("stop_loss")
    targets = risk.get("targets", [])
    qty = risk.get("quantity", 0)
    rr = risk.get("rrRatio", 0)
    
    if not entry or not sl:
        return "No actionable trade plan available."
    
    # Start narrative
    narrative = f"**Trade Plan: {ticker}**\n\n"
    
    # Entry details
    narrative += f"**Entry:** ₹{entry:,.2f}\n"
    narrative += f"**Stop Loss:** ₹{sl:,.2f} ({_calculate_sl_pct(entry, sl):.1f}% risk)\n"
    
    # Targets
    if targets:
        narrative += f"**Target 1:** ₹{targets[0]:,.2f}"
        if timeline.get("available"):
            t1_est = timeline.get("t1_estimate", "N/A")
            narrative += f" (~{t1_est})"
        narrative += "\n"
        
        if len(targets) > 1:
            narrative += f"**Target 2:** ₹{targets[1]:,.2f}"
            if timeline.get("available"):
                t2_est = timeline.get("t2_estimate", "N/A")
                narrative += f" (~{t2_est})"
            narrative += "\n"
    
    # Risk/Reward
    narrative += f"\n**Risk/Reward Ratio:** {rr:.2f}:1"
    if rr >= 2.0:
        narrative += " (Excellent)"
    elif rr >= 1.5:
        narrative += " (Good)"
    else:
        narrative += " (Marginal)"
    narrative += "\n"
    
    # Position sizing
    capital_at_risk = qty * (entry - sl)
    narrative += f"\n**Position Size:** {qty} shares (₹{qty * entry:,.0f} investment)\n"
    narrative += f"**Capital at Risk:** ₹{capital_at_risk:,.0f}\n"
    
    # Pattern context
    if pattern_meta.get("available"):
        pattern_name = pattern_meta.get("pattern", "")
        age = pattern_meta.get("age_candles", 0)
        quality = pattern_meta.get("quality", 0)
        
        narrative += f"\n**Pattern:** {_format_pattern_name(pattern_name)}\n"
        narrative += f"Pattern Age: {age} candles | Quality: {quality}/10\n"
    
    return narrative.strip()


def generate_scoring_breakdown_narrative(eval_ctx: Dict[str, Any]) -> str:
    """
    ✅ NEW: Generate human-readable scoring breakdown.
    
    Args:
        eval_ctx: Evaluation context from resolver
    
    Returns:
        Narrative explanation of scoring components
    """
    scoring = eval_ctx.get("scoring", {})
    
    tech = scoring.get("technical", {})
    fund = scoring.get("fundamental", {})
    hybrid = scoring.get("hybrid", {})
    
    tech_score = tech.get("score", 0)
    fund_score = fund.get("score", 0)
    hybrid_score = hybrid.get("score", 0)
    
    try:
        narrative = "**Scoring Breakdown:**\n\n"
        
        # Technical pillar
        narrative += f"**Technical Score: {tech_score:.1f}/10**\n"
        tech_penalties = tech.get("penalties", {})
        
        # ✅ FIX: Extract the list of reasons safely
        if isinstance(tech_penalties, dict):
            penalty_list = tech_penalties.get("reasons", [])
        elif isinstance(tech_penalties, list):
            penalty_list = tech_penalties
        else:
            penalty_list = []

        if penalty_list:
            narrative += "\n**📉 Technical Drag:**\n"
            for p in penalty_list[:3]:
                narrative += f"  • {p}\n"
        narrative += "\n"
        
        # Fundamental pillar
        narrative += f"**Fundamental Score: {fund_score:.1f}/10**\n"
        fund_breakdown = fund.get("breakdown", {})
        
        if fund_breakdown:
            # ✅ FIX 1: Extract score/contribution from nested dicts
            top_metrics = []
            for metric, value in fund_breakdown.items():
                # Handle both dict and non-dict values
                if isinstance(value, dict):
                    score_val = value.get('score', value.get('weighted', value.get('contribution', 0)))
                else:
                    score_val = value if value is not None else 0
                
                top_metrics.append((metric, score_val))
            
            # Now safe to sort by numeric values
            top_metrics = sorted(
                top_metrics,
                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:3]
            
            narrative += "Top metrics:\n"
            for metric, score_val in top_metrics:
                if isinstance(score_val, (int, float)) and score_val > 0:
                    narrative += f"  • {_format_metric_name(metric)}: {score_val:.1f}\n"
        
        narrative += "\n"
        
        # Hybrid pillar
        narrative += f"**Hybrid Score: {hybrid_score:.1f}/10**\n"
        hybrid_breakdown = hybrid.get("breakdown", {})
        
        if hybrid_breakdown:
            narrative += "Components:\n"
            
            # ✅ FIX 2: Extract score/contribution from nested dicts
            for metric, value in list(hybrid_breakdown.items())[:3]:
                # Handle both dict and non-dict values
                if isinstance(value, dict):
                    score_val = value.get('score', value.get('weighted', value.get('contribution', 0)))
                else:
                    score_val = value if value is not None else 0
                
                if isinstance(score_val, (int, float)) and score_val > 0:
                    narrative += f"  • {_format_metric_name(metric)}: {score_val:.1f}\n"
    
    except Exception as e:
        logger.error(f"Error generating scoring breakdown narrative: {e}")
        import traceback
        logger.error(traceback.format_exc())  # ✅ BONUS: Better error logging
        return "Error generating scoring breakdown narrative."
    
    return narrative.strip()

def generate_rr_explanation_narrative(exec_ctx: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation of RR calculation.
    
    Args:
        exec_ctx: Execution context from resolver
    
    Returns:
        Narrative explaining RR calculation and adjustments
    """
    risk = exec_ctx.get("risk", {})
    market_adj = exec_ctx.get("market_adjusted_targets", {})
    
    structural_rr = market_adj.get("structural_rr", risk.get("rrRatio", 0))
    exec_rr_t1 = market_adj.get("execution_rr_t1", 0)
    exec_rr_t2 = market_adj.get("execution_rr_t2", 0)
    spread = market_adj.get("spread_cost", 0)
    
    entry = market_adj.get("execution_entry", risk.get("entry_price", 0))
    sl = market_adj.get("execution_sl", risk.get("stop_loss", 0))
    t1 = market_adj.get("execution_t1", 0)
    t2 = market_adj.get("execution_t2", 0)
    
    # Start narrative
    narrative = f"**Risk/Reward Analysis**\n\n"
    
    # Structural vs Execution
    narrative += f"**Pattern-Based (Structural) RR:** {structural_rr:.2f}:1\n"
    narrative += f"This represents the ideal geometry of the chart pattern.\n\n"
    
    # Execution adjustments
    narrative += f"**After Real-World Adjustments:**\n\n"
    
    # Spread impact
    if spread > 0:
        spread_pct = (spread / entry * 100) if entry > 0 else 0
        narrative += f"Spread Cost: ₹{spread:.2f} ({spread_pct:.2f}% of entry)\n"
        narrative += f"  • Reduces reward (bid-ask on target exit)\n"
        narrative += f"  • Increases risk (slippage on stop loss)\n\n"
    
    # Show calculations
    base_reward_t1 = t1 - entry if t1 and entry else 0
    base_risk = entry - sl if entry and sl else 0
    
    if base_reward_t1 and base_risk:
        narrative += f"**T1 Calculation:**\n"
        narrative += f"  Base Reward: ₹{base_reward_t1:.2f}\n"
        narrative += f"  After Spread: ₹{base_reward_t1 - spread:.2f}\n"
        narrative += f"  Base Risk: ₹{base_risk:.2f}\n"
        narrative += f"  After Spread: ₹{base_risk + spread:.2f}\n"
        narrative += f"  **Execution RR T1: {exec_rr_t1:.2f}:1**\n\n"
    
    if t2 and entry:
        base_reward_t2 = t2 - entry
        narrative += f"**T2 Calculation:**\n"
        narrative += f"  Base Reward: ₹{base_reward_t2:.2f}\n"
        narrative += f"  After Spread: ₹{base_reward_t2 - spread:.2f}\n"
        narrative += f"  **Execution RR T2: {exec_rr_t2:.2f}:1**\n\n"
    
    # Assessment
    narrative += "**Assessment:**\n"
    
    if exec_rr_t1 >= 1.5:
        narrative += f"✅ T1 execution RR ({exec_rr_t1:.2f}) meets minimum requirement (1.5)\n"
    else:
        narrative += f"⚠️ T1 execution RR ({exec_rr_t1:.2f}) below minimum (1.5)\n"
        
        if exec_rr_t2 >= 2.0:
            narrative += f"✅ But T2 execution RR ({exec_rr_t2:.2f}) is acceptable!\n"
            narrative += f"\n**Recommendation:** Use T2 (₹{t2:.2f}) as primary target instead of T1.\n"
        else:
            narrative += f"❌ T2 execution RR ({exec_rr_t2:.2f}) also below threshold (2.0)\n"
            narrative += f"\n**Reason:** Spread cost of ₹{spread:.2f} significantly impacts short-term targets.\n"
    
    # Why it matters
    narrative += f"\n**Why This Matters:**\n"
    narrative += f"Execution RR accounts for real trading costs that aren't visible on the chart. "
    narrative += f"While the pattern structure is sound (RR {structural_rr:.2f}), actual execution "
    narrative += f"conditions reduce your effective profit potential."
    
    return narrative

# ==========================================
# 🔧 HELPER FUNCTIONS
# ==========================================
def safe_numeric_extract(value, default=0):
    """Safely extract numeric value from dict or return value."""
    if isinstance(value, dict):
        return value.get('score', value.get('weighted', value.get('contribution', default)))
    return value if isinstance(value, (int, float)) else default
    
def _fmt_money(v: Optional[float]) -> str:
    """Format value as Indian Rupees."""
    try:
        return f"₹{float(v):,.2f}"
    except Exception:
        return "None"


def _format_setup_name(setup_type: str) -> str:
    """Convert setup type to human-readable name."""
    if not setup_type or setup_type == "GENERIC":
        return "Generic Setup"
    
    # Use explanation if available
    if setup_type in SETUP_EXPLANATIONS:
        return setup_type.replace("_", " ").title()
    
    return setup_type.replace("_", " ").title()


def _format_gate_name(gate_name: str) -> str:
    """Convert gate name to human-readable format."""
    replacements = {
        "adx": "ADX (Trend Strength)",
        "trendStrength": "Trend Strength",
        "volatilityQuality": "Volatility Quality",
        "atrPct": "ATR %",
        "rvol": "Relative Volume",
        "confidence": "Confidence",
        "rrRatio": "Risk/Reward Ratio",
        "technicalScore": "Technical Score",
        "fundamentalScore": "Fundamental Score",
        "volatility_guards": "Volatility Guards",
        "structure_validation": "Structure Validation",
        "sl_distance_validation": "Stop Loss Distance",
    }
    
    return replacements.get(gate_name, gate_name.replace("_", " ").title())


def _format_pattern_name(pattern: str) -> str:
    """Convert pattern name to human-readable format."""
    pattern_names = {
        "cupHandle": "Cup and Handle",
        "minerviniStage2": "VCP (Minervini Stage 2)",
        "darvasBox": "Darvas Box",
        "bollingerSqueeze": "Bollinger Squeeze",
        "flagPennant": "Flag/Pennant",
        "goldenCross": "Golden Cross",
        "threeLineStrike": "Three Line Strike",
        "doubleTopBottom": "Double Top/Bottom",
        "ichimokuSignals": "Ichimoku Signal",
    }
    
    return pattern_names.get(pattern, pattern.replace("_", " ").title())


def _humanize_adjustment(adjustment_str: str) -> str:
    """Convert technical adjustment string to human-readable format."""
    # Remove category prefixes
    text = re.sub(r'^(volume_modifiers\.|trend_strength_bands\.|penalty\.|bonus\.|enhancement\.)', '', adjustment_str)
    
    # Extract amount and reason
    match = re.match(r'([^:]+):\s*([+-]?\d+\.?\d*)\s*\(([^)]+)\)', text)
    if match:
        name, amount, reason = match.groups()
        return f"{reason} ({amount})"
    
    return adjustment_str


def _calculate_sl_pct(entry: float, sl: float) -> float:
    """Calculate stop loss percentage."""
    try:
        return abs((entry - sl) / entry * 100)
    except (ZeroDivisionError, TypeError):
        return 0.0


# ==========================================
# 📊 LEGACY FUNCTIONS (Maintained for Compatibility)
# ==========================================

def summarize_patterns(indicators: Dict[str, Any]) -> str:
    """
    Scans for ALL your patterns (Cup, VCP, etc.) and reports the active one.
    """
    active_patterns = []
    
    keys = PATTERN_INDICATOR_MAPPINGS.keys()
            
    for k in keys:
        p = indicators.get(k, {})
        if p.get("found"):
            key = _format_metric_name(k)
            name = key.replace("_", " ").title()
            meta = p.get("meta", {})
            desc = f"**{name}**"
            
            if k == "cupHandle":
                desc += f" (Depth: {meta.get('depth_pct')}%)"
            elif k == "minerviniStage2":
                desc += f" (Contraction: {meta.get('tightness')})"
            elif k == "bollingerSqueeze":
                desc += " (Volatility Compression)"
                
            active_patterns.append(desc)
            
    if not active_patterns:
        return "No classical chart patterns detected currently."
        
    return "🚀 **Chart Patterns:** " + ", ".join(active_patterns)


def get_active_pattern_details(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dictionary of definition/implication for every ACTIVE pattern.
    """
    details = {}
    for key, info in PATTERN_LIBRARY.items():
        p = indicators.get(key)
        if p and isinstance(p, dict) and p.get("found"):
            details[key] = info
    return details


def summarize_trade_recommendation(tr: Dict[str, Any]) -> str:
    """Legacy function - maintained for backward compatibility."""
    if not tr:
        return "No analysis available."
    
    signal = tr.get("signal", "N/A")
    setup = tr.get("setup_type", "Generic").replace("_", " ").title()
    conf = (
        tr.get("confidence") or
        tr.get("setup_confidence") or
        tr.get("final_confidence") or
        0
    )

    is_generic = setup.upper() == "GENERIC"
    is_executable = tr.get("entry") and tr.get("stop_loss")

    if "BUY" in signal and not is_generic and is_executable:
        targets = tr.get("targets", {})
        t1 = _fmt_money(targets.get("t1"))
        t2 = _fmt_money(targets.get("t2"))
        return (
            f"**Actionable {setup} Detected.** "
            f"Confidence is high ({conf}%) with a clear path to {t1}. "
            f"The risk-reward profile supports entry near {_fmt_money(tr.get('entry'))}."
        )

    if not tr.get("entry_permission", True):
        reason = tr.get("reason", "Entry conditions not met")
        return (
            f"**Setup identified but entry blocked.** "
            f"Reason: {reason}. Monitor for improvement."
        )

    if "WAIT" in signal or "NA_" in signal:
        reason = tr.get("reason", "conditions not met")
        
        if "VOLATILITY" in signal:
            return (
                f"**High Quality {setup} Setup identified ({conf}%)**, but volatility is too high. "
                f"Current ATR indicates chop/whipsaw risk. **Wait for VIX/ATR to cool down** before entering."
            )
        if "RESISTANCE" in signal:
            return (
                f"**Setup is forming**, but price is blocked by immediate resistance. "
                f"Do not buy yet. **Wait for a breakout above {tr.get('debug', {}).get('indicators_snapshot', {}).get('price')}** to confirm."
            )
        if "ENTRY_PERMISSION" in signal:
            return (
                f"**Technically valid {setup}**, but it failed the Entry Gate. "
                f"Reason: {reason}. Monitor for improved momentum."
            )
            
    if setup.upper() == "GENERIC":
        return (
            f"**Market structure detected (Generic bias).** "
            f"No executable setup yet. Confidence: {conf}%. "
            f"Wait for volume expansion or a valid breakout pattern."
        )
            
    return f"Current structure is **{setup}** but lacks conviction ({conf}%). {tr.get('reason')}."


# ==========================================
# 🆕 MAIN SUMMARY BUILDER (Context-Driven)
# ==========================================

def build_enhanced_summaries(
    eval_ctx: Dict[str, Any],
    exec_ctx: Optional[Dict[str, Any]] = None,
    opportunity_result: Optional[Dict[str, Any]] = None,
    eligibility_score: Optional[float] = None,
    ticker: str = "",
    horizon: str = "short_term"
) -> Dict[str, str]:
    """
    ✅ NEW: Build comprehensive summaries from evaluation context.
    
    Args:
        eval_ctx: Evaluation context from resolver
        exec_ctx: Optional execution context
        opportunity_result: Optional opportunity scoring result
        eligibility_score: Optional structural eligibility score
        ticker: Stock symbol
        horizon: Trading timeframe
    
    Returns:
        Dictionary of narrative summaries
    """
    summaries = {}
    try:
        
        # 1. Confidence Narrative
        summaries["confidence_narrative"] = generate_confidence_narrative(eval_ctx, horizon)
        
        # 2. Gate Validation Narrative
        summaries["gate_validation"] = generate_gate_validation_narrative(eval_ctx)
        
        # 3. Scoring Breakdown
        summaries["scoring_breakdown"] = generate_scoring_breakdown_narrative(eval_ctx)
        
        # 4. Opportunity Narrative (if available)
        if opportunity_result and eligibility_score is not None:
            summaries["opportunity_narrative"] = generate_opportunity_narrative(
                eligibility_score,
                opportunity_result,
                horizon
            )
        
        # 5. Trade Plan Narrative (if available)
        if exec_ctx:
            summaries["trade_plan"] = generate_trade_plan_narrative(exec_ctx, ticker)
        
        # 6. Setup Explanation
        setup_type = eval_ctx.get("setup", {}).get("type", "GENERIC")
        if setup_type in SETUP_EXPLANATIONS:
            summaries["setup_explanation"] = SETUP_EXPLANATIONS[setup_type]
        
        # 7. Strategy Explanation
        strategy = eval_ctx.get("strategy", {}).get("primary_strategy", "")
        if strategy in STRATEGY_LIBRARY:
            summaries["strategy_explanation"] = STRATEGY_LIBRARY[strategy]

        # 🆕 Add RR explanation if exec_ctx available
        if exec_ctx:
            summaries["rr_explanation"] = generate_rr_explanation_narrative(exec_ctx)
    
    except Exception as e:
        logger.debug(f"Error building enhanced summaries: {e}")
        raise  # Re-raise to signal failure
    
    return summaries


def build_all_summaries(result: Dict[str, Any]) -> Dict[str, str]:
    """
    Legacy function - maintained for backward compatibility.
    Maps old result structure to summaries.
    """
    indicators = result.get("indicators", {}) or {}
    tr = result.get("trade_recommendation", {}) or {}
    prof = result.get("profile_report", {}) or {}
    strat_report = result.get("strategy_report", {}) or {}
    
    # Get Best Strategy Name safely
    best_strat = "unknown"
    if strat_report.get("summary"):
        best_strat = strat_report["summary"].get("best_strategy", "unknown")
    
    return {
        "trade": summarize_trade_recommendation(tr),
        "patterns": summarize_patterns(indicators),
        "pattern_details": get_active_pattern_details(indicators),
        "strategy_details": STRATEGY_LIBRARY.get(best_strat.lower(), "Standard strategy analysis."),
        "market": f"Market Trend: {result.get('macro_trend_status', 'Neutral')}",
        "risk": f"Suggested Stop Loss: {_fmt_money(tr.get('stop_loss'))} ({tr.get('execution_hints', {}).get('risk_note', 'Standard Risk')})"
    }