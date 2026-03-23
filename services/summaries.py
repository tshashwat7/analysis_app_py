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
from markupsafe import escape  # ✅ P0-1: For XSS protection
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
    "swing_breakout": "Capitalizes on momentum-based price swings. We look for strong breakouts from consolidations or upper Bollinger Band breaks for short-term trend following.",
    "swing_pullback": "Mean-reversion focused swaps. We look for oversold dips or pullbacks to key supports within established uptrends for buy-low-sell-high opportunities.",
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
    Generate human-readable confidence calculation explanation.
    Uses structured_adjustments from resolver (pre-classified dicts).
    Returns HTML-safe string.
    """
    conf = eval_ctx.get("confidence", {})
    setup_type = eval_ctx.get("setup", {}).get("type", "GENERIC")

    base = conf.get("base", 50)
    final = conf.get("final", 50)
    clamped = conf.get("clamped", 50)
    clamp_range = conf.get("clamp_range", [30, 95])
    divergence_mult = conf.get("divergence_multiplier", 1.0)

    # Use structured_adjustments — resolver already classified these
    structured = conf.get("structured_adjustments", [])

    bonuses = [a for a in structured if a.get("direction") == "positive"]
    penalties = [a for a in structured if a.get("direction") == "negative"]

    setup_name = escape(_format_setup_name(setup_type))
    safe_horizon = escape(horizon)
    narrative = f"<b>Confidence Score: {clamped}%</b> ({safe_horizon})<br><br>"
    narrative += f"Base floor: <b>{base}%</b> ({setup_name}, {safe_horizon}).<br>"
    narrative += f"Adjustments: <b>{final - base:+.1f}%</b> → Raw {final:.1f}%<br><br>"

    if bonuses:
        narrative += "<b>✅ Boosting Factors:</b><br>"
        for a in bonuses[:5]:
            narrative += f"&nbsp;&nbsp;• {_humanize_adjustment_structured(a)}<br>"
        narrative += "<br>"

    if penalties:
        narrative += "<b>⚠️ Risk Factors:</b><br>"
        for a in penalties[:5]:
            narrative += f"&nbsp;&nbsp;• {_humanize_adjustment_structured(a)}<br>"
        narrative += "<br>"

    if divergence_mult != 1.0:
        if divergence_mult < 1.0:
            severity = "Severe" if divergence_mult <= 0.55 else ("Moderate" if divergence_mult <= 0.75 else "Minor")
            label = "bearish divergence"
            icon = "⚡"
        else:
            severity = "Strong" if divergence_mult >= 1.2 else "Minor"
            label = "bullish boost"
            icon = "🚀"
        narrative += f"<b>{icon} Divergence Logic:</b> {severity} {label} applied ({divergence_mult:.2f}×)<br><br>"

    if clamped != final:
        # ✅ P3-4: Correct bearish vocabulary (Ceiling -> Vulnerability)
        direction = eval_ctx.get("signal_intent", "BULLISH")
        limit_label = "ceiling" if direction == "BULLISH" else "vulnerability floor"
        
        if clamped == clamp_range[1]:
            narrative += f"Raw score ({final:.1f}%) reached {safe_horizon} {limit_label}, clamped to <b>{clamped}%</b>.<br>"
        elif clamped == clamp_range[0]:
            narrative += f"Raw score ({final:.1f}%) below {safe_horizon} {limit_label}, raised to <b>{clamped}%</b>.<br>"

    return narrative.strip()


# Actionable guidance for each gate when it fails
GATE_GUIDANCE = {
    "adx": "Trend is too weak — price is choppy, not directional. Wait for ADX > 25.",
    "rvol": "Volume is below average — institutional interest not yet confirmed. Wait for RVOL > 1.5×.",
    "trendStrength": "Insufficient momentum. Watch for higher lows + rising MA alignment.",
    "confidence": "Too many conflicting signals. Wait for a cleaner setup with fewer headwinds.",
    "rrRatio": "Risk/Reward too thin after costs. Wait for a pullback to improve entry price.",
    "atrPct": "Volatility outside safe range. Wait for ATR to normalize before entering.",
    "volatilityQuality": "Choppy price action. Look for smoother trend bars before entry.",
    "technicalScore": "Technical indicators are mixed. Wait for alignment across RSI, MACD, and MAs.",
    "fundamentalScore": "Fundamentals don't support the setup. Verify earnings and growth metrics.",
    "volatility_guards": "Volatility conditions unsafe for this setup type.",
    "structure_validation": "Price structure doesn't match the required pattern geometry.",
    "sl_distance_validation": "Stop loss too far from entry — position size would be too small to be meaningful.",
}


def generate_gate_validation_narrative(eval_ctx: Dict[str, Any]) -> str:
    """
    Generate human-readable gate validation explanation.
    Reads structural_gates.by_gate, opportunity_gates.gates, and execution_rules.
    Returns HTML-safe string with actionable guidance for failed gates.
    """
    structural_gates = eval_ctx.get("structural_gates", {})
    opportunity_gates = eval_ctx.get("opportunity_gates", {})
    execution_rules = eval_ctx.get("execution_rules", {})

    # Check if we have any gate data at all
    if not structural_gates and not opportunity_gates and not execution_rules:
        return "No entry gate information available."

    all_passed = (
        structural_gates.get("overall", {}).get("passed", True) and
        opportunity_gates.get("overall", {}).get("passed", True) and
        execution_rules.get("overall", {}).get("passed", True)
    )

    if all_passed:
        narrative = "<b>✅ All Gates Passed</b><br><br>"
    else:
        narrative = "<b>⛔ Gate Failures Detected</b><br><br>"

    # ── Structural Gates ─────────────────────────────────────────────────
    by_gate = structural_gates.get("by_gate", {})
    if by_gate:
        narrative += "<b>Structural Gates:</b><br>"
        for gate_name, result in by_gate.items():
            status = result.get("status", "skipped")
            actual = result.get("actual")
            required = result.get("required") or {}

            icon = "✅" if status == "passed" else ("⚠️" if status == "skipped" else "❌")
            narrative += f"&nbsp;&nbsp;{icon} {_format_gate_name(gate_name)}"

            if actual is not None:
                if isinstance(actual, (int, float)):
                    narrative += f": {actual:.2f}"
                else:
                    narrative += f": {escape(str(actual))}"
                if isinstance(required, dict):
                    if required.get("min") is not None:
                        narrative += f" (need ≥{required['min']})"
                    elif required.get("max") is not None:
                        narrative += f" (need ≤{required['max']})"

            if status == "failed" and gate_name in GATE_GUIDANCE:
                narrative += f'<br>&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#92400e;font-size:0.8em;">💡 {GATE_GUIDANCE[gate_name]}</span>'
            narrative += "<br>"
        narrative += "<br>"

    # ── Opportunity Gates ────────────────────────────────────────────────
    opp_gates = opportunity_gates.get("gates", {})
    if opp_gates:
        narrative += "<b>Opportunity Gates:</b><br>"
        for gate_name, result in opp_gates.items():
            status = result.get("status", "skipped")
            actual = result.get("actual")
            required = result.get("required") or {}

            icon = "✅" if status == "passed" else ("⚠️" if status == "skipped" else "❌")
            narrative += f"&nbsp;&nbsp;{icon} {_format_gate_name(gate_name)}"

            if actual is not None:
                if isinstance(actual, (int, float)):
                    fmt = f"{actual:.0f}%" if gate_name == "confidence" else f"{actual:.2f}"
                    narrative += f": {fmt}"
                else:
                    narrative += f": {escape(str(actual))}"
                if isinstance(required, dict) and required.get("min") is not None:
                    narrative += f" (need ≥{required['min']})"

            if status == "failed" and gate_name in GATE_GUIDANCE:
                narrative += f'<br>&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#92400e;font-size:0.8em;">💡 {GATE_GUIDANCE[gate_name]}</span>'
            narrative += "<br>"
        narrative += "<br>"

    # ── Execution Rules ─────────────────────────────────────────────────
    exec_summary = execution_rules.get("summary", {})
    warnings = exec_summary.get("warnings", [])
    violations = exec_summary.get("violations", [])
    if warnings or violations:
        narrative += "<b>Execution Warnings:</b><br>"
        for w in warnings:
            narrative += f"&nbsp;&nbsp;⚠️ {_format_gate_name(w)}<br>"
        for v in violations:
            narrative += f"&nbsp;&nbsp;🚫 {_format_gate_name(v)}<br>"

    return narrative.strip()


# Specific next-action triggers when entry is blocked
BLOCKED_NEXT_ACTIONS = {
    "LOW_CONFIDENCE": "Watch for RSI divergence or a volume surge above 2× average to re-trigger.",
    "GATE_FAILED": "Entry unlocks when failed gates turn green — check the Gate Status panel above.",
    "NO_PATTERN": "Set an alert for a breakout above the nearest resistance level.",
    "PATTERN_EXPIRED": "Pattern has aged out. Wait for a fresh setup to form (check back in 3-5 sessions).",
    "LOW_RR": "Risk/Reward too thin at current price. Wait for a pullback to improve entry.",
    "DIRECTION_NEUTRAL": "No directional bias yet. Watch for MACD crossover or Supertrend flip.",
    "VOLATILITY": "Volatility too high for safe entry. Wait for ATR% to contract below threshold.",
}


def generate_opportunity_narrative(
    eligibility_score: float,
    opportunity_result: Dict[str, Any],
    horizon: str
) -> str:
    """
    Generate human-readable opportunity scoring explanation.
    Returns HTML-safe string. When blocked, includes specific next-action triggers.
    """
    final_score = opportunity_result.get("final_decision_score", 0)
    bonus = opportunity_result.get("opportunity_bonus", 0)
    trade_ctx = opportunity_result.get("trade_context", {})
    
    gate_passed = trade_ctx.get("gate_passed", False)
    setup = trade_ctx.get("setup", "GENERIC")
    strategy = trade_ctx.get("strategy", "generic")
    confidence = trade_ctx.get("confidence", 50)
    
    safe_horizon = escape(horizon)
    narrative = f"<b>Opportunity Score: {final_score:.1f}/10</b> ({safe_horizon})<br><br>"
    
    narrative += f"<b>Base Structural Eligibility: {eligibility_score:.1f}/10</b><br>"
    narrative += f"Combines technical, fundamental, and hybrid scoring weighted for {horizon} trading.<br><br>"
    
    if gate_passed:
        narrative += f"<b>Opportunity Bonus: +{bonus:.1f}</b><br>"
        narrative += f"Setup: {escape(_format_setup_name(setup))}<br>"
        narrative += f"Strategy: {escape(strategy.replace('_', ' ').title())}<br>"
        narrative += f"Confidence: {confidence}%<br><br>"
        narrative += "✅ All entry gates passed — this is a tradeable opportunity.<br>"
    else:
        narrative += "<b>No Opportunity Bonus</b> (entry gates not met)<br><br>"
        block_reason = trade_ctx.get("block_reason", "Unknown")
        narrative += f"⛔ <b>Blocked:</b> {escape(str(block_reason))}<br><br>"
        
        # Match block reason to specific next-action
        next_action = None
        block_upper = block_reason.upper()
        if "CONFIDENCE" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["LOW_CONFIDENCE"]
        elif "GATE" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["GATE_FAILED"]
        elif "PATTERN" in block_upper and "EXPIRED" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["PATTERN_EXPIRED"]
        elif "PATTERN" in block_upper or "NO_PATTERN" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["NO_PATTERN"]
        elif "RR" in block_upper or "REWARD" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["LOW_RR"]
        elif "DIRECTION" in block_upper or "NEUTRAL" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["DIRECTION_NEUTRAL"]
        elif "VOLATIL" in block_upper or "ATR" in block_upper:
            next_action = BLOCKED_NEXT_ACTIONS["VOLATILITY"]
        
        if next_action:
            narrative += f'<span style="color:#0369a1;">💡 <b>Next step:</b> {next_action}</span><br>'
        else:
            narrative += '<span style="color:#6b7280;">💡 Monitor this stock — conditions may improve.</span><br>'
    
    return narrative.strip()


def generate_trade_plan_narrative(exec_ctx: Dict[str, Any], ticker: str) -> str:
    """
    Generate human-readable trade plan explanation.
    Returns HTML-safe string.
    """
    risk = exec_ctx.get("risk", {})
    timeline = exec_ctx.get("timeline", {})
    pattern_meta = exec_ctx.get("pattern_meta", {})
    
    entry = risk.get("entry_price")
    sl = risk.get("stop_loss")
    targets = risk.get("targets", [])
    qty = risk.get("quantity", 0)
    rr = risk.get("rrRatio") or 0.0
    
    if not entry or not sl:
        return "No actionable trade plan available."
    
    # ✅ P0-1: Escape dynamic ticker for XSS protection
    safe_ticker = escape(ticker)
    narrative = f"<b>Trade Plan: {safe_ticker}</b><br><br>"
    
    narrative += f"<b>Entry:</b> ₹{entry:,.2f}<br>"
    # ✅ P3-4: Terminology fix for bearish setups
    is_bearish = risk.get("direction", "").lower() == "bearish"
    sl_label = "Stop Loss"
    if is_bearish:
        sl_label = "Buy Stop (SL)"
        
    narrative += f"<b>{sl_label}:</b> ₹{sl:,.2f} ({_calculate_sl_pct(entry, sl):.1f}% risk)<br>"
    
    if targets:
        narrative += f"<b>Target 1:</b> ₹{targets[0]:,.2f}"
        if timeline.get("available"):
            narrative += f" (~{timeline.get('t1_estimate', 'N/A')})"
        narrative += "<br>"
        
        if len(targets) > 1:
            narrative += f"<b>Target 2:</b> ₹{targets[1]:,.2f}"
            if timeline.get("available"):
                narrative += f" (~{timeline.get('t2_estimate', 'N/A')})"
            narrative += "<br>"
    
    rr_label = "Excellent" if rr >= 2.0 else ("Good" if rr >= 1.5 else "Marginal")
    narrative += f"<br><b>Risk/Reward Ratio:</b> {rr:.2f}:1 ({rr_label})<br>"
    
    capital_at_risk = qty * (entry - sl)
    narrative += f"<br><b>Position Size:</b> {qty} shares (₹{qty * entry:,.0f} investment)<br>"
    narrative += f"<b>Capital at Risk:</b> ₹{capital_at_risk:,.0f}<br>"
    
    if pattern_meta.get("available"):
        pattern_name = pattern_meta.get("pattern", "")
        age = pattern_meta.get("age_candles", 0)
        quality = pattern_meta.get("quality", 0)
        narrative += f"<br><b>Pattern:</b> {escape(_format_pattern_name(pattern_name))}<br>"
        narrative += f"Pattern Age: {age} candles | Quality: {quality}/10<br>"
    
    return narrative.strip()


def generate_scoring_breakdown_narrative(eval_ctx: Dict[str, Any]) -> str:
    """
    Generate human-readable scoring breakdown.
    Returns HTML-safe string.
    """
    scoring = eval_ctx.get("scoring", {})
    
    tech = scoring.get("technical", {})
    fund = scoring.get("fundamental", {})
    hybrid = scoring.get("hybrid", {})
    
    tech_score = tech.get("score", 0)
    fund_score = fund.get("score", 0)
    hybrid_score = hybrid.get("score", 0)
    
    try:
        narrative = "<b>Scoring Breakdown:</b><br><br>"
        
        narrative += f"<b>Technical Score: {tech_score:.1f}/10</b><br>"
        tech_penalties = tech.get("penalties", {})
        
        if isinstance(tech_penalties, dict):
            penalty_list = tech_penalties.get("reasons", [])
        elif isinstance(tech_penalties, list):
            penalty_list = tech_penalties
        else:
            penalty_list = []

        if penalty_list:
            narrative += "<br><b>📉 Technical Penalties:</b><br>"
            for p in penalty_list[:3]:
                if isinstance(p, dict):
                    metric = p.get("metric", "Unknown")
                    score_val = p.get("score", 0)
                    safe_metric = escape(_format_metric_name(metric))
                    narrative += f"&nbsp;&nbsp;• {safe_metric}: {score_val:.1f}/10<br>"
                else:
                    narrative += f"&nbsp;&nbsp;• {escape(str(p))}<br>"
        
        narrative += f"<b>Fundamental Score: {fund_score:.1f}/10</b><br>"
        fund_breakdown = fund.get("breakdown", {})
        
        if fund_breakdown:
            top_metrics = []
            for metric, value in fund_breakdown.items():
                if isinstance(value, dict):
                    score_val = value.get('score', value.get('weighted', value.get('contribution', 0)))
                else:
                    score_val = value if value is not None else 0
                top_metrics.append((metric, score_val))
            
            top_metrics = sorted(
                top_metrics,
                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )[:3]
            
            narrative += "Top metrics:<br>"
            for metric, score_val in top_metrics:
                if isinstance(score_val, (int, float)) and score_val > 0:
                    # ✅ P0-1: Escape dynamic metric name
                    safe_metric = escape(_format_metric_name(metric))
                    narrative += f"&nbsp;&nbsp;• {safe_metric}: {score_val:.1f}<br>"
        
        narrative += "<br>"
        
        narrative += f"<b>Hybrid Score: {hybrid_score:.1f}/10</b><br>"
        hybrid_breakdown = hybrid.get("breakdown", {})
        
        if hybrid_breakdown:
            narrative += "Components:<br>"
            for metric, value in list(hybrid_breakdown.items())[:3]:
                if isinstance(value, dict):
                    score_val = value.get('score', value.get('weighted', value.get('contribution', 0)))
                else:
                    score_val = value if value is not None else 0
                if isinstance(score_val, (int, float)) and score_val > 0:
                    narrative += f"&nbsp;&nbsp;• {escape(_format_metric_name(metric))}: {score_val:.1f}<br>"
    
    except Exception as e:
        logger.error(f"Error generating scoring breakdown narrative: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "Error generating scoring breakdown narrative."
    
    return narrative.strip()

def generate_rr_explanation_narrative(exec_ctx: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation of RR calculation.
    Returns HTML-safe string. Leads with verdict, then shows calculations.
    """
    risk = exec_ctx.get("risk", {})
    market_adj = exec_ctx.get("market_adjusted_targets", {})
    
    structural_rr = market_adj.get("structural_rr")
    if structural_rr is None:
        structural_rr = risk.get("rrRatio")
    if structural_rr is None:
        structural_rr = 0.0
    
    exec_rr_t1 = market_adj.get("execution_rr_t1") or 0.0
    exec_rr_t2 = market_adj.get("execution_rr_t2") or 0.0
    spread = market_adj.get("spread_cost", 0)
    
    entry = market_adj.get("execution_entry", risk.get("entry_price", 0))
    sl = market_adj.get("execution_sl", risk.get("stop_loss", 0))
    t1 = market_adj.get("execution_t1", 0)
    t2 = market_adj.get("execution_t2", 0)
    
    # ── VERDICT FIRST ────────────────────────────────────────────────────
    narrative = "<b>Risk/Reward Analysis</b><br><br>"

    # Guard: if no market-adjusted data yet, fall back to structural RR
    if not exec_rr_t1 and not exec_rr_t2:
        if structural_rr >= 1.5:
            narrative += f'<div style="background:#dcfce7;color:#166534;padding:8px 12px;border-radius:6px;margin-bottom:10px;">'
            narrative += f'✅ <b>Structural RR: {structural_rr:.1f}:1</b> — execution costs not yet calculated.</div>'
        elif structural_rr > 0:
            narrative += f'<div style="background:#fef3c7;color:#92400e;padding:8px 12px;border-radius:6px;margin-bottom:10px;">'
            narrative += f'⚠️ <b>Structural RR: {structural_rr:.1f}:1</b> — review after execution costs applied.</div>'
        else:
            narrative += f'<div style="background:#f3f4f6;color:#374151;padding:8px 12px;border-radius:6px;margin-bottom:10px;">'
            narrative += f'ℹ️ RR not yet calculated — trade plan pending.</div>'
        return narrative

    if exec_rr_t1 >= 1.5:
        narrative += f'<div style="background:#dcfce7;color:#166534;padding:8px 12px;border-radius:6px;margin-bottom:10px;">'
        narrative += f'✅ <b>Favorable</b> — you risk ₹1 to make ₹{exec_rr_t1:.1f} at T1 (after costs).</div>'
    elif exec_rr_t2 and exec_rr_t2 >= 2.0:
        narrative += f'<div style="background:#fef3c7;color:#92400e;padding:8px 12px;border-radius:6px;margin-bottom:10px;">'
        narrative += f'⚠️ <b>Marginal T1</b> ({exec_rr_t1:.2f}:1), but T2 gives {exec_rr_t2:.1f}:1 — consider scaling into T2.</div>'
    else:
        narrative += f'<div style="background:#fee2e2;color:#991b1b;padding:8px 12px;border-radius:6px;margin-bottom:10px;">'
        narrative += f'❌ <b>Poor R/R after costs.</b> Pattern is valid but entry now reduces edge ({exec_rr_t1:.2f}:1 T1).'
        if spread > 0:
            narrative += f' Spread ₹{spread:.2f} is eating the reward.'
        narrative += '</div>'
    
    # ── DETAILS (collapsible-friendly) ───────────────────────────────────
    narrative += f"<b>Pattern-Based (Structural) RR:</b> {structural_rr:.2f}:1<br>"
    narrative += f"<span style='color:#6b7280;font-size:0.85em;'>The ideal geometry of the chart pattern.</span><br><br>"
    
    if spread > 0:
        spread_pct = (spread / entry * 100) if entry > 0 else 0
        narrative += f"<b>Spread Cost:</b> ₹{spread:.2f} ({spread_pct:.2f}% of entry)<br>"
        narrative += "&nbsp;&nbsp;• Reduces reward (bid-ask on target exit)<br>"
        narrative += "&nbsp;&nbsp;• Increases risk (slippage on stop loss)<br><br>"
    
    base_reward_t1 = t1 - entry if t1 and entry else 0
    base_risk = entry - sl if entry and sl else 0
    
    if base_reward_t1 and base_risk:
        narrative += "<b>T1 Calculation:</b><br>"
        narrative += f"&nbsp;&nbsp;Base Reward: ₹{base_reward_t1:.2f}<br>"
        narrative += f"&nbsp;&nbsp;After Spread: ₹{base_reward_t1 - spread:.2f}<br>"
        narrative += f"&nbsp;&nbsp;Base Risk: ₹{base_risk:.2f}<br>"
        narrative += f"&nbsp;&nbsp;After Spread: ₹{base_risk + spread:.2f}<br>"
        narrative += f"&nbsp;&nbsp;<b>Execution RR T1: {exec_rr_t1:.2f}:1</b><br><br>"
    
    if t2 and entry:
        base_reward_t2 = t2 - entry
        narrative += "<b>T2 Calculation:</b><br>"
        narrative += f"&nbsp;&nbsp;Base Reward: ₹{base_reward_t2:.2f}<br>"
        narrative += f"&nbsp;&nbsp;After Spread: ₹{base_reward_t2 - spread:.2f}<br>"
        narrative += f"&nbsp;&nbsp;<b>Execution RR T2: {exec_rr_t2:.2f}:1</b><br><br>"
    
    # Recommendation if T1 fails but T2 is ok
    if exec_rr_t1 < 1.5 and exec_rr_t2 and exec_rr_t2 >= 2.0:
        narrative += f'<span style="color:#0369a1;">💡 <b>Recommendation:</b> Use T2 (₹{t2:,.2f}) as primary target instead of T1.</span><br><br>'
    
    narrative += '<span style="color:#6b7280;font-size:0.85em;">'
    narrative += 'Execution RR accounts for real trading costs not visible on the chart. '
    narrative += f'While the pattern structure is sound (RR {structural_rr:.2f}), actual execution '
    narrative += 'conditions affect your effective profit potential.</span>'
    
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


def _humanize_adjustment_structured(adj: Dict[str, Any]) -> str:
    """Convert structured_adjustments dict to readable string."""
    name = adj.get("name", "").replace("_", " ").title()
    delta = adj.get("delta", 0)
    source = adj.get("source", "")

    SOURCE_LABELS = {
        "conditional": "",
        "volume_modifiers": "Volume",
        "universal_adjustments": "Universal",
        "setup_validation": "Setup",
        "execution": "Execution",
    }
    prefix = SOURCE_LABELS.get(source, source.replace("_", " ").title())
    label = f"{prefix}: {name}" if prefix else name
    return f"{label} ({delta:+.1f}%)"


def _humanize_adjustment(adjustment_str: str) -> str:
    """Convert technical adjustment string to human-readable format (legacy fallback)."""
    # Strip category prefixes
    text = re.sub(r'^(setup_penalties\.|setup_bonuses\.|execution_warning\.|execution_violation\.)', '', adjustment_str)
    text = re.sub(r'^(volume_modifiers\.|trend_strength_bands\.|penalty\.|bonus\.|enhancement\.)', '', text)

    # Format: "name: +10.0" or "name: -20.0"
    match = re.match(r'([^:]+):\s*([+-]?\d+\.?\d*)$', text.strip())
    if match:
        name, amount = match.groups()
        name_clean = name.replace("_", " ").title()
        return f"{name_clean} ({float(amount):+.1f}%)"

    return text


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
            desc = f"<b>{name}</b>"
            
            if k == "cupHandle":
                desc += f" (Depth: {meta.get('depth_pct')}%)"
            elif k == "minerviniStage2":
                # ✅ P1-6: Use contraction_pct with ATR fallback
                c_pct = meta.get('contraction_pct')
                if c_pct:
                    desc += f" (Contraction: {c_pct}%)"
                else:
                    desc += " (Tight Consolidation)"
            elif k == "bollingerSqueeze":
                desc += " (Volatility Compression)"
                
            active_patterns.append(desc)
            
    if not active_patterns:
        return "No classical chart patterns detected currently."
        
    return "🚀 <b>Chart Patterns:</b> " + ", ".join(active_patterns)


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
    """Legacy function - maintained for backward compatibility. Returns HTML."""
    if not tr:
        return "No analysis available."
    
    # Use trade_signal (from finalize_trade_decision), fallback to signal
    signal = escape(tr.get("trade_signal", tr.get("signal", "N/A")))
    setup = escape(tr.get("setup_type", "Generic").replace("_", " ").title())
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
        # targets is a dict with t1/t2 keys from signal_engine
        if isinstance(targets, dict):
            t1 = _fmt_money(targets.get("t1"))
        elif isinstance(targets, list):
            t1 = _fmt_money(targets[0] if targets else None)
        else:
            t1 = "N/A"
        return (
            f"<b>Actionable {setup} Detected.</b> "
            f"Confidence is high ({conf}%) with a clear path to {t1}. "
            f"The risk-reward profile supports entry near {_fmt_money(tr.get('entry'))}."
        )

    if not tr.get("entry_permission", True):
        reason = escape(tr.get("reason", "Entry conditions not met"))
        return (
            f"<b>Setup identified but entry blocked.</b> "
            f"Reason: {reason}. Monitor for improvement."
        )

    if "WAIT" in signal or "WATCH" in signal or "NA_" in signal:
        reason = tr.get("reason", "conditions not met")
        
        if "VOLATILITY" in signal:
            return (
                f"<b>High Quality {setup} Setup identified ({conf}%)</b>, but volatility is too high. "
                f"Current ATR indicates chop/whipsaw risk. <b>Wait for VIX/ATR to cool down</b> before entering."
            )
        if "RESISTANCE" in signal:
            return (
                f"<b>Setup is forming</b>, but price is blocked by immediate resistance. "
                f"Do not buy yet. <b>Wait for a breakout</b> to confirm."
            )
        if "ENTRY_PERMISSION" in signal:
            return (
                f"<b>Technically valid {setup}</b>, but it failed the Entry Gate. "
                f"Reason: {reason}. Monitor for improved momentum."
            )

    if "BLOCKED" in signal:
        reason = escape(tr.get("reason", "Execution gates not met"))
        return (
            f"<b>{setup} setup detected but entry is blocked.</b> "
            f"Reason: {reason}. Check gate status for specifics."
        )

    if setup.upper() == "GENERIC":
        return (
            f"<b>Market structure detected (Generic bias).</b> "
            f"No executable setup yet. Confidence: {conf}%. "
            f"Wait for volume expansion or a valid breakout pattern."
        )
            
    return f"Current structure is <b>{setup}</b> but lacks conviction ({conf}%). {escape(tr.get('reason'))}."


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
        strategy = eval_ctx.get("strategy", {}).get("primary", "")
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
    if isinstance(strat_report, dict):
        summary_node = strat_report.get("best") or strat_report.get("summary") or {}
        if isinstance(summary_node, dict):
            best_strat = summary_node.get("strategy") or summary_node.get("best_strategy", "unknown")
    
    return {
        "trade": summarize_trade_recommendation(tr),
        "patterns": summarize_patterns(indicators),
        "pattern_details": get_active_pattern_details(indicators),
        "strategy_details": STRATEGY_LIBRARY.get(best_strat.lower(), "Standard strategy analysis."),
        "market": f"Market Trend: {escape(result.get('macro_trend_status', 'Neutral'))}",
        "risk": f"Suggested Stop Loss: {_fmt_money(tr.get('stop_loss'))} ({tr.get('execution_hints', {}).get('risk_note', 'Standard Risk')})"
    }