# services/summaries.py
from typing import Dict, Any, Optional
from datetime import datetime

# ==========================================
# 🧠 KNOWLEDGE BASE (Educational Content)
# ==========================================

PATTERN_LIBRARY = {
    "cup_handle": {
        "what": "A bullish continuation pattern resembling a tea cup.",
        "implies": "The stock has consolidated gains and shaken out weak hands in the 'handle'. A breakout implies the prior uptrend is resuming with renewed energy."
    },
    "minervini_stage2": {
        "what": "Volatility Contraction Pattern (VCP) defined by Mark Minervini.",
        "implies": "Supply is drying up as institutions accumulate. Successive contractions in volatility indicate the stock is primed for an explosive breakout."
    },
    "darvas_box": {
        "what": "A momentum strategy tracking stocks making new highs in a 'box' range.",
        "implies": "The stock is in a strong uptrend, stepping up like a staircase. A breakout from the current box signals a new leg of momentum."
    },
    "bollinger_squeeze": {
        "what": "A period of extremely low volatility where Bollinger Bands narrow.",
        "implies": "The 'calm before the storm'. Energy is building up, and a violent expansion in price (breakout) is imminent."
    },
    "flag_pennant": {
        "what": "A brief pause or consolidation in a strong vertical trend.",
        "implies": "The market is taking a breath before continuing the sprint. A breakout confirms the next leg up."
    },
    "golden_cross": {
        "what": "The 50-day Moving Average crosses above the 200-day Moving Average.",
        "implies": "A major long-term trend shift from bearish to bullish. Often signals the start of a sustained bull market."
    },
    "three_line_strike": {
        "what": "A sharp 4-candle reversal pattern.",
        "implies": "Trapped traders are forced to cover positions, creating a powerful snap-back reversal in the opposite direction."
    },
    "double_top_bottom": {
        "what": "Price tests a key level twice and reverses (W or M shape).",
        "implies": "A strong rejection of a price level. Double Bottom (W) indicates a support floor; Double Top (M) indicates a resistance ceiling."
    },
    "ichimoku_signals": {
        "what": "A comprehensive system showing support, resistance, and trend.",
        "implies": "Price is interacting with the 'Cloud'. A breakout above the cloud signals clear skies (uptrend) ahead."
    }
}

STRATEGY_LIBRARY = {
    "swing": "Capitalizes on short-term price swings (3-10 days). We look for oversold dips in uptrends or mean-reversion setups.",
    "day_trading": "Focuses on intraday volatility and liquidity. We look for explosive volume and range expansion for quick profits.",
    "trend_following": "The 'Big Money' approach. We ignore small fluctuations and ride the major Moving Averages (50/200) for months.",
    "momentum": "Buying strength. We look for stocks hitting new highs with high Relative Strength (RSI) and volume surges.",
    "value": "Buying $1 for $0.50. We look for low P/E, P/B, and strong fundamentals that the market has undervalued.",
    "position_trading": "Long-term wealth building. We combine strong fundamentals (EPS growth) with a bullish primary trend.",
    "minervini": "Specific Growth + Momentum strategy. We look for VCP patterns in stocks with high earnings acceleration.",
    "canslim": "William O'Neil's strategy. We combine Earnings (C,A), New Highs (N), and Market Leaders (L)."
}

def _fmt_money(v: Optional[float]) -> str:
    try:
        return f"₹{float(v):,.2f}"
    except Exception:
        return "None"

def summarize_patterns(indicators: Dict[str, Any]) -> str:
    """
    Scans for ALL your patterns (Cup, VCP, etc.) and reports the active one.
    """
    active_patterns = []
    
    keys = ["cup_handle", "minervini_stage2", "darvas_box", "bollinger_squeeze", 
            "flag_pennant", "golden_cross", "three_line_strike"]
            
    for k in keys:
        p = indicators.get(k, {})
        if p.get("found"):
            name = k.replace("_", " ").title()
            meta = p.get("meta", {})
            desc = f"**{name}**"
            
            if k == "cup_handle":
                desc += f" (Depth: {meta.get('depth_pct')}%)"
            elif k == "minervini_stage2":
                desc += f" (Contraction: {meta.get('tightness')})"
            elif k == "bollinger_squeeze":
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
    if not tr: return "No analysis available."
    
    signal = tr.get("signal", "N/A")
    setup = tr.get("setup_type", "Generic").replace("_", " ").title()
    conf = tr.get("setup_confidence", 0)
    
    # 1. POSITIVE SCENARIO (Execute)
    if "BUY" in signal and "WAIT" not in signal:
        targets = tr.get("targets", {})
        t1 = _fmt_money(targets.get("t1"))
        t2 = _fmt_money(targets.get("t2"))
        return (
            f"**Actionable {setup} Detected.** "
            f"Confidence is high ({conf}%) with a clear path to {t1}. "
            f"The risk-reward profile supports entry near {_fmt_money(tr.get('entry'))}."
        )

    # 2. WATCHLIST SCENARIO (Valid Setup, but Blocked)
    if "WAIT" in signal or "NA_" in signal:
        # Decode the blocker from the signal string or debug info
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
            
    # 3. NEGATIVE SCENARIO
    return f"Current structure is **{setup}** but lacks conviction ({conf}%). {tr.get('reason')}."

def summarize_risk_execution(tr: Dict[str, Any]) -> str:
    rr = tr.get("RR_Ratio")
    trailing = tr.get("Trailing_Stop_Loss")
    advice = tr.get("advice") or None
    rr_text = "moderate"
    if rr is not None:
        try:
            rr_val = float(rr)
            if rr_val >= 2:
                rr_text = "favorable"
            elif rr_val < 1:
                rr_text = "unfavorable"
        except Exception:
            rr_text = "moderate"
    text = f"Risk-to-reward ratio is **{rr_text} ({rr or 'N/A'})**, indicating a "
    text += "balanced trade setup." if rr_text == "moderate" else ("strong opportunity." if rr_text == "favorable" else "low probability setup.")
    if trailing:
        text += f" Suggested trailing stop-loss: {_fmt_money(trailing)}."
    if advice:
        text += f" Advice: {advice}."
    return text

def summarize_entry_stop(indicators: Dict[str, Any]) -> str:
    entry = indicators.get('Entry Price (Confirm)', {}).get('value') if indicators else None
    sl2 = indicators.get('Suggested SL (2xATR)', {}).get('value') if indicators else None
    atr = indicators.get('ATR (14)', {}).get('value') if indicators else None
    entry_str = _fmt_money(entry) if entry not in (None, 'N/A') else "None"
    sl_str = _fmt_money(sl2) if sl2 not in (None, 'N/A') else "None"
    atr_str = f"₹{float(atr):.2f}/day" if atr not in (None, 'N/A') else "N/A"
    if entry in (None, 'N/A'):
        return f"The stock is volatile (~{atr_str}). No buy signal yet (Confirm Above: {entry_str}). If entered, a safe stop-loss is {sl_str}."
    else:
        return f"The stock is volatile (~{atr_str}). A buy signal confirms above {entry_str}. Suggested stop-loss (2×ATR) is {sl_str}."

def summarize_market_status(trend: str, close: Any, index: str) -> str:
    trend = trend or "N/A"
    idx = index or "NIFTY50"
    
    close_val = None
    if isinstance(close, (float, int, str)):
        try:
            close_val = float(close)
        except (ValueError, TypeError):
            close_val = None

    # --- THIS IS THE FIX ---
    # We check 'is not None' instead of 'if close'
    if close_val is not None:
        close_text = f"last close at {close_val:,.2f}"
    else:
        close_text = "no recent close data"
    # --- END FIX ---

    if "BULL" in trend.upper() or "Uptrend" in trend:
        msg = f"The broader market ({idx}) remains **bullish**, trading above its long-term average ({close_text})."
    elif "BEAR" in trend.upper() or "Downtrend" in trend:
        msg = f"The broader market ({idx}) is **bearish**, trading below its long-term average ({close_text})."
    else:
        msg = f"Market trend for {idx} is neutral with {close_text}."
    return msg

def summarize_score_breakdown(final_score: float, meta_scores: Dict[str, float], profile_report: Dict) -> str:
    """
    Intelligent Attribution: Tells you WHAT is driving the score.
    """
    drivers = []
    drags = []
    
    # Analyze the metric_details from profile_report
    details = profile_report.get("metric_details", {})
    
    for k, score in details.items():
        if score >= 9: drivers.append(k.replace("_", " ").title())
        elif score <= 3: drags.append(k.replace("_", " ").title())
            
    msg = f"**Final Score: {final_score}/10.** "
    
    if drivers:
        top_3 = ", ".join(drivers[:3])
        msg += f"Strength is driven by **{top_3}**. "
    
    if drags:
        top_2 = ", ".join(drags[:2])
        msg += f"However, **{top_2}** are creating drag."
        
    return msg

def summarize_actionable_risk(indicators: Dict[str, Any]) -> str:
    atr = indicators.get('ATR (14)', {}).get('value') if indicators else None
    sl2 = indicators.get('Suggested SL (2xATR)', {}).get('value') if indicators else None
    atr_str = f"₹{float(atr):.2f}" if atr not in (None,'N/A') else "N/A"
    sl_str = _fmt_money(sl2) if sl2 not in (None,'N/A') else "None"
    return f"Volatility is moderate (~{atr_str} daily). Based on this, the 2×ATR stop-loss is {sl_str}."

def summarize_corporate_actions(actions: list) -> str:
    if not actions:
        return "No upcoming corporate actions."
    texts = []
    for a in actions:
        t = a.get("type","").replace("Upcoming ","")
        amt = a.get("amount") or a.get("ratio") or ""
        exd = a.get("ex_date") or ""
        if "Dividend" in t:
            texts.append(f"Upcoming dividend of ₹{amt} on {exd}.")
        elif "Bonus" in t:
            texts.append(f"Bonus issue {amt} effective {exd}.")
        elif "Split" in t:
            texts.append(f"Stock split {amt} on {exd}.")
        else:
            texts.append(f"{t} announced for {exd}.")
    return " ".join(texts)

def build_all_summaries(result: Dict[str, Any]) -> Dict[str, str]:
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
        "score": summarize_score_breakdown(
            prof.get("final_score", 0), 
            result.get("meta_scores", {}),
            prof
        ),
        "patterns": summarize_patterns(indicators),
        "pattern_details": get_active_pattern_details(indicators), # <--- NEW
        "strategy_details": STRATEGY_LIBRARY.get(best_strat.lower(), "Standard strategy analysis."), # <--- NEW
        "market": f"Market Trend: {result.get('macro_trend_status', 'Neutral')}",
        "risk": f"Suggested Stop Loss: {_fmt_money(tr.get('stop_loss'))} ({tr.get('execution_hints', {}).get('risk_note', 'Standard Risk')})"
    }