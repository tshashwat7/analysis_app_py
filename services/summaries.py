# services/summaries.py
from typing import Dict, Any, Optional
from datetime import datetime

def _fmt_money(v: Optional[float]) -> str:
    try:
        return f"₹{float(v):,.2f}"
    except Exception:
        return "None"

def summarize_trade_recommendation(tr: Dict[str, Any]) -> str:
    if not tr:
        return "No trade recommendation available."
    rec = (tr.get("recommendation") or "").upper()
    reason = tr.get("reason") or ""
    entry = tr.get("entry_price_used")
    target = tr.get("target_price")
    sl = tr.get("suggested_sl")

    if rec in ("BUY", "STRONG BUY"):
        msg = f"The model suggests a **{rec}**, supported by strong momentum signals."
    elif rec in ("SELL", "STRONG SELL"):
        msg = f"The model suggests a **{rec}**, as the long-term trend indicates weakness."
    else:
        msg = f"The model gives a **{rec or 'neutral'}** stance, awaiting confirmation."

    if reason:
        msg += f" Reason: {reason.strip()}."

    if entry:
        msg += f" Entry zone near {_fmt_money(entry)}."
    if target:
        msg += f" Target around {_fmt_money(target)}."
    if sl:
        msg += f" Stop-loss near {_fmt_money(sl)}."
    if not any([entry, target, sl]):
        msg += " No valid entry, target, or stop levels detected yet."
    return msg

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

#
# --- REPLACE this function in summaries.py ---
#
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

#
# --- REPLACE this function in summaries.py ---
#
def summarize_score_breakdown(final_score: float, meta_scores: Dict[str, float]) -> str:
    """
    Summarizes the stock's archetype based on its meta-scores.
    """
    try:
        if not meta_scores:
            return "Score breakdown is not available."

        # Find the highest and lowest scoring archetypes
        best_archetype = max(meta_scores, key=meta_scores.get)
        best_score = meta_scores[best_archetype]
        
        worst_archetype = min(meta_scores, key=meta_scores.get)
        worst_score = meta_scores[worst_archetype]

        msg = f"This stock's **Final Score is {final_score:.1f}/10**. "
        
        # Describe the stock based on its strongest and weakest points
        if best_score >= 7.5:
            msg += f"Its primary strength is **{best_archetype.title()}** (Score: {best_score:.1f}). "
        else:
            msg += "It shows a balanced profile. "

        if worst_score <= 3.0:
            msg += f"Its main weakness is **{worst_archetype.title()}** (Score: {worst_score:.1f})."
        
        return msg
        
    except Exception as e:
        logger.warning(f"Error in summarize_score_breakdown: {e}")
        return "Could not generate score summary."

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
    summaries = {
        "trade": summarize_trade_recommendation(tr),
        "risk": summarize_risk_execution(tr),
        "entry_stop": summarize_entry_stop(indicators),
        "market": summarize_market_status(
            result.get("macro_trend_status"),
            result.get("macro_close"),
            result.get("macro_index_name")
        ),
        "score": summarize_score_breakdown(
            result.get("profile_report", {}).get("final_score", 0), # Get the real final score
            result.get("meta_scores", {})                          # Pass the archetype scores
        ),
        "actionable_risk": summarize_actionable_risk(indicators),
        "corporate": summarize_corporate_actions(result.get("corporate_actions") or []),
    }
    return summaries
