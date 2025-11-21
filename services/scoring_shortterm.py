# services/scoring_shortterm.py
from typing import Tuple, Dict, Any, Optional

from services.data_fetch import _to_float

def score_short_bull_run(indicators: Dict[str, Dict[str, Any]]) -> Tuple[int, str, list]:
    """
    Adaptive short-term swing scoring.
    Consumes output of services/indicators.py and produces:
        - Numerical confidence (0–100)
        - Signal label ("STRONG BUY", "BULLISH SWING", etc.)
        - List of reasoning lines
    """
    reasons = []
    score = 0.0
    total_weight = 0.0

    # --- Short-term focused weights ---
    weights = {
        "MACD Hist Z-Score": 1.8,        # Momentum conviction
        "Price Action": 1.5,             # Immediate buying pressure
        "RSI": 1.2,                      # Momentum health
        "Stoch %K": 0.7,                 # Reactive oscillator
        "MFI": 0.8,                      # Volume-weighted momentum
        "Short MA Cross (5/20)": 1.0,    # Quick trend reversal signal
        "Ichimoku Cloud": 0.9,           # Trend structure
        "Volume Spike Ratio": 0.8,       # Participation confirmation
        "200 DMA Trend (Slope)": 0.6,    # Long-term filter
        "BB Width": 0.4,                 # Volatility expansion
    }

    def get_score(key: str, default: int = 5) -> int:
        return indicators.get(key, {}).get("score", default)

    # --- Weighted aggregation ---
    for metric, weight in weights.items():
        m_score = get_score(metric)
        m_value = indicators.get(metric, {}).get("value")
        total_weight += 10 * weight
        score += m_score * weight

        if m_score == 10:
            reasons.append(f"{metric}: ✅ Strong ({m_value})")
        elif m_score == 0:
            reasons.append(f"{metric}: ⚠️ Weak ({m_value})")
        else:
            reasons.append(f"{metric}: Neutral ({m_value})")

    # --- ADX Context (non-weighted informational) ---
    adx_val = _to_float(indicators.get("ADX", {}).get("value"))
    if adx_val and adx_val > 25:
        reasons.append(f"ADX {round(adx_val,1)} → Trend strength confirmed.")

    # --- Normalize to 0–100 ---
    bull_score = int(max(0, min(100, (score / total_weight) * 100))) if total_weight else 0

    # --- Risk Vetoes (Safety filters) ---
    # RSI Overbought
    rsi_val = _to_float(indicators.get("RSI", {}).get("value"))
    if rsi_val and rsi_val > 70:
        bull_score = min(bull_score, 60)
        reasons.append(f"⚠️ RSI {round(rsi_val,1)} > 70 → Overbought, capping confidence to 60.")

    # MFI Overbought
    mfi_val = _to_float(indicators.get("MFI", {}).get("value"))
    if mfi_val and mfi_val > 80:
        bull_score = min(bull_score, 60)
        reasons.append(f"⚠️ MFI {round(mfi_val,1)} > 80 → Overbought (money flow exhaustion).")

    # Stochastic Overbought
    stoch_k = _to_float(indicators.get("Stoch %K", {}).get("value"))
    stoch_d = _to_float(indicators.get("Stoch %D", {}).get("value"))
    if stoch_k and stoch_d and stoch_k > 80 and stoch_d > 80:
        bull_score = min(bull_score, 60)
        reasons.append(f"⚠️ Stochastic {round(stoch_k,1)}/{round(stoch_d,1)} > 80 → Overbought, confidence capped.")

    # ATR Volatility Veto
    atr_pct = _to_float(indicators.get("ATR %", {}).get("value"))
    if atr_pct and atr_pct > 5.0:
        bull_score = max(0, bull_score - 20)
        reasons.append(f"⚠️ ATR% {round(atr_pct,1)} > 5 → High volatility, -20 penalty.")

    # Volatility Expansion Reward
    bb_width = _to_float(indicators.get("BB Width", {}).get("value"))
    if bb_width and bb_width > 6.0:
        bull_score = min(100, bull_score + 5)
        reasons.append(f"✅ BB Width {round(bb_width,2)} expanding → Breakout potential (+5).")

    # --- Final Signal Mapping ---
    if bull_score >= 80:
        signal = "STRONG BUY"
    elif bull_score >= 65:
        signal = "BULLISH SWING"
    elif bull_score >= 45:
        signal = "NEUTRAL"
    else:
        signal = "BEARISH / AVOID"

    return bull_score, signal, reasons
