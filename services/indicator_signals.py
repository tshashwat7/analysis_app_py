from typing import Dict, Any, Optional
import logging
from services.data_fetch import _to_float

logger = logging.getLogger(__name__)

def compute_signals(indicators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes UNWEIGHTED Confidence (Consensus) and applies overrides 
    (ATR penalty, 200 DMA trend) to produce the final trading signal.
    """
    reasons = []
    score = 0
    total = 0
    penalty = 0

    def val(key):
        item = indicators.get(key)
        if not isinstance(item, dict):
            return None
        return item.get("value")

    def add_metric(points: int, reason: str):
        nonlocal score, total
        total += 10
        score += points
        reasons.append(reason)

    def retrieve_and_add_metric(key: str, reason_template: str):
        """Fetches score and adds formatted reason text."""
        item = indicators.get(key)
        if not item:
            return

        s = item.get("score")
        v = item.get("value")
        if s is None or (s == 0 and (v is None or str(v).upper() in ["N/A", "NONE", "NA"])):
            return

        status = "Bullish" if s == 10 else "Bearish" if s == 0 else "Neutral"
        add_metric(s, reason_template.format(status=status))

    # --- ATR Volatility Penalty ---
    atr_pct = val("atr_pct")  # now returns numeric 2.32 directly
    if atr_pct is not None and atr_pct > 3.0:
        penalty = int((atr_pct - 3.0) * 5)
        reasons.append(f"RISK ALERT: High ATR ({atr_pct:.2f}%) suggests elevated volatility risk.")


    # --- 200 DMA Trend (Slope) check ---
    dma_trend = str(val("dma_200_slope")).lower() if val("dma_200_slope") else "n/a"

    # =========================================================
    # 1. Metric Consumption (Unweighted)
    # =========================================================

    # RSI
    rsi = indicators.get("rsi")
    if rsi:
        rsi_score = rsi.get("score", 5)
        rsi_val = rsi.get("value")
        zone = rsi.get("zone", "Neutral")
        # Use the explicit zone in the reason, but still add the score properly
        add_metric(rsi_score, f"RSI ({rsi_val}): {zone}")

    # MACD Cross + Histogram Z (Logic retained, as it combines two scores)
    macd_cross = indicators.get("macd_cross", {}).get("score", 5)
    macd_z = indicators.get("macd_hist_z", {}).get("score", 5)
    final_macd = max(macd_cross, macd_z)
    if final_macd == 10:
        add_metric(10, "MACD: Bullish crossover or positive momentum")
    elif final_macd == 0:
        add_metric(0, "MACD: Bearish crossover or negative momentum")
    else:
        add_metric(5, "MACD: Neutral momentum")

    # Stochastic %K (Logic retained, as it extracts the crossover signal)
    stoch = indicators.get("stoch_k")
    if stoch:
        s_score = stoch.get("score", 5)
        stoch_cross = val("stoch_cross")
        if stoch_cross == "Bullish":
            reasons.append("SHORT-TERM ENTRY: Stochastic %K just crossed %D upward.")
        elif stoch_cross == "Bearish":
            reasons.append("SHORT-TERM EXIT: Stochastic %K just crossed %D downward.")
        add_metric(s_score, f"Stochastic %K: {('Bullish' if s_score == 10 else 'Bearish' if s_score == 0 else 'Neutral')} bias")

    # --- All other indicators ---
    retrieve_and_add_metric("price_vs_200dma_pct", "Price vs 200 DMA: {status}")
    retrieve_and_add_metric("adx", "ADX: {status} trend strength")
    retrieve_and_add_metric("vwap_bias", "VWAP: {status} intraday bias")
    retrieve_and_add_metric("vol_trend", "Volume Trend: {status}")
    retrieve_and_add_metric("rvol", "Relative Volume (RVOL): {status}")
    retrieve_and_add_metric("bb_low", "Bollinger Band Low: {status}")
    retrieve_and_add_metric("bb_width", "Bollinger Width: {status}")
    retrieve_and_add_metric("entry_confirm", "Entry Price Confirmation: {status}")
    retrieve_and_add_metric("dma_20_50_cross", "20/50 DMA Crossover: {status}")
    retrieve_and_add_metric("dma_200_slope", "200 DMA Slope: {status}")
    retrieve_and_add_metric("ichi_cloud", "Ichimoku Cloud: {status}")
    retrieve_and_add_metric("obv_div", "OBV Divergence: {status}")
    retrieve_and_add_metric("atr_14", "ATR Volatility: {status}")
    retrieve_and_add_metric("vol_spike_ratio", "Volume Spike Ratio: {status}")
    retrieve_and_add_metric("rel_strength_nifty", "Relative Strength vs NIFTY: {status}")
    retrieve_and_add_metric("price_action", "Price Action: {status}")

    # =========================================================
    # 2. Final Signal Mapping & Overrides
    # =========================================================  
    confidence = int((score / total) * 100) if total > 0 else 0

    # Apply volatility penalty
    if penalty > 0:
        confidence = max(20, confidence - int(penalty))
        reasons.append(f"Global penalty: -{penalty}% due to volatility risk")

    # Map confidence → signal
    if confidence >= 70:
        signal = "BUY"
    elif confidence >= 40:
        signal = "HOLD"
    else:
        signal = "SELL"

    # RSI override
    rsi_zone = rsi.get("zone", "Neutral") if rsi else "Neutral"
    if confidence >= 60 and "Overbought" in rsi_zone:
        signal = "HOLD"
        reasons.append("Override: Overbought RSI → HOLD bias")

    # 200 DMA slope override
    if dma_trend == "falling":
        if signal == "BUY":
            signal = "HOLD"
            confidence = max(40, confidence - 20)
            reasons.append("200 DMA Falling → BUY→HOLD downgrade")
        elif signal == "HOLD":
            signal = "SELL"
            confidence = max(20, confidence - 10)
            reasons.append("200 DMA Falling → HOLD→SELL downgrade")

    return {
        "signal": signal,
        "confidence": max(0, min(100, confidence)),
        "reasons": reasons
    }
