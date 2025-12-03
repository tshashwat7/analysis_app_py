# services/strategy_analyzer.py
"""
Strategy Analyzer (Production Hardened)
-----------------
Produces strategy-fit checks & scores for:
- Swing, Day Trading, Trend Following, Momentum, Value, Income, Position Trading

Features:
- Centralized Threshold Configuration
- Robust Type Handling (Dict vs Scalar)
- Detailed "Reasoning" and "Raw Data" returns for UI
- Wick Rejection Integration
"""

from typing import Dict, Any, List, Optional
import math
import logging

logger = logging.getLogger(__name__)

# -------------------------
# 1. Configuration (Centralized Tuning)
# -------------------------
STRATEGY_CONFIG = {
    "swing": {"fit_thresh": 50},
    "day_trading": {"fit_thresh": 60},
    "trend_following": {"fit_thresh": 60},
    "position_trading": {"fit_thresh": 65},
    "momentum": {"fit_thresh": 60},
    "value": {"fit_thresh": 50},
    "income": {"fit_thresh": 55},
}

# -------------------------
# 2. Robust Helpers
# -------------------------
try:
    from services.signal_engine import _get_val, _get_str, _is_squeeze_on, _get_ma_keys
except Exception:
    # Minimal fallbacks
    def _get_val(data, key, default=None):
        return data.get(key, default) if isinstance(data, dict) else default
    
    def _get_str(data, key, default=""):
        v = _get_val(data, key, default)
        return str(v).lower() if v else default

    def _is_squeeze_on(indicators):
        return "on" in _get_str(indicators, "ttm_squeeze")

    def _get_ma_keys(horizon):
        return {"fast": "ema_20", "mid": "ema_50", "slow": "ema_200"}

def safe_get_numeric(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Robust numeric extractor: accepts raw scalar, dicts with value/raw/score, or missing.
    Defaults to 0.0 to prevent NoneType comparison errors in logic.
    """
    if data is None: return default
    
    # Try getting the object
    v = _get_val(data, key, None)
    
    if v is None: return default
    
    # If it's a dict, hunt for the value
    if isinstance(v, dict):
        for k in ("value", "raw", "score"):
            if k in v and v[k] is not None:
                try: return float(v[k])
                except: pass
        return default
    
    # If it's a scalar
    try: return float(v)
    except: return default

def _norm_score(x: float) -> float:
    return max(0.0, min(100.0, float(x) if x else 0.0))

def _build_result(name, score, reasons, details):
    thresh = STRATEGY_CONFIG.get(name, {}).get("fit_thresh", 50)
    final_score = _norm_score(score)
    return {
        "strategy": name,
        "fit": final_score >= thresh,
        "score": round(final_score, 1),
        "reasons": reasons,
        "details": details # For UI Debugging
    }

# -------------------------
# 3. Strategy Implementations
# -------------------------

def check_swing_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Swing: Bollinger Band Mean Reversion + RSI + Volume + Wick Rejection"""
    reasons, score = [], 0.0
    
    # Fetch Data
    price = safe_get_numeric(indicators, "price")
    bb_low = safe_get_numeric(indicators, "bb_low")
    bb_high = safe_get_numeric(indicators, "bb_high")
    rsi = safe_get_numeric(indicators, "rsi", default=50)
    rvol = safe_get_numeric(indicators, "rvol", default=1.0)
    # [NEW] Wick Rejection (Lower is better for Bullish breakouts, but High is bad for Longs)
    wick_ratio = safe_get_numeric(indicators, "wick_rejection", default=0.0) 
    
    # 1. Band Proximity (Reversion)
    if bb_low > 0 and price <= bb_low * 1.02:
        score += 35; reasons.append("Price near lower BB (Buy Zone)")
    elif bb_high > 0 and price >= bb_high * 0.98:
        score += 20; reasons.append("Price near upper BB (Sell Zone)")
        
    # 2. RSI Regime (30-50 for dips)
    if rsi <= 45: score += 25; reasons.append("RSI suppressed (Setup for bounce)")
    elif 45 < rsi < 60: score += 10; reasons.append("RSI Neutral")
    else: score -= 5; reasons.append("RSI High (Momentum, not Swing)")
        
    # 3. Volume & Squeeze
    if rvol > 1.2: score += 10; reasons.append("Elevated Volume")
    if _is_squeeze_on(indicators): score += 20; reasons.append("Volatility Squeeze Active")

    # 4. [NEW] Wick Rejection Logic
    # If we are looking to buy (price low), a high UPPER wick is bad (selling pressure).
    # Wick Ratio > 2.0 = "Shooting Star" shape (Bad for long)
    if wick_ratio > 2.0:
        score -= 15; reasons.append(f"High Rejection Wick ({wick_ratio:.1f}x) - Risk")
    elif wick_ratio < 0.5:
        score += 5; reasons.append("Strong Close (Low Wick)")

    details = {"price": price, "bb_low": bb_low, "rsi": rsi, "wick": wick_ratio}
    return _build_result("swing", score, reasons, details)


def check_day_trading_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Day Trading: Volatility (ATR/Beta), Liquidity (RVOL), VWAP"""
    reasons, score = [], 0.0
    
    rvol = safe_get_numeric(indicators, "rvol", 1.0)
    vwap = safe_get_numeric(indicators, "vwap")
    price = safe_get_numeric(indicators, "price")
    atr_pct = safe_get_numeric(indicators, "atr_pct")
    beta = safe_get_numeric(fundamentals, "beta", 1.0)
    gap = safe_get_numeric(indicators, "gap_percent", 0.0)
    
    # 1. Volatility
    if beta > 1.2: score += 10; reasons.append(f"High Beta {beta:.2f}")
    if abs(gap) > 1.0: score += 15; reasons.append(f"Gap Play ({gap:.1f}%)")
    
    if atr_pct > 2.0: score += 15; reasons.append("High Intraday Range (ATR%)")
    elif atr_pct < 0.8: score -= 20; reasons.append("Stock too dead (Low ATR)")
        
    # 2. Liquidity (Critical)
    if rvol > 2.0: score += 30; reasons.append("Explosive Relative Volume (>2x)")
    elif rvol > 1.2: score += 15
    else: score -= 10
        
    # 3. Trend Alignment
    if vwap > 0:
        if price > vwap: score += 20; reasons.append("Above VWAP (Intraday Bull)")
        elif price < vwap: score += 5; reasons.append("Below VWAP (Intraday Bear)")
        
    details = {"rvol": rvol, "atr_pct": atr_pct, "beta": beta, "gap": gap}
    return _build_result("day_trading", score, reasons, details)


def check_trend_following_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None, horizon: str = "short_term") -> Dict[str, Any]:
    """Trend: MA Alignment, ADX, Ichimoku, DMI"""
    reasons, score = [], 0.0
    
    ma_keys = _get_ma_keys(horizon)
    fast = safe_get_numeric(indicators, ma_keys["fast"])
    mid = safe_get_numeric(indicators, ma_keys["mid"])
    slow = safe_get_numeric(indicators, ma_keys["slow"])
    price = safe_get_numeric(indicators, "price")
    
    adx = safe_get_numeric(indicators, "adx")
    di_plus = safe_get_numeric(indicators, "di_plus")
    di_minus = safe_get_numeric(indicators, "di_minus")
    ichi = _get_str(indicators, "ichi_cloud")

    # 1. MA Alignment
    if fast and mid and slow:
        if price > fast > mid > slow: score += 30; reasons.append("Perfect Bullish Alignment")
        elif fast > slow: score += 10; reasons.append("Golden Cross State")
            
    # 2. Trend Strength
    if adx > 25: score += 20; reasons.append("Strong Trend (ADX>25)")
    
    # 3. Directional Movement (DMI)
    if di_plus > 0 and di_minus > 0:
        if di_plus > di_minus: score += 10; reasons.append("DI+ > DI-")
        else: score -= 5
    
    # 4. Ichimoku Safe Check
    if "bull" in ichi: score += 15; reasons.append("Ichimoku Cloud Bullish")
    
    details = {"adx": adx, "alignment": f"{int(fast)}/{int(mid)}/{int(slow)}"}
    return _build_result("trend_following", score, reasons, details)


def check_position_trading_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Position: Hybrid (Fundamental Growth + Long Term Technical Trend)"""
    reasons, score = [], 0.0
    
    roe = safe_get_numeric(fundamentals, "roe")
    eps_g = safe_get_numeric(fundamentals, "eps_growth_5y")
    pe = safe_get_numeric(fundamentals, "pe_ratio")
    
    price = safe_get_numeric(indicators, "price")
    dma200 = safe_get_numeric(indicators, "ema_200") # Proxy
    
    if roe > 15: score += 15; reasons.append("High ROE (>15%)")
    if eps_g > 10: score += 15; reasons.append("Consistent Earnings Growth")
    if 0 < pe < 30: score += 10; reasons.append("Reasonable Valuation")
    
    if dma200 > 0 and price > dma200: 
        score += 40; reasons.append("Above 200 DMA (Primary Uptrend)")
            
    details = {"roe": roe, "eps_g": eps_g, "pe": pe}
    return _build_result("position_trading", score, reasons, details)


def check_momentum_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Momentum: High RSI, Rising MACD, Relative Strength"""
    reasons, score = [], 0.0
    
    rsi = safe_get_numeric(indicators, "rsi")
    macd_hist = safe_get_numeric(indicators, "macd_histogram")
    rsi_slope = safe_get_numeric(indicators, "rsi_slope")
    rel_strength = safe_get_numeric(indicators, "rel_strength_nifty")
    
    if rsi >= 60: score += 25; reasons.append("Strong RSI (>60)")
    if macd_hist > 0: score += 20; reasons.append("Positive Momentum (MACD Hist > 0)")
    if rsi_slope > 0: score += 15; reasons.append("Accelerating Momentum")
    if rel_strength > 0: score += 20; reasons.append("Outperforming Benchmark")
    
    details = {"rsi": rsi, "macd_h": macd_hist, "rs_nifty": rel_strength}
    return _build_result("momentum", score, reasons, details)


def check_value_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Value: Low P/E, Low P/B, FCF"""
    reasons, score = [], 0.0
    
    pe = safe_get_numeric(fundamentals, "pe_ratio")
    pb = safe_get_numeric(fundamentals, "pb_ratio")
    fcf = safe_get_numeric(fundamentals, "fcf_yield")
    
    if 0 < pe < 15: score += 35; reasons.append("Undervalued P/E (<15)")
    if 0 < pb < 1.5: score += 25; reasons.append("Low P/B (<1.5)")
    if fcf > 5: score += 20; reasons.append("High FCF Yield (>5%)")
    
    details = {"pe": pe, "pb": pb, "fcf": fcf}
    return _build_result("value", score, reasons, details)


def check_income_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Income: Dividend Yield, Payout Safety"""
    reasons, score = [], 0.0
    
    dy = safe_get_numeric(fundamentals, "dividend_yield")
    payout = safe_get_numeric(fundamentals, "payout_ratio")
    
    if dy > 4.0: score += 40; reasons.append("High Yield (>4%)")
    elif dy > 2.0: score += 20
    
    if 0 < payout < 60: score += 30; reasons.append("Safe Payout Ratio (<60%)")
    elif payout > 90: score -= 20; reasons.append("Unsafe Payout Ratio (>90%)")
    
    details = {"yield": dy, "payout": payout}
    return _build_result("income", score, reasons, details)


# -------------------------
# 4. Main Entry Point
# -------------------------
def analyze_strategies(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None, horizon: str = "short_term") -> Dict[str, Any]:
    """
    Analyzes all strategies and returns a report with:
    - Individual strategy details
    - A 'summary' object with best fit and ALL fitting candidates.
    """
    strategies = [
        check_swing_fit, check_day_trading_fit, check_trend_following_fit,
        check_position_trading_fit, check_momentum_fit, check_value_fit, check_income_fit
    ]
    
    results = {}
    best_score, best_strat = -1, None
    fit_candidates = []
    
    for strategy_fn in strategies:
        name = "unknown"
        try:
            # Handle horizon arg
            if strategy_fn.__name__ == "check_trend_following_fit":
                res = strategy_fn(indicators, fundamentals, horizon=horizon)
            else:
                res = strategy_fn(indicators, fundamentals)
            
            name = res["strategy"]
            results[name] = res
            
            # Track Fits
            if res["fit"]:
                fit_candidates.append(name)
            
            # Track Best
            if res["score"] > best_score:
                best_score = res["score"]
                best_strat = name
                
        except Exception as e:
            logger.error(f"Strategy error {name}: {e}")
            results[name] = {"fit": False, "score": 0, "reasons": [f"Error: {str(e)}"], "details": {}}

    results["summary"] = {
        "best_strategy": best_strat, 
        "best_score": best_score,
        "all_fits": fit_candidates # [NEW] For UI filtering
    }
    return results