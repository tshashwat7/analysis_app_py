# services/analyzers/strategy_analyzer.py
"""
Strategy Analyzer (Production Hardened + Pattern Aware)
-----------------
Produces strategy-fit checks & scores for:
- Swing, Day Trading, Trend Following, Momentum, Value, Income, Position Trading
- NEW: Minervini (VCP), CANSLIM (Growth + Cup)

Features:
- Consumes Pattern Engine results (Tier S & A Patterns)
- Centralized Threshold Configuration
- Robust Type Handling
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
    # NEW STRATEGIES
    "minervini": {"fit_thresh": 70}, # Stricter
    "canslim": {"fit_thresh": 65}
}

# -------------------------
# 2. Robust Helpers
# -------------------------
def _get_val(data, key, default=None):
    return data.get(key, default) if isinstance(data, dict) else default

def _get_str(data, key, default=""):
    v = _get_val(data, key, default)
    return str(v).lower() if v else default

def _is_squeeze_on(indicators):
    return "on" in _get_str(indicators, "ttm_squeeze")

def _get_ma_keys(horizon):
    # Dynamic key mapping based on horizon
    if horizon == "long_term": return {"fast": "wma_10", "mid": "wma_40", "slow": "wma_50"}
    if horizon == "multibagger": return {"fast": "mma_6", "mid": "mma_12", "slow": "mma_12"}
    return {"fast": "ema_20", "mid": "ema_50", "slow": "ema_200"}

def safe_get_numeric(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Robust numeric extractor: accepts raw scalar, dicts with value/raw/score, or missing.
    Prioritizes 'score' for Patterns, 'value' for Indicators.
    """
    if data is None: return default
    
    # Try getting the object
    v = _get_val(data, key, None)
    
    if v is None: return default
    
    # If it's a dict (Metric or Pattern result)
    if isinstance(v, dict):
        # 1. If it's a Pattern result, 'score' is the confidence (0-100)
        if "found" in v: 
            return float(v.get("score", 0.0))
            
        # 2. If it's a standard Metric
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
        "details": details 
    }

# -------------------------
# 3. Strategy Implementations
# -------------------------

def check_swing_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Swing: Catching reversals or dips in trend."""
    reasons, score = [], 0.0
    price = safe_get_numeric(indicators, "price")
    bb_low = safe_get_numeric(indicators, "bb_low")
    rsi = safe_get_numeric(indicators, "rsi", default=50)
    
    # --- PATTERN MUSCLE ---
    # 1. Double Bottom (Strong Reversal)
    # Using the 'score' from the pattern engine
    db_score = safe_get_numeric(indicators, "double_top_bottom", 0)
    
    # Check if it's actually bullish (Double Bottom) not Bearish (Double Top)
    is_double_bottom = False
    if db_score > 0 and isinstance(indicators.get("double_top_bottom"), dict):
        meta = indicators["double_top_bottom"].get("meta", {})
        if meta.get("type") == "bullish":
            is_double_bottom = True

    # --- SCORING ---
    # 1. Band Proximity
    if bb_low > 0 and price <= bb_low * 1.02:
        score += 35; reasons.append("Price near Buy Zone (BB Low)")
    
    # 2. RSI Dip
    if rsi <= 45: 
        score += 25; reasons.append("RSI Oversold/Dip")
    
    # 3. Pattern Confirmation (The "Trigger")
    if db_score > 0 and is_double_bottom:
        score += 40; reasons.append(f"Double Bottom Reversal ({int(db_score)}%)")
        
    # 4. Squeeze (Volatility Contraction often precedes swing moves)
    if _is_squeeze_on(indicators): score += 10
    
    return _build_result("swing", score, reasons, {"rsi": rsi, "double_bottom": is_double_bottom})

def check_day_trading_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    reasons, score = [], 0.0
    rvol = safe_get_numeric(indicators, "rvol", 1.0)
    atr_pct = safe_get_numeric(indicators, "atr_pct")
    
    # Patterns
    strike_score = safe_get_numeric(indicators, "three_line_strike", 0)
    
    # 1. Volatility & Liquidity
    if rvol > 1.5: score += 25
    if atr_pct > 1.5: score += 15
    
    # 2. Patterns (Triggers)
    if strike_score > 0: 
        score += 40; reasons.append("3-Line Strike Reversal")
        
    return _build_result("day_trading", score, reasons, {})

def check_trend_following_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None, horizon: str = "short_term") -> Dict[str, Any]:
    """Trend: MA Alignment, ADX, Ichimoku, Golden Cross"""
    reasons, score = [], 0.0
    
    ma_keys = _get_ma_keys(horizon)
    fast = safe_get_numeric(indicators, ma_keys["fast"])
    mid = safe_get_numeric(indicators, ma_keys["mid"])
    slow = safe_get_numeric(indicators, ma_keys["slow"])
    price = safe_get_numeric(indicators, "price")
    adx = safe_get_numeric(indicators, "adx")
    
    # Pattern Muscle
    ichi_score = safe_get_numeric(indicators, "ichimoku_signals", 0)
    golden_score = safe_get_numeric(indicators, "golden_cross", 0)

    # 1. MA Alignment
    if fast and mid and slow and price > fast > mid > slow:
        score += 30; reasons.append("Bullish MA Alignment")
        
    # 2. Trend Strength
    if adx > 25: score += 20; reasons.append("Strong Trend (ADX)")
    
    # 3. Ichimoku Cloud/Cross
    if ichi_score > 0:
        score += 25; reasons.append("Ichimoku Signal (Cloud/TK)")
        
    # 4. Golden Cross (Regime Confirmation)
    # Check if bullish
    if golden_score > 0:
        meta = indicators.get("golden_cross", {}).get("meta", {})
        if meta.get("type") == "bullish":
            score += 25; reasons.append("Golden Cross (Major Trend)")
    
    return _build_result("trend_following", score, reasons, {})

def check_momentum_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Momentum: Patterns + Indicators"""
    reasons, score = [], 0.0
    
    rsi = safe_get_numeric(indicators, "rsi")
    
    # Pattern Muscles
    darvas_score = safe_get_numeric(indicators, "darvas_box", 0)
    flag_score = safe_get_numeric(indicators, "flag_pennant", 0)
    squeeze_score = safe_get_numeric(indicators, "bollinger_squeeze", 0)
    
    if rsi >= 60: score += 20
    
    # Patterns are the key for momentum
    if darvas_score > 0: score += 30; reasons.append("Darvas Box Breakout")
    if flag_score > 0: score += 30; reasons.append("Bull Flag Continuation")
    if squeeze_score > 0: score += 20; reasons.append("Volatility Squeeze")
    
    return _build_result("momentum", score, reasons, {})

# --- NEW: MINERVINI VCP STRATEGY ---
def check_minervini_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """Minervini SEPA: Trend Template + VCP Pattern + Relative Strength."""
    reasons, score = [], 0.0
    
    # 1. Pattern Muscle
    vcp_score = safe_get_numeric(indicators, "minervini_stage2", 0)
    
    if vcp_score > 0:
        score += 50; reasons.append(f"VCP Pattern Confirmed ({int(vcp_score)}%)")
    else:
        # Fallback check for Trend Template
        ma_50 = safe_get_numeric(indicators, "ema_50")
        ma_200 = safe_get_numeric(indicators, "ema_200")
        price = safe_get_numeric(indicators, "price")
        
        if price > ma_50 > ma_200:
            score += 20; reasons.append("Stage 2 Trend Alignment")
        else:
            score -= 50; reasons.append("Not in Stage 2 Trend")

    # 2. Relative Strength
    rs_nifty = safe_get_numeric(indicators, "rel_strength_nifty")
    if rs_nifty > 0: score += 20; reasons.append("Outperforming Market")

    # 3. Near 52W Highs
    pos_52w = safe_get_numeric(fundamentals, "52w_position", 0)
    if pos_52w > 85: 
        score += 20; reasons.append("Near 52W Highs")
    elif pos_52w < 50:
        score -= 20; reasons.append("Deep in base")

    return _build_result("minervini", score, reasons, {"vcp": vcp_score, "rs": rs_nifty})

# --- NEW: CANSLIM STRATEGY ---
def check_canslim_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """CANSLIM: Earnings + Cup Pattern + Volume."""
    reasons, score = [], 0.0
    
    # C & A: Earnings
    q_growth = safe_get_numeric(fundamentals, "quarterly_growth", 0)
    a_growth = safe_get_numeric(fundamentals, "eps_growth_3y", 0)
    
    if q_growth > 20: score += 20; reasons.append("Strong Qtr Growth (>20%)")
    if a_growth > 15: score += 15; reasons.append("Strong Annual Growth")

    # N: New Pattern (Cup & Handle)
    cup_score = safe_get_numeric(indicators, "cup_handle", 0)
    pos_52w = safe_get_numeric(fundamentals, "52w_position", 0)
    
    if cup_score > 0:
        score += 30; reasons.append(f"Cup & Handle Pattern ({int(cup_score)}%)")
    elif pos_52w > 90:
        score += 15; reasons.append("Breaking to New Highs")

    # S: Volume
    rvol = safe_get_numeric(indicators, "rvol", 1.0)
    if rvol > 1.2: score += 10; reasons.append("Demand Volume")

    # L: Leader
    rs_nifty = safe_get_numeric(indicators, "rel_strength_nifty")
    if rs_nifty > 5: score += 15; reasons.append("Market Leader")

    return _build_result("canslim", score, reasons, {"cup": cup_score})

# --- LEGACY: VALUE & INCOME & POSITION (Unchanged logic, just simplified) ---

def check_value_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    reasons, score = [], 0.0
    pe = safe_get_numeric(fundamentals, "pe_ratio")
    pb = safe_get_numeric(fundamentals, "pb_ratio")
    if 0 < pe < 15: score += 35
    if 0 < pb < 1.5: score += 25
    return _build_result("value", score, reasons, {})

def check_income_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    reasons, score = [], 0.0
    dy = safe_get_numeric(fundamentals, "dividend_yield")
    if dy > 3.0: score += 40
    return _build_result("income", score, reasons, {})

def check_position_trading_fit(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    reasons, score = [], 0.0
    eps_g = safe_get_numeric(fundamentals, "eps_growth_5y")
    dma200 = safe_get_numeric(indicators, "ema_200")
    price = safe_get_numeric(indicators, "price")
    
    if eps_g > 10: score += 30
    if price > dma200: score += 40; reasons.append("Primary Uptrend")
    
    # Pattern Bonus: Golden Cross
    golden_score = safe_get_numeric(indicators, "golden_cross", 0)
    if golden_score > 0: score += 20; reasons.append("Golden Cross Confirmation")
    
    return _build_result("position_trading", score, reasons, {})

# -------------------------
# 4. Main Entry Point
# -------------------------
def analyze_strategies(indicators: Dict[str, Any], fundamentals: Dict[str, Any] = None, horizon: str = "short_term") -> Dict[str, Any]:
    """
    Analyzes all strategies including NEW Pattern-based ones.
    """
    strategies = [
        check_swing_fit, 
        check_day_trading_fit, 
        check_trend_following_fit,
        check_position_trading_fit, 
        check_momentum_fit, 
        check_value_fit, 
        check_income_fit,
        # --- NEW ---
        check_minervini_fit,
        check_canslim_fit
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
            
            if res["fit"]:
                fit_candidates.append(name)
            
            if res["score"] > best_score:
                best_score = res["score"]
                best_strat = name
                
        except Exception as e:
            logger.error(f"Strategy error {name}: {e}")
            results[name] = {"fit": False, "score": 0, "reasons": [f"Error: {str(e)}"], "details": {}}

    results["summary"] = {
        "best_strategy": best_strat, 
        "best_score": best_score,
        "all_fits": fit_candidates
    }
    return results