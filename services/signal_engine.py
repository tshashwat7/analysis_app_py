# services/signal_engine.py
"""
Signal Engine v10.6 (Golden Master)
- BASE: V1 Robustness (Strict scoring, Hybrid Metrics, Penalty Logic).
- FEATURE: V2 Trend Surfing (Supertrend Value Trailing, ADX Filters).
- RESTORED: PSAR Checks, RR Validation, Execution Hints.
- TUNED: Auto-Tuning for Volatility (Dynamic Stops & Dynamic Confidence).
- REFINED: Multi-path Entry Logic (Breakout OR Strong Trend OR PSAR).
- ARCHITECTURE: Handles both Flat and Nested Indicator dictionaries.
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import statistics
import operator
import logging
from datetime import datetime

from services.data_fetch import _safe_float

# Config / constants 
from config.constants import (
    HORIZON_PROFILE_MAP,
    VALUE_WEIGHTS,
    GROWTH_WEIGHTS,
    QUALITY_WEIGHTS,
    MOMENTUM_WEIGHTS,
    ENABLE_VOLATILITY_QUALITY,
    MACD_MOMENTUM_THRESH,
    RSI_SLOPE_THRESH,
    TREND_THRESH,
    VOL_BANDS,
    ATR_MULTIPLIERS,
)
from services.tradeplan.time_estimator import estimate_hold_time

logger = logging.getLogger(__name__)

MISSING_KEYS = set()
STRATEGY_TIME_MULTIPLIERS = {
    "momentum": 0.7,        # Moves fast
    "day_trading": 0.5,     # Very fast
    "swing": 1.0,           # Standard
    "trend_following": 1.2, # Trends take time
    "position_trading": 1.5,# Slow grind
    "value": 1.5,           # Market takes time to recognize value
    "income": 2.0,          # Holding forever
    "unknown": 1.0
}
# -------------------------
# Helpers
# -------------------------
def ensure_numeric(x, default=0.0):
    """Safely coerce any input (dict/str/None) to float."""
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, dict):
            # Prioritize 'value', then 'raw', then 'score'
            return float(x.get("value") or x.get("raw") or x.get("score") or default)
        # Handle string percentages "12.5%"
        if isinstance(x, str):
            clean = x.replace("%", "").replace(",", "").strip()
            return float(clean) if clean else float(default)
        return float(default)
    except Exception:
        return float(default)

# -------------------------
# Dynamic MA Configuration
# -------------------------
MA_KEYS_BY_HORIZON = {
    "intraday":    {"fast": "ema_20", "mid": "ema_50", "slow": "ema_200", "trend": "ema_20_50_200_trend"},
    "short_term":  {"fast": "ema_20", "mid": "ema_50", "slow": "ema_200", "trend": "ema_20_50_200_trend"},
    "long_term":   {"fast": "wma_10", "mid": "wma_40", "slow": "wma_50",  "trend": "wma_10_40_50_trend"},
    "multibagger": {"fast": "mma_6",  "mid": "mma_12", "slow": "mma_12",  "trend": "mma_6_12_12_trend"} 
}

def _reset_missing_keys():
    """Clear missing metric tracker before each compute cycle."""
    MISSING_KEYS.clear()

def _get_ma_keys(horizon: str) -> Dict[str, str]:
    """Return the correct MA keys for the given horizon."""
    return MA_KEYS_BY_HORIZON.get(horizon, MA_KEYS_BY_HORIZON["short_term"])

# -------------------------
# Unified Data Accessors (Restored V1 Robustness)
# -------------------------
def _get_val(data: Dict[str, Any], key: str, default: Any = None, missing_log: set = None):
    if missing_log is None: missing_log = MISSING_KEYS
    if not data or key not in data:
        missing_log.add(key)
        return default
    entry = data[key]
    if isinstance(entry, (int, float)): return float(entry)
    if isinstance(entry, dict):
        val = entry.get("value") or entry.get("raw")
        if val is None:
            missing_log.add(key)
            return default
        return _safe_float(val)
    # fallback
    if entry is None:
        missing_log.add(key)
        return default
    return _safe_float(entry)

def _get_str(data: Dict[str, Any], key: str, default: str = "") -> str:
    if not data or key not in data:
        return default
    entry = data[key]
    if isinstance(entry, str):
        return entry.lower()
    if isinstance(entry, dict):
        val = entry.get("value") or entry.get("raw") or entry.get("desc")
        return str(val).lower() if val else default
    return str(entry).lower()

# --- Dynamic Fallback Helpers ---

def _get_slow_ma(indicators: Dict[str, Any], horizon: str) -> Optional[float]:
    """Tries the specific horizon slow MA first, then falls back to common long MAs."""
    keys = _get_ma_keys(horizon)
    val = _get_val(indicators, keys["slow"])
    if val is not None:
        return val
    return (_get_val(indicators, "ema_200") or 
            _get_val(indicators, "dma_200") or 
            _get_val(indicators, "wma_50") or 
            _get_val(indicators, "mma_12"))

def _get_dynamic_slope(indicators: Dict[str, Any], horizon: str) -> Optional[float]:
    """Tries horizon-specific slope, then generic fallbacks."""
    pref_key = {
        "intraday": "ema_20_slope",
        "short_term": "ema_20_slope", 
        "long_term": "wma_50_slope", # Based on your logs
        "multibagger": "dma_200_slope" # Fallback
    }.get(horizon)
    
    # Try preferred key
    val = _get_val(indicators, pref_key) if pref_key else None
    if val is not None: return val
        
    # Fallback list
    fallback_keys = ["ema_slope", "ema_20_slope", "wma_50_slope", "wma_20_slope", "dma_200_slope"]
    for k in fallback_keys:
        v = _get_val(indicators, k)
        if v is not None: return v
    return None

def _is_squeeze_on(indicators: Dict[str, Any]) -> bool:
    """Robust squeeze detection."""
    val = _get_str(indicators, "ttm_squeeze")
    return any(x in val for x in ("on", "sqz", "squeeze_on", "squeeze on"))

# -------------------------
# Score Coercion & Rules (Restored V1)
# -------------------------
def _coerce_score_field(metric_entry: Any) -> Optional[float]:
    if not metric_entry: return None
    if isinstance(metric_entry, dict):
        s = metric_entry.get("score")
        if s is not None: return _safe_float(s)
        raw = metric_entry.get("raw")
        rv = _safe_float(raw)
        if rv is not None and 0 <= rv <= 10: return rv
    elif isinstance(metric_entry, (int, float)):
        return float(metric_entry)
    return None

_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    "in": lambda a, b: a in b if b is not None else False,
}

def _rule_matches(raw_val, op: str, tgt) -> bool:
    if raw_val is None: return False
    if op == "in":
        try: return raw_val in tgt
        except: return False
    try:
        if isinstance(raw_val, (int, float)):
            return _OPS[op](float(raw_val), float(tgt))
        return _OPS[op](raw_val, tgt)
    except: return False

# -------------------------
# Hybrid Metrics (Restored V1 Full Suite)
# -------------------------
def enrich_hybrid_metrics(fundamentals: dict, indicators: dict) -> dict:
    hybrids = {}
    roe = _get_val(fundamentals, "roe")
    atr_pct = _get_val(indicators, "atr_pct")
    
    # 1. Volatility-Adjusted ROE
    if roe is not None and atr_pct is not None and atr_pct > 0:
        ratio = roe / atr_pct
        score = 10 if ratio >= 10 else 7 if ratio >= 5 else 3 if ratio >= 2 else 0
        hybrids["volatility_adjusted_roe"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"ROE/Vol = {ratio:.2f}", "alias": "Volatility-Adjusted ROE", "source": "hybrid"
        }

    # 2. Price vs Intrinsic Value
    pe = _get_val(fundamentals, "pe_ratio")
    eps_growth = _get_val(fundamentals, "eps_growth_5y")
    price = _get_val(indicators, "price") or _get_val(fundamentals, "current_price")
    
    if price and pe and eps_growth and eps_growth > 0:
        try:
            intrinsic_value = price * (1 / (pe / eps_growth))
            ratio = price / intrinsic_value if intrinsic_value != 0 else 999.0
            score = 10 if ratio < 0.8 else 7 if ratio < 1.0 else 3 if ratio < 1.2 else 0
            hybrids["price_vs_intrinsic_value"] = {
                "raw": ratio, "value": round(ratio, 2), "score": score,
                "desc": f"Price/IV = {ratio:.2f}", "alias": "Price vs Intrinsic Value", "source": "hybrid"
            }
        except: pass

    # 3. FCF Yield vs Volatility
    fcf_yield = _get_val(fundamentals, "fcf_yield")
    if fcf_yield is not None and atr_pct is not None:
        ratio = fcf_yield / max(atr_pct, 0.1)
        score = 10 if ratio >= 10 else 8 if ratio >= 5 else 5 if ratio >= 2 else 2
        hybrids["fcf_yield_vs_volatility"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"FCF/Vol = {ratio:.2f}", "alias": "FCF Yield vs Volatility", "source": "hybrid"
        }

    # 4. Trend Consistency
    adx = _get_val(indicators, "adx")
    supertrend = _get_str(indicators, "supertrend_signal")
    if adx is not None:
        score = 10 if adx >= 25 else 7 if adx >= 20 else 4
        if "bull" in supertrend: score = min(10, score + 1)
        hybrids["trend_consistency"] = {
            "raw": adx, "value": adx, "score": min(10, score),
            "desc": f"ADX {adx:.1f}", "alias": "Trend Consistency", "source": "hybrid"
        }

    # 5. Price vs 200DMA
    dma_200 = _get_val(indicators, "dma_200") or _get_val(indicators, "ema_200")
    if price and dma_200:
        ratio = (price / dma_200) - 1
        score = 10 if ratio > 0.1 else 7 if ratio > 0.0 else 3 if ratio > -0.05 else 0
        hybrids["price_vs_200dma_pct"] = {
            "raw": ratio, "value": round(ratio * 100, 2), "score": score,
            "desc": f"Price vs 200DMA: {ratio*100:.2f}%", "alias": "Price vs 200 DMA (%)", "source": "hybrid"
        }

    # 6. Fundamental Momentum
    q_growth = _get_val(fundamentals, "quarterly_growth")
    eps_5y = _get_val(fundamentals, "eps_growth_5y")
    if q_growth is not None and eps_5y is not None:
        ratio = (q_growth + eps_5y / 5) / 2
        score = 10 if ratio >= 15 else 7 if ratio >= 10 else 4 if ratio >= 5 else 1
        hybrids["fundamental_momentum"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"Growth Mom = {ratio:.2f}%", "alias": "Fundamental Momentum", "source": "hybrid"
        }

    # 7. Earnings Consistency
    net_margin = _get_val(fundamentals, "net_profit_margin")
    if roe is not None and net_margin is not None:
        ratio = (roe + net_margin) / 2
        score = 10 if ratio >= 25 else 7 if ratio >= 15 else 4 if ratio >= 8 else 1
        hybrids["earnings_consistency_index"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"Consistency = {ratio:.2f}", "alias": "Earnings Consistency Index", "source": "hybrid"
        }

    return hybrids

# -------------------------
# Composite Calculations (Restored V1 + Slope Fixes)
# -------------------------
def compute_trend_strength(indicators: Dict[str, Any], horizon: str = "short_term") -> Dict[str, Any]:
    try:
        adx = _get_val(indicators, "adx")
        ema_slope = _get_dynamic_slope(indicators, horizon)
        # Check raw slope if dynamic lookup returned simple float
        if ema_slope is None:
            # Fallback to look inside dict if key exists but _get_val returned float
            # Not needed if _get_val handles dict extraction properly
            pass

        di_plus = _get_val(indicators, "di_plus")
        di_minus = _get_val(indicators, "di_minus")
        supertrend = _get_str(indicators, "supertrend_signal")

        adx_score = 0.0
        if adx is not None:
            if adx >= TREND_THRESH["strong_floor"]: adx_score = 10.0
            elif adx >= TREND_THRESH["moderate_floor"]: adx_score = 8.0
            elif adx >= TREND_THRESH["weak_floor"]: adx_score = 4.0
            else: adx_score = 2.0

        # RESTORED: String check for "Rising" descriptions + Numeric check
        ema_score = 0.0
        if ema_slope is not None:
            # If slope is passed as a string (e.g. from an indicator description), parse it
            if isinstance(ema_slope, str) and "rising" in ema_slope.lower():
                ema_score = 10.0
            else:
                # Numeric Logic
                try:
                    v = abs(float(ema_slope))
                    if v >= 2.0: ema_score = 10.0
                    elif v >= 1.0: ema_score = 8.0
                    elif v >= 0.5: ema_score = 6.0
                    elif v >= 0.2: ema_score = 4.0
                    else: ema_score = 2.0
                except ValueError:
                    pass

        di_score = 0.0
        if di_plus is not None and di_minus is not None:
            spread = di_plus - di_minus
            if spread >= TREND_THRESH["di_spread_strong"]: di_score = 10.0
            elif spread >= 10: di_score = 7.0
            elif spread >= 0: di_score = 5.0
            else: di_score = 2.0

        st_score = 0.0
        if "bull" in supertrend: st_score = 10.0
        elif "bear" in supertrend: st_score = 0.0

        components = [
            ("adx", adx_score, 0.4),
            ("ema", ema_score, 0.3),
            ("di", di_score, 0.2),
            ("st", st_score, 0.1),
        ]
        
        raw = sum(val * w for (_, val, w) in components)
        score = max(0.0, min(10.0, round(raw, 2)))
        desc = f"Trend Strength (ADX {adx})"
        return {"raw": raw, "value": score, "score": int(round(score)), "desc": desc, "alias": "trend_strength", "source": "composite"}
    except Exception as e:
        logger.debug("Trend strength compute error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "trend_strength", "source": "composite"}

def compute_momentum_strength(indicators: Dict[str, Any]) -> Dict[str, Any]:
    try:
        rsi = _get_val(indicators, "rsi")
        rsi_slope = _get_val(indicators, "rsi_slope")
        macd_hist = _get_val(indicators, "macd_histogram")
        stoch_k = _get_val(indicators, "stoch_k")
        stoch_d = _get_val(indicators, "stoch_d")
        
        rsi_score = 0.0
        if rsi is not None:
            if rsi >= 70: rsi_score = 8.0
            elif rsi >= 60: rsi_score = 7.0
            elif rsi >= 50: rsi_score = 5.0
            elif rsi >= 40: rsi_score = 4.0
            else: rsi_score = 2.0

        slope_score = 0.0
        if rsi_slope is not None:
            if rsi_slope >= RSI_SLOPE_THRESH["acceleration_floor"]: slope_score = 8.0
            elif rsi_slope >= 0.0: slope_score = 4.0
            elif rsi_slope <= RSI_SLOPE_THRESH["deceleration_ceiling"]: slope_score = 2.0
            else: slope_score = 3.0

        macd_score = 0.0
        if macd_hist is not None:
            if macd_hist >= MACD_MOMENTUM_THRESH["acceleration_floor"]: macd_score = 8.0
            elif macd_hist <= MACD_MOMENTUM_THRESH["deceleration_ceiling"]: macd_score = 2.0
            else: macd_score = 5.0

        stoch_score = 0.0
        if stoch_k is not None and stoch_d is not None:
            if stoch_k > stoch_d and stoch_k >= 50: stoch_score = 8.0
            elif stoch_k > stoch_d: stoch_score = 6.0
            elif stoch_k < stoch_d: stoch_score = 3.0

        raw = (rsi_score * 0.25) + (slope_score * 0.25) + (macd_score * 0.30) + (stoch_score * 0.20)
        score = max(0.0, min(10.0, round(raw, 2)))
        desc = f"Momentum (RSI {rsi}, MACD_hist {macd_hist})"
        return {"raw": raw, "value": score, "score": int(round(score)), "desc": desc, "alias": "momentum_strength", "source": "composite"}
    except Exception as e:
        logger.debug("Momentum compute error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "momentum_strength", "source": "composite"}

def compute_roe_stability(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    try:
        history = fundamentals.get("roe_history")
        vals = []
        if isinstance(history, list) and len(history) >= 3:
            vals = [v for v in ([_safe_float(x) for x in history]) if v is not None]
        else:
            r5 = fundamentals.get("roe_5y")
            if isinstance(r5, dict):
                vals = [v for v in (_safe_float(x) for x in r5.values()) if v is not None]

        if not vals:
            return {"raw": None, "value": None, "score": None, "desc": "No ROE history", "alias": "roe_stability", "source": "composite"}

        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        if std < 2.0: score = 10
        elif std < 4.0: score = 8
        elif std < 7.0: score = 5
        else: score = 1

        desc = f"ROE stability stddev={std:.2f}"
        return {"raw": std, "value": round(std, 2), "score": int(score), "desc": desc, "alias": "roe_stability", "source": "composite"}
    except Exception as e:
        logger.debug("ROE stability error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "ROE Stability", "source": "composite"}

def compute_volatility_quality(indicators: Dict[str, Any]) -> Dict[str, Any]:
    try:
        atr_pct = _get_val(indicators, "atr_pct")
        bb_width = _get_val(indicators, "bb_width")
        true_range = _get_val(indicators, "true_range")
        hv10 = _get_val(indicators, "hv_10")
        hv20 = _get_val(indicators, "hv_20")
        atr_sma_ratio = _get_val(indicators, "atr_sma_ratio")
        
        LOW_VOL = VOL_BANDS["low_vol_ceiling"]
        MODERATE_VOL = VOL_BANDS["moderate_vol_ceiling"]
        HIGH_VOL = VOL_BANDS["high_vol_floor"]

        atr_score = 0.0
        if atr_pct is not None:
            if atr_pct <= LOW_VOL: atr_score = 10.0
            elif atr_pct <= MODERATE_VOL: atr_score = 8.0
            elif atr_pct <= HIGH_VOL: atr_score = 6.0
            elif atr_pct <= HIGH_VOL + 1.5: atr_score = 4.0
            else: atr_score = 2.0

        bbw_score = 0.0
        if bb_width is not None:
            if bb_width <= 0.01: bbw_score = 10.0
            elif bb_width <= 0.02: bbw_score = 8.0
            elif bb_width <= 0.04: bbw_score = 6.0
            elif bb_width <= 0.08: bbw_score = 4.0
            else: bbw_score = 2.0

        tr_score = 0.0
        if true_range is not None and atr_pct is not None and atr_pct > 0:
            ratio = true_range / max(atr_pct, 1e-9)
            if ratio <= 0.5: tr_score = 10.0
            elif ratio <= 1.0: tr_score = 8.0
            elif ratio <= 1.5: tr_score = 6.0
            elif ratio <= 2.0: tr_score = 4.0
            else: tr_score = 2.0
        elif true_range is not None:
            if true_range <= 0.5: tr_score = 10.0
            elif true_range <= 1.0: tr_score = 8.0
            else: tr_score = 3.0

        hv_score = 0.0
        if hv10 is not None and hv20 is not None:
            if hv10 < hv20 and hv20 < 25: hv_score = 10.0
            elif hv10 < hv20: hv_score = 8.0
            elif hv10 <= hv20: hv_score = 6.0
            else: hv_score = 3.0
        elif hv20 is not None:
            if hv20 < LOW_VOL * 10: hv_score = 10.0
            elif hv20 < MODERATE_VOL * 10: hv_score = 8.0
            else: hv_score = 2.0

        atr_sma_score = 0.0
        if atr_sma_ratio is not None:
            if atr_sma_ratio < 0.8: atr_sma_score = 10.0
            elif atr_sma_ratio < 0.95: atr_sma_score = 8.0
            elif atr_sma_ratio < 1.05: atr_sma_score = 6.0
            else: atr_sma_score = 3.0

        squeeze_bonus = 0.0
        if _is_squeeze_on(indicators):
            squeeze_bonus = 0.5

        raw = (
            (bbw_score * 0.30) +
            (hv_score * 0.25) +
            (tr_score * 0.20) +
            (atr_sma_score * 0.15) +
            (atr_score * 0.10)
        )
        raw = raw + squeeze_bonus
        score = max(0.0, min(10.0, round(raw, 2)))
        desc = f"VolQuality(atr%={atr_pct}, bbw={bb_width})"
        return {"raw": raw, "value": score, "score": int(round(score)), "desc": desc, "alias": "volatility_quality", "source": "composite"}
    except Exception as e:
        logger.debug("Volatility quality compute error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "volatility quality", "source": "composite"}

# -------------------------
# Scoring Logic (Restored V1 + Penalties)
# -------------------------
def _get_metric_entry(key: str, fundamentals: Dict[str, Any], indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not key: return None
    if indicators and key in indicators: return indicators[key]
    if fundamentals and key in fundamentals: return fundamentals[key]
    return None

def _compute_weighted_score(metrics_map, fundamentals, indicators):
    weighted_sum = 0.0
    weight_sum = 0.0
    details = {}
    
    CRITICAL = {"vwap_bias", "adx", "supertrend_signal", "price_vs_200dma_pct"}

    for metric_key, rule in metrics_map.items():
        try:
            entry = _get_metric_entry(metric_key, fundamentals, indicators)

            if isinstance(rule, dict):
                weight = float(rule.get("weight", 0.0))
                direction = rule.get("direction", "normal")
            else:
                weight = float(rule)
                direction = "normal"

            if weight <= 0: continue

            if not entry:
                if metric_key in CRITICAL:
                    # Optional logging
                    pass
                continue

            score_val = _coerce_score_field(entry)
            
            if score_val is None:
                raw = entry.get("raw")
                if isinstance(raw, str):
                    r = raw.lower()
                    if r in ("strong_buy", "buy", "bullish"): score_val = 8.5
                    elif r in ("hold", "neutral"): score_val = 5.0
                    elif r in ("sell", "bearish", "strong_sell"): score_val = 1.5

            if score_val is None: continue

            s = float(score_val)
            if direction == "invert": s = 10.0 - s

            weighted_sum += s * weight
            weight_sum += weight
            details[metric_key] = s

        except Exception as e:
            logger.debug(f"Error scoring '{metric_key}': {e}", exc_info=False)

    if weight_sum == 0: return 0.0, 0.0, {}
    return weighted_sum, weight_sum, details

def _apply_penalties(penalties_map: Dict[str, Dict[str, Any]], fundamentals: Dict[str, Any], indicators: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    penalty_total = 0.0
    applied = []
    for metric_key, rule in (penalties_map or {}).items():
        entry = _get_metric_entry(metric_key, fundamentals, indicators)
        if not entry: continue
        
        raw = entry.get("raw") or entry.get("value") or entry.get("score")
        raw_num = _safe_float(raw) if not isinstance(raw, (list, dict)) else raw
        op = rule.get("operator")
        tgt = rule.get("value")
        pen = _safe_float(rule.get("penalty")) or 0.0
        
        if _rule_matches(raw_num, op, tgt):
            penalty_total += float(pen)
            applied.append({
                "metric": metric_key,
                "op": op,
                "value": tgt,
                "penalty": float(pen),
                "raw": raw
            })

    penalty_total = min(max(penalty_total, 0.0), 0.95)
    return penalty_total, applied

# -------------------------
# Profile Scoring Logic
# -------------------------
def compute_profile_score(profile_name, fundamentals, indicators, profile_map=None):
    """
    Compute profile score and return the dictionary shape expected by UI/templates.
    """
    profile = (profile_map or HORIZON_PROFILE_MAP).get(profile_name)
    missing_metrics = set()
    if not profile:
        raise KeyError(f"Profile '{profile_name}' not defined")

    metrics_map = profile.get("metrics", {})
    penalties_map = profile.get("penalties", {}) or {}
    thresholds = profile.get("thresholds", {"buy": 8, "hold": 6, "sell": 4})

    # 1. Base Score
    weighted_sum, weight_sum, metric_details = _compute_weighted_score(metrics_map, fundamentals, indicators)

    for mk in metrics_map.keys():
        if mk not in indicators and mk not in fundamentals:
            missing_metrics.add(mk)

    base_score = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
    base_score = round(base_score, 2)

    # 2. Penalties
    penalty_total, applied_penalties = _apply_penalties(penalties_map, fundamentals, indicators)

    # 3. Final Score (Clamped)
    final_score = base_score - penalty_total
    final_score = max(0.0, min(10.0, round(final_score, 2)))

    if final_score >= thresholds.get("buy", 8): cat = "BUY"
    elif final_score >= thresholds.get("hold", 6): cat = "HOLD"
    else: cat = "SELL"

    return {
        "profile": profile_name,
        "base_score": base_score,
        "final_score": final_score,
        "category": cat,
        "metric_details": metric_details,
        "penalty_total": round(penalty_total, 4), # 'no attribute penalty_total'
        "applied_penalties": applied_penalties,
        "thresholds": thresholds,
        "notes": profile.get("notes", ""),
        "missing_keys": list(missing_metrics)
    }

def compute_all_profiles(ticker: str, fundamentals: Dict[str, Any], indicators: Dict[str, Any], profile_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute all profiles with robust error handling to prevent UI 500 errors.
    """
    # Defensive copy
    indicators = (indicators or {}).copy()
    fundamentals = (fundamentals or {}).copy()
    _reset_missing_keys()

    pm = profile_map or HORIZON_PROFILE_MAP

    # 1) Enrich fundamentals
    hybrids = enrich_hybrid_metrics(fundamentals or {}, indicators or {})
    if hybrids:
        fundamentals.update(hybrids)

    profiles_out: Dict[str, Any] = {}
    best_fit, best_score = None, -1.0
    missing_map: Dict[str, List[str]] = {}

    # Base indicators snapshot
    base_inds = indicators.copy() if isinstance(indicators, dict) else {}

    # Pre-compute globals where possible (Safe Wrappers)
    try:
        base_inds["trend_strength"] = compute_trend_strength(base_inds)
    except Exception: pass

    try:
        base_inds["momentum_strength"] = compute_momentum_strength(base_inds)
    except Exception: pass

    try:
        base_inds["roe_stability"] = compute_roe_stability(fundamentals)
    except Exception: pass

    if "compute_volatility_quality" in globals() and ENABLE_VOLATILITY_QUALITY:
        try:
            base_inds["volatility_quality"] = compute_volatility_quality(base_inds)
        except Exception: pass

    # Iterate Profiles
    for pname in pm.keys():
        try:
            # A. Select Indicators (Horizon Slice vs Flat)
            if isinstance(indicators, dict) and isinstance(indicators.get(pname), dict):
                inds_for_profile = indicators[pname].copy()
            else:
                inds_for_profile = base_inds.copy()

            # B. Horizon-Specific Composites
            try:
                inds_for_profile["trend_strength"] = compute_trend_strength(inds_for_profile, horizon=pname)
            except (TypeError, Exception):
                if "trend_strength" not in inds_for_profile:
                    inds_for_profile["trend_strength"] = compute_trend_strength(inds_for_profile)

            # Ensure other composites exist
            if "momentum_strength" not in inds_for_profile:
                inds_for_profile["momentum_strength"] = compute_momentum_strength(inds_for_profile)
            
            if "roe_stability" not in inds_for_profile:
                inds_for_profile["roe_stability"] = compute_roe_stability(fundamentals)

            if "compute_volatility_quality" in globals() and "volatility_quality" not in inds_for_profile:
                 try:
                    inds_for_profile["volatility_quality"] = compute_volatility_quality(inds_for_profile)
                 except: pass

            # C. Score
            out = compute_profile_score(pname, fundamentals, inds_for_profile, profile_map=pm)
            profiles_out[pname] = out
            
        except Exception as e:
            logger.exception("compute_all_profiles failed for profile %s: %s", pname, e)
            # SAFE FALLBACK: Contains all keys required by result.html
            profiles_out[pname] = {
                "profile": pname,
                "error": str(e),
                "base_score": 0.0,
                "final_score": 0.0,
                "penalty_total": 0.0,  # Fixes the Attribute Error on Crash
                "category": "HOLD",
                "metric_details": {},
                "applied_penalties": [],
                "thresholds": {},
                "missing_keys": []
            }

        # Track Best
        fs = profiles_out[pname].get("final_score", 0.0)
        try:
            if fs is not None and float(fs) > float(best_score):
                best_score = float(fs)
                best_fit = pname
        except Exception: pass

    # Aggregate
    avg_signal = (
        round(sum(p.get("final_score", 0) for p in profiles_out.values()) / len(profiles_out), 2)
        if profiles_out else 0.0
    )

    # Collect Missing
    for pname, pdata in profiles_out.items():
        mk = pdata.get("missing_keys", [])
        if mk:
            missing_map[pname] = mk

    total_missing = sorted({v for arr in missing_map.values() for v in arr}) if missing_map else []

    summary = {
        "ticker": ticker,
        "best_fit": best_fit,
        "best_score": best_score,
        "aggregate_signal": avg_signal,
        "profiles": profiles_out,
        "missing_indicators": missing_map,
        "missing_count": {k: len(v) for k, v in missing_map.items()},
        "missing_unique": total_missing,
    }

    logger.info(f"compute_all_profiles: missing metrics {missing_map} for symbol {ticker}")
    return summary

# -------------------------
# Trade plan
# -------------------------
def should_trade_current_volatility(indicators: Dict[str, Any], setup_type: str = "GENERIC") -> Tuple[bool, str]:
    vol_qual = _get_val(indicators, "volatility_quality")
    atr_pct = _get_val(indicators, "atr_pct")
    
    if vol_qual is None or atr_pct is None:
        return True, "Missing vol data, proceed cautiously"
    
    # Skip extreme volatility
    if atr_pct > VOL_BANDS["high_vol_floor"] + 2.0: 
        return False, f"Extreme volatility ({atr_pct}%), avoid breakouts"
    # 2. Breakout Exemption
    # Breakouts are inherently "High Volatility / Low Stability" events.
    # We relax the floor significantly for them.
    if setup_type == "MOMENTUM_BREAKOUT":
        if vol_qual < 2.0: return False, "Volatility chaotic/untradable even for breakout"
        return True, "Volatility expansion allowed for Breakout"
    
    if vol_qual < 4.0:
        return False, f"Low volatility quality ({vol_qual}), potential chop"
    
    return True, "Volatility regime favorable"

def calculate_position_size(indicators: dict, setup_conf: float, setup_type: str, horizon) -> float:
    base_risk = 0.01
    conf_factor = setup_conf / 100.0
    
    quality_multipliers = {
        "DEEP_PULLBACK": 1.5,
        "VOLATILITY_SQUEEZE": 1.3,
        "TREND_PULLBACK": 1.0,
        "MOMENTUM_BREAKOUT": 0.8,
        "TREND_FOLLOWING": 1.0
    }
    multiplier = quality_multipliers.get(setup_type, 1.0)
    
    vol_qual = _get_val(indicators, "volatility_quality")
    vol_factor = 1.0
    if vol_qual:
        vol_factor = 1.2 if vol_qual > 7 else 0.9 if vol_qual < 5 else 1.0
    
    max_pct = 0.02 if horizon != "intraday" else 0.01
    position = base_risk * conf_factor * multiplier * vol_factor
    return round(min(position, max_pct), 4)

def classify_setup(indicators: Dict[str, Any], horizon: str = "short_term") -> str:
    """
    Classify trade setup (Long AND Short detection).
    Hardenings: Explicit Polarity checks for Trend Following.
    """
    ma_keys = _get_ma_keys(horizon)
    fast_k, mid_k, slow_k = ma_keys["fast"], ma_keys["mid"], ma_keys["slow"]

    close = ensure_numeric(_get_val(indicators, "price"))
    open_price = ensure_numeric(_get_val(indicators, "Open"))
    prev_close = ensure_numeric(_get_val(indicators, "prev_close"))
    
    ma_fast = ensure_numeric(_get_val(indicators, fast_k))
    ma_mid  = ensure_numeric(_get_val(indicators, mid_k))
    ma_slow = _get_slow_ma(indicators, horizon) # can be None
    
    bb_upper = ensure_numeric(_get_val(indicators, "bb_high"))
    bb_lower = ensure_numeric(_get_val(indicators, "bb_low"))
    
    rsi = ensure_numeric(_get_val(indicators, "rsi"))
    macd_hist = ensure_numeric(_get_val(indicators, "macd_histogram"))
    rvol = ensure_numeric(_get_val(indicators, "rvol"))
    trend_strength = ensure_numeric(_get_val(indicators, "trend_strength"))
    
    st_val = ensure_numeric(_get_val(indicators, "supertrend_value")) 
    is_squeeze = _is_squeeze_on(indicators)

    wick_ratio = ensure_numeric(_get_val(indicators, "wick_rejection"), default=0.0)

    if not close: return "GENERIC"

    candidates = []

    # Detect Trend Context
    is_uptrend = False
    if ma_slow and close > ma_slow: is_uptrend = True
    
    is_downtrend = False
    if ma_slow and close < ma_slow: is_downtrend = True

    # --- 1. LONG SETUPS ---
    
    # A. MOMENTUM BREAKOUT 
    # We require wick_ratio < 2.5 (Wick shouldn't be 2.5x bigger than the body)
    if (bb_upper > 0 and close >= (bb_upper * 0.98) and 
        rsi > 60 and rvol > 1.5 and trend_strength >= 6):
        
        # The "Bull Trap" Guard
        is_clean_candle = (wick_ratio < 2.5)
        
        if not (st_val and close < st_val) and is_clean_candle:
            candidates.append((100, "MOMENTUM_BREAKOUT"))
    
    # ===========================
    # QUALITY ACCUMULATION (Revised)
    # ===========================

    adx_val = ensure_numeric(_get_val(indicators, "adx"))
    bb_mid  = ensure_numeric(_get_val(indicators, "bb_mid"))
    bb_low  = ensure_numeric(_get_val(indicators, "bb_low"))

    rsi     = ensure_numeric(_get_val(indicators, "rsi"))
    ma_slow = _get_slow_ma(indicators, horizon)

    # Safety guards
    if close and bb_mid and bb_low:

        is_ranging = (adx_val < 30)

        # Allow small margin ABOVE mid band (strong stocks consolidate above 20-SMA)
        is_in_buy_zone = (close > bb_low) and (close <= bb_mid * 1.02)

        # Avoid falling knife / avoid overbought
        is_stable = (35 < rsi < 65)

        # Avoid downtrends
        is_not_downtrend = not (ma_slow and close < ma_slow)

        # Optional slope guard
        mid_slope = ensure_numeric(_get_val(indicators, "bb_mid_slope"))
        is_mid_rising = mid_slope >= 0 if mid_slope is not None else (ma_fast >= ma_mid)

        if is_ranging and is_in_buy_zone and is_stable and is_not_downtrend and is_mid_rising:
            candidates.append((30, "QUALITY_ACCUMULATION"))


    # B. VOLATILITY SQUEEZE
    if is_squeeze:
        candidates.append((95, "VOLATILITY_SQUEEZE"))

    # C. LONG PULLBACKS
    if is_uptrend:
        if ma_fast and (ma_fast * 0.95) <= close <= (ma_fast * 1.05):
            if rsi > 50: candidates.append((75, "TREND_PULLBACK"))
        elif ma_mid and (ma_mid * 0.98) <= close <= (ma_mid * 1.02):
            if rsi > 40: candidates.append((70, "DEEP_PULLBACK"))

    # D. TREND FOLLOWING (Long) - Explicit Polarity Check (Fix 1)
    if is_uptrend and (not is_downtrend) and rsi >= 55 and macd_hist > 0:
        candidates.append((40, "TREND_FOLLOWING"))
        
    # --- 2. SHORT SETUPS ---

    # A. MOMENTUM BREAKDOWN
    if (bb_lower > 0 and close <= (bb_lower * 1.02) and 
        rsi < 40 and rvol > 1.5):
        if not (st_val and close > st_val):
             candidates.append((100, "MOMENTUM_BREAKDOWN"))

    # B. SHORT PULLBACKS
    if is_downtrend:
        if ma_fast and (ma_fast * 0.95) <= close <= (ma_fast * 1.05):
            if rsi < 50: candidates.append((75, "BEAR_PULLBACK"))
        elif ma_mid and (ma_mid * 0.98) <= close <= (ma_mid * 1.02):
            if rsi < 60: candidates.append((70, "DEEP_BEAR_PULLBACK"))

    # C. TREND FOLLOWING (Short) - Explicit Polarity Check (Fix 1)
    if is_downtrend and (not is_uptrend) and rsi <= 45 and macd_hist < 0:
        candidates.append((40, "BEAR_TREND_FOLLOWING"))

    candidates.append((10, "NEUTRAL / CHOPPY"))
    candidates.sort(key=lambda x: (-x[0], x[1])) 
    return candidates[0][1]

def calculate_setup_confidence(indicators: Dict[str, Any], 
                               trend_strength: float, 
                               macro_trend_status: str = "N/A",
                               setup_type: str = "GENERIC",
                               horizon: str = "short_term") -> int:
    """
    Horizon-aware confidence calculation.
    """
    # 1. Fetch correct keys
    ma_keys = _get_ma_keys(horizon)
    fast_k, mid_k, slow_k = ma_keys["fast"], ma_keys["mid"], ma_keys["slow"]

    close = _get_val(indicators, "price")
    ma_fast = _get_val(indicators, fast_k)
    ma_mid  = _get_val(indicators, mid_k)
    ma_slow = _get_slow_ma(indicators, horizon)
    st_val = _get_val(indicators, "supertrend_value") # NEW V2
    
    vwap = _get_val(indicators, "vwap") or ma_fast
    rsi_slope = _get_val(indicators, "rsi_slope")
    macd_hist = _get_val(indicators, "macd_histogram")
    rvol = _get_val(indicators, "rvol")
    obv_div = _get_str(indicators, "obv_div")

    # -----------------------------
    # TREND SCORE (FIXED SCALE)
    # -----------------------------
    # Use ensure_numeric to safely coerce dicts/strings -> floats
    trend_score = 0
    close_val = ensure_numeric(close, default=0.0)
    ma_mid_val = ensure_numeric(ma_mid, default=0.0)
    ma_slow_val = ensure_numeric(ma_slow, default=0.0)
    st_val_num = ensure_numeric(st_val, default=None) if st_val is not None else None
    trend_strength_num = ensure_numeric(trend_strength, default=0.0)

    # Price above moving averages adds to trend score
    if close_val and ma_slow_val and close_val > ma_slow_val:
        trend_score += 15
    if close_val and ma_mid_val and close_val > ma_mid_val:
        trend_score += 10

    # FIXED SCALE: trend_strength is 0..10 -> treat >7 as strong, >5 moderate
    if trend_strength_num >= 7.0:  # Strong (70%+ of max)
        trend_score += 15
    elif trend_strength_num >= 5.0:  # Moderate
        trend_score += 8  # Up from 8
    elif trend_strength_num >= 3.0:  # Weak trend exists
        trend_score += 3

    # Boost when price above supertrend (confirming trend)
    if st_val_num is not None and close_val > st_val_num:
        trend_score += 10

    # Cap trend score to avoid over-weighting
    trend_score = min(40, trend_score)

    mom_score = 0
    if macd_hist and macd_hist > 0: mom_score += 15
    if close and vwap and close > vwap: mom_score += 10
    if rsi_slope and rsi_slope > 0: mom_score += 10
    vol_qual = _get_val(indicators, "volatility_quality")

    # Reward stability generally, OR reward the setup being a Breakout (which implies Vol Expansion)
    if setup_type == "MOMENTUM_BREAKOUT":
        mom_score += 5  # Breakouts get the points because Vol Expansion is their "Quality"
    elif vol_qual and vol_qual > 6: 
        mom_score += 5
    mom_score = min(40, mom_score)

    vol_score = 0
    if rvol and rvol > 1.0: vol_score += 10
    if rvol and rvol > 2.0: vol_score += 5
    if "confirm" in obv_div or "bull" in obv_div: vol_score += 5
    vol_score = min(20, vol_score)

    total_conf = trend_score + mom_score + vol_score

    macro = (macro_trend_status or "").lower()
    is_bearish_macro = any(x in macro for x in ["bear", "down", "corrective", "distribution", "weak"])
    
    if is_bearish_macro:
        if setup_type in ["MOMENTUM_BREAKOUT", "TREND_FOLLOWING"]:
            total_conf *= 0.65 
        elif setup_type in ["TREND_PULLBACK", "DEEP_PULLBACK"]:
            total_conf *= 0.90
        else:
            total_conf *= 0.85

    boost_map = {
        "MOMENTUM_BREAKOUT": 1.10,
        "TREND_PULLBACK": 1.07,
        "DEEP_PULLBACK": 1.05,
        "OVERSOLD_IN_UPTREND": 1.05,
        "VOLATILITY_SQUEEZE": 1.05,
        "OVERSOLD_REVERSAL": 0.90
    }
    boost = boost_map.get(setup_type, 1.0)
    
    final_conf = int(total_conf * boost)
    return min(100, max(0, final_conf))

def generate_trade_plan(profile_report: Dict[str, Any],
                        indicators: Dict[str, Any],
                        macro_trend_status: str = "N/A",
                        horizon: str = "short_term",
                        strategy_report: Dict[str, Any] = None,
                        fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Master Trade Plan: Symmetric Long/Short Guards, Risk Clamps, and Setup Floors.
    Hardenings applied: Strict Setup Matching, Debug String Updates.
    AUTO-TUNED: Dynamic Confidence Floors + Dynamic ATR based on Volatility.
    """
    final_score = profile_report.get("final_score", 0)
    category = profile_report.get("category", "HOLD")

    # --- 1. Robust Data ---
    price_val = ensure_numeric(_get_val(indicators, "price"))
    atr_val = ensure_numeric(_get_val(indicators, "atr_dynamic"))
    
    psar_trend = _get_str(indicators, "psar_trend") or ""
    psar_level = ensure_numeric(_get_val(indicators, "psar_level"), default=None)
    
    st_signal_str = _get_str(indicators, "supertrend_signal") or ""
    st_val_num = ensure_numeric(_get_val(indicators, "supertrend_value"), default=None)
    
    st_direction = "BULL" if "bull" in st_signal_str.lower() else "BEAR" if "bear" in st_signal_str.lower() else None

    # --- 2. Classification & Confidence ---
    setup_type = classify_setup(indicators, horizon=horizon)
    ts_val = ensure_numeric(_get_val(indicators, "trend_strength"))
    
    try:
        setup_conf = calculate_setup_confidence(indicators, ts_val, macro_trend_status, setup_type, horizon=horizon)
    except TypeError:
        setup_conf = calculate_setup_confidence(indicators, ts_val, macro_trend_status, setup_type)

    can_trade_vol, vol_reason = should_trade_current_volatility(indicators, setup_type=setup_type)

    plan = {
        "signal": "NO_TRADE", "setup_type": setup_type, "setup_confidence": setup_conf,
        "entry": price_val, "stop_loss": None, "targets": {"t1": None, "t2": None},
        "rr_ratio": 0, "execution_hints": {},
        "reason": "Analysis Inconclusive"
    }

    if price_val <= 0 or atr_val <= 0:
        plan["reason"] = "Data Error: Invalid Price or ATR"
        return plan

    # --- 3. ENTRY PERMISSIONS ---
    can_enter = False
    
    profile_cfg = HORIZON_PROFILE_MAP.get(horizon, HORIZON_PROFILE_MAP.get("short_term", {}))
    required_conf_base = profile_cfg.get("thresholds", {"buy": 7.0}).get("buy", 7.0) * 10
    
    adx_val = ensure_numeric(_get_val(indicators, "adx"))
    adx_req = 20 if horizon == "intraday" else 25
    rvol_val = ensure_numeric(_get_val(indicators, "rvol"), default=1.0)

    # LOGIC A: BREAKOUTS / BREAKDOWNS
    if setup_type in ["MOMENTUM_BREAKOUT", "MOMENTUM_BREAKDOWN"]:
        if setup_conf >= required_conf_base: can_enter = True

    # LOGIC B: LONG PULLBACKS
    elif setup_type in ["TREND_PULLBACK", "DEEP_PULLBACK", "TREND_FOLLOWING"]:
        discounted_conf = required_conf_base - 15
        if setup_conf >= discounted_conf and ts_val >= 5.0: 
            can_enter = True

    # LOGIC C: SHORT PULLBACKS
    elif setup_type in ["BEAR_PULLBACK", "DEEP_BEAR_PULLBACK", "BEAR_TREND_FOLLOWING"]:
        discounted_conf = required_conf_base - 15
        if setup_conf >= discounted_conf and adx_val >= adx_req:
            can_enter = True

    # LOGIC D: SQUEEZES
    elif setup_type in ["VOLATILITY_SQUEEZE", "OVERSOLD_REVERSAL"]:
         if setup_conf >= (required_conf_base - 5): can_enter = True

    # LOGIC E: QUALITY ACCUMULATION

    top_strat = (strategy_report or {}).get("summary", {}).get("best_strategy", "")
    top_strat_l = str(top_strat or "").lower()
    is_value_strat = top_strat_l in ("value", "position_trading", "income", "dividend", "quality")

    # Normalize setup_type string for robust checks
    st_l = (setup_type or "").lower()

    # Determine direct technical accumulation
    is_acc_setup = st_l.startswith("quality_accumulation") or ("accumulation" in st_l)

    # Force accumulation if neutral AND fundamentals & strategy align
    force_accumulation = False
    if not fundamentals or not isinstance(fundamentals, dict):
        logger.debug("Accumulation check skipped - no fundamental data")
        force_accumulation = False
    elif fundamentals :
        momentum_setups = ("MOMENTUM_BREAKOUT","MOMENTUM_BREAKDOWN","VOLATILITY_SQUEEZE")
        weak_technical = setup_type not in momentum_setups
        top_strat = (strategy_report or {}).get("summary", {}).get("best_strategy")
        is_value_strat = top_strat in ("value", "position_trading", "income", "dividend")

        if weak_technical and is_value_strat:
            pe = ensure_numeric(_get_val(fundamentals or {}, "pe_ratio"))
            roe = ensure_numeric(_get_val(fundamentals or {}, "roe"))
            eps_g = ensure_numeric(_get_val(fundamentals or {}, "eps_growth_5y"))

            if final_score >= 7.0 and pe > 0 and pe < 25 and roe > 12 and eps_g >= 0:
                force_accumulation = True

        if is_acc_setup or force_accumulation:

            # Fetch BBs safely
            bb_high = ensure_numeric(_get_val(indicators, "bb_high"), default=0.0)
            bb_low  = ensure_numeric(_get_val(indicators, "bb_low"),  default=0.0)
            bb_mid  = ensure_numeric(_get_val(indicators, "bb_mid"),  default=0.0)

            # Basic data sanity
            if price_val <= 0 or bb_low <= 0 or bb_mid <= 0 or bb_high <= 0:
                logger.debug("Accumulation skipped - missing BBs or price. price=%s bb_low=%s bb_mid=%s bb_high=%s", price_val, bb_low, bb_mid, bb_high)
            else:
                # Score gate (slightly relaxed)
                if final_score < 6.0:
                    plan["signal"] = "WAIT"
                    plan["reason"] = f"Consolidating, but total score ({final_score}/10) is too low for accumulation."
                    return plan

                # Strategy alignment check
                if not is_value_strat and not force_accumulation:
                    plan["signal"] = "WAIT"
                    plan["reason"] = f"Consolidation but top strategy '{top_strat}' not accumulation-friendly."
                    return plan

                # Stop loss: 2% below BB low (structure break)
                sl_final = bb_low * 0.98

                # T1: mid if below mid else upper band
                t1 = bb_mid if price_val <= bb_mid else bb_high
                t2 = bb_high * 1.05

                # Ensure sane risk
                risk = price_val - sl_final
                if risk <= 0 or math.isnan(risk):
                    logger.warning("Accumulation risk non-positive: price=%s sl=%s", price_val, sl_final)
                    plan["signal"] = "WAIT"
                    plan["reason"] = "Bad stop/risk calculation for accumulation"
                    return plan

                plan.update({
                    "signal": "BUY_ACCUMULATE",
                    "reason": "Range Accumulation: fundamentals + consolidation match",
                    "entry": price_val,
                    "stop_loss": round(sl_final, 4),
                    "targets": {"t1": round(t1, 4), "t2": round(t2, 4)}
                })

                # RR
                plan["rr_ratio"] = round((t1 - price_val) / risk, 2) if risk > 0 else None

                # Position sizing: smaller per-chunk allocation for accumulation
                # Use existing calculate_position_size (returns fraction)
                chunk_size = calculate_position_size(indicators, setup_conf, "QUALITY_ACCUMULATION", horizon)
                # recommend pyramid: 3 chunks total but each = chunk_size / n_chunks
                n_chunks = 3
                per_chunk = round(min(chunk_size, 0.02) / n_chunks, 4)
                plan["position_size"] = per_chunk
                plan["execution_hints"].update({
                    "strategy": "Range Accumulation",
                    "pyramid": {
                        "chunks": n_chunks,
                        "per_chunk": per_chunk,
                        "total_target": round(per_chunk * n_chunks, 4)
                    },
                    "note": "Buy in tranches; stop if BB low breaks."
                })

                # Est time
                plan["est_time"] = estimate_hold_time(price_val=price_val,t1=plan["targets"]["t1"],t2=plan["targets"]["t2"],atr_val=atr_val,horizon=horizon,indicators=indicators,strategies=(strategy_report or {}).get("summary", {}))
                logger.info("Accumulation plan created for %s: t1=%s t2=%s sl=%s score=%s strategy=%s force=%s",
                            profile_report.get("ticker", "<t>"), t1, t2, sl_final, final_score, top_strat, force_accumulation)

                return plan
    # --- end accumulation block ---

    # Volatility Guard
    if not can_trade_vol and "MOMENTUM" not in setup_type and "SQUEEZE" not in setup_type:
        plan["signal"] = "WAIT"
        plan["reason"] = vol_reason
        return plan

    # ----------------------------------------------------
    # DYNAMIC CONFIDENCE TUNING (The "Quality Gate")
    # ----------------------------------------------------
    BASE_FLOORS = {
        "MOMENTUM_BREAKOUT": 55, "MOMENTUM_BREAKDOWN": 55,
        "VOLATILITY_SQUEEZE": 50,
        "TREND_PULLBACK": 53, "BEAR_PULLBACK": 53,
        "DEEP_PULLBACK": 45,  "DEEP_BEAR_PULLBACK": 45,
        "TREND_FOLLOWING": 57, "BEAR_TREND_FOLLOWING": 57
    }
    base_floor = BASE_FLOORS.get(setup_type, 65)

    # Get Volatility Personality (0 = Chaos, 10 = Stable)
    vol_qual = ensure_numeric(_get_val(indicators, "volatility_quality"), default=5.0)

    floor_adjustment = 0
    if vol_qual <= 4.0:
        floor_adjustment = 5  # Chaos: Strict
        plan["execution_hints"]["entry_mode"] = "Strict (High Vol)"
    elif vol_qual >= 8.0:
        floor_adjustment = -5 # Stable: Relaxed
        plan["execution_hints"]["entry_mode"] = "Relaxed (High Quality)"
    else:
        plan["execution_hints"]["entry_mode"] = "Normal"

    final_floor = base_floor + floor_adjustment

    # Filter marginal trades
    if not can_enter and setup_conf < final_floor:
        plan["signal"] = "WAIT"
        plan["reason"] = f"Confidence {setup_conf}% < {final_floor}% (tuned for VolQual {vol_qual})"
        return plan

    plan["position_size"] = calculate_position_size(indicators, setup_conf, setup_type, horizon=horizon)
    
    mst = str(macro_trend_status or "").lower()
    macro_bearish = "down" in mst or "bear" in mst
    macro_bullish = "up" in mst or "bull" in mst
    is_squeeze = _is_squeeze_on(indicators)
    strat_summary = strategy_report.get("summary", {}) if strategy_report else {}
    # ----------------------------------------------------
    # ENTRY VALIDATION
    # ----------------------------------------------------
    
    # Valid Long?
    is_valid_buy = False
    if category == "BUY":
        if ("bull" in (psar_trend or "").lower()) or (setup_type == "MOMENTUM_BREAKOUT") or (ts_val > 7.0) or (st_direction == "BULL") or can_enter:
             is_valid_buy = True
    elif can_enter and st_direction == "BULL":
        is_valid_buy = True

    # Valid Short?
    is_valid_short = False
    if category == "SELL":
        if ("bear" in (psar_trend or "").lower()) or (setup_type == "MOMENTUM_BREAKDOWN") or (st_direction == "BEAR") or can_enter:
            is_valid_short = True
    elif can_enter and st_direction == "BEAR":
         is_valid_short = True
    elif setup_type == "MOMENTUM_BREAKDOWN": 
         is_valid_short = True

    # ----------------------------------------------------
    # DYNAMIC ATR CONFIG (The "Universal Tuner")
    # ----------------------------------------------------
    # 1. Fetch Baseline Config
    atr_cfg = ATR_MULTIPLIERS.get(horizon, ATR_MULTIPLIERS.get("short_term", {"sl": 2.0, "tp": 3.0}))
    base_sl = ensure_numeric(atr_cfg.get("sl"), default=2.0)
    base_tp = ensure_numeric(atr_cfg.get("tp"), default=3.0)

    # 2. Auto-Tune Based on Personality (Using previously computed vol_qual)
    if vol_qual >= 8.0:
        # STABLE: Needs tighter stops
        atr_sl_mult = 1.5
        atr_tp_mult = 2.5
        plan["execution_hints"]["volatility_mode"] = "Stable (Tightened)"
    elif vol_qual <= 4.0:
        # VOLATILE: Needs wider stops
        atr_sl_mult = 3.0 
        atr_tp_mult = 5.0
        plan["execution_hints"]["volatility_mode"] = "Volatile (Widened)"
    else:
        # NORMAL
        atr_sl_mult = base_sl
        atr_tp_mult = base_tp
        plan["execution_hints"]["volatility_mode"] = "Normal"

    # --- LONG BRANCH ---
    if is_valid_buy:
        dist_to_st = (st_val_num - price_val) / price_val if (st_val_num and price_val < st_val_num) else 999
        if st_direction == "BEAR" and dist_to_st < 0.015 and setup_type != "MOMENTUM_BREAKOUT" and not is_squeeze:
             plan["signal"] = "WAIT_RESISTANCE"
             plan["reason"] = f"Price too close to ST Res {st_val_num}"
             return plan

        sl_atr = price_val - (atr_val * atr_sl_mult)
        sl_final = sl_atr
        strategy = "ATR"

        # Supertrend Trail (Clamp)
        if st_val_num and price_val > st_val_num:
             clamped_st = max(st_val_num, price_val - (atr_val * 2.0))
             sl_final = max(sl_atr, clamped_st)
             strategy = "Supertrend (Clamped)"

        # PSAR Tightening
        if psar_level and psar_level > 0 and psar_level > sl_final and psar_level < price_val:
             if psar_level > sl_final:
                sl_final = psar_level
                strategy = "PSAR"
        
        # Noise Clamp
        if (price_val - sl_final) < (atr_val * 0.5): 
            sl_final = price_val - (atr_val * 0.5)
            strategy += " + Noise Clamp"

        risk = price_val - sl_final
        plan["stop_loss"] = round(sl_final, 4)
        plan["targets"]["t1"] = round(price_val + (risk * 1.5), 4)
        plan["targets"]["t2"] = round(price_val + (atr_val * atr_tp_mult), 4)
        plan["rr_ratio"] = round((plan["targets"]["t2"] - price_val) / risk, 2) if risk > 0 else 0

        # Relax Volume for Long Pullbacks
        if setup_type in ("TREND_PULLBACK", "DEEP_PULLBACK") and rvol_val < 0.8:
            plan["execution_hints"]["pullback_low_volume_ok"] = True

        wick_val = ensure_numeric(_get_val(indicators, "wick_rejection"), default=0.0)
        if setup_type == "MOMENTUM_BREAKOUT" and wick_val > 1.5:
             plan["execution_hints"]["caution_wick"] = f"High upper wick ({wick_val}x). Wait for high breach."
        
        plan["signal"] = "RISKY_BUY" if macro_bearish else "BUY_SQUEEZE" if is_squeeze else "BUY_TREND"
        plan["reason"] = f"Long {setup_type} | Conf {setup_conf}%"
        plan["est_time"] = estimate_hold_time(price_val=price_val,t1=plan["targets"].get("t1"),t2=plan["targets"].get("t2"),atr_val=atr_val,horizon=horizon,indicators=indicators,strategies=strat_summary)
        return plan

    # --- SHORT BRANCH ---
    if is_valid_short:
        dist_to_st = (price_val - st_val_num) / price_val if (st_val_num and price_val > st_val_num) else 999
        
        if st_direction == "BULL" and (st_val_num and price_val > st_val_num) and dist_to_st < 0.015 and setup_type != "MOMENTUM_BREAKDOWN" and not is_squeeze:
             plan["signal"] = "WAIT_SUPPORT"
             plan["reason"] = f"Price too close to ST Supp {st_val_num}"
             return plan

        if setup_type == "MOMENTUM_BREAKDOWN" and rvol_val < 1.0:
            plan["signal"] = "WAIT_LOW_VOL"
            plan["reason"] = "Breakdown requires Volume > 1.0"
            return plan

        sl_atr = price_val + (atr_val * atr_sl_mult)
        sl_final = sl_atr
        strategy = "ATR"

        # Supertrend Trail (Clamp)
        if st_val_num and price_val < st_val_num:
             clamped_st = min(st_val_num, price_val + (atr_val * 2.0))
             sl_final = min(sl_atr, clamped_st)
             strategy = "Supertrend (Clamped)"

        # PSAR Tighten (Shorts)
        if psar_level and psar_level > price_val:
            if psar_level < sl_final: 
                sl_final = psar_level
                strategy = "PSAR"

        # Noise Clamp (Short)
        if (sl_final - price_val) < (atr_val * 0.5): 
            sl_final = price_val + (atr_val * 0.5)
            strategy += " + Noise Clamp"

        risk = sl_final - price_val
        plan["stop_loss"] = round(sl_final, 4)
        plan["targets"]["t1"] = round(price_val - (risk * 1.5), 4)
        plan["targets"]["t2"] = round(price_val - (atr_val * atr_tp_mult), 4)
        plan["rr_ratio"] = round((price_val - plan["targets"]["t2"]) / risk, 2) if risk > 0 else 0

        # Relax Volume for Short Pullbacks
        if setup_type in ("BEAR_PULLBACK", "DEEP_BEAR_PULLBACK") and rvol_val < 0.8:
            plan["execution_hints"]["pullback_low_volume_ok"] = True

        plan["signal"] = "RISKY_SHORT" if macro_bullish else "SHORT_SQUEEZE" if is_squeeze else "SHORT_TREND"
        plan["reason"] = f"Short {setup_type} | Conf {setup_conf}%"
        plan["est_time"] = estimate_hold_time(price_val=price_val,t1=plan["targets"].get("t1"),t2=plan["targets"].get("t2"),atr_val=atr_val,horizon=horizon,indicators=indicators,strategies=strat_summary)

        return plan
    
    plan["signal"] = "WAIT"
    plan["reason"] = f"Score {final_score} Neutral/Inconclusive (No valid Setup)"
    return plan

# ----------------------------------------------------------------------
# Meta-Category Scoring (Restored)
# ----------------------------------------------------------------------
def score_value_profile(fundamentals: Dict[str, Any]) -> float:
    w, tot, _ = _compute_weighted_score(VALUE_WEIGHTS, fundamentals, {})
    return round((w / tot), 2) if tot > 0 else 0.0

def score_growth_profile(fundamentals: Dict[str, Any]) -> float:
    w, tot, _ = _compute_weighted_score(GROWTH_WEIGHTS, fundamentals, {})
    return round((w / tot), 2) if tot > 0 else 0.0

def score_quality_profile(fundamentals: Dict[str, Any]) -> float:
    w, tot, _ = _compute_weighted_score(QUALITY_WEIGHTS, fundamentals, {})
    return round((w / tot), 2) if tot > 0 else 0.0

def score_momentum_profile(fundamentals: Dict[str, Any], indicators: Dict[str, Any]) -> float:
    w, tot, _ = _compute_weighted_score(MOMENTUM_WEIGHTS, fundamentals, indicators)
    return round((w / tot), 2) if tot > 0 else 0.0