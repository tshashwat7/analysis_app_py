# services/signal_engine.py
"""
Signal Engine v6.3 - Robust, Zero-Default, Horizon-Aware
- Implements strict scoring (0 default for missing data).
- Dynamic MA and Slope lookups.
- Standardized metric keys (price_vs_200dma_pct).
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

logger = logging.getLogger(__name__)

MISSING_KEYS = set()

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
# Unified Data Accessors (Helpers)
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

# --- NEW: Dynamic Fallback Helpers ---

def _get_slow_ma(indicators: Dict[str, Any], horizon: str) -> Optional[float]:
    """
    Tries the specific horizon slow MA first, then falls back to common long MAs.
    """
    keys = _get_ma_keys(horizon)
    val = _get_val(indicators, keys["slow"])
    if val is not None:
        return val
    # Fallbacks in order of preference
    return (_get_val(indicators, "ema_200") or 
            _get_val(indicators, "dma_200") or 
            _get_val(indicators, "wma_50") or 
            _get_val(indicators, "mma_12"))

def _get_dynamic_slope(indicators: Dict[str, Any], horizon: str) -> Optional[float]:
    """
    Tries horizon-specific slope, then generic fallbacks.
    """
    # Preferred mapping
    pref_key = {
        "intraday": "ema_20_slope",
        "short_term": "ema_20_slope", # Short term often relies on fast MA momentum
        "long_term": "wma_50_slope",
        "multibagger": "dma_200_slope" 
    }.get(horizon)
    
    val = _get_val(indicators, pref_key) if pref_key else None
    if val is not None: 
        return val
        
    # Fallback list
    fallback_keys = ["ema_slope", "ema_20_slope", "wma_50_slope", "dma_200_slope"]
    for k in fallback_keys:
        v = _get_val(indicators, k)
        if v is not None: return v
    return None

def _is_squeeze_on(indicators: Dict[str, Any]) -> bool:
    """Robust squeeze detection."""
    val = _get_str(indicators, "ttm_squeeze")
    # Checks for substring matches
    return any(x in val for x in ("on", "sqz", "squeeze_on", "squeeze on"))

# -------------------------
# Score Coercion & Rules
# -------------------------
def _coerce_score_field(metric_entry: Any) -> Optional[float]:
    if not metric_entry:
        return None
    if isinstance(metric_entry, dict):
        s = metric_entry.get("score")
        if s is not None:
            return _safe_float(s)
        raw = metric_entry.get("raw")
        rv = _safe_float(raw)
        if rv is not None and 0 <= rv <= 10:
            return rv
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
    if raw_val is None:
        return False
    if op == "in":
        try:
            return raw_val in tgt
        except Exception:
            return False
    try:
        if isinstance(raw_val, (int, float)):
            return _OPS[op](float(raw_val), float(tgt))
        return _OPS[op](raw_val, tgt)
    except Exception:
        return False

# -------------------------
# Hybrid metrics
# -------------------------
def enrich_hybrid_metrics(fundamentals: dict, indicators: dict) -> dict:
    hybrids = {}
    roe = _get_val(fundamentals, "roe")
    atr_pct = _get_val(indicators, "atr_pct")
    
    # Volatility-Adjusted ROE
    if roe is not None and atr_pct is not None and atr_pct > 0:
        ratio = roe / atr_pct
        score = 10 if ratio >= 10 else 7 if ratio >= 5 else 3 if ratio >= 2 else 0
        hybrids["volatility_adjusted_roe"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"ROE/Vol = {ratio:.2f}", "alias": "Volatility-Adjusted ROE", "source": "hybrid"
        }

    # Price vs Intrinsic Value
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
        except Exception:
            pass

    # FCF Yield vs Volatility
    fcf_yield = _get_val(fundamentals, "fcf_yield")
    if fcf_yield is not None and atr_pct is not None:
        ratio = fcf_yield / max(atr_pct, 0.1)
        score = 10 if ratio >= 10 else 8 if ratio >= 5 else 5 if ratio >= 2 else 2
        hybrids["fcf_yield_vs_volatility"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"FCF/Vol = {ratio:.2f}", "alias": "FCF Yield vs Volatility", "source": "hybrid"
        }

    # Trend Consistency
    adx = _get_val(indicators, "adx")
    supertrend = _get_str(indicators, "supertrend_signal")
    if adx is not None:
        score = 10 if adx >= 25 else 7 if adx >= 20 else 4
        if "bull" in supertrend: score = min(10, score + 1)
        hybrids["trend_consistency"] = {
            "raw": adx, "value": adx, "score": min(10, score),
            "desc": f"ADX {adx:.1f}", "alias": "Trend Consistency", "source": "hybrid"
        }

    # Price vs 200DMA (Standardized Key)
    # Try dynamic fallback for 200DMA value
    dma_200 = _get_val(indicators, "dma_200") or _get_val(indicators, "ema_200")
    if price and dma_200:
        ratio = (price / dma_200) - 1
        score = 10 if ratio > 0.1 else 7 if ratio > 0.0 else 3 if ratio > -0.05 else 0
        # Explicitly name it price_vs_200dma_pct
        hybrids["price_vs_200dma_pct"] = {
            "raw": ratio, "value": round(ratio * 100, 2), "score": score,
            "desc": f"Price vs 200DMA: {ratio*100:.2f}%", "alias": "Price vs 200 DMA (%)", "source": "hybrid"
        }

    # Fundamental Momentum
    q_growth = _get_val(fundamentals, "quarterly_growth")
    eps_5y = _get_val(fundamentals, "eps_growth_5y")
    if q_growth is not None and eps_5y is not None:
        ratio = (q_growth + eps_5y / 5) / 2
        score = 10 if ratio >= 15 else 7 if ratio >= 10 else 4 if ratio >= 5 else 1
        hybrids["fundamental_momentum"] = {
            "raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"Growth Mom = {ratio:.2f}%", "alias": "Fundamental Momentum", "source": "hybrid"
        }

    # Earnings Consistency
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
# Composite Calculations (Strict 0 Defaults)
# -------------------------
def compute_trend_strength(indicators: Dict[str, Any], horizon: str = "short_term") -> Dict[str, Any]:
    try:
        adx = _get_val(indicators, "adx")
        # Use dynamic slope lookup
        ema_slope = _get_dynamic_slope(indicators, horizon)
        
        di_plus = _get_val(indicators, "di_plus")
        di_minus = _get_val(indicators, "di_minus")
        supertrend = _get_str(indicators, "supertrend_signal")

        # STRICT DEFAULT: 0.0
        adx_score = 0.0
        if adx is not None:
            if adx >= TREND_THRESH["strong_floor"]: adx_score = 10.0
            elif adx >= TREND_THRESH["moderate_floor"]: adx_score = 8.0
            elif adx >= TREND_THRESH["weak_floor"]: adx_score = 4.0
            else: adx_score = 2.0

        ema_score = 0.0
        if ema_slope is not None:
            if isinstance(ema_slope, str):
                ema_score = 10.0 if "rising" in ema_slope.lower() else 0.0
            else:
                v = abs(ema_slope)
                if v >= 2.0: ema_score = 10.0
                elif v >= 1.0: ema_score = 8.0
                elif v >= 0.5: ema_score = 6.0
                elif v >= 0.2: ema_score = 4.0
                else: ema_score = 2.0

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
        # If supertrend missing, score stays 0

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
            else: slope_score = 3.0 # slight penalty for neutral

        macd_score = 0.0
        if macd_hist is not None:
            if macd_hist >= MACD_MOMENTUM_THRESH["acceleration_floor"]: macd_score = 8.0
            elif macd_hist <= MACD_MOMENTUM_THRESH["deceleration_ceiling"]: macd_score = 2.0
            else: macd_score = 5.0 # neutral range

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

        # STRICT DEFAULT: 0.0
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
# Scoring Logic
# -------------------------
def _get_metric_entry(key: str, fundamentals: Dict[str, Any], indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    if indicators and key in indicators:
        return indicators[key]
    if fundamentals and key in fundamentals:
        return fundamentals[key]
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

            if weight <= 0:
                continue

            if not entry:
                if metric_key in CRITICAL:
                    # weight_sum += weight
                    # details[metric_key] = 0.0 # Explicit penalty
                    logger.warning(f"critical metrics {metric_key} is missing from entry")
                continue

            score_val = _coerce_score_field(entry)
            
            if score_val is None:
                raw = entry.get("raw")
                if isinstance(raw, str):
                    r = raw.lower()
                    if r in ("strong_buy", "buy", "bullish"): score_val = 8.5
                    elif r in ("hold", "neutral"): score_val = 5.0
                    elif r in ("sell", "bearish", "strong_sell"): score_val = 1.5

            if score_val is None:
                continue

            s = float(score_val)
            if direction == "invert":
                s = 10.0 - s

            weighted_sum += s * weight
            weight_sum += weight
            details[metric_key] = s

        except Exception as e:
            logger.debug(f"Error scoring '{metric_key}': {e}", exc_info=False)

    if weight_sum == 0:
        return 0.0, 0.0, {}

    return weighted_sum, weight_sum, details

def _apply_penalties(penalties_map: Dict[str, Dict[str, Any]], fundamentals: Dict[str, Any], indicators: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    penalty_total = 0.0
    applied = []
    for metric_key, rule in (penalties_map or {}).items():
        entry = _get_metric_entry(metric_key, fundamentals, indicators)
        if not entry:
            continue
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
# Profile logic
# -------------------------
def compute_profile_score(profile_name, fundamentals, indicators, profile_map=None):
    profile = (profile_map or HORIZON_PROFILE_MAP).get(profile_name)
    missing_metrics = set()
    if not profile:
        raise KeyError(f"Profile '{profile_name}' not defined")

    metrics_map = profile.get("metrics", {})
    penalties_map = profile.get("penalties", {})
    thresholds = profile.get("thresholds", {"buy": 8, "hold": 6, "sell": 4})

    weighted_sum, weight_sum, metric_details = _compute_weighted_score(metrics_map, fundamentals, indicators)

    for mk in metrics_map.keys():
        if mk not in indicators and mk not in fundamentals:
            missing_metrics.add(mk)

    base_score = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
    base_score = round(base_score, 2)

    penalty_total, applied_penalties = _apply_penalties(
        penalties_map, fundamentals, indicators
    )

    final_score = base_score - penalty_total
    final_score = max(0.0, min(10.0, round(final_score, 2)))

    if final_score >= thresholds["buy"]: cat = "BUY"
    elif final_score >= thresholds["hold"]: cat = "HOLD"
    else: cat = "SELL"

    return {
        "profile": profile_name,
        "base_score": base_score,
        "final_score": final_score,
        "category": cat,
        "metric_details": metric_details,
        "penalty_total": round(penalty_total, 4),
        "applied_penalties": applied_penalties,
        "thresholds": thresholds,
        "notes": profile.get("notes", ""),
        "missing_keys": list(missing_metrics)
    }

def compute_all_profiles(ticker: str, fundamentals: Dict[str, Any], indicators: Dict[str, Any], profile_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # 1. Ensure idempotency by copying
    indicators = indicators.copy()
    fundamentals = fundamentals.copy()
    _reset_missing_keys()

    pm = profile_map or HORIZON_PROFILE_MAP

    hybrids = enrich_hybrid_metrics(fundamentals or {}, indicators or {})
    if hybrids:
        fundamentals.update(hybrids)

    # Pre-compute composites
    # Note: Pass horizon if known, else defaults to short_term slope lookups
    # If you need specific horizon-based composites, you'd need to loop per profile.
    # For now, we use a generic compute which falls back to common keys.
    composites = {
        "trend_strength": compute_trend_strength(indicators), 
        "momentum_strength": compute_momentum_strength(indicators),
        "roe_stability": compute_roe_stability(fundamentals)
    }
    indicators.update(composites)

    if ENABLE_VOLATILITY_QUALITY:
        indicators["volatility_quality"] = compute_volatility_quality(indicators)

    profiles_out = {}
    best_fit, best_score = None, -1.0

    for pname in pm.keys():
        try:
            out = compute_profile_score(pname, fundamentals, indicators, profile_map=pm)
            profiles_out[pname] = out
        except Exception as e:
            logger.exception("Failed compute_profile_score %s for %s: %s", pname, ticker, e)
            out = {"profile": pname, "error": str(e)}
            profiles_out[pname] = out
            
        fs = out.get("final_score", 0.0) if isinstance(out, dict) else 0.0
        if fs > best_score:
            best_score = fs
            best_fit = pname

    avg_signal = (
        round(sum(p.get("final_score", 0) for p in profiles_out.values()) / len(profiles_out), 2)
        if profiles_out else 0.0
    )
    missing_map = {}
    for pname, pdata in profiles_out.items():
        mk = pdata.get("missing_keys", [])
        if mk:
            missing_map[pname] = mk

    total_missing = sorted({v for arr in missing_map.values() for v in arr})
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
    logger.info(f"missing metrics {missing_map} for symbol {ticker}" )
    return summary

# -------------------------
# Trade plan
# -------------------------
def should_trade_current_volatility(indicators: Dict[str, Any], setup_type: str = "GENERIC") -> Tuple[bool, str]:
    vol_qual = _get_val(indicators, "volatility_quality")
    atr_pct = _get_val(indicators, "atr_pct")
    
    if vol_qual is None or atr_pct is None:
        return True, "Missing vol data, proceed cautiously"
    
    # Skip extreme volatility (> 4.5% daily range usually unsafe for standard stops)
    if atr_pct > VOL_BANDS["high_vol_floor"] + 2.0: 
        return False, f"Extreme volatility ({atr_pct}%), avoid breakouts"
    # 2. Breakout Exemption
    # Breakouts are inherently "High Volatility / Low Stability" events.
    # We relax the floor significantly for them.
    if setup_type == "MOMENTUM_BREAKOUT":
        if vol_qual < 2.0: 
            return False, "Volatility chaotic/untradable even for breakout"
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
    Classify trade setup using Horizon-specific Moving Averages.
    """
    # 1. Fetch correct keys for this horizon
    ma_keys = _get_ma_keys(horizon)
    fast_k, mid_k, slow_k = ma_keys["fast"], ma_keys["mid"], ma_keys["slow"]

    # 2. Retrieve values
    close = _get_val(indicators, "price")
    open_price = _get_val(indicators, "Open")
    prev_close = _get_val(indicators, "prev_close")
    
    ma_fast = _get_val(indicators, fast_k)
    ma_mid  = _get_val(indicators, mid_k)
    
    # Dynamic Slow MA + Fallback
    ma_slow = _get_slow_ma(indicators, horizon)

    vwap = _get_val(indicators, "vwap")
    if vwap is None: vwap = ma_fast

    bb_upper = _get_val(indicators, "bb_high")
    rsi = _get_val(indicators, "rsi")
    
    macd_hist = _get_val(indicators, "macd_histogram")
    rvol = _get_val(indicators, "rvol")
    trend_strength = _get_val(indicators, "trend_strength")
    
    # Robust squeeze check
    is_squeeze = _is_squeeze_on(indicators)

    if not close: return "GENERIC"

    candidates = []

    # MOMENTUM BREAKOUT
    if (bb_upper and close >= (bb_upper * 0.98) and 
        (rsi and rsi > 65) and 
        (rvol and rvol > 1.8) and
        (trend_strength and trend_strength >= 7)): 
        candidates.append((100, "MOMENTUM_BREAKOUT"))

    # VOLATILITY SQUEEZE
    if is_squeeze:
        if (macd_hist and macd_hist > 0.2) or (rsi and rsi > 58):
            candidates.append((95, "VOLATILITY_SQUEEZE"))

    # PULLBACKS
    # Determine trend status (either via explicit slow MA or Trend Alignment Key)
    # This acts as the "fallback check" for trend if slow MA is missing
    is_uptrend = False
    if ma_slow and close > ma_slow:
        is_uptrend = True
    elif ma_slow is None:
        # Fallback: Check the explicit trend alignment key (e.g., ema_20_50_200_trend)
        trend_val = _get_val(indicators, ma_keys["trend"])
        if trend_val is not None and trend_val > 0:
            is_uptrend = True

    # Use dynamic slope
    slow_slope = _get_dynamic_slope(indicators, horizon)

    if is_uptrend and (slow_slope is None or slow_slope > -0.05):
        # Shallow Pullback to Fast MA
        if ma_fast and (ma_fast * 0.98) <= close <= (ma_fast * 1.02):
            if (prev_close is None or close > prev_close) and (rsi and rsi > 50): 
                 candidates.append((75, "TREND_PULLBACK"))
        # Deep Pullback to Mid MA
        elif ma_mid and ma_fast and (ma_fast > ma_mid) and (ma_mid * 0.98) <= close <= (ma_mid * 1.02):
            is_green_candle = (open_price is None) or (close > open_price)
            if rsi and rsi > 45 and is_green_candle:
                candidates.append((70, "DEEP_PULLBACK"))

    # OVERSOLD BOUNCES
    if (rsi and rsi < 30 and (rvol and rvol > 1.3)):
        if is_uptrend:
            if macd_hist is None or macd_hist >= -0.2: 
                candidates.append((60, "OVERSOLD_IN_UPTREND"))
        else:
            candidates.append((30, "OVERSOLD_REVERSAL"))

    # TREND FOLLOWING
    if (ma_fast and close > ma_fast):
        if rsi and rsi >= 55:
            if macd_hist and macd_hist >= 0.1:
                if trend_strength and trend_strength >= 6:
                    dist_to_ma = (close - ma_fast) / ma_fast
                    max_dist = 0.06 if trend_strength >= 8 else 0.04
                    if dist_to_ma < max_dist:  
                        candidates.append((40, "TREND_FOLLOWING"))

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
    
    vwap = _get_val(indicators, "vwap") or ma_fast
    rsi_slope = _get_val(indicators, "rsi_slope")
    macd_hist = _get_val(indicators, "macd_histogram")
    rvol = _get_val(indicators, "rvol")
    obv_div = _get_str(indicators, "obv_div")

    trend_score = 0
    if close and ma_slow and close > ma_slow: trend_score += 15
    if close and ma_mid and close > ma_mid: trend_score += 10
    if trend_strength and trend_strength > 25: trend_score += 15
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
                        horizon: str = "short_term") -> Dict[str, Any]:
    """
    Generate plan integrated with ATR_MULTIPLIERS from config.
    """
    final_score = profile_report.get("final_score", 0)
    category = profile_report.get("category", "HOLD")

    price = _get_val(indicators, "price")
    atr = _get_val(indicators, "atr_14")
    
    psar_trend = _get_str(indicators, "psar_trend")
    psar_level = _get_val(indicators, "psar_level")
    squeeze_signal = _get_str(indicators, "ttm_squeeze")

    r1 = _get_val(indicators, "resistance_1")
    r2 = _get_val(indicators, "resistance_2")
    r3 = _get_val(indicators, "resistance_3")
    s1 = _get_val(indicators, "support_1")
    s2 = _get_val(indicators, "support_2")
    s3 = _get_val(indicators, "support_3")

    # PASS HORIZON TO CLASSIFY SETUP
    setup_type = classify_setup(indicators, horizon=horizon)
    ts_val = _get_val(indicators, "trend_strength") or 0
    # PASS HORIZON TO CONFIDENCE CALC
    setup_conf = calculate_setup_confidence(indicators, ts_val, macro_trend_status, setup_type, horizon=horizon)

    can_trade_vol, vol_reason = should_trade_current_volatility(indicators, setup_type=setup_type)
    
    plan = {
        "signal": "NO_TRADE",
        "setup_type": setup_type,
        "setup_confidence": setup_conf,
        "position_size": 0.0,
        "reason": "Analysis Inconclusive",
        "entry": price,
        "stop_loss": None,
        "targets": {"t1": None, "t2": None},
        "rr_ratio": 0,
        "move_stop_to_breakeven_after_t1": False,
        "execution_hints": {},
        "analytics": {
            "volatility_quality": _get_val(indicators, "volatility_quality"),
            "trend_strength": ts_val,
            "macro_regime": macro_trend_status,
            "atr_pct": _get_val(indicators, "atr_pct"),
            "timestamp": datetime.now().isoformat()
        }
    }

    if price is None:
        plan["reason"] = "Data Error: Current Price Missing"
        return plan
    
    if not can_trade_vol and setup_type in ["MOMENTUM_BREAKOUT", "TREND_FOLLOWING"]:
        plan["signal"] = "WAIT"
        plan["reason"] = vol_reason
        return plan

    SETUP_CONFIDENCE_FLOORS = {
        "MOMENTUM_BREAKOUT": 70,
        "VOLATILITY_SQUEEZE": 65,
        "TREND_PULLBACK": 68,
        "DEEP_PULLBACK": 60, 
        "TREND_FOLLOWING": 72
    }
    floor = SETUP_CONFIDENCE_FLOORS.get(setup_type, 65)
    if setup_conf < floor:
        plan["signal"] = "WAIT"
        plan["reason"] = f"Confidence {setup_conf}% < {floor}% floor for {setup_type}"
        return plan

    mst = str(macro_trend_status or "").lower().strip()
    if mst in ("n/a", "na", "", "unknown", None):
        macro_bearish = macro_bullish = False
    else:
        macro_bearish = ("down" in mst or "bear" in mst)
        macro_bullish = ("up" in mst or "bull" in mst)

    is_squeeze = _is_squeeze_on(indicators)
    is_bullish_psar = "bull" in psar_trend
    is_bearish_psar = "bear" in psar_trend

    if atr is None or atr == 0:
        plan["signal"] = "HOLD_NO_RISK"
        plan["reason"] = "Missing ATR"
        return plan

    plan["position_size"] = calculate_position_size(indicators, setup_conf, setup_type, horizon=horizon)

    # ----------------------------------------------------
    # LONG LOGIC
    # ----------------------------------------------------
    if category == "BUY" and is_bullish_psar:
        if macro_bearish:
            plan["signal"] = "RISKY_BUY"
            plan["reason"] = "Score high but macro bearish - reduce size"
        else:
            plan["signal"] = "BUY_SQUEEZE" if is_squeeze else "BUY_TREND"
            plan["reason"] = f"Bullish Score {final_score} + Trend Alignment"

        # Horizon-aware ATR multipliers
        atr_cfg = ATR_MULTIPLIERS.get(horizon, ATR_MULTIPLIERS.get("shortterm", {"tp": 2.0, "sl": 1.0}))
        atr_sl_mult = atr_cfg.get("sl", 1.0)
        atr_tp_mult = atr_cfg.get("tp", 2.0)

        # prefer PSAR as SL if reasonable else ATR-based
        sl_theoretical = psar_level if (psar_level is not None and psar_level < price) else (price - (atr_sl_mult * atr))
        raw_risk_dist = price - sl_theoretical
        actual_risk = max(raw_risk_dist, atr * (atr_sl_mult))
        # MAX_SL_PCT fallback (configurable); default 5%
        MAX_SL_PCT = 0.05
        actual_risk = min(actual_risk, price * MAX_SL_PCT)

        plan["stop_loss"] = round(price - actual_risk, 4)
        # targets: prefer pivot if meaningful else ATR-based
        t1 = price + (actual_risk * 1.5)
        min_target = t1
        tgt_calc = min_target
        pivot_candidate = None
        for r in [r1, r2, r3]:
            if r is not None and r > min_target:
                pivot_candidate = r
                break
        if pivot_candidate:
            tgt_calc = pivot_candidate
        if tgt_calc <= t1:
            # robust ATR fallback
            tgt_calc = price + (atr * atr_tp_mult * 1.0)
            if tgt_calc <= t1:
                tgt_calc = t1 + (actual_risk * 0.5)

        plan["targets"]["t1"] = round(t1, 4)
        plan["targets"]["t2"] = round(tgt_calc, 4)
        plan["move_stop_to_breakeven_after_t1"] = True

        safe_risk = max(actual_risk, 1e-9)
        plan["rr_ratio"] = round((tgt_calc - price) / safe_risk, 2)

        plan["execution_hints"] = {
            "t1_desc": "1.5R take-profit; move stop to breakeven after hit",
            "t2_desc": "Pivot target or ATR-based fallback"
        }

        MIN_RR_BY_SETUP = {
            "MOMENTUM_BREAKOUT": 1.5,
            "VOLATILITY_SQUEEZE": 1.3,
            "TREND_PULLBACK": 1.4,
            "DEEP_PULLBACK": 1.2,
            "TREND_FOLLOWING": 1.5
        }
        min_rr = MIN_RR_BY_SETUP.get(setup_type, 1.3)
        if plan["rr_ratio"] < min_rr:
            plan["signal"] = "SKIPPED_LOW_RR"
            plan["reason"] = f"RR {plan['rr_ratio']} < {min_rr} required for {setup_type}"
            plan["analytics"]["skipped_low_rr"] = True
        return plan

    # ----------------------------------------------------
    # SHORT LOGIC
    # ----------------------------------------------------
    if category == "SELL" and is_bearish_psar:
        if macro_bullish:
            plan["signal"] = "RISKY_SHORT"
            plan["reason"] = "Score bearish but macro bullish - tight stops"
        else:
            plan["signal"] = "SHORT_SQUEEZE" if is_squeeze else "SHORT_TREND"
            plan["reason"] = f"Bearish Score {final_score} + Trend Breakdown"

        atr_cfg = ATR_MULTIPLIERS.get(horizon, ATR_MULTIPLIERS.get("shortterm", {"tp": 2.0, "sl": 1.0}))
        atr_sl_mult = atr_cfg.get("sl", 1.0)
        atr_tp_mult = atr_cfg.get("tp", 2.0)

        sl_theoretical = psar_level if (psar_level is not None and psar_level > price) else (price + (atr_sl_mult * atr))
        raw_risk = sl_theoretical - price
        actual_risk = max(raw_risk, atr * (atr_sl_mult))
        MAX_SL_PCT = 0.05
        actual_risk = min(actual_risk, price * MAX_SL_PCT)

        plan["stop_loss"] = round(price + actual_risk, 4)
        t1 = price - (actual_risk * 1.5)
        min_target = price - (actual_risk * 2.0)
        tgt_calc = min_target
        pivot_candidate = None
        for s in [s1, s2, s3]:
            if s is not None and s < min_target:
                pivot_candidate = s
                break
        if pivot_candidate:
            tgt_calc = pivot_candidate
        if tgt_calc >= t1:
            tgt_calc = price - (atr * atr_tp_mult * 1.0)
            if tgt_calc >= t1:
                tgt_calc = t1 - (actual_risk * 0.5)

        plan["targets"]["t1"] = round(t1, 4)
        plan["targets"]["t2"] = round(tgt_calc, 4)
        plan["move_stop_to_breakeven_after_t1"] = True

        safe_risk = max(actual_risk, 1e-9)
        plan["rr_ratio"] = round((price - tgt_calc) / safe_risk, 2)
        plan["execution_hints"] = {
            "t1_desc": "1.5R take-profit; move stop to breakeven after hit",
            "t2_desc": "Pivot support target or ATR fallback"
        }

        # [PHASE-1] Dynamic RR Thresholds
        MIN_RR_BY_SETUP = {
            "MOMENTUM_BREAKOUT": 1.5,
            "VOLATILITY_SQUEEZE": 1.3,
            "TREND_PULLBACK": 1.4,
            "DEEP_PULLBACK": 1.2,
            "TREND_FOLLOWING": 1.5
        }
        min_rr = MIN_RR_BY_SETUP.get(setup_type, 1.3)
        if plan["rr_ratio"] < min_rr:
            plan["signal"] = "SKIPPED_LOW_RR"
            plan["reason"] = f"RR {plan['rr_ratio']} < {min_rr} required for {setup_type}"
            plan["analytics"]["skipped_low_rr"] = True
        return plan
    
    plan["signal"] = "WAIT"
    if category == "HOLD":
        plan["reason"] = f"Score {final_score} is neutral. No clear edge."
    elif category == "BUY":
        plan["reason"] = "Fundamental Buy, but Technical trend not aligned"
    elif category == "SELL":
        plan["reason"] = "Fundamental Sell, but Technical trend not aligned"
    return plan

# ----------------------------------------------------------------------
# Meta-Category Scoring
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