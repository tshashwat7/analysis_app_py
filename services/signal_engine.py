# services/signal_engine_v3_phase4_final.py
"""
Signal Engine v3 Phase-4 Final

- Golden Master + Phase-3 features
- Adds Advanced Volatility Quality composite (toggleable)
- Removes redundant composite calculations in compute_all_profiles
- Keeps all previous hybrids, scoring, penalties, and trade-plan logic
- Implements Priority-Based Setup Classification with Context
- Fixes Squeeze False Positives
- Adds Deep Pullback (50EMA) Support
- Distinguishes Trend Dips vs Counter-Trend Reversals
- 3-Factor Confidence Model with VWAP Fallback & Macro Sensitivity

"""

from typing import Dict, Any, Optional, Tuple, List
import math
import statistics
import operator
import logging
from datetime import datetime

from config.constants import MACD_MOMENTUM_THRESH, RSI_SLOPE_THRESH, TREND_THRESH, VOL_BANDS
from services.data_fetch import _safe_float, _safe_get_raw_float

# Import constants; ENABLE_VOLATILITY_QUALITY fallback
try:
    from config.constants import (
        HORIZON_PROFILE_MAP, VALUE_WEIGHTS, GROWTH_WEIGHTS, 
        QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, ENABLE_VOLATILITY_QUALITY
    )
except Exception:
    HORIZON_PROFILE_MAP = {}
    VALUE_WEIGHTS = {}
    GROWTH_WEIGHTS = {}
    QUALITY_WEIGHTS = {}
    MOMENTUM_WEIGHTS = {}
    ENABLE_VOLATILITY_QUALITY = False

logger = logging.getLogger(__name__)


# -------------------------
# Helpers
# -------------------------


def _coerce_score_field(metric_entry: Dict[str, Any]) -> Optional[float]:
    if not metric_entry:
        return None
    s = metric_entry.get("score")
    if s is not None:
        sf = _safe_float(s)
        return float(sf) if sf is not None else None
    raw = metric_entry.get("raw")
    rv = _safe_float(raw)
    if rv is not None and 0 <= rv <= 10:
        return rv
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
# Hybrid metrics (kept from v3)
# -------------------------
def enrich_hybrid_metrics(fundamentals: dict, indicators: dict) -> dict:
    hybrids = {}

    # Volatility-Adjusted ROE
    roe = _safe_float(fundamentals.get("roe", {}).get("raw") if isinstance(fundamentals.get("roe"), dict) else fundamentals.get("roe"))
    atr_pct = _safe_float(indicators.get("atr_pct", {}).get("value") if isinstance(indicators.get("atr_pct"), dict) else indicators.get("atr_pct"))
    if roe is not None and atr_pct is not None and atr_pct > 0:
        ratio = roe / atr_pct
        score = 10 if ratio >= 10 else 7 if ratio >= 5 else 3 if ratio >= 2 else 0
        hybrids["volatility_adjusted_roe"] = {
            "raw": ratio,
            "value": round(ratio, 2),
            "score": score,
            "desc": f"ROE/Vol = {ratio:.2f} (ROE {roe:.1f}%, ATR {atr_pct:.2f}%)",
            "alias": "Volatility-Adjusted ROE",
            "source": "hybrid",
        }

    # Price vs Intrinsic Value
    pe = _safe_float(fundamentals.get("pe_ratio", {}).get("raw") if isinstance(fundamentals.get("pe_ratio"), dict) else fundamentals.get("pe_ratio"))
    eps_growth = _safe_float(fundamentals.get("eps_growth_5y", {}).get("raw") if isinstance(fundamentals.get("eps_growth_5y"), dict) else fundamentals.get("eps_growth_5y"))
    price = _safe_float(indicators.get("Price", {}).get("value") if isinstance(indicators.get("Price"), dict) else indicators.get("Price")) or _safe_float(fundamentals.get("current_price"))
    if price is not None and pe is not None and eps_growth is not None and eps_growth > 0:
        try:
            intrinsic_value = price * (1 / (pe / eps_growth))
            ratio = price / intrinsic_value if intrinsic_value != 0 else float("inf")
            score = 10 if ratio < 0.8 else 7 if ratio < 1.0 else 3 if ratio < 1.2 else 0
            hybrids["price_vs_intrinsic_value"] = {
                "raw": ratio,
                "value": round(ratio, 2),
                "score": score,
                "desc": f"Price/IV = {ratio:.2f} ({'Undervalued' if ratio < 1 else 'Overvalued'})",
                "alias": "Price vs Intrinsic Value",
                "source": "hybrid",
            }
        except Exception:
            pass

    # FCF Yield vs Volatility
    fcf_yield = _safe_float(fundamentals.get("fcf_yield", {}).get("raw") if isinstance(fundamentals.get("fcf_yield"), dict) else fundamentals.get("fcf_yield"))
    if fcf_yield is not None and atr_pct is not None:
        ratio = fcf_yield / max(atr_pct, 0.1)
        score = 10 if ratio >= 10 else 8 if ratio >= 5 else 5 if ratio >= 2 else 2
        hybrids["fcf_yield_vs_volatility"] = {
            "raw": ratio,
            "value": round(ratio, 2),
            "score": score,
            "desc": f"FCF/Vol = {ratio:.2f} (Yield {fcf_yield:.2f}%, ATR {atr_pct:.2f}%)",
            "alias": "FCF Yield vs Volatility",
            "source": "hybrid",
        }

    # Trend Consistency (ADX + Supertrend)
    adx = _safe_float(indicators.get("adx", {}).get("value") if isinstance(indicators.get("adx"), dict) else indicators.get("adx"))
    supertrend = str(indicators.get("supertrend_signal", {}).get("raw") if isinstance(indicators.get("supertrend_signal"), dict) else indicators.get("supertrend_signal") or "").lower()
    if adx is not None:
        score = 10 if adx >= 25 else 7 if adx >= 20 else 4
        if "bull" in supertrend:
            score = min(10, score + 1)
        hybrids["trend_consistency"] = {
            "raw": adx,
            "value": adx,
            "score": min(10, score),
            "desc": f"ADX {adx:.1f} ({'Bullish' if 'bull' in supertrend else 'Neutral'})",
            "alias": "Trend Consistency",
            "source": "hybrid",
        }

    # Price vs 200DMA
    dma_200 = _safe_float(indicators.get("dma_200", {}).get("value") if isinstance(indicators.get("dma_200"), dict) else indicators.get("dma_200"))
    if price is not None and dma_200 is not None:
        ratio = (price / dma_200) - 1
        score = 10 if ratio > 0.1 else 7 if ratio > 0.0 else 3 if ratio > -0.05 else 0
        hybrids["price_vs_200dma"] = {
            "raw": ratio,
            "value": round(ratio * 100, 2),
            "score": score,
            "desc": f"Price vs 200DMA: {ratio*100:.2f}%",
            "alias": "Price vs 200DMA %",
            "source": "hybrid",
        }

    # Fundamental Momentum
    q_growth = _safe_float(fundamentals.get("quarterly_growth", {}).get("raw") if isinstance(fundamentals.get("quarterly_growth"), dict) else fundamentals.get("quarterly_growth"))
    eps_5y = _safe_float(fundamentals.get("eps_growth_5y", {}).get("raw") if isinstance(fundamentals.get("eps_growth_5y"), dict) else fundamentals.get("eps_growth_5y"))
    if q_growth is not None and eps_5y is not None:
        ratio = (q_growth + eps_5y / 5) / 2
        score = 10 if ratio >= 15 else 7 if ratio >= 10 else 4 if ratio >= 5 else 1
        hybrids["fundamental_momentum"] = {
            "raw": ratio,
            "value": round(ratio, 2),
            "score": score,
            "desc": f"Growth Momentum = {ratio:.2f}%",
            "alias": "Fundamental Momentum",
            "source": "hybrid",
        }

    # Earnings Consistency Index
    net_margin = _safe_float(fundamentals.get("net_profit_margin", {}).get("raw") if isinstance(fundamentals.get("net_profit_margin"), dict) else fundamentals.get("net_profit_margin"))
    if roe is not None and net_margin is not None:
        ratio = (roe + net_margin) / 2
        score = 10 if ratio >= 25 else 7 if ratio >= 15 else 4 if ratio >= 8 else 1
        hybrids["earnings_consistency_index"] = {
            "raw": ratio,
            "value": round(ratio, 2),
            "score": score,
            "desc": f"Consistency Index = {ratio:.2f}",
            "alias": "Earnings Consistency Index",
            "source": "hybrid",
        }

    return hybrids


# -------------------------
# Composite helpers (trend, momentum, roe stability)
# -------------------------
def compute_trend_strength(indicators: Dict[str, Any]) -> Dict[str, Any]:
    try:
        adx = _safe_float(indicators.get("adx", {}).get("value") if isinstance(indicators.get("adx"), dict) else indicators.get("adx"))
        ema_slope = _safe_float(indicators.get("ema_slope", {}).get("value") if isinstance(indicators.get("ema_slope"), dict) else indicators.get("ema_slope"))
        if ema_slope is None:
            ema_slope = _safe_float(indicators.get("ema_slope_20_50", {}).get("value") if isinstance(indicators.get("ema_slope_20_50"), dict) else indicators.get("ema_slope_20_50"))
        di_plus = _safe_float(indicators.get("di_plus", {}).get("value") if isinstance(indicators.get("di_plus"), dict) else indicators.get("di_plus"))
        di_minus = _safe_float(indicators.get("di_minus", {}).get("value") if isinstance(indicators.get("di_minus"), dict) else indicators.get("di_minus"))
        supertrend = str(indicators.get("supertrend_signal", {}).get("raw") if isinstance(indicators.get("supertrend_signal"), dict) else indicators.get("supertrend_signal") or "").lower()

        adx_score = 0.0
        if adx is not None:
            # 🆕 FIX: Use TREND_THRESH constants instead of hardcoded numbers
            if adx >= TREND_THRESH["strong_floor"]:
                adx_score = 10.0
            elif adx >= TREND_THRESH["moderate_floor"]:
                adx_score = 8.0
            elif adx >= TREND_THRESH["weak_floor"]:
                adx_score = 4.0
            else: # Below weak_floor (20.0)
                adx_score = 2.0

        ema_score = 0.0
        if ema_slope is not None:
            v = abs(ema_slope)
            if v >= 2.0:
                ema_score = 10.0
            elif v >= 1.0:
                ema_score = 8.0
            elif v >= 0.5:
                ema_score = 6.0
            elif v >= 0.2:
                ema_score = 4.0
            else:
                ema_score = 2.0

        di_score = 0.0
        if di_plus is not None and di_minus is not None:
            spread = di_plus - di_minus
            # 🆕 FIX: Use TREND_THRESH["di_spread_strong"]
            if spread >= TREND_THRESH["di_spread_strong"]: 
                di_score = 10.0
            elif spread >= 10: # Remaining threshold kept hardcoded or moved to constants
                di_score = 7.0
            elif spread >= 0:
                di_score = 5.0
            else:
                di_score = 2.0

        st_score = 1.0
        if "bull" in supertrend:
            st_score = 2.0
        elif "bear" in supertrend:
            st_score = 0.0
        else:
            st_score = 1.0

        components = [
            ("adx", adx_score, 0.4),
            ("ema", ema_score, 0.3),
            ("di", di_score, 0.2),
            ("st", st_score, 0.1),
        ]
        raw = sum(val * w for (_, val, w) in components)
        score = max(0.0, min(10.0, round(raw, 2)))
        desc = f"Trend Strength (ADX {adx}, EMA_slope {ema_slope}, DI {di_plus}-{di_minus})"
        return {"raw": raw, "value": score, "score": int(round(score)), "desc": desc, "alias": "trend_strength", "source": "composite"}
    except Exception as e:
        logger.debug("Trend strength compute error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "trend_strength", "source": "composite"}

def compute_momentum_strength(indicators: Dict[str, Any]) -> Dict[str, Any]:
    try:
        rsi = _safe_get_raw_float(indicators.get("rsi"))
        rsi_slope = _safe_get_raw_float(indicators.get("rsi_slope"))
        macd_hist = _safe_get_raw_float(indicators.get("macd_histogram"))
        stoch_k = _safe_get_raw_float(indicators.get("stoch_k"))
        stoch_d = _safe_get_raw_float(indicators.get("stoch_d"))
        rsi_score = 5.0
        if rsi is not None:
            if rsi >= 70:
                rsi_score = 8.0
            elif rsi >= 60:
                rsi_score = 7.0
            elif rsi >= 50:
                rsi_score = 5.0
            elif rsi >= 40:
                rsi_score = 4.0
            else:
                rsi_score = 2.0

        slope_score = 5.0
        if rsi_slope is not None:
            # 🆕 FIX: Use RSI_SLOPE_THRESH
            if rsi_slope >= RSI_SLOPE_THRESH["acceleration_floor"]: 
                slope_score = 8.0
            elif rsi_slope >= 0.0:
                slope_score = 4.0
            elif rsi_slope <= RSI_SLOPE_THRESH["deceleration_ceiling"]:
                slope_score = 2.0

        macd_score = 5.0
        if macd_hist is not None:
            # 🆕 FIX: Use MACD_MOMENTUM_THRESH (assuming z-score logic is now applied to macd_hist)
            if macd_hist >= MACD_MOMENTUM_THRESH["acceleration_floor"]:
                macd_score = 8.0
            elif macd_hist <= MACD_MOMENTUM_THRESH["deceleration_ceiling"]:
                macd_score = 2.0

        stoch_score = 5.0
        if stoch_k is not None and stoch_d is not None:
            if stoch_k > stoch_d and stoch_k >= 50:
                stoch_score = 8.0
            elif stoch_k > stoch_d:
                stoch_score = 6.0
            elif stoch_k < stoch_d:
                stoch_score = 3.0

        raw = (rsi_score * 0.25) + (slope_score * 0.25) + (macd_score * 0.30) + (stoch_score * 0.20)
        score = max(0.0, min(10.0, round(raw, 2)))
        desc = f"Momentum (RSI {rsi}, RSI_slope {rsi_slope}, MACD_hist {macd_hist})"
        return {"raw": raw, "value": score, "score": int(round(score)), "desc": desc, "alias": "momentum_strength", "source": "composite"}
    except Exception as e:
        logger.debug("Momentum compute error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "momentum_strength", "source": "composite"}

def compute_roe_stability(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    try:
        history = fundamentals.get("roe_history")
        if isinstance(history, list) and len(history) >= 3:
            vals = [v for v in ([_safe_float(x) for x in history]) if v is not None]
        else:
            r5 = fundamentals.get("roe_5y")
            if isinstance(r5, dict):
                vals = [v for v in (_safe_float(x) for x in r5.values()) if v is not None]
            else:
                vals = []

        if not vals:
            return {"raw": None, "value": None, "score": None, "desc": "No ROE history", "alias": "roe_stability", "source": "composite"}

        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        if std < 2.0:
            score = 10
        elif std < 4.0:
            score = 8
        elif std < 7.0:
            score = 5
        else:
            score = 1

        desc = f"ROE stability stddev={std:.2f}"
        return {"raw": std, "value": round(std, 2), "score": int(score), "desc": desc, "alias": "roe_stability", "source": "composite"}
    except Exception as e:
        logger.debug("ROE stability error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "roe_stability", "source": "composite"}

def compute_volatility_quality(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced volatility quality composite (0-10).
    Inputs (preferred indicator keys):
      - atr_pct            indicators["atr_pct"] or indicators["atr_pct"]["value"]
      - bb_width           indicators["bb_width"] or indicators["bb_width"]["value"]
      - true_range         indicators["true_range"] or indicators["true_range"]["value"]
      - hv_10, hv_20       indicators["hv_10"], indicators["hv_20"] (historical vol in %)
      - atr_sma_ratio      indicators["atr_sma_ratio"] (ATR / ATR_SMA)
      - bbp (bollinger percent) optional
      - squeeze_flag       indicators["ttm_squeeze"] optional
    Returns metric object.
    """
    try:
        # fetch raw inputs with defensive access
        atr_pct = _safe_float(indicators.get("atr_pct", {}).get("value") if isinstance(indicators.get("atr_pct"), dict) else indicators.get("atr_pct"))
        bb_width = _safe_float(indicators.get("bb_width", {}).get("value") if isinstance(indicators.get("bb_width"), dict) else indicators.get("bb_width"))
        true_range = _safe_float(indicators.get("true_range", {}).get("value") if isinstance(indicators.get("true_range"), dict) else indicators.get("true_range"))
        hv10 = _safe_float(indicators.get("hv_10", {}).get("value") if isinstance(indicators.get("hv_10"), dict) else indicators.get("hv_10"))
        hv20 = _safe_float(indicators.get("hv_20", {}).get("value") if isinstance(indicators.get("hv_20"), dict) else indicators.get("hv_20"))
        atr_sma_ratio = _safe_float(indicators.get("atr_sma_ratio", {}).get("value") if isinstance(indicators.get("atr_sma_ratio"), dict) else indicators.get("atr_sma_ratio"))
        bbp = _safe_float(indicators.get("bbp", {}).get("value") if isinstance(indicators.get("bbp"), dict) else indicators.get("bbp"))
        squeeze_flag = str(indicators.get("ttm_squeeze", {}).get("value") if isinstance(indicators.get("ttm_squeeze"), dict) else indicators.get("ttm_squeeze") or "").lower()
        LOW_VOL = VOL_BANDS["low_vol_ceiling"]          # 1.0%
        MODERATE_VOL = VOL_BANDS["moderate_vol_ceiling"] # 2.5%
        HIGH_VOL = VOL_BANDS["high_vol_floor"]

        # normalize/score components to 0-10
        # 1) ATR% score (high ATR% -> lower quality)
        atr_score = 5.0
        if atr_pct is not None:
            # 🆕 FIX: Use VOL_BANDS
            if atr_pct <= LOW_VOL:
                atr_score = 10.0
            elif atr_pct <= MODERATE_VOL:
                atr_score = 8.0
            elif atr_pct <= HIGH_VOL: # Standard volatility
                atr_score = 6.0
            elif atr_pct <= HIGH_VOL + 1.5: # Allows a small buffer above floor
                atr_score = 4.0
            else:
                atr_score = 2.0

        # 2) BB Width score (tight BB -> higher quality)
        bbw_score = 5.0
        if bb_width is not None:
            # smaller bandwidth -> higher score, scaled heuristically
            if bb_width <= 0.01:
                bbw_score = 10.0
            elif bb_width <= 0.02:
                bbw_score = 8.0
            elif bb_width <= 0.04:
                bbw_score = 6.0
            elif bb_width <= 0.08:
                bbw_score = 4.0
            else:
                bbw_score = 2.0

        # 3) True Range compression: compare true_range to atr_pct (if both present)
        tr_score = 5.0
        if true_range is not None and atr_pct is not None and atr_pct > 0:
            ratio = true_range / max(atr_pct, 1e-9)
            # ratio <<1 -> compressed; ratio >>1 -> expanding volatility
            if ratio <= 0.5:
                tr_score = 10.0
            elif ratio <= 1.0:
                tr_score = 8.0
            elif ratio <= 1.5:
                tr_score = 6.0
            elif ratio <= 2.0:
                tr_score = 4.0
            else:
                tr_score = 2.0
        elif true_range is not None:
            # fallback using absolute true_range
            v = true_range
            if v <= 0.5:
                tr_score = 10.0
            elif v <= 1.0:
                tr_score = 8.0
            elif v <= 2.0:
                tr_score = 6.0
            else:
                tr_score = 3.0

        # 4) Historical Volatility regime (HV10 vs HV20) - lower HV and HV10 < HV20 (calmer) is good
        hv_score = 5.0
        if hv10 is not None and hv20 is not None:
            # If hv10 < hv20 -> compressing -> good
            if hv10 < hv20 and hv20 < 25:
                hv_score = 10.0
            elif hv10 < hv20:
                hv_score = 8.0
            elif hv10 <= hv20:
                hv_score = 6.0
            else:
                hv_score = 3.0
        elif hv20 is not None:
            # single long term hv
            if hv20 < LOW_VOL * 10: # e.g., < 10%
                hv_score = 10.0
            elif hv20 < MODERATE_VOL * 10: # e.g., < 25%
                hv_score = 8.0
            elif hv20 < HIGH_VOL * 10: # e.g., < 40%
                hv_score = 5.0
            else:
                hv_score = 2.0

        # 5) ATR_SMA ratio (if ATR is below its SMA -> compression)
        atr_sma_score = 5.0
        if atr_sma_ratio is not None:
            if atr_sma_ratio < 0.8:
                atr_sma_score = 10.0
            elif atr_sma_ratio < 0.95:
                atr_sma_score = 8.0
            elif atr_sma_ratio < 1.05:
                atr_sma_score = 6.0
            else:
                atr_sma_score = 3.0

        # 6) Squeeze penalty: if squeeze is "on" (i.e. low BB but building), we should boost quality for "setup" but be cautious:
        squeeze_bonus = 0.0
        if "on" in squeeze_flag or "sqz" in squeeze_flag:
            # if it's a squeeze (narrow BB), it's a high-quality consolidation for breakouts
            squeeze_bonus = 0.5  # small boost

        # Weights (tunable): give importance to BB width, HV regime, TR compression, ATR_SMA, ATR%
        raw = (
            (bbw_score * 0.30) +
            (hv_score * 0.25) +
            (tr_score * 0.20) +
            (atr_sma_score * 0.15) +
            (atr_score * 0.10)
        ) / 1.0

        # Apply tiny squeeze bonus (keeps raw in 0-10 range)
        raw = raw + squeeze_bonus
        score = max(0.0, min(10.0, round(raw, 2)))
        desc = f"VolQuality(atr%={atr_pct}, bbw={bb_width}, tr={true_range}, hv10={hv10}, hv20={hv20})"
        return {"raw": raw, "value": score, "score": int(round(score)), "desc": desc, "alias": "volatility_quality", "source": "composite"}
    except Exception as e:
        logger.debug("Volatility quality compute error: %s", e)
        return {"raw": None, "value": None, "score": None, "desc": "err", "alias": "volatility_quality", "source": "composite"}


# -------------------------
# Scoring primitives and penalty engine (kept from v3)
# -------------------------
def _get_metric_entry(key: str, fundamentals: Dict[str, Any], indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    if indicators and key in indicators:
        return indicators.get(key)
    if fundamentals and key in fundamentals:
        return fundamentals.get(key)
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
                    weight_sum += weight
                continue

            score_val = _coerce_score_field(entry)
            if score_val is None:
                raw = entry.get("raw")
                if isinstance(raw, str):
                    r = raw.lower()
                    if r in ("strong_buy", "buy", "bullish"):
                        score_val = 8.5
                    elif r in ("hold", "neutral"):
                        score_val = 5.0
                    elif r in ("sell", "bearish", "strong_sell"):
                        score_val = 1.5

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
        logger.warning("No metric weights available (weight_sum==0). Returning safe defaults.")
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
# Profile computation injection (composites computed here only)
# -------------------------
def compute_profile_score(profile_name, fundamentals, indicators, profile_map=None):
    profile = (profile_map or HORIZON_PROFILE_MAP).get(profile_name)
    if not profile:
        raise KeyError(f"Profile '{profile_name}' not defined")

    # Compute composites if missing (single place)
    if "trend_strength" not in indicators:
        indicators["trend_strength"] = compute_trend_strength(indicators)
    if "momentum_strength" not in indicators:
        indicators["momentum_strength"] = compute_momentum_strength(indicators)
    if "roe_stability" not in fundamentals:
        fundamentals["roe_stability"] = compute_roe_stability(fundamentals)
    # Volatility composite is toggleable
    if ENABLE_VOLATILITY_QUALITY and "volatility_quality" not in indicators:
        indicators["volatility_quality"] = compute_volatility_quality(indicators)

    metrics_map = profile.get("metrics", {})
    penalties_map = profile.get("penalties", {})
    thresholds = profile.get("thresholds", {"buy": 8, "hold": 6, "sell": 4})

    weighted_sum, weight_sum, metric_details = _compute_weighted_score(
        metrics_map, fundamentals, indicators
    )

    base_score = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
    base_score = round(base_score, 2)

    penalty_total, applied_penalties = _apply_penalties(
        penalties_map, fundamentals, indicators
    )

    penalty_total = min(max(penalty_total, 0.0), 0.95)
    final_score = base_score * (1.0 - penalty_total)
    final_score = max(0.0, min(10.0, round(final_score, 2)))

    if final_score >= thresholds["buy"]:
        cat = "BUY"
    elif final_score >= thresholds["hold"]:
        cat = "HOLD"
    else:
        cat = "SELL"

    return {
        "profile": profile_name,
        "base_score": base_score,
        "final_score": final_score,
        "category": cat,
        "metric_details": metric_details,
        "penalty_total": round(penalty_total, 4),
        "applied_penalties": applied_penalties,
        "thresholds": thresholds,
        "notes": profile.get("notes", "")
    }


def compute_all_profiles(ticker: str, fundamentals: Dict[str, Any], indicators: Dict[str, Any], profile_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    compute_all_profiles no longer pre-computes composites redundantly.
    compute_profile_score will compute necessary composites (trend/momentum/roe_stability/volatility) once per profile invocation.
    """
    pm = profile_map or HORIZON_PROFILE_MAP

    # enrich hybrids (v1)
    hybrids = enrich_hybrid_metrics(fundamentals or {}, indicators or {})
    if hybrids:
        fundamentals.update(hybrids)

    profiles_out = {}
    best_fit, best_score = None, -1.0

    for pname in pm.keys():
        try:
            out = compute_profile_score(pname, fundamentals, indicators, profile_map=pm)
            logger.debug(f"{ticker} | {pname.upper():<12} | base={out.get('base_score'):>4} | final={out.get('final_score'):>4} | {out.get('category')}")
        except Exception as e:
            logger.exception("Failed compute_profile_score %s for %s: %s", pname, ticker, e)
            out = {"profile": pname, "error": str(e)}
        profiles_out[pname] = out
        fs = out.get("final_score") or 0.0
        if fs > best_score:
            best_score = fs
            best_fit = pname

    avg_signal = (
        round(sum(p.get("final_score", 0) for p in profiles_out.values()) / len(profiles_out), 2)
        if profiles_out else 0.0
    )

    summary = {
        "ticker": ticker,
        "best_fit": best_fit,
        "best_score": best_score,
        "aggregate_signal": avg_signal,
        "profiles": profiles_out,
    }
    return summary


# -------------------------
# Trade plan (unchanged from v3, includes T1/T2 handling previously added)
# -------------------------

def classify_setup(indicators: Dict[str, Any]) -> str:
    """
    Determines trading setup using a Priority Queue system.
    Evaluates ALL conditions and picks the highest-priority match.
    """
    # 1. Extract Metrics
    close = _safe_float(indicators.get("Price", {}).get("value"))
    ema20 = _safe_float(indicators.get("20 EMA", {}).get("value"))
    ema50 = _safe_float(indicators.get("50 EMA", {}).get("value"))
    ema200 = _safe_float(indicators.get("200 EMA", {}).get("value"))
    bb_upper = _safe_float(indicators.get("BB High", {}).get("value"))
    rsi = _safe_float(indicators.get("RSI", {}).get("value"))
    macd_hist = _safe_float(indicators.get("MACD Histogram (Raw Momentum)", {}).get("value"))
    squeeze = str(indicators.get("ttm_squeeze", {}).get("value") or "").lower().strip()
    rvol = _safe_float(indicators.get("Relative Volume (RVOL)", {}).get("value"))
    
    if not close: return "GENERIC"

    # 2. Define Candidates List: (Priority, SetupName)
    candidates = []

    # --- A. MOMENTUM BREAKOUT (Priority: 100) ---
    # Strict: Price High + High RSI + Volume Expansion
    if (bb_upper and close >= (bb_upper * 0.98) and 
        (rsi and rsi > 60) and 
        (rvol and rvol > 1.2)):
        candidates.append((100, "MOMENTUM_BREAKOUT"))

    # --- B. VOLATILITY SQUEEZE (Priority: 90) ---
    # Strict String Match (Fixes Issue 3)
    if squeeze in ["squeeze on", "on", "squeeze_on"]:
        candidates.append((90, "VOLATILITY_SQUEEZE"))

    # --- C. PULLBACKS (Priority: 70-75) ---
    # Context: Must be in Long-Term Uptrend (Above 200 EMA)
    if ema200 and close > ema200:
        # 1. Shallow Pullback (Priority 75) - Dip to 20 EMA zone
        if ema20 and (ema20 * 0.97) <= close <= (ema20 * 1.01):
            candidates.append((75, "TREND_PULLBACK"))
        
        # 2. Deep Pullback (Priority 70) - Dip to 50 EMA zone (Fix Issue 1)
        elif ema50 and (ema50 * 0.97) <= close <= (ema50 * 1.01):
            candidates.append((70, "DEEP_PULLBACK"))

    # --- D. OVERSOLD BOUNCE (Priority: 60 vs 30) ---
    # Context: Panic Selling (RSI < 30 + High Volume)
    if (rsi and rsi < 30 and (rvol and rvol > 1.3)):
        # High Quality: Oversold in Uptrend (Buy the Dip) - Fix Issue 2
        if ema200 and close > ema200:
            candidates.append((60, "OVERSOLD_IN_UPTREND"))
        # Risky: Oversold in Downtrend (Falling Knife)
        else:
            candidates.append((30, "OVERSOLD_REVERSAL"))

    # --- E. TREND FOLLOWING (Priority: 40) ---
    # Strict: Price > 20EMA AND Momentum Confirmations (Fix Issue 4)
    if (ema20 and close > ema20 and 
        (rsi and rsi > 50) and 
        (macd_hist and macd_hist > 0)):
        candidates.append((40, "TREND_FOLLOWING"))

    # --- F. DEFAULT ---
    candidates.append((10, "NEUTRAL / CHOPPY"))

    # 3. Pick Winner
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# 🚀 FINAL: 3-FACTOR CONFIDENCE MODEL
# ==============================================================================

def calculate_setup_confidence(indicators: Dict[str, Any], 
                               trend_strength: float, 
                               macro_trend_status: str = "N/A",
                               setup_type: str = "GENERIC") -> int:
    """
    Calculates a weighted confidence score (0-100%) with proper fallbacks.
    """
    # Extract Helpers
    close = _safe_float(indicators.get("Price", {}).get("value"))
    ema20 = _safe_float(indicators.get("20 EMA", {}).get("value"))
    ema50 = _safe_float(indicators.get("50 EMA", {}).get("value"))
    ema200 = _safe_float(indicators.get("200 EMA", {}).get("value"))
    
    # Fix Issue 5: VWAP Fallback
    vwap = _safe_float(indicators.get("VWAP", {}).get("value"))
    if vwap is None or vwap == 0:
        vwap = ema20 # Use 20 EMA as proxy for intraday average control
        
    rsi_slope = _safe_float(indicators.get("RSI Slope", {}).get("value")) 
    macd_hist = _safe_float(indicators.get("MACD Histogram (Raw Momentum)", {}).get("value"))
    rvol = _safe_float(indicators.get("Relative Volume (RVOL)", {}).get("value"))
    obv_div = str(indicators.get("OBV Divergence", {}).get("value") or "").lower()

    # --- COMPONENT A: TREND CONFIDENCE (Max 40) ---
    trend_score = 0
    if close and ema200 and close > ema200: trend_score += 15   # Long-term alignment
    if close and ema50 and close > ema50: trend_score += 10     # Mid-term alignment
    if trend_strength and trend_strength > 25: trend_score += 15 # ADX Strength
    trend_score = min(40, trend_score)

    # --- COMPONENT B: MOMENTUM CONFIDENCE (Max 40) ---
    mom_score = 0
    if macd_hist and macd_hist > 0: mom_score += 15             # Momentum Positive
    if close and vwap and close > vwap: mom_score += 10         # Buyers in control
    if rsi_slope and rsi_slope > 0: mom_score += 10             # Momentum Accelerating
    # Volatility Quality Bonus
    vol_qual = _safe_float(indicators.get("volatility_quality", {}).get("value"))
    if vol_qual and vol_qual > 6: mom_score += 5
    mom_score = min(40, mom_score)

    # --- COMPONENT C: VOLUME CONFIDENCE (Max 20) ---
    vol_score = 0
    if rvol and rvol > 1.0: vol_score += 10
    if rvol and rvol > 2.0: vol_score += 5
    if "confirm" in obv_div or "bull" in obv_div: vol_score += 5
    vol_score = min(20, vol_score)

    # --- TOTAL RAW SCORE ---
    # Formula: 40% Trend + 40% Momentum + 20% Volume
    total_conf = trend_score + mom_score + vol_score

    # --- MACRO & BIAS ADJUSTMENTS ---
    # Fix Issue 6: Expanded Bearish Keywords
    macro = (macro_trend_status or "").lower()
    is_bearish_macro = any(x in macro for x in ["bear", "down", "corrective", "distribution", "weak"])
    
    if is_bearish_macro:
        # If macro is weak, punish bullish setups
        if setup_type in ["MOMENTUM_BREAKOUT", "TREND_FOLLOWING", "VOLATILITY_SQUEEZE"]:
            total_conf *= 0.85  # -15% Penalty

    # Setup Boost (Feedback Loop)
    boost_map = {
        "MOMENTUM_BREAKOUT": 1.10,
        "TREND_PULLBACK": 1.07,
        "DEEP_PULLBACK": 1.05,
        "OVERSOLD_IN_UPTREND": 1.05,
        "VOLATILITY_SQUEEZE": 1.05,
        "OVERSOLD_REVERSAL": 0.90 # Penalty for counter-trend
    }
    boost = boost_map.get(setup_type, 1.0)
    
    final_conf = int(total_conf * boost)
    return min(100, max(0, final_conf))




def generate_trade_plan(profile_report: Dict[str, Any],
                        indicators: Dict[str, Any],
                        macro_trend_status: str = "N/A") -> Dict[str, Any]:
    final_score = profile_report.get("final_score", 0)
    category = profile_report.get("category", "HOLD")

    price = _safe_float(indicators.get("Price", {}).get("value") if isinstance(indicators.get("Price"), dict) else indicators.get("Price"))
    atr = _safe_float(indicators.get("atr_14", {}).get("value") if isinstance(indicators.get("atr_14"), dict) else indicators.get("atr_14"))
    adx = _safe_float(indicators.get("adx", {}).get("value") if isinstance(indicators.get("adx"), dict) else indicators.get("adx"))

    psar_trend = str(indicators.get("psar_trend", {}).get("value") if isinstance(indicators.get("psar_trend"), dict) else indicators.get("psar_trend") or "").lower()
    psar_level = _safe_float(indicators.get("psar_level", {}).get("value") if isinstance(indicators.get("psar_level"), dict) else indicators.get("psar_level"))
    squeeze_signal = str(indicators.get("ttm_squeeze", {}).get("value") if isinstance(indicators.get("ttm_squeeze"), dict) else indicators.get("ttm_squeeze") or "").lower()

    r1 = _safe_float(indicators.get("resistance_1", {}).get("value") if isinstance(indicators.get("resistance_1"), dict) else indicators.get("resistance_1"))
    r2 = _safe_float(indicators.get("resistance_2", {}).get("value") if isinstance(indicators.get("resistance_2"), dict) else indicators.get("resistance_2"))
    r3 = _safe_float(indicators.get("resistance_3", {}).get("value") if isinstance(indicators.get("resistance_3"), dict) else indicators.get("resistance_3"))
    s1 = _safe_float(indicators.get("support_1", {}).get("value") if isinstance(indicators.get("support_1"), dict) else indicators.get("support_1"))
    s2 = _safe_float(indicators.get("support_2", {}).get("value") if isinstance(indicators.get("support_2"), dict) else indicators.get("support_2"))
    s3 = _safe_float(indicators.get("support_3", {}).get("value") if isinstance(indicators.get("support_3"), dict) else indicators.get("support_3"))

# --- NEW: Run Advanced Logic ---
    setup_type = classify_setup(indicators)
    ts_val = _safe_float(indicators.get("trend_strength", {}).get("value"))
    setup_conf = calculate_setup_confidence(indicators, ts_val, macro_trend_status, setup_type)
    
    loggable_data = {
        "type": setup_type,
        "confidence": setup_conf,
        "timestamp": datetime.now().isoformat(),
        "trend_score": ts_val,
        "macro": macro_trend_status
    }

    plan = {
        "signal": "NO_TRADE",
        "setup_type": setup_type,       
        "setup_confidence": setup_conf, 
        "log_data": loggable_data,
        "reason": "Analysis Inconclusive",
        "entry": price, # Fallback
        "stop_loss": None,
        "targets": {"t1": None, "t2": None},
        "rr_ratio": 0,
        "move_stop_to_breakeven_after_t1": False,
        "execution_hints": {}
    }

    if price is None:
        plan["reason"] = "Data Error: Current Price Missing"
        return plan

    mst = str(macro_trend_status or "").lower().strip()
    if mst in ("n/a", "na", "", "unknown", None):
        macro_bearish = macro_bullish = False
    else:
        macro_bearish = ("down" in mst or "bear" in mst)
        macro_bullish = ("up" in mst or "bull" in mst)

    is_squeeze = ("on" in squeeze_signal) or ("squeeze" in squeeze_signal) or ("sqz" in squeeze_signal)
    is_bullish_psar = "bull" in psar_trend
    is_bearish_psar = "bear" in psar_trend

    if atr is None or atr == 0:
        plan["signal"] = "HOLD_NO_RISK"
        plan["reason"] = "Missing ATR"
        return plan

    # LONG
    if category == "BUY" and is_bullish_psar:
        if macro_bearish:
            plan["signal"] = "RISKY_BUY"
            plan["reason"] = "Score high but macro bearish - reduce size"
        else:
            plan["signal"] = "BUY_SQUEEZE" if is_squeeze else "BUY_TREND"
            plan["reason"] = f"Bullish Score {final_score} + Trend Alignment"

        sl_calc = psar_level if (psar_level is not None and psar_level < price) else (price - (2 * atr))
        plan["stop_loss"] = round(sl_calc, 2)
        raw_risk = price - sl_calc
        risk = max(raw_risk, atr * 1.2)
        t1 = price + risk
        min_target = price + (risk * 1.5)
        tgt_calc = min_target
        for r in [r1, r2, r3]:
            if r is not None and r > min_target:
                tgt_calc = r
                break
        plan["targets"]["t1"] = round(t1, 2)
        plan["targets"]["t2"] = round(tgt_calc, 2)
        plan["move_stop_to_breakeven_after_t1"] = True
        plan["rr_ratio"] = round((tgt_calc - price) / max(risk, 1e-9), 2)
        plan["execution_hints"] = {
            "t1_desc": "1R take-profit; move stop to breakeven after hit",
            "t2_desc": "Pivot target or 1.5R fallback"
        }
        return plan

    # SHORT
    if category == "SELL" and is_bearish_psar:
        if macro_bullish:
            plan["signal"] = "RISKY_SHORT"
            plan["reason"] = "Score bearish but macro bullish - tight stops"
        else:
            plan["signal"] = "SHORT_SQUEEZE" if is_squeeze else "SHORT_TREND"
            plan["reason"] = f"Bearish Score {final_score} + Trend Breakdown"

        sl_calc = psar_level if (psar_level is not None and psar_level > price) else (price + (2 * atr))
        plan["stop_loss"] = round(sl_calc, 2)
        raw_risk = sl_calc - price
        risk = max(raw_risk, atr * 1.2)
        t1 = price - risk
        min_target = price - (risk * 1.5)
        tgt_calc = min_target
        for s in [s1, s2, s3]:
            if s is not None and s < min_target:
                tgt_calc = s
                break
        plan["targets"]["t1"] = round(t1, 2)
        plan["targets"]["t2"] = round(tgt_calc, 2)
        plan["move_stop_to_breakeven_after_t1"] = True
        plan["rr_ratio"] = round((price - tgt_calc) / max(risk, 1e-9), 2)
        plan["execution_hints"] = {
            "t1_desc": "1R take-profit; move stop to breakeven after hit",
            "t2_desc": "Pivot support target or 1.5R fallback"
        }
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
# Meta-Category (Archetype) Scoring — Required for main.py
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
