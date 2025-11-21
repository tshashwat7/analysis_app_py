# services/recommendation.py
"""

Preserves your hybrid approach:
 - BB High / ATR / R:R target priority
 - Hybrid trailing SL using ATR, adaptive fractal swing-low, initial SL
 - Adaptive lookback computed from volatility (ATR or returns)
 - Tactical / strategic decision rules (bull_score, bull_signal, long_term_rec)
 - Returns same keys: Trailing_Stop_Loss, RR_Ratio, recommendation, reason, etc.

Improvements:
 - safer numeric conversion with logging
 - avoids silent excepts
 - defensive checks for missing price_history keys
 - consistent rounding/None handling
"""

from typing import Dict, Any, Optional, List
import logging

from services.data_fetch import _clamp, _num

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic config only if not configured by application
    logging.basicConfig(level=logging.INFO)


def _find_adaptive_swing_low(lows: List[float], lookback: int, prominence: int = 2) -> Optional[float]:
    """
    Find the most recent pivot low using a fractal definition:
    low[i] < low[i-1..i-prominence] and low[i] < low[i+1..i+prominence]
    Search from the most recent bar backwards within lookback window.
    If no pivot found, return the minimum low in the lookback window.
    """
    try:
        if not lows or len(lows) < (prominence * 2 + 1):
            return None
        n = len(lows)
        start = max(0, n - lookback)
        for i in range(n - 1, start + prominence - 1, -1):
            left = lows[i - prominence:i]
            right = lows[i + 1:i + 1 + prominence]
            if left and right and all(lows[i] < x for x in left) and all(lows[i] < x for x in right):
                return float(lows[i])
        # fallback: min in window
        window = lows[start:]
        if window:
            return float(min(window))
    except Exception as e:
        logger.warning("Error finding adaptive swing low: %s", e, exc_info=True)
    return None


def build_trade_recommendation(
    indicators: Dict[str, Dict[str, Any]],
    bull_score: int,
    bull_signal: str,
    long_term_rec: str,
    technical_score: int,
    fundamental_score: int,
    price_history: Optional[Dict[str, Any]] = None,
    reward_multiplier: float = 2.0
) -> Dict[str, Any]:
    """
    Extended recommendation with adaptive hybrid trailing SL (ATR-driven swing-low detection).
    price_history expected as {'high': [...], 'low': [...], 'close': [...]} (oldest -> newest).
    """

    out = {
        "recommendation": "NO_RECOMMENDATION",
        "reason": "Insufficient data or no clear trade signal",
        "entry_price_used": None,
        "suggested_sl": None,
        "target_price": None,
        "expected_profit_pct": None,
        "method": None,
        "Trailing_Stop_Loss": None,
        "RR_Ratio": None,
        "tactical_source": {"bull_score": bull_score, "bull_signal": bull_signal},
        "strategic_source": {"long_term_rec": long_term_rec, "technical_score": technical_score, "fundamental_score": fundamental_score}
    }

    # Safely extract numeric indicators
    try:
        price = _num(indicators.get("Price", {}).get("value"))
        entry = _num(indicators.get("Entry Price (Confirm)", {}).get("value"))
        initial_sl = _num(indicators.get("Suggested SL (2xATR)", {}).get("value"))
        atr = _num(indicators.get("ATR (14)", {}).get("value"))
        bb_high = _num(indicators.get("BB High", {}).get("value"))
        bb_low = _num(indicators.get("BB Low", {}).get("value"))
    except Exception as e:
        logger.exception("Error extracting indicators: %s", e)
        return out

    if price is None:
        # Can't produce a recommendation without current price
        return out

    # Determine target priority: BB High > ATR multiplier > Risk-Reward fallback
    target, method = None, None
    try:
        if bb_high is not None and bb_high > price:
            target = bb_high
            method = "BB_HIGH"
        elif atr is not None:
            target = price + reward_multiplier * atr
            method = "ATR_MULTIPLIER"
        elif entry is not None and initial_sl is not None:
            rr = entry - initial_sl
            if rr > 0:
                target = entry + reward_multiplier * rr
                method = "RISK_REWARD"
    except Exception as e:
        logger.warning("Error computing target: %s", e, exc_info=True)

    expected_pct = None
    try:
        if target is not None and price > 0:
            expected_pct = round(((target - price) / price) * 100, 2)
    except Exception:
        expected_pct = None

    # Trailing SL: components
    atr_sl = None
    if atr is not None:
        try:
            atr_sl = price - 2.0 * atr
        except Exception:
            atr_sl = None

    # Adaptive swing SL using price history & volatility
    swing_sl = None
    try:
        if price_history and isinstance(price_history, dict):
            lows = price_history.get("low", []) or []
            closes = price_history.get("close", []) or []

            # Compute volatility factor
            vol_factor = None
            if atr is not None and price and price > 0:
                vol_factor = atr / price
            else:
                # fallback estimate from recent returns
                try:
                    if closes and len(closes) >= 5:
                        returns = []
                        for i in range(1, len(closes)):
                            prev = closes[i - 1]
                            curr = closes[i]
                            if prev and prev != 0:
                                returns.append(abs((curr - prev) / prev))
                        # use last up to 20 returns
                        recent = returns[-20:] if returns else []
                        vol_factor = (sum(recent) / max(1, len(recent))) if recent else 0.02
                    else:
                        vol_factor = 0.02
                except Exception:
                    vol_factor = 0.02

            # Map vol_factor to lookback range [10, 60]
            try:
                lookback = int(_clamp(round(vol_factor * 400), 10, 60))
            except Exception:
                lookback = 20

            # attempt to find fractal pivot low within lookback
            try:
                swing_candidate = _find_adaptive_swing_low(lows, lookback=lookback, prominence=2)
                if swing_candidate is not None:
                    swing_sl = swing_candidate
            except Exception:
                swing_sl = None
    except Exception as e:
        logger.warning("Error while computing swing SL: %s", e, exc_info=True)

    # Fallbacks for swing_sl
    try:
        if swing_sl is None:
            if bb_low is not None:
                swing_sl = bb_low
            elif price_history and isinstance(price_history, dict):
                lows = price_history.get("low", []) or []
                if lows:
                    swing_sl = float(min(lows[-20:])) if len(lows) >= 1 else None
    except Exception:
        swing_sl = None

    # If still None, use initial_sl or a conservative ATR-based SL
    try:
        if swing_sl is None and initial_sl is not None:
            swing_sl = initial_sl
    except Exception:
        pass

    # initial SL fallback if not provided
    init_sl = None
    try:
        if initial_sl is not None:
            init_sl = initial_sl
        else:
            if atr is not None:
                init_sl = price - 3 * atr
    except Exception:
        init_sl = None

    # Compose hybrid trailing SL — ensure we pick the most protective (highest) SL number
    try:
        sl_candidates = [x for x in [atr_sl, swing_sl, init_sl] if x is not None]
        trailing_sl = float(max(sl_candidates)) if sl_candidates else None
    except Exception:
        trailing_sl = None

    # Compute RR (using entry if provided else current price)
    try:
        entry_for_calc = entry if entry is not None else price
        rr_ratio = None
        if trailing_sl is not None and entry_for_calc is not None and target is not None:
            risk = entry_for_calc - trailing_sl
            reward = target - entry_for_calc
            if risk > 0:
                rr_ratio = round(reward / risk, 2)
            else:
                rr_ratio = None
    except Exception:
        rr_ratio = None

    # Decision rules
    try:
        tactical_buy = (isinstance(bull_score, (int, float)) and bull_score >= 70) \
                       or (isinstance(bull_signal, str) and "strong" in bull_signal.lower())
        tactical_hold = (isinstance(bull_score, (int, float)) and 50 <= bull_score < 70)
        strategic_sell = str(long_term_rec).strip().upper().startswith("SELL")
        strategic_hold = "HOLD" in str(long_term_rec).upper()
    except Exception as e:
        logger.warning("Error evaluating decision flags: %s", e, exc_info=True)
        tactical_buy = tactical_hold = strategic_sell = strategic_hold = False

    # Build outputs depending on rules (preserve original messaging & fields)
    try:
        # Tactical BUY branch (strong momentum)
        if tactical_buy:
            rec = "BUY"
            if entry is not None and price <= entry:
                reason = "Strong momentum; price near ideal entry"
            else:
                reason = "Strong momentum; price above ideal entry but still BUY on momentum"

            advice_msg = None
            if rr_ratio is None or (isinstance(rr_ratio, (int, float)) and rr_ratio <= 0):
                risk_buffer = (atr * 0.5) if atr else 0
                ideal_low = (trailing_sl + risk_buffer) if trailing_sl else (price * 0.98)
                ideal_high = entry_for_calc
                try:
                    advice_msg = f"Wait for price pullback near ₹{ideal_low:.2f} - ₹{ideal_high:.2f} for favorable R:R"
                except Exception:
                    advice_msg = None

            out.update({
                "recommendation": rec,
                "reason": reason,
                "entry_price_used": round(entry_for_calc, 4) if entry_for_calc is not None else None,
                "suggested_sl": round(initial_sl, 4) if initial_sl is not None else None,
                "target_price": round(target, 4) if target is not None else None,
                "expected_profit_pct": expected_pct,
                "method": method,
                "Trailing_Stop_Loss": round(trailing_sl, 4) if trailing_sl is not None else None,
                "RR_Ratio": rr_ratio,
                "advice": advice_msg
            })
            return out

        # Strategic SELL branch
        if strategic_sell:
            out.update({
                "recommendation": "SELL",
                "reason": "Long-term trend suggests exit",
                "Trailing_Stop_Loss": round(trailing_sl, 4) if trailing_sl is not None else None,
                "RR_Ratio": rr_ratio
            })
            return out

        # Tactical hold or strategic hold (conservative buy if price is near entry)
        if tactical_hold or strategic_hold:
            if entry is not None and price <= entry:
                out.update({
                    "recommendation": "BUY (Conservative)",
                    "reason": "Neutral zone but acceptable entry price",
                    "entry_price_used": round(entry_for_calc, 4) if entry_for_calc is not None else None,
                    "suggested_sl": round(initial_sl, 4) if initial_sl is not None else None,
                    "target_price": round(target, 4) if target is not None else None,
                    "expected_profit_pct": expected_pct,
                    "method": method,
                    "Trailing_Stop_Loss": round(trailing_sl, 4) if trailing_sl is not None else None,
                    "RR_Ratio": rr_ratio
                })
                return out
            else:
                out.update({
                    "recommendation": "HOLD",
                    "reason": "Wait for better price or momentum",
                    "Trailing_Stop_Loss": round(trailing_sl, 4) if trailing_sl is not None else None,
                    "RR_Ratio": rr_ratio
                })
                return out

        # Default — no decisive signal
        out.update({
            "Trailing_Stop_Loss": round(trailing_sl, 4) if trailing_sl is not None else None,
            "RR_Ratio": rr_ratio
        })
        return out

    except Exception as e:
        logger.exception("Unhandled error assembling recommendation: %s", e)
        # Final safe return
        out.update({
            "Trailing_Stop_Loss": round(trailing_sl, 4) if locals().get("trailing_sl") is not None else None,
            "RR_Ratio": locals().get("rr_ratio", None)
        })
        return out
