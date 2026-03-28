# config/multibagger/multibagger_screener.py
"""
Multibagger Screener — Phase 1
================================
Fast fundamental + technical hard-rejection filter.

DESIGN:
    - Stateless, pure function. No DB writes.
    - Called by mb_scheduler.py once per symbol in the universe.
    - Symbols that pass are forwarded to multibagger_evaluator.run_mb_resolver().
    - Missing data is handled gracefully: if yfinance doesn't provide a
      fundamental value (common for NSE stocks), that filter is SKIPPED
      with a debug log rather than rejecting the stock.

TECHNICAL FILTERS use weekly indicator keys:
    maFast = MMA(6w), maMid = MMA(12w), maSlow = MMA(24w)
    These are the standardised output keys from indicators.py for
    horizon="multibagger".

PRICE / 52W HIGH DATA SOURCES:
    close   → indicators["price"]["value"]     (set by compute_indicators in indicators.py)
    high52w → fundamentals["high52w"]["value"] (set by fundamentals.py, passed via _extract_price_data)
    The key is "high52w" (no underscore) — confirmed in config_helpers._extract_price_data.
"""

import logging
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any

from config.multibagger.multibagger_config import MULTIBAGGER_CONFIG

logger = logging.getLogger(__name__)

_GATEKEEPER  = MULTIBAGGER_CONFIG["gatekeeper"]
_FUND_GATES  = _GATEKEEPER["fundamentals"]
_TECH_GATES  = _GATEKEEPER["technicals"]
_UNIV_GATES  = _GATEKEEPER["universe"]



# =============================================================================
# UNIVERSE FILTER
# =============================================================================

def _check_universe_filters(symbol: str, meta: Dict, fundamentals: Dict = None) -> str:
    """
    Check sector exclusions, listing age, minimum price, and minimum market cap.

    Args:
        symbol:       NSE symbol
        meta:         Row from StockMeta table (may be empty dict if not yet populated)
        fundamentals: Optional fundamentals dict for price/marketCap checks

    Returns:
        "" if passes, rejection reason string if fails.
    """
    fundamentals = fundamentals or {}

    sector = meta.get("sector") or ""
    if sector in _UNIV_GATES.get("exclude_sectors", []):
        return f"EXCLUDED_SECTOR:{sector}"

    # listing_days requires DB metadata — skip gracefully if unavailable
    listing_days = meta.get("listing_days")
    if listing_days is not None:
        min_days = _UNIV_GATES.get("min_listing_days", 365)
        if listing_days < min_days:
            return f"TOO_NEW:{listing_days}d<{min_days}d"

    # min_price — sourced from fundamentals.currentPrice or meta
    min_price = _UNIV_GATES.get("min_price")
    if min_price is not None:
        price_data = fundamentals.get("currentPrice") or meta.get("price")
        if isinstance(price_data, dict):
            price_val = price_data.get("raw") or price_data.get("value")
        else:
            price_val = price_data
        if price_val is not None:
            try:
                if float(price_val) < min_price:
                    return f"PRICE_TOO_LOW:{price_val}<{min_price}"
            except (ValueError, TypeError):
                pass

    # min_market_cap (in Crores) — sourced from fundamentals.marketCap
    min_mcap = _UNIV_GATES.get("min_market_cap")
    if min_mcap is not None:
        mcap_data = fundamentals.get("marketCap")
        if isinstance(mcap_data, dict):
            mcap_val = mcap_data.get("raw") or mcap_data.get("value")
        else:
            mcap_val = mcap_data
        if mcap_val is not None:
            try:
                # Normalize absolute rupees to Crores (1e7)
                mcap_crores = float(mcap_val) / 10000000.0
                if mcap_crores < min_mcap:
                    return f"MCAP_TOO_LOW:{mcap_crores:.0f}Cr<{min_mcap}Cr"
            except (ValueError, TypeError):
                pass

    return ""


# =============================================================================
# FUNDAMENTAL FILTER
# =============================================================================

def _check_fundamental_filters(symbol: str, fundamentals: Dict) -> str:
    """
    Evaluate fundamental hard gates.

    The fundamentals dict has the format:
        {"roe": {"raw": 22.1, "value": 22.1, "score": 9, ...}, ...}

    Gate values are extracted via .get("raw") with .get("value") fallback.
    If raw is None (yfinance data gap), the filter is SKIPPED — lenient
    fallback for Indian market data coverage gaps.

    Returns:
        "" if all checked gates pass, rejection reason string otherwise.
    """
    for metric_key, constraint in _FUND_GATES.items():
        metric_data = fundamentals.get(metric_key, {})

        # Extract raw value — same priority as gate_evaluator._unwrap
        if isinstance(metric_data, dict):
            raw_val = metric_data.get("raw")
            if raw_val is None:
                raw_val = metric_data.get("value")
        else:
            raw_val = metric_data

        if raw_val is None:
            logger.debug(
                f"[MB Phase1] {symbol}: {metric_key} missing — skipping filter"
            )
            continue

        try:
            raw_val = float(raw_val)
        except (ValueError, TypeError):
            logger.debug(f"[MB Phase1] {symbol}: {metric_key} non-numeric — skipping")
            continue

        min_val = constraint.get("min")
        max_val = constraint.get("max")

        if min_val is not None and raw_val < min_val:
            return f"{metric_key}_BELOW_MIN:{raw_val:.2f}<{min_val}"
        if max_val is not None and raw_val > max_val:
            return f"{metric_key}_ABOVE_MAX:{raw_val:.2f}>{max_val}"

    return ""


# =============================================================================
# TECHNICAL FILTER
# =============================================================================

def _check_technical_filters(symbol: str, indicators: Dict, fundamentals: Dict = None) -> str:
    """
    Evaluate weekly Stage 2 alignment and 52W proximity.

    Uses standardised MA keys produced by indicators.py for "multibagger":
        maFast → MMA(6w), maMid → MMA(12w), maSlow → MMA(24w)

    Price sources (confirmed from indicators.py and config_helpers._extract_price_data):
        close   → indicators["price"]["value"]
        high52w → fundamentals["high52w"] with raw/value extraction
                  Key is "high52w" (no underscore) — NOT inside the price dict.

    Returns:
        "" if passes, rejection reason string otherwise.
    """
    fundamentals = fundamentals or {}

    # ── Current close price ──────────────────────────────────────────────
    # Verify "currentPrice" is the actual key in your indicators output
    price_entry = indicators.get("currentPrice") or indicators.get("close") or indicators.get("price") or {}
    close = price_entry.get("value") if isinstance(price_entry, dict) else price_entry

    # ── Stage 2 trend alignment: Close > MMA6 > MMA12 > MMA24 ───────────
    if _TECH_GATES.get("trend_alignment", {}).get("required", True):
        def _ma_val(key):
            d = indicators.get(key, {})
            return d.get("value") if isinstance(d, dict) else d

        ma_fast = _ma_val("maFast")
        ma_mid  = _ma_val("maMid")
        ma_slow = _ma_val("maSlow")

        if None in (close, ma_fast, ma_mid, ma_slow):
            missing = [k for k, v in
                       {"close": close, "maFast": ma_fast, "maMid": ma_mid, "maSlow": ma_slow}.items()
                       if v is None]
            return f"MA_DATA_MISSING:{missing}"

        stage2 = (close > ma_fast) and (ma_fast > ma_mid) and (ma_mid > ma_slow)
        if not stage2:
            return (
                f"NOT_STAGE2:"
                f"close={close:.2f},mmaFast={ma_fast:.2f},"
                f"mmaMid={ma_mid:.2f},mmaSlow={ma_slow:.2f}"
            )

    # ── Distance from 52W high ───────────────────────────────────────────
    # high52w lives in fundamentals (keyed as "high52w", no underscore),
    # confirmed in config_helpers._extract_price_data line 135.
    max_dd = _TECH_GATES.get("distance_from_high", {}).get("max_drawdown_pct")
    if max_dd is not None and close:
        h52_data = fundamentals.get("high52w")
        if isinstance(h52_data, dict):
            high52w = h52_data.get("raw") or h52_data.get("value")
        else:
            high52w = h52_data

        if high52w is not None:
            try:
                high52w = float(high52w)
                if high52w > 0:
                    drawdown_pct = (high52w - close) / high52w * 100
                    if drawdown_pct > max_dd:
                        return f"TOO_FAR_FROM_HIGH:{drawdown_pct:.1f}%>{max_dd}%"
            except (ValueError, TypeError):
                pass  # Skip if non-numeric

    return ""


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def evaluate_single_screener(
    symbol:       str,
    fundamentals: Dict,
    indicators:   Dict,
    meta:         Dict = None,
) -> Tuple[bool, str]:
    """
    Run Phase 1 gatekeeper on a single stock.
    """
    # ✅ 10.1.1 P2 FIX: Verify MA key contract at runtime
    # If indicators.py changes key names, the screener must fail explicitly, not silently.
    required_keys = ["maFast", "maMid", "maSlow"]
    if not any(k in indicators for k in required_keys):
        return False, f"MA_DATA_MISSING:Indicators keys {list(indicators.keys())} mismatch {required_keys}"

    meta = meta or {}



    reason = _check_universe_filters(symbol, meta, fundamentals)
    if reason:
        logger.info(f"[MB Phase1] ❌ {symbol} rejected: {reason}")
        return False, reason

    reason = _check_fundamental_filters(symbol, fundamentals)
    if reason:
        logger.info(f"[MB Phase1] ❌ {symbol} rejected: {reason}")
        return False, reason

    reason = _check_technical_filters(symbol, indicators, fundamentals)
    if reason:
        logger.info(f"[MB Phase1] ❌ {symbol} rejected: {reason}")
        return False, reason

    logger.info(f"[MB Phase1] ✅ {symbol} passed gatekeeper")
    return True, ""


# =============================================================================
# BULK SCANNER (MULTITHREADED)
# =============================================================================

def worker_eval_single(symbol: str, meta: Dict = None) -> Dict[str, Any]:
    """
    Worker task: Fetch data and run Phase 1 screener for a single symbol.
    Provides robust error isolation for bulk runs.
    """
    from services.indicator_cache import compute_indicators_cached
    from services.fundamentals import compute_fundamentals
    
    try:
        # 1. Fetch Data (Individually robust via _wrap_calc inside these services)
        # ✅ P0-2 FIX: Remove force_refresh=True to use Parquet cache
        indicators, patterns = compute_indicators_cached(
            symbol, horizon="multibagger", force_refresh=False
        )
        fundamentals = compute_fundamentals(symbol)
        
        # 2. Run Screener
        passed, reason = evaluate_single_screener(
            symbol, fundamentals, indicators, meta=meta
        )
        
        return {
            "symbol":       symbol,
            "passed":       passed,
            "reason":       reason,
            "fundamentals": fundamentals,
            "indicators":   indicators,
            "patterns":     patterns,
            "status":       "SUCCESS"
        }
    except Exception as e:
        err_msg = str(e)
        logger.error(f"[MB Bulk Worker] {symbol} failed: {err_msg}")
        
        # ✅ P1-1: Distinguish error types
        is_transient = any(x in err_msg.lower() for x in ["timeout", "rate limit", "connection", "remote end closed"])
        status = "TRANSIENT_ERROR" if is_transient else "PERMANENT_ERROR"
        
        return {
            "symbol": symbol,
            "passed": False,
            "reason": f"WORKER_ERROR:{status}:{err_msg}",
            "status": status
        }


def run_bulk_screener(
    symbols:      List[str],
    meta_map:     Dict[str, Dict] = None,
    max_workers:  int = 10,
) -> List[Dict[str, Any]]:
    """
    Leverage multithreading to scan a bulk list of stocks for yfinance data.
    
    Args:
        symbols:     List of NSE ticker strings
        max_workers: Concurrency level (default 10 to balance speed/rate limits)
        meta_map:    Optional mapping of symbol -> StockMeta dict
        
    Returns:
        List of result dicts from worker_eval_single.
    """
    meta_map = meta_map or {}
    results = []
    
    logger.info(f"[MB Screener] Starting bulk scan for {len(symbols)} symbols with {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(worker_eval_single, sym, meta_map.get(sym, {})): sym 
            for sym in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logger.error(f"[MB Screener] Fatal error processing {sym}: {e}")
                results.append({
                    "symbol": sym, 
                    "passed": False, 
                    "reason": f"CRITICAL_FUTURE_ERROR:{e}",
                    "status": "FATAL"
                })

    logger.info(f"[MB Screener] Bulk scan complete. {len(results)} symbols processed.")
    return results

