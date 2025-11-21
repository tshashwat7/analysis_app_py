# services/flowchart_helper.py
"""
Builds a complete flowchart payload by merging:
 - quick_score (technical + combined)
 - fundamentals (base)
 - extended metrics (metrics_ext)
and returns a normalized dict ready for frontend consumption.

Uses cached compute functions (fundamentals.py, indicators.py)
so repeated symbol calls avoid recomputation.

Async entrypoint:
    build_flowchart_payload(symbol, index="NIFTY50")
Safe for FastAPI async routes.
"""

import logging
import asyncio
import numpy as np
import json
import hashlib
from typing import Dict, Any, Optional
from config.constants import flowchart_mapping

from services.data_fetch import (
    get_price_history,
    safe_history,
    fetch_data,
    run_sync_or_async,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------
# Import service modules with graceful fallbacks
# -------------------------------------------------------------------
try:
    from services.fundamentals import compute_fundamentals
except Exception:
    compute_fundamentals = None
    logger.warning("⚠️ compute_fundamentals not importable")

try:
    from services.indicators import compute_indicators as _qc_core
except Exception:
    _qc_core = None
    logger.warning("⚠️ compute_indicators not found")

try:
    from services.metrics_ext import compute_extended_metrics as _compute_extended_metrics
except Exception:
    _compute_extended_metrics = None
    logger.warning("⚠️ compute_extended_metrics not found")


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def df_hash(df):
    """Create lightweight hash signature for a DataFrame."""
    try:
        return f"{len(df)}_{df.index[-1].date() if not df.empty else 'empty'}"
    except Exception:
        return "empty"


def default_converter(o):
    """Ensure all types are JSON serializable."""
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    return str(o)


# -------------------------------------------------------------------
# Flowchart mapping
# -------------------------------------------------------------------
def build_flowchart_mapping(
    payload: Dict[str, Any],
    quick_score_data: Dict[str, Any],
    fundamentals_data: Dict[str, Any],
    extended_metrics: Dict[str, Any],
    index: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize and merge metrics for frontend flowchart."""

    payload["flowchart"] = {}

    for out_key, (src, sub, metric_key) in flowchart_mapping.items():
        val, score = "N/A", 0
        try:
            if src == "quick_score":
                q = quick_score_data or {}
                candidate = q.get(sub, {}).get(metric_key) if sub else q.get(metric_key)
                if isinstance(candidate, dict):
                    val, score = candidate.get("value", "N/A"), candidate.get("score", 0)
                elif candidate is not None:
                    val = candidate

            elif src == "fundamentals":
                f = fundamentals_data or {}
                item = f.get(metric_key)
                if isinstance(item, dict):
                    val, score = item.get("value", "N/A"), item.get("score", 0)
                elif item is not None:
                    val = item

            elif src == "extended":
                e = extended_metrics or {}
                item = e.get(metric_key)
                if isinstance(item, dict):
                    val, score = item.get("value", "N/A"), item.get("score", 0)
                elif item is not None:
                    val = item
            elif src == "macro_sentiment":
                m = payload.get("macro_sentiment", {})
                item = m.get(metric_key)
                if isinstance(item, dict):
                    val, score = item.get("value", "N/A"), item.get("score", 0)
                elif item is not None:
                    val = item

            elif src == "payload" and metric_key == "index":
                val = index or "N/A"

            payload["flowchart"][out_key] = {"value": val, "score": score}

        except Exception as e:
            logger.exception(f"Mapping failed for {out_key}: {e}")
            payload["flowchart"][out_key] = {"value": "N/A", "score": 0}

    payload["raw"] = {
        "quick_score": quick_score_data or {},
        "fundamentals": fundamentals_data or {},
        "extended": extended_metrics or {},
        "macro_sentiment": payload.get("macro_sentiment", {}),
    }

    return payload


# -------------------------------------------------------------------
# Async builder
# -------------------------------------------------------------------
async def build_flowchart_payload(symbol: str, index: str = "NIFTY50") -> Dict[str, Any]:
    """Asynchronously build the complete flowchart payload."""
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "index": index,
        "quick_score": {},
        "fundamentals": {},
        "extended": {},
        "flowchart": {},
        "macro_sentiment": {},
    }

    # 0️⃣ Macro Sentiment (cached in-memory)

    # 1️⃣ Quick Score (technical)
    quick_score_data = {}
    try:
        if _qc_core:
            benchmark_df = fetch_data(period="2y", interval="1wk")
            df = safe_history(symbol, period="2y")
            if df is None or getattr(df, "empty", True):
                df = get_price_history(symbol)
            if df is None or getattr(df, "empty", True):
                logger.info("⚠️ No price history for %s", symbol)

            # Precompute df hash for indicator cache key
            hash_df = df_hash(df)
            hash_bench = df_hash(benchmark_df)

            quick_score_data = await run_sync_or_async(_qc_core, symbol, hash_df, hash_bench)
            logger.info("✅ Got quick_score for %s", symbol)
    except Exception as e:
        logger.exception("Error fetching quick_score for %s: %s", symbol, e)


    payload["quick_score"] = quick_score_data or {}

    # 2️⃣ Fundamentals (cached internally)
    fundamentals_data = {}
    try:
        if compute_fundamentals:
            fundamentals_data = await run_sync_or_async(compute_fundamentals, symbol)
            logger.info("✅ Got fundamentals for %s", symbol)
    except Exception as e:
        logger.exception("Error computing fundamentals for %s: %s", symbol, e)

    payload["fundamentals"] = fundamentals_data or {}

    # 3️⃣ Extended metrics
    extended_metrics = {}
    try:
        if _compute_extended_metrics:
            if asyncio.iscoroutinefunction(_compute_extended_metrics):
                extended_metrics = await _compute_extended_metrics(symbol)
            else:
                loop = asyncio.get_running_loop()
                extended_metrics = await loop.run_in_executor(None, lambda: _compute_extended_metrics(symbol))
            logger.info("✅ Got extended metrics for %s", symbol)
    except Exception as e:
        logger.exception("Error computing extended metrics for %s: %s", symbol, e)

    payload["extended"] = extended_metrics or {}

    # Final merge
    payload = build_flowchart_mapping(
        payload, quick_score_data, fundamentals_data, extended_metrics, index=index
    )

    return json.loads(json.dumps(payload, default=default_converter))
