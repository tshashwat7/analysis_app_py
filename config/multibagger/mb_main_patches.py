# config/multibagger/mb_main_patches.py
"""
MB Main Patches
================
Provides utilities to bridge the weekly Multibagger pipeline with the
main Orchestrator's SignalCache table.

P1-3 FIX: Hydrate MB scores into SignalCache so the main dashboard can
display multibagger conviction alongside other horizons.
"""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from services.db import _write_signal_cache_with_retry
from config.config_utility.market_utils import get_current_utc

logger = logging.getLogger(__name__)

def hydrate_mb_score_to_signal_cache(
    symbol: str,
    score: float,
    conf: float,
    tier: str,
    setup: str,
    strategy: Optional[str] = None
):
    """
    Update the 'multi_score' field in SignalCache.horizon_scores JSON.
    Uses the Orchestrator's retry-safe writer to avoid lock contention.
    """
    try:
        def _writer(db, entry):
            # entry is a SignalCache instance
            horizon_scores = entry.horizon_scores or {}
            
            # Update the specific multibagger score
            # We preserve other horizon scores (intra, short, long)
            horizon_scores["multi_score"] = round(score, 2)
            
            # Update the JSON column
            entry.horizon_scores = horizon_scores
            
            # If the currently selected horizon is multibagger, update score/conf/rec
            # ✅ Fix 7.1-2: Remove illegal signal_text overwrite (signal_text must be BUY/SELL/etc)
            if entry.selected_horizon == "multibagger":
                entry.score = round(score, 2)
                entry.conf_score = int(conf)
                entry.recommendation = f"MB_{tier}"
                entry.updated_at = get_current_utc()

        _write_signal_cache_with_retry(symbol, _writer)
        logger.debug(f"[MB Patch] ✅ Hydrated {symbol} score into SignalCache")
    except Exception as e:
        logger.error(f"[MB Patch] ❌ Failed to hydrate {symbol}: {e}")
