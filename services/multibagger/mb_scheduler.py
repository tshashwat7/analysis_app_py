# services/multibagger/mb_scheduler.py
"""
Multibagger Scheduler — Weekly Orchestrator
=============================================
Runs the full MB pipeline (Phase 1 + Phase 2) on Sunday midnight IST.

THREAD SAFETY:
    The scheduler runs in a daemon thread started during FastAPI lifespan.
    run_mb_resolver() creates a fresh MBConfigResolver per symbol — no
    global state is patched. Safe to run concurrently with the live API.

    Phase 2 (resolver) is compute-heavy: ~200-400ms per symbol on weekly
    data. For 500 stocks, that's ~2.5 minutes. Schedule only during
    confirmed off-hours (Sunday midnight IST = Saturday 18:30 UTC).

INTEGRATION:
    In main.py lifespan startup:
        from services.multibagger.mb_scheduler import start_mb_scheduler
        start_mb_scheduler()

UNIVERSE:
    Symbol list read from UNIVERSE_CSV path defined in config/constants.py:
        NSE_UNIVERSE_CSV = "data/nifty500.csv"
    CSV must have a "symbol" column with NSE ticker strings (e.g. "RELIANCE.NS").
"""

import csv
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from config.config_utility.market_utils import get_current_ist, get_current_utc, IST

logger = logging.getLogger(__name__)

# ── Shared cycle status — imported by mb_routes.py for /status endpoint ──────
# Stored here (the authoritative writer) rather than in mb_routes.py so the
# scheduler can update it directly without a circular import.
cycle_status: dict = {
    "last_run_at":    None,
    "last_run_result": None,  # "SUCCESS" | "ERROR" | "RUNNING"
    "total_symbols":  0,
    "passed_phase1":  0,
    "errors":         0,
}

# ── Controls ──────────────────────────────────────────────────────────────────
_MB_ENABLED     = True    # Toggle without restarting via ENV or feature flag
_MAX_ERRORS     = 50      # Abort cycle if error count exceeds this
_BATCH_SIZE     = 10      # Process in batches to allow graceful shutdown
_SLEEP_BETWEEN  = 0.1     # Seconds between batches (avoid yfinance rate limits)


# =============================================================================
# SCHEDULING HELPERS
# =============================================================================

def _now_ist() -> datetime:
    return get_current_ist()


def _next_sunday_ist() -> datetime:
    """
    Return the next Sunday midnight IST.

    If called on a Sunday (weekday == 6), returns today at midnight IST
    (days_ahead = 0 → scheduled for today, not next week).
    Fix: use `< 0` not `<= 0` so Sunday runs today.
    """
    now        = _now_ist()
    days_ahead = 6 - now.weekday()   # Monday=0 … Sunday=6
    if days_ahead < 0:              # Any negative (already past) — push to next week
        days_ahead += 7

    # Replace to midnight of target day
    target = now + timedelta(days=days_ahead)
    return target.replace(hour=0, minute=0, second=0, microsecond=0)


def _seconds_until(target: datetime) -> float:
    now = _now_ist()
    return max(0.0, (target - now).total_seconds())


# =============================================================================
# UNIVERSE LOADER
# =============================================================================

def _load_universe() -> List[str]:
    """Load symbol list from NSE_UNIVERSE_CSV."""
    try:
        from config.constants import NSE_UNIVERSE_CSV
        symbols = []
        with open(NSE_UNIVERSE_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("SYMBOL") or row.get("Symbol") or row.get("ticker")
                if sym:
                    symbols.append(sym.strip())
        logger.info(f"[MB Scheduler] Loaded {len(symbols)} symbols from {NSE_UNIVERSE_CSV}")
        return symbols
    except Exception as e:
        logger.error(f"[MB Scheduler] Failed to load universe: {e}")
        return []


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

# Logic moved to services.multibagger.multibagger_screener.worker_eval_single
# for thread safety and reuse.


def _run_phase2(symbol: str, fundamentals: dict, indicators: dict, patterns: dict) -> Optional[Dict]:
    """Run Phase 2 resolver and return result dict."""
    from services.multibagger.multibagger_evaluator import run_mb_resolver
    return run_mb_resolver(symbol, fundamentals, indicators, patterns)


def _determine_conviction_tier(result: dict) -> Optional[str]:
    """Map final_decision_score + confidence to conviction tier."""
    from services.multibagger.multibagger_config import MULTIBAGGER_CONFIG
    tiers = MULTIBAGGER_CONFIG["output_schema"]["conviction_tier"]

    score = result.get("final_decision_score", 0)
    conf  = result.get("confidence", 0)

    for tier, thresholds in [("HIGH", tiers["HIGH"]), ("MEDIUM", tiers["MEDIUM"]), ("LOW", tiers["LOW"])]:
        if score >= thresholds["score_min"] and conf >= thresholds["confidence_min"]:
            return tier
    return None


def _upsert_candidate(symbol: str, result: dict, conviction_tier: Optional[str]):
    """Write or update a MultibaggerCandidate row."""
    from services.db import SessionLocal
    from services.multibagger.mb_db_model import MultibaggerCandidate

    db = SessionLocal()
    try:
        now  = get_current_utc()
        row  = db.query(MultibaggerCandidate).filter_by(symbol=symbol).first()
        prev_tier = row.conviction_tier if row else None

        if not row:
            row = MultibaggerCandidate(symbol=symbol)
            db.add(row)

        row.conviction_tier      = conviction_tier
        row.fundamental_score    = result.get("fundamental_score")
        row.technical_score      = result.get("technical_score")
        row.hybrid_score         = result.get("hybrid_score")
        row.final_score          = result.get("final_score")
        row.final_decision_score = result.get("final_decision_score")
        row.confidence           = result.get("confidence")
        row.primary_setup        = result.get("setup")
        row.primary_strategy     = result.get("strategy")
        row.gatekeeper_passed    = True
        row.gatekeeper_passed_at = now
        row.rejection_reason     = None
        row.last_evaluated       = now

        # Build thesis dict
        opp = result.get("opportunity", {})
        row.thesis_json = {
            "eval_ctx_keys":    list((result.get("eval_ctx") or {}).keys()),
            "opportunity":      opp,
            "scoring_breakdown": (result.get("eval_ctx") or {}).get("scoring", {}),
        }

        # Track tier changes
        if prev_tier != conviction_tier:
            row.prev_conviction_tier = prev_tier
            row.tier_changed_at      = now

        # Estimate re-evaluation date (4 weeks)
        from datetime import timedelta
        row.re_evaluate_date = now + timedelta(weeks=4)

        db.commit()
        logger.info(f"[MB] ✅ {symbol} upserted | tier={conviction_tier}")
    except Exception as e:
        logger.error(f"[MB] DB write error for {symbol}: {e}")
        db.rollback()
    finally:
        db.close()


def _reject_candidate(symbol: str, reason: str):
    """Write Phase 1 rejection to DB (preserves historical record)."""
    from services.db import SessionLocal
    from services.multibagger.mb_db_model import MultibaggerCandidate

    db = SessionLocal()
    try:
        now = get_current_utc()
        row = db.query(MultibaggerCandidate).filter_by(symbol=symbol).first()
        if row:
            # Update existing — don't clear good scores if previously passed
            row.gatekeeper_passed = False
            row.rejection_reason  = reason
            row.last_evaluated    = now
            row.conviction_tier   = None
        else:
            row = MultibaggerCandidate(
                symbol            = symbol,
                gatekeeper_passed = False,
                rejection_reason  = reason,
                last_evaluated    = now,
            )
            db.add(row)
        db.commit()
    except Exception as e:
        logger.error(f"[MB] DB reject error for {symbol}: {e}")
        db.rollback()
    finally:
        db.close()


# =============================================================================
# CYCLE RUNNER
# =============================================================================

def run_mb_cycle():
    """
    Execute one full MB pipeline cycle over the entire universe.
    Phase 1 (Bulk) → filter → Phase 2 (Sequential/Hybrid) → upsert.
    """
    from services.multibagger.multibagger_screener import run_bulk_screener
    from services.db import SessionLocal, StockMeta

    logger.info("[MB Scheduler] ▶ Starting weekly MB cycle")
    cycle_status["last_run_result"] = "RUNNING"
    cycle_status["last_run_at"]     = get_current_utc().isoformat()

    symbols = _load_universe()
    if not symbols:
        logger.warning("[MB Scheduler] Empty universe — aborting cycle")
        cycle_status["last_run_result"] = "ERROR:empty_universe"
        return

    total = len(symbols)
    cycle_status["total_symbols"] = total
    passed_count = 0
    error_count  = 0

    # 1. Prepare Meta Data for Screener
    db = SessionLocal()
    try:
        meta_map = {}
        all_meta = db.query(StockMeta).all()
        for m in all_meta:
            meta_map[m.symbol] = {
                "sector": m.sector,
                "industry": m.industry,
                "listing_days": None
            }
    finally:
        db.close()

    # 2. Sequential Batches for Bulk Execution
    # We batch symbols to avoid overwhelming memory if the universe is huge (e.g. 5000 stocks)
    # and to allow periodic status updates.
    INTERNAL_BATCH = 50 
    
    for i in range(0, total, INTERNAL_BATCH):
        if not _MB_ENABLED:
            logger.info("[MB Scheduler] Disabled mid-cycle — stopping")
            break

        batch_symbols = symbols[i:i + INTERNAL_BATCH]
        
        # Phase 1: Robust Bulk Scan
        batch_results = run_bulk_screener(batch_symbols, max_workers=10, meta_map=meta_map)
        
        for res in batch_results:
            symbol = res["symbol"]
            try:
                if res["status"] == "ERROR":
                    error_count += 1
                    continue

                if not res["passed"]:
                    _reject_candidate(symbol, res["reason"])
                    continue

                passed_count += 1

                # Phase 2: Detailed Scorer
                result = _run_phase2(symbol, res["fundamentals"], res["indicators"], res["patterns"])
                if not result:
                    logger.warning(f"[MB] {symbol}: Phase 2 returned None")
                    continue

                conviction_tier = _determine_conviction_tier(result)
                _upsert_candidate(symbol, result, conviction_tier)

            except Exception as e:
                error_count += 1
                logger.error(f"[MB] {symbol}: cycle error — {e}")
                if error_count >= _MAX_ERRORS:
                    logger.error("[MB Scheduler] Max errors reached — aborting cycle")
                    cycle_status["last_run_result"] = f"ERROR:max_errors({error_count})"
                    cycle_status["passed_phase1"]   = passed_count
                    cycle_status["errors"]          = error_count
                    return

        # Brief cooldown between batches
        time.sleep(_SLEEP_BETWEEN)

    cycle_status["passed_phase1"]   = passed_count
    cycle_status["errors"]          = error_count
    cycle_status["last_run_result"] = "SUCCESS"

    logger.info(
        f"[MB Scheduler] ✅ Cycle complete | "
        f"Total={total} Passed={passed_count} Errors={error_count}"
    )


# =============================================================================
# DAEMON THREAD
# =============================================================================

def _scheduler_loop():
    """Background loop — sleeps until Sunday midnight IST, then runs cycle."""
    logger.info("[MB Scheduler] Daemon thread started")
    while True:
        next_run    = _next_sunday_ist()
        wait_secs   = _seconds_until(next_run)
        next_run_str = next_run.strftime("%Y-%m-%d %H:%M IST")

        logger.info(f"[MB Scheduler] Next run: {next_run_str} (in {wait_secs/3600:.1f}h)")
        time.sleep(wait_secs)

        if _MB_ENABLED:
            try:
                run_mb_cycle()
            except Exception as e:
                logger.error(f"[MB Scheduler] Cycle crashed: {e}", exc_info=True)
        else:
            logger.info("[MB Scheduler] Disabled — skipping this run")

        # Sleep 60s after a run to avoid re-triggering on the same Sunday
        time.sleep(60)


def start_mb_scheduler():
    """
    Start the MB scheduler daemon thread.
    Call once from FastAPI lifespan startup.
    """
    t = threading.Thread(target=_scheduler_loop, name="mb-scheduler", daemon=True)
    t.start()
    logger.info("[MB Scheduler] ✅ Daemon thread started")
    return t
