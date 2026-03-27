# config/multibagger/mb_scheduler.py
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
        from config.multibagger.mb_scheduler import start_mb_scheduler
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
    "p1_errors":      0,      # ✅ P3-1: Separate P1 errors
    "p2_errors":      0,      # ✅ P3-1: Separate P2 errors
}

# ✅ P1-4: Prevent manual runs from racing with scheduled runs
_MB_CYCLE_LOCK = threading.Lock()

# ── Controls ──────────────────────────────────────────────────────────────────
_MB_ENABLED     = True    # Toggle without restarting via ENV or feature flag
_MAX_ERRORS     = 50      # Abort cycle if error count exceeds this
_BATCH_SIZE     = 10      # Process in batches to allow graceful shutdown
_SLEEP_BETWEEN  = 0.1     # ✅ P3-4: Seconds between batches for resource pacing and rate limits


# =============================================================================
# SCHEDULING HELPERS
# =============================================================================

def _now_ist() -> datetime:
    return get_current_ist()


def _next_sunday_ist() -> datetime:
    """
    Return the next Sunday midnight IST.
    """
    now = _now_ist()
    # weekday() is 6 for Sunday. 
    days_ahead = 6 - now.weekday()
    if days_ahead < 0:              # Past Sunday (e.g. Monday=0)
        days_ahead += 7
    
    target = now + timedelta(days=days_ahead)
    target = target.replace(hour=0, minute=0, second=0, microsecond=0)

    # ✅ P1-5 FIX: If it's already Sunday and we already ran today, skip to next week
    last_run_str = cycle_status.get("last_run_at")
    if last_run_str:
        try:
            # last_run_at is UTC ISO string
            last_run = datetime.fromisoformat(last_run_str).replace(tzinfo=timezone.utc)
            # Compare target (IST midnight) with last run (converted to IST)
            # If last_run was today (Sunday), we must skip
            if last_run.astimezone(IST).date() == target.date():
                target += timedelta(days=7)
        except (ValueError, TypeError):
            pass

    # Final guard: if target is in the past (e.g. 00:05 and target is 00:00 today), skip
    if (target - now).total_seconds() < 0:
        target += timedelta(days=7)

    return target


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
                sym = row.get("SYMBOL") or row.get("Symbol") or row.get("symbol") or row.get("ticker")
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

# Logic moved to config.multibagger.multibagger_screener.worker_eval_single
# for thread safety and reuse.


def _run_phase2(symbol: str, fundamentals: dict, indicators: dict, patterns: dict) -> Optional[Dict]:
    """Run Phase 2 resolver and return result dict."""
    from config.multibagger.multibagger_evaluator import run_mb_resolver
    return run_mb_resolver(symbol, fundamentals, indicators, patterns)


def _determine_conviction_tier(result: dict) -> Optional[str]:
    """Map final_decision_score + confidence to conviction tier."""
    from config.multibagger.multibagger_config import MULTIBAGGER_CONFIG
    tiers = MULTIBAGGER_CONFIG["output_schema"]["conviction_tier"]

    score = result.get("final_decision_score", 0)
    conf  = result.get("confidence", 0)

    for tier, thresholds in [("HIGH", tiers["HIGH"]), ("MEDIUM", tiers["MEDIUM"]), ("LOW", tiers["LOW"])]:
        if score >= thresholds["score_min"] and conf >= thresholds["confidence_min"]:
            return tier
            
    # ✅ P1-2 FIX: Assign WATCH for stocks that pass gates but have low conviction
    return "WATCH"


from services.db import SessionLocal, backoff_retry_db

@backoff_retry_db(retries=5)
def _upsert_candidate(symbol: str, result: dict, conviction_tier: Optional[str]):
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
        # Populate entry_trigger and hold duration from resolver result
        row.entry_trigger         = result.get("entry_trigger", "TECHNICAL_SETUP")
        row.estimated_hold_months = result.get("estimated_hold_months")

        # ✅ P3-2 FIX: Clean up thesis_json to remove internal exposure
        opp = result.get("opportunity", {})
        eval_ctx = result.get("eval_ctx") or {}
        public_keys = [k for k in eval_ctx.keys() if not k.startswith("_")]

        # ✅ P3 FIX: Store actual values in public_context, not just keys
        eval_ctx_data = result.get("eval_ctx") or {}
        public_context = {k: eval_ctx_data.get(k) for k in public_keys}
        scoring_breakdown = eval_ctx.get("scoring", {})
        
        row.thesis_json = {
            "public_context":    public_context,
            "opportunity":       opp,
            "scoring_breakdown": scoring_breakdown,
        }

        # Track tier changes
        if prev_tier != conviction_tier:
            row.prev_conviction_tier = prev_tier
            row.tier_changed_at      = now

        # Estimate re-evaluation date (4 weeks)
        # ✅ P3 FIX: Tier-based re-evaluation scaling
        tier_weeks = {
            "HIGH":  8,
            "MED":   4,
            "LOW":   2,
            "WATCH": 4
        }
        weeks = tier_weeks.get(conviction_tier, 4)
        row.re_evaluate_date = now + timedelta(weeks=weeks)
        
        db.commit()
        logger.info(f"[MB] ✅ {symbol} upserted | tier={conviction_tier}")
    except Exception as e:
        logger.error(f"[MB] DB write error for {symbol}: {e}")
        db.rollback()
    finally:
        db.close()


@backoff_retry_db(retries=5)
def _reject_candidate(symbol: str, reason: str):
    """Write Phase 1 rejection to DB (preserves historical record)."""
    from services.db import SessionLocal
    from config.multibagger.mb_db_model import MultibaggerCandidate

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
    if not _MB_CYCLE_LOCK.acquire(blocking=False):
        logger.warning("[MB Scheduler] Cycle already in progress — skipping")
        return

    try:
        _run_mb_cycle_internal()
    finally:
        _MB_CYCLE_LOCK.release()

def _run_mb_cycle_internal():
    """Internal implementation of the cycle."""
    from config.multibagger.multibagger_screener import run_bulk_screener
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
    p1_error_count = 0
    p2_error_count = 0

    # 1. Prepare Meta Data for Screener
    db = SessionLocal()
    try:
        meta_map = {}
        all_meta = db.query(StockMeta).all()
        for m in all_meta:
            meta_map[m.symbol] = {
                "sector": m.sector,
                "industry": m.industry,
                "listing_days": None # Default — updated per batch from yf firstTradeDate
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
        
        # 1.1 Hydrate listing_days from fundamentals (yfinance gap fix)
        from services.fundamentals import compute_fundamentals
        for sym in batch_symbols:
            if sym in meta_map:
                f_data = compute_fundamentals(sym)
                # firstTradeDateEpochUtc is absolute seconds
                first_trade = f_data.get("firstTradeDateEpochUtc")
                if first_trade:
                    try:
                        ft_dt = datetime.fromtimestamp(float(first_trade), tz=timezone.utc)
                        days = (get_current_utc() - ft_dt).days
                        meta_map[sym]["listing_days"] = days
                    except (ValueError, TypeError):
                        pass

        # Phase 1: Robust Bulk Scan
        batch_results = run_bulk_screener(batch_symbols, max_workers=10, meta_map=meta_map)
        
        for res in batch_results:
            symbol = res["symbol"]
            try:
                if res["status"] in ("ERROR", "PERMANENT_ERROR", "TRANSIENT_ERROR"):
                    p1_error_count += 1
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

                # ✅ P1-3 FIX: Hydrate MB score into main SignalCache dashboard
                from config.multibagger.mb_main_patches import hydrate_mb_score_to_signal_cache
                hydrate_mb_score_to_signal_cache(
                    symbol   = symbol,
                    score    = result.get("final_decision_score", 0),
                    conf     = result.get("confidence", 0),
                    tier     = conviction_tier or "WATCH",
                    setup    = result.get("setup", "GENERIC"),
                    strategy = result.get("strategy")
                )

            except Exception as e:
                p2_error_count += 1
                logger.error(f"[MB] {symbol}: cycle error — {e}")
                if (p1_error_count + p2_error_count) >= _MAX_ERRORS:
                    logger.error("[MB Scheduler] Max errors reached — aborting cycle")
                    cycle_status["last_run_result"] = f"ERROR:max_errors({p1_error_count + p2_error_count})"
                    cycle_status["passed_phase1"]   = passed_count
                    cycle_status["p1_errors"]        = p1_error_count
                    cycle_status["p2_errors"]        = p2_error_count
                    return

        # Brief cooldown between batches
        time.sleep(_SLEEP_BETWEEN)

    cycle_status["passed_phase1"]   = passed_count
    cycle_status["p1_errors"]        = p1_error_count
    cycle_status["p2_errors"]        = p2_error_count
    cycle_status["last_run_result"] = "SUCCESS"

    logger.info(
        f"[MB Scheduler] ✅ Cycle complete | "
        f"Total={total} Passed={passed_count} P1_Errors={p1_error_count} P2_Errors={p2_error_count}"
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


def check_and_recover_missed_run():
    """
    ✅ P1-5 FIX: If startup happens mid-week and the last run was more than 7 days ago,
    trigger a recovery run (if not already Sunday).
    """
    try:
        last_run_iso = cycle_status.get("last_run_at")
        if not last_run_iso:
            logger.info("[MB Recovery] No last run recorded. Assuming fresh start.")
            # Trigger first-time run immediately if enabled
            if _MB_ENABLED:
                threading.Thread(target=run_mb_cycle, name="mb-recovery-startup", daemon=True).start()
            return

        last_run = datetime.fromisoformat(last_run_iso)
        # Ensure last_run is aware
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
            
        now = get_current_utc()
        age_days = (now - last_run).days
        
        # If last run was more than 7 days ago, we missed a Sunday cycle
        if age_days >= 7:
            logger.warning(f"[MB Recovery] Missed weekly cycle detected! Last run was {age_days} days ago. Triggering now.")
            if _MB_ENABLED:
                threading.Thread(target=run_mb_cycle, name="mb-recovery-missed", daemon=True).start()
        else:
            logger.info(f"[MB Recovery] Last run was {age_days} days ago — no recovery needed.")
            
    except Exception as e:
        logger.error(f"[MB Recovery] CRITICAL: Startup recovery failed — {e}")


def start_mb_scheduler():
    """
    Start the MB scheduler daemon thread.
    Call once from FastAPI lifespan startup.
    """
    # ✅ P1-5 FIX: Check for missed run on startup
    check_and_recover_missed_run()
    
    t = threading.Thread(target=_scheduler_loop, name="mb-scheduler", daemon=True)
    t.start()
    logger.info("[MB Scheduler] ✅ Daemon thread started")
    return t
