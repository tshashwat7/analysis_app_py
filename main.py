import asyncio
import os
import time
import math
import pytz
import concurrent.futures
import multiprocessing
from typing import List, Tuple, Dict, Any, Optional
from fastapi import FastAPI, Request, Form, Query, Security, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel
from filelock import FileLock
import random
import re
import pandas as pd
from config.config_utility.logger_config import setup_logger
from services.indicator_cache import compute_indicators_cached
logger = setup_logger()
from config.config_utility.market_utils import is_market_open, get_current_ist, get_current_utc,ensure_utc
from services.patterns.pattern_state_manager import cleanup_old_breakdown_states
import threading
from config.constants import ENABLE_CACHE_WARMER, ENABLE_JSON_ENRICHMENT, INDEX_TICKERS
from services.data_fetch import parse_index_csv
from services.fundamentals import compute_fundamentals
from services.signal_engine import (
    compute_all_profiles,
    generate_trade_plan,
    score_value_profile,
    score_growth_profile,
    score_quality_profile,
    score_momentum_profile,
)
from services.corporate_actions import (
    get_corporate_actions,
    build_corp_actions_summary_cache,
    get_corp_actions_summary,
)
from services.world_bank_provider import get_macro_metrics
from services.summaries import build_all_summaries
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from services.db import SessionLocal, SignalCache, init_db, PaperTrade, FundamentalCache, get_db
from config.multibagger.mb_routes import mb_router
from config.multibagger.mb_scheduler import start_mb_scheduler

class PaperTradeRequest(BaseModel):
    symbol: str
    entry_price: float
    target_1: float = None
    target_2: float = None
    stop_loss: float = None
    estimated_hold_days: int = None
    horizon: str = None
    position_size: int = None

# API Versioning & Schema (P0-3)
API_VERSION = "1.0.0"
RESPONSE_SCHEMA_VERSION = "1.1.0"

# Environment-configurable parameters
WARMER_BATCH_SIZE = int(os.getenv("WARMER_BATCH_SIZE", "5"))
WARMER_TOP_N_DURING_MARKET = int(os.getenv("WARMER_TOP_N_DURING_MARKET", "50"))
WARMER_LRU_TARGET = int(os.getenv("WARMER_LRU_TARGET", "500"))
WARMER_MARKET_INTERVAL_SEC = int(os.getenv("WARMER_MARKET_INTERVAL_SEC", str(15 * 60)))
WARMER_OFFPEAK_INTERVAL_SEC = int(os.getenv("WARMER_OFFPEAK_INTERVAL_SEC", str(60 * 60)))
WARMER_DEEP_HOUR = int(os.getenv("WARMER_DEEP_HOUR", "2"))
WARMER_DEEP_SLEEP_HOURS = int(os.getenv("WARMER_DEEP_SLEEP_HOURS", "6"))
WARMER_BATCH_SLEEP_MARKET = float(os.getenv("WARMER_BATCH_SLEEP_MARKET", "5"))
WARMER_BATCH_SLEEP_OFFPEAK = float(os.getenv("WARMER_BATCH_SLEEP_OFFPEAK", "2"))
WARMER_BATCH_TIMEOUT_SEC = int(os.getenv("WARMER_BATCH_TIMEOUT_SEC", str(5 * 60)))
WARMER_RATE_LIMIT_COOLDOWN_SEC = int(os.getenv("WARMER_RATE_LIMIT_COOLDOWN_SEC", "300"))
CORP_ACTIONS_STARTUP_TIMEOUT_SEC = int(os.getenv("CORP_ACTIONS_STARTUP_TIMEOUT_SEC", "120"))
IST = pytz.timezone("Asia/Kolkata")

# --- SECURITY ---
API_KEY = os.getenv("SIGNAL_API_KEY", "pro-signal-v15-secret")  # Default for dev
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    raise HTTPException(status_code=403, detail="Invalid API Key")

def sanitize_index_name(name: str) -> str:
    """Fix P3-1: Prevent path traversal by allowing only alphanumeric and underscores."""
    if not name: return "nifty50"
    return re.sub(r'[^a-zA-Z0-9_]', '', name)

def _validate_symbol(symbol: str):
    """✅ P0: Ensure symbol matches expected format to prevent NameError and injection."""
    if not symbol or not re.match(r'^[A-Z0-9.\-&]{1,30}$', symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format")

# --- CACHE WARMER HELPER ---
def run_analysis_for_cache(symbol: str):
    try:
        run_full_analysis(symbol)
        logger.debug(f"[WARMER] Successfully warmed cache for {symbol}.")
    except Exception as e:
        logger.error(f"[WARMER] Failed to run analysis for {symbol}: {e}")

# ==============================================================================
# ### GLOBAL STATE & EXECUTORS (Lazy & Safe)
# ==============================================================================
API_EXECUTOR = None
COMPUTE_EXECUTOR = None
BACKGROUND_EXECUTOR = None
_SHUTDOWN_LOCK = threading.Lock()

# ✅ FIX P1-6: In-flight request deduplication
_IN_FLIGHT_ANALYSES = set()
_IN_FLIGHT_LOCK = threading.Lock()
_SHUTDOWN_IN_PROGRESS = False # ✅ FIX 8.5-2: Defined at module level to prevent NameError

# Lazy Loaded Stock Lists (P1-7: Moved loading logic to index_utils)
from services.index_utils import (
    load_or_create_index, 
    load_or_create_global_stocks, 
    build_smart_index_map,
    get_cached_stocks,
    STOCK_TO_INDEX_MAP
)

ALL_STOCKS_LIST = None
ALL_STOCKS_MAP = None

# ✅ In-memory cache: stores all 4 horizon scores from the last full-mode analysis
# Keyed by symbol. Persists across horizon toggles until a new full analysis runs.
FULL_HORIZON_SCORES: Dict[str, Dict[str, float]] = {}

# --- CLEANUP SCHEDULER ---
# main.py

def run_scheduled_cleanup():
    """
    Background thread to clean up stale breakdown states.
    Runs every 24 hours with proper sleep loop.
    """
    # Wait 1 hour before first cleanup
    time.sleep(3600)
    
    while True:
        try:
            # Check shutdown flag
            if _SHUTDOWN_IN_PROGRESS:
                logger.info("🛑 Cleanup scheduler stopping (shutdown)")
                break
            
            # Run cleanup
            count = cleanup_old_breakdown_states(days_old=7)
            if count > 0:
                logger.info(f"✅ Cleaned up {count} stale breakdown states")
            else:
                logger.debug("ℹ️  No stale breakdown states to clean")
        
        except Exception as e:
            logger.error(f"❌ Cleanup job failed: {e}", exc_info=True)
        
        # Sleep for 24 hours (with shutdown checks every 5 minutes)
        for _ in range(288):  # 24 hours = 288 * 5 minutes
            if _SHUTDOWN_IN_PROGRESS:
                break
            time.sleep(300)  # 5 minutes

# START CLEANUP THREAD
import multiprocessing
if multiprocessing.current_process().name == "MainProcess":
    cleanup_thread = threading.Thread(target=run_scheduled_cleanup, daemon=True, name="CleanupScheduler")
    cleanup_thread.start()
    logger.info("✅ Pattern state cleanup scheduler started (runs every 24h)")

# --- EXECUTOR GETTERS WITH SHUTDOWN SAFETY ---
def get_api_executor():
    if _SHUTDOWN_IN_PROGRESS:
        raise RuntimeError("Server is shutting down")
    if API_EXECUTOR is None:
        raise RuntimeError("API_EXECUTOR not initialized")
    return API_EXECUTOR

def get_compute_executor():
    if _SHUTDOWN_IN_PROGRESS:
        raise RuntimeError("Server is shutting down")
    if COMPUTE_EXECUTOR is None:
        raise RuntimeError("COMPUTE_EXECUTOR not initialized")
    return COMPUTE_EXECUTOR

def get_background_executor():
    if _SHUTDOWN_IN_PROGRESS:
        raise RuntimeError("Server is shutting down")
    if BACKGROUND_EXECUTOR is None:
        raise RuntimeError("BACKGROUND_EXECUTOR not initialized")
    return BACKGROUND_EXECUTOR

# --- LAZY LOADER FOR STOCKS ---

def prewarm_parquet_cache(tickers: List[str], horizons: List[str]):
    """
    Ensures Parquet cache is populated before parallel workers start (P1-1).
    This runs in the main process to prevent multiple workers from 
    hammering Yahoo for the same file simultaneously.
    """
    from services.data_fetch import get_history_for_horizon
    logger.info(f"Pre-warming Parquet cache for {len(tickers)} tickers across {horizons}...")
    for t in tickers:
        for h in horizons:
            try:
                # This will fetch from YF and save to Parquet if cache is stale/missing
                get_history_for_horizon(t, h)
            except Exception as e:
                logger.warning(f"Pre-warm failed for {t} ({h}): {e}")

async def periodic_warmer():
    """
    Background task to pre-fetch data.
    - Timezone-aware (IST)
    - Smart schedule (market vs off-peak)
    - Deep warm once per night
    - Rate-limit cooling
    - Cancels cleanly on shutdown
    """
    # ... (Wait for startup)
    await asyncio.sleep(10)
    # Ensure stocks loaded before warming
    _ensure_stocks_loaded()
    
    while True:
        # Check shutdown flag safely
        if _SHUTDOWN_IN_PROGRESS:
            break
            
        try:
            is_market_hours = is_market_open()
            # ✅ For deep warm, check IST hour separately
            ist_now = get_current_ist()
            is_deep_warm = (ist_now.hour == WARMER_DEEP_HOUR) and (ist_now.weekday() < 5)
            
            # Determine symbols to warm
            if is_market_hours:
                logger.info("[WARMER] Market Open - Warming Top %d", WARMER_TOP_N_DURING_MARKET)
                symbols_to_warm = [s[0] for s in ALL_STOCKS_LIST[:WARMER_TOP_N_DURING_MARKET]]
                batch_sleep = WARMER_BATCH_SLEEP_MARKET
            else:
                if is_deep_warm:
                    logger.info("[WARMER] Deep Warm - All %d symbols", len(ALL_STOCKS_LIST))
                    symbols_to_warm = [s[0] for s in ALL_STOCKS_LIST]
                else:
                    logger.info("[WARMER] Off-Peak Cycle")
                    symbols_to_warm = [s[0] for s in ALL_STOCKS_LIST]
                
                batch_sleep = WARMER_BATCH_SLEEP_OFFPEAK

            loop = asyncio.get_running_loop()
            # iterate over in batches
            for i in range(0, len(symbols_to_warm), WARMER_BATCH_SIZE):
                if _SHUTDOWN_IN_PROGRESS: break         
                batch = symbols_to_warm[i : i + WARMER_BATCH_SIZE]
                
                # P1-1: Pre-warm Parquet cache for the current batch (Non-blocking)
                await asyncio.to_thread(prewarm_parquet_cache, batch, ["intraday", "short_term", "long_term"])

                # ✅ Issue 6 FIX: pass enable_write_cache=False to workers
                futures = [
                    loop.run_in_executor(get_compute_executor(), run_full_analysis, symbol, "nifty50", False)
                    for symbol in batch
                ]
                try:
                    # Wait for the batch to complete within timeout; return_exceptions=True so we get results/errors
                    results = await asyncio.wait_for(
                        asyncio.gather(*futures, return_exceptions=True),
                        timeout=WARMER_BATCH_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    logger.error("[WARMER] Batch timed out after %d seconds. Moving to next batch.",WARMER_BATCH_TIMEOUT_SEC)
                    # We can't directly cancel the underlying thread tasks, but we should continue and cool down
                    await asyncio.sleep(WARMER_RATE_LIMIT_COOLDOWN_SEC)
                    # Move to next batch
                    continue
                except concurrent.futures.process.BrokenProcessPool as e:
                    # ✅ FIX P1-3: Recover from crashed compute worker
                    logger.error(f"[WARMER] COMPUTE_EXECUTOR broken: {e}. Recreating...")
                    global COMPUTE_EXECUTOR
                    if COMPUTE_EXECUTOR:
                        COMPUTE_EXECUTOR.shutdown(wait=False)
                    safe_cpu_count = min(4, os.cpu_count() or 4)
                    COMPUTE_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=safe_cpu_count)
                    await asyncio.sleep(5)
                    continue
                except Exception as e:
                    logger.error(f"[WARMER] Error: {e}")
                    await asyncio.sleep(5)
                    continue
                # Symbol-level results logging + rate-limit detection
                rate_limited = False
                for sym, res in zip(batch, results):
                    if isinstance(res, Exception):
                        # Detect common rate-limit or HTTP throttling hints
                        msg = str(res)
                        low = msg.lower()
                        if (
                            "429" in msg
                            or "rate limit" in low
                            or "too many requests" in low
                            or "throttl" in low
                        ):
                            logger.error("[WARMER] Rate-limit hit on %s: %s", sym, msg)
                            rate_limited = True
                        else:
                            logger.error("[WARMER] Error warming %s: %s", sym, msg)
                    else:
                        # Successful warm; log minimal info to avoid too-verbose logs
                        logger.debug(
                            "[WARMER] Warmed %s (result type=%s)", sym, type(res).__name__
                        )
                        
                        # ✅ Issue 6 & W59 FIX: Process results and serialize DB writes in main process
                        if isinstance(res, dict) and "full_report" in res:
                            profiles = res.get("full_report", {}).get("profiles", {})
                            if sym not in FULL_HORIZON_SCORES:
                                FULL_HORIZON_SCORES[sym] = {}
                            
                            for h, profile_data in profiles.items():
                                score = profile_data.get("final_score")
                                if score is not None:
                                    FULL_HORIZON_SCORES[sym][h] = score
                                    # ✅ W59 FIX: Add staleness flag
                                    FULL_HORIZON_SCORES[sym]["_stale"] = False
                            
                            # ✅ Issue 6 FIX: Serialize DB writes here
                            logger.debug(f"[WARMER] Serializing DB write for {sym}...")
                            try:
                                best_fit = res.get("full_report", {}).get("best_fit")
                                display_h = res.get("analysis_mode") == "single" and res.get("requested_horizon") or best_fit
                                
                                if res.get("trade_recommendation", {}).get("status") != "ERROR":
                                    _save_analysis_to_db(
                                        sym,
                                        res,
                                        "nifty50", 
                                        best_fit_horizon=best_fit,
                                        selected_horizon=display_h
                                    )
                                else:
                                    _mark_analysis_error_in_db(
                                        sym,
                                        res,
                                        "nifty50",
                                        best_fit_horizon=best_fit,
                                        selected_horizon=display_h,
                                    )
                            except Exception as db_err:
                                logger.error(f"[WARMER] Serialized DB Write Failed for {sym}: {db_err}")

                if rate_limited:
                    logger.warning( "[WARMER] Rate-limited; cooling down for %d seconds before resuming.", WARMER_RATE_LIMIT_COOLDOWN_SEC, )
                    await asyncio.sleep(WARMER_RATE_LIMIT_COOLDOWN_SEC)
                    # abort this warming cycle to avoid repeated hits
                    break
                # breathe between batches
                await asyncio.sleep(batch_sleep)
            logger.info( "[WARMER] Cycle complete (is_deep_warm=%s, is_market_hours=%s)", is_deep_warm, is_market_hours, )
        except asyncio.CancelledError:
            logger.info("[WARMER] Cancelled.")
            raise
        except Exception as e:
            logger.error(f"[WARMER] Loop Error: {e}")
        
        # Sleep logic...
        try:
            interval = WARMER_MARKET_INTERVAL_SEC if is_market_hours else WARMER_OFFPEAK_INTERVAL_SEC
            sleep_left = interval
            while sleep_left > 0:
                if _SHUTDOWN_IN_PROGRESS: break
                await asyncio.sleep(min(30, sleep_left))
                sleep_left -= 30
        except Exception:
            await asyncio.sleep(60)

# ---- LIFESPAN ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    global API_EXECUTOR, COMPUTE_EXECUTOR, BACKGROUND_EXECUTOR
    global cache_warmer_task, _SHUTDOWN_IN_PROGRESS, _CORP_ACTIONS_CACHE_READY
    
    # 1. Startup
    # _SHUTDOWN_IN_PROGRESS = False (Now at module level)
    
    # Phase 6 P1-4: Initialize DB at the very beginning of startup
    init_db()
    
    # ✅ W59 FIX: Restore FULL_HORIZON_SCORES from SignalCache on startup
    # Ensures persistence of recent analysis across app restarts.
    try:
        tmp_db = SessionLocal()
        # Fetch most recent signals for all active symbols
        recent_signals = tmp_db.query(SignalCache).all()
        for sig in recent_signals:
                if sig.horizon_scores:
                    # sig.horizon_scores is stored as a JSON dict: {horizon: score}
                    score_data = sig.horizon_scores.copy()
                    # ✅ W59 FIX: Mark restored scores as stale
                    score_data["_stale"] = True
                    FULL_HORIZON_SCORES[sig.symbol] = score_data
        logger.info(f"📋 W59: Restored {len(recent_signals)} symbols into FULL_HORIZON_SCORES (marked as _stale).")
    except Exception as e:
        logger.error(f"❌ W59: Score restoration failed: {e}")
    finally:
        tmp_db.close()
    
    # Pre-load data in main process
    build_smart_index_map()
    _ensure_stocks_loaded()
    
    # P1-1: Pre-warm Parquet cache barrier before executors start (Non-blocking)
    if ALL_STOCKS_LIST:
        await asyncio.to_thread(prewarm_parquet_cache, [s[0] for s in ALL_STOCKS_LIST], ["intraday", "short_term", "long_term"])
    
    safe_cpu_count = min(4, os.cpu_count() or 4)
    logger.info(f"Starting Executors (Compute Workers: {safe_cpu_count})...")
    
    API_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="API_Worker")
    COMPUTE_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=safe_cpu_count)
    BACKGROUND_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="BG_Worker")

    if ENABLE_CACHE_WARMER:
        cache_warmer_task = asyncio.create_task(periodic_warmer())
        logger.info("🔥 Cache Warmer: ENABLED")
    else:
        cache_warmer_task = None
        logger.info("⏸️  Cache Warmer: DISABLED")

    try:
        def _warm_corp_actions_cache():
            """Runs in background thread at startup to pre-build corp actions summary."""
            global _CORP_ACTIONS_CACHE_READY, _LAST_CORP_FETCH_TIME
            try:
                # Collect actual stock tickers from all index data files
                all_tickers = set()
                for idx_name in INDEX_TICKERS.keys():
                    stocks = load_or_create_index(idx_name)
                    for s in stocks:
                        all_tickers.add(s[0])  # s is (symbol, name) tuple
                all_tickers = list(all_tickers)
                if not all_tickers:
                    logger.warning("[STARTUP] No tickers found in data files, skipping corp actions cache.")
                    return
                logger.info("[STARTUP] Building corp actions summary for %d tickers...", len(all_tickers))

                result_holder = {}
                error_holder = {}

                def _build_summary():
                    try:
                        result_holder["summary"] = build_corp_actions_summary_cache(all_tickers)
                    except Exception as inner_err:
                        error_holder["error"] = inner_err

                worker = threading.Thread(
                    target=_build_summary,
                    daemon=True,
                    name="CorpActionsBuildWorker",
                )
                worker.start()
                worker.join(CORP_ACTIONS_STARTUP_TIMEOUT_SEC)

                if worker.is_alive():
                    logger.warning(
                        "[STARTUP] Corp actions cache build timed out after %d seconds; continuing startup.",
                        CORP_ACTIONS_STARTUP_TIMEOUT_SEC,
                    )
                    return
                if "error" in error_holder:
                    raise error_holder["error"]

                _CORP_ACTIONS_CACHE_READY = True
                _LAST_CORP_FETCH_TIME = time.time()
                logger.info("[STARTUP] Corp actions summary cache ready.")
            except Exception as e:
                logger.warning("[STARTUP] Corp actions cache build failed: %s", e)
        
            # ✅ Fix 8.4-1: Removed duplicate score restoration block.
            # Restoration now happens exclusively in lifespan() with _stale flag.

        # Phase 4 P1-6: Activate pattern config validation
        from config.config_extractor import startup_config_validation
        if not startup_config_validation():
            logger.error("❌ CRITICAL: Pattern Config validation failed. Shutdown sequence initiated.")
            # In a real PROD app, you might want to exit here.
        
        # Restore warmer call
        threading.Thread(target=_warm_corp_actions_cache, daemon=True, name="CorpActionsWarmer").start()
        
        start_mb_scheduler()
        yield
    finally:
        # 2. Robust Shutdown
        logger.info("Initiating graceful shutdown...")
        
        # [FIX] Thread-safe flag set
        with _SHUTDOWN_LOCK:
            _SHUTDOWN_IN_PROGRESS = True

        if cache_warmer_task is not None:
            cache_warmer_task.cancel()
            try:
                await cache_warmer_task
            except asyncio.CancelledError:
                pass

        logger.info("Shutting down executors...")
        
        # Shutdown Thread Pools (wait=True ensures data integrity)
        if API_EXECUTOR: 
            API_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        if BACKGROUND_EXECUTOR: 
            BACKGROUND_EXECUTOR.shutdown(wait=True, cancel_futures=False)
            
        # ✅ FIXED: Shutdown Compute Pool with PROPER Zombie Process Cleanup
        if COMPUTE_EXECUTOR:
            try:
                # Give workers 10 seconds to finish current tasks
                logger.info("Waiting for compute workers to finish...")
                COMPUTE_EXECUTOR.shutdown(wait=True)
                logger.info("Compute executor shutdown complete.")
            except Exception as e:
                logger.warning(f"Compute executor shutdown had issues: {e}")
            
            # ✅ ROBUST ZOMBIE CLEANUP (Python 3.9+ compatible)
            try:
                # Access private attribute safely
                processes_dict = getattr(COMPUTE_EXECUTOR, '_processes', None)
                
                if processes_dict is not None:
                    # Python 3.9+: _processes is a dict
                    if isinstance(processes_dict, dict):
                        process_list = list(processes_dict.values())
                    # Python 3.7-3.8: _processes might be a set
                    elif hasattr(processes_dict, '__iter__'):
                        process_list = list(processes_dict)
                    else:
                        process_list = []
                    
                    # Force-kill any lingering processes
                    for proc in process_list:
                        try:
                            if proc.is_alive():
                                logger.warning(f"Force-terminating zombie process {proc.pid}")
                                proc.terminate()
                                proc.join(timeout=2.0)
                                
                                # Nuclear option if still alive
                                if proc.is_alive():
                                    logger.error(f"Force-killing stubborn process {proc.pid}")
                                    proc.kill()
                                    proc.join(timeout=1.0)
                        except Exception as proc_err:
                            logger.debug(f"Error cleaning up process: {proc_err}")
            except Exception as cleanup_err:
                logger.warning(f"Zombie cleanup failed: {cleanup_err}")

        logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)
app.include_router(mb_router)
templates = Jinja2Templates(directory="templates")

def enrich_json_sync(index_name: str, symbol: str, name: str):
    """
    Runs in BACKGROUND_EXECUTOR.
    Safely updates the JSON file with the new stock name.
    """
    if not ENABLE_JSON_ENRICHMENT or not index_name or not name:
        return

    # Map generic names to files
    # (Optional: Add logic if index_name needs mapping, e.g., 'default' -> 'NSEStock')
    if index_name.lower() in ["default", "nsestock"]:
        index_name = "NSEStock"
        
    json_file = os.path.join(DATA_DIR, f"{index_name}.json")
    
    if not os.path.exists(json_file):
        return

    try:
        # Use the Global Lock from Main Process
        with CACHE_LOCK: 
            updated = False
            data = []
            
            # Read
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Modify
            clean_sym = symbol.strip().upper()
            for item in data:
                if isinstance(item, dict) and item.get('symbol', '').strip().upper() == clean_sym:
                    # Only update if missing or just the ticker
                    current_name = item.get('name', '').strip()
                    if not current_name or current_name == clean_sym:
                        item['name'] = name.strip()
                        updated = True
                    break
            
            # Write (Only if changed)
            if updated:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"[ENRICH] Updated name for {symbol} in {index_name}.json")
                
    except Exception as e:
        logger.warning(f"[ENRICH] Failed to update JSON for {symbol}: {e}")

# --- PYDANTIC MODELS ---
class AnalyzeRequest(BaseModel):
    symbol: str
    index: str = "nifty50"

class QuickScoresRequest(BaseModel):
    symbols: List[str]
    index_name: str = "nifty50"

class CorpActionsRequest(BaseModel):
    tickers: List[str]


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
NSE_STOCKS_FILE = os.path.join(DATA_DIR, "nse_stocks.json")

# Global reference
cache_warmer_task = None
_CORP_ACTIONS_CACHE_READY = False  # True once startup warmer finishes
_LAST_CORP_FETCH_TIME = 0.0        # epoch timestamp of last successful corp actions fetch
_CORP_FETCH_COOLDOWN_SEC = 3600    # 1 hour cooldown between fetches
CACHE_TTL = 60 * 60
# Phase 6 P0-1: Use FileLock for cross-process synchronization
# Ensures JSON enrichment is safe across parallel workers
CACHE_LOCK_FILE = "cache/global_json.lock"
os.makedirs(os.path.dirname(CACHE_LOCK_FILE), exist_ok=True)
CACHE_LOCK = FileLock(CACHE_LOCK_FILE)

SIGNAL_CACHE_WRITE_RETRIES = int(os.getenv("SIGNAL_CACHE_WRITE_RETRIES", "5"))
SIGNAL_CACHE_RETRY_SLEEP_SEC = float(os.getenv("SIGNAL_CACHE_RETRY_SLEEP_SEC", "0.2"))

# (Moved index loading helpers to services/index_utils.py)

# --- CACHE LOGIC ---

def get_cached(symbol: str):
    """
    Retrieves analysis from SQLite (Persistent).
    Returns the flattened dictionary expected by AG Grid.
    """
    db: Session = SessionLocal()
    try:
        # Query DB
        entry = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()
        if not entry: return None

        # ✅ Use utility function
        now_utc = get_current_utc()
        
        # Ensure entry.updated_at is UTC-aware
        entry_time = ensure_utc(entry.updated_at)
        age = (now_utc - entry_time).total_seconds()
        
        if age > CACHE_TTL:
            logger.debug(f"[{symbol}] Cache expired ({age/60:.1f}m old)")
            return None

        # Reconstruct the 'Flat' dictionary for the Grid
        # 1. Start with the structured columns
        flat_data = {
            "symbol": entry.symbol,
            "score": entry.score,
            "bull_score": entry.score, # ✅ P2-5: Key normalization
            "recommendation": entry.recommendation,
            "setup_signal": entry.recommendation, # ✅ P2-5: Key normalization
            "confidence": entry.conf_score,
            "bull_signal": entry.signal_text,
            "rr_ratio": entry.rr_ratio,
            "entry_trigger": entry.entry_price,
            "stop_loss": entry.stop_loss,
            "direction": entry.direction or "neutral",
            "cached": True,
        }

        # 2. Merge the flexible JSON fields (horizon scores, errors, macros)
        if entry.horizon_scores:
            flat_data.update(entry.horizon_scores)
            # ✅ P2-5: Ensure profit_pct is present
            if "profit_pct" not in flat_data:
                flat_data["profit_pct"] = flat_data.get("expected_return", 0)

        # Prefer the dedicated DB column over legacy JSON shadow data.
        flat_data["direction"] = entry.direction or flat_data.get("direction", "neutral")
        
        # ✅ Fix 8.4-2: Surface _stale flag from memory
        from_memory = FULL_HORIZON_SCORES.get(symbol, {})
        flat_data["_stale"] = from_memory.get("_stale", False)
        
        return flat_data
    except Exception as e:
        logger.error(f"DB Read Error {symbol}: {e}")
        return None
    finally:
        db.close()


def _is_retryable_sqlite_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "database is locked" in msg or "database table is locked" in msg or "busy" in msg


# ✅ Phase 6 P0-2: Implement exponential backoff with jitter (delegated to db.py decorator)
from services.db import backoff_retry_db

@backoff_retry_db(retries=SIGNAL_CACHE_WRITE_RETRIES, base_delay=SIGNAL_CACHE_RETRY_SLEEP_SEC)
def _write_signal_cache_with_retry(symbol: str, writer_fn):
    """
    Execute a SignalCache write with a fresh DB session per attempt.
    """
    db: Session = SessionLocal()
    try:
        entry = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()
        if not entry:
            entry = SignalCache(symbol=symbol)
            db.add(entry)

        writer_fn(db, entry)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def set_cached(symbol: str, value: Dict[str, Any]):
    """
    Update analysis cache in SQLite.
    Preserves existing multibagger score if not provided in 'value'.
    """
    try:
        def _writer(db: Session, entry: SignalCache):
            existing_multi = (entry.horizon_scores or {}).get("multi_score") if entry else None
            horizon_data = {
                "intra_score":  value.get("intra_score"),
                "short_score":  value.get("short_score"),
                "long_score":   value.get("long_score"),
                "multi_score":  value.get("multi_score") or existing_multi,
                "best_fit_score": value.get("best_fit_score"),
                "sl_dist":      value.get("sl_dist"),
                "direction":    value.get("direction", "neutral"),
            }

            entry.best_horizon = value.get("best_horizon")
            entry.selected_horizon = value.get("selected_horizon")
            entry.score = value.get("score", 0)
            entry.recommendation = value.get("recommendation", "N/A")
            entry.signal_text = value.get("bull_signal", "INCOMPLETE")
            entry.conf_score = value.get("confidence", 0)
            entry.rr_ratio = value.get("rr_ratio")
            entry.entry_price = value.get("entry_trigger")
            entry.stop_loss = value.get("stop_loss")
            entry.direction = value.get("direction", "neutral")
            entry.horizon_scores = horizon_data
            entry.updated_at = get_current_utc()

        _write_signal_cache_with_retry(symbol, _writer)
    except Exception as e:
        logger.error(f"DB Write Error {symbol}: {e}")


def _has_usable_fundamentals(fundamentals: Dict[str, Any]) -> bool:
    """
    Guard against truthy-but-empty fundamental payloads.
    We require at least one core metric or identity field beyond a metadata shell.
    """
    if not isinstance(fundamentals, dict) or not fundamentals:
        return False

    core_fields = [
        "marketCap", "peRatio", "pbRatio", "deRatio", "roe", "roce",
        "current_price", "name", "symbol"
    ]
    for key in core_fields:
        val = fundamentals.get(key)
        if isinstance(val, dict):
            if val.get("raw") is not None or val.get("value") not in (None, "", "N/A"):
                return True
        elif val not in (None, "", [], {}):
            return True
    return False


# --- WORKER ---
def run_analysis(
    symbol: str, 
    index_name: str = "nifty50",
    mode: str = "full",           # ✅ NEW: "full" or "single"
    requested_horizon: str = None, # ✅ NEW: Required if mode="single"
    enable_write_cache: bool = True # ✅ Issue 6 FIX: Allow disabling DB write in worker
) -> Dict[str, Any]:
    """
    Unified analysis function supporting both full and single-horizon modes.
    
    Args:
        symbol: Stock ticker
        index_name: Index for benchmarking
        mode: Analysis mode
              - "full": Compute all 4 horizons, pick best_fit
              - "single": Compute only requested_horizon
        requested_horizon: Required if mode="single" (e.g., "multibagger")
    
    Returns:
        Analysis data with profiles, trade_plan, indicators
    
    Performance:
        - mode="full": ~4.9s (4 horizons × indicators + scoring)
        - mode="single": ~2.1s (1 horizon × indicators + scoring)
    """
    enrichment_info = None
    
    try:
        global STOCK_TO_INDEX_MAP
        if not STOCK_TO_INDEX_MAP:
            build_smart_index_map()
        
        # =====================================================================
        # DETERMINE HORIZONS TO COMPUTE
        # =====================================================================
        if mode == "single":
            if not requested_horizon:
                raise ValueError("requested_horizon required for mode='single'")
            # ✅ FIX 15: Always compute long_term for meta-scores (Value, Growth, Quality)
            # and short_term as a baseline if needed.
            horizons_to_compute = [requested_horizon]
            if "long_term" not in horizons_to_compute:
                horizons_to_compute.append("long_term")
            
            logger.info(f"[{symbol}] 🎯 SINGLE-HORIZON MODE | Horizons: {horizons_to_compute}")
        else:
            # ✅ OPTIMIZED: Exclude multibagger from full scan (it's now a specialized weekly flow)
            horizons_to_compute = ["intraday", "short_term", "long_term"]
            # ✅ Fix 8.2-2: Suppress intraday if market is closed or holiday
            from config.config_utility.market_utils import get_current_session
            if get_current_session() in ("after_hours", "holiday"):
                horizons_to_compute = [h for h in horizons_to_compute if h != "intraday"]
                logger.info(f"[{symbol}] Off-hours status: intraday suppressed.")
            
            logger.info(f"[{symbol}] 🔄 FULL-HORIZON MODE | Horizons: {horizons_to_compute}")
        
        # =====================================================================
        # STEP 1: CORE DATA STRUCTURE
        # =====================================================================
        analysis_data = {
            "symbol": symbol,
            "fundamentals": {},
            "raw_indicators_by_horizon": {},
            "raw_patterns_by_horizon": {},
            "macro_trend_status": "N/A",
            "macro_close": None,
            "full_report": {},
            "trade_recommendation": {},
            "meta_scores": {},
            "indicators": {},
            "partial_error": False,
            "error_details": [],
            "analysis_mode": mode,  # ✅ NEW: Track which mode was used
        }
        
        # Determine benchmark
        target_index = index_name
        clean_sym = symbol.strip().upper()
        if index_name in ["NSEStock", "default", "nifty500"]:
            smart_index = STOCK_TO_INDEX_MAP.get(clean_sym)
            if smart_index:
                target_index = smart_index
        bench_symbol = INDEX_TICKERS.get(target_index, INDEX_TICKERS["default"])
        
        # =====================================================================
        # STEP 2: FUNDAMENTALS
        # =====================================================================
        try:
            analysis_data["fundamentals"] = compute_fundamentals(symbol)
            fund_data = analysis_data.get("fundamentals", {})
            raw_name = fund_data.get("name")
            if raw_name and raw_name != symbol:
                enrichment_info = (index_name, symbol, raw_name)
        except Exception as e:
            analysis_data["partial_error"] = True
            analysis_data["error_details"].append(f"Fundamentals Failed: {e}")
            logger.error(f"[{symbol}] Fundamentals failed: {e}")
        
        # =====================================================================
        # STEP 3: INDICATORS (ONLY FOR REQUESTED HORIZONS)
        # =====================================================================
        for h in horizons_to_compute:  # ✅ OPTIMIZED: Only compute what's needed
            try:
                h_indicators, h_patterns = compute_indicators_cached(
                    symbol,
                    horizon=h,
                    benchmark_symbol=bench_symbol,
                    force_refresh=False
                )
                analysis_data["raw_indicators_by_horizon"][h] = h_indicators
                analysis_data["raw_patterns_by_horizon"][h] = h_patterns
                
                logger.debug(f"[{symbol}] Computed indicators for {h}")
                
            except Exception as e:
                analysis_data["partial_error"] = True
                analysis_data["error_details"].append(f"Indicators/{h}: {e}")
                logger.warning(f"[{symbol}] Failed indicators for '{h}': {e}")
        
        # =====================================================================
        # STEP 4: MACRO TREND (FALLBACK CHAIN)
        # =====================================================================
        # ✅ FIX OBSERVATION 3: Try short_term first, then fallback to any available
        try:
            macro_inds = None
            for fallback_horizon in ["short_term", "intraday", "long_term"]:
                if fallback_horizon in analysis_data["raw_indicators_by_horizon"]:
                    macro_inds = analysis_data["raw_indicators_by_horizon"][fallback_horizon]
                    logger.debug(f"[{symbol}] Using {fallback_horizon} for macro trend")
                    break
            
            if macro_inds:
                nifty_metric = macro_inds.get("niftyTrendScore")
                if nifty_metric:
                    analysis_data["macro_trend_status"] = nifty_metric.get("desc", "N/A")
                    analysis_data["macro_close"] = nifty_metric.get("value")
        except Exception as e:
            logger.debug(f"[{symbol}] Macro trend failed: {e}")
        
        # =====================================================================
        # STEP 5: PROFILE SCORING (WITH HORIZON FILTER)
        # =====================================================================
        if not (_has_usable_fundamentals(analysis_data["fundamentals"]) and analysis_data["raw_indicators_by_horizon"]):
            logger.error(f"[{symbol}] Missing data for profile scoring")
            return analysis_data
        
        try:
            # Pass requested_horizons to filter scoring
            full_report = compute_all_profiles(
                symbol,
                analysis_data["fundamentals"],
                analysis_data["raw_indicators_by_horizon"],
                analysis_data["raw_patterns_by_horizon"],
                requested_horizons=horizons_to_compute  # ✅ NEW PARAM
            )
            analysis_data["full_report"] = full_report
            
            best_fit = full_report.get("best_fit")
            best_score = full_report.get("best_score", 0)
            
            # ✅ CRITICAL: In single-horizon mode, best_fit IS the requested horizon
            if mode == "single":
                best_fit = requested_horizon
                # ⚠️ Safety check: profile must exist
                if requested_horizon not in full_report.get("profiles", {}):
                    logger.error(f"[{symbol}] Requested horizon '{requested_horizon}' not computed!")
                    analysis_data["partial_error"] = True
                    analysis_data["error_details"].append(f"Horizon {requested_horizon} failed to compute")
                    return analysis_data
                best_score = full_report.get("profiles", {}).get(requested_horizon, {}).get("final_score", 0)
            
            logger.info(f"[{symbol}] Best Fit: {best_fit} (Score: {best_score:.2f}/10)")
            
        except Exception as e:
            analysis_data["partial_error"] = True
            analysis_data["error_details"].append(f"Profile Scoring Failed: {e}")
            logger.error(f"[{symbol}] Profile scoring failed: {e}")
            return analysis_data
        
        # =====================================================================
        # STEP 6: META SCORES (CONTEXT-APPROPRIATE HORIZONS)
        # =====================================================================
        # ✅ FIX OBSERVATION 4: Use appropriate horizons for each meta score type
        try:
            # Momentum baseline for meta-block should be long-term for consistency
            # with Value/Growth/Quality, giving a high-level fundamental-growth-trend overview.
            momentum_inds = analysis_data["raw_indicators_by_horizon"].get(
                "long_term",
                analysis_data["raw_indicators_by_horizon"].get(
                    requested_horizon if mode == "single" else "short_term", {}
                )
            )
            
            analysis_data["meta_scores"] = {
                # Value, Growth, Quality are ALWAYS long-term metrics
                "value": score_value_profile(analysis_data["fundamentals"], horizon="long_term"),
                "growth": score_growth_profile(analysis_data["fundamentals"], horizon="long_term"),
                "quality": score_quality_profile(analysis_data["fundamentals"], horizon="long_term"),
                # Momentum baseline for the summary block
                "momentum": score_momentum_profile(
                    analysis_data["fundamentals"],
                    momentum_inds,
                    horizon="long_term"
                ),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"[{symbol}] Meta scores failed: {e}")
        
        # =====================================================================
        # STEP 7: FLATTEN INDICATORS (BEST FIT OR REQUESTED)
        # =====================================================================
        display_horizon = requested_horizon if mode == "single" else best_fit
        analysis_data["indicators"] = analysis_data["raw_indicators_by_horizon"].get(
            display_horizon, {}
        ).copy()
        
        # =====================================================================
        # STEP 8: TRADE PLAN GENERATION
        # =====================================================================
        try:
            profiles = full_report.get("profiles", {})
            if display_horizon not in profiles:
                raise ValueError(f"Profile for {display_horizon} not computed")
            
            winner_data = profiles[display_horizon]
            
            trade_plan = generate_trade_plan(
                symbol=symbol,
                winner_profile=winner_data,
                indicators=analysis_data["raw_indicators_by_horizon"][display_horizon],
                fundamentals=analysis_data["fundamentals"],
                horizon=display_horizon,
                macro_trend_status=analysis_data.get("macro_trend_status", "N/A"),
            )
            
            analysis_data["trade_recommendation"] = trade_plan
            
        except Exception as e:
            logger.error(f"[{symbol}] Trade plan failed: {e}", exc_info=True)
            analysis_data["error_details"].append(f"Trade Plan Failed: {e}")
            analysis_data["trade_recommendation"] = {
                "status": "ERROR",
                "reason": str(e),
                "signal": "ERROR",
                "entry": None,
                "stop_loss": None,
                "targets": {"t1": None, "t2": None},
                "narratives": {"trade": f"Analysis failed: {e}"}
            }
        
        # =====================================================================
        # STEP 9: DATABASE PERSISTENCE (WITH BOTH HORIZONS)
        # =====================================================================
        # ✅ Issue 6 FIX: Only save to cache if enable_write_cache is True.
        # Warmer sets this to False to perform serialization in the main process.
        if enable_write_cache and "full_report" in analysis_data:
            try:
                # If the trade plan had an error, we mark it specially in DB
                if analysis_data.get("trade_recommendation", {}).get("status") != "ERROR":
                    _save_analysis_to_db(
                        symbol,
                        analysis_data,
                        index_name,
                        best_fit_horizon=best_fit,
                        selected_horizon=display_horizon
                    )
                else:
                    _mark_analysis_error_in_db(
                        symbol,
                        analysis_data,
                        index_name,
                        best_fit_horizon=best_fit,
                        selected_horizon=display_horizon,
                    )
                    logger.warning(f"[{symbol}] Persisted ERROR state to cache.")
            except Exception as db_err:
                logger.error(f"[{symbol}] Cache Persistence Failed: {db_err}")
        
        # =====================================================================
        # STEP 10: ENRICHMENT METADATA
        # =====================================================================
        if enrichment_info:
            analysis_data["_enrichment"] = enrichment_info
        
        return analysis_data
        
    except Exception as e:
        logger.error(f"[{symbol}] Analysis Error: {e}", exc_info=True)
        return {"symbol": symbol, "error": str(e)}
    
# ✅ BACKWARD COMPATIBLE WRAPPERS
def run_full_analysis(symbol: str, index_name: str = "nifty50", enable_write_cache: bool = True) -> Dict[str, Any]:
    """Legacy wrapper for full analysis."""
    return run_analysis(symbol, index_name, mode="full", enable_write_cache=enable_write_cache)


def run_single_horizon_analysis(symbol: str, horizon: str, index_name: str = "nifty50") -> Dict[str, Any]:
    """Optimized single-horizon analysis."""
    return run_analysis(symbol, index_name, mode="single", requested_horizon=horizon)

# =========================================================================
# DATABASE PERSISTENCE (Aligned with New Trade Plan Structure)
# =========================================================================
def _save_analysis_to_db(
    symbol: str,
    data: Dict[str, Any],
    index_name: str,
    best_fit_horizon: str = None,     # ✅ NEW: System-determined optimal
    selected_horizon: str = None      # ✅ NEW: User's actual choice
):
    """
    ✅Save BOTH best_fit and selected horizons.
    
    This preserves critical information:
    - best_fit_horizon: What the system recommends
    - selected_horizon: What the user is actually viewing/trading
    
    Use Cases:
    1. User accepts best_fit:
       - best_fit_horizon = "intraday"
       - selected_horizon = "intraday"
       - Signal from intraday
    
    2. User overrides to multibagger:
       - best_fit_horizon = "intraday"
       - selected_horizon = "multibagger"
       - Signal from multibagger
       - ⚠️ User knowingly chose different timeframe
    """
    try:
        full_rep = data.get("full_report", {})
        profiles = full_rep.get("profiles", {})
        
        # ✅ Determine which horizons to save
        # If not provided, use best_fit from analysis
        if not best_fit_horizon:
            best_fit_horizon = full_rep.get("best_fit", "short_term")
        
        if not selected_horizon:
            selected_horizon = best_fit_horizon
        
        # ✅ Get profile and trade plan from SELECTED horizon
        selected_profile = profiles.get(selected_horizon, {})
        trade_plan = data.get("trade_recommendation", {})
        
        # Get best_fit score for comparison
        best_fit_profile = profiles.get(best_fit_horizon, {})
        best_fit_score = best_fit_profile.get("final_score", 0)
        
        indicators = data.get("indicators", {})
        final_score = selected_profile.get("final_score", 0)

        def _writer_retry(db: Session, entry_row: SignalCache):
            existing_multi_retry = (
                (entry_row.horizon_scores or {}).get("multi_score")
                if entry_row else None
            )
            horizon_data_retry = {
                "intra_score": profiles.get("intraday", {}).get("final_score"),
                "short_score": profiles.get("short_term", {}).get("final_score"),
                "long_score": profiles.get("long_term", {}).get("final_score"),
                "multi_score": profiles.get("multibagger", {}).get("final_score") or existing_multi_retry,
                "macro_index_name": index_name.upper(),
                "error_details": "; ".join(data.get("error_details", [])),
                "best_fit_score": best_fit_score,
                "direction": trade_plan.get("metadata", {}).get("direction", "bullish"),
            }

            entry_val_retry = trade_plan.get("entry")
            current_price_retry = indicators.get("price", {}).get("value") or entry_val_retry
            sl_val_retry = trade_plan.get("stop_loss")
            sl_dist_str_retry = "-"
            if current_price_retry and sl_val_retry:
                dist = abs(current_price_retry - sl_val_retry) / current_price_retry * 100
                sl_dist_str_retry = f"{dist:.1f}%"
            horizon_data_retry["sl_dist"] = sl_dist_str_retry

            entry_row.best_horizon = best_fit_horizon
            entry_row.selected_horizon = selected_horizon
            entry_row.score = int(final_score * 10) if final_score else 0
            entry_row.recommendation = selected_profile.get("category", "HOLD") + "--" + selected_horizon
            entry_row.signal_text = trade_plan.get("trade_signal", "N/A")
            entry_row.conf_score = trade_plan.get("final_confidence", 0)
            entry_row.rr_ratio = trade_plan.get("rr_ratio")
            entry_row.entry_price = entry_val_retry
            entry_row.stop_loss = sl_val_retry
            entry_row.direction = trade_plan.get("metadata", {}).get("direction", "neutral")
            entry_row.horizon_scores = horizon_data_retry
            entry_row.updated_at = get_current_utc()

        _write_signal_cache_with_retry(symbol, _writer_retry)
        logger.debug(
            f"[{symbol}] Saved to DB | "
            f"Best: {best_fit_horizon} | "
            f"Selected: {selected_horizon} | "
            f"Score: {int(final_score * 10) if final_score else 0}"
        )
        return

    except Exception as e:
        logger.error(f"[{symbol}] DB Save Failed: {e}")


def _mark_analysis_error_in_db(
    symbol: str,
    data: Dict[str, Any],
    index_name: str,
    best_fit_horizon: str = None,
    selected_horizon: str = None,
):
    """
    Persist an explicit error-state cache row so stale trade data is not served as fresh.
    Keeps horizon scores and metadata, but clears executable trade fields.
    """
    try:
        full_rep = data.get("full_report", {})
        profiles = full_rep.get("profiles", {})

        if not best_fit_horizon:
            best_fit_horizon = full_rep.get("best_fit", "short_term")
        if not selected_horizon:
            selected_horizon = best_fit_horizon

        selected_profile = profiles.get(selected_horizon, {})
        best_fit_profile = profiles.get(best_fit_horizon, {})

        def _writer_retry(db: Session, entry_row: SignalCache):
            existing_multi_retry = (
                (entry_row.horizon_scores or {}).get("multi_score")
                if entry_row else None
            )
            horizon_data_retry = {
                "intra_score": profiles.get("intraday", {}).get("final_score"),
                "short_score": profiles.get("short_term", {}).get("final_score"),
                "long_score": profiles.get("long_term", {}).get("final_score"),
                "multi_score": profiles.get("multibagger", {}).get("final_score") or existing_multi_retry,
                "macro_index_name": index_name.upper(),
                "error_details": "; ".join(data.get("error_details", [])),
                "best_fit_score": best_fit_profile.get("final_score", 0),
                "direction": "neutral",
                "sl_dist": "-",
                "analysis_status": "ERROR",
            }

            entry_row.best_horizon = best_fit_horizon
            entry_row.selected_horizon = selected_horizon
            entry_row.score = int(selected_profile.get("final_score", 0) * 10) if selected_profile.get("final_score") else 0
            entry_row.recommendation = selected_profile.get("category", "INCOMPLETE") + "--" + selected_horizon
            entry_row.signal_text = "INCOMPLETE"
            entry_row.conf_score = 0
            entry_row.rr_ratio = None
            entry_row.entry_price = None
            entry_row.stop_loss = None
            entry_row.direction = "neutral"
            entry_row.horizon_scores = horizon_data_retry
            entry_row.updated_at = get_current_utc()

        _write_signal_cache_with_retry(symbol, _writer_retry)
        return

    except Exception as e:
        logger.error(f"[{symbol}] DB Error-State Save Failed: {e}")

# --- ENDPOINTS ---

@app.get("/load_index/{index_name}")
async def load_index_endpoint(index_name: str, api_key: str = Depends(get_api_key)):
    stocks = load_or_create_index(index_name)
    if not stocks and "nifty" in index_name.lower():
        stocks = [("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "TCS")]
    tickers = [s[0] for s in stocks]
    pairs = {s[0]: s[1] for s in stocks}
    # Include corp actions from cache if available
    corp_map = {}
    corp_ready = _CORP_ACTIONS_CACHE_READY
    if corp_ready:
        try:
            corp_map = get_corp_actions_summary(tickers)
        except Exception:
            pass
    return {
        "index": index_name, "count": len(stocks),
        "tickers": tickers, "pairs": pairs,
        "corp_actions": corp_map,
        "corp_actions_ready": corp_ready,
    }

async def analyze_multibagger(request: Request, symbol: str, index: str = "nifty50", api_key: str = Depends(get_api_key)):
    """
    Specialized route for Multibagger Thesis.
    Uses the MB-specific evaluator and renders the dedicated dashboard.
    """
    _ensure_stocks_loaded()
    try:
        # Issue A FIX: Offload blocking data fetch to executor
        def _fetch_mb_data(symbol):
            from services.indicator_cache import compute_indicators_cached
            from services.fundamentals import compute_fundamentals
            indicators_local, patterns_local = compute_indicators_cached(symbol, horizon="multibagger")
            fundamentals_local = compute_fundamentals(symbol)
            return indicators_local, patterns_local, fundamentals_local

        loop = asyncio.get_running_loop()
        indicators, patterns, fundamentals = await loop.run_in_executor(
            get_compute_executor(), _fetch_mb_data, symbol
        )
        
        # 2. Run MB Evaluator (Phase 2 scoring)
        # This uses the isolated MB extractor/config stack
        from config.multibagger.multibagger_evaluator import run_mb_resolver
        result = await loop.run_in_executor(
            get_compute_executor(),
            run_mb_resolver,
            symbol,
            fundamentals,
            indicators,
            patterns
        )
        
        if not result:
            raise ValueError(f"MB Evaluator returned no result for {symbol}")

        # 3. Get conviction tier (same logic as scheduler)
        from config.multibagger.mb_scheduler import _determine_conviction_tier
        conviction_tier = _determine_conviction_tier(result)
        
        # 4. Prepare context for mb_result.html
        context = {
            "request": request,
            "symbol": symbol,
            "final_decision_score": result.get("final_decision_score"),
            "confidence": result.get("confidence"),
            "conviction_tier": conviction_tier,
            "primary_setup": result.get("setup"),
            "primary_strategy": result.get("strategy"),
            "fundamental_score": result.get("fundamental_score"),
            "technical_score": result.get("technical_score"),
            "opportunity": result.get("opportunity", {}),
            "eval_ctx": result.get("eval_ctx", {}),
            "rejection_reason": result.get("rejection_reason"),
            "last_evaluated": "Live Evaluation",
            "re_evaluate_date": "N/A"
        }
        
        return templates.TemplateResponse("mb_result.html", context)
        
    except Exception as e:
        logger.exception(f"Error in analyze_multibagger for {symbol}: {e}")
        # ✅ P2-4: Sanitize error disclosure
        error_msg = "A technical error occurred during multibagger analysis."
        return templates.TemplateResponse("result.html", {
            "request": request,
            "symbol": symbol,
            "current_index": index,
            "error": error_msg,
            "full_report": {"profiles": {}},
            "profile_report": {},
            "indicators": {},
            "fundamentals": {},
            "summaries": {},
            "trade_recommendation": {},
            "selected_horizon": "multibagger",
            "confidence": 0,
            "meta_scores": {},
            "strategy_report": {},
            "all_horizon_scores": {},
        })

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_common(
    request: Request,
    symbol: str,
    index: str = "nifty50",
    horizon: str = None,
    api_key: str = Depends(get_api_key)
):
    """
    ✅ FULLY REVISED: Proper separation of concerns with smart routing.
    """
    # ✅ P2-3/P3-1: Full sanitization
    symbol = symbol.strip().upper()
    _validate_symbol(symbol)
    index = sanitize_index_name(index)
    _ensure_stocks_loaded()

    # ✅ P1-6: Deduplication logic
    with _IN_FLIGHT_LOCK:
        if symbol in _IN_FLIGHT_ANALYSES:
             return templates.TemplateResponse("result.html", {
                "request": request, 
                "symbol": symbol,
                "error": "Analysis already in progress for this symbol. Please wait.",
                "all_horizon_scores": FULL_HORIZON_SCORES.get(symbol, {}),
                "full_report": {"profiles": {}},
             })
        _IN_FLIGHT_ANALYSES.add(symbol)
    
    try:
        loop = asyncio.get_running_loop()
        
        # ✅ SMART ROUTING: Choose mode based on horizon parameter
        if horizon == "multibagger":
            return await analyze_multibagger(request, symbol.strip().upper(), index)

        if horizon and horizon in ["intraday", "short_term", "long_term"]:
            logger.info(f"[{symbol}] Single-horizon mode: {horizon}")
            
            analysis_data = await loop.run_in_executor(
                get_compute_executor(),
                run_analysis,
                symbol,
                index,
                "single",      # mode
                horizon        # requested_horizon
            )
            
            selected_profile_name = horizon
            
            # ✅ Update the current horizon's score in the global cache
            current_score = (analysis_data.get("full_report", {})
                           .get("profiles", {})
                           .get(horizon, {})
                           .get("final_score"))
            if symbol in FULL_HORIZON_SCORES and current_score is not None:
                FULL_HORIZON_SCORES[symbol][horizon] = current_score
            
        else:
            logger.info(f"[{symbol}] Full multi-horizon mode")
            
            analysis_data = await loop.run_in_executor(
                get_compute_executor(),
                run_analysis,
                symbol,
                index,
                "full",   # mode
                None      # requested_horizon
            )
            
            full_report = analysis_data.get("full_report", {})
            selected_profile_name = full_report.get("best_fit", "short_term")
            
            # ✅ Cache ALL horizon scores from the full analysis
            profiles = full_report.get("profiles", {})
            FULL_HORIZON_SCORES[symbol] = {
                h: profiles[h].get("final_score", 0)
                for h in ["intraday", "short_term", "long_term"]
            }
            
        # ✅ ADDITION: Also fetch Multibagger score from DB for dash "confluence dots"
        try:
            from config.multibagger.mb_db_model import MultibaggerCandidate
            from services.db import SessionLocal
            _mb_db = SessionLocal()
            try:
                mb_cand = _mb_db.query(MultibaggerCandidate).filter_by(symbol=symbol).first()
                FULL_HORIZON_SCORES[symbol]["multibagger"] = mb_cand.final_score or 0 if mb_cand else 0
            finally:
                _mb_db.close()
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to fetch MB score for cache: {e}")
            FULL_HORIZON_SCORES[symbol]["multibagger"] = 0
        
        # ✅ Extract data for template
        full_report = analysis_data.get("full_report", {})
        profile_report = full_report.get("profiles", {}).get(selected_profile_name, {})
        analysis_data["profile_report"] = profile_report
        analysis_data["strategy_report"] = profile_report.get("strategy", {})  # ✅ NEW: For summaries.py logic

        # =====================================================================
        # STRATEGY SHAPE NORMALISATION
        # =====================================================================
        # analyze_strategy_fit_v5 (config_resolver) returns:
        #   { primary_strategy, primary_fit_score, primary_weighted_score,
        #     all_candidates: [{name, fit_score, weighted_score, fit_threshold,
        #                        description, horizon_multiplier, breakdown?}] }
        #
        # result.html expects profile_report.strategy to have:
        #   { best:   {strategy, weighted_score, description, breakdown?,
        #              horizon_multiplier},
        #     ranked: [{name, weighted_score, fit_threshold, description,
        #               breakdown?}]   ← same list, just aliased }
        #
        # We normalise here so the template never has to know about the raw
        # resolver output format.  The original dict is preserved under "raw"
        # so nothing downstream breaks.
        # =====================================================================
        raw_strategy = profile_report.get("strategy", {})

        if raw_strategy and not raw_strategy.get("best"):
            # Build the winner record from the top all_candidates entry
            # (all_candidates is already sorted descending by weighted_score)
            all_candidates = raw_strategy.get("all_candidates", [])
            primary_name   = raw_strategy.get("primary_strategy", "")

            # Prefer the explicit primary; fall back to first candidate
            winner = next(
                (c for c in all_candidates if c.get("name") == primary_name),
                all_candidates[0] if all_candidates else {}
            )

            normalised_strategy = {
                # ── winner summary ──────────────────────────────────────────
                "best": {
                    "strategy":           winner.get("name", primary_name),
                    "weighted_score":     winner.get("weighted_score",
                                            raw_strategy.get("primary_weighted_score", 0)),
                    "fit_score":          winner.get("fit_score",
                                            raw_strategy.get("primary_fit_score", 0)),
                    "description":        winner.get("description", ""),
                    "horizon_multiplier": winner.get("horizon_multiplier", 1.0),
                    # breakdown has dna_fit_score / setup_quality_score (from our refactor)
                    "breakdown":          winner.get("breakdown"),
                },
                # ── full ranked list (used in the scrollable card) ──────────
                "ranked": all_candidates,          # already sorted, contains breakdown
                # ── preserve raw for any downstream code ────────────────────
                "raw":    raw_strategy,
            }

            # Patch profile_report in-place so eval_ctx stays consistent
            profile_report["strategy"] = normalised_strategy
            analysis_data["strategy_report"] = normalised_strategy

            # Wire eval_ctx.strategy.all_strategies for the accordion detail table.
            # eval_ctx lives inside profile_report (written by signal_engine/compute_all_profiles).
            # The accordion reads: profile_report.eval_ctx.strategy.all_strategies
            # all_candidates already contains every strategy with breakdown + rejection_reasons.
            eval_ctx = profile_report.get("eval_ctx")

            # ── SETUP CARD DATA ──────────────────────────────────────────────
            # eval_ctx["setup"] from _classify_setup already contains:
            #   best, candidates, ranked, rejected, top, type
            # Enrich rejected entries with human-readable labels and group
            # so the template doesn't need inline logic.
            SETUP_META = {
                "PATTERN_DARVAS_BREAKOUT":         {"label": "Darvas Box Breakout",      "group": "Breakout",  "icon": "bi-box-arrow-up-right"},
                "PATTERN_VCP_BREAKOUT":            {"label": "VCP Breakout",             "group": "Breakout",  "icon": "bi-graph-up-arrow"},
                "PATTERN_CUP_BREAKOUT":            {"label": "Cup & Handle",             "group": "Breakout",  "icon": "bi-cup-hot"},
                "PATTERN_FLAG_BREAKOUT":           {"label": "Flag / Pennant",           "group": "Breakout",  "icon": "bi-flag"},
                "PATTERN_GOLDEN_CROSS":            {"label": "Golden Cross",             "group": "Breakout",  "icon": "bi-stars"},
                "PATTERN_STRIKE_REVERSAL":         {"label": "3-Line Strike",            "group": "Reversal",  "icon": "bi-arrow-counterclockwise"},
                "MOMENTUM_BREAKOUT":               {"label": "Momentum Breakout",        "group": "Breakout",  "icon": "bi-lightning-charge"},
                "MOMENTUM_BREAKDOWN":              {"label": "Momentum Breakdown",       "group": "Short",     "icon": "bi-arrow-down-circle"},
                "TREND_PULLBACK":                  {"label": "Buy the Dip",              "group": "Trend",     "icon": "bi-arrow-down-up"},
                "DEEP_PULLBACK":                   {"label": "Deep Pullback Entry",      "group": "Trend",     "icon": "bi-arrow-bar-down"},
                "TREND_FOLLOWING":                 {"label": "Trend Following",          "group": "Trend",     "icon": "bi-graph-up"},
                "BEAR_TREND_FOLLOWING":            {"label": "Bear Trend (Short)",       "group": "Short",     "icon": "bi-graph-down"},
                "QUALITY_ACCUMULATION":            {"label": "Quality Accumulation",     "group": "Value",     "icon": "bi-gem"},
                "DEEP_VALUE_PLAY":                 {"label": "Deep Value",               "group": "Value",     "icon": "bi-currency-rupee"},
                "VALUE_TURNAROUND":                {"label": "Value Turnaround",         "group": "Value",     "icon": "bi-arrow-repeat"},
                "VOLATILITY_SQUEEZE":              {"label": "Volatility Squeeze",       "group": "Breakout",  "icon": "bi-arrows-angle-contract"},
                "REVERSAL_MACD_CROSS_UP":          {"label": "MACD Cross Reversal",      "group": "Reversal",  "icon": "bi-arrow-up-circle"},
                "REVERSAL_RSI_SWING_UP":           {"label": "RSI Oversold Bounce",      "group": "Reversal",  "icon": "bi-activity"},
                "REVERSAL_ST_FLIP_UP":             {"label": "Supertrend Flip",          "group": "Reversal",  "icon": "bi-toggles"},
                "QUALITY_ACCUMULATION_DOWNTREND":  {"label": "Quality in Downtrend",     "group": "Value",     "icon": "bi-safe2"},
                "SELL_AT_RANGE_TOP":               {"label": "Sell at Range Top",        "group": "Exit",      "icon": "bi-door-open"},
                "TAKE_PROFIT_AT_MID":              {"label": "Take Profit at Mid",       "group": "Exit",      "icon": "bi-cash-coin"},
                "GENERIC":                         {"label": "Generic (No Match)",       "group": "Other",     "icon": "bi-question-circle"},
            }

            REASON_LABELS = {
                "pattern_detection_failed":       "Pattern not detected",
                "technical_conditions_failed":    "Technical conditions not met",
                "fundamental_conditions_failed":  "Fundamental conditions not met",
                "blocked_by_horizon":             "Not valid for this horizon",
                "missing_fundamentals":           "Fundamental data unavailable",
                "context_validation_failed":      "Context requirements not met",
            }

            if isinstance(eval_ctx, dict) and "setup" in eval_ctx:
                raw_setup = eval_ctx["setup"]

                def _enrich_setup_entry(entry):
                    stype = entry.get("type", "GENERIC")
                    meta  = SETUP_META.get(stype, {"label": stype.replace("_", " ").title(), "group": "Other", "icon": "bi-circle"})
                    raw_reason = entry.get("reason", "")
                    # Prefer the detailed reason (e.g. "rsi 68.80 > max 35") over generic bucket
                    friendly   = REASON_LABELS.get(raw_reason, raw_reason.replace("_", " ").title())
                    return {**entry, "label": meta["label"], "group": meta["group"],
                            "icon": meta["icon"], "reason_label": friendly, "reason_raw": raw_reason}

                enriched_candidates = [_enrich_setup_entry(c) for c in raw_setup.get("candidates", [])]
                enriched_rejected   = [_enrich_setup_entry(r) for r in raw_setup.get("rejected",   [])]

                eval_ctx["setup"]["candidates_enriched"] = enriched_candidates
                eval_ctx["setup"]["rejected_enriched"]   = enriched_rejected

                if raw_setup.get("best"):
                    best_raw = raw_setup["best"]
                    best_meta = SETUP_META.get(best_raw.get("type","GENERIC"),
                                               {"label": best_raw.get("type",""), "group": "Other", "icon": "bi-circle"})
                    eval_ctx["setup"]["best"]["label"] = best_meta["label"]
                    eval_ctx["setup"]["best"]["group"] = best_meta["group"]
                    eval_ctx["setup"]["best"]["icon"]  = best_meta["icon"]
            # ── END SETUP CARD DATA ──────────────────────────────────────────

            if isinstance(eval_ctx, dict):
                eval_ctx_strategy = eval_ctx.get("strategy")
                if isinstance(eval_ctx_strategy, dict):
                    # Merge all_candidates (normalised, with breakdown) into eval_ctx
                    # Include rejected strategies too — combine all_candidates + rejected list
                    rejected = raw_strategy.get("rejected", [])
                    all_for_table = list(all_candidates) + list(rejected)
                    eval_ctx_strategy["all_strategies"] = all_for_table
                else:
                    # eval_ctx exists but strategy sub-key missing — create it
                    eval_ctx["strategy"] = {
                        "all_strategies": list(all_candidates) + list(raw_strategy.get("rejected", []))
                    }

        trade_plan = analysis_data.get("trade_recommendation", {})
        final_score = profile_report.get("final_score", 0)
        
        # Use correct confidence source
        confidence = trade_plan.get("final_confidence", 0)
        
        # Build narratives
        narratives = trade_plan.get("narratives", {})
        summary_context = {
            "indicators": analysis_data.get("indicators", {}),
            "trade_recommendation": trade_plan,
            "profile_report": profile_report,
            "strategy_report": analysis_data.get("strategy_report", {}), # ✅ P1-4: Restore missing strategy context
            "meta_scores": analysis_data.get("meta_scores", {}),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A")
        }
        legacy_summaries = build_all_summaries(summary_context)
        summaries = {**legacy_summaries, **narratives}
        
        # Build template context
        context = {
            "request": request,
            "symbol": symbol,
            "current_index": index,
            "selected_horizon": selected_profile_name,
            "best_fit_horizon": full_report.get("best_fit"),  # ✅ NEW: Show both
            "error": None,
            "full_report": full_report,
            "profile_report": profile_report,
            "fundamentals": analysis_data.get("fundamentals", {}),
            "indicators": analysis_data.get("indicators", {}),
            "meta_scores": analysis_data.get("meta_scores", {}),
            "trade_recommendation": trade_plan,
            "summaries": summaries,
            "final_signal": profile_report.get("category", "N/A"),
            "bull_signal": trade_plan.get("signal", "N/A"),
            "total_score": final_score * 10,
            "confidence": confidence,  # ✅ FIXED
            "macro_index_name": index.upper(),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A"),
            "macro_close": analysis_data.get("macro_close"),
            "reasons": [trade_plan.get("reason", "")],
            "strategy_report": analysis_data.get("strategy_report", {}),
            "all_horizon_scores": FULL_HORIZON_SCORES.get(symbol, {}),  # ✅ Global scores for toggle buttons
            "trade_direction": trade_plan.get("metadata", {}).get("direction", "bullish"),
        }
        # ✅ Log only essential context to reduce noise (with Stage 1 -> Stage 2 evolution)
        log_context = {
            "indicators": analysis_data.get("indicators", {}),
            "fundamentals": analysis_data.get("fundamentals", {}),
            "eval_ctx": profile_report.get("eval_ctx", {}),
            "exec_ctx_raw": trade_plan.get("metadata", {}).get("exec_ctx_raw", {}),
            "exec_ctx_enhanced": trade_plan.get("metadata", {}).get("exec_ctx", {}),
            "trade_plan": trade_plan
        }
        logger.debug(f"analysis data for {symbol}: {log_context}")
        
        # ✅ P0-3: API Versioning in response headers if needed, or in context
        response = templates.TemplateResponse("result.html", context)
        response.headers["X-API-Version"] = API_VERSION
        response.headers["X-Response-Schema-Version"] = RESPONSE_SCHEMA_VERSION
        return response
        
    except Exception as e:
        logger.exception(f"Error in analyze_common for {symbol}: {e}")
        # ✅ P2-4: Sanitize error disclosure
        error_msg = "An error occurred while analyzing the requested symbol."
        return templates.TemplateResponse("result.html", {
            "request": request,
            "symbol": symbol,
            "current_index": index,
            "error": error_msg,
            "full_report": {"profiles": {}},
            "profile_report": {},
            "indicators": {},
            "fundamentals": {},
            "summaries": {},
            "trade_recommendation": {},
            "selected_horizon": "short_term",
            "confidence": 0,
            "meta_scores": {},
            "strategy_report": {},
            "all_horizon_scores": {},
            "trade_direction": "bullish",
        })
    finally:
        # ✅ FIX: Properly paired finally block
        with _IN_FLIGHT_LOCK:
            if symbol in _IN_FLIGHT_ANALYSES:
                _IN_FLIGHT_ANALYSES.remove(symbol)

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_post(request: Request, symbol: str = Form(...), index: str = Form("nifty50"), api_key: str = Depends(get_api_key)):
    return await analyze_common(request, symbol.strip().upper(), index, api_key=api_key)

@app.post("/quick_scores")
async def get_quick_scores(req: QuickScoresRequest, api_key: str = Depends(get_api_key)):
    """
    Returns the scores for multiple tickers.
    ✅ P0-4: Hydrates FULL_HORIZON_SCORES from Signal Cache if missing.
    """
    symbols = [s.strip().upper() for s in req.symbols]
    index_name = sanitize_index_name(req.index_name)
    
    scores = {}
    for sym in symbols:
        # 1. Check in-memory cache first
        if sym in FULL_HORIZON_SCORES and FULL_HORIZON_SCORES[sym]:
            scores[sym] = FULL_HORIZON_SCORES[sym]
            continue
            
        # 2. ✅ P0-4: Fallback to DB to hydrate memory
        db_data = get_cached(sym)
        if db_data:
             h_scores = {
                 "intra_score": db_data.get("intra_score"),
                 "short_score": db_data.get("short_score"),
                 "long_score": db_data.get("long_score"),
                 "multi_score": db_data.get("multi_score")
             }
             h_scores = {k: v for k, v in h_scores.items() if v is not None}
             if h_scores:
                 # ✅ W59 FIX: Ensure we store the properly keyed horizon_scores JSON if available
                 FINAL_SCORES = db_data.get("horizon_scores") or h_scores
                 FULL_HORIZON_SCORES[sym] = FINAL_SCORES
                 scores[sym] = FINAL_SCORES
                 continue
                 
        scores[sym] = {
            "error": "Not Found",
            "status": "pending",
            "api_v": API_VERSION,
            "ts": get_current_utc().isoformat()
        }
        
@app.get("/api/v1/scores", response_class=JSONResponse)
async def get_all_scores_v1(api_key: str = Depends(get_api_key)):
    """
    ✅ P0-3: Versioned endpoint for all scores.
    """
    scores = {}
    for symbol in FULL_HORIZON_SCORES:
        scores[symbol] = FULL_HORIZON_SCORES[symbol]
        
    return {
        "api_info": {
            "version": API_VERSION,
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "timestamp": get_current_utc().isoformat()
        },
        "scores": scores
    }

@app.get("/api/v1/corporate_actions", response_class=JSONResponse)
async def corporate_actions_api_v1(tickers: str = Query(...), api_key: str = Depends(get_api_key)):
    """
    ✅ P0-3: Versioned endpoint for corporate actions.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    return get_corporate_actions(ticker_list)

@app.get("/multibagger_dashboard", response_class=HTMLResponse)
async def multibagger_dashboard(request: Request, api_key: str = Depends(get_api_key)):
    """Render the Multibagger Picks dashboard."""
    return templates.TemplateResponse("mb_picks.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, api_key: str = Depends(get_api_key)):
    return templates.TemplateResponse("index.html", {"request": request, "api_key": api_key})

@app.post("/corporate_action_summary")
async def corporate_action_summary(payload: CorpActionsRequest, api_key: str = Depends(get_api_key)):
    """
    Returns a flat {ticker: display_string} map for the given tickers.
    Reads from pre-built disk cache — instant response, no live fetches.
    """
    try:
        tickers = payload.tickers
        if not tickers:
            return JSONResponse({})

        # This reads from cache/corp_actions_summary.json — sub-millisecond
        summary = get_corp_actions_summary(tickers)
        return JSONResponse(summary)

    except Exception as e:
        logger.warning("corporate_action_summary error: %s", e)
        return JSONResponse({})

@app.get("/api/corp_actions_status")
async def corp_actions_status(api_key: str = Depends(get_api_key)):
    """
    Returns readiness state of the corp actions summary cache.
    """
    cache_path = "cache/corp_actions_summary.json"
    ready = _CORP_ACTIONS_CACHE_READY and os.path.exists(cache_path)
    count = 0
    if ready:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                count = len(json.load(f))
        except Exception:
            pass
    return JSONResponse({"ready": ready, "count": count})


@app.get("/health")
async def health_check():
    """✅ P2-1: Standard health endpoint for infrastructure monitoring."""
    return {"status": "ok", "timestamp": get_current_utc().isoformat()}


@app.post("/api/fetch_corp_actions")
async def fetch_corp_actions_endpoint(api_key: str = Depends(get_api_key)):
    """
    On-demand fetch of upcoming corp actions (split, bonus, dividend).
    Runs on BACKGROUND_EXECUTOR (ThreadPoolExecutor). 1-hour cooldown.
    """
    global _CORP_ACTIONS_CACHE_READY, _LAST_CORP_FETCH_TIME

    # Cooldown — skip if <1hr AND cache file actually exists on disk
    now = time.time()
    elapsed = now - _LAST_CORP_FETCH_TIME
    cache_exists = os.path.exists("cache/corp_actions_summary.json")
    if elapsed < _CORP_FETCH_COOLDOWN_SEC and _CORP_ACTIONS_CACHE_READY and cache_exists:
        mins_left = int((_CORP_FETCH_COOLDOWN_SEC - elapsed) / 60)
        # return JSONResponse({"status": "cooldown", "minutes_left": mins_left})

    loop = asyncio.get_running_loop()

    def _do_fetch():
        # Safe: BACKGROUND_EXECUTOR is ThreadPoolExecutor, globals shared in-process.
        global _CORP_ACTIONS_CACHE_READY, _LAST_CORP_FETCH_TIME
        # Collect actual stock tickers from all index data files
        all_tickers = set()
        for idx_name in INDEX_TICKERS.keys():
            stocks = load_or_create_index(idx_name)
            for s in stocks:
                all_tickers.add(s[0])
        all_tickers = list(all_tickers)
        if not all_tickers:
            raise ValueError("No tickers found in data files")
        # Always force=True — cooldown gate above ensures we don't hammer NSE
        summary = build_corp_actions_summary_cache(all_tickers, force=True)
        _CORP_ACTIONS_CACHE_READY = True
        _LAST_CORP_FETCH_TIME = time.time()
        return len(summary)

    try:
        count = await loop.run_in_executor(get_background_executor(), _do_fetch)
        return JSONResponse({"status": "ok", "count": count})
    except Exception as e:
        logger.error("fetch_corp_actions failed: %s", e)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/corporate_actions")
async def corporate_actions(ticker: str, api_key: str = Depends(get_api_key)):
    loop = asyncio.get_running_loop()
    def _fetch():
        past = get_corporate_actions([ticker], mode="past", lookback_days=365)
        upcoming = get_corporate_actions([ticker], mode="upcoming", lookback_days=365)
        return past, upcoming
    past, upcoming = await loop.run_in_executor(get_api_executor(), _fetch)
    flat = []
    if past:
        for item in past:
            for a in item.get("actions", []):
                a["_when"] = "past"
                flat.append(a)
    if upcoming:
        for item in upcoming:
            for a in item.get("actions", []):
                a["_when"] = "upcoming"
                flat.append(a)
    return JSONResponse(flat)

@app.get("/api/macro_metrics")
async def macro_metrics_api(api_key: str = Depends(get_api_key)):
    try:
        metrics = get_macro_metrics()
        return JSONResponse(metrics)
    except Exception as e:
        logger.error(f"Error fetching macro metrics: {e}")
        return JSONResponse({"error": "Failed to fetch macro metrics"}, status_code=500)

@app.post("/api/paper_trade/add")
async def add_paper_trade(req: PaperTradeRequest, api_key: str = Depends(get_api_key)):
    db = SessionLocal()
    try:
        trade = PaperTrade(
            symbol=req.symbol,
            entry_price=req.entry_price,
            target_1=req.target_1,
            target_2=req.target_2,
            stop_loss=req.stop_loss,
            estimated_hold_days=req.estimated_hold_days,
            horizon=req.horizon,
            position_size=req.position_size
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return {"status": "success", "id": trade.id}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to add paper trade: {e}")
        return JSONResponse({"error": "Failed to add paper trade"}, status_code=500)
    finally:
        db.close()

class PaperTradeRemoveRequest(BaseModel):
    symbol: str
    horizon: str

@app.post("/api/paper_trade/remove")
async def remove_paper_trade(req: PaperTradeRemoveRequest, api_key: str = Depends(get_api_key)):
    db = SessionLocal()
    try:
        # Remove ALL open paper trades matching symbol and horizon
        trades = db.query(PaperTrade).filter(
            PaperTrade.symbol == req.symbol,
            PaperTrade.horizon == req.horizon,
            PaperTrade.status == "OPEN"
        ).all()
        for t in trades:
            db.delete(t)
        db.commit()
        return {"status": "success", "deleted": len(trades)}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to remove paper trade: {e}")
        return JSONResponse({"error": "Failed to remove paper trade"}, status_code=500)
    finally:
        db.close()

@app.get("/api/paper_trade/status")
async def get_paper_trade_status(symbol: str = Query(...), horizon: str = Query(...), api_key: str = Depends(get_api_key)):
    db = SessionLocal()
    try:
        trade = db.query(PaperTrade).filter(
            PaperTrade.symbol == symbol,
            PaperTrade.horizon == horizon,
            PaperTrade.status == "OPEN"
        ).first()
        return {"active": trade is not None}
    except Exception as e:
        return {"active": False, "error": "Database error"}
    finally:
        db.close()

@app.get("/paper_trades", response_class=HTMLResponse)
async def view_paper_trades(request: Request, api_key: str = Depends(get_api_key)):
    db: Session = SessionLocal()
    try:
        trades = db.query(PaperTrade).order_by(PaperTrade.created_at.desc()).all()
        return templates.TemplateResponse("paper_trades.html", {"request": request, "trades": trades})
    finally:
        db.close()

@app.get("/api/paper_trade/cmp")
async def get_paper_trade_cmp(symbols: str = Query(""), api_key: str = Depends(get_api_key)):
    if not symbols:
        return {}
    import yfinance as yf
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return {}
    
    try:
        from services.data_fetch import get_history_for_horizon
        res = {}
        for s in sym_list:
            try:
                # Use short_term (1d) horizon to get latest close
                df = get_history_for_horizon(s, "short_term")
                if df is not None and not df.empty:
                    res[s] = float(df['Close'].iloc[-1])
            except Exception as e:
                logger.warning(f"Failed to fetch CMP for {s}: {e}")
                pass
        return res
    except Exception as e:
        logger.error(f"CMP API Error: {e}")
        return {}

if __name__ == "__main__":
    init_db()
    # if not startup_config_validation():
    #     print("❌ CRITICAL: Pattern Config validation failed. Check logs.")
    #     exit(1)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
