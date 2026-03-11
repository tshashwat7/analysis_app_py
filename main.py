# main.py
import asyncio
import os
import json
import time
import datetime
import math
import threading
import concurrent.futures
import pytz
from typing import Dict, Any, List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from config.config_utility.logger_config import setup_logger
from services.indicator_cache import compute_indicators_cached
logger = setup_logger()

# --- Modular services ---
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
from services.db import SessionLocal, SignalCache, init_db, PaperTrade

class PaperTradeRequest(BaseModel):
    symbol: str
    entry_price: float
    target_1: float = None
    target_2: float = None
    stop_loss: float = None
    estimated_hold_days: int = None
    horizon: str = None
    position_size: int = None

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
IST = pytz.timezone("Asia/Kolkata")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
_SHUTDOWN_IN_PROGRESS = False
_SHUTDOWN_LOCK = threading.Lock()

# Lazy Loaded Stock Lists
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
def _ensure_stocks_loaded():
    global ALL_STOCKS_LIST, ALL_STOCKS_MAP
    if ALL_STOCKS_LIST is not None:
        return
    
    # Load logic
    ALL_STOCKS_LIST = load_or_create_global_stocks()
    if not ALL_STOCKS_LIST:
        ALL_STOCKS_LIST = get_cached_stocks(NSE_STOCKS_FILE)
    ALL_STOCKS_MAP = {s[0]: s[1] for s in ALL_STOCKS_LIST}
    logger.info(f"Loaded {len(ALL_STOCKS_LIST)} stocks into memory.")

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
                # Kick off analysis of the batch using the compute executor
                futures = [
                    loop.run_in_executor(get_compute_executor(), run_full_analysis, symbol)
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
                            "[WARMER] Warmed %s (result type=%s)",
                            sym,
                            type(res).__name__,
                        )

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
    _SHUTDOWN_IN_PROGRESS = False
    
    # Pre-load data in main process
    build_smart_index_map()
    _ensure_stocks_loaded()
    
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
                build_corp_actions_summary_cache(all_tickers)
                _CORP_ACTIONS_CACHE_READY = True
                _LAST_CORP_FETCH_TIME = time.time()
                logger.info("[STARTUP] Corp actions summary cache ready.")
            except Exception as e:
                logger.warning("[STARTUP] Corp actions cache build failed: %s", e)
        
        threading.Thread(target=_warm_corp_actions_cache, daemon=True, name="CorpActionsWarmer").start()
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
CACHE_LOCK = threading.Lock()
STOCK_TO_INDEX_MAP: Dict[str, str] = {}

# --- STOCK LOADING HELPERS ---

def load_or_create_index(index_name: str):
    json_file = os.path.join(DATA_DIR, f"{index_name}.json")
    csv_file = os.path.join(DATA_DIR, f"{index_name}.csv")

    # If JSON exists, load it
    if os.path.exists(json_file):
        stocks = get_cached_stocks(json_file)
        if stocks: return stocks
    # If JSON missing but CSV exists → build JSON
    if os.path.exists(csv_file):
        logger.info(f"Parsing CSV for index: {index_name}")
        pairs = parse_index_csv(csv_file)
        if pairs:
            json_data = [{"symbol": s, "name": n} for s, n in pairs]
            try:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)
            except Exception: pass
            return pairs
    return []

def load_or_create_global_stocks():
    """
    Loads NSEStock.json (global stock universe).
    If missing/corrupted -> rebuild from NSEStock.csv.
    """
    return load_or_create_index("NSEStock")

def build_smart_index_map():
    global STOCK_TO_INDEX_MAP
    priority_files = ["NSEStock", "niftyauto", "niftybank", "niftyfmcg", "niftyinfra", 
                      "niftyit", "niftypharma", "niftyrealty", "nifty500", "smallcap250", 
                      "microcap250", "smallcap100", "midcap150", "niftynext50", "nifty100", "nifty50"]
    for filename in priority_files:
        # Ensure the file actually exists before trying to load
        filepath = os.path.join(DATA_DIR, f"{filename}.json")
        if os.path.exists(filepath):
            stocks = get_cached_stocks(filepath)
            for symbol, _ in stocks:
                STOCK_TO_INDEX_MAP[symbol.strip().upper()] = filename
    logger.info(f"[INIT] Smart Index Map built for {len(STOCK_TO_INDEX_MAP)} symbols.")

def get_cached_stocks(index_file: str) -> List[Tuple[str, str]]:
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                out = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "symbol" in item:
                            out.append((item["symbol"], item.get("name", item["symbol"])))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            out.append((item[0], item[1]))
                return out
        except Exception:
            return []
    return []

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
            "recommendation": entry.recommendation,
            "confidence": entry.conf_score,
            "bull_signal": entry.signal_text,
            "rr_ratio": entry.rr_ratio,
            "entry_trigger": entry.entry_price,
            "stop_loss": entry.stop_loss,
            "cached": True,
        }

        # 2. Merge the flexible JSON fields (horizon scores, errors, macros)
        if entry.horizon_scores:
            flat_data.update(entry.horizon_scores)
        return flat_data
    except Exception as e:
        logger.error(f"DB Read Error {symbol}: {e}")
        return None
    finally:
        db.close()

def set_cached(symbol: str, value: Dict[str, Any]):
    """
    Saves analysis to SQLite (Persistent).
    Maps the 'flat' grid dictionary into the structured DB columns.
    """
    db: Session = SessionLocal()
    try:
        with CACHE_LOCK:  # Keep Lock to prevent SQLite race conditions
            # Check if exists
            entry = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()
            
            horizon_data = {
                "intra_score": value.get("intra_score"),
                "short_score": value.get("short_score"),
                "long_score": value.get("long_score"),
                "multi_score": value.get("multi_score"),
                "sl_dist": value.get("sl_dist"),
                "macro_index_name": value.get("macro_index_name"),
                "error_details": value.get("error_details"),
                # ✅ NEW: persist so get_cached returns them on cache hit
                "t1": value.get("t1"),
                "t2": value.get("t2"),
                "profit_pct": value.get("profit_pct"),
                "setup_signal": value.get("setup_signal"),
                "best_fit_horizon": value.get("best_fit_horizon"),
                "best_strategy": value.get("best_strategy"),
                "fitting_strategies": value.get("fitting_strategies"),
                "top_pattern": value.get("top_pattern"),
                "bull_score": value.get("bull_score"),
                "profile_category": value.get("profile_category"),
            }

            # Helper to safely get float
            def _f(k): return value.get(k) if isinstance(value.get(k), (int, float)) else 0.0

            if not entry:
                entry = SignalCache(
                    symbol=symbol,
                    score=_f("score"),
                    recommendation=str(value.get("recommendation", "HOLD")),
                    best_horizon=str(value.get("recommendation", "")).split("--")[-1],
                    signal_text=str(value.get("bull_signal", "")),
                    conf_score=int(_f("confidence")),
                    rr_ratio=_f("rr_ratio"),
                    entry_price=_f("entry_trigger"),
                    stop_loss=_f("stop_loss"),
                    horizon_scores=horizon_data,
                    # Auto-updated by DB default, but explicit doesn't hurt
                    updated_at=get_current_utc(),
                )
                db.add(entry)
            else:
                # Update existing
                entry.score = _f("score")
                entry.recommendation = str(value.get("recommendation", "HOLD"))
                entry.signal_text = str(value.get("bull_signal", ""))
                entry.conf_score = int(_f("confidence"))
                entry.rr_ratio = _f("rr_ratio")
                entry.entry_price = _f("entry_trigger")
                entry.horizon_scores = horizon_data
                entry.updated_at = get_current_utc()
            db.commit()
    except Exception as e:
        logger.error(f"DB Write Error {symbol}: {e}")
        db.rollback()
    finally:
        db.close()

# --- WORKER ---
def run_analysis(
    symbol: str, 
    index_name: str = "nifty50",
    mode: str = "full",           # ✅ NEW: "full" or "single"
    requested_horizon: str = None # ✅ NEW: Required if mode="single"
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
    db = None
    enrichment_info = None
    
    try:
        global STOCK_TO_INDEX_MAP
        if not STOCK_TO_INDEX_MAP:
            build_smart_index_map()
        
        db = SessionLocal()
        
        # =====================================================================
        # DETERMINE HORIZONS TO COMPUTE
        # =====================================================================
        if mode == "single":
            if not requested_horizon:
                raise ValueError("requested_horizon required for mode='single'")
            horizons_to_compute = [requested_horizon]
            logger.info(f"[{symbol}] 🎯 SINGLE-HORIZON MODE | Horizon: {requested_horizon}")
        else:
            horizons_to_compute = ["intraday", "short_term", "long_term", "multibagger"]
            logger.info(f"[{symbol}] 🔄 FULL-HORIZON MODE | Computing all 4")
        
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
            for fallback_horizon in ["short_term", "intraday", "long_term", "multibagger"]:
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
        if not (analysis_data["fundamentals"] and analysis_data["raw_indicators_by_horizon"]):
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
            # Get indicators from any available horizon for momentum
            momentum_horizon = requested_horizon if mode == "single" else "short_term"
            momentum_inds = analysis_data["raw_indicators_by_horizon"].get(
                momentum_horizon,
                analysis_data["raw_indicators_by_horizon"].get("intraday", {})
            )
            
            analysis_data["meta_scores"] = {
                # Value, Growth, Quality are ALWAYS long-term metrics
                "value": score_value_profile(
                    analysis_data["fundamentals"], 
                    horizon="multibagger"  # ✅ FIXED: Always use long-term
                ),
                "growth": score_growth_profile(
                    analysis_data["fundamentals"], 
                    horizon="long_term"    # ✅ FIXED: Always use long-term
                ),
                "quality": score_quality_profile(
                    analysis_data["fundamentals"], 
                    horizon="long_term"    # ✅ FIXED: Always use long-term
                ),
                # Momentum adapts to context
                "momentum": score_momentum_profile(
                    analysis_data["fundamentals"],
                    momentum_inds,
                    horizon=momentum_horizon  # ✅ Use appropriate horizon
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
        if "full_report" in analysis_data and analysis_data["trade_recommendation"].get("status") != "ERROR":
            _save_analysis_to_db(
                db, 
                symbol, 
                analysis_data, 
                index_name,
                best_fit_horizon=best_fit,          # ✅ NEW: System optimal
                selected_horizon=display_horizon     # ✅ NEW: User choice
            )
        
        # =====================================================================
        # STEP 10: ENRICHMENT METADATA
        # =====================================================================
        if enrichment_info:
            analysis_data["_enrichment"] = enrichment_info
        
        return analysis_data
        
    except Exception as e:
        logger.error(f"[{symbol}] Analysis Error: {e}", exc_info=True)
        return {"symbol": symbol, "error": str(e)}
    
    finally:
        if db:
            db.close()


# ✅ BACKWARD COMPATIBLE WRAPPERS
def run_full_analysis(symbol: str, index_name: str = "nifty50") -> Dict[str, Any]:
    """Legacy wrapper for full analysis."""
    return run_analysis(symbol, index_name, mode="full")


def run_single_horizon_analysis(symbol: str, horizon: str, index_name: str = "nifty50") -> Dict[str, Any]:
    """Optimized single-horizon analysis."""
    return run_analysis(symbol, index_name, mode="single", requested_horizon=horizon)

# =========================================================================
# DATABASE PERSISTENCE (Aligned with New Trade Plan Structure)
# =========================================================================
def _save_analysis_to_db(
    db: Session,
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
        
        # Build horizon scores JSON
        horizon_data = {
            "intra_score": profiles.get("intraday", {}).get("final_score"),
            "short_score": profiles.get("short_term", {}).get("final_score"),
            "long_score": profiles.get("long_term", {}).get("final_score"),
            "multi_score": profiles.get("multibagger", {}).get("final_score"),
            "macro_index_name": index_name.upper(),
            "error_details": "; ".join(data.get("error_details", [])),
            "best_fit_score": best_fit_score,  # ✅ NEW: For comparison
        }
        
        # Calculate SL distance
        entry_val = trade_plan.get("entry")
        current_price = indicators.get("price", {}).get("value") or entry_val
        sl_val = trade_plan.get("stop_loss")
        
        sl_dist_str = "-"
        if current_price and sl_val:
            dist = abs(current_price - sl_val) / current_price * 100
            sl_dist_str = f"{dist:.1f}%"
        horizon_data["sl_dist"] = sl_dist_str
        
        # Upsert database entry
        entry_row = db.query(SignalCache).filter(
            SignalCache.symbol == symbol
        ).first()
        
        if not entry_row:
            entry_row = SignalCache(symbol=symbol)
            db.add(entry_row)
        
        # Save BOTH horizons
        entry_row.best_horizon = best_fit_horizon        # System optimal
        entry_row.selected_horizon = selected_horizon    # User choice
        
        # Trade data is from SELECTED horizon
        entry_row.score = int(final_score * 10) if final_score else 0
        entry_row.recommendation = (
            selected_profile.get("category", "HOLD") + 
            "--" + 
            selected_horizon
        )
        entry_row.signal_text = trade_plan.get("trade_signal", "N/A")
        entry_row.conf_score = trade_plan.get("final_confidence", 0)  # ✅ FIXED
        entry_row.rr_ratio = trade_plan.get("rr_ratio")
        entry_row.entry_price = entry_val
        entry_row.stop_loss = sl_val
        entry_row.horizon_scores = horizon_data
        entry_row.updated_at = get_current_utc()
        
        db.commit()
        
        logger.debug(
            f"[{symbol}] Saved to DB | "
            f"Best: {best_fit_horizon} | "
            f"Selected: {selected_horizon} | "
            f"Score: {entry_row.score}"
        )
        
    except Exception as e:
        logger.error(f"[{symbol}] DB Save Failed: {e}")
        db.rollback()

# --- ENDPOINTS ---

@app.post("/quick_scores")
async def quick_scores(payload: QuickScoresRequest):
    # Lazy Load Stock List
    _ensure_stocks_loaded()
    
    symbols = [s.strip().upper() for s in payload.symbols if s.strip()]
    index_name = payload.index_name
    results = {}
    to_fetch = []

    for s in symbols:
        c = get_cached(s)
        if c: results[s] = {**c, "cached": True}
        else: to_fetch.append(s)

    if not to_fetch: return results

    loop = asyncio.get_running_loop()
    calc_tasks = [loop.run_in_executor(get_compute_executor(), run_full_analysis, s, index_name) for s in to_fetch]
    analyzed_data_list = await asyncio.gather(*calc_tasks)

    for analysis_data in analyzed_data_list:
        sym = analysis_data.get("symbol")
        if not sym: continue
        # Extract & Strip Enrichment Data
        enrich_payload = analysis_data.pop("_enrichment", None)
        
        # Fire & Forget Background Task
        if enrich_payload:
            idx, s, n = enrich_payload
            # Runs sync function in thread pool. No await needed.
            loop.run_in_executor(get_background_executor(), enrich_json_sync, idx, s, n)
        try:
            full_rep = analysis_data.get("full_report", {})
            profiles = full_rep.get("profiles", {})
            best_profile = profiles.get(full_rep.get("best_fit", "short_term"), {})
            trade_plan = analysis_data.get("trade_recommendation", {})
            indicators = analysis_data.get("indicators", {})
            final_score = best_profile.get("final_score", 0)
            confidence = trade_plan.get("final_confidence", 0)
            signal_str = trade_plan.get("signal", "N/A")
            error_status = analysis_data.get("error_details", [])
            
            bull_score = 0
            if "SQUEEZE" in signal_str: bull_score = 95
            elif "TREND" in signal_str: bull_score = 85
            elif "DIP" in signal_str: bull_score = 75
            elif "HOLD" in signal_str: bull_score = 50
            else: bull_score = 20

            rr = trade_plan.get("rr_ratio")
            entry_val = trade_plan.get("entry")
            sl_val = trade_plan.get("stop_loss")
            current_price = indicators.get("price", {}).get("value") or entry_val
            sl_dist_str = "-"
            if current_price and sl_val and current_price > 0:
                dist = abs(current_price - sl_val) / current_price * 100
                sl_dist_str = f"{dist:.1f}%"

            # ✅ FIX: strategy_report = profile_report.strategy
            # Structure: { "best": {strategy, weighted_score,...}, "primary": str,
            #              "ranked": [{name, weighted_score, fit_threshold,...},...],
            #              "summary": {total, qualified, best_strategy, rejected} }
            strat_data = analysis_data.get("strategy_report", {})
            strat_summ = strat_data.get("summary", {})
            best_strat = (
                strat_summ.get("best_strategy")
                or (strat_data.get("best") or {}).get("strategy")
                or strat_data.get("primary")
                or "N/A"
            )
            # ✅ FIX: "all_fits" key never existed in summary — read from ranked list
            ranked = strat_data.get("ranked", [])
            all_fits = ", ".join(
                c["name"] for c in ranked
                if c.get("weighted_score", 0) >= c.get("fit_threshold", 50)
            )

            pattern_keys = ["goldenCross", "doubleTopBottom", "cupHandle", "darvasBox", 
                            "flagPennant", "minerviniStage2", "bollingerSqueeze", "threeLineStrike", "ichimokuSignals"]
            top_pattern_name = ""
            top_pattern_score = 0
            for pk in pattern_keys:
                p_obj = indicators.get(pk)
                if p_obj and isinstance(p_obj, dict) and p_obj.get("found"):
                    s = p_obj.get("score", 0)
                    if s > top_pattern_score:
                        top_pattern_score = s
                        top_pattern_name = pk.replace("_", " ").title()

            name_map = {"Bollinger Squeeze": "Squeeze", "Minervini Stage2": "VCP", "Three Line Strike": "3-Strike", 
                        "Golden Cross": "Gold Cross", "Double Top Bottom": "Double T/B", "Cup Handle": "Cup", 
                        "Darvas Box": "Darvas", "Flag Pennant": "Flag", "Ichimoku Signals": "Ichimoku"}
            top_pattern_name = name_map.get(top_pattern_name, top_pattern_name)

            # ✅ FIX: derive clean BUY/SELL/HOLD from trade_signal (not the raw setup string)
            # trade_plan["trade_signal"] = "BUY"/"SELL"/"HOLD"
            # trade_plan["signal"]       = "BOLLINGER_SQUEEZE_BUY" (full setup label)
            trade_signal_clean = trade_plan.get("trade_signal") or (
                "BUY"  if any(k in signal_str for k in ("BUY", "SQUEEZE", "TREND", "DIP"))
                else "SELL" if any(k in signal_str for k in ("SELL", "SHORT"))
                else "HOLD"
            )

            # ✅ NEW: profit to T1 = (t1 - entry) / entry * 100
            t1_val = (trade_plan.get("targets") or {}).get("t1")
            t2_val = (trade_plan.get("targets") or {}).get("t2")
            profit_pct = None
            if entry_val and t1_val and entry_val > 0:
                profit_pct = round((t1_val - entry_val) / entry_val * 100, 2)

            def safe_val(v):
                if v is None: return None
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
                return v

            flat_output = {
                "symbol": sym,
                "score": int(final_score * 10) if final_score else 0,
                "confidence": confidence,
                "recommendation": (best_profile.get("profile_signal", best_profile.get("category", "HOLD")) + "--" + full_rep.get("best_fit", "")),
                "bull_score": bull_score,
                "bull_signal": trade_signal_clean,
                "profile_category": best_profile.get("profile_signal", best_profile.get("category", "HOLD")),
                "setup_signal": signal_str.replace("_", " "), # ✅ NEW: full label e.g. "BOLLINGER SQUEEZE BUY"
                "rr_ratio": rr if rr else 0,
                "entry_trigger": entry_val if entry_val else 0,
                "t1": safe_val(t1_val),                       # ✅ NEW
                "t2": safe_val(t2_val),                       # ✅ NEW
                "profit_pct": safe_val(profit_pct),           # ✅ NEW: (T1-entry)/entry*100
                "sl_dist": sl_dist_str,
                "best_strategy": best_strat,
                "fitting_strategies": all_fits,
                "best_fit_horizon": full_rep.get("best_fit", ""),  # ✅ NEW
                "intra_score": safe_val(profiles.get("intraday", {}).get("final_score")),
                "short_score": safe_val(profiles.get("short_term", {}).get("final_score")),
                "long_score": safe_val(profiles.get("long_term", {}).get("final_score")),
                "multi_score": safe_val(profiles.get("multibagger", {}).get("final_score")),
                "macro_index_name": index_name.upper(),
                "top_pattern": top_pattern_name,
                "error_details": "; ".join(error_status) if error_status else "",
            }
            results[sym] = flat_output
        except Exception as e:
            logger.error(f"[{sym}] Error flattening: {e}")
            results[sym] = {"symbol": sym, "recommendation": "Error"}
    return results

@app.get("/load_index/{index_name}")
async def load_index_endpoint(index_name: str):
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

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_common(
    request: Request,
    symbol: str,
    index: str = "nifty50",
    horizon: str = None
):
    """
    ✅ FULLY REVISED: Proper separation of concerns with smart routing.
    
    Routes:
    - horizon=None     → mode="full" (compute all 4, pick best)
    - horizon=specific → mode="single" (compute only 1)
    """
    _ensure_stocks_loaded()
    
    try:
        loop = asyncio.get_running_loop()
        
        # ✅ SMART ROUTING: Choose mode based on horizon parameter
        if horizon and horizon in ["intraday", "short_term", "long_term", "multibagger"]:
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
            
            # ✅ Cache ALL 4 horizon scores from the full analysis
            profiles = full_report.get("profiles", {})
            FULL_HORIZON_SCORES[symbol] = {
                h: profiles[h].get("final_score", 0)
                for h in ["intraday", "short_term", "long_term", "multibagger"]
                if h in profiles
            }
        
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
        }
        logger.debug(f"analysis data for {symbol}: {analysis_data}")
        return templates.TemplateResponse("result.html", context)
        
    except Exception as e:
        logger.exception(f"Error in analyze_common for {symbol}: {e}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "symbol": symbol,
            "current_index": index,
            "error": str(e),
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
        })
    
async def analyze_common_old(request: Request, symbol: str, index: str = "nifty50", horizon: str = None):
    # Lazy Load
    _ensure_stocks_loaded()
    try:
        loop = asyncio.get_running_loop()
        analysis_data = await loop.run_in_executor(get_compute_executor(), run_full_analysis, symbol, index)
        full_report = analysis_data.get("full_report", {})
        selected_profile_name = horizon if (horizon and horizon in full_report.get("profiles", {})) else full_report.get("best_fit", "short_term")
        profile_report = full_report.get("profiles", {}).get(selected_profile_name, {})
        analysis_data["profile_report"] = profile_report
        analysis_data["strategy_report"] = profile_report.get("strategy", {})  # ✅ NEW: For summaries.py logic
        
        current_horizon_inds = analysis_data.get("raw_indicators_by_horizon", {}).get(selected_profile_name)
        if not current_horizon_inds: current_horizon_inds = analysis_data.get("indicators", {})

        # ... (Prepare standard summaries/scores variables) ...
        best_fit = full_report.get("best_fit", "short_term")

        if selected_profile_name != best_fit:
            trade_plan = generate_trade_plan(
                symbol,
                profile_report,
                current_horizon_inds,
                analysis_data.get("fundamentals", {}),
                selected_profile_name,
                analysis_data.get("macro_trend_status", "N/A"),
                )
        else:
            trade_plan = analysis_data.get("trade_recommendation", {})

        final_score = profile_report.get("final_score", 0)
        meta_scores = analysis_data.get("meta_scores", {})
        
        # ✅ ENHANCED: Get narratives from trade_plan (generated in generate_trade_plan)
        narratives = trade_plan.get("narratives", {})

        # Build legacy summaries for backward compatibility
        summary_context = {
            "indicators": analysis_data.get("indicators", {}),
            "trade_recommendation": trade_plan,
            "profile_report": profile_report,
            "meta_scores": analysis_data.get("meta_scores", {}),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A")
        }
        legacy_summaries = build_all_summaries(summary_context)

        # Merge: narratives + legacy summaries
        summaries = {**legacy_summaries, **narratives}

        if narratives:
            logger.info(f"[{symbol}] Enhanced narratives available: {list(narratives.keys())}")
        else:
            logger.warning(f"[{symbol}] No enhanced narratives in trade_plan")

        context = {
            "request": request, "symbol": symbol, "current_index": index, "selected_horizon": selected_profile_name,
            "error": None, "full_report": full_report, "profile_report": profile_report,
            "fundamentals": analysis_data.get("fundamentals", {}), "indicators": analysis_data.get("indicators", {}),
            "meta_scores": {"value": meta_scores.get("value", 0) or 0, "growth": meta_scores.get("growth", 0) or 0,
                            "quality": meta_scores.get("quality", 0) or 0, "momentum": meta_scores.get("momentum", 0) or 0},
            "trade_recommendation": trade_plan, "summaries": summaries,
            "final_signal": profile_report.get("category", "HOLD"), "bull_signal": trade_plan.get("signal", "N/A"),
            "total_score": final_score * 10, "confidence": trade_plan.get("final_confidence", 0), "macro_index_name": index.upper(),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A"), "macro_close": analysis_data.get("macro_close"),
            "reasons": [trade_plan.get("reason", "")], "strategy_report": analysis_data.get("strategy_report", {}),
        }
        logger.debug(f" context passed to result.html for {symbol} is {context}")#//////////////////////////////////////////////////////////////////////////////////
        return templates.TemplateResponse("result.html", context)
    except Exception as e:
        logger.exception(f"Error in analyze_common for {symbol}: {e}")
        return templates.TemplateResponse("result.html", {
            "request": request, "symbol": symbol, "current_index": index, "error": str(e),
            "full_report": {"profiles": {}}, "profile_report": {}, "indicators": {}, "fundamentals": {},
            "summaries": {}, "trade_recommendation": {}, "selected_horizon": "short_term", "confidence": 0,
            "meta_scores": {}, "strategy_report": {},
        })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_post(request: Request, symbol: str = Form(...), index: str = Form("nifty50")):
    return await analyze_common(request, symbol.strip().upper(), index)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/corporate_action_summary")
async def corporate_action_summary(payload: CorpActionsRequest):
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
async def corp_actions_status():
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


@app.post("/api/fetch_corp_actions")
async def fetch_corp_actions_endpoint():
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
async def corporate_actions(ticker: str):
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
async def macro_metrics_api():
    try:
        metrics = get_macro_metrics()
        return JSONResponse(metrics)
    except Exception as e:
        logger.error(f"Error fetching macro metrics: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/paper_trade/add")
async def add_paper_trade(req: PaperTradeRequest):
    try:
        db = SessionLocal()
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
        return {"status": "success", "id": trade.id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

class PaperTradeRemoveRequest(BaseModel):
    symbol: str
    horizon: str

@app.post("/api/paper_trade/remove")
async def remove_paper_trade(req: PaperTradeRemoveRequest):
    try:
        db = SessionLocal()
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
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        db.close()

@app.get("/api/paper_trade/status")
async def get_paper_trade_status(symbol: str = Query(...), horizon: str = Query(...)):
    try:
        db = SessionLocal()
        trade = db.query(PaperTrade).filter(
            PaperTrade.symbol == symbol,
            PaperTrade.horizon == horizon,
            PaperTrade.status == "OPEN"
        ).first()
        return {"active": trade is not None}
    except Exception as e:
        return {"active": False, "error": str(e)}
    finally:
        db.close()

@app.get("/paper_trades", response_class=HTMLResponse)
async def view_paper_trades(request: Request):
    db: Session = SessionLocal()
    trades = db.query(PaperTrade).order_by(PaperTrade.created_at.desc()).all()
    db.close()
    return templates.TemplateResponse("paper_trades.html", {"request": request, "trades": trades})

@app.get("/api/paper_trade/cmp")
async def get_paper_trade_cmp(symbols: str = Query("")):
    if not symbols:
        return {}
    import yfinance as yf
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return {}
    
    try:
        loop = asyncio.get_running_loop()
        def _fetch_cmp():
            res = {}
            for s in sym_list:
                try:
                    tkr = yf.Ticker(s)
                    h = tkr.history(period="1d")
                    if not h.empty:
                        res[s] = float(h['Close'].iloc[-1])
                except:
                    pass
            return res
            
        prices = await loop.run_in_executor(get_api_executor(), _fetch_cmp)
        return prices
    except Exception as e:
        return {}

if __name__ == "__main__":
    init_db()
    # if not startup_config_validation():
    #     print("❌ CRITICAL: Pattern Config validation failed. Check logs.")
    #     exit(1)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)