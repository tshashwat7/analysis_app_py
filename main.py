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
from config.logger_config import setup_logger
logger = setup_logger()

# --- Modular services ---
from config.market_utils import is_market_open, get_current_ist, get_current_utc,ensure_utc
from services.patterns.pattern_state_manager import cleanup_old_breakdown_states
import threading
from config.constants import ENABLE_CACHE_WARMER, ENABLE_JSON_ENRICHMENT, INDEX_TICKERS
from services.data_fetch import parse_index_csv
from services.fundamentals import compute_fundamentals
from services.indicators import compute_indicators
from config.config_resolver import get_config
from services.signal_engine import (
    compute_all_profiles,
    compute_momentum_strength,
    compute_trend_strength,
    compute_volatility_quality,
    enrich_hybrid_metrics_multi_horizon,
    generate_trade_plan,
    score_value_profile,
    score_growth_profile,
    score_quality_profile,
    score_momentum_profile,
)
from services.corporate_actions import get_corporate_actions
from services.summaries import build_all_summaries
from services.metrics_ext import compute_extended_metrics_sync
from services.flowchart_helper import build_flowchart_payload
from sqlalchemy.orm import Session
from services.db import SessionLocal, SignalCache, init_db
from services.strategy_analyzer import analyze_strategies

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
    global cache_warmer_task, _SHUTDOWN_IN_PROGRESS
    
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
        if API_EXECUTOR: API_EXECUTOR.shutdown(wait=True)
        if BACKGROUND_EXECUTOR: BACKGROUND_EXECUTOR.shutdown(wait=True)
            
        # Shutdown Compute Pool with FORCE KILL (Your Zombie Process Fix)
        if COMPUTE_EXECUTOR:
            # Longer timeout for DB safety
            COMPUTE_EXECUTOR.shutdown(wait=True)
            
            # Robust Zombie Cleanup Check if _processes is actually a dictionary before accessing .values()
            processes = getattr(COMPUTE_EXECUTOR, '_processes', None)
            if processes: 
                for proc in processes.values():
                    try:
                        if proc.is_alive():
                            proc.terminate()
                            proc.join(timeout=2.0)
                            if proc.is_alive():
                                proc.kill()
                    except Exception:
                        pass

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

def run_full_analysis(symbol: str, index_name: str = "nifty50") -> Dict[str, Any]:
    db = None
    enrichment_info = None
    try:
        global STOCK_TO_INDEX_MAP
        if not STOCK_TO_INDEX_MAP:
            build_smart_index_map()
            
        db = SessionLocal()
        analysis_data = {
            "symbol": symbol, 
            "fundamentals": {}, 
            "raw_indicators_by_horizon": {},  # ✅ Nested structure for profiles
            "indicators": {},                  # ✅ Flat structure for UI/Trade Plan
            "full_report": {}, 
            "trade_recommendation": {}, 
            "meta_scores": {},
            "macro_trend_status": "N/A", 
            "macro_close": None, 
            "strategy_report": {}, 
            "partial_error": False, 
            "error_details": []
        }
        horizons = ["intraday", "short_term", "long_term", "multibagger"]

        # Determine benchmark
        target_index = index_name
        clean_sym = symbol.strip().upper()
        if index_name in ["NSEStock", "default", "nifty500"]:
            smart_index = STOCK_TO_INDEX_MAP.get(clean_sym)
            if smart_index: 
                target_index = smart_index
        bench_symbol = INDEX_TICKERS.get(target_index, INDEX_TICKERS["default"])

        # =====================================================================
        # STEP 1: FUNDAMENTALS
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
        # STEP 2: INDICATORS (Multi-Horizon)
        # =====================================================================
        for h in horizons:
            try:
                h_indicators = compute_indicators(
                    symbol, horizon=h, benchmark_symbol=bench_symbol
                )
                analysis_data["raw_indicators_by_horizon"][h] = h_indicators
                logger.info(f"[{symbol}] Computed {len(h_indicators)} indicators for {h}")
                get_config(horizon=h, indicators=analysis_data['raw_indicators_by_horizon'][h], fundamentals=analysis_data['fundamentals'])
            except Exception as e:
                analysis_data["partial_error"] = True
                analysis_data["error_details"].append(f"Indicators/{h}: {e}")
                logger.warning(f"[{symbol}] Failed indicators for '{h}': {e}")

        # =====================================================================
        # STEP 3: ENRICH HYBRIDS (Multi-Horizon Aware)
        # =====================================================================
        # logger.debug(f"indicators - {analysis_data["raw_indicators_by_horizon"]}")
        # logger.debug(f"fundamentals - {analysis_data["fundamentals"]}")
        if analysis_data["fundamentals"] and analysis_data["raw_indicators_by_horizon"]:
            try:
                enriched_fund, enriched_inds = enrich_hybrid_metrics_multi_horizon(
                    analysis_data["fundamentals"], 
                    analysis_data["raw_indicators_by_horizon"]
                )
                # SORT ENRICH_FUNDS BY Key
                enriched_fund = dict(sorted(enriched_fund.items()))
                enriched_inds = {k: dict(sorted(v.items())) for k, v in enriched_inds.items()}
                # Update nested structure
                analysis_data["fundamentals"] = enriched_fund
                analysis_data["raw_indicators_by_horizon"] = enriched_inds
                fundamentals = analysis_data['fundamentals']
                for horizon in horizons:
                    get_config(horizon, indicators=analysis_data['raw_indicators_by_horizon'][horizon], 
                            fundamentals=fundamentals)
                    logger.info(f"✅ PRIMED {horizon} cache: {len(analysis_data['raw_indicators_by_horizon'][horizon])} keys")
            except Exception as e:
                logger.warning(f"[{symbol}] Hybrid enrichment failed: {e}")
                analysis_data["error_details"].append(f"Hybrid Failed: {e}")

        # =====================================================================
        # STEP 4: MACRO TREND (using short_term as baseline)
        # =====================================================================
        try:
            short_inds = analysis_data["raw_indicators_by_horizon"].get("short_term", {})
            nifty_metric = short_inds.get("nifty_trend_score")
            if nifty_metric:
                analysis_data["macro_trend_status"] = nifty_metric.get("desc", "N/A")
                analysis_data["macro_close"] = nifty_metric.get("value")
                if not analysis_data["macro_trend_status"] or \
                   analysis_data["macro_trend_status"] == "N/A":
                    analysis_data["macro_trend_status"] = "Neutral"
        except Exception as e:
            logger.debug(f"[{symbol}] Macro trend failed: {e}")

        # =====================================================================
        # STEP 5: PROFILE SCORING (using nested structure)
        # =====================================================================
        if analysis_data["fundamentals"] and analysis_data["raw_indicators_by_horizon"]:
            try:
                full_report = compute_all_profiles(
                    symbol, 
                    analysis_data["fundamentals"], 
                    analysis_data["raw_indicators_by_horizon"]
                )
                analysis_data["full_report"] = full_report

                # ✅ Log pure profile selection
                logger.info(
                    f"[{symbol}] Profile Selection: {full_report.get('best_fit')} "
                    f"(Pure Score: {full_report.get('best_score'):.2f}/10)"
                )
                
                # Meta scores (baseline: short_term)
                short_inds = analysis_data["raw_indicators_by_horizon"].get("short_term", {})
                analysis_data["meta_scores"] = {
                    "value": score_value_profile(analysis_data["fundamentals"]),
                    "growth": score_growth_profile(analysis_data["fundamentals"]),
                    "quality": score_quality_profile(analysis_data["fundamentals"]),
                    "momentum": score_momentum_profile(
                        analysis_data["fundamentals"], short_inds
                    ),
                }
            except Exception as e:
                analysis_data["partial_error"] = True
                analysis_data["error_details"].append(f"Scoring Failed: {e}")
                logger.error(f"[{symbol}] Profile scoring failed: {e}")

        # =====================================================================
        # STEP 6: SMART MERGE FOR UI (Create Flat indicators dict)
        # =====================================================================
        best_horizon_for_ui = "short_term"  # Default
        
        if "full_report" in analysis_data:
            best_horizon_for_ui = analysis_data["full_report"].get("best_fit", "short_term")
        
        # ✅ CRITICAL: Merge best-fit horizon into flat dict for UI
        analysis_data["indicators"] = analysis_data["raw_indicators_by_horizon"].get(
            best_horizon_for_ui, {}
        ).copy()
        
        logger.info(f"[{symbol}] UI using '{best_horizon_for_ui}' indicators ({len(analysis_data['indicators'])} keys)")

        # =====================================================================
        # STEP 7: STRATEGY ANALYSIS
        # =====================================================================
        try:
            full_rep = analysis_data.get("full_report", {})
            best_horizon = full_rep.get("best_fit", "short_term")
            
            # Use best-fit horizon's indicators
            target_inds = analysis_data["raw_indicators_by_horizon"].get(best_horizon, {})
            
            strategy_report = analyze_strategies(
                target_inds, 
                analysis_data["fundamentals"], 
                horizon=best_horizon
            )
            analysis_data["strategy_report"] = strategy_report
            
        except Exception as e:
            logger.error(f"[{symbol}] Strategy Analyzer Failed: {e}")
            analysis_data["strategy_report"] = {}

        # =====================================================================
        # STEP 8: TRADE PLAN (with Composites)
        # =====================================================================
        if "full_report" in analysis_data:
            try:
                full_rep = analysis_data["full_report"]
                best_horizon = full_rep.get("best_fit", "short_term")
                best_profile_data = full_rep.get("profiles", {}).get(best_horizon, {})
                
                # Get enriched indicators for this horizon
                horizon_inds = analysis_data["raw_indicators_by_horizon"].get(
                    best_horizon, {}
                ).copy()
                
                # Ensure composites exist
                if "trend_strength" not in horizon_inds:
                    horizon_inds["trend_strength"] = compute_trend_strength(horizon_inds, best_horizon)
                    horizon_inds["momentum_strength"] = compute_momentum_strength(horizon_inds)
                    horizon_inds["volatility_quality"] = compute_volatility_quality(horizon_inds)
                    logger.info(f"[{symbol}] Added composites to {best_horizon}")

                # ✅ This now applies ALL execution validation (penalties/gates/enhancements)
                trade_plan = generate_trade_plan(
                    best_profile_data, 
                    horizon_inds,
                    analysis_data.get("macro_trend_status", "N/A"),
                    horizon=best_horizon, 
                    strategy_report=analysis_data["strategy_report"],
                    fundamentals=analysis_data["fundamentals"],
                )
                analysis_data["trade_recommendation"] = trade_plan
                
                # ✅ Log trade decision
                logger.info(
                    f"[{symbol}] Trade Decision: {trade_plan.get('status')} | "
                    f"Signal: {trade_plan.get('trade_signal')} | "
                    f"Confidence Flow: {trade_plan.get('confidence_flow')}"
                )

            except Exception as e:
                logger.error(f"[{symbol}] Trade Plan Failed: {e}", exc_info=True)
                analysis_data["error_details"].append(f"Trade Plan Failed: {e}")
        # =====================================================================
        # STEP 9: DATABASE PERSISTENCE
        # =====================================================================
        if "full_report" in analysis_data:
            _save_analysis_to_db(db, symbol, analysis_data, index_name)

        # =====================================================================
        # STEP 10: ENRICHMENT METADATA (for background task)
        # =====================================================================
        if enrichment_info:
            analysis_data["_enrichment"] = enrichment_info

        return analysis_data
        
    except Exception as e:
        logger.error(f"[{symbol}] Worker Error: {e}", exc_info=True)
        return {"symbol": symbol, "error": str(e)}
    finally:
        if db: 
            db.close()

def _save_analysis_to_db(db: Session, symbol: str, data: Dict[str, Any], index_name: str):
    try:
        full_rep = data.get("full_report", {})
        profiles = full_rep.get("profiles", {})
        best_profile = profiles.get(full_rep.get("best_fit", "short_term"), {})
        trade_plan = data.get("trade_recommendation", {})
        indicators = data.get("indicators", {})
        final_score = best_profile.get("final_score", 0)

        horizon_data = {
            "intra_score": profiles.get("intraday", {}).get("final_score"),
            "short_score": profiles.get("short_term", {}).get("final_score"),
            "long_score": profiles.get("long_term", {}).get("final_score"),
            "multi_score": profiles.get("multibagger", {}).get("final_score"),
            "macro_index_name": index_name.upper(),
            "error_details": "; ".join(data.get("error_details", [])),
        }
        entry_val = trade_plan.get("entry")
        current_price = indicators.get("price", {}).get("value") or entry_val
        sl_val = trade_plan.get("stop_loss")
        sl_dist_str = "-"
        if current_price and sl_val:
            dist = abs(current_price - sl_val) / current_price * 100
            sl_dist_str = f"{dist:.1f}%"
        horizon_data["sl_dist"] = sl_dist_str

        entry_row = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()
        if not entry_row:
            entry_row = SignalCache(symbol=symbol)
            db.add(entry_row)

        entry_row.score = int(final_score * 10) if final_score else 0
        entry_row.recommendation = (best_profile.get("category", "HOLD") + "--" + best_profile.get("profile", ""))
        entry_row.best_horizon = full_rep.get("best_fit", "short_term")
        entry_row.signal_text = trade_plan.get("signal", "N/A")
        entry_row.conf_score = int(final_score * 10)
        entry_row.rr_ratio = trade_plan.get("rr_ratio")
        entry_row.entry_price = entry_val
        entry_row.stop_loss = sl_val
        entry_row.horizon_scores = horizon_data
        entry_row.updated_at = get_current_utc()
        db.commit()
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
            confidence = int(final_score * 10) if final_score else 0
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

            strat_summ = analysis_data.get("strategy_report", {}).get("summary", {})
            best_strat = strat_summ.get("best_strategy", "N/A")
            all_fits = ", ".join(strat_summ.get("all_fits", []))

            pattern_keys = ["golden_cross", "double_top_bottom", "cup_handle", "darvas_box", 
                            "flag_pennant", "minervini_stage2", "bollinger_squeeze", "three_line_strike", "ichimoku_signals"]
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

            def safe_val(v):
                if v is None: return None
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
                return v

            flat_output = {
                "symbol": sym,
                "score": int(final_score * 10) if final_score else 0,
                "confidence": confidence,
                "recommendation": (best_profile.get("category", "HOLD") + "--" + best_profile.get("profile", "")),
                "bull_score": bull_score,
                "bull_signal": signal_str.replace("_", " "),
                "rr_ratio": rr if rr else 0,
                "entry_trigger": entry_val if entry_val else 0,
                "sl_dist": sl_dist_str,
                "best_strategy": best_strat,
                "fitting_strategies": all_fits,
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
    return {"index": index_name, "count": len(stocks), "tickers": tickers, "pairs": pairs}

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_common(request: Request, symbol: str, index: str = "nifty50", horizon: str = None):
    # Lazy Load
    _ensure_stocks_loaded()
    try:
        loop = asyncio.get_running_loop()
        analysis_data = await loop.run_in_executor(get_compute_executor(), run_full_analysis, symbol, index)
        full_report = analysis_data.get("full_report", {})
        selected_profile_name = horizon if (horizon and horizon in full_report.get("profiles", {})) else full_report.get("best_fit", "short_term")
        profile_report = full_report.get("profiles", {}).get(selected_profile_name, {})
        analysis_data["profile_report"] = profile_report
        
        current_horizon_inds = analysis_data.get("raw_indicators_by_horizon", {}).get(selected_profile_name)
        if not current_horizon_inds: current_horizon_inds = analysis_data.get("indicators", {})

        # ... (Prepare standard summaries/scores variables) ...
        best_fit = full_report.get("best_fit", "short_term")

        if selected_profile_name != best_fit:
            trade_plan = generate_trade_plan(
                profile_report,
                current_horizon_inds,
                analysis_data.get("macro_trend_status", "N/A"),
                horizon=selected_profile_name,
                strategy_report=analysis_data.get("strategy_report", {}),
                fundamentals=analysis_data.get("fundamentals", {}),
            )
        else:
            trade_plan = analysis_data.get("trade_recommendation", {})

        final_score = profile_report.get("final_score", 0)
        meta_scores = analysis_data.get("meta_scores", {})
        summary_context = {
            "indicators": analysis_data.get("indicators", {}),
            "trade_recommendation": trade_plan,
            "profile_report": profile_report,
            "meta_scores": analysis_data.get("meta_scores", {}),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A")
        }
        summaries = build_all_summaries(summary_context)

        context = {
            "request": request, "symbol": symbol, "current_index": index, "selected_horizon": selected_profile_name,
            "error": None, "full_report": full_report, "profile_report": profile_report,
            "fundamentals": analysis_data.get("fundamentals", {}), "indicators": analysis_data.get("indicators", {}),
            "meta_scores": {"value": meta_scores.get("value", 0) or 0, "growth": meta_scores.get("growth", 0) or 0,
                            "quality": meta_scores.get("quality", 0) or 0, "momentum": meta_scores.get("momentum", 0) or 0},
            "trade_recommendation": trade_plan, "summaries": summaries,
            "final_signal": profile_report.get("category", "HOLD"), "bull_signal": trade_plan.get("signal", "N/A"),
            "total_score": final_score * 10, "confidence": int(final_score * 10), "macro_index_name": index.upper(),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A"), "macro_close": analysis_data.get("macro_close"),
            "reasons": [trade_plan.get("reason", "")], "strategy_report": analysis_data.get("strategy_report", {}),
        }
        logger.debug(f" context passed to result.html for {symbol} is {context}")
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
    try:
        tickers = payload.tickers
        if not tickers: return JSONResponse({})
        summary = {}
        loop = asyncio.get_running_loop()
        def fetch_acts(t): return t, get_corporate_actions([t], mode="upcoming", lookback_days=7)
        tasks = [loop.run_in_executor(get_api_executor(), fetch_acts, t) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple):
                t, acts = res
                try:
                    if acts and acts[0].get("actions"):
                        valid_action = None
                        today = datetime.date.today()
                        for action in acts[0]["actions"]:
                            ex_date_str = action.get("ex_date")
                            if ex_date_str:
                                try:
                                    if datetime.date.fromisoformat(ex_date_str) >= today:
                                        valid_action = action
                                        break
                                except ValueError: pass
                        if valid_action:
                            a = valid_action
                            typ = a.get("type", "").replace("Upcoming ", "")
                            amt, exd = a.get("amount", ""), a.get("ex_date", "")
                            parts = []
                            if "Dividend" in typ: parts.append("Dividend Announced")
                            else: parts.append(typ)
                            if amt: parts.append(f"₹{amt}")
                            if exd: parts.append(f"(Ex: {exd})")
                            summary[t] = " ".join(parts)
                except Exception: pass
        return JSONResponse(summary)
    except Exception: return JSONResponse({})

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

@app.get("/metrics_ext")
async def metrics_ext_route(ticker: str):
    return await compute_extended_metrics_sync(ticker)

@app.get("/flowchart", response_class=HTMLResponse)
async def show_flowchart(request: Request):
    return templates.TemplateResponse("Stock_Buy-sell_Flowcharts.html", {"request": request})

@app.get("/flowchart_payload")
async def flowchart_payload(symbol: str, index: str = Query("NIFTY50")):
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_api_executor(), build_flowchart_payload, symbol, index)
    except Exception as e:
        return {"error": str(e)}

def startup_config_validation():
    """
    Validates MASTER_CONFIG at startup using ConfigValidator.
    Critical: Catches structural errors before any analysis runs.
    """
    from config.config_validators import ConfigValidator
    from config.master_config import MASTER_CONFIG
    import sys
    
    logger.info("="*70)
    logger.info("VALIDATING MASTER_CONFIG v2.0...")
    logger.info("="*70)
    
    validator = ConfigValidator(MASTER_CONFIG)
    report = validator.validate_all()
    
    if not report.is_valid:
        logger.error("❌ CRITICAL: Config validation FAILED!")
        report.print_report()
        
        # Block startup if errors exist
        logger.error("Fix config errors in master_config.py before starting.")
        sys.exit(1)
    
    # Show warnings but don't block
    if report.warnings:
        logger.warning(f"⚠️  Config has {len(report.warnings)} warnings:")
        for warning in report.warnings[:5]:  # Show first 5
            logger.warning(f"  - {warning.path}: {warning.message}")
    
    logger.info("✅ Config validation passed!")
    logger.info(f"   Profiles: {len(MASTER_CONFIG.get('horizons', {}))}")
    logger.info(f"   Patterns: {len(MASTER_CONFIG.get('global', {}).get('pattern_entry_rules', {}))}")
    logger.info("="*70)
    
    return True

if __name__ == "__main__":
    init_db()
    if not startup_config_validation():
        print("❌ CRITICAL: Pattern Config validation failed. Check logs.")
        exit(1)
    run_scheduled_cleanup()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)