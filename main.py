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
import traceback
from typing import Dict, Any, List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, Body, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import logging

# --- Modular services ---
from config.constants import ENABLE_CACHE_WARMER, INDEX_TICKERS
from services.data_fetch import (
    safe_history,
    save_stocks_list,
    parse_index_csv,
    get_price_history,
    fetch_data,
)
from services.fundamentals import compute_fundamentals
from services.indicators import compute_indicators
from services.signal_engine import (
    compute_all_profiles,
    enrich_hybrid_metrics,
    generate_trade_plan,
    score_value_profile,
    score_growth_profile,
    score_quality_profile,
    score_momentum_profile,
)
from services.corporate_actions import get_corporate_actions
from services.summaries import build_all_summaries
from services.metrics_ext import compute_extended_metrics_sync
from services.flowchart_helper import build_flowchart_payload, df_hash
from sqlalchemy.orm import Session
from services.db import SessionLocal, SignalCache, init_db
from services.strategy_analyzer import analyze_strategies

# Environment-configurable parameters (tweak as needed)
WARMER_BATCH_SIZE = int(os.getenv("WARMER_BATCH_SIZE", "5"))
WARMER_TOP_N_DURING_MARKET = int(os.getenv("WARMER_TOP_N_DURING_MARKET", "50"))
WARMER_LRU_TARGET = int(os.getenv("WARMER_LRU_TARGET", "500"))
WARMER_MARKET_INTERVAL_SEC = int(
    os.getenv("WARMER_MARKET_INTERVAL_SEC", str(15 * 60))
)  # 15 minutes during market
WARMER_OFFPEAK_INTERVAL_SEC = int(
    os.getenv("WARMER_OFFPEAK_INTERVAL_SEC", str(60 * 60))
)  # 60 minutes off-peak
WARMER_DEEP_HOUR = int(
    os.getenv("WARMER_DEEP_HOUR", "2")
)  # hour (IST) to run a deep warm
WARMER_DEEP_SLEEP_HOURS = int(os.getenv("WARMER_DEEP_SLEEP_HOURS", "6"))
WARMER_BATCH_SLEEP_MARKET = float(os.getenv("WARMER_BATCH_SLEEP_MARKET", "5"))
WARMER_BATCH_SLEEP_OFFPEAK = float(os.getenv("WARMER_BATCH_SLEEP_OFFPEAK", "2"))
WARMER_BATCH_TIMEOUT_SEC = int(
    os.getenv("WARMER_BATCH_TIMEOUT_SEC", str(5 * 60))
)  # per-batch timeout
WARMER_RATE_LIMIT_COOLDOWN_SEC = int(
    os.getenv("WARMER_RATE_LIMIT_COOLDOWN_SEC", "300")
)  # 5 minutes
IST = pytz.timezone("Asia/Kolkata")

init_db()


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- CACHE WARMER ---


def run_analysis_for_cache(symbol: str):
    """Synchronous worker function to execute the heavy lifting."""
    try:
        run_full_analysis(symbol)
        logger.debug(f"[WARMER] Successfully warmed cache for {symbol}.")
    except Exception as e:
        logger.error(f"[WARMER] Failed to run analysis for {symbol}: {e}")


# ----------------------------------------------------------------------
# SMART CACHE WARMER (Robust Version)
# ----------------------------------------------------------------------

logger = logging.getLogger("warmer")


async def periodic_warmer():
    """
    Background task to pre-fetch data.
    - Timezone-aware (IST)
    - Smart schedule (market vs off-peak)
    - Deep warm once per night
    - Rate-limit cooling
    - Cancels cleanly on shutdown
    """
    # small startup delay
    await asyncio.sleep(10)
    while True:
        is_market_hours = False
        try:
            now = datetime.datetime.now(IST)
            is_weekend = now.weekday() >= 5
            hour = now.hour
            # Market hours (9:00 to 16:00 IST)
            is_market_hours = (9 <= hour < 16) and (not is_weekend)
            # Deep warm once a day (configurable hour)
            is_deep_warm = (hour == WARMER_DEEP_HOUR) and (not is_weekend)
            # Determine symbols to warm
            if is_market_hours:
                logger.info(
                    "[WARMER] Market Open (IST): warming Top %d symbols",
                    WARMER_TOP_N_DURING_MARKET,
                )
                symbols_to_warm = [
                    s[0] for s in ALL_STOCKS_LIST[:WARMER_TOP_N_DURING_MARKET]
                ]
                batch_sleep = WARMER_BATCH_SLEEP_MARKET
            else:
                if is_deep_warm:
                    logger.info(
                        "[WARMER] Off-Peak (IST) - DEEP WARM: warming all %d symbols",
                        len(ALL_STOCKS_LIST),
                    )
                    symbols_to_warm = [s[0] for s in ALL_STOCKS_LIST]
                    batch_sleep = WARMER_BATCH_SLEEP_OFFPEAK
                else:
                    # Off-peak light warm: warm a subset per cycle to avoid constant full re-runs
                    # We'll cycle through the universe across multiple cycles.
                    logger.info("[WARMER] Off-Peak (IST) - Light warm cycle")
                    symbols_to_warm = [
                        s[0] for s in ALL_STOCKS_LIST
                    ]  # keep list; we will batch it
                    batch_sleep = WARMER_BATCH_SLEEP_OFFPEAK
            loop = asyncio.get_running_loop()
            # iterate over in batches
            for i in range(0, len(symbols_to_warm), WARMER_BATCH_SIZE):
                batch = symbols_to_warm[i : i + WARMER_BATCH_SIZE]
                # Kick off analysis of the batch using the compute executor
                futures = [
                    loop.run_in_executor(COMPUTE_EXECUTOR, run_full_analysis, symbol)
                    for symbol in batch
                ]
                try:
                    # Wait for the batch to complete within timeout; return_exceptions=True so we get results/errors
                    results = await asyncio.wait_for(
                        asyncio.gather(*futures, return_exceptions=True),
                        timeout=WARMER_BATCH_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[WARMER] Batch timed out after %d seconds. Moving to next batch.",
                        WARMER_BATCH_TIMEOUT_SEC,
                    )
                    # We can't directly cancel the underlying thread tasks, but we should continue and cool down
                    await asyncio.sleep(WARMER_RATE_LIMIT_COOLDOWN_SEC)
                    # Move to next batch
                    continue
                except Exception as e:
                    logger.error(
                        "[WARMER] Unexpected gather error: %s\n%s",
                        e,
                        traceback.format_exc(),
                    )
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
                    logger.warning(
                        "[WARMER] Rate-limited; cooling down for %d seconds before resuming.",
                        WARMER_RATE_LIMIT_COOLDOWN_SEC,
                    )
                    await asyncio.sleep(WARMER_RATE_LIMIT_COOLDOWN_SEC)
                    # abort this warming cycle to avoid repeated hits
                    break
                # breathe between batches
                await asyncio.sleep(batch_sleep)
            logger.info(
                "[WARMER] Cycle complete (is_deep_warm=%s, is_market_hours=%s)",
                is_deep_warm,
                is_market_hours,
            )
        except asyncio.CancelledError:
            logger.info("[WARMER] Cancelled - exiting periodic_warmer cleanly.")
            raise
        except Exception as e:
            logger.error(
                "[WARMER] Critical Loop Error: %s\n%s", e, traceback.format_exc()
            )
        # Decide how long to sleep before next cycle
        try:
            if is_market_hours:
                interval = WARMER_MARKET_INTERVAL_SEC
            else:
                # If we did a deep warm, sleep longer so deep warms don't repeat too often
                if "is_deep_warm" in locals() and is_deep_warm:
                    interval = WARMER_DEEP_SLEEP_HOURS * 3600
                else:
                    interval = WARMER_OFFPEAK_INTERVAL_SEC
        except Exception:
            interval = WARMER_OFFPEAK_INTERVAL_SEC

        # Respect cancellation while sleeping by breaking into small sleeps
        sleep_left = interval
        while sleep_left > 0:
            try:
                await asyncio.sleep(min(30, sleep_left))
            except asyncio.CancelledError:
                logger.info("[WARMER] Cancelled during sleep - exiting.")
                raise
            sleep_left -= 30


# ---- Lifespan with clean shutdown / cancelation ----


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_smart_index_map()
    global cache_warmer_task

    logger.info("Starting API Executor and Background Executor...")

    # ✅ CONDITIONAL START
    if ENABLE_CACHE_WARMER:
        cache_warmer_task = asyncio.create_task(periodic_warmer())
        logger.info("🔥 Cache Warmer: ENABLED")
    else:
        cache_warmer_task = None  # ✅ Explicitly set to None
        logger.info("⏸️  Cache Warmer: DISABLED (development mode)")

    try:
        yield
    finally:
        # ✅ CONDITIONAL SHUTDOWN
        if cache_warmer_task is not None:
            cache_warmer_task.cancel()
            try:
                await cache_warmer_task
            except asyncio.CancelledError:
                logger.info("Cache warmer canceled cleanly.")
            except Exception as e:
                logger.warning("Cache warmer raised during cancel: %s", e)

        # Shutdown executors
        try:
            API_EXECUTOR.shutdown(wait=False)
            COMPUTE_EXECUTOR.shutdown(wait=False)
            BACKGROUND_EXECUTOR.shutdown(wait=False)
        except Exception:
            logger.warning("Executor shutdown issue", exc_info=True)

        logger.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# --- PYDANTIC MODELS ---
class AnalyzeRequest(BaseModel):
    symbol: str
    index: str = "nifty50"


class QuickScoresRequest(BaseModel):
    symbols: List[str]
    index_name: str = "nifty50"


class CorpActionsRequest(BaseModel):
    tickers: List[str]


logger = logging.getLogger("stock_analyzer")
logging.basicConfig(level=logging.INFO)

# --- FILES ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
NSE_STOCKS_FILE = os.path.join(DATA_DIR, "nse_stocks.json")

# ==============================================================================
# ### PATCH FIX: SEPARATE EXECUTORS FOR API AND BACKGROUND TASKS
# ==============================================================================
# API Executor: Higher thread count for responsive user interface
API_THREADS = 5
API_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=API_THREADS, thread_name_prefix="API_Worker"
)

MAX_CPU = max(1, (os.cpu_count() or 2) - 1)
COMPUTE_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CPU)

# Background Executor: Lower thread count to prevent resource starvation
BG_THREADS = 2
BACKGROUND_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=BG_THREADS, thread_name_prefix="BG_Worker"
)

# Global reference for the warmer task
cache_warmer_task = None
CACHE_WARMING_INTERVAL_SECONDS = 15 * 60  # 15 minutes

# --- CACHE HELPERS ---
CACHE_LOCK = threading.Lock()
CACHE_TTL = 60 * 60  # 1 hour
STOCK_TO_INDEX_MAP: Dict[str, str] = {}


def load_or_create_index(index_name: str):
    json_file = os.path.join(DATA_DIR, f"{index_name}.json")
    csv_file = os.path.join(DATA_DIR, f"{index_name}.csv")

    # If JSON exists, load it
    if os.path.exists(json_file):
        stocks = get_cached_stocks(json_file)
        if stocks:
            return stocks

    # If JSON missing but CSV exists → build JSON
    if os.path.exists(csv_file):
        logger.info(f"Parsing CSV for index: {index_name}")
        pairs = parse_index_csv(csv_file)  # -> List[(symbol, name)]

        if pairs:
            json_data = [{"symbol": s, "name": n} for s, n in pairs]
            try:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)
                logger.info(f"Created {json_file} with {len(pairs)} stocks")
            except Exception as e:
                logger.error(f"Failed writing {json_file}: {e}")

            return pairs

    logger.warning(f"No JSON or CSV found for {index_name}")
    return []


def load_or_create_global_stocks():
    """
    Loads NSEStock.json (global stock universe).
    If missing/corrupted -> rebuild from NSEStock.csv.
    """
    return load_or_create_index("NSEStock")


def build_smart_index_map():
    global STOCK_TO_INDEX_MAP

    priority_files = [
        "NSEStock",
        "niftyauto",
        "niftybank",
        "niftyfmcg",
        "niftyinfra",
        "niftyit",
        "niftypharma",
        "niftyrealty",
        "nifty500",
        "smallcap250",
        "microcap250",
        "smallcap100",
        "midcap150",
        "niftynext50",
        "nifty100",
        # 5. The Gold Standard (Overwrites everything else)
        "nifty50",
    ]

    for filename in priority_files:
        # Ensure the file actually exists before trying to load
        filepath = os.path.join(DATA_DIR, f"{filename}.json")

        # (Optional: If you only have .csv for some, you might need to parse_index_csv here
        # or ensure your load_index logic converts them to JSON first.
        # The get_cached_stocks helper usually handles JSON reading.)
        if os.path.exists(filepath):
            stocks = get_cached_stocks(filepath)
            for symbol, _ in stocks:
                STOCK_TO_INDEX_MAP[symbol.strip().upper()] = filename

    logger.info(f"[INIT] Smart Index Map built for {len(STOCK_TO_INDEX_MAP)} symbols.")


def get_cached(symbol: str):
    """
    Retrieves analysis from SQLite (Persistent).
    Returns the flattened dictionary expected by AG Grid.
    """
    db: Session = SessionLocal()
    try:
        # Query DB
        entry = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()

        if not entry:
            return None

        # Check TTL (Optional: if you want data to expire even from DB)
        age = (datetime.datetime.now() - entry.updated_at).total_seconds()
        if age > CACHE_TTL:
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

            # Pack the "Extra" fields into the JSON column
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
            def _f(k):
                return value.get(k) if isinstance(value.get(k), (int, float)) else 0.0

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
                    stop_loss=_f(
                        "stop_loss"
                    ),  # Ensure this key exists in quick_scores or default to 0
                    horizon_scores=horizon_data,
                    updated_at=datetime.datetime.now(),
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
                entry.updated_at = datetime.datetime.now()

            db.commit()
    except Exception as e:
        logger.error(f"DB Write Error {symbol}: {e}")
        db.rollback()
    finally:
        db.close()


def get_cached_stocks(index_file: str) -> List[Tuple[str, str]]:
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                out: List[Tuple[str, str]] = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "symbol" in item:
                            out.append(
                                (item["symbol"], item.get("name", item["symbol"]))
                            )
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            out.append((item[0], item[1]))
                        elif isinstance(item, str):
                            out.append((item, item))
                return out
        except Exception as e:
            logger.exception("Error reading cached stocks: %s", e)
            return []
    return []


# --- GLOBAL STOCK LIST LOADER ---
ALL_STOCKS_LIST = load_or_create_global_stocks()
if not ALL_STOCKS_LIST:
    ALL_STOCKS_LIST = get_cached_stocks(NSE_STOCKS_FILE)
ALL_STOCKS_MAP = {s[0]: s[1] for s in ALL_STOCKS_LIST}


# --- CORE ANALYSIS FUNCTIONS ---


def run_full_analysis(symbol: str, index_name: str = "nifty50") -> Dict[str, Any]:
    """
    Worker function running in a separate process.
    Fully self-contained: Manages its own DB session and Memory state.
    """
    db = None
    try:
        db = SessionLocal()
        analysis_data = {
            "symbol": symbol,
            "fundamentals": {},
            "indicators": {},
            "full_report": {},
            "trade_recommendation": {},
            "meta_scores": {},
            "macro_trend_status": "N/A",
            "macro_close": None,
            "strategy_report": {},
            "partial_error": False,
            "error_details": [],
        }
        horizons = ["intraday", "short_term", "long_term", "multibagger"]

        # ... (Keep Index/Symbol Logic 1-2 unchanged) ...
        target_index = index_name
        clean_sym = symbol.strip().upper()
        if index_name in ["NSEStock", "default", "nifty500"]:
            smart_index = STOCK_TO_INDEX_MAP.get(clean_sym)
            if smart_index:
                target_index = smart_index
        bench_symbol = INDEX_TICKERS.get(target_index, INDEX_TICKERS["default"])

        # 1. Fundamentals
        try:
            analysis_data["fundamentals"] = compute_fundamentals(symbol)
        except Exception as e:
            analysis_data["partial_error"] = True
            analysis_data["error_details"].append(f"Fundamentals Failed: {e}")
            logger.error(f"[{symbol}] Fundamentals failed: {e}")

        # 2. Indicators
        all_indicators = {}
        analysis_data["raw_indicators_by_horizon"] = {}
        for h in horizons:
            try:
                h_indicators = compute_indicators(
                    symbol, horizon=h, benchmark_symbol=bench_symbol
                )
                analysis_data["raw_indicators_by_horizon"][h] = h_indicators

                # ... (Keep warning logic) ...
                for key, new_val in h_indicators.items():
                    if key in all_indicators:
                        old_v = all_indicators[key].get("value")
                        new_v = new_val.get("value")
                        if str(old_v) != str(new_v):
                            logger.debug(f"[{symbol}] WARNING: Overwrote {key}!")
                    all_indicators.update(h_indicators)

            except Exception as e:
                analysis_data["partial_error"] = True
                analysis_data["error_details"].append(f"Indicators for {h} Failed: {e}")
                logger.warning(
                    f"[{symbol}] Failed to compute indicators for horizon '{h}': {e}"
                )

        analysis_data["indicators"] = all_indicators

        # 3. Hybrids
        if analysis_data["fundamentals"] and analysis_data["indicators"]:
            try:
                hybrids = enrich_hybrid_metrics(
                    analysis_data["fundamentals"], analysis_data["indicators"]
                )
                analysis_data["fundamentals"].update(hybrids)
            except Exception as e:
                analysis_data["error_details"].append(
                    f"Hybrid Metric Calculation Failed: {e}"
                )

        # 4. Macro Context
        try:
            nifty_metric = analysis_data["indicators"].get("nifty_trend_score")
            if nifty_metric:
                analysis_data["macro_trend_status"] = nifty_metric.get("desc", "N/A")
                analysis_data["macro_close"] = nifty_metric.get("value")
                if (
                    not analysis_data["macro_trend_status"]
                    or analysis_data["macro_trend_status"] == "N/A"
                ):
                    analysis_data["macro_trend_status"] = "Neutral"
        except Exception:
            pass

        # 🔥 5. Profile Scores (Compute this FIRST to find the Best Fit)
        if analysis_data["fundamentals"] and analysis_data["indicators"]:
            try:
                full_report = compute_all_profiles(
                    symbol,
                    analysis_data["fundamentals"],
                    analysis_data["raw_indicators_by_horizon"],
                )
                analysis_data["full_report"] = full_report

                # Meta Scores
                analysis_data["meta_scores"] = {
                    "value": score_value_profile(analysis_data["fundamentals"]),
                    "growth": score_growth_profile(analysis_data["fundamentals"]),
                    "quality": score_quality_profile(analysis_data["fundamentals"]),
                    "momentum": score_momentum_profile(
                        analysis_data["fundamentals"], analysis_data["indicators"]
                    ),
                }
            except Exception as e:
                analysis_data["partial_error"] = True
                analysis_data["error_details"].append(f"Scoring/Trade Plan Failed: {e}")
        # 6. [NEW] Strategy Analysis (Dynamic Horizon) We now know the 'best_fit' from step 5, so we use it!
        try:
            # 1. Determine the Best Horizon (Default to short_term if failed)
            full_rep = analysis_data.get("full_report", {})
            best_horizon = full_rep.get("best_fit", "short_term")
            # 2. Grab the specific indicators for that horizon
            # (This prevents Long Term RSI from overwriting Intraday RSI)
            target_inds = analysis_data["raw_indicators_by_horizon"].get(best_horizon)
            if not target_inds:
                target_inds = analysis_data.get("indicators", {})  # Fallback to merged
            # 3. Run Strategy Engine on the BEST horizon
            strategy_report = analyze_strategies(
                target_inds, analysis_data["fundamentals"], horizon=best_horizon
            )
            analysis_data["strategy_report"] = strategy_report

        except Exception as e:
            logger.error(f"[{symbol}] Strategy Analyzer Failed: {e}")
            analysis_data["strategy_report"] = {}

        # 7. Generate Trade Plan (Now using the Strategy Report)
        if "full_report" in analysis_data:
            try:
                full_rep = analysis_data["full_report"]
                best_profile_name = full_rep.get("best_fit", "short_term")
                best_profile_data = full_rep.get("profiles", {}).get(
                    best_profile_name, {}
                )
                best_horizon = full_rep.get("best_fit", "short_term")
                horizon_inds = analysis_data["raw_indicators_by_horizon"].get(
                    best_horizon
                )

                trade_plan = generate_trade_plan(
                    best_profile_data,
                    horizon_inds,
                    analysis_data.get("macro_trend_status", "N/A"),
                    horizon=best_profile_name,
                    strategy_report=analysis_data["strategy_report"],
                    fundamentals=analysis_data["fundamentals"],
                )
                analysis_data["trade_recommendation"] = trade_plan

            except Exception as e:
                analysis_data["error_details"].append(f"Trade Plan Failed: {e}")

        # 8. SAVE TO SQLITE (Keep unchanged)
        if "full_report" in analysis_data:
            _save_analysis_to_db(db, symbol, analysis_data, index_name)

        return analysis_data

    except Exception as e:
        logger.error(f"[{symbol}] Worker Error: {e}")
        return {"symbol": symbol, "error": str(e)}

    finally:
        if db:
            db.close()


# Helper function to handle the DB write inside the worker
def _save_analysis_to_db(
    db: Session, symbol: str, data: Dict[str, Any], index_name: str
):
    try:
        full_rep = data.get("full_report", {})
        profiles = full_rep.get("profiles", {})
        best_profile = profiles.get(full_rep.get("best_fit", "short_term"), {})
        trade_plan = data.get("trade_recommendation", {})
        indicators = data.get("indicators", {})

        final_score = best_profile.get("final_score", 0)

        # Flatten Logic (Same as your original set_cached, but using the passed 'db' session)
        horizon_data = {
            "intra_score": profiles.get("intraday", {}).get("final_score"),
            "short_score": profiles.get("short_term", {}).get("final_score"),
            "long_score": profiles.get("long_term", {}).get("final_score"),
            "multi_score": profiles.get("multibagger", {}).get("final_score"),
            "macro_index_name": index_name.upper(),
            "error_details": "; ".join(data.get("error_details", [])),
        }

        # Calculate Sl Distance
        entry = trade_plan.get("entry")
        current_price = indicators.get("price", {}).get("value") or entry
        sl_val = trade_plan.get("stop_loss")
        sl_dist_str = "-"
        if current_price and sl_val:
            dist = abs(current_price - sl_val) / current_price * 100
            sl_dist_str = f"{dist:.1f}%"
        horizon_data["sl_dist"] = sl_dist_str

        # Upsert Logic
        entry_row = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()

        if not entry_row:
            entry_row = SignalCache(symbol=symbol)
            db.add(entry_row)

        entry_row.score = int(final_score * 10) if final_score else 0
        entry_row.recommendation = (
            best_profile.get("category", "HOLD")
            + "--"
            + best_profile.get("profile", "")
        )
        entry_row.best_horizon = full_rep.get("best_fit", "short_term")
        entry_row.signal_text = trade_plan.get("signal", "N/A")
        entry_row.conf_score = int(final_score * 10)  # Simplified conf
        entry_row.rr_ratio = trade_plan.get("rr_ratio")
        entry_row.entry_price = entry
        entry_row.stop_loss = sl_val
        entry_row.horizon_scores = horizon_data

        # Explicit timestamp update (Phase 1 Improvement C)
        entry_row.updated_at = datetime.datetime.now(datetime.timezone.utc)

        db.commit()
    except Exception as e:
        logger.error(f"[{symbol}] DB Save Failed: {e}")
        db.rollback()


# --- ENDPOINTS ---


@app.post("/quick_scores")
async def quick_scores(payload: QuickScoresRequest):
    """
    The Core Endpoint for the Hybrid Dashboard.
    Receives chunk of symbols, runs analysis on API_EXECUTOR.
    """
    symbols = [s.strip().upper() for s in payload.symbols if s.strip()]
    index_name = payload.index_name

    results: Dict[str, Any] = {}
    to_fetch = []

    for s in symbols:
        c = get_cached(s)
        if c:
            results[s] = {**c, "cached": True}
        else:
            to_fetch.append(s)

    if not to_fetch:
        return results

    # ### KEEPING YOUR ROBUST WORKER LOGIC
    loop = asyncio.get_running_loop()

    calc_tasks = [
        loop.run_in_executor(COMPUTE_EXECUTOR, run_full_analysis, s, index_name)
        for s in to_fetch
    ]

    analyzed_data_list = await asyncio.gather(*calc_tasks)

    for analysis_data in analyzed_data_list:
        sym = analysis_data.get("symbol")
        if not sym:
            continue

        try:
            full_rep = analysis_data.get("full_report", {})
            profiles = full_rep.get("profiles", {})
            best_profile = profiles.get(full_rep.get("best_fit", "short_term"), {})
            trade_plan = analysis_data.get("trade_recommendation", {})
            indicators = analysis_data.get("indicators", {})  # Need this for Price

            final_score = best_profile.get("final_score", 0)
            confidence = int(final_score * 10) if final_score else 0
            signal_str = trade_plan.get("signal", "N/A")
            error_status = analysis_data.get("error_details", [])
            # Bull Score Logic
            bull_score = 0
            if "SQUEEZE" in signal_str:
                bull_score = 95
            elif "TREND" in signal_str:
                bull_score = 85
            elif "DIP" in signal_str:
                bull_score = 75
            elif "HOLD" in signal_str:
                bull_score = 50
            else:
                bull_score = 20

            def safe_val(v):
                if v is None:
                    return None
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v

            # --- PHASE 1 INJECTION START: Extract R:R and SL Distance ---
            rr = trade_plan.get("rr_ratio")
            entry_val = trade_plan.get("entry")
            sl_val = trade_plan.get("stop_loss")

            # Calculate SL Distance % (Visual Aid)
            # We use 'Price' from indicators, or fallback to 'entry'
            current_price = indicators.get("price", {}).get("value")
            if not current_price:
                current_price = entry_val

            sl_dist_str = "-"
            if current_price and sl_val and current_price > 0:
                dist = abs(current_price - sl_val) / current_price * 100
                sl_dist_str = f"{dist:.1f}%"

            strat_summ = analysis_data.get("strategy_report", {}).get("summary", {})
            best_strat = strat_summ.get("best_strategy", "N/A")
            all_fits = ", ".join(strat_summ.get("all_fits", []))
            # --- PHASE 1 INJECTION END ---
            pattern_keys = [
                "golden_cross",
                "double_top_bottom",
                "cup_handle",
                "darvas_box",
                "flag_pennant",
                "minervini_stage2",
                "bollinger_squeeze",
                "three_line_strike",
                "ichimoku_signals",
            ]

            top_pattern_name = ""
            top_pattern_score = 0

            for pk in pattern_keys:
                p_obj = indicators.get(pk)
                if p_obj and isinstance(p_obj, dict) and p_obj.get("found"):
                    s = p_obj.get("score", 0)
                    if s > top_pattern_score:
                        top_pattern_score = s
                        # Format Name: "darvas_box" -> "Darvas Box"
                        top_pattern_name = pk.replace("_", " ").title()

            # Shorten names for grid to save space
            name_map = {
                "Bollinger Squeeze": "Squeeze",
                "Minervini Stage2": "VCP",
                "Three Line Strike": "3-Strike",
                "Golden Cross": "Gold Cross",
                "Double Top Bottom": "Double T/B",
                "Cup Handle": "Cup",
                "Darvas Box": "Darvas",
                "Flag Pennant": "Flag",
                "Ichimoku Signals": "Ichimoku",
            }
            top_pattern_name = name_map.get(top_pattern_name, top_pattern_name)
            # -----------------------------------------
            # 3. Flatten for Grid
            flat_output = {
                "symbol": sym,
                "score": int(final_score * 10) if final_score else 0,
                "confidence": confidence,
                "recommendation": (
                    best_profile.get("category", "HOLD")
                    + "--"
                    + best_profile.get("profile", "")
                ),
                "bull_score": bull_score,
                "bull_signal": signal_str.replace("_", " "),
                # --- NEW COLUMNS FOR GRID ---
                "rr_ratio": rr if rr else 0,
                "entry_trigger": entry_val if entry_val else 0,
                "sl_dist": sl_dist_str,
                "best_strategy": best_strat,
                "fitting_strategies": all_fits,  # Useful for Grid tooltip
                # ----------------------------
                # Horizon Scores
                "intra_score": safe_val(
                    profiles.get("intraday", {}).get("final_score")
                ),
                "short_score": safe_val(
                    profiles.get("short_term", {}).get("final_score")
                ),
                "long_score": safe_val(
                    profiles.get("long_term", {}).get("final_score")
                ),
                "multi_score": safe_val(
                    profiles.get("multibagger", {}).get("final_score")
                ),
                # Macro context
                "macro_index_name": index_name.upper(),
                # Pattern
                "top_pattern": top_pattern_name,
                "error_details": "; ".join(error_status) if error_status else "",
            }

            set_cached(sym, flat_output)
            results[sym] = flat_output
        except Exception as e:
            logger.error(f"[{sym}] Error flattening analysis data: {e}")
            results[sym] = {"symbol": sym, "recommendation": "Error"}

    return results


@app.get("/load_index/{index_name}")
async def load_index(index_name: str):
    # First try JSON
    stocks = load_or_create_index(index_name)

    # Fallback mock data for missing index files
    if not stocks and "nifty" in index_name.lower():
        stocks = [("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "TCS")]

    tickers = [s[0] for s in stocks]
    pairs = {s[0]: s[1] for s in stocks}

    return {
        "index": index_name,
        "count": len(stocks),
        "tickers": tickers,
        "pairs": pairs,
    }


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_common(
    request: Request, symbol: str, index: str = "nifty50", horizon: str = None
):
    try:
        # 1. Trigger the analysis
        loop = asyncio.get_running_loop()
        analysis_data = await loop.run_in_executor(
            COMPUTE_EXECUTOR, run_full_analysis, symbol, index
        )
        full_report = analysis_data.get("full_report", {})
        # 2. Determine Active Horizon
        if horizon and horizon in full_report.get("profiles", {}):
            selected_profile_name = horizon
        else:
            selected_profile_name = full_report.get("best_fit", "short_term")

        # 3. Get the specific data for that profile
        profile_report = full_report.get("profiles", {}).get(selected_profile_name, {})
        analysis_data["profile_report"] = profile_report
        # 1. Grab indicators SPECIFIC to the requested button (selected_profile_name)
        #    This ensures Intraday button uses Intraday ATR/Patterns
        current_horizon_inds = analysis_data.get("raw_indicators_by_horizon", {}).get(
            selected_profile_name
        )

        # Fallback to flattened if missing (safety)
        if not current_horizon_inds:
            current_horizon_inds = analysis_data.get("indicators", {})

        # ... (Prepare standard summaries/scores variables) ...
        trade_plan = generate_trade_plan(
            profile_report,
            current_horizon_inds,
            analysis_data.get("macro_trend_status", "N/A"),
            horizon=selected_profile_name,
            strategy_report=analysis_data.get("strategy_report", {}),
            fundamentals=analysis_data.get("fundamentals", {}),
        )
        final_score = profile_report.get("final_score", 0)
        meta_scores = analysis_data.get("meta_scores", {})

        summary_payload = {
            **profile_report,
            "score": final_score * 10,
            "recommendation": profile_report.get("category", "HOLD"),
            "bull_signal": trade_plan.get("signal", "N/A"),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A"),
            "macro_close": analysis_data.get("macro_close"),
            "macro_index_name": index.upper(),
        }
        summaries = build_all_summaries(summary_payload)

        # 4. PASS 'selected_horizon' TO TEMPLATE
        context = {
            "request": request,
            "symbol": symbol,
            "current_index": index,
            "selected_horizon": selected_profile_name,
            "error": None,
            "full_report": full_report,
            "profile_report": profile_report,
            "fundamentals": analysis_data.get("fundamentals", {}),
            "indicators": analysis_data.get("indicators", {}),
            "meta_scores": {
                "value": meta_scores.get("value", 0) or 0,
                "growth": meta_scores.get("growth", 0) or 0,
                "quality": meta_scores.get("quality", 0) or 0,
                "momentum": meta_scores.get("momentum", 0) or 0,
            },
            "trade_recommendation": trade_plan,
            "summaries": summaries,
            "final_signal": profile_report.get("category", "HOLD"),
            "bull_signal": trade_plan.get("signal", "N/A"),
            "total_score": final_score * 10,
            "confidence": int(final_score * 10),
            "macro_index_name": index.upper(),
            "macro_trend_status": analysis_data.get("macro_trend_status", "N/A"),
            "macro_close": analysis_data.get("macro_close"),
            "reasons": [trade_plan.get("reason", "")],
            "strategy_report": analysis_data.get("strategy_report", {}),
        }
        return templates.TemplateResponse("result.html", context)

    except Exception as e:
        logger.exception(f"Error in analyze_common for {symbol}: {e}")
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "symbol": symbol,
                "current_index": index,  # Added to avoid missing var error
                "error": str(e),
                "full_report": {"profiles": {}},
                "profile_report": {},  # Fixed: Added missing key
                "indicators": {},
                "fundamentals": {},
                "summaries": {},
                "trade_recommendation": {},
                "selected_horizon": "short_term",
                "confidence": 0,
                "meta_scores": {},
                "strategy_report": {},
            },
        )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_post(
    request: Request, symbol: str = Form(...), index: str = Form("nifty50")
):
    return await analyze_common(request, symbol.strip().upper(), index)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/corporate_action_summary")
async def corporate_action_summary(payload: CorpActionsRequest):
    try:
        tickers = payload.tickers
        if not tickers:
            return JSONResponse({})

        summary = {}
        # ### PATCH FIX: Use API_EXECUTOR here too
        loop = asyncio.get_running_loop()

        # Wrapper for concurrent executor submission
        def fetch_acts(t):
            return t, get_corporate_actions([t], mode="upcoming", lookback_days=7)

        tasks = [loop.run_in_executor(API_EXECUTOR, fetch_acts, t) for t in tickers]

        # Wait for results
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
                                    ex_date = datetime.date.fromisoformat(ex_date_str)
                                    if ex_date >= today:
                                        valid_action = action
                                        break
                                except ValueError:
                                    pass

                        # ✅ Preserved User Logic: Formatting string exactly as before
                        if valid_action:
                            a = valid_action
                            typ = a.get("type", "").replace("Upcoming ", "")
                            amt = a.get("amount", "")
                            exd = a.get("ex_date", "")
                            parts = []
                            if "Dividend" in typ:
                                parts.append("Dividend Announced")
                            else:
                                parts.append(typ)
                            if amt:
                                parts.append(f"₹{amt}")
                            if exd:
                                parts.append(f"(Ex: {exd})")
                            summary[t] = " ".join(parts)

                except Exception:
                    pass
        return JSONResponse(summary)
    except Exception:
        return JSONResponse({})


@app.get("/corporate_actions")
async def corporate_actions(ticker: str):
    # ### PATCH FIX: Offload to API Executor
    loop = asyncio.get_running_loop()

    def _fetch():
        past = get_corporate_actions([ticker], mode="past", lookback_days=365)
        upcoming = get_corporate_actions([ticker], mode="upcoming", lookback_days=365)
        return past, upcoming

    past, upcoming = await loop.run_in_executor(API_EXECUTOR, _fetch)

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
    return templates.TemplateResponse(
        "Stock_Buy-sell_Flowcharts.html", {"request": request}
    )


@app.get("/flowchart_payload")
async def flowchart_payload(symbol: str, index: str = Query("NIFTY50")):
    try:
        # ### PATCH FIX: Heavy payload generation on API Executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            API_EXECUTOR, build_flowchart_payload, symbol, index
        )
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
