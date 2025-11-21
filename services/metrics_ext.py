"""
Production-ready services/metrics_ext.py
- Single, cleaned implementation with:
  * Modular metric helpers
  * Single thread pool executor for blocking calls
  * Correct Sharpe/Sortino with risk-free subtraction and annualization
  * Tight CF vs Net Profit scoring (always applied)
  * Unified macro scoring helpers and sanity checks
  * Centralized logging and cache handling

Dependencies expected:
- services.data_fetch: _retry, safe_float, safe_get
- services.world_bank_provider: get_macro_metrics
- yfinance installed

Notes:
- This file avoids blocking the event loop by running blocking calls on a shared ThreadPoolExecutor.
- Keep RISK_FREE_RATE configurable via env/config in your real deployment.
"""

import os
import time
import math
import asyncio
import nest_asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from services.world_bank_provider import get_macro_metrics
from services.data_fetch import _retry, safe_float, safe_get
from services import macro_filter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------
# Macro Trend Config <-- NEW SECTION
# --------------------
# Default to Nifty50. Can be configured via environment variable.
INDEX_NAME = os.getenv("MACRO_INDEX_NAME", "nifty50") 
TICKER_MAP = {
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
    "default": "^NSEI",
    "us_sp500": "^GSPC"
}

# --------------------
# Config
# --------------------
CACHE_TTL = int(os.getenv("METRICS_CACHE_TTL", "600"))
RISK_FREE_RATE = None  # will fetch dynamically below
_THREAD_POOL_SIZE = int(os.getenv("METRICS_THREAD_POOL", "6"))

# --------------------
# Cache + Executor
# --------------------
_metrics_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=_THREAD_POOL_SIZE)

# --------------------
# Utilities
# --------------------

def safe_div(a: Optional[float], b: Optional[float], default: Optional[float] = None) -> Optional[float]:
    try:
        if a is None or b is None:
            return default
        b = float(b)
        if abs(b) < 1e-12:
            return default
        return float(a) / b
    except Exception:
        return default


def score_metric(value: Optional[float], thresholds: list, reverse_score: bool = False) -> int:
    """
    Flexible scoring helper.

    Args:
        value: numeric value to score
        thresholds: list of (cond, score) where cond is a callable accepting value
        reverse_score: if True, reverses scoring direction (lower value = higher score)
                       useful for metrics like drawdown, volatility, recovery days, etc.

    Example:
        score_metric(15, [(lambda x: x < 20, 10), (lambda x: x < 30, 5)])          → 10
        score_metric(200, [(lambda x: x > 100, 10), (lambda x: x > 50, 5)], True)  → 10
    """
    if value is None or isinstance(value, str):
        return 0

    try:
        # Normal scoring: first condition that matches
        for cond, score in thresholds:
            if cond(value):
                return int(score)

        # If reverse scoring requested and no condition matched, flip order & try again
        if reverse_score:
            for cond, score in reversed(thresholds):
                if cond(value):
                    return int(score)

    except Exception:
        pass

    return 0



def normalize_score(value: Optional[float], min_val: float, max_val: float) -> int:
    try:
        if value is None or math.isnan(value):
            return 0
        v = float(value)
        v_clamped = max(min_val, min(max_val, v))
        return int(round(10 * (v_clamped - min_val) / (max_val - min_val)))
    except Exception:
        return 0


async def run_blocking(func: Callable, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


def annualize_return_from_daily_mean(daily_mean: float) -> Optional[float]:
    try:
        return (1 + daily_mean) ** 252 - 1
    except Exception:
        return None


def calc_sharpe_from_daily(returns: np.ndarray, risk_free_rate_annual: float = RISK_FREE_RATE) -> Optional[float]:
    try:
        if returns is None or len(returns) < 2:
            return None
        # risk-free daily
        rf_daily = risk_free_rate_annual / 252.0
        excess = returns - rf_daily
        mean_excess = np.mean(excess)
        std_daily = np.std(returns, ddof=0)
        if std_daily == 0:
            return None
        # annualize: mean_excess / std_daily * sqrt(252)
        return float(mean_excess / std_daily * math.sqrt(252))
    except Exception:
        return None


def calc_sortino_from_daily(returns: np.ndarray, risk_free_rate_annual: float = RISK_FREE_RATE) -> Optional[float]:
    try:
        if returns is None or len(returns) < 2:
            return None
        rf_daily = risk_free_rate_annual / 252.0
        excess = returns - rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0:
            return None
        downside_std = np.std(downside, ddof=0)
        if downside_std == 0:
            return None
        return float(np.mean(excess) / downside_std * math.sqrt(252))
    except Exception:
        return None


def calculate_market_cap_cagr(symbol: str, years: int = 5):
    """Estimate Market Cap CAGR (or Price CAGR proxy if share history unavailable)."""
    try:
        if years <= 0:
            return {"raw": None, "value": "Invalid period", "score": 0}

        # Fetch historical data with adjusted prices
        df = yf.download(
            symbol,
            period=f"{years+1}y",
            interval="1mo",
            auto_adjust=True,
            progress=False
        )

        # Defensive checks
        if df is None or getattr(df, "empty", True):
            return {"raw": None, "value": "N/A (no data)", "score": 0}

        # Handle potential multi-level columns (in rare cases)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        if "Close" not in df.columns:
            return {"raw": None, "value": "N/A (no Close data)", "score": 0}

        close = df["Close"].dropna().astype(float)
        if len(close) < 2:
            return {"raw": None, "value": "N/A (insufficient data)", "score": 0}

        beginning = float(close.iloc[0])
        ending = float(close.iloc[-1])

        if math.isnan(beginning) or math.isnan(ending) or beginning <= 0 or ending <= 0:
            return {"raw": None, "value": "N/A (invalid prices)", "score": 0}

        # CAGR computation
        cagr = (ending / beginning) ** (1 / years) - 1
        cagr_pct = round(cagr * 100, 2)

        # --- Scoring Logic ---
        if cagr_pct >= 25:
            score = 10
        elif cagr_pct >= 15:
            score = 8
        elif cagr_pct >= 5:
            score = 5
        elif cagr_pct > 0:
            score = 2
        else:
            score = 0

        return {
            "raw": cagr_pct,
            "value": f"{cagr_pct:.2f}% ({years}Y CAGR)",
            "score": score
        }

    except Exception as e:
        return {"raw": None, "value": f"Error: {str(e)}", "score": 0}


def score_cf_vs_profit(ratio: Optional[float]) -> int:
    if ratio is None:
        return 0
    deviation = abs(1.0 - ratio)
    if deviation <= 0.10:
        return 10
    if deviation <= 0.20:
        return 7
    if deviation <= 0.30:
        return 5
    if deviation <= 0.50:
        return 3
    return 0


def macro_score(value: Optional[float], good: float, ok: float, inverse: bool = False) -> int:
    if value is None:
        return 0
    try:
        v = float(value)
        if inverse:
            return 10 if v < good else (5 if v < ok else 0)
        else:
            return 10 if v >= good else (5 if v >= ok else 0)
    except Exception:
        return 0


# --------------------
# Main metric computation
# --------------------
async def compute_extended_metrics(ticker_symbol: str) -> Dict[str, Dict[str, Any]]:
    # 1) cache check (non-blocking)
    async with _cache_lock:
        cached = _metrics_cache.get(ticker_symbol)
        if cached and (time.time() - cached["ts"] < CACHE_TTL):
            logger.debug("[CACHE HIT] %s", ticker_symbol)
            return cached["data"]

    metrics: Dict[str, Dict[str, Any]] = {}

    try:
        # 2) fetch macro data on threadpool
        macro_data_raw = await run_blocking(lambda: get_macro_metrics())

        def _get_macro_value(key: str) -> Optional[float]:
            try:
                v = macro_data_raw.get(key)
                if not v:
                    return None
                val = v.get("value")
                if val is None or val == "N/A":
                    return None
                return safe_float(val)
            except Exception:
                return None
        
        # 3A️⃣ Dynamic Risk-Free Rate (from World Bank or RBI data)
        try:
            rf = _get_macro_value("10Y Govt Bond Yield (%)")
            if rf is not None and rf > 0:
                risk_free_dynamic = rf / 100
            else:
                risk_free_dynamic = 0.06
            logger.info(f"Dynamic Risk-Free Rate set to {risk_free_dynamic*100:.2f}%")
        except Exception as e:
            risk_free_dynamic = 0.06
            logger.warning(f"Failed to fetch dynamic RFR: {e}")

        # 3) fetch ticker (blocking) and history
        t = await run_blocking(lambda: _retry(lambda: yf.Ticker(ticker_symbol)))
        if not t:
            logger.error("Failed to initialize ticker for %s", ticker_symbol)
            return {}

        info = safe_get(vars(t), "info", {}) or {}
        financials = getattr(t, "financials", pd.DataFrame())
        balance_sheet = getattr(t, "balance_sheet", pd.DataFrame())
        cashflow = getattr(t, "cashflow", pd.DataFrame())
        earnings = getattr(t, "earnings", pd.DataFrame())
        analyst_ratings = getattr(t, "recommendations", pd.DataFrame())
        hist = await run_blocking(lambda: _retry(lambda: t.history(period="1y")))
        if hist is None:
            hist = pd.DataFrame()

        latest_financials = financials.iloc[:, 0] if not getattr(financials, "empty", True) else pd.Series()
        latest_cashflow = cashflow.iloc[:, 0] if not getattr(cashflow, "empty", True) else pd.Series()
        latest_balance_sheet = balance_sheet.iloc[:, 0] if not getattr(balance_sheet, "empty", True) else pd.Series()

        # ----- FUNDAMENTAL METRICS -----
        # EPS CAGR (try EPS column then fallback to Earnings)

        try:
            eps_series = pd.Series()
            
            # 1. Primary Attempt: Use the t.earnings DataFrame (if available)
            if isinstance(earnings, pd.DataFrame) and not earnings.empty:
                # Assuming earnings DataFrame has one row per year/quarter, and 'EPS' column exists
                col_name = next((c for c in earnings.columns if "eps" in c.lower() or "earn" in c.lower()), None)
                if col_name:
                    # Get the last 5 annual periods, ensure they are floats
                    eps_series = earnings[col_name].dropna().tail(5).apply(safe_float)
            
            # 2. Secondary Attempt (CRITICAL FIX): Use the financials DataFrame
            # Financials contains annual data where 'Diluted EPS' is a key row.
            if len(eps_series.dropna()) < 2:
                # Transpose financials to get years as columns and metrics as index
                if not financials.empty and 'Diluted EPS' in financials.index:
                    # We are using .loc to get the row, then .dropna().tail(5) to get the last 5 years.
                    eps_series = financials.loc['Diluted EPS'].dropna().tail(5).apply(safe_float)
                # Check for other common EPS keys in financials
                elif not financials.empty and 'Basic Average Shares' in financials.index and 'Net Income' in financials.index:
                    # Fallback to Net Income / Shares Outstanding (much less accurate, but a last resort)
                    net_income_series = financials.loc['Net Income'].dropna()
                    shares_series = financials.loc['Basic Average Shares'].dropna()
                    # Align indices and calculate EPS
                    temp_df = pd.DataFrame({'NI': net_income_series, 'Shares': shares_series}).dropna()
                    if len(temp_df) >= 2:
                        eps_series = (temp_df['NI'] / temp_df['Shares']).tail(5).apply(safe_float)

            # --- CAGR Calculation ---
            valid_eps = eps_series.dropna()
            
            if valid_eps.empty or len(valid_eps) < 2:
                metrics["EPS Growth Consistency (5Y CAGR)"] = {"value": "N/A", "score": 0}
            else:
                # Ensure the series is sorted chronologically if it's based on financials/earnings (should be).
                start, end = float(valid_eps.iloc[0]), float(valid_eps.iloc[-1])
                
                # Check for valid start/end for CAGR calculation (must be > 0)
                if start <= 0 or end <= 0:
                    metrics["EPS Growth Consistency (5Y CAGR)"] = {"value": "N/A", "score": 0}
                else:
                    n = len(valid_eps) - 1
                    cagr = ((end / start) ** (1 / n) - 1) * 100
                    metrics["EPS Growth Consistency (5Y CAGR)"] = {"value": round(cagr, 2),
                                                                 "score": score_metric(cagr, [(lambda x: x > 15, 10), (lambda x: x > 5, 5)])}
        except Exception as e:
            logger.debug("EPS CAGR failed: %s", e)
            metrics["EPS Growth Consistency (5Y CAGR)"] = {"value": "N/A", "score": 0}


        # Operating CF vs Net Profit (use tighter scoring always)
        try:
            op_cf = safe_float(latest_cashflow.get("Total Cash From Operating Activities") or latest_cashflow.get("Operating Cash Flow"))
            net_inc = safe_float(latest_financials.get("Net Income") or latest_financials.get("Net Income Applicable To Common Shares"))
            ratio = safe_div(op_cf, net_inc)
            val = round(ratio, 2) if ratio is not None else "N/A"
            metrics["Operating CF vs Net Profit"] = {"value": val, "score": score_cf_vs_profit(ratio)}
        except Exception as e:
            logger.debug("CF vs Net failed: %s", e)
            metrics["Operating CF vs Net Profit"] = {"value": "N/A", "score": 0}

        # R&D Intensity
        try:
            # 1. Primary Attempt: Use the info dictionary
            rd = safe_float(info.get("researchDevelopment"))
            rev = safe_float(info.get("totalRevenue"))
            
            # 2. Fallback for R&D: Check multiple income statement keys
            if rd is None or rd == 0:
                rd = safe_float(
                    # CRITICAL ADDITION: The fully spelled-out, capitalized key
                    latest_financials.get("Research And Development") or 
                    latest_financials.get("Research Development") or
                    latest_financials.get("Research and Development Expense") or
                    latest_financials.get("R&D") or
                    latest_financials.get("Rnd") or
                    latest_financials.get("R And D") or
                    latest_financials.get("Product Development")
                )
            
            # 3. Fallback for Revenue: Check multiple income statement keys
            if rev is None or rev == 0:
                rev = safe_float(
                    latest_financials.get("Total Revenue") or
                    latest_financials.get("Revenue") or
                    latest_financials.get("Total Operating Revenue")
                )
            
            # 4. Calculation
            if rd is not None and rd > 0 and rev is not None and rev > 0:
                pct = (rd / rev) * 100
                metrics["R&D Intensity (%)"] = {
                    "value": round(pct, 2), 
                    "score": score_metric(pct, [(lambda x: x > 5, 10), (lambda x: x > 2, 5)])
                }
            else:
                metrics["R&D Intensity (%)"] = {"value": "N/A", "score": 0}
        
        except Exception as e:
            logger.debug("R&D failed: %s", e)
            metrics["R&D Intensity (%)"] = {"value": "N/A", "score": 0}

        # Asset Turnover
        try:
            rev_val = rev if 'rev' in locals() and rev is not None else safe_float(latest_financials.get("Total Revenue") or latest_financials.get("Revenue"))
            assets = safe_float(latest_balance_sheet.get("Total Assets") or latest_balance_sheet.get("Assets"))
            ratio_at = safe_div(rev_val, assets)
            metrics["Asset Turnover Ratio"] = {"value": round(ratio_at, 2) if ratio_at is not None else "N/A",
                                               "score": score_metric(ratio_at, [(lambda x: x > 1, 10), (lambda x: x > 0.5, 5)])}
        except Exception as e:
            logger.debug("Asset turnover failed: %s", e)
            metrics["Asset Turnover Ratio"] = {"value": "N/A", "score": 0}

        # ----- RISK METRICS -----
        try:
            if not hist.empty and "Close" in hist.columns:
                daily = hist["Close"].pct_change().dropna()
                returns = daily.values

                sharpe = calc_sharpe_from_daily(returns, risk_free_rate_annual=risk_free_dynamic)
                sortino = calc_sortino_from_daily(returns, risk_free_rate_annual=risk_free_dynamic)


                rr = hist["Close"].dropna()
                if len(rr) > 0:
                    # --- Drawdown Computation ---
                    rr_peak = rr.cummax()
                    drawdown_series = 1 - (rr / rr_peak)
                    max_dd_decimal = drawdown_series.max()
                    max_dd = safe_float(max_dd_decimal * 100)

                    # --- Drawdown Recovery Period ---
                    recovery_days = None
                    trough_date = drawdown_series.idxmax()
                    prior_peak = rr[:trough_date].idxmax()
                    recovery_period_prices = rr[trough_date:]
                    recovery_dates = recovery_period_prices[recovery_period_prices >= rr[prior_peak]]

                    if not recovery_dates.empty:
                        recovery_date = recovery_dates.index[0]
                        recovery_days = (recovery_date - trough_date).days
                    else:
                        recovery_days = (rr.index[-1] - trough_date).days  # fallback if never recovered

                else:
                    max_dd = None
                    recovery_days = None

                # --- Calmar Ratio ---
                ann_return = annualize_return_from_daily_mean(daily.mean())
                calmar = safe_div(
                    ann_return * 100 if ann_return is not None else None,
                    abs(max_dd) if max_dd not in (None, 0) else None
                )

                # --- Metrics ---
                metrics["Sharpe Ratio"] = {
                    "value": round(sharpe, 2) if sharpe is not None else "N/A",
                    "score": normalize_score(sharpe, -1.0, 3.0)
                }
                metrics["Sortino Ratio"] = {
                    "value": round(sortino, 2) if sortino is not None else "N/A",
                    "score": normalize_score(sortino, -1.0, 3.0)
                }
                metrics["Max Drawdown (%)"] = {
                    "value": round(max_dd, 2) if max_dd is not None else "N/A",
                    "score": score_metric(max_dd, [
                        (lambda x: x < 20, 10),
                        (lambda x: x < 30, 5)
                    ])
                }
                metrics["Calmar Ratio"] = {
                    "value": round(calmar, 2) if calmar is not None else "N/A",
                    "score": normalize_score(calmar, -1.0, 2.0)
                }
                metrics["Drawdown Recovery Period"] = {
                    "value": recovery_days if recovery_days is not None else "N/A",
                    "score": score_metric(recovery_days, [
                        (lambda x: x <= 90, 10),
                        (lambda x: x <= 180, 7),
                        (lambda x: x <= 365, 3)
                    ], reverse_score=True)
                }
                metrics["Max Drawdown Dates"] = {
                    "value": f"{prior_peak.date()} → {trough_date.date()}",
                    "score": 0
                }

            else:
                for k in [
                    "Sharpe Ratio", "Sortino Ratio",
                    "Max Drawdown (%)", "Calmar Ratio",
                    "Drawdown Recovery Period"
                ]:
                    metrics[k] = {"value": "N/A", "score": 0}

        except Exception as e:
            logger.debug("Risk metrics failed: %s", e)
            for k in [
                "Sharpe Ratio", "Sortino Ratio",
                "Max Drawdown (%)", "Calmar Ratio",
                "Drawdown Recovery Period"
            ]:
                metrics[k] = {"value": "N/A", "score": 0}

        # ----- OWNERSHIP & SENTIMENT -----
        try:
            promoter_pct = safe_float(info.get("insidersPercentHeld"))
            inst_pct = safe_float(info.get("institutionsPercentHeld"))
            promoter_pct = promoter_pct * 100 if promoter_pct is not None else None
            inst_pct = inst_pct * 100 if inst_pct is not None else None
            metrics["Promoter Holding (%)"] = {"value": round(promoter_pct, 2) if promoter_pct is not None else "N/A", "score": score_metric(promoter_pct, [(lambda x: x > 40, 10), (lambda x: x > 20, 5)])}
            metrics["Institutional Ownership (%)"] = {"value": round(inst_pct, 2) if inst_pct is not None else "N/A", "score": score_metric(inst_pct, [(lambda x: x > 30, 10), (lambda x: x > 10, 5)])}
        except Exception as e:
            logger.debug("Ownership failed: %s", e)
            metrics["Promoter Holding (%)"] = {"value": "N/A", "score": 0}
            metrics["Institutional Ownership (%)"] = {"value": "N/A", "score": 0}

        # Analyst ratings
        try:
            if analyst_ratings is not None and not analyst_ratings.empty:
                latest_row = analyst_ratings.iloc[-1] 

                # Safely extract all rating counts
                period = latest_row.get('period')
                strong_buy_count = float(latest_row.get('strongBuy', 0) or 0)
                buy_count = float(latest_row.get('buy', 0) or 0)
                hold_count = float(latest_row.get('hold', 0) or 0)
                sell_count = float(latest_row.get('sell', 0) or 0)
                strong_sell_count = float(latest_row.get('strongSell', 0) or 0)

                # Calculate total recommendations and total 'Buy' type
                total_recommendations = strong_buy_count + buy_count + hold_count + sell_count + strong_sell_count
                total_buy_type = strong_buy_count + buy_count

                if total_recommendations > 0:
                    buy_percentage = (total_buy_type / total_recommendations) * 100

                    # --- Scoring Logic for Analyst Buy Percentage ---
                    # Score is based on the percentage of analysts recommending Strong Buy or Buy
                    if buy_percentage >= 80: # Very strong consensus
                        score = 10
                    elif buy_percentage >= 60: # Solid majority buy
                        score = 8
                    elif buy_percentage >= 40: # Balanced/Slightly bullish
                        score = 5
                    elif buy_percentage >= 20: # Slightly bearish/High hold count
                        score = 2
                    else: # Strong sell consensus
                        score = 0
                    
                    # Store the result in the metrics dictionary
                    metrics["Analyst Ratings"] = {
                        "raw": period,
                        "value": f"{buy_percentage:.2f}% Buy",
                        "score": score
                    }
                else:
                    # Handle case where counts are zero (but dataframe wasn't empty)
                    metrics["Analyst Ratings"] = {"raw": None, "value": "N/A (No Recommends)", "score": 0}

            else:
                # Handle case where the dataframe is None or empty
                metrics["Analyst Ratings"] = {"raw": None, "value": "N/A", "score": 0}

        except Exception as e:
            logger.debug("Analyst ratings failed: %s", e)
            # Ensure a value is always set upon failure
            metrics["Analyst Ratings"] = {"value": "N/A", "score": 0}

        # market cap cagr
        try:
            cgr = calculate_market_cap_cagr(ticker_symbol)
            metrics["Market Cap CAGR"] = {
                "raw": cgr["raw"],
                "value": cgr["value"],
                "score": cgr["score"]
            }
                    
        except Exception as e:         
            logger.debug("Market Cap CAGR failed: %s", e)
            metrics["Market Cap CAGR"] = {"value": "N/A", "score": 0}

        # ----- MACRO & MARKET -----
        try:
            vix_t = await run_blocking(lambda: _retry(lambda: yf.Ticker("^INDIAVIX")))
            vix_close = safe_float(safe_get(getattr(vix_t, "info", {}), "regularMarketPrice")) if vix_t else None
            if vix_close is not None:
                # Contrarian scoring: panic → bullish, complacency → bearish
                if vix_close >= 25:
                    score = 10
                elif vix_close >= 20:
                    score = 7
                elif vix_close >= 15:
                    score = 5
                elif vix_close >= 12:
                    score = 2
                else:
                    score = 0
            else:
                score = 0
            metrics["VIX (Volatility Index)"] = {
                "value": round(vix_close, 2) if vix_close else "N/A",
                "score": score,
            }
        except Exception as e:
            logger.debug("VIX failed: %s", e)
            metrics["VIX (Volatility Index)"] = {"value": "N/A", "score": 0}


        try:
            gdp_growth = _get_macro_value("GDP Growth (%)")
            metrics["GDP Growth (%)"] = {"value": round(gdp_growth, 2) if gdp_growth is not None else "N/A", "score": macro_score(gdp_growth, 7.0, 5.0)}
        except Exception as e:
            logger.debug("GDP failed: %s", e)
            metrics["GDP Growth (%)"] = {"value": "N/A", "score": 0}

        try:
            inflation = _get_macro_value("Inflation Rate (%)")
            metrics["Inflation Rate (%)"] = {"value": round(inflation, 2) if inflation is not None else "N/A", "score": macro_score(inflation, 4.0, 6.0, inverse=True)}
        except Exception as e:
            logger.debug("Inflation failed: %s", e)
            metrics["Inflation Rate (%)"] = {"value": "N/A", "score": 0}

        try:
            inr_rate = _get_macro_value("USD/INR")
            metrics["Currency Trend (USD/INR)"] = {"value": round(inr_rate, 2) if inr_rate is not None else "N/A",
                                                    "score": macro_score(inr_rate, 75.0, 80.0, inverse=True)}
        except Exception as e:
            logger.debug("Currency failed: %s", e)
            metrics["Currency Trend (USD/INR)"] = {"value": "N/A", "score": 0}

        try:
            crude = _get_macro_value("Crude Oil ($)")
            metrics["Crude Oil ($)"] = {"value": round(crude, 2) if crude is not None else "N/A", "score": macro_score(crude, 60.0, 80.0, inverse=True)}
        except Exception as e:
            logger.debug("Crude failed: %s", e)
            metrics["Crude Oil ($)"] = {"value": "N/A", "score": 0}

        # ----------------------------------------------------
        # 🟢 Macro Trend Check (from macro_filter)
        # ----------------------------------------------------
        try:
            loop = asyncio.get_running_loop()
            macro = await loop.run_in_executor(
                _executor,
                lambda: macro_filter.check_macro_trend(INDEX_NAME, TICKER_MAP)
            )

            # Extract safe fields
            idx_name = macro.get("name", INDEX_NAME.upper())
            trend = macro.get("trend", "N/A")
            last_close = macro.get("last_close", "N/A")
            score = macro.get("score", 0)
            confidence = macro.get("confidence", 0.0)
            sma_value = macro.get("sma_value")

            # Add formatted trend and score to metrics
            metrics[f"{idx_name} Trend Status"] = {
                "value": trend,
                "score": score,
                "raw": trend,
                "confidence": confidence
            }

            metrics[f"{idx_name} Last Close"] = {
                "value": last_close,
                "score": 0,
                "raw": sma_value if sma_value is not None else "N/A"
            }

            # Optional: derived indicator for UI clarity
            metrics[f"{idx_name} Macro Confidence (%)"] = {
                "value": f"{confidence}%",
                "score": 0,
                "raw": confidence
            }

            logger.info(f"✅ Macro trend computed for {idx_name}: {trend} (Confidence {confidence:.2f}%)")

        except Exception as e:
            logger.warning(f"⚠️ Macro Trend check failed (ignored): {e}")
            resolved_index = INDEX_NAME.upper()
            metrics[f"{resolved_index} Trend Status"] = {"value": "N/A (Error)", "score": 0, "raw": "N/A"}
            metrics[f"{resolved_index} Last Close"] = {"value": "N/A", "score": 0, "raw": "N/A"}
            metrics[f"{resolved_index} Macro Confidence (%)"] = {"value": "N/A", "score": 0, "raw": "N/A"}

        # ----- Cache store -----
        async with _cache_lock:
            _metrics_cache[ticker_symbol] = {"ts": time.time(), "data": metrics}
            logger.debug("Cached metrics for %s", ticker_symbol)

    except Exception as e:
        logger.exception("compute_extended_metrics failed for %s: %s", ticker_symbol, e)

    return metrics

def compute_extended_metrics_sync(symbol: str):
    """
    Safe synchronous wrapper for compute_extended_metrics.
    Works even when called from within a running event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop — schedule task and block until done
        nest_asyncio.apply()
        return loop.run_until_complete(compute_extended_metrics(symbol))
    else:
        # No event loop running
        return asyncio.run(compute_extended_metrics(symbol))



# Quick smoke when run as script
if __name__ == "__main__":
    import asyncio

    async def main():
        res = await compute_extended_metrics("RELIANCE.NS")
        for k, v in res.items():
            print(k, v)

    asyncio.run(main())
