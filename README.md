Python Stock Signal Engine
This is a Python stock analysis app that generates 'buy/hold/sell' signals. It works by fetching technical and fundamental data, scoring it against predefined investment profiles (like 'intraday' or 'long_term'), and generating a final signal.

🎯 Core Aim & Architecture
The primary goal of this project is to solve the multi-timeframe analysis problem.
An intraday trader and a long-term investor need different metrics, but they also share metrics that must be calculated on different timeframes. For example:
An intraday trader needs a 15-minute VWAP.
A long-term investor needs a 1-day 200 DMA.
An intraday trader also needs to know the 1-day 200 DMA for context.
This engine is architected to correctly select, fetch, and calculate the right metric on the right timeframe for the right profile, all on demand.

🗺️ What The Files Do
The project is broken down into a "separation of concerns" architecture:

1. config/constants.py (The "Brain")
HORIZON_PROFILE_MAP: This is the most important part. It's a dictionary that defines the scoring logic for each profile (e.g., "intraday"), specifying which metrics to use and their weights.
INDICATOR_METRIC_MAP: This is the master "routing table." It maps a metric key (e.g., "adx") to its calculation function (compute_adx) and, crucially, its required data horizon (e.g., "short_term" for daily data).

2. services/data_fetch.py (The "Engine & Tools")

Provides data fetching functions like get_history_for_horizon (which reads from HORIZON_FETCH_CONFIG) and the core fetch_data.
Contains the universal _wrap_calc helper. This is a robust decorator that standardizes all function outputs (both single and multi-metric) into a safe, JSON-serializable {"value": ..., "score": ..., "desc": ...} format.

3. services/fundamentals.py (The "Fundamental Library")

A collection of calc_... functions (e.g., calc_pe_ratio, calc_roe) that calculate individual fundamental metrics.
This file is orchestrated by its own METRIC_FUNCTIONS registry.

4. services/indicators.py (The "Technical Library & Orchestrator")

This is the core of the project.
Part 1 (Library): Contains all the individual compute_... functions (e.g., compute_rsi, compute_bollinger_bands) that calculate technical indicators from a dataframe.
Part 2 (Orchestrator): Contains the master compute_indicators() function. This is the "smart" function that drives the logic.

5. services/signal_engine.py (The "Decision Maker")

compute_profile_score(): This is the main scoring function. It takes the data from the calculators and applies the rules from HORIZON_PROFILE_MAP to generate the final final_score and category ("BUY", "HOLD", "SELL").
enrich_hybrid_metrics(): (Optional) Creates advanced factors by blending technical and fundamental data (e.g., volatility_adjusted_roe).

🔄 How It Works: The 3-Step Scoring Flow
The system is designed to generate a score for a single profile (e.g., "short_term") by combining data from three separate, specialized components.

Here is the end-to-end flow to get a score for RELIANCE.NS on the "short_term" profile:

Step 1: fundamentals.py (The Fundamental Engine)
First, the system calls compute_fundamentals("RELIANCE.NS").
This function is called only once per stock, as fundamentals are timeframe-agnostic.
It fetches all necessary data (info, financials, balance sheet) once.
It runs all calc_... functions (like calc_pe_ratio, calc_roe) and returns a single, large dictionary: fundamentals = {"roe": {...}, "pe_ratio": {...}, ...}

Step 2: indicators.py (The Technical Engine)
Next, the system calls compute_indicators("RELIANCE.NS", horizon="short_term").
This is the "smart" orchestrator that handles multi-timeframe logic.
It looks up "short_term" in HORIZON_PROFILE_MAP to get the list of required metrics (e.g., ["supertrend_signal", "adx", "price_vs_200dma_pct", ...]).
It loops only over these required metrics.
For each metric (e.g., "supertrend_signal"):
It looks up "supertrend_signal" in INDICATOR_METRIC_MAP to find its rule: {"func": compute_supertrend, "horizon": "short_term"}.
It sees "horizon": "short_term" (daily) and calls compute_supertrend(df_daily).
For another metric (e.g., "vwap_bias", if requested by an intraday profile):
It would look up "vwap_bias" in INDICATOR_METRIC_MAP.
It would find the rule: {"func": compute_vwap, "horizon": "intraday"}.
It would correctly fetch the 15-minute data and call compute_vwap(df_intraday).
It returns a single dictionary of all technical metrics for that profile: indicators = {"supertrend_signal": {...}, "adx": {...}, ...}

Step 3: signal_engine.py (The Decision Maker)
Finally, the system combines all the data by calling compute_profile_score(...).
This function receives both the fundamentals and indicators dictionaries.
It looks up the "short_term" profile in HORIZON_PROFILE_MAP to get the scoring weights (e.g., {"supertrend_signal": 0.12, "quarterly_growth": 0.05, ...}).
It then loops through these weights, pulling the corresponding metric from either the indicators dict or the fundamentals dict.
It calculates the base_score, applies penalties, and returns the final report: {"final_score": 7.2, "category": "BUY", ...}

🚀 How to Use (Example)
This architecture is designed to be called profile-by-profile.

Python

from services.indicators import compute_indicators
from services.fundamentals import compute_fundamentals
from services.signal_engine import compute_profile_score
from config.constants import HORIZON_PROFILE_MAP
import json

# --- Define what you want to analyze ---
SYMBOL = "RELIANCE.NS"
PROFILE_NAME = "short_term" # or "intraday", "long_term", "multibagger"

# 1. Calculate all fundamental data (this is timeframe-agnostic)
print(f"Calculating fundamentals for {SYMBOL}...")
fundamentals = compute_fundamentals(SYMBOL)

# 2. Calculate technical indicators for the *specific* profile
# This is the "smart" function that handles all routing.
print(f"Calculating '{PROFILE_NAME}' indicators for {SYMBOL}...")
indicators = compute_indicators(SYMBOL, horizon=PROFILE_NAME)

# 3. Run the signal engine to get the final score for that profile
print("Running signal engine...")
profile_report = compute_profile_score(
    profile_name=PROFILE_NAME,
    fundamentals=fundamentals,
    indicators=indicators
)

# 4. View the final report
print(json.dumps(profile_report, indent=2))

--------------------------------------------------------------------------------------------------------------------------------------

Well-Organized Recommendations Per File:
Based on our comprehensive review, here's a prioritized action plan organized by file:
📄 README.md
🔧 Immediate Updates
Fix typos: "breifly" → "briefly" and scan for other spelling errors
Add Requirements section: Document Python version, core dependencies (yfinance, pandas, numpy, fastapi, pydantic, etc.)
Add Running Tests section: If tests exist, document how to run them with examples
Add architecture diagram: Visual representation of the Brain → Engine → Decision Maker flow
🏗️ Architectural & Cross-Cutting Concerns
🔥 Critical (Fix Before Production)
Implement proper error handling: Add try/except in _wrap_calc decorator to return {"value": None, "score": 0, "desc": "Error: ..."} on failure
Validate profile weights: On startup, ensure all HORIZON_PROFILE_MAP weights sum to 1.0 (with auto-rebalancing or error)
Add startup configuration validation: Verify all metric keys in profiles exist in INDICATOR_METRIC_MAP and METRIC_FUNCTIONS
Implement data freshness validation: Add timestamp checks to detect stale yfinance data and log warnings
⚠️ High Priority
Separate fundamentals caching: Cache fundamentals for 24 hours (quarterly data) vs technical data (5 min - 1 hour)
Implement graceful degradation: Ensure single metric failure doesn't crash entire score calculation
Standardize metric structure: Use consistent dict format {"weight": 0.12, "direction": "normal"} for all metrics
Add comprehensive logging: Include score breakdowns per metric for debugging
Add health check endpoint: /health returning cache size, threads, timestamp
🔧 Medium Priority
Migrate to async/await: Refactor yfinance calls to use asyncio.to_thread or async libraries
Build backtesting framework: Validate signals against historical data (e.g., March 2020, 2021 bull run)
Add plugin architecture: Create self-registering metric system to avoid editing multiple files when adding metrics
Create CONTRIBUTING.md: If others will contribute, document development setup and guidelines
🎛️ main.py
🔥 Critical (Memory & Stability)
Replace in-memory cache: Use cachetools.TTLCache(maxsize=1000, ttl=3600) instead of plain Dict to prevent OOM
Use global ThreadPoolExecutor: Move executor to module level with lifespan management:
Python
Copy
EXECUTOR = ThreadPoolExecutor(max_workers=20)
@app.on_event("shutdown")
def shutdown(): EXECUTOR.shutdown(wait=True)
Add Pydantic models: Replace Dict = Body(...) with proper request/response models for type safety and auto-docs
⚠️ High Priority
Fix race conditions: Implement cache-aside pattern with per-symbol locks to prevent duplicate computations
Add input sanitization: Validate symbols against regex ^[A-Z\.]{3,20}$ and reject malformed input
Add rate limiting: Implement @limiter.limit("30/minute") on /quick_scores endpoint
Implement shared data context: Pass single shared_data dict to prevent redundant fetches across functions
🔧 Medium Priority
Standardize return types: Use JSONResponse consistently; let frontend handle rendering
Add timing decorators: @timed("metric_name") to log performance of compute_indicators, compute_fundamentals
Refactor deep nesting: Break run_full_analysis into smaller functions: fetch_core_data(), compute_scores(), build_trade_plan()
Add startup validation: @app.on_event("startup") to validate profiles, weights, and mappings
⚙️ constants.py
🔥 Critical (Data Integrity)
Fix PEG ratio calculation: Divide PE by growth rate as decimal (profit_g / 100), not percentage
Fix ROIC tax assumption: Derive actual tax rate from financials instead of hardcoding 25%
Fix market cap currency: Normalize to consistent units (crores) for all Indian stocks
Remove duplicate definitions: Delete duplicate STOCH_THRESHOLDS and consolidate ATR_MULTIPLIERS
⚠️ High Priority
Auto-generate flowchart_mapping: Create programmatic generation from TECHNICAL_METRIC_MAP and FUNDAMENTAL_ALIAS_MAP, then add special cases manually
Dynamic sector PE mapping: Fetch sector averages from external source with fallback to static values
Add version constant: CONFIG_VERSION = "2.1.0" for environment management
🔧 Medium Priority
Centralize magic numbers: Move constants like QUICK_SCORES_MAX_WORKERS, CONFIDENCE_MULTIPLIER to top of file
Standardize all metric weights: Ensure every metric uses dict structure, even if simple
📊 data_fetch.py
🔥 Critical (Cache Safety)
Normalize cache keys: Uppercase/strip symbols in @lru_cache and return immutable dict copies
Add strict mode: Replace silent return pd.DataFrame() with optional exception raising for debugging
Implement data freshness checks: Validate last timestamp in DataFrame against max_age_minutes parameter
⚠️ High Priority
Batch CSV parsing: Use ThreadPoolExecutor to fetch safe_info for all symbols in parallel during index parsing
Add proper error types: Create DataFetchError exception instead of generic Exception
Add retry logging: Log each retry attempt with symbol and failure reason
🔧 Medium Priority
Add type hints: Include return types like -> pd.DataFrame for all functions
Document horizon logic: Add docstring explaining get_history_for_horizon timeframe selection
💰 fundamentals.py
🔥 Critical (Calculation Accuracy)
Fix PEG ratio: Use decimal growth rate: profit_g_decimal = profit_g / 100.0 if profit_g > 2 else profit_g
Fix ROIC tax: Calculate effective tax rate from actual tax expense/EIT
Fix market cap comparison: Ensure currency units match across all ratios (P/S, P/B)
⚠️ High Priority
Consistent missing data handling: All calc_ functions should return None for missing data, letting _wrap_calc handle scoring
Improve dividend yield fallback: Use info.get("dividendYield") if available before calculating manually
Add sector normalization: Map yfinance's varied sector names to standard categories
🔧 Medium Priority
Cache intermediate results: Store base_data["fields"] in module-level TTL cache
Add unit tests: Test each calc_ function with mock yfinance responses
Document fallback chains: Comment the priority order (info → financials → balance sheet)
📈 indicators.py
🔥 Critical (Signal Reliability)
Fix Ichimoku calculation: Use manual calculation only with proper shifting to avoid lookahead bias and NaN values
Replace RSI slope: Remove non-standard indicator; replace with RSI2 (2-period RSI) for mean-reversion signals
Fix volume spike scoring: Score based on deviation from 1.0: score = 10 if spike_ratio > 1.5 else 7 if spike_ratio > 1.2 else 5
⚠️ High Priority
Implement module-level caching: Replace per-call _LocalFetchCache with shared TTLCache(maxsize=100, ttl=300)
Pre-fetch benchmark data: Fetch all needed benchmarks once at start of compute_indicators
Fix PSAR trend logic: Only use PSAR as stop-loss if trend is Bullish and PSAR < price
🔧 Medium Priority
MACD bundle optimization: Ensure MACD_12_26_9 calculation prevents redundant signal/histogram fetches
Add parameter validation: Check that length parameters are positive and DataFrame has sufficient rows
Standardize return format: Ensure all functions return consistent {"metric_name": {"value": ..., "score": ..., "desc": ...}}
🎯 signal_engine.py
🔥 Critical (Scoring Logic)
Fix penalty application: Use multiplicative, not additive: final_score = base_score * (1.0 - penalty_total)
Fix division by zero: In _compute_weighted_score, return (0.0, 1.0, {}) if weight_sum == 0
Remove direction inversion: Invert scores at metric source (e.g., calc_pe_ratio), not in scoring loop
Fix PSAR stop-loss: Check psar_trend == "Bullish" before applying PSAR level
⚠️ High Priority
Make macro trend a penalty: Replace rigid filter with penalty = 0.2 if "Downtrend" in macro_trend_status else 0
Add short-selling logic: Generate both long and short trade plans based on trend
Validate hybrid metrics: Remove duplicate calculations; compute each hybrid once and reuse
🔧 Medium Priority
Penalty capping: Keep min(penalty_total, 1.0) but document the rationale
Add score confidence intervals: Calculate standard deviation of metric scores
Document trade plan logic: Add comments explaining stop/target calculation strategies
📋 Implementation Priority Summary
1: Fix Critical Bugs (All Files)
signal_engine.py: Multiplicative penalties, division by zero protection
fundamentals.py: PEG ratio, ROIC tax, market cap currency
indicators.py: Ichimoku calculation, RSI slope removal
main.py: Replace in-memory cache, global executor
Add data freshness validation across all services
2: Configuration & Validation
constants.py: Auto-generate flowchart_mapping, dynamic sector PE
main.py: Add Pydantic models, startup validation, health endpoint
data_fetch.py: Batch CSV parsing, strict error mode
Add unit tests for 5 core metrics (P/E, RSI, SuperTrend, ROE, Volume Spike)
3: Performance & Scale
main.py: Implement shared data context, async fetching
indicators.py: Module-level caching, pre-fetch benchmarks
Add rate limiting and input sanitization
Add timing decorators and performance logging
4: Developer Experience & Polish
Generate plugin architecture skeleton
README.md: Add requirements, tests section, architecture diagram
Standardize return types and add docstrings
Create CONTRIBUTING.md and backtesting framework stub