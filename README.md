# 📈 Pro Stock Analyzer & Trading Engine

A professional-grade, end-to-end **Algorithmic Trading System** built with a **Hybrid Data Architecture**. A modular, end-to-end **Stock Analysis + Trade Decision System** built using FastAPI, AG Grid, pandas, yfinance, and custom scoring logic. It combines high-frequency technical analysis, fundamental screening, and smart signal logic into a blazing-fast dashboard for Indian Stocks (NSE).The app transforms raw market + fundamental data into **actionable trading signals** with risk management, horizon-based scoring, confluence detection, and trade planning.

---

## 🚀 Key Differentiators

### **1. Hybrid "Data Lakehouse" Architecture**
Unlike basic scripts that hit API rate limits, this engine uses a split-storage strategy:
- **Time-Series Data (OHLCV):** Stored in local **Apache Parquet** files (Data Lake).
  - *Benefit:* 50x faster loading than CSV, zero-lag charts, and bypasses API limits.
  - *Optimization:* Loads ONLY the needed rows (e.g., last 400 for swing) via **Polars LazyFrame**, preventing RAM bloat.
- **Relational Data & Fundamentals:** Stored in **SQLite** (`trade.db`).
  - *Benefit:* Persists trade history and caches heavy Fundamental JSON blobs (P/E, ROE) to prevent redundant API calls.

### **2. Smart Signal Logic (Phase 2 Engine)**
It doesn't just say "BUY". It classifies the *nature* of the move using priority logic:
- **🚀 MOMENTUM BREAKOUT:** High RSI + Volume Expansion + Band Break.
- **⚓ TREND PULLBACK:** Dip to 20/50 EMA in a long-term uptrend.
- **🔥 VOLATILITY SQUEEZE:** TTM Squeeze firing (Bollinger inside Keltner).
- **🧲 MEAN REVERSION:** Oversold bounce with volume confirmation.

### **3. GIL-FREE MULTIPROCESSING ENGINE**
Heavy technical analysis (RSI, EMA clouds, VWAP, MACD, ATR, Supertrend, etc.) now runs inside a **ProcessPoolExecutor**, allowing true parallel CPU execution.

**Benefits:**
- ✅ No UI freezes during heavy scans
- ✅ No blocking of the FastAPI request loop
- ✅ Full CPU utilization across all cores
- ✅ Seamless scanning of 500+ symbols

### **4. Execution-Ready Tools**
- **Position Sizer:** Sticky footer calculates exact share quantity based on ₹ Risk.
- **Tactical Hints:** Displays specific entry zones, targets, and "Move-to-Breakeven" rules.
- **Trade Logging:** "Mark Active" button persists trades to the database for review.

---

---

## 🏗️ System Architecture

| Layer | Key Files | Function |
|:---|:---|:---|
| **Persistence** | `db.py`, `data_layer.py` | **Hybrid Engine**: Parquet (Time-Series) + SQLite (Signal/Fundamental Cache). |
| **Data Layer** | `data_fetch.py` | **3-Tier Cache**: RAM → Disk (Parquet) → Yahoo. Implements Smart Freshness & TTL. |
| **Corp Actions** | `corporate_actions.py` | **Hybrid Scraper**: Scrapes MoneyControl for Upcoming actions (Zero API cost), uses Yahoo for History. |
| **Computation** | `indicators.py` | **Batch Processor**: Fetches horizon data *once* per stock and computes 20+ indicators in memory. |
| **Fundamentals** | `fundamentals.py` | **DB-Cached**: Stores raw financial statements in SQLite to strictly limit API calls to once per 24h. |
| **Logic Core** | `signal_engine.py` | **The Brain**: Weighted scoring, Priority Setup Classification, 3-Factor Confidence Algorithm. |
| **Orchestrator** | `main.py` | FastAPI app. Manages **Smart Index Mapping** and Separate Executors (API vs Compute). |
| **Frontend** | `index.html` | Interactive AG Grid dashboard + Jinja2 templates. |
---

## 🧠 Core Concepts & Logic

### **Horizon Profiles**
The engine evaluates every stock across 4 distinct time horizons simultaneously. Defined in `constants.py`.
- **Intraday:** 5m-15m data (Scalping)
- **Short Term:** Daily data (Swing Trading)
- **Long Term:** Weekly data (Trend Following)
- **Multibagger:** Monthly data (Fundamental Investing)

### **The Decision Engine**
Each stock is evaluated using a funnel:
1. **Raw Metrics** → **Normalization (0–10 scale)**
2. **Weighted Score** → **Penalties & Bonus Adjustments**
3. **Setup Classification** (Priority Queue)
4. **3-Factor Confidence Model**

#### **A. Setup Classification (Priority Queue)**
The engine prioritizes the most dominant behavior to avoid conflicting signals:
1.  **MOMENTUM BREAKOUT (Priority 100):** Price > BB Upper + High RSI + Volume Expansion.
2.  **VOLATILITY SQUEEZE (Priority 90):** Bollinger Bands inside Keltner Channels.
3.  **TREND PULLBACK (Priority 75):** Dip to **20 EMA** (Shallow) or **50 EMA** (Deep) while above 200 EMA.
4.  **OVERSOLD BOUNCE (Priority 60):** RSI < 30 with Volume Climax.
5.  **TREND FOLLOWING (Priority 40):** Simple alignment of Price > EMA20 > EMA50.

#### **B. 3-Factor Confidence Model**
Every signal gets a confidence score (0-100%) based on confluence:
* **Trend (40%):** Is price above 200 EMA? Is ADX > 25?
* **Momentum (40%):** Is MACD rising? Is RSI Slope positive? Is Price > VWAP?
* **Volume (20%):** Is Relative Volume (RVOL) > 1.5? Is OBV confirming price?
* **Macro Penalty:** If Market (Nifty) is Bearish, total confidence is slashed by **15%**.

#### **C. Custom Hybrid Metrics**
Unique metrics calculated in `signal_engine.py`:
* **Volatility-Adjusted ROE:** (ROE / ATR%) — Finds stable compounders.
* **Trend Consistency:** Combines ADX + Supertrend status.
* **Price vs Intrinsic Value:** Graham-style valuation check relative to current price.

### **Trade Planning Engine**
Found inside `signal_engine.py`, it generates specific execution parameters:
- **Entry Price:** Calculated via Pivot Points or VWAP.
- **Stop Loss (SL):** Dynamic volatility-based SL (ATR Multiplier or PSAR).
- **Targets:** Target 1 (1R Conservative) and Target 2 (Pivot Resistance).
- **Risk/Reward Ratio (RR):** Calculated dynamically based on Entry/SL.
- **Execution Hints:** Text instructions (e.g., "Move SL to Breakeven after T1").

---

## 🧩 Smart Workflows

### **1. 3-Tier Caching Strategy**
Implemented in `data_fetch.py` to ensure sub-millisecond response times:
1.  **L1 (RAM):** Instant access using `LRUCache` (TTL: 15m Intraday / 6h Daily).
2.  **L2 (Disk/Parquet):** Persistent storage handled by `ParquetStore`.
3.  **L3 (Source API):** Yahoo Finance (only hit if L1 and L2 miss).

### **2. Smart Benchmarking**
- **Auto-Detection:** The system automatically maps stocks to their "Home Index" (e.g., `INFY` → `Nifty IT`, `SUZLON` → `Smallcap 100`).
- **Relative Strength:** Calculates performance against the *relevant* benchmark, preventing false "Underperformance" signals when a Smallcap lags behind Nifty 50 but beats other Smallcaps.

### **3. The "Morning Routine"**
- On startup, the **Cache Warmer** runs in the background.
- It prioritizes **Nifty 50** stocks to ensure the market leaders are ready instantly.
- Fetches fresh data from Yahoo → Saves compressed **Parquet** files.

### **4. Quick Market Scan (The Grid)**
- **Endpoint:** `/quick_scores` (POST)
- **Function:** Returns lightweight summary for the AG Grid.
- **Optimization:** Reads directly from **Local Parquet/SQLite Cache** (Sub-10ms response).

### **5. Full Deep Analysis (The Dashboard)**
- **Endpoint:** `/analyze?symbol={SYMBOL}` (GET)
- **Function:** Returns full breakdown:
  - Metric-wise scores & penalties.
  - Detailed Trade Plan with Targets/SL.
  - Fundamental health check.
  - Interactive Profile Switcher (Intraday vs Long Term).

### **6. Corporate Actions Architecture**
- **BULK MODE:** Uses ONLY **Equitymaster** (Zero YFinance calls). Safe for scanning 1500+ symbols.
- **SINGLE-STOCK MODE:** When analyzing a specific stock, fetches detailed history via Yahoo and caches it in JSON.

---

## 💻 Dashboard Features

### **Index View (`index.html`)**
- **Confluence Dots:** Visual "Traffic Light" (● ● ●) showing alignment across Intraday/Swing/Long-Term.
- **Actionable Columns:** Shows **R:R Ratio**, **Risk %**, and **Setup Type** badges (🚀, 📉).
- **Live Filtering:** Sort by "Squeeze", "Trend", or specific Score thresholds.

### **Details View (`result.html`)**
- **Profile Switcher:** Toggle between **Intraday** (Scalp) and **Long Term** (Invest) scoring logic instantly.
- **Transparency:** "Top Drivers" table shows exactly *which* indicators boosted the score.
- **Risk Management:** Integrated sticky footer calculator for position sizing based on Stop Loss.

---

## 📦 Installation

```
pip install -r requirements.txt
```

### **Start server**
```
uvicorn main:app --reload
```

### **Access UI**
```
http://localhost:8000
```

---

## 📁 Project Structure

```
/
├── data/
│   ├── store/                 # Parquet Data Lake (OHLCV)
│   ├── trade.db               # SQLite Database (Logs/Meta)
│   └── *.json                 # Index definitions
│
├── services/
│   ├── data_layer.py          # Parquet I/O Engine
│   ├── db.py                  # SQL Models
│   ├── data_fetch.py
│   ├── fundamentals.py
│   ├── indicators.py
│   ├── signal_engine.py
│   ├── corporate_actions.py
│   ├── summaries.py
│   └── metrics_ext.py
│
├── constants.py
├── main.py
├── templates/
│   ├── index.html
│   ├── result.html
│
└── requirements.txt
```

---

## 🛠️ Technologies Used

- **FastAPI(Async Web Framework)**
- **AG Grid**
- **Pandas / Numpy(Data Processing)**
- **Yfinance(Market Data Source)**
- **ThreadPoolExecutors(Concurrency)**
- **cachetools**
- **Jinja Templates**
- **Polars / PyArrow(Ultra-fast Parquet I/O)**
- **SQLAlchemy / SQLite(Relational Database)**
- **Pandas-TA(Technical Indicators)**

---

## 🧭 OPTIMIZED PROCESSING WORKFLOW (FULL PIPELINE)

```
        User Action (UI)  
            ↓
        FastAPI Endpoint  
            ↓
        Check SQLite Cache (Instant Return if warm)  
            ↓
        If missing → Dispatch to ProcessPoolExecutor  
            ↓
        Load OHLCV Window via Polars LazyFrame Tail  
            ↓
        Indicator Computation (Pandas-TA)  
            ↓
        Signal Scoring + Setup Classification  
            ↓
        Write to SQLite + Return to UI  
```

# 🗂 PARQUET LAKEHOUSE DIAGRAM

```
           Yahoo Finance (Initial OHLC Source)
                        │
           Morning Full Window Refresh Job
                        │
                ┌─────────────────────┐
                │   Parquet Store     │
                │ (per symbol/interval) 
                └─────────────────────┘
                        │
             Polars scan_parquet().tail(N)
                        │
              L1 Trimmed LRU Cache (RAM)
                        │
                 Indicator Computation
                        │
                  Signal Scoring Engine
                        │
                    SQLite Persistence
```

---

# 📈 PERFORMANCE BENCHMARKS (REAL WORLD)

| Action | v1.0 | v2.0 | Improvement |
|--------|-------|-------|------------|
| Nifty50 Scan | 45s | 8s | **5.6x faster** |
| Memory | Unbounded | <200MB | **Stable** |
| UI Freezes | Frequent | None | **ProcessPool** |
| Page Refresh | Re-scan | Instant | **SQLite Cache** |
| OHLC Fetch | Always YF | Mostly Parquet | **10–20x faster** |

---
## 📈 Roadmap

[ ] Backtesting Module: Replay historical Parquet data against signal_engine logic.
[ ] Broker Integration: Connect to Zerodha/Angel One for 1-click execution.
[ ] Alerts: Webhook integration for "Squeeze Fired" notifications.
[ ] ML Integration: Predict probability of breakout success.
---

---
##  Pseudocode — compact, readable (sIGNAL eNGINE)

Below is condensed pseudocode for each core function. It keeps the same structure and naming as your code so UI/orchestrator mapping stays intact.

compute_all_profiles(ticker, fundamentals, indicators, profile_map)
copy inputs, reset missing keys
enrich fundamentals with hybrid metrics (enrich_hybrid_metrics)

base_inds = indicators.copy()
try compute global composites:
    base_inds["trend_strength"] = compute_trend_strength(base_inds)
    base_inds["momentum_strength"] = compute_momentum_strength(base_inds)
    base_inds["roe_stability"] = compute_roe_stability(fundamentals)
    optionally compute volatility_quality

for each profile_name in profile_map:
    # choose per-profile indicator slice if provided, else use base_inds
    inds_for_profile = indicators.get(profile_name, base_inds).copy()
    # compute horizon-specific composites safely:
    safe compute trend_strength(inds_for_profile, horizon=profile_name) with fallback
    ensure momentum_strength, roe_stability, volatility_quality exist
    out = compute_profile_score(profile_name, fundamentals, inds_for_profile, profile_map)
    store out in profiles_out[profile_name]
    handle exceptions by returning safe fallback dict with penalty_total key

aggregate best_fit, best_score, avg_signal, missing indicators
return summary (ticker, best_fit, best_score, aggregate_signal, profiles, missing indicators…)

compute_profile_score(profile_name, fundamentals, indicators, profile_map)
profile = profile_map[profile_name]
metrics_map = profile["metrics"]
penalties_map = profile["penalties"]
thresholds = profile["thresholds"]

weighted_sum, weight_sum, metric_details = _compute_weighted_score(metrics_map, fundamentals, indicators)
base_score = (weighted_sum / weight_sum)  # normalized

penalty_total, applied_penalties = _apply_penalties(penalties_map, fundamentals, indicators)

final_score = clamp(base_score - penalty_total, 0, 10)

category = BUY if final_score >= thresholds["buy"] else HOLD if >= thresholds["hold"] else SELL

return {
  profile, base_score, final_score, category,
  metric_details, penalty_total, applied_penalties, thresholds, missing_keys...
}

classify_setup(indicators, horizon)
resolve MA keys (fast, mid, slow)
close, open, prev_close = indicator values
ma_fast, ma_mid, ma_slow = indicator values
bb_upper, bb_lower, rsi, macd_hist, rvol, trend_strength, st_val, is_squeeze

if no close: return "GENERIC"

determine context:
  is_uptrend = ma_slow exists and close > ma_slow
  is_downtrend = ma_slow exists and close < ma_slow

candidates = []

# LONG candidates
if breakout conditions (bb_upper, close ~ upper, rsi high, rvol high, trend_strength):
  if not close < st_val: candidates.append(MOMENTUM_BREAKOUT)

if is_squeeze: candidates.append(VOLATILITY_SQUEEZE)

if is_uptrend:
  if close near ma_fast (±5%) and rsi > 50: TREND_PULLBACK
  elif close near ma_mid and rsi > 40: DEEP_PULLBACK

if is_uptrend and not is_downtrend and rsi >= 55 and macd_hist > 0:
  TREND_FOLLOWING

# SHORT candidates (mirror)
if breakdown conditions (bb_lower, close <= lower, rsi low, rvol high):
  if not close > st_val: MOMENTUM_BREAKDOWN

if is_downtrend:
  if close near ma_fast and rsi < 50: BEAR_PULLBACK
  elif close near ma_mid and rsi < 60: DEEP_BEAR_PULLBACK

if is_downtrend and not is_uptrend and rsi <= 45 and macd_hist < 0:
  BEAR_TREND_FOLLOWING

fallback NEUTRAL/CHOPPY

return highest-priority candidate

calculate_setup_confidence(indicators, trend_strength, macro_trend_status, setup_type, horizon)
resolve MA keys and values, st_val, rsi_slope, macd_hist, rvol, obv_div

trend_score: points for above slow MA, above mid MA, trend_strength numeric buckets, above supertrend

mom_score: momentum signals (macd_hist > 0, price > vwap, rsi_slope > 0), breakouts get bonus, vol_quality bonus

vol_score: rvol buckets and obv confirmations

total_conf = trend_score + mom_score + vol_score
apply macro discount if macro_trend_status shows bearish
apply setup-specific boost factor
final_confidence = clamp(int(total_conf * boost), 0, 100)
return final_confidence

should_trade_current_volatility(indicators, setup_type)
vol_qual, atr_pct = indicators
if missing vol data -> return True but cautious message
if atr_pct extremely high -> return False (avoid)
if setup_type == MOMENTUM_BREAKOUT:
    if vol_qual < minimal threshold -> False else True (breakouts allowed)
if vol_qual < 4.0 -> False (potential chop)
else True

generate_trade_plan(profile_report, indicators, macro_trend_status, horizon)

(Full flow summarized, this is the main decision tree.)

Pseudocode (flow)
# 0. read profile_report.final_score & category (category in {BUY,HOLD,SELL})
# 1. fetch core indicators: price, atr, psar_trend/level, supertrend (st_val, st_signal), rvol, adx
# 2. determine st_direction ("BULL"/"BEAR"/None), is_squeeze, is_bullish_psar
# 3. classify_setup -> setup_type
# 4. ts_val = trend_strength numeric
# 5. setup_conf = calculate_setup_confidence(..., setup_type, horizon)
# 6. can_trade_vol, vol_reason = should_trade_current_volatility(...)
# 7. build base plan dict with analytics, execution_hints

# Gate 1: price/atr sanity checks -> HOLD_NO_RISK / Data Error

# 8. ENTRY PERMISSIONS logic -> set can_enter True/False
   required_conf_base = horizon threshold * 10
   - If setup_type is MOMENTUM_BREAKOUT or MOMENTUM_BREAKDOWN:
       require setup_conf >= required_conf_base
   - Else if LONG PULLBACK types:
       discounted_conf = required_conf_base - 15
       require setup_conf >= discounted_conf and ts_val >= 5.0
   - Else if SHORT PULLBACK types:
       discounted_conf = required_conf_base - 15
       require setup_conf >= discounted_conf and adx >= adx_req
   - Else if SQUEEZE or OVERSOLD_REVERSAL:
       require setup_conf >= required_conf_base - 5

   If not can_trade_vol and setup is not Breakout/Squeeze -> WAIT + reason

   Confidence floor map exists per setup; if not can_enter and setup_conf < floor -> WAIT

# 9. position_size = calculate_position_size(...)

# 10. Determine is_valid_buy / is_valid_short
   BUY allowed if:
      category == BUY and (bullish PSAR OR breakout OR ts_val > 7 OR ST BULL OR can_enter)
      OR (not category BUY but can_enter and ST is BULL)
   SHORT allowed if:
      category == SELL and (bearish PSAR OR breakdown OR ts_val > 7 OR ST BEAR OR can_enter)
      OR (not category SELL but can_enter and ST is BEAR)
      OR explicit MOMENTUM_BREAKDOWN override

# 11. IF is_valid_buy:
     - ST Resistance Guard:
         if ST is BEAR and price < ST and dist < 1.5% and setup not explosive -> WAIT_RESISTANCE
     - ATR config: get multipliers
     - STOP LOSS calculation:
         sl_atr = price - (atr * sl_mult)
         sl_final = sl_atr by default
         if ST exists and price > ST: clamp using candidate_sl = max(ST, price - 2*ATR); sl_final = max(sl_atr, candidate_sl)
         if PSAR tighter and valid -> sl_final = psar_level
         apply Noise clamp: ensure (price - sl_final) >= 0.5*ATR
     - targets:
         t1 = price + 1.5 * risk
         t2 = price + ATR * tp_mult
     - rr_ratio calculation and rr_floor enforcement (by horizon)
     - pullback low-volume hint for pullbacks if rvol < 0.8
     - final signal: RISKY_BUY if macro_bearish else BUY_SQUEEZE (if squeeze) or BUY_TREND
     - return plan

# 12. IF is_valid_short:
     - ST Support Guard:
         if ST is BULL and price > ST and dist < 1.5% and setup not explosive -> WAIT_SUPPORT
     - For MOMENTUM_BREAKDOWN require rvol >= 1.0 (otherwise WAIT_LOW_VOL)
     - SL calc:
         sl_atr = price + (atr * sl_mult)
         sl_final = sl_atr
         if ST exists above price: clamped_st = min(ST, price + 2*ATR); sl_final = min(sl_atr, clamped_st)
         if PSAR tightening above price and tighter -> sl_final = psar_level
         apply Noise clamp: ensure (sl_final - price) >= 0.5*ATR
     - targets:
         t1 = price - 1.5 * risk
         t2 = price - ATR * tp_mult
     - rr_ratio calc and floor check
     - pullback low-volume hint symmetric to longs
     - final signal: RISKY_SHORT if macro_bullish else SHORT_SQUEEZE or SHORT_TREND
     - return plan

# 13. FALLBACK:
   plan["signal"] = "WAIT"
   plan["reason"] = "Score neutral/inconclusive"
   return plan
---

## Decision tree (textual flowchart) — generate_trade_plan

Start → read indicators, profile_report.
Sanity Check: price & atr valid? No → return HOLD_NO_RISK or Data Error.
Classify Setup → setup_type.
Compute setup_confidence → setup_conf.
Volatility Gate: should_trade_current_volatility
• If fails and not Breakout/Squeeze → WAIT (exit).
Entry Gate (can_enter) using setup_type rules and floors.
Category & Confirmation → decide is_valid_buy or is_valid_short.

If is_valid_buy:
Check Supertrend resistance guard (price < ST and ST=BEAR and within threshold) → WAIT_RESISTANCE.
Compute SL candidate(s): ATR fallback, ST clamp (max 2*ATR clamp), PSAR tighten.
Apply Noise clamp (min 0.5 ATR).
Compute T1(1.5R) & T2(ATR*tp_mult), RR, enforce min RR → WAIT_LOW_RR.
Add execution hints (pullback low-volume ok, stop_strategy).
Return BUY_* or RISKY_BUY.

If is_valid_short (symmetric):
Check Supertrend support guard (price > ST and ST=BULL and within threshold) → WAIT_SUPPORT.
For MOMENTUM_BREAKDOWN require rvol > 1.0.
Compute SL candidate(s): ATR, ST (clamped to 2*ATR), PSAR tighten, Noise clamp.
Targets: T1, T2 (mirror), RR check → WAIT_LOW_RR.
Return SHORT_* or RISKY_SHORT.
Else → WAIT with neutral reason.

Guards & hard limits summarized
Supertrend resistance/support guard (reject entries if price within 1.5% of hostile ST, unless explosive setups like breakouts/squeezes).
ATR clamp to Supertrend: ST can be used as a trailing stop but SL can't expose you to more than 2 * ATR from entry.
Noise clamp: ensure stop is at least 0.5 * ATR away from price (prevents getting whipsawed).
Volume checks:
Breakouts/breakdowns require elevated RVOL.
Pullbacks accept low RVOL (healthy consolidation) — hint only, not force.
Volatility guard: extreme ATR% disables entries except explicit breakouts that still meet minimum vol quality.
Confidence floors per setup; can_enter may override floors (smart pullbacks & breakouts).
PSAR tightening: use PSAR if it yields tighter (better) stop.
RR check & minimum RR by horizon/setup: reject trades with too-low reward for risk.

## 🤝 Contributing
PRs are welcome.  
Guidelines:
- Keep layers separated  
- No calculation logic in API routes  
- Add metric-level docstrings  
- Use fast, cached reads for indicators  

---

## 📝 License
Private project — all rights reserved.

---

If you need help extending or deploying this project, feel free to reach out!