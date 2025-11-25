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