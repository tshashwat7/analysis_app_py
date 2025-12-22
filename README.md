# 📈 Pro Stock Analyzer & Trading Engine

An institutional-grade **Algorithmic Trading System** built with a **Hybrid Data Architecture** for the Indian Market (NSE). A modular, end-to-end **Stock Analysis + Trade Decision System** built using FastAPI, AG Grid, pandas, yfinance, and custom scoring logic. This fully modular engine combines high-frequency technical analysis, fundamental screening, and a **Pattern-Aware Trade Intelligence Layer** to generate actionable trade plans.

The system processes a stock from **Raw OHLCV → Actionable Trade Plan** using a multi-layer decision pipeline typically found on proprietary trading desks.

> **Core Philosophy:** Most scanners only look at indicators. This engine understands **Market Structure**. Price does not move randomly—it follows geometry (Cup depth, Flag poles, Box ranges). This engine quantifies that geometry.

---

## 🚀 Key Differentiators

### **1. Unified Configuration Architecture (MASTER_Config)**
* **Single Source of Truth:** All horizon-specific parameters (ATR periods, MA windows, SL/TP multipliers) centralized in `constants.py`.
* **Pattern-Config Injection:** Each pattern receives horizon-aware config (e.g., VCP lookback adapts from 50 days to 50 weeks).
* **Zero Desync Risk:** Eliminates scattered magic numbers across modules.

### **2. Pattern Lifecycle Management**
* **Age Tracking:** Every pattern returns `age_candles` and `formation_timestamp` in metadata.
* **Stateful Invalidation:** Database-backed breakdown tracking (e.g., Darvas box breakdown duration).
* **Stale Pattern Detection:** Automatic expiry of old formations (prevents trading 30-day-old Cup & Handles).

### **3. Hybrid Data Architecture**
* **Polymorphic Indicators:** Automatically switches math based on horizon (e.g., Intraday uses `ATR(10), EMA (20/50/200)`, Short-Term uses `ATR(14), EMA (20/50/200)`, Long-Term uses `ATR(20), WMA (10/40/50)` Multibagger uses `ATR(12)` | MMA (6/12/12)).
* **3-Tier Caching:** RAM -> Parquet (Data Lake) -> Yahoo Finance.
* **Zero-Cost Corp Actions:** Scrapes Equitymaster for upcoming dividends/splits to avoid API costs.

### *4. Pattern Recognition Engine (The "Eyes")**
The system includes a dedicated detection layer that identifies 9 specific institutional setups:
* **Breakout:** Cup & Handle (O'Neil), Darvas Box, Bull Flag/Pennant.
* **Volatility:** Minervini VCP (Volatility Contraction), Bollinger Squeeze.
* **Trend:** Golden/Death Cross, Ichimoku Cloud/TK Cross.
* **Reversal:** Double Top/Bottom, Three-Line Strike.

### *5. Geometric Trade Planning**
It overrides generic ATR targets with **Pattern Geometry**:
* **Smart Targets:** If a "Cup & Handle" is found, T1 is calculated based on `Rim + 0.618 × Depth`.
* **Dynamic Stops:** Stops are auto-tuned based on Volatility Personality (Tight for stable stocks, Wide for volatile ones).
* **Pattern-Aware Time:** Uses "Pattern Physics" to estimate holding time (e.g., *VCP = Fast Breakout*, *Golden Cross = Slow Regime Change*).
---

## 🏗️ System Architecture

| Layer | Key Files | Function |
|:---|:---|:---|
| **Persistence** | `db.py`, `data_layer.py` | **Hybrid Engine**: Parquet (Time-Series) + SQLite (Signal/Fundamental Cache). |
| **Data Layer** | `data_fetch.py` | Hybrid Fetcher (Yahoo + Parquet + Cache Warmer). |
| **Corp Actions** | `corporate_actions.py` | **Hybrid Scraper**: Scrapes MoneyControl for Upcoming actions (Zero API cost), uses Yahoo for History. |
| **Fundamentals** | `fundamentals.py` | **DB-Cached**: Stores raw financial statements in SQLite to strictly limit API calls to once per 24h. |
| **Indicators** | `indicators.py` | Computes 30+ technicals (RSI, MACD, ADX, Supertrend, etc.) per horizon. |
| **Orchestrator** | `main.py` | FastAPI app. Manages **Smart Index Mapping** and Separate Executors (API vs Compute). |
| **Pattern Engine** | `services/patterns/` | **Modular Detectors**: `darvas.py`, `cup_handle.py`, `minervini_vcp.py`, etc. |
| **Pattern State** | `pattern_state_manager.py` | **Lifecycle Tracker**: Monitors pattern age, breakdown states, invalidation logic. |
| **Config Layer** | `constants.py` (MASTER_CONFIG) | **Centralized Config**: Single source of truth for all horizon-specific parameters. |
| **Timezone Utils** | `market_utils.py` | **Modular Timezone**: Centralized IST handling, consistent datetime formatting. |
| **Fusion Layer** | `pattern_fusion.py` | Merges pattern results into the main indicator stream. |
| **Strategy Core** | `strategy_analyzer.py` | Checks fit for 9 strategies (Minervini, CANSLIM, Swing, Trend, etc.). |
| **Planning** | `trade_enhancer.py` | **Geometric Logic**: Overrides targets/stops based on detected patterns. |
| **Execution** | `time_estimator.py` | Calculates "Time to Target" using Slope + Pattern Speed factors. |
| **UI** | `result.html` | Dashboard with Pattern Radar, Interactive Charts, and PDF Export. |
---

## 🧠 The Decision Pipeline

The engine processes every stock through this specific pipeline:

1.  **Data Ingestion:** Fetches OHLCV data for Intraday (15m), Daily, Weekly, and Monthly.
2.  **Metric Computation:** Calculates raw indicators (RSI, EMAs, ATR) for all 4 horizons.
3.  **Pattern Scanning:** Runs the 9-Pattern Library to find structural setups.
4.  **Strategy Fitting:** Scores the stock against styles (e.g., *"Is this a Minervini Setup?"*).
5.  **Signal Generation:** Determines the primary signal (e.g., `MOMENTUM_BREAKOUT` or `RISKY_SHORT`).
6.  **Plan Construction:**
    * Calculates Base ATR Targets.
    * **Enhancer:** If a Pattern is found (e.g., Flag), overwrites targets using Flag Pole height.
    * **Sanity Check:** Ensures Risk:Reward > 1:2.
7.  **Persistence:** Saves the final analysis to SQLite for the Index Grid.

---

## 📚 Pattern Library (Supported Setups)

| Pattern | Type | Timeframe | Speed Factor | Age Tracking |
|:---|:---|:---|:---|:---|
| **Minervini VCP** | Volatility Contraction | Swing | **1.8x** (Explosive) | ✅ Contraction start |
| **Cup & Handle** | Accumulation | Long Term | **1.2x** (Measured) | ✅ Left rim formation |
| **Darvas Box** | Trend Continuation | Swing | **1.3x** (Fast) | ✅ Box consolidation start |
| **Bollinger Squeeze** | Volatility Breakout | Intraday/Daily | **1.5x** (Fast) | ⚠️ Estimated (7-day) |
| **Golden Cross** | Regime Change | Long Term | **0.8x** (Slow Grind) | ✅ Crossover bar |
| **Double Bottom** | Reversal | Swing | **0.9x** (Structural) | ✅ First trough |
| **Three-Line Strike** | Reversal | Short Term | **2.5x** (Violent) | ✅ Always fresh (1 bar) |
| **Flag/Pennant** | Continuation | Swing | **1.4x** (Fast) | ✅ Pole start |
| **Ichimoku Signal** | Trend Entry | All | **1.1x** (Steady) | ✅ TK cross (fresh=1) |


## 🧬 Strategy Analyzer (The "Why")
-Evaluates the stock against nine distinct trading styles to determine fit:
**Value:** Undervalued / Margin of Safety checks.
**Momentum:** High RSI + Pattern Breakouts.
**Minervini Trend Template:** Growth + Technical Compliance (Stage 2).
**Swing:** 1–3 week moves based on reversals.
**Trend Following:** Medium-term MA alignment.
**CANSLIM:** Growth fundamentals + Technical strength.
**Intraday Scalping:** Fast reversals and volatility expansion.
**Long-Term Investing:** Weekly moving average structure.
**Multibagger Framework:** Compounding fundamentals + Monthly trends.

```text
┌──────────────────────────────────────────────┐
│                Data Layer                    │
│  Yahoo → Cache → Parquet Lakehouse           │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│              Indicator Engine                │
│  Polymorphic metrics per horizon             │
│  (Intraday, Daily, Weekly, Monthly)          │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│          Pattern Recognition Engine          │
│  Cup | Darvas | VCP | Squeeze | GC | DB |    │
│  Flag | 3-Line Strike | Ichimoku             │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│             Pattern Fusion Layer             │
│  Normalizes, ranks, merges patterns          │
│  into technical stream                       │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│             Strategy Analyzer                │
│  Value | Trend | Momentum | Minervini | VCP  │
│  Long-term | Swing | Intraday personalities  │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│              Trade Enhancer                  │
│  Geometric targets + dynamic SL based on     │
│  detected patterns                           │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│          Pattern-Aware Time Estimator        │
│  Combines ATR + slopes + strategy + pattern  │
│  physics to estimate holding time            │
└──────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│                    UI                        │
│  Result Dashboard + Pattern Radar + PDF      │
└──────────────────────────────────────────────┘
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


### 💡 WHAT SYSTEM ACTUALLY DOES

```
USER INPUT: Stock Symbol (e.g., "TCS.NS")
    ↓
[FastAPI Handler] → Validates input, routes to analyzer
    ↓
[Horizon Loop] → Fetches 4 separate DataFrames (intraday, short, long, multi)
    ↓
[For Each Horizon]:
    ├─ Calculate 50+ Technical Indicators
    ├─ Run Pattern Detection (auto-merged into indicators)
    ├─ Compute Fundamentals if available
    ├─ Create Hybrid Metrics (ROE/Vol, Price/IV, etc.)
    ├─ Score against 4 Investment Profiles (Value, Growth, Quality, Momentum)
    └─ Generate Trade Plan with:
        • Entry price
        • Target price (3xATR based)
        • Stop loss (2xATR based)
        • Estimated hold time
        • Confidence score
    ↓
[Results Aggregation]:
    ├─ Find best-fit profile across all horizons
    ├─ Calculate average signal strength
    ├─ Persist to SQLite cache (1 hour TTL)
    └─ Format for frontend
    ↓
[Beautiful Dashboard]:
    ├─ ag-Grid table with sortable columns
    ├─ Pattern detection results highlighted
    ├─ Trade plan details with calculated levels
    ├─ Risk/reward analysis
    └─ Fundamental metrics (if available)
    ↓
USER SEES: Complete institutional-grade analysis
---

---

## 🔧 Configuration Architecture

### **MASTER_Config (Centralized Parameters)**
**Location:** `constants.py`

All horizon-specific parameters are now centralized in a single configuration dictionary:
MASTER_CONFIG = {
"intraday": {
"atr_period": 10,
"ema_fast": 20,
"ema_mid": 50,
"ema_slow": 200,
"sl_multiplier": 1.5,
"tp_multiplier": 2.5,
"min_confidence": 70,
"pattern_config": {
"minervini_vcp": {"lookback": 50},
"darvas_box": {"lookback": 50, "box_length": 5},
"flag_pennant": {"pole_back": 15, "flag_back": 5}
}
},
"short_term": {...},
"long_term": {...},
"multibagger": {...}
}


**Benefits:**
- ✅ Eliminates hardcoded magic numbers
- ✅ Enables A/B testing via config swaps
- ✅ Ensures consistency between indicators and patterns
- ✅ Horizon-aware pattern detection (e.g., 50 daily bars = 50 weekly bars)

### **Pattern Configuration Injection**
Each pattern class receives horizon-specific config:
Pattern initialization
pattern = MinerviniVCPPattern(
config=MASTER_CONFIG["short_term"]["pattern_config"]["minervini_vcp"]
)

Pattern uses config internally
self.lookback = self.config.get("lookback", 50) # Default fallback

### **Timezone Management (market_utils.py)**
**Location:** `services/market_utils.py`

Centralized timezone handling ensures consistency:

from services.market_utils import get_current_market_time, format_market_time

Always returns timezone-aware IST datetime
now = get_current_market_time()

Consistent ISO8601 formatting with timezone
timestamp = format_market_time(now) # "2025-12-14T12:30:00+05:30"


**Core Functions:**
- `get_market_timezone()` - Returns pytz IST timezone object
- `get_current_market_time()` - Returns timezone-aware datetime
- `to_market_time(dt)` - Converts any datetime to IST
- `format_market_time(dt)` - Formats as ISO8601 with timezone
- `get_current_market_day()` - Returns weekday name

**Integration Points:**
- Pattern metadata (`formation_timestamp`)
- Database models (`created_at`, `updated_at`)
- Signal engine (market hours detection)
- State manager (breakdown tracking)
---

## 🔄 Pattern Lifecycle Management

### **Age Tracking (All Patterns)**
Every pattern now returns age metadata:

result["meta"] = {
"age_candles": 15, # Bars since pattern formed
"formation_timestamp": "2025-12-01T09:15:00+05:30", # ISO8601 with TZ
"pattern_duration_candles": 10 # Pattern-specific metric
}

### **Pattern Invalidation (Trade Enhancer)**
**Location:** `services/tradeplan/trade_enhancer.py` → `check_pattern_invalidation()`

Monitors active patterns for breakdown/failure:

| Pattern | Invalidation Trigger | Action |
|---------|---------------------|--------|
| **Darvas Box** | Price < `box_low` | EXIT_IMMEDIATELY |
| **Cup & Handle** | Price < `handle_low × 0.95` | EXIT_ON_CLOSE |
| **VCP/Stage 2** | Price < `pivot × 0.92` (8% stop) | EXIT_ON_CLOSE |
| **Flag/Pennant** | Price < `flag_low` | EXIT_IMMEDIATELY |
| **Golden Cross** | 50 EMA crosses below 200 EMA | EXIT_ON_CLOSE |

### **Breakdown State Tracking**
**Location:** `services/patterns/pattern_state_manager.py`

Database-backed state management for pattern invalidation:

**Schema:**
CREATE TABLE pattern_breakdown_state (
symbol TEXT,
pattern_name TEXT,
horizon TEXT,
candle_count INTEGER, -- Days below threshold
price_at_breakdown REAL,
threshold_level REAL,
condition TEXT,
created_at TIMESTAMP,
updated_at TIMESTAMP,
PRIMARY KEY (symbol, pattern_name, horizon)
);

**Core Functions:**
- `save_breakdown_state()` - Day 1: Pattern breaks key level
- `update_breakdown_state()` - Day 2+: Increment candle count
- `get_breakdown_state()` - Retrieve active breakdown tracking
- `delete_breakdown_state()` - Pattern recovered or invalidated

**Usage Example:**
Day 1: Price breaks below Darvas box_low
save_breakdown_state(
symbol="RELIANCE",
pattern_name="darvas_box",
horizon="short_term",
price=1195.00,
threshold=1200.00
)

Day 2: Still below threshold
new_count = update_breakdown_state("RELIANCE", "darvas_box", "short_term")

If new_count >= 2: Invalidate pattern
if new_count >= 2:
delete_breakdown_state("RELIANCE", "darvas_box", "short_term")

# Trigger EXIT signal

### **Automatic Cleanup**
Background scheduler runs every 24 hours to delete stale breakdown states (>30 days old).


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
# 📈 Pro Stock Analyzer & Trading Engine v3.0

An institutional-grade **Algorithmic Trading Intelligence Engine** for the Indian Market (NSE). Unlike basic scanners that rely on simple indicators, this engine understands **Market Structure**, combining pattern recognition, strategy personality, and geometric trade planning into a single decision pipeline.

> **Core Philosophy:** Price does not move randomly—it follows geometry. A "Cup & Handle" has a measurable depth that projects a specific target. A "Volatile" stock requires wider stops than a "Stable" one. This engine quantifies that physics.

---

## 🚀 Key Capabilities

### **1. Pattern Recognition Engine (The "Eyes")**
A dedicated detection layer that identifies 9 specific institutional setups:
* **Breakout:** Cup & Handle (O'Neil), Darvas Box, Bull Flag/Pennant.
* **Volatility:** Minervini VCP (Volatility Contraction), Bollinger Squeeze.
* **Trend:** Golden/Death Cross, Ichimoku Cloud/TK Cross.
* **Reversal:** Double Top/Bottom, Three-Line Strike.

### **2. Geometric Trade Planning**
It overrides generic ATR targets with **Pattern Geometry**:
* **Smart Targets:** If a "Cup & Handle" is found, T1 is calculated based on `Rim + 0.618 × Depth`.
* **Dynamic Stops:** Stops are auto-tuned based on Volatility Personality (Tight for stable stocks, Wide for volatile ones).
* **Pattern-Aware Time:** Uses "Pattern Physics" to estimate holding time (e.g., *VCP = Fast Breakout*, *Golden Cross = Slow Regime Change*).

### **3. Strategy Analyzer (The "Why")**
Evaluates the stock against nine distinct trading styles to determine fit:
* **Minervini Trend Template:** Growth + Technical Compliance (Stage 2).
* **CANSLIM:** Growth fundamentals + Technical strength.
* **Value:** Undervalued / Margin of Safety checks.
* **Momentum/Swing:** High RSI + Pattern Breakouts.

### **4. Polymorphic Indicator Engine**
The engine adapts its math based on the time horizon:
| Horizon | Purpose | ATR Logic | MA Logic |
|:---|:---|:---|:---|
| **Intraday** | Scalping | `ATR(10)` | EMA (20/50/200) |
| **Short-Term** | Swing Trading | `ATR(14)` | EMA (20/50/200) |
| **Long-Term** | Investing | `ATR(20)` | WMA (10/40/50) |
| **Multibagger** | Deep Value | `ATR(12)` | MMA (6/12/12) |

---

## 🧠 The Decision Pipeline

The engine processes every stock through this specific pipeline:

1.  **Data Ingestion:** Fetches OHLCV data for Intraday (15m), Daily, Weekly, and Monthly.
2.  **Pattern Scanning:** Runs the 9-Pattern Library to find structural setups.
3.  **Strategy Fitting:** Scores the stock against styles (e.g., *"Is this a Minervini Setup?"*).
4.  **Signal Generation:** Determines the primary signal (e.g., `MOMENTUM_BREAKOUT` or `RISKY_SHORT`) using a priority queue.
5.  **Plan Construction:**
    * **Enhancer:** If a Pattern is found (e.g., Flag), overwrites targets using Flag Pole height.
    * **Sanity Check:** Ensures Risk:Reward > 1:2.
    * **Time Estimator:** Calculates `Distance / (ATR * Speed_Factor)`.
6.  **Persistence:** Saves the final analysis to SQLite for the Index Grid.

---

## 🖥️ Dashboard & Features

### **Index View (Discovery)**
* **Confluence Dots:** Visual "Traffic Light" (● ● ●) showing alignment across Intraday/Swing/Long-Term.
* **Pattern Badges:** Filter stocks by specific patterns (e.g., "Show me all VCPs").
* **Live Filtering:** Sort by "Squeeze", "Trend", or Score thresholds.

### **Result View (Deep Dive)**
* **Pattern Radar:** Visual card showing active patterns and their confidence scores.
* **Visual Trade Range:** Progress bar showing Entry position relative to Stop Loss and Target.
* **Smart PDF Export:** Generates a dense, single-page PDF report with side-by-side Technicals/Fundamentals.
* **Profile Switcher:** Instantly toggle analysis between Intraday (Scalp) and Multibagger (Invest).

---

## 🏗️ Technical Architecture

### **Hybrid Data Layer**
* **Parquet Lakehouse:** Stores OHLCV time-series data locally for sub-millisecond access.
* **SQLite Cache:** Stores computed signals and heavy fundamental JSONs.
* **Smart Warmer:** Background process pre-fetches data for Nifty 50/500 stocks during market hours.
* **Corp Actions:** Hybrid fetcher scrapes MoneyControl/Equitymaster for upcoming dividends (Zero API cost).

### **Performance Benchmarks**
| Action | v1.0 | v3.0 | Improvement |
|--------|-------|-------|------------|
| Nifty50 Scan | 45s | 8s | **5.6x faster** |
| UI Freezes | Frequent | None | **ProcessPoolExecutor** |
| OHLC Fetch | Always YF | mostly Parquet | **10–20x faster** |

---

## 📁 Project Structure

```text
/
├── data/
│   ├── store/                 # Parquet Data Lake (OHLCV)
│   ├── trade.db               # SQLite Database (Logs/Meta)
│   └── *.json                 # Index definitions
│
├── services/
│   ├── patterns/              # The "Eyes" (Cup, VCP, Darvas...)
│   │   ├── base.py            # Base pattern class
│   │   ├── pattern_state_manager.py  # Breakdown tracking
│   │   ├── darvas.py
│   │   ├── cup_handle.py
│   │   └── ...            
│   ├── analyzers/             # The "Brain" (Strategy, Patterns)
│   ├── tradeplan/             # The "Planner" (Enhancer, Estimator)
│   ├── fusion/                # Merges Patterns into Indicators
│   ├── data_layer.py          # Parquet I/O Engine
│   ├── db.py                  # SQL Models
│   ├── data_fetch.py
│   ├── fundamentals.py
│   ├── indicators.py
│   ├── signal_engine.py       # Core Decision Logic
│   ├── corporate_actions.py
│   ├── summaries.py
│   └── metrics_ext.py
│
├── config
│   ├── constants.py        # master config
│   ├── market_utils.py        # Timezone utilities 
│   ├── logger_config.py        # modular logging  
├── main.py                     # FastAPI Orchestrator
├── templates/                  # Jinja2 Dashboards
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
[ ] ML Integration: Predict probability of breakout success.
---

---

## 🧠 Logic Deep Dive
This is an excellent, comprehensive deep dive. It perfectly bridges the gap between "high-level features" and "developer implementation."

Here is the formatted version, matching the exact style (bold headers, bullet points, and clean hierarchy) used in the rest of your README. You can paste this directly under the **Logic Deep Dive** section.

---

## 🧠 Logic Deep Dive

The engine uses a deterministic, priority-driven decision framework to classify market setups, validate execution conditions, and generate a geometric, risk-aware trade plan.

### **1. Signal Classification Engine (Priority Queue)**
The classifier processes all possible setups in descending priority. The **first matching condition** becomes the active signal.

* **🚀 Momentum Breakout (Highest Priority)**
    * **Logic:** `Price > BB Upper` AND `RSI > 60` AND `RVOL > 1.5×` AND `Trend Strength > 6`.
    * **Context:** Used for explosive upside events only.
* **🎯 Volatility Squeeze**
    * **Logic:** Bollinger Bands inside Keltner Channels (`TTM Squeeze = ON`).
    * **Context:** Signals volatility contraction before expansion; direction is decided post-breakout.
* **💎 Quality Accumulation**
    * **Logic:** `Price in Lower BB Half` AND `ADX < 30` (Ranging) AND `Fundamentals Strong` (PE < 25, ROE > 12%).
    * **Context:** Used for long-term value accumulation candidates.
* **📘 Trend Pullback**
    * **Logic:** `Price > 200 EMA` (Uptrend) AND `Price near 20/50 EMA` AND `RSI > 50`.
    * **Context:** Standard continuation-pullback entry.
* **📈 Trend Following**
    * **Logic:** `20 EMA > 50 EMA > 200 EMA` (Perfect Alignment) AND `MACD Hist > 0`.
    * **Context:** Used when the trend is fully mature.

### **2. Accumulation Mode (Smart Money Logic)**
Designed to detect **multibagger-grade accumulation bases** despite weak short-term signals.

* **Fundamental Gate:** `PE < 25`, `ROE > 12%`, `EPS Growth > 0`.
* **Technical Gate:** `Price > BB Lower Band`, `Price < BB Mid × 1.02`, `ADX < 30`.
* **Action:** Generates **BUY_ACCUMULATE** with staged position sizing.

### **3. Entry Guards & Safety Filters**
Every potential trade is validated through multiple protective layers.

* **🟦 Macro Trend Guard:** If NIFTY Trend = Bearish, reduce long confidence by **15%**.
* **🟥 Supertrend Guard:**
    * Longs blocked when `Price < Supertrend Bearish` (unless Breakout).
    * Shorts blocked when `Price > Supertrend Bullish` (unless Breakdown).
* **🟨 Volatility Guard:**
    * `ATR% > 4%` → Reject trade (except Breakouts).
    * `Volatility Quality < 4` → Avoid choppy markets.

### **4. Best Horizon Selection (Profile Competition)**
The engine computes and scores all horizons simultaneously:
* Intraday (15m)
* Short-Term (Daily)
* Long-Term (Weekly)
* Multibagger (Monthly)

**Selection Example:**
* Intraday: 4.5
* **Short Term: 8.2 (Selected)**
* Long Term: 6.0
* Multibagger: 5.4

The **highest-scoring profile** becomes the active view on the dashboard.

### **5. Trade Plan Construction (Geometric Planning)**

* **🎯 Entry Permission Framework:**
    * **Breakouts:** Require ≥70% confidence.
    * **Squeezes:** Require ≥65% confidence.
    * **Pullbacks:** Require ≥55% confidence + `Trend Strength ≥ 5`.
* **🔻 Stop-Loss Geometry:**
    * **Base SL:** `Entry − (ATR × SL_MULT)`.
    * **Supertrend Clamp:** Uses ST if tighter.
    * **PSAR Tightening:** Uses PSAR if tighter.
    * **Noise Filter:** SL must be `≥ 0.5× ATR` away.
* **🎯 Target Calculation:**
    * **T1:** `Entry + (1.5 × Risk)`.
    * **T2:** `Entry + (ATR × TP_MULT)`.
* **📏 Pattern Overrides:**
    * **Cup & Handle:** Rim depth projection.
    * **Darvas Box:** Box height projection.
    * **Flag/Pennant:** Pole height projection.
* **📊 Risk/Reward Enforcement:**
    * **Intraday:** 1:1.5
    * **Swing:** 1:2
    * **Long-Term:** 1:2.5
    * *Trades failing RR floor are rejected.*

### **6. Volatility & Volume Controls**

* **Volatility Rules:**
    * Excessively high ATR% = Dangerous.
    * Low Volatility Quality = Chop (Avoid).
* **Volume Rules:**
    * **Breakouts:** Require strong RVOL.
    * **Breakdowns:** Require RVOL ≥ 1.0.
    * **Pullbacks:** Allow low RVOL (healthy consolidation).
* **Supertrend Proximity:**
    * Avoid longs directly under bearish ST.
    * Avoid shorts directly over bullish ST.

### **7. Setup Confidence Model (0–100%)**
Final Confidence = **Trend + Momentum + Volume ± Macro Adjustment**

* **Trend Component:**
    * Above 200 EMA → **+20**
    * Above 50 EMA → **+10**
    * Supertrend Alignment → **+10**
    * Trend Strength Tiers
* **Momentum Component:**
    * MACD Histogram Positive
    * RSI Slope Positive
    * Above VWAP
    * Breakout Bonus
* **Volume Component:**
    * RVOL Tiers
    * OBV Confirmation
    * Volume Spike Detection
* **Final Adjustments:**
    * Macro Penalty (if Bearish Index)
    * Setup Boost (VCP, Breakout, Squeeze)
    * *Confidence is clipped to 0–100% range.*

````````````````````````````````````````
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

Have replaced generic `WAIT` with **specific rejection codes**:

| Old Signal | New Signal | Meaning |
|------------|------------|---------|
| `WAIT` | `NA_INVALID_INPUTS` | Price/ATR data missing |
| `WAIT` | `NA_VOLATILITY_BLOCKED` | Market too chaotic |
| `WAIT` | `NA_ENTRY_PERMISSION_FAILED` | Setup doesn't meet entry rules |
| `WAIT` | `NA_LOW_CONFIDENCE` | Below dynamic confidence floor |
| `WAIT` | `NA_DIVERGENCE_DETECTED` | Bearish divergence warning |
| `WAIT` | `NA_POOR_VOLUME` | Volume drought detected |
| `WAIT_LOW_RR` | (Unchanged) | Risk:Reward < 1.5:1 |