# config/constants.py
# === Technical Indicator Constants ===

# RSI, MACD, BB etc. existing constants assumed here

# --- Stochastic Oscillator Defaults ---
STOCH_FAST = {"k_period": 5, "d_period": 3, "smooth": 3}
STOCH_SLOW = {"k_period": 14, "d_period": 3, "smooth": 3}
STOCH_THRESHOLDS = {"overbought": 80, "oversold": 20}

# config/constants.py
ENABLE_CACHE = False

STOCH_THRESHOLDS = {
    "overbought": 80,
    "oversold": 20,
}

# ATR-based stoploss/target multipliers
ATR_MULTIPLIERS = {
    "intraday": {"tp": 1.2, "sl": 0.6},
    "shortterm": {"tp": 2.0, "sl": 1.0},
    "longterm": {"tp": 3.0, "sl": 1.5},
}

flowchart_mapping = {
    # 1️⃣ TECHNICAL INDICATORS
    "RSI": ("quick_score", None, "RSI"),
    "MACD": ("quick_score", None, "MACD"),
    "EMA Crossover": ("quick_score", None, "EMA Crossover Trend"),
    "20 EMA": ("quick_score", None, "20 EMA"),
    "50 EMA": ("quick_score", None, "50 EMA"),
    "Bollinger Bands": ("quick_score", None, "BB Signal"),
    "BB High": ("quick_score", None, "BB High"),
    "BB Low": ("quick_score", None, "BB Low"),
    "ATR (Volatility)": ("quick_score", None, "ATR (14)"),
    "Volume Spike": ("quick_score", None, "Volume Spike Signal"),
    "Ichimoku Cloud": ("quick_score", None, "Ichimoku Cloud"),
    "Stochastic Oscillator": ("quick_score", None, "Stoch %K"),
    "Relative Volume (RVOL)": ("quick_score", None, "Relative Volume (RVOL)"),
    "OBV Divergence": ("quick_score", None, "OBV Divergence"),
    "Pivot Points / Fibonacci Levels": ("quick_score", None, "Entry Price (Confirm)"),

    # 2️⃣ FUNDAMENTAL METRICS
    "P/E Ratio": ("fundamentals", None, "Valuation (P/E)"),
    "PEG Ratio": ("fundamentals", None, "PEG Ratio"),
    "ROE": ("fundamentals", None, "Return on Equity (ROE)"),
    "Debt-to-Equity": ("fundamentals", None, "Debt to Equity"),
    "Free Cash Flow Growth": ("fundamentals", None, "FCF Yield (%)"),  # you don't have CAGR yet
    "Dividend Yield": ("fundamentals", None, "Dividend Yield"),
    "Management Quality": ("extended", None, "Promoter Holding (%)"),
    "EPS Growth Consistency": ("extended", None, "EPS Growth Consistency (5Y CAGR)"),
    "Interest Coverage Ratio": ("fundamentals", None, "Interest Coverage"),
    "Operating Cash Flow vs Net Profit": ("extended", None, "Operating CF vs Net Profit"),
    "R&D Intensity": ("extended", None, "R&D Intensity (%)"),
    "Asset Turnover Ratio": ("fundamentals", None, "Asset Turnover Ratio"),
    "Book Value Growth": ("fundamentals", None, "Price to Book (P/B)"),

    # 3️⃣ MULTIBAGGER IDENTIFICATION
    "ROIC": ("fundamentals", None, "ROIC (%)"), 
    "Earnings Growth": ("fundamentals", None, "Net Profit Qtr Growth YoY %"),
    "Promoter Holding": ("extended", None, "Promoter Holding (%)"),
    "Market Cap CAGR": ("extended", None, "Market Cap CAGR"),
    "TAM Growth": ("extended", None, "TAM Growth"),
    "Debt/Equity": ("fundamentals", None, "Debt to Equity"),
    "PEG (Forward)": ("fundamentals", None, "PEG Ratio"),  # fallback since no Forward P/E
    "Promoter Pledge": ("extended", None, "Promoter Pledge"),
    "Institutional Ownership Trend": ("extended", None, "Institutional Ownership (%)"),
    "Innovation/R&D Pipeline": ("extended", None, "R&D Intensity (%)"),
    "Sector Leadership": ("fundamentals", None, "Sector"),

    # 4️⃣ SENTIMENT & BEHAVIORAL FACTORS
    "VIX": ("extended", None, "VIX (Volatility Index)"),
    "Analyst Ratings": ("extended", None, "Analyst Ratings"),
    "Fear & Greed Index": ("extended", None, "Fear & Greed Index"),
    "Retail Sentiment": ("extended", None, "Retail Sentiment"),
    "Put-Call Ratio (PCR)": ("extended", None, "Put-Call Ratio (PCR)"),
    "Insider Trading Activity": ("extended", None, "Insider Trading Activity"),
    "Advance-Decline Line (A/D)": ("extended", None, "Advance-Decline Line (A/D)"),
    "News/Google Trends": ("extended", None, "News/Google Trends"),

    # 5️⃣ MACRO-ECONOMIC CONTEXT
    "GDP Growth": ("extended", None, "GDP Growth (%)"),
    "Inflation Rate": ("extended", None, "Inflation Rate (%)"),
    "Interest Rate Trend": ("extended", None, "Repo Rate (%)"),  # from macro_sentiment
    "Crude Oil Prices": ("extended", None, "Crude Oil ($)"),
    "Bond Yield vs Equity Spread": ("extended", None, "10Y Bond Yield (%)"),  # macro proxy
    "Currency Trend": ("extended", None, "Currency Trend (USD/INR)"),
    "PMI": ("extended", None, "PMI"),
    "Sector Rotation": ("payload", None, "index"),

    # 6️⃣ RISK MANAGEMENT
    "Stop-Loss": ("quick_score", None, "Suggested SL (2xATR)"),
    "Position Sizing": ("payload", None, None),
    "Max Drawdown": ("extended", None, "Max Drawdown (%)"),
    "Beta": ("fundamentals", None, "Beta"),
    "Sharpe Ratio": ("extended", None, "Sharpe Ratio"),
    "Sortino Ratio": ("extended", None, "Sortino Ratio"),
    "Correlation Matrix": ("extended", None, "Correlation Matrix"),
    "Drawdown Recovery Period": ("extended", None, "Drawdown Recovery Period"),
}

TECHNICAL_WEIGHTS = {
    "rsi": 1.0,
    "macd_cross": 1.0, 
    "macd_hist_z": 0.8,
    "price_vs_200dma_pct": 1.0,
    "adx": 1.0,
    "vwap_bias": 0.8,
    "vol_trend": 0.6,
    "rvol": 0.6,
    "stoch_k": 0.6,
    "bb_low": 0.4,
    "bb_width": 0.3,
    "entry_confirm": 0.5,
    "dma_20_50_cross": 0.8, 
    "dma_200_slope": 0.8,
    "ichi_cloud": 1.0, 
    "obv_div": 0.6, 
    "atr_14": 0.8,
    "vol_spike_ratio": 0.5,
    "rel_strength_nifty": 0.6,
    "price_action": 0.7,
    "supertrend_signal": 1.0,
    "cci": 0.6,
    "bb_percent_b": 0.4,
    "cmf_signal": 0.6,
    "donchian_signal": 0.8,
    "reg_slope": 0.8,
}

# -----------------------------
# Master Metric Map (Internal Key -> UI Alias)
# -----------------------------
TECHNICAL_METRIC_MAP = {
    "price": "Current Price",
    # Indicators from compute_... functions
    "rsi": "RSI",
    "mfi": "MFI",
    "short_ma_cross": "Short MA Cross (5/20)",
    "adx": "ADX",
    "adx_signal": "ADX Signal",
    "stoch_k": "Stoch %K",
    "stoch_d": "Stoch %D",
    "stoch_cross": "Stoch Crossover",
    "ema_20": "20 EMA",
    "ema_50": "50 EMA",
    "ema_200": "200 EMA",
    "ema_cross_trend": "EMA Crossover Trend",
    "vwap": "VWAP",
    "vwap_bias": "VWAP Bias",
    "bb_high": "BB High",
    "bb_mid": "BB Mid",
    "bb_low": "BB Low",
    "bb_width": "BB Width",
    "rvol": "Relative Volume (RVOL)",
    "obv_div": "OBV Divergence",
    "vpt": "VPT",
    "vol_spike_ratio": "Volume Spike Ratio",
    "vol_spike_signal": "Volume Spike Signal",
    "atr_14": "ATR (14)",
    "atr_pct": "ATR %",
    "ichi_cloud": "Ichimoku Cloud",
    "ichi_span_a": "Ichimoku Span A",
    "ichi_span_b": "Ichimoku Span B",
    "ichi_tenkan": "Tenkan-sen",
    "ichi_kijun": "Kijun-sen",
    "price_action": "Price Action",
    "macd": "MACD",
    "macd_cross": "MACD Cross",
    "macd_hist_z": "MACD Hist Z-Score",
    "macd_histogram": "MACD Histogram (Raw Momentum)",
    # Inline indicators from compute_indicators
    "price_vs_200dma_pct": "Price vs 200 DMA (%)",
    "dma_200": "200 DMA",
    "vol_trend": "Volume Trend",
    "vol_vs_avg20": "Volume vs Avg20",
    "dma_20": "20 DMA",
    "dma_50": "50 DMA",
    "dma_20_50_cross": "20/50 DMA Cross",
    "rel_strength_nifty": "Relative Strength vs NIFTY (%)",
    "entry_confirm": "Entry Price (Confirm)",
    "dma_200_slope": "200 DMA Trend (Slope)",
    "sl_2x_atr": "Suggested SL (2xATR)",
    "supertrend_signal": "SuperTrend Signal",
    "cci": "Commodity Channel Index (CCI)",
    "bb_percent_b": "Bollinger %B",
    "cmf_signal": "Chaikin Money Flow (CMF)",
    "donchian_signal": "Donchian Channel Breakout",
    "reg_slope": "Regression Slope (Trend Angle)",
    "nifty_trend_score": "NIFTY Trend Score",
    "pivot_point": "Pivot Point (Daily)",
    "resistance_1": "Resistance 1 (Fib)",
    "resistance_2": "Resistance 2 (Fib)",
    "resistance_3": "Resistance 3 (Fib)",
    "support_1": "Support 1 (Fib)",
    "support_2": "Support 2 (Fib)",
    "support_3": "Support 3 (Fib)",
    
    # --- New Timing Indicators ---
    "psar_trend": "Parabolic SAR Trend",
    "psar_level": "PSAR Level",
    "ttm_squeeze": "TTM Squeeze Signal",
    "kc_upper": "Keltner Upper",
    "kc_lower": "Keltner Lower",
    "ema_20_slope": "20 EMA Slope (Angle)",
    "ema_50_slope": "50 EMA Slope (Angle)",
    "true_range": "True Range (Raw)",
    "true_range_pct": "True Range % of Price",
    "hv_10": "Historical Volatility (10D)",
    "hv_20": "Historical Volatility (20D)",
}


# -------------------------
# Updated fundamental weights (short keys)
# -------------------------
FUNDAMENTAL_WEIGHTS = {
    # --- Valuation (20%) ---
    "pe_ratio": 0.05,
    "pb_ratio": 0.04,
    "peg_ratio": 0.03,
    "fcf_yield": 0.05,
    "dividend_yield": 0.03,

    # --- Profitability / Returns (25%) ---
    "roe": 0.10,
    "roce": 0.07,
    "roic": 0.08,

    # --- Leverage / Liquidity (15%) ---
    "de_ratio": 0.05,
    "interest_coverage": 0.05,
    "current_ratio": 0.03,
    "ocf_vs_profit": 0.02,

    # --- Efficiency / Quality (20%) ---
    "asset_turnover": 0.04,
    "piotroski_f": 0.07,
    "r_d_intensity": 0.04,
    "earnings_stability": 0.05,

    # --- Growth (15%) ---
    "eps_growth_5y": 0.06,
    "fcf_growth_3y": 0.05,
    "market_cap_cagr": 0.04,

    # --- Ownership / Market Sentiment (5%) ---
    "promoter_holding": 0.015,
    "institutional_ownership": 0.015,
    "beta": 0.01,
    "52w_position": 0.01,
    "dividend_payout": 0.03,
    "yield_vs_avg": 0.02,
}

# Safety normalization (ensures sum = 1.0)
_total = sum(FUNDAMENTAL_WEIGHTS.values())
if abs(_total - 1.0) > 1e-3:
    FUNDAMENTAL_WEIGHTS = {k: v / _total for k, v in FUNDAMENTAL_WEIGHTS.items()}


# -------------------------
# Updated alias map (short_key -> human label)
# -------------------------
FUNDAMENTAL_ALIAS_MAP = {
    # Valuation
    "pe_ratio": "P/E Ratio",
    "pb_ratio": "Price to Book (P/B)",
    "peg_ratio": "PEG Ratio",
    "fcf_yield": "FCF Yield (%)",
    "dividend_yield": "Dividend Yield (%)",
    # Profitability / Returns
    "roe": "Return on Equity (ROE)",
    "roce": "Return on Capital Employed (ROCE)",
    "roic": "Return on Invested Capital (ROIC)",
    "net_profit_margin": "Net Profit Margin (%)",
    "operating_margin": "Operating Margin (%)",
    # Growth (mostly provided by metrics_ext, kept mapping)
    "eps_growth_5y": "EPS Growth (5Y CAGR)",
    "profit_growth_3y": "Profit Growth (3Y CAGR)",
    "fcf_growth_3y": "FCF Growth (3Y CAGR)",
    "market_cap_cagr": "Market Cap CAGR (%)",
    # Leverage / Liquidity
    "de_ratio": "Debt to Equity",
    "interest_coverage": "Interest Coverage Ratio",
    "current_ratio": "Current Ratio",
    "ocf_vs_profit": "Operating CF vs Net Profit",
    # Efficiency / Quality
    "asset_turnover": "Asset Turnover Ratio",
    "piotroski_f": "Piotroski F-Score",
    "r&d_intensity": "R&D Intensity (%)",
    "earnings_stability": "Earnings Stability",
    # Ownership / Market
    "promoter_holding": "Promoter Holding (%)",
    "institutional_ownership": "Institutional Ownership (%)",
    "beta": "Beta",
    "52w_position": "52W Position (off-high %)",
    "analyst_rating": "Analyst Rating (Momentum)",
    "quarterly_growth": "Quarterly Growth (EPS/Rev)",
    "short_interest": "Short Interest",
    "trend_strength": "Trend Strength (EMA 50/200)",
    "net_profit_margin": "Net Profit Margin (%)", 
    "operating_margin": "Operating Margin (%)",  
    "ebitda_margin": "EBITDA Margin (%)",     
    "pe_vs_sector": "P/E vs Sector Avg",     
    "dividend_payout": "Dividend Payout (%)",
    "yield_vs_avg": "Yield vs 5Y Avg",
    "revenue_growth_5y": "Revenue Growth (5Y CAGR)",
    "days_to_earnings": "Days to Next Earnings",
}

# -------------------------
# Field candidates map for YFinance / DataFrame key normalization
# -------------------------
FUNDAMENTAL_FIELD_CANDIDATES = {
    # Income Statement
    "revenue": [
        "Total Revenue", "Revenue", "totalRevenue", "Sales", "Net Sales", "Operating Revenue"
    ],
    "net_income": [
        "Net Income", "netIncome", "NetIncome", "Profit After Tax", "Net Profit"
    ],
    "operating_income": [
        "Operating Income", "EBIT", "Ebit", "Operating Profit", "OperatingProfit"
    ],
    "ebit": [
        "EBIT", "Ebit", "Operating Income", "Operating Profit"
    ],
    "ebitda": [
        "EBITDA", "Ebitda", "Operating Profit Before Depreciation"
    ],
    "cogs": [
        "Cost Of Revenue", "Cost of Goods Sold", "COGS", "Total Expenses", "Operating Expense"
    ],
    "interest_expense": [
        "Interest Expense", "Interest And Debt Expense", "Finance Cost", "Interest"
    ],
    "tax_expense": [
        "Income Tax Expense", "Tax Provision", "Total Tax Expense"
    ],
    "pre_tax_income": [
        "Pretax Income", "Income Before Tax", "Income Before Tax Expense", "PretaxProfit"
    ],

    # Balance Sheet
    "total_assets": [
        "Total Assets", "totalAssets", "Assets"
    ],
    "total_liabilities": [
        "Total Liabilities", "totalLiab", "Liabilities"
    ],
    "total_equity": [
        "Total Stockholders Equity", "totalStockholdersEquity",
        "Shareholders Equity", "Total Equity", "Equity"
    ],
    "current_assets": [
        "Total Current Assets", "totalCurrentAssets", "Current Assets", "currentAssets"
    ],
    "current_liabilities": [
        "Total Current Liabilities", "totalCurrentLiabilities", "Current Liabilities", "currentLiabilities"
    ],
    "cash_equivalents": [
        "Cash And Cash Equivalents", "cashAndCashEquivalents", "Cash", "Cash Balance"
    ],
    "total_debt": [
        "Total Debt", "totalDebt", "Long Term Debt", "longTermDebt", "Borrowings"
    ],

    # Cash Flow Statement
    "ocf": [
        "Total Cash From Operating Activities", "totalCashFromOperatingActivities", "Operating Cash Flow"
    ],
    "capex": [
        "Capital Expenditures", "capitalExpenditures", "CapEx", "Purchase Of Fixed Assets"
    ],
    "free_cash_flow": [
        "Free Cash Flow", "freeCashflow", "freeCashFlow"
    ],

    # Other metrics / ratios
    "rd_expense": [
        "Research And Development", "Research Development", "Research and Development Expense",
        "R&D", "Rnd"
    ],
    "eps": [
        "EPS", "Diluted EPS", "Basic EPS", "Earnings Per Sare", "eps"
    ],
    "shares_outstanding": [
        "Basic Average Shares", "Shares Outstanding", "Weighted Average Shares"
    ],
    "gross_profit": [
        "Gross Profit", "GrossIncome"
    ],
    "market_cap": [
        "marketCap", "Market Capitalization", "market_cap"
    ],
    "dividend": [
        "Dividends Paid", "dividendRate", "Cash Dividends Paid"
    ],
    "fcf_yield": [
        "Free Cash Flow Yield", "fcfYield"
    ],
    "promoter_holding": [
        "heldPercentInsiders", "insiderPercent", "insidersPercent", "Insider Ownership"
    ],
    "institutional_ownership": [
        "heldPercentInstitutions", "institutionPercent", "institutionsPercent", "Institutional Ownership"
    ],
    "dividend_yield": ["dividendYield"],
    "analyst_rating": ["recommendations", "recommendationKey", "recommendationMean"],
    "quarterly_growth": ["earningsQuarterlyGrowth", "revenueQuarterlyGrowth"],
    "short_interest": ["shortRatio", "sharesPercentSharesOut", "shortPercentOfFloat"],
    "trend_strength": [],
    }

SECTOR_PE_AVG = {
    "Technology": 58.7,
    "Financial Services": 26.4,
    "Healthcare": 38.5,
    "Consumer Defensive": 40.1,
    "Consumer Cyclical": 33.8,
    "Energy": 18.9,
    "Industrials": 29.4,
}

HORIZON_PROFILE_MAP = {

    # ============================
    #  INTRADAY PROFILE
    # ============================
    "intraday": {
        "metrics": {
            # --- Core directional bias & volatility ---
            "vwap_bias": 0.18,            # VWAP-based intraday trend direction
            "price_action": 0.15,         # Candlestick structure / closing strength
            "supertrend_signal": 0.00,    # Trend alignment on short frame
            "stoch_cross": 0.00,          # Momentum exhaustion / entry zone
            "macd_histogram": 0.00,       # Intraday momentum intensity
            "rsi_slope": 0.00,            # Strength of RSI momentum turn

            # --- Liquidity & volatility context ---
            "vol_spike_ratio": 0.08,      # Relative volume compared to mean
            "rvol": 0.05,                 # Confirmation of active participation
            "bb_percent_b": 0.05,         # Position in Bollinger Band (overbought/oversold)
            "adx": 0.05,                  # Trend presence confirmation
            "gap_percent": 0.0,           # <-- REMOVED (set to 0, it's a filter not a score)
            "ichi_cloud": 0.02,
            "momentum_strength": 0.24,
            "volatility_quality": 0.15,

        },
        "penalties": {
            "atr_pct": {"operator": "<", "value": 0.75, "penalty": 0.5},            # Avoid ultra-low volatility days
            "rvol": {"operator": "<", "value": 0.8, "penalty": 0.3},                # Avoid dead volume setups
            "bb_width": {"operator": "<", "value": 3.0, "penalty": 0.2},
            "gap_percent": {"operator": "<", "value": 1.0, "penalty": 0.4},  # <-- Kept as a penalty
            "nifty_trend_score": {"operator": "<", "value": 4, "penalty": 0.3},
        },
        "thresholds": {"buy": 7.5, "hold": 5.5, "sell": 3.5},
        "notes": "Momentum, liquidity and volatility-driven scalping framework. Ideal for 5–15 min setups."
    },


    # ============================
    #  SHORT-TERM PROFILE
    # ============================
    "short_term": {
        "metrics": {
# --- NEW Composites (Priority) ---
        "trend_strength": 0.20,      
        "momentum_strength": 0.05,  
        "volatility_quality": 0.10, 

        # --- Remaining Core Technicals (Must Keep) ---
        "macd_cross": 0.10,          
        "dma_20_50_cross": 0.08,
        "price_vs_200dma_pct": 0.05,
        "reg_slope": 0.05,
        "psar_trend": {"weight": 0.05, "direction": "normal"},
        "ttm_squeeze": {"weight": 0.05, "direction": "normal"},

        # --- RESTORED FUNDAMENTAL / HYBRID CONTEXT ---
        "quarterly_growth": 0.05,    
        "analyst_rating": 0.04,      
        "fcf_yield": 0.03,           
        "pe_vs_sector": 0.03,        
        "nifty_trend_score": 0.04,
        
        # --- RESTORED VOLUME/FLOW CONTEXT ---
        "cmf_signal": 0.05,
        "obv_div": 0.05,
        "rvol": 0.05,
        "bb_percent_b": 0.05,
        "ps_ratio": {"weight": 0.03, "direction": "invert"},
        
        # --- CLEANUP (Set to 0.0) ---
        "supertrend_signal": 0.00,
        "adx": 0.00,
        "rsi_slope": 0.00,
        "bb_width": 0.00,
        },
        "penalties": {
            "days_to_earnings": {"operator": "<", "value": 7, "penalty": 1.0},      # Avoid pre-earnings risk
            "bb_percent_b": {"operator": ">", "value": 0.95, "penalty": 0.3},       # Avoid overextended zones
            "beta": {"operator": ">", "value": 1.8, "penalty": 0.1},                # Avoid extreme volatility
            "52w_position": {"operator": ">", "value": 85, "penalty": 0.15},        # Avoid extended highs
            "adx": {"operator": "<", "value": 20, "penalty": 0.3},                  # Weak trend = penalty
            "vol_spike_ratio": {"operator": "<", "value": 1.2, "penalty": 0.2},     # Avoid dead-volume setups
            "price_vs_200dma_pct": {"operator": "<", "value": 0, "penalty": 0.4},
            "short_interest": {"operator": ">", "value": 10.0, "penalty": 0.2} # <-- ADDED (Risk flag)
        },
        "thresholds": {"buy": 7.0, "hold": 5.5, "sell": 4.0},
        "notes": "Catalyst-driven swing trades combining trend continuation, volume expansion, and short-term growth signals."
    },


    # ============================
    #  LONG-TERM PROFILE
    # ============================
    "long_term": {
        "metrics": {
            # --- Core profitability & balance sheet quality ---
            "roe": 0.08,
            "roce": 0.08,
            "roic": 0.08,
            "net_profit_margin": 0.08,
            "fcf_yield": 0.08,
            "eps_growth_5y": 0.08,
            "fcf_growth_3y": 0.05,
            "revenue_growth_5y": 0.05,
            "piotroski_f": 0.05,
            "promoter_holding": 0.05,
            "dividend_yield": 0.05,
            "dividend_payout": 0.05, # <-- Renamed key to match fundamentals.py
            "earnings_stability": 0.05,
            "current_ratio": 0.04,
            "de_ratio": 0.04,

            # --- Trend confirmations (technical + hybrid) ---
            "price_vs_200dma_pct": 0.05,
            "dma_200_slope": 0.04,
            "rel_strength_nifty": 0.04,
            "supertrend_signal": 0.03,
            "trend_consistency": 0.03,
            "earnings_consistency_index": 0.03,
        },
        "penalties": {
            "fcf_yield": {"operator": "<", "value": 2, "penalty": 0.3},
            "roe": {"operator": "<", "value": 10, "penalty": 0.3},
            "price_vs_200dma_pct": {"operator": "<", "value": 0, "penalty": 0.4},
            "dividend_payout": {"operator": ">", "value": 80.0, "penalty": 0.3}, # <-- Renamed key
            "beta": {"operator": ">", "value": 1.5, "penalty": 0.15},
            "promoter_pledge": {"operator": ">", "value": 15.0, "penalty": 0.2},
            "ocf_vs_profit": {"operator": "<", "value": 0.8, "penalty": 0.4} # <-- ADDED (Critical Quality check)
        },
        "thresholds": {"buy": 7.5, "hold": 6.0, "sell": 4.0},
        "notes": "Quality + stability + compounding trend structure. Ideal for multi-month positional holdings."
    },


    # ============================
    #  MULTIBAGGER PROFILE
    # ============================
    "multibagger": {
        "metrics": {
            # --- Core secular growth + capital efficiency ---
            "revenue_growth_5y": 0.12,
            "eps_growth_5y": 0.10,
            "fcf_growth_3y": 0.08,
            "roic": 0.08,
            "roe": 0.08,
            "peg_ratio": 0.08,
            "r_d_intensity": 0.08,
            "market_cap_cagr": 0.06,
            "promoter_holding": 0.05,
            "piotroski_f": 0.05,
            "earnings_stability": 0.04,          # Prevent over-cyclicals
            "rel_strength_nifty": 0.04,
            "dma_200_slope": 0.04,
            "fundamental_momentum": 0.04,        # Hybrid: accelerating fundamental growth
            "fcf_yield_vs_volatility": 0.04,     # Hybrid: cash flow stability vs volatility
            "institutional_ownership": 0.03,
            "quarterly_growth": 0.03,
            "ocf_vs_profit": {"weight": 0.06, "direction": "normal"} # <-- ADDED (Cash flow check)
        },
        "penalties": {
            "peg_ratio": {"operator": ">", "value": 2.0, "penalty": 0.3},
            "fcf_margin": {"operator": "<", "value": 0, "penalty": 0.4},
            "de_ratio": {"operator": ">", "value": 1.0, "penalty": 0.2},
            "52w_position": {"operator": ">", "value": 85, "penalty": 0.2},
            "market_cap": {"operator": ">", "value": 5e11, "penalty": 0.3},
            "market_cap_floor": {"operator": "<", "value": 5e9, "penalty": 0.3}, # <-- ADDED (Liquidity floor)
            "roe": {"operator": "<", "value": 12, "penalty": 0.2},
            "rel_strength_nifty": {"operator": "<", "value": 0, "penalty": 0.3},
            "institutional_ownership": {"operator": ">", "value": 85, "penalty": 0.3},
            "quarterly_growth": {"operator": "<", "value": 3, "penalty": 0.2},
            "promoter_pledge": {"operator": ">", "value": 10.0, "penalty": 0.3}
        },
        "thresholds": {"buy": 8.0, "hold": 6.5, "sell": 4.5},
        "notes": "High-growth, high-efficiency compounding opportunities with strong sponsor alignment and volatility-adjusted quality."
    }
}
HORIZON_FETCH_CONFIG = {
    "intraday":  {"period": "5d",  "interval": "15m"},
    "short_term": {"period": "3mo", "interval": "1d"},
    "long_term":  {"period": "2y",  "interval": "1wk"},
    "multibagger": {"period": "5y", "interval": "1mo"}
}


QUALITY_WEIGHTS = {
    # Higher is better
    "roe": {"weight": 1.0, "direction": "normal"},
    "roce": {"weight": 1.0, "direction": "normal"},
    "roic": {"weight": 1.0, "direction": "normal"},
    "piotroski_f": {"weight": 1.0, "direction": "normal"},
    "ocf_vs_profit": {"weight": 1.0, "direction": "normal"},
    "interest_coverage": {"weight": 1.0, "direction": "normal"},
    "earnings_stability": {"weight": 1.0, "direction": "normal"},
    "net_profit_margin": {"weight": 1.0, "direction": "normal"},
    # Lower is better
    "de_ratio": {"weight": 1.0, "direction": "invert"},
    "promoter_pledge": {"weight": 1.0, "direction": "normal"},
    "roe_stability": {"weight": 0.10, "direction": "invert"}, # Lower standard deviation = higher score (invert)
    "volatility_quality": {"weight": 0.10, "direction": "normal"},
}
GROWTH_WEIGHTS = {
    "eps_growth_3y": {"weight": 1.0, "direction": "normal"},
    "revenue_growth_5y": {"weight": 1.0, "direction": "normal"},
    "quarterly_growth": {"weight": 1.0, "direction": "normal"},
    "fcf_growth_3y": {"weight": 1.0, "direction": "normal"},
    "market_cap_cagr": {"weight": 1.0, "direction": "normal"}
}
VALUE_WEIGHTS = {
    # Lower is better
    "pe_ratio": {"weight": 1.0, "direction": "invert"},
    "pb_ratio": {"weight": 1.0, "direction": "invert"},
    "peg_ratio": {"weight": 1.0, "direction": "invert"},
    "pe_vs_sector": {"weight": 1.0, "direction": "invert"},
    # Higher is better
    "fcf_yield": {"weight": 1.0, "direction": "normal"},
    "dividend_yield": {"weight": 1.0, "direction": "normal"}
}
MOMENTUM_WEIGHTS = {# 🆕 CORE COMPOSITES (PRIORITY)
    "momentum_strength": {"weight": 0.30, "direction": "normal"}, # RSI, MACD, Stoch Bundle
    "trend_strength": {"weight": 0.40, "direction": "normal"},    # ADX, EMA Slope, ST Bundle
    "volatility_quality": {"weight": 0.10, "direction": "normal"},# New Volatility Setup Score

    # CONTEXTUAL/HYBRID MOMENTUM (MUST KEEP)
    "vwap_bias": {"weight": 0.05, "direction": "normal"},
    "price_action": {"weight": 0.05, "direction": "normal"},
    "nifty_trend_score": {"weight": 0.05, "direction": "normal"}, # Macro Context
    "52w_position": {"weight": 0.05, "direction": "normal"},     # Hybrid/Sentiment Context
    
    # ⚠️ CONTEXTUAL/VOLUME (MUST KEEP)
    "rvol": {"weight": 0.00, "direction": "normal"}, # Weight mass moved to composite, keep key for context
    "obv_div": {"weight": 0.00, "direction": "normal"}, # Weight mass moved to composite, keep key for context
    
    "psar_trend": {"weight": 0.00, "direction": "normal"},
    "ttm_squeeze": {"weight": 0.00, "direction": "normal"},

}

MACRO_INDEX_TICKERS = {
    "nifty50": "^NSEI",       # Nifty 50 Index (Primary Benchmark)
    "nifty100": "^NIFTY100.NS",
    "niftynext50": "^NIFTYNEXT50.NS",
    "nifty500": "^NIFTY500.NS",
    "midcap150": "^NIFTYMCAP150.NS",
    "default": "^NSEI" # Default fallback
}

ENABLE_VOLATILITY_QUALITY = True

# 1. Trend Strength Analyzer Thresholds (Based on ADX Score)
TREND_THRESH = {
    "weak_floor": 20.0,      
    "moderate_floor": 25.0,  
    "strong_floor": 40.0,
    "di_spread_strong": 20.0 # DI+ vs DI- spread for high momentum trend
}

# 2. Volatility Regime Bands (Used by Volatility Quality composite score)
VOL_BANDS = {
    # Absolute volatility levels (percent)
    "low_vol_ceiling": 1.0,  # Below 1.0% ATR/HV is considered very low/tight
    "moderate_vol_ceiling": 2.5, # Below 2.5% ATR/HV is standard swing/high-quality
    "high_vol_floor": 4.0    # Above 4.0% ATR/HV is considered highly volatile/risky
}

# 3. Momentum Slopes Thresholds
RSI_SLOPE_THRESH = {
    "acceleration_floor": 0.05, # Slope > 0.05 (bullish acceleration)
    "deceleration_ceiling": -0.05 # Slope < -0.05 (bearish deceleration)
}

MACD_MOMENTUM_THRESH = {
    "acceleration_floor": 0.5, # MACD Histogram Z-Score > 0.5
    "deceleration_ceiling": -0.5 # MACD Histogram Z-Score < -0.5
}