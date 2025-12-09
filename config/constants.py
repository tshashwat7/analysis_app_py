# config/constants.py
# === Technical Indicator Constants ===

# RSI, MACD, BB etc. existing constants assumed here

# --- Stochastic Oscillator Defaults ---
import os


STOCH_FAST = {"k_period": 5, "d_period": 3, "smooth": 3}
STOCH_SLOW = {"k_period": 14, "d_period": 3, "smooth": 3}
STOCH_THRESHOLDS = {"overbought": 80, "oversold": 20}

# config/constants.py
ENABLE_CACHE = False
ENABLE_CACHE_WARMER = os.getenv("ENABLE_CACHE_WARMER", "false").lower() == "true"
ENABLE_JSON_ENRICHMENT = os.getenv("ENABLE_JSON_ENRICHMENT", "true").lower() == "true"
STOCH_THRESHOLDS = {
    "overbought": 80,
    "oversold": 20,
}

# ATR-based stoploss/target multipliers
ATR_MULTIPLIERS = {
    "short_term": {"tp": 3.0, "sl": 2.0},   # ← NEW (wider stops!)
    "long_term": {"tp": 3.5, "sl": 2.0},
    "multibagger": {"tp": 4.0, "sl": 2.5},
    "intraday": {"tp": 2.0, "sl": 1.5}
}

ATR_HORIZON_CONFIG = {
    "intraday": 10,     # Faster reaction for scalping
    "short_term": 14,   # Standard Swing default
    "long_term": 20,    # Smoother for weekly charts
    "multibagger": 12   # Monthly (1 Year Rolling)
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
    "Free Cash Flow Growth": (
        "fundamentals",
        None,
        "FCF Yield (%)",
    ),  # you don't have CAGR yet
    "Dividend Yield": ("fundamentals", None, "Dividend Yield"),
    "Management Quality": ("extended", None, "Promoter Holding (%)"),
    "EPS Growth Consistency": ("extended", None, "EPS Growth Consistency (5Y CAGR)"),
    "Interest Coverage Ratio": ("fundamentals", None, "Interest Coverage"),
    "Operating Cash Flow vs Net Profit": (
        "extended",
        None,
        "Operating CF vs Net Profit",
    ),
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
    "PEG (Forward)": (
        "fundamentals",
        None,
        "PEG Ratio",
    ),  # fallback since no Forward P/E
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
    "Bond Yield vs Equity Spread": (
        "extended",
        None,
        "10Y Bond Yield (%)",
    ),  # macro proxy
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
    "ema_20_50_cross": 0.8,
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

TECHNICAL_METRIC_MAP = {
    # Price / meta
    "price": "Current Price",
    "prev_close": "Previous Close",

    # Trend / Moving averages “DMA” should be considered Daily Moving Average (SMA),But your dynamic logic never generates dma_XX anymore we're using:EMA for daily (intraday/short_term) WMA-label for weekly (long_term) MMA-label for monthly (multibagger)
    "dma_20": "20 DMA",
    "dma_50": "50 DMA",
    "dma_200": "200 DMA",
    "dma_10": "10 DMA",
    "dma_40": "40 DMA",
    "wma_50": "50WMA",
    "price_vs_50wma_pct": "Price vs 50WMA (%)",
    "price_vs_200dma_pct": "Price vs 200 DMA (%)",
    "dma_200_slope": "200 DMA Slope",
    # === Dynamic MA Mapping (Fully Horizon-Aware) ===
    # Intraday / Short Term (EMA-based)
    "ema_20": "20 EMA (Short-Term Trend)",
    "ema_50": "50 EMA (Medium-Term Trend)",
    "ema_200": "200 EMA (Long-Term Trend)",
    # EMA Crossovers (Intraday / Short-Term)
    "ema_20_50_cross": "EMA 20/50 Crossover",
    "ema_20_200_cross": "EMA 20/200 Crossover",
    "ema_50_200_cross": "EMA 50/200 Crossover",
    # EMA Trend Stacking
    "ema_20_50_200_trend": "EMA Trend Alignment (20 > 50 > 200)",
    # Long-Term Horizon (Weekly MAs) — WMA prefix, SMA math
    "wma_10": "10-Week MA",
    "wma_40": "40-Week MA",
    "wma_50": "50-Week MA",
    # Weekly Crossover
    "wma_10_40_cross": "Weekly MA Crossover (10/40)",
    # Weekly Trend Stacking
    "wma_10_40_50_trend": "Weekly Trend Alignment (10 > 40 > 50)",
    # Multibagger Horizon (Monthly MAs) — MMA prefix, SMA math
    "mma_6": "6-Month MA",
    "mma_12": "12-Month MA",
    # Monthly Crossover
    "mma_6_12_cross": "Monthly MA Crossover (6/12)",
    # Monthly Trend Stacking
    "mma_6_12_12_trend": "Monthly Trend Alignment (6 > 12 > 12)",
    # Generic Crossover Trend Key (Used by dynamic MA Trend)
    "ma_cross_trend": "Composite MA Trend Signal",
    "ema_20_slope": "20 EMA Slope",
    "ema_50_slope": "50 EMA Slope",
    "wma_50_slope": "50 WMA Slope",
    "mma_12_slope": "12-Month MA Slope",
    "ma_cross_setup": "MA Crossover Setup",
    "ma_trend_setup": "MA Trend Setup",

    # Momentum
    "rsi": "RSI",
    "rsi_slope": "RSI Slope",
    "dma_20_50_cross": "20/50 DMA Cross",
    "dma_10_40_cross": "10/40 DMA Cross",
    "short_ma_cross": "Short MA Cross",
    "macd": "MACD",
    "macd_cross": "MACD Cross",
    "macd_hist_z": "MACD Hist Z-Score",
    "macd_histogram": "MACD Histogram (Raw)",
    "mfi": "MFI",
    "stoch_k": "Stoch %K",
    "stoch_d": "Stoch %D",
    "stoch_cross": "Stoch Crossover",
    "cci": "CCI",
    "adx": "ADX",
    "adx_signal": "ADX Signal",
    "di_plus": "DI+",
    "di_minus": "DI-",

    # Volatility / volume
    "atr_14": "ATR (14)",
    "atr_pct": "ATR %",
    "true_range": "True Range (Raw)",
    "true_range_pct": "True Range % of Price",
    "hv_10": "Historical Volatility (10D)",
    "hv_20": "Historical Volatility (20D)",
    "rvol": "Relative Volume (RVOL)",
    "vol_spike_ratio": "Volume Spike Ratio",
    "vol_spike_signal": "Volume Spike Signal",
    "vol_trend": "Volume Trend",
    "vpt": "VPT",
    "cmf_signal": "Chaikin Money Flow (CMF)",
    "obv_div": "OBV Divergence",

    # Bands / Channel
    "bb_high": "BB High",
    "bb_mid": "BB Mid",
    "bb_low": "BB Low",
    "bb_width": "BB Width",
    "bb_percent_b": "Bollinger %B",
    "ttm_squeeze": "TTM Squeeze Signal",
    "kc_upper": "Keltner Upper",
    "kc_lower": "Keltner Lower",
    "donchian_signal": "Donchian Channel Breakout",
    "ichi_cloud": "Ichimoku Cloud",
    "ichi_span_a": "Ichimoku Span A",
    "ichi_span_b": "Ichimoku Span B",
    "ichi_tenkan": "Tenkan-sen",
    "ichi_kijun": "Kijun-sen",

    # Levels / pivots
    "pivot_point": "Pivot Point (Daily)",
    "resistance_1": "Resistance 1 (Fib)",
    "resistance_2": "Resistance 2 (Fib)",
    "resistance_3": "Resistance 3 (Fib)",
    "support_1": "Support 1 (Fib)",
    "support_2": "Support 2 (Fib)",
    "support_3": "Support 3 (Fib)",
    "entry_confirm": "Entry Price (Confirm)",
    "gap_percent": "Gap %",

    # Misc / signals
    "psar_trend": "Parabolic SAR Trend",
    "psar_level": "PSAR Level",
    "supertrend_signal": "SuperTrend Signal",
    "supertrend_value": "Supertrend Value",
    "price_action": "Price Action",
    "vwap": "VWAP",
    "vwap_bias": "VWAP Bias",

    # Relative / benchmark
    "rel_strength_nifty": "Relative Strength vs NIFTY (%)",
    "nifty_trend_score": "NIFTY Trend Score",

    # Composite placeholders (some are computed in signal_engine but include them so profile keys don't break)
    "trend_strength": "Trend Strength",
    "momentum_strength": "Momentum Strength",
    "volatility_quality": "Volatility Quality",
    "fundamental_momentum": "Fundamental Momentum",
    "price_vs_avg": "Price vs Average",

    # Utility / reporting
    "sl_2x_atr": "Suggested SL (2xATR)",
    "technical_score": "Technical Score",
    "Horizon": "Horizon",
    "wick_rejection": "Wick Rejection",
    "atr_dynamic": "Dynamic ATR",
    "sl_atr_dynamic": "Stop Loss (Dynamic ATR)",
    "risk_per_share_pct": "Risk Per Share (%)",
    "atr_sma_ratio": "ATR/SMA Ratio",

    #pattern Key
    "darvas_box": "Darvas Box Pattern",
    "cup_handle": "Cup & Handle Pattern",
    "flag_pennant": "Flag/Pennant Pattern",
    "bollinger_squeeze": "Bollinger Squeeze",
    "golden_cross": "Golden/Death Cross",
    "double_top_bottom": "Double Top/Bottom",
    "three_line_strike": "Three-Line Strike",
    "minervini_stage2": "Minervini VCP / Stage 2",
    "ichimoku_signals": "Ichimoku Signals",





}

CORE_TECHNICAL_SETUP_METRICS = [
        "rsi", 
        "ema_20", 
        "ema_200", 
        "bb_high", 
        "bb_low", 
        "ttm_squeeze", 
        "atr_14",          # Needed for Stop Loss
        "price_action",    # Good context
        "volatility_quality" # Needed for Confidence Score
    ]
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

FUNDAMENTAL_ALIAS_MAP = {
    "pe_ratio": "P/E Ratio",
    "pb_ratio": "Price to Book (P/B)",
    "peg_ratio": "PEG Ratio",
    "ps_ratio": "Price-to-Sales (P/S)",
    "pe_vs_sector": "P/E vs Sector",
    "fcf_yield": "FCF Yield (%)",
    "dividend_yield": "Dividend Yield (%)",
    "dividend_payout": "Dividend Payout (%)",
    "market_cap": "Market Cap",
    "market_cap_cagr": "Market Cap CAGR (%)",
    # Profitability / returns
    "roe_history": "ROE History",
    "roe": "Return on Equity (ROE)",
    "roce": "Return on Capital Employed (ROCE)",
    "roic": "Return on Invested Capital (ROIC)",
    "net_profit_margin": "Net Profit Margin (%)",
    "operating_margin": "Operating Margin (%)",
    "ebitda_margin": "EBITDA Margin (%)",
    "fcf_margin": "FCF Margin (%)",
    # Growth
    "revenue_growth_5y": "Revenue Growth (5Y CAGR)",
    "profit_growth_3y": "Profit Growth (3Y CAGR)",
    "eps_growth_5y": "EPS Growth (5Y CAGR)",
    "eps_growth_3y": "EPS Growth (3Y CAGR)",
    "fcf_growth_3y": "FCF Growth (3Y CAGR)",
    "quarterly_growth": "Quarterly Growth (EPS/Rev)",
    # Health / liquidity
    "de_ratio": "Debt to Equity",
    "interest_coverage": "Interest Coverage Ratio",
    "current_ratio": "Current Ratio",
    "ocf_vs_profit": "Operating CF vs Net Profit",
    # Quality / efficiency
    "piotroski_f": "Piotroski F-Score",
    "asset_turnover": "Asset Turnover Ratio",
    "r_d_intensity": "R&D Intensity (%)",
    "earnings_stability": "Earnings Stability",
    # Ownership / market
    "promoter_holding": "Promoter Holding (%)",
    "promoter_pledge": "Promoter Pledge (%)",
    "institutional_ownership": "Institutional Ownership (%)",
    "short_interest": "Short Interest",
    "analyst_rating": "Analyst Rating (Momentum)",
    "52w_position": "52W Position (off-high %)",
    "beta": "Beta",
    "days_to_earnings": "Days to Next Earnings",
    "ps_ratio": "Price-to-Sales (P/S)",
    # reporting/meta
    "base_score": "Base Fundamental Score",
    "final_score": "Final Fundamental Score",
    "_meta": "Meta",
    "52w_high": "52 week high",
    "52w_low": "52 week low",

}
FUNDAMENTAL_FIELD_CANDIDATES = {
    # Income Statement
    "revenue": [
        "Total Revenue",
        "Revenue",
        "totalRevenue",
        "Sales",
        "Net Sales",
        "Operating Revenue",
        "Total Sales",
        "Gross Sales",
    ],
    "net_income": [
        "Net Income",
        "netIncome",
        "NetIncome",
        "Profit After Tax",
        "Net Profit",
        "Profit",
        "PAT",
        "Net Loss",
        "Net Income Common Stockholders",
    ],
    "operating_income": [
        "Operating Income",
        "EBIT",
        "Ebit",
        "Operating Profit",
        "OperatingProfit",
        "Profit from Operations",
    ],
    "ebit": [
        "EBIT",
        "Ebit",
        "Operating Income",
        "Operating Profit",
        "Profit from Operations",
    ],
    "ebitda": [
        "EBITDA",
        "Ebitda",
        "Operating Profit Before Depreciation",
        "Normalized EBITDA",
    ],
    "cogs": [
        "Cost Of Revenue",
        "Cost of Goods Sold",
        "COGS",
        "Total Expenses",
        "Operating Expense",
    ],
    "interest_expense": [
        "Interest Expense",
        "Interest And Debt Expense",
        "Finance Cost",
        "Interest",
        "Total Interest Expense",
    ],
    "tax_expense": ["Income Tax Expense", "Tax Provision", "Total Tax Expense", "Tax"],
    "pre_tax_income": [
        "Pretax Income",
        "Income Before Tax",
        "Income Before Tax Expense",
        "PretaxProfit",
    ],
    # Balance Sheet
    "total_assets": ["Total Assets", "totalAssets", "Assets"],
    "current_assets": [
        "Total Current Assets",
        "totalCurrentAssets",
        "Current Assets",
        "currentAssets",
    ],
    "current_liabilities": [
        "Total Current Liabilities",
        "totalCurrentLiabilities",
        "Current Liabilities",
        "currentLiabilities",
    ],
    "cash_equivalents": [
        "Cash And Cash Equivalents",
        "cashAndCashEquivalents",
        "Cash",
        "Cash Balance",
        "Cash & Equivalents",
        "Cash & Bank Balances",
    ],
    "total_liabilities": [
        "Total Liabilities",
        "Total Current Liabilities",
        "Liabilities",
        "Total Liab",
    ],
    "total_equity": [
        "Total Equity",
        "Total Stockholders Equity",
        "totalStockholdersEquity",
        "Shareholders Equity",
        "Equity",
        "Stockholders Equity",
        "Shareholder Equity",
        "Total Common Equity",
        "Total Stockholder Equity",
        "Shareholder's funds",
        "Total shareholders' funds",
    ],
    "total_debt": [
        "Total Debt",
        "totalDebt",
        "Long Term Debt",
        "Short Long Term Debt",
        "Long Term Borrowings",
        "Short Term Borrowings",
        "Debt",
        "Borrowings",
    ],
    "pure_borrowings": [
        "Short Term Borrowings",
        "ShortTermBorrowings",
        "Short Term Debt",
        "ShortTermDebt",
        "Long Term Borrowings",
        "LongTermBorrowings",
        "Long Term Debt",
        "LongTermDebt",
        "Borrowings",
        "borrowings",
    ],
    # Cash Flow Statement
    "ocf": [
        "Total Cash From Operating Activities",
        "totalCashFromOperatingActivities",
        "Operating Cash Flow",
        "Cash Flow From Operating Activities",
    ],
    "capex": [
        "Capital Expenditures",
        "capitalExpenditures",
        "CapEx",
        "Purchase Of Fixed Assets",
    ],
    "free_cash_flow": [
        "Free Cash Flow",
        "freeCashflow",
        "freeCashFlow",
        "FCF",
        "Free Cash Flow (FCF)",
    ],
    # Other metrics / ratios
    "rd_expense": [
        "Research And Development",
        "Research Development",
        "Research and Development Expense",
        "R&D",
        "Rnd",
    ],
    "eps": ["EPS", "Diluted EPS", "Basic EPS", "Earnings Per Sare", "eps"],
    "shares_outstanding": [
        "Basic Average Shares",
        "Shares Outstanding",
        "Weighted Average Shares",
        "Shares",
    ],
    "gross_profit": ["Gross Profit", "GrossIncome"],
    "market_cap": ["marketCap", "Market Capitalization", "market_cap", "Market Cap"],
    "book_value": ["bookValue", "Book Value", "Book value per share"],
    "dividend": ["Dividends Paid", "dividendRate", "Cash Dividends Paid"],
    "fcf_yield": ["Free Cash Flow Yield", "fcfYield"],
    "promoter_holding": [
        "heldPercentInsiders",
        "insiderPercent",
        "insidersPercent",
        "Insider Ownership",
    ],
    "institutional_ownership": [
        "heldPercentInstitutions",
        "institutionPercent",
        "institutionsPercent",
        "Institutional Ownership",
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
    "intraday": {
        "metrics": {
            "vwap_bias": 0.18,
            "price_action": 0.15,
            "rsi_slope": 0.12,
            "vol_spike_ratio": 0.08,
            "rvol": 0.05,
            "bb_percent_b": 0.05,
            "adx": 0.05,
            "ttm_squeeze": 0.04,
            "momentum_strength": 0.20,      # composite (signal_engine should compute and add into indicators before scoring)
            "volatility_quality": 0.15,
        },
        "penalties": {
            "atr_pct": {"operator": "<", "value": 0.75, "penalty": 0.5},
            "rvol": {"operator": "<", "value": 0.8, "penalty": 0.3},
            "bb_width": {"operator": "<", "value": 3.0, "penalty": 0.2},
            "gap_percent": {"operator": "<", "value": 1.0, "penalty": 0.4},
            "nifty_trend_score": {"operator": "<", "value": 4, "penalty": 0.3},
        },
        "thresholds": {"buy": 7.5, "hold": 5.5, "sell": 3.5},
    },

    "short_term": {
        "metrics": {
            "trend_strength": 0.20,
            "momentum_strength": 0.05,
            "volatility_quality": 0.10,
            "macd_cross": 0.10,
            "ema_20_50_cross": 0.08,
            "price_vs_200dma_pct": 0.05,
            "psar_trend": 0.05,
            "ttm_squeeze": 0.05,
            "quarterly_growth": 0.05,
            "analyst_rating": 0.04,
            "fcf_yield": 0.03,
            "pe_vs_sector": 0.03,
            "nifty_trend_score": 0.04,
            "cmf_signal": 0.05,
            "obv_div": 0.05,
            "rvol": 0.05,
            "bb_percent_b": 0.05,
            "ps_ratio": 0.03,
            "rsi_slope": 0.05,      # Helps detect momentum acceleration
            "ema_20_slope": 0.05,   # Helps detect trend velocity
        },
        "penalties": {
            "days_to_earnings": {"operator": "<", "value": 7, "penalty": 1.0},
            "bb_percent_b": {"operator": ">", "value": 0.95, "penalty": 0.3},
            "beta": {"operator": ">", "value": 1.8, "penalty": 0.1},
            "52w_position": {"operator": ">", "value": 85, "penalty": 0.15},
            "adx": {"operator": "<", "value": 20, "penalty": 0.3},
            "vol_spike_ratio": {"operator": "<", "value": 1.2, "penalty": 0.2},
            "price_vs_200dma_pct": {"operator": "<", "value": 0, "penalty": 0.4},
            "short_interest": {"operator": ">", "value": 10.0, "penalty": 0.2},
        },
        "thresholds": {"buy": 7.0, "hold": 5.5, "sell": 4.0},
    },

    "long_term": {
        "metrics": {
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
            "dividend_payout": 0.05,
            "earnings_stability": 0.05,
            "current_ratio": 0.04,
            "de_ratio": 0.04,
            "price_vs_200dma_pct": 0.05,
            "dma_200_slope": 0.04,
            "rel_strength_nifty": 0.04,
            "supertrend_signal": 0.03,
        },
        "penalties": {
            "fcf_yield": {"operator": "<", "value": 2, "penalty": 0.3},
            "roe": {"operator": "<", "value": 10, "penalty": 0.3},
            "price_vs_200dma_pct": {"operator": "<", "value": 0, "penalty": 0.4},
            "dividend_payout": {"operator": ">", "value": 80.0, "penalty": 0.3},
            "beta": {"operator": ">", "value": 1.5, "penalty": 0.15},
            "promoter_pledge": {"operator": ">", "value": 15.0, "penalty": 0.2},
            "ocf_vs_profit": {"operator": "<", "value": 0.8, "penalty": 0.4},
        },
        "thresholds": {"buy": 7.5, "hold": 6.0, "sell": 4.0},
    },

    "multibagger": {
        "metrics": {
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
            "earnings_stability": 0.04,  # Prevent over-cyclicals
            "rel_strength_nifty": 0.04,
            "mma_12_slope": 0.04,
            "fundamental_momentum": 0.04,
            "institutional_ownership": 0.03,
            "quarterly_growth": 0.03,
            "ocf_vs_profit": 0.06,
        },
        "penalties": {
            "peg_ratio": {"operator": ">", "value": 2.0, "penalty": 0.3},
            "fcf_margin": {"operator": "<", "value": 0, "penalty": 0.4},
            "de_ratio": {"operator": ">", "value": 1.0, "penalty": 0.2},
            "52w_position": {"operator": ">", "value": 85, "penalty": 0.2},
            "market_cap": {"operator": ">", "value": 5e11, "penalty": 0.3},
            "market_cap_floor": {"operator": "<", "value": 5e9, "penalty": 0.3},
            "roe": {"operator": "<", "value": 12, "penalty": 0.2},
            "rel_strength_nifty": {"operator": "<", "value": 0, "penalty": 0.3},
            "institutional_ownership": {"operator": ">", "value": 85, "penalty": 0.3},
            "quarterly_growth": {"operator": "<", "value": 3, "penalty": 0.2},
            "promoter_pledge": {"operator": ">", "value": 10.0, "penalty": 0.3},
        },
        "thresholds": {"buy": 8.0, "hold": 6.5, "sell": 4.5},
    }
}

HORIZON_FETCH_CONFIG = {
    "intraday": {
        "period": "1mo",   # CHANGED from '5d'. Gives ~500 bars (25 * 20 days). Plenty for EMA200 + Warmup.
        "interval": "15m", 
        "label": "Intraday"
    },
    "short_term": {
        "period": "5y",    # CHANGED from '3mo' to 2y. Gives ~250 candles. Enough for 200 DMA.
        "interval": "1d", 
        "label": "Short Term"
    },
    "long_term": {
        "period": "5y",    # changed. from 2y to 5y ~104 weekly bars. (Note: WMA 200 needs ~4 years, but you use WMA 50 here, so 2y is fine)
        "interval": "1wk", 
        "label": "Long Term"
    },
    "multibagger": {
        "period": "10y",   # CHANGED from '5y' to be safe, though 5y (60 months) is usually enough for MMA 12.
        "interval": "1mo", 
        "label": "Multibagger"
    },
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
    "roe_stability": {
        "weight": 0.10,
        "direction": "invert",
    },  # Lower standard deviation = higher score (invert)
    "volatility_quality": {"weight": 0.10, "direction": "normal"},
}
GROWTH_WEIGHTS = {
    "eps_growth_3y": {"weight": 1.0, "direction": "normal"},
    "revenue_growth_5y": {"weight": 1.0, "direction": "normal"},
    "quarterly_growth": {"weight": 1.0, "direction": "normal"},
    "fcf_growth_3y": {"weight": 1.0, "direction": "normal"},
    "market_cap_cagr": {"weight": 1.0, "direction": "normal"},
}
VALUE_WEIGHTS = {
    # Lower is better
    "pe_ratio": {"weight": 1.0, "direction": "invert"},
    "pb_ratio": {"weight": 1.0, "direction": "invert"},
    "peg_ratio": {"weight": 1.0, "direction": "invert"},
    "pe_vs_sector": {"weight": 1.0, "direction": "invert"},
    # Higher is better
    "fcf_yield": {"weight": 1.0, "direction": "normal"},
    "dividend_yield": {"weight": 1.0, "direction": "normal"},
}
MOMENTUM_WEIGHTS = {  # 🆕 CORE COMPOSITES (PRIORITY)
    "momentum_strength": {
        "weight": 0.30,
        "direction": "normal",
    },  # RSI, MACD, Stoch Bundle
    "trend_strength": {
        "weight": 0.40,
        "direction": "normal",
    },  # ADX, EMA Slope, ST Bundle
    "volatility_quality": {
        "weight": 0.10,
        "direction": "normal",
    },  # New Volatility Setup Score
    # CONTEXTUAL/HYBRID MOMENTUM (MUST KEEP)
    "vwap_bias": {"weight": 0.05, "direction": "normal"},
    "price_action": {"weight": 0.05, "direction": "normal"},
    "nifty_trend_score": {"weight": 0.05, "direction": "normal"},  # Macro Context
    "52w_position": {"weight": 0.05, "direction": "normal"},  # Hybrid/Sentiment Context
    # ⚠️ CONTEXTUAL/VOLUME (MUST KEEP)
    "rvol": {
        "weight": 0.00,
        "direction": "normal",
    },  # Weight mass moved to composite, keep key for context
    "obv_div": {
        "weight": 0.00,
        "direction": "normal",
    },  # Weight mass moved to composite, keep key for context
    "psar_trend": {"weight": 0.00, "direction": "normal"},
    "ttm_squeeze": {"weight": 0.00, "direction": "normal"},
}

INDEX_TICKERS = {
    # Broad Market Indices (Comprehensive Coverage)
    "nifty50": "^NSEI",  # Nifty 50 Index (Primary Benchmark)
    "nifty100": "^CNX100",  # Top 100 stocks
    "niftynext50": "^NSMIDCP",  # Top 51-100 stocks
    "nifty500": "^CRSLDX",  # Top 500 stocks (Wide Coverage)
    "midcap150": "NIFTYMIDCAP150.NS",  # Mid-cap segment
    "smallcap100": "^CNXSC",  # Small-cap segment
    "smallcap250": "NIFTYSMLCAP250.NS",  # Broader small-cap coverage
    "microcap250": "NIFTY_MICROCAP250.NS",  # Micro-cap segment
    # Sectoral Indices (Industry-specific Insight)
    "niftybank": "^NSEBANK",  # Banking sector
    "niftyit": "^CNXIT.NS",  # Information Technology sector
    "niftypharma": "^CNXPHARMA.NS",  # Pharmaceutical sector
    "niftyfmcg": "^CNXFMCG.NS",  # Fast Moving Consumer Goods sector
    "niftyauto": "^CNXAUTO.NS",  # Automotive sector
    "niftyrealty": "^CNXREALTY.NS",  # Realty sector
    "niftyinfra": "^CNXINFRA.NS",  # Infrastructure sector
    # Bombay Stock Exchange (BSE) Indices
    "sensex": "^BSESN",  # BSE Sensex (BSE Benchmark)
    "bsemidcap": "^BSEMC.BO",  # BSE Mid-cap segment
    "bsesmallcap": "^BSESC.BO",  # BSE Small-cap segment
    # Default fallback
    "default": "^NSEI",
}

ENABLE_VOLATILITY_QUALITY = True

# 1. Trend Strength Analyzer Thresholds (Based on ADX Score)
TREND_THRESH = {
    "weak_floor": 20.0,
    "moderate_floor": 25.0,
    "strong_floor": 40.0,
    "di_spread_strong": 20.0,  # DI+ vs DI- spread for high momentum trend
}

# 2. Volatility Regime Bands (Used by Volatility Quality composite score)
VOL_BANDS = {
    # Absolute volatility levels (percent)
    "low_vol_ceiling": 1.0,  # Below 1.0% ATR/HV is considered very low/tight
    "moderate_vol_ceiling": 2.5,  # Below 2.5% ATR/HV is standard swing/high-quality
    "high_vol_floor": 4.0,  # Above 4.0% ATR/HV is considered highly volatile/risky
}

# 3. Momentum Slopes Thresholds
RSI_SLOPE_THRESH = {
    # Default fallback
    "default": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
    
    # Intraday: Requires sharper moves to filter noise
    "intraday": {"acceleration_floor": 0.10, "deceleration_ceiling": -0.10},
    
    # Short Term: Standard Swing
    "short_term": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
    
    # Long Term: Slower moves are significant
    "long_term": {"acceleration_floor": 0.03, "deceleration_ceiling": -0.03},
    
    # Multibagger: Monthly charts move very slowly
    "multibagger": {"acceleration_floor": 0.02, "deceleration_ceiling": -0.02},
}

MACD_MOMENTUM_THRESH = {
    "acceleration_floor": 0.5,  # MACD Histogram Z-Score > 0.5
    "deceleration_ceiling": -0.5,  # MACD Histogram Z-Score < -0.5
}
