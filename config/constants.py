# config/constants.py

import os

#todo remove
STOCH_FAST = {"k_period": 5, "d_period": 3, "smooth": 3}
STOCH_SLOW = {"k_period": 14, "d_period": 3, "smooth": 3}
STOCH_THRESHOLDS = {"overbought": 80, "oversold": 20}

# config/constants.py
ENABLE_CACHE = False
ENABLE_CACHE_WARMER = os.getenv("ENABLE_CACHE_WARMER", "false").lower() == "true"
ENABLE_JSON_ENRICHMENT = os.getenv("ENABLE_JSON_ENRICHMENT", "true").lower() == "true"
ENABLE_VOLATILITY_QUALITY = True
NSE_UNIVERSE_CSV = "data/nifty500.csv"


# ==========================OLD==================================
ADX_HORIZON_CONFIG = {
    "intraday": 10,     # Fast
    "short_term": 14,   # Standard
    "long_term": 14,
    "multibagger": 20   # Slow/Smooth
}

STOCH_HORIZON_CONFIG = {
    "intraday": {"k": 8, "d": 3, "smooth": 3},   # Faster
    "short_term": {"k": 14, "d": 3, "smooth": 3}, # Standard
    "long_term": {"k": 14, "d": 3, "smooth": 3},
    "multibagger": {"k": 21, "d": 5, "smooth": 5} # Very smooth
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
    "ATR (Volatility)": ("quick_score", None, "atr14"),
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
#todo remove
TECHNICAL_WEIGHTS = {
    "rsi": 1.0,
    "macdCross": 1.0,
    "macdHistZ": 0.8,
    "priceVsMaSlowPct": 1.0,
    "adx": 1.0,
    "vwapBias": 0.8,
    "volTrend": 0.6,
    "rvol": 0.6,
    "stochK": 0.6,
    "bbLow": 0.4,
    "bbWidth": 0.3,
    "entryConfirm": 0.5,
    "ema20_50Cross": 0.8,
    "dma200Slope": 0.8,
    "ichiCloud": 1.0,
    "obvDiv": 0.6,
    "atr14": 0.8,
    "volSpikeRatio": 0.5,
    "relStrengthNifty": 0.6,
    "priceAction": 0.7,
    "supertrendsignal": 1.0,
    "cci": 0.6,
    "bbpercentb": 0.4,
    "cmfSignal": 0.6,
    "donchianSignal": 0.8,
    "regSlope": 0.8,
}

TECHNICAL_METRIC_MAP = {
    # Price / meta
    "price": "Current Price",
    "prevClose": "Previous Close",

    # Trend / Moving averages “DMA” should be considered Daily Moving Average (SMA),But your dynamic logic never generates dma_XX anymore we're using:EMA for daily (intraday/short_term) WMA-label for weekly (long_term) MMA-label for monthly (multibagger)
    "dma20": "20 DMA",
    "dma50": "50 DMA",
    "dma200": "200 DMA",
    "dma10": "10 DMA",
    "dma40": "40 DMA",
    "wma50": "50WMA",
    "price_vs_50wma_pct": "Price vs 50WMA (%)",
    "priceVsMaSlowPct": "Price vs 200 DMA (%)",
    "dma200Slope": "200 DMA Slope",
    # === Dynamic MA Mapping (Fully Horizon-Aware) ===
    # Intraday / Short Term (EMA-based)
    "ema20": "20 EMA (Short-Term Trend)",
    "ema50": "50 EMA (Medium-Term Trend)",
    "ema200": "200 EMA (Long-Term Trend)",
    # EMA Crossovers (Intraday / Short-Term)
    "ema20_50Cross": "EMA 20/50 Crossover",
    "ema_20_200_cross": "EMA 20/200 Crossover",
    "ema_50_200_cross": "EMA 50/200 Crossover",
    # EMA Trend Stacking
    "ema_20_50_200_trend": "EMA Trend Alignment (20 > 50 > 200)",
    # Long-Term Horizon (Weekly MAs) — WMA prefix, SMA math
    "wma10": "10-Week MA",
    "wma40": "40-Week MA",
    "wma50": "50-Week MA",
    # Weekly Crossover
    "wma_10_40_cross": "Weekly MA Crossover (10/40)",
    # Weekly Trend Stacking
    "wma_10_40_50_trend": "Weekly Trend Alignment (10 > 40 > 50)",
    # Multibagger Horizon (Monthly MAs) — MMA prefix, SMA math
    "mma6": "6-Month MA",
    "mma12": "12-Month MA",
    # Monthly Crossover
    "mma_6_12_cross": "Monthly MA Crossover (6/12)",
    # Monthly Trend Stacking
    "mma_6_12_12_trend": "Monthly Trend Alignment (6 > 12 > 12)",
    # Generic Crossover Trend Key (Used by dynamic MA Trend)
    "maCrossTrend": "Composite MA Trend Signal",
    "ema20Slope": "20 EMA Slope",
    "ema50Slope": "50 EMA Slope",
    "wma50Slope": "50 WMA Slope",
    "mma12Slope": "12-Month MA Slope",
    "maCrossSetup": "MA Crossover Setup",
    "maTrendSetup": "MA Trend Setup",

    # Momentum
    "rsi": "RSI",
    "rsislope": "RSI Slope",
    "dma20_50Cross": "20/50 DMA Cross",
    "dma10_40Cross": "10/40 DMA Cross",
    "shortMaCross": "Short MA Cross",
    "macd": "MACD",
    "macdCross": "MACD Cross",
    "macdHistZ": "MACD Hist Z-Score",
    "macdHistogram": "MACD Histogram (Raw)",
    "mfi": "MFI",
    "stochK": "Stoch %K",
    "stochD": "Stoch %D",
    "stoch_cross": "Stoch Crossover",
    "cci": "CCI",
    "adx": "ADX",
    "adx_signal": "ADX Signal",
    "diPlus": "DI+",
    "diMinus": "DI-",

    # Volatility / volume
    "atr14": "atr14",
    "atrPct": "ATR %",
    "trueRange": "True Range (Raw)",
    "trueRangePct": "True Range % of Price",
    "hv10": "Historical Volatility (10D)",
    "hv20": "Historical Volatility (20D)",
    "rvol": "Relative Volume (RVOL)",
    "volSpikeRatio": "Volume Spike Ratio",
    "volSpikeSignal": "Volume Spike Signal",
    "volTrend": "Volume Trend",
    "vpt": "VPT",
    "cmfSignal": "Chaikin Money Flow (CMF)",
    "obvDiv": "OBV Divergence",

    # Bands / Channel
    "bbHigh": "BB High",
    "bbMid": "BB Mid",
    "bbLow": "BB Low",
    "bbWidth": "BB Width",
    "bbpercentb": "Bollinger %B",
    "ttmSqueeze": "TTM Squeeze Signal",
    "kcUpper": "Keltner Upper",
    "kcLower": "Keltner Lower",
    "donchianSignal": "Donchian Channel Breakout",
    "ichiCloud": "Ichimoku Cloud",
    "ichiSpanA": "Ichimoku Span A",
    "ichiSpanB": "Ichimoku Span B",
    "ichiTenkan": "Tenkan-sen",
    "ichiKijun": "Kijun-sen",

    # Levels / pivots
    "pivotPoint": "Pivot Point (Daily)",
    "resistance1": "Resistance 1 (Fib)",
    "resistance2": "Resistance 2 (Fib)",
    "resistance3": "Resistance 3 (Fib)",
    "support1": "Support 1 (Fib)",
    "support2": "Support 2 (Fib)",
    "support3": "Support 3 (Fib)",
    "entryConfirm": "Entry Price (Confirm)",
    "gapPercent": "Gap %",

    # Misc / signals
    "psarTrend": "Parabolic SAR Trend",
    "psarLevel": "PSAR Level",
    "supertrendsignal": "SuperTrend Signal",
    "supertrendValue": "Supertrend Value",
    "priceAction": "Price Action",
    "vwap": "VWAP",
    "vwapBias": "VWAP Bias",

    # Relative / benchmark
    "relStrengthNifty": "Relative Strength vs NIFTY (%)",
    "niftyTrendScore": "NIFTY Trend Score",

    # Composite placeholders (some are computed in signal_engine but include them so profile keys don't break)
    "fundamentalMomentum": "Fundamental Momentum",
    "priceVsAvg": "Price vs Average",

    # Utility / reporting
    "sl2xAtr": "Suggested SL (2xATR)",
    "technicalScore": "Technical Score",
    "Horizon": "Horizon",
    "wickRejection": "Wick Rejection",
    "atrDynamic": "Dynamic ATR",
    "slAtrDynamic": "Stop Loss (Dynamic ATR)",
    "riskPerSharePct": "Risk Per Share (%)",
    "atrSmaRatio": "ATR/SMA Ratio",

    #pattern Key
    "darvasBox": "Darvas Box Pattern",
    "cupHandle": "Cup & Handle Pattern",
    "flagPennant": "Flag/Pennant Pattern",
    "bollingerSqueeze": "Bollinger Squeeze",
    "goldenCross": "Golden/Death Cross",
    "bullishNecklinePattern": "Bullish Neckline Breakout",
    "bearishNecklinePattern": "Bearish Neckline Breakdown",
    "threeLineStrike": "Three-Line Strike",
    "minerviniStage2": "Minervini VCP / Stage 2",
    "ichimokuSignals": "Ichimoku Signals",
    "trendStrength": "Trend Strength (Composite ADX+Slope+DI)",
    "momentumStrength": "Momentum Strength (Composite RSI+MACD+Stoch)",
    "volatilityQuality": "Volatility Quality (0-10 Score)",
    
    # Universal MA Keys
    "maFastSlope": "Primary MA Slope (Horizon-Aware)",
    "maMidSlope": "Secondary MA Slope",
    "maSlowSlope": "Tertiary MA Slope",




}

CORE_TECHNICAL_SETUP_METRICS = [
        "rsi", 
        "ema20", 
        "ema200", 
        "bbHigh", 
        "bbLow", 
        "ttmSqueeze", 
        "atr14",          # Needed for Stop Loss
        "priceAction",    # Good context
        "volatilityQuality" # Needed for Confidence Score
    ]
# -------------------------
# Updated fundamental weights (short keys)
# -------------------------
FUNDAMENTAL_WEIGHTS = {
    # --- Valuation (20%) ---
    "peRatio": 0.05,
    "pbRatio": 0.04,
    "pegRatio": 0.03,
    "fcfYield": 0.05,
    "dividendYield": 0.03,
    # --- Profitability / Returns (25%) ---
    "roe": 0.10,
    "roce": 0.07,
    "roic": 0.08,
    # --- Leverage / Liquidity (15%) ---
    "deRatio": 0.05,
    "interestCoverage": 0.05,
    "currentRatio": 0.03,
    "ocfVsProfit": 0.02,
    # --- Efficiency / Quality (20%) ---
    "assetTurnover": 0.04,
    "piotroskiF": 0.07,
    "RDIntensity": 0.04,
    "earningsStability": 0.05,
    # --- Growth (15%) ---
    "epsgrowth5y": 0.06,
    "fcfGrowth3y": 0.05,
    "marketCapCagr": 0.04,
    # --- Ownership / Market Sentiment (5%) ---
    "promoterHolding": 0.015,
    "institutionalOwnership": 0.015,
    "beta": 0.01,
    "position52w": 0.01,
    "dividendPayout": 0.03,
    "yieldVsAvg": 0.02,
}

# Safety normalization (ensures sum = 1.0)
_total = sum(FUNDAMENTAL_WEIGHTS.values())
if abs(_total - 1.0) > 1e-3:
    FUNDAMENTAL_WEIGHTS = {k: v / _total for k, v in FUNDAMENTAL_WEIGHTS.items()}

FUNDAMENTAL_ALIAS_MAP = {
    "peRatio": "P/E Ratio",
    "pbRatio": "Price to Book (P/B)",
    "pegRatio": "PEG Ratio",
    "psRatio": "Price-to-Sales (P/S)",
    "peVsSector": "P/E vs Sector",
    "fcfYield": "FCF Yield (%)",
    "dividendYield": "Dividend Yield (%)",
    "dividendPayout": "Dividend Payout (%)",
    "marketCap": "Market Cap",
    "marketCapCagr": "Market Cap CAGR (%)",
    # Profitability / returns
    "roeHistory": "ROE History",
    "roe": "Return on Equity (ROE)",
    "roce": "Return on Capital Employed (ROCE)",
    "roic": "Return on Invested Capital (ROIC)",
    "netProfitMargin": "Net Profit Margin (%)",
    "operatingMargin": "Operating Margin (%)",
    "ebitdaMargin": "EBITDA Margin (%)",
    "fcfMargin": "FCF Margin (%)",
    # Growth
    "revenueGrowth5y": "Revenue Growth (5Y CAGR)",
    "profitGrowth3y": "Profit Growth (3Y CAGR)",
    "epsgrowth5y": "EPS Growth (5Y CAGR)",
    "epsGrowth3y": "EPS Growth (3Y CAGR)",
    "fcfGrowth3y": "FCF Growth (3Y CAGR)",
    "quarterlyGrowth": "Quarterly Growth (EPS/Rev)",
    # Health / liquidity
    "deRatio": "Debt to Equity",
    "interestCoverage": "Interest Coverage Ratio",
    "currentRatio": "Current Ratio",
    "ocfVsProfit": "Operating CF vs Net Profit",
    # Quality / efficiency
    "piotroskiF": "Piotroski F-Score",
    "assetTurnover": "Asset Turnover Ratio",
    "RDIntensity": "R&D Intensity (%)",
    "earningsStability": "Earnings Stability",
    # Ownership / market
    "promoterHolding": "Promoter Holding (%)",
    "promoterpledge": "Promoter Pledge (%)",
    "institutionalOwnership": "Institutional Ownership (%)",
    "shortInterest": "Short Interest",
    "analystRating": "Analyst Rating (Momentum)",
    "position52w": "52W Position (off-high %)",
    "beta": "Beta",
    "days_to_earnings": "Days to Next Earnings",
    "psRatio": "Price-to-Sales (P/S)",
    # reporting/meta
    "base_score": "Base Fundamental Score",
    "final_score": "Final Fundamental Score",
    "_meta": "Meta",
    "high52w": "52 week high",
    "low52w": "52 week low",
    "drawdown52wHigh":"Drawdown from 52W High",
    "priceVs52wHighPct":"Price vs 52w high percentage",
    "volatilityAdjustedRoe": "ROE/Volatility Ratio",
    "priceToIntrinsicValue": "Price vs Intrinsic Value",
    "fcfYieldVsVolatility": "FCF Yield vs Volatility",
    "earningsConsistencyIndex": "Earnings Consistency Index",
    "roeStability": "ROE Stability (StdDev)"

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
    "marketCap": ["marketCap", "Market Capitalization", "marketCap", "Market Cap"],
    "book_value": ["bookValue", "Book Value", "Book value per share"],
    "dividend": ["Dividends Paid", "dividendRate", "Cash Dividends Paid"],
    "fcfYield": ["Free Cash Flow Yield", "fcfYield"],
    "promoterHolding": [
        "heldPercentInsiders",
        "insiderPercent",
        "insidersPercent",
        "Insider Ownership",
    ],
    "institutionalOwnership": [
        "heldPercentInstitutions",
        "institutionPercent",
        "institutionsPercent",
        "Institutional Ownership",
    ],
    "dividendYield": ["dividendYield"],
    "analystRating": ["recommendations", "recommendationKey", "recommendationMean"],
    "quarterlyGrowth": ["earningsQuarterlyGrowth", "revenueQuarterlyGrowth"],
    "shortInterest": ["shortRatio", "sharesPercentSharesOut", "shortPercentOfFloat"],
    "trendStrength": [],
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

# ============================================================================
# PRODUCTION-READY HORIZON_PROFILE_MAP
# Uses Standardized Dynamic Keys + Fixes from Analysis
# ============================================================================

HORIZON_PROFILE_MAP = {
    "intraday": {
        "metrics": {
            "maFastSlope": 0.20,          # ✅ Boosted (was 0.15)
            "rsislope": 0.20,              # ✅ Boosted (was 0.15)
            "priceAction": 0.15,           # ✅ Kept
            "vwapBias": 0.15,
            "volSpikeRatio": 0.10,
            "volatilityQuality": 0.05,     # ✅ Reduced (let engine handle)
            "maTrendSignal": 0.05,
            "momentumStrength": 0.10,
        },
        "penalties": {
            # ✅ REMOVED: bbWidth (Let squeeze score high!)
            # ✅ REMOVED: niftyTrendScore (Stock-specific plays allowed)
            "gapPercent": {"operator": "<", "value": 0.1, "penalty": 0.1},  # ✅ Relaxed
            "maFastSlope": {"operator": "<", "value": -2, "penalty": 0.3},  # ✅ Relaxed (was 0)
            "atrPct": {"operator": "<", "value": 0.4, "penalty": 0.3},      # ✅ FIXED (was 0.75)
        },
        "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 3.5},  # Was: 6.5, 5.0, 3.5
    },

    "short_term": {
        "metrics": {
            "trendStrength": 0.15,         # ✅ Key metric
            "maTrendSignal": 0.10,
            "priceVsPrimaryTrendPct": 0.08,  # ✅ Reduced (pullbacks OK)
            "maFastSlope": 0.05,
            "supertrendsignal": 0.10,
            "momentumStrength": 0.12,      # ✅ Boosted (was 0.10)
            "rsislope": 0.08,              # ✅ Boosted (was 0.05)
            "macdCross": 0.05,
            "cmfSignal": 0.05,
            "obvDiv": 0.05,
            "volatilityQuality": 0.05,
            "rvol": 0.05,
            "quarterlyGrowth": 0.03,
            "analystRating": 0.02,
            "niftyTrendScore": 0.02,      # ✅ Reduced (was 0.05)
        },
        "penalties": {
            "days_to_earnings": {"operator": "<", "value": 3, "penalty": 0.5},  # ✅ FIXED (was 7 days, -1.0)
            # ✅ REMOVED: priceVsPrimaryTrendPct (Allow dip buys)
            "maFastSlope": {"operator": "<", "value": -5, "penalty": 0.3},    # ✅ Kept (severe downtrend)
            "rvol": {"operator": "<", "value": 0.5, "penalty": 0.2}            # ✅ Relaxed (was 0.8)
        },
        "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 3.8},  # Was: 6.5, 5.0, 4.0
    },

    "long_term": {
        # ✅ No changes needed - already balanced
        "metrics": {
            "maTrendSignal": 0.15,
            "maFastSlope": 0.10,
            "priceVsPrimaryTrendPct": 0.10,
            "roe": 0.10,
            "roce": 0.08,
            "roic": 0.08,
            "earningsStability": 0.08,
            "fcfYield": 0.08,
            "epsgrowth5y": 0.06,
            "piotroskiF": 0.05,
            "deRatio": 0.03,
            "promoterHolding": 0.05,
            "relStrengthNifty": 0.04,
            # 🟢 SYNC WITH STRATEGY ANALYZER: Valuation Check
            "pegRatio": 0.05 
        },
        "penalties": {
            "priceVsPrimaryTrendPct": {"operator": "<", "value": 0, "penalty": 0.5},
            "roe": {"operator": "<", "value": 10, "penalty": 0.3},
            "fcfYield": {"operator": "<", "value": 2, "penalty": 0.3},
            "promoterpledge": {"operator": ">", "value": 15.0, "penalty": 0.2},
            # 🟢 FRAUD CHECK: High Profit but No Cash Flow
            "ocfVsProfit": {"operator": "<", "value": 0.6, "penalty": 0.5} 
        },
        "thresholds": {"buy": 7.5, "hold": 6.0, "sell": 4.0},
    },

    "multibagger": {
        # ✅ No changes needed
        "metrics": {
            "maTrendSignal": 0.10,
            "maFastSlope": 0.10,
            "priceVsPrimaryTrendPct": 0.05,
            "epsgrowth5y": 0.10,
            "revenueGrowth5y": 0.10,
            # 🟢 SYNC WITH CANSLIM STRATEGY: Recent Growth
            "quarterlyGrowth": 0.05, 
            "marketCapCagr": 0.08,
            "roic": 0.10,
            "roe": 0.08,
            "pegRatio": 0.08,
            "RDIntensity": 0.05,
            "promoterHolding": 0.05,
            "institutionalOwnership": 0.03,
            "ocfVsProfit": 0.06,
            "relStrengthNifty": 0.05,
        },
        "penalties": {
            "maFastSlope": {"operator": "<", "value": 0, "penalty": 0.5},
            "pegRatio": {"operator": ">", "value": 3.0, "penalty": 0.3},
            "marketCap": {"operator": ">", "value": 1e12, "penalty": 0.5},
            "deRatio": {"operator": ">", "value": 1.0, "penalty": 0.2},
            "roe": {"operator": "<", "value": 12, "penalty": 0.2},
            "institutionalOwnership": {"operator": ">", "value": 85, "penalty": 0.3},
            "promoterpledge": {"operator": ">", "value": 10.0, "penalty": 0.4}
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
        "period": "3y",    # CHANGED from '3mo' to 2y. Gives ~250 candles. Enough for 200 DMA.
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
    "piotroskiF": {"weight": 1.0, "direction": "normal"},
    "ocfVsProfit": {"weight": 1.0, "direction": "normal"},
    "interestCoverage": {"weight": 1.0, "direction": "normal"},
    "earningsStability": {"weight": 1.0, "direction": "normal"},
    "netProfitMargin": {"weight": 1.0, "direction": "normal"},
    "deRatio": {"weight": 1.0, "direction": "normal"},
    "promoterpledge": {"weight": 1.0, "direction": "normal"},
    "roeStability": {"weight": 0.10, "direction": "normal"},
    "volatilityQuality": {"weight": 0.10, "direction": "normal"},
}

GROWTH_WEIGHTS = {
    "epsGrowth3y": {"weight": 1.0, "direction": "normal"},
    "revenueGrowth5y": {"weight": 1.0, "direction": "normal"},
    "quarterlyGrowth": {"weight": 1.0, "direction": "normal"},
    "fcfGrowth3y": {"weight": 1.0, "direction": "normal"},
    "marketCapCagr": {"weight": 1.0, "direction": "normal"},
}

VALUE_WEIGHTS = {
    "peRatio": {"weight": 1.0, "direction": "normal"},
    "pbRatio": {"weight": 1.0, "direction": "normal"},
    "pegRatio": {"weight": 1.0, "direction": "normal"},
    "peVsSector": {"weight": 1.0, "direction": "normal"},
    "fcfYield": {"weight": 1.0, "direction": "normal"},
    "dividendYield": {"weight": 1.0, "direction": "normal"},
}

MOMENTUM_WEIGHTS = {  # 🆕 CORE COMPOSITES (PRIORITY)
    "momentumStrength": {
        "weight": 0.30,
        "direction": "normal",
    },  # RSI, MACD, Stoch Bundle
    "trendStrength": {
        "weight": 0.40,
        "direction": "normal",
    },  # ADX, EMA Slope, ST Bundle
    "volatilityQuality": {
        "weight": 0.10,
        "direction": "normal",
    },  # New Volatility Setup Score
    # CONTEXTUAL/HYBRID MOMENTUM (MUST KEEP)
    "vwapBias": {"weight": 0.05, "direction": "normal"},
    "priceAction": {"weight": 0.05, "direction": "normal"},
    "niftyTrendScore": {"weight": 0.05, "direction": "normal"},  # Macro Context
    "position52w": {"weight": 0.05, "direction": "normal"},  # Hybrid/Sentiment Context
    # ⚠️ CONTEXTUAL/VOLUME (MUST KEEP)
    "rvol": {
        "weight": 0.00,
        "direction": "normal",
    },  # Weight mass moved to composite, keep key for context
    "obvDiv": {
        "weight": 0.00,
        "direction": "normal",
    },  # Weight mass moved to composite, keep key for context
    "psarTrend": {"weight": 0.00, "direction": "normal"},
    "ttmSqueeze": {"weight": 0.00, "direction": "normal"},
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

# In constants.py - ADD THIS:
VOL_BANDS_HORIZON_MULTIPLIERS = {
    "intraday": 1.0,      # Use base thresholds (4% = risky)
    "short_term": 1.0,    # Daily data, use base
    "long_term": 2.5,     # Weekly: 4% × 2.5 = 10% ceiling
    "multibagger": 4.0    # Monthly: 4% × 4 = 16% ceiling ✅
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

# ============================================================================
# VOLATILITY QUALITY MINIMUM THRESHOLDS (UNCHANGED - These are already correct)
# ============================================================================
VOL_QUAL_MINS = {
    "intraday": {
        "MOMENTUM_BREAKOUT": 2.5,
        "VOLATILITY_SQUEEZE": 4.0,
        "TREND_PULLBACK": 3.0,
        "default": 2.5
    },
    "short_term": {
        "MOMENTUM_BREAKOUT": 3.0,
        "VOLATILITY_SQUEEZE": 5.0,
        "TREND_PULLBACK": 3.5,
        "default": 3.0
    },
    "long_term": {
        "MOMENTUM_BREAKOUT": 4.0,
        "VOLATILITY_SQUEEZE": 6.0,
        "TREND_PULLBACK": 4.5,
        "default": 4.0
    },
    "multibagger": {
        "MOMENTUM_BREAKOUT": 5.0,
        "VOLATILITY_SQUEEZE": 7.0,
        "TREND_PULLBACK": 5.0,
        "default": 4.5
    },
    "default": 3.0
}

# ============================================================================
# VOLATILITY REGIME BANDS (✅ CORRECTLY SCALED FOR TIMEFRAMES)
# ============================================================================
VOL_BANDS = {
    "intraday": {
        "min": 1.5,      # Dead money below 1.5% daily range
        "ideal": 3.0,    # Sweet spot for scalping
        "max": 6.0       # Panic selling/buying above 6%
    },
    "short_term": {      # DAILY CANDLES (Baseline)
        "min": 1.0,      # Need some swing
        "ideal": 2.5,    # Healthy volatility
        "max": 5.0       # Too choppy above 5%
    },
    "long_term": {       # ✅ WEEKLY CANDLES (Scale ~2.2x from daily)
        "min": 2.0,      # <2% weekly = dead stock (0.4% daily equivalent)
        "ideal": 5.5,    # Healthy weekly trend (1.1% daily equivalent)
        "max": 12.0      # >12% weekly = unstable (2.4% daily equivalent)
    },
    "multibagger": {     # ✅ MONTHLY CANDLES (Scale ~4.6x from daily)
        "min": 3.0,      # <3% monthly = bond-like (0.65% daily equivalent)
        "ideal": 8.0,    # Steady compounder (1.75% daily equivalent)
        "max": 25.0      # >25% monthly = extreme risk (5.5% daily equivalent)
    }
}

# ============================================================================
# VOLATILITY QUALITY SCORING (✅ SCALED FOR TIMEFRAMES)
# ============================================================================
# Used in compute_volatility_quality() to score ATR%
VOL_SCORING_THRESHOLDS = {
    "intraday": {
        "excellent": 4.0,   # ATR% <= 4% = excellent for scalping
        "good": 6.0,        # ATR% <= 6% = acceptable
        "fair": 8.0,        # ATR% <= 8% = risky but tradeable
        "poor": 10.0        # ATR% > 10% = panic/news event
    },
    "short_term": {         # DAILY CANDLES (Baseline)
        "excellent": 2.5,   # ATR% <= 2.5% = clean swing
        "good": 4.0,        # ATR% <= 4% = normal swing volatility
        "fair": 6.0,        # ATR% <= 6% = choppy
        "poor": 8.0         # ATR% > 8% = avoid
    },
    "long_term": {          # ✅ WEEKLY CANDLES (Scale ~2.2x)
        "excellent": 5.5,   # ATR% <= 5.5% weekly (2.5% daily equiv)
        "good": 9.0,        # ATR% <= 9% weekly (4% daily equiv)
        "fair": 13.0,       # ATR% <= 13% weekly (6% daily equiv) ← HAL is here
        "poor": 18.0        # ATR% > 18% weekly = breakdown
    },
    "multibagger": {        # ✅ MONTHLY CANDLES (Scale ~4.6x)
        "excellent": 11.5,  # ATR% <= 11.5% monthly (2.5% daily equiv)
        "good": 18.0,       # ATR% <= 18% monthly (4% daily equiv) ← HAL (13.76%) is here
        "fair": 27.0,       # ATR% <= 27% monthly (6% daily equiv)
        "poor": 36.0        # ATR% > 36% monthly = extreme
    }
}

# ============================================================================
# SCALING REFERENCE TABLE (For Documentation)
# ============================================================================
# Timeframe   | Days | √Days | Daily 2.5% → Scaled | Daily 5% → Scaled
# ------------|------|-------|---------------------|-------------------
# Intraday    | 1    | 1.0x  | 2.5%                | 5%
# Short-Term  | 1    | 1.0x  | 2.5%                | 5%
# Long-Term   | 5    | 2.24x | 5.6%                | 11.2%
# Multibagger | 21   | 4.58x | 11.5%               | 22.9%


# ============================================================================
# TREND STRENGTH THRESHOLDS (UNCHANGED)
# ============================================================================
TREND_THRESH = {
    "weak_floor": 20.0,
    "moderate_floor": 25.0,
    "strong_floor": 40.0,
    "diSpread_strong": 20.0,
}

RVOL_SURGE_THRESHOLD, RVOL_DROUGHT_THRESHOLD, VOLUME_CLIMAX_SPIKE = 3.0, 0.7, 2.0
ATR_SL_MAX_PERCENT, ATR_SL_MIN_PERCENT = 0.03, 0.01
STRATEGY_TIME_MULTIPLIERS = {'momentum': 0.7, 'day_trading': 0.5, 'swing': 1.0,
                             'trend_following': 1.2, 'position_trading': 1.5, 'value': 1.5,
                             'income': 2.0, 'unknown': 1.0}
TREND_WEIGHTS = {'primary': 0.50, 'secondary': 0.30, 'acceleration': 0.20}
RR_REGIME_ADJUSTMENTS = {'strong_trend': {'t1_mult': 2.0, 't2_mult': 4.0},
                         'normal_trend': {'t1_mult': 1.5, 't2_mult': 3.0},
                         'weak_trend': {'t1_mult': 1.2, 't2_mult': 2.5}}
HORIZON_T2_CAPS = {
    "intraday": 0.04,     # Max 4% expansion
    "short_term": 0.10,   # Max 10% expansion
    "long_term": 0.20,    # Max 20% expansion
    "multibagger": 1.00   # Uncapped (100%)
}

SIGNAL_ENGINE = {
    # ------------------------------------------------------------------
    # Risk / Reward rules (UPDATED)
    # ------------------------------------------------------------------
    "RR_RULES": {
        "min_rr_t1": 1.5,
        "min_rr_by_horizon": {  # ✅ ADD THIS
            "intraday": 1.1,
            "short_term": 1.3,
            "long_term": 1.5,
            "multibagger": 1.5
        },
        "default_multipliers": {  # ✅ ADD THIS
            "t1_mult": 1.5,
            "t2_mult": 3.0
        }
    },

    # ------------------------------------------------------------------
    # Volume Analysis (NEW)
    # ------------------------------------------------------------------\
    "VOLUME_ANALYSIS": {
        "intraday": {
            "rvol_surge_threshold": 3.0,
            "rvol_drought_threshold": 0.7
        },
        "short_term": {
            "rvol_surge_threshold": 2.5,  # Lower bar for daily
            "rvol_drought_threshold": 0.7
        },
        "long_term": {
            "rvol_surge_threshold": 2.0,  # Weekly volume is less volatile
            "rvol_drought_threshold": 0.8
        },
        "multibagger": {
            "rvol_surge_threshold": 1.8,  # Monthly volume even more stable
            "rvol_drought_threshold": 0.8
        }
    },

    # ------------------------------------------------------------------
    # Stop Loss Configuration (NEW)
    # ------------------------------------------------------------------
    "STOP_LOSS_MULTIPLIERS": {  # ✅ ADD THIS ENTIRE SECTION
        "volatility_based": {
            "high_quality": {"threshold": 8.0, "mult": 1.5},
            "low_quality": {"threshold": 4.0, "mult": 3.0},
            "default": 2.0
        }
    },
    
    "STOP_LOSS_VALIDATION": {  # ✅ ADD THIS
        "max_atr_multiplier": 5.0
    },

    # ------------------------------------------------------------------
    # Pattern Priority (NEW)
    # ------------------------------------------------------------------
    "PATTERN_PRIORITY": [  # ✅ ADD THIS ENTIRE SECTION
        {"pattern": "darvasBox", "setup_name": "PATTERN_DARVAS_BREAKOUT", "min_score": 85},
        {"pattern": "minerviniStage2", "setup_name": "PATTERN_VCP_BREAKOUT", "min_score": 85},
        {"pattern": "cupHandle", "setup_name": "PATTERN_CUP_BREAKOUT", "min_score": 80},
        {"pattern": "threeLineStrike", "setup_name": "PATTERN_STRIKE_REVERSAL", "min_score": 80},
        {"pattern": "goldenCross", "setup_name": "PATTERN_GOLDEN_CROSS", "min_score": 75},
        {"pattern": "flagPennant", "setup_name": "PATTERN_FLAG_BREAKOUT", "min_score": 80}
    ],

    # ------------------------------------------------------------------
    # Divergence Adjustments (NEW)
    # ------------------------------------------------------------------
    "DIVERGENCE_ADJUSTMENTS": {  # ✅ ADD THIS
        "opposing_divergence_sl_mult": 0.8
    },

    # ------------------------------------------------------------------
    # Consolidation Detection (NEW)
    # ------------------------------------------------------------------
    "CONSOLIDATION_DETECTION": {  # ✅ ADD THIS
        "bb_atr_ratio_threshold": 0.5
    },

    # ------------------------------------------------------------------
    # Proximity Rejection (NEW)
    # ------------------------------------------------------------------
    "PROXIMITY_REJECTION": {
        "intraday": {"resistance_mult": 1.003, "support_mult": 0.997},
        "short_term": {"resistance_mult": 1.005, "support_mult": 0.995},
        "long_term": {"resistance_mult": 1.010, "support_mult": 0.990},
        "multibagger": {"resistance_mult": 1.020, "support_mult": 0.980}
    },

    # ------------------------------------------------------------------
    # Dynamic confidence floors
    # ------------------------------------------------------------------
    "BASE_FLOORS": {
        "MOMENTUM_BREAKOUT": {
            "intraday": 50,
            "short_term": 55,
            "long_term": 60,
            "multibagger": 65
        },
        "TREND_PULLBACK": {
            "intraday": 48,
            "short_term": 53,
            "long_term": 58,
            "multibagger": 60
        }
    },

    "CONFIDENCE_HORIZON_DISCOUNT": {
        "intraday": 10
    },

    "CONFIDENCE_ADX_NORMALIZATION": {
        "adx_min": 10,
        "adx_range": 30,
        "adx_scale": 12,
        "min_floor": 35,
        "max_floor": 75
    },

    # ------------------------------------------------------------------
    # Horizon-specific target expansion caps
    # ------------------------------------------------------------------
    "HORIZON_T2_CAPS": {
        "intraday": 0.04,
        "short_term": 0.10,
        "long_term": 0.20,
        "multibagger": 1.00
    },

    # ------------------------------------------------------------------
    # Target / resistance calculation constants
    # ------------------------------------------------------------------
    "TARGET_BUFFERS": {
        "min_clearance_mult": 1.002,      # entry * 1.002
        "min_profit_pct": {
            "intraday": 0.3,
            "short_term": 0.5,
            "long_term": 1.0,
            "multibagger": 2.0
        },
        "t1_resistance_mult": 0.96,
        "t2_resistance_mult": 0.98,
        "next_resistance_mult": 1.03,
        "future_resistance_mult": 1.05,
        "t2_fallback_1": 1.15,
        "t2_fallback_2": 1.12,
        "t2_fallback_3": 1.20
    },

    "MAX_PRICE_MOVE": {
        "short_term": 1.10
    },

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    "POSITION_SIZING": {
        "base_risk": 0.01,
        "max_pct": {
            "intraday": 0.01,
            "default": 0.02
        },
        "setup_multipliers": {
            "DEEP_PULLBACK": 1.5,
            "VOLATILITY_SQUEEZE": 1.3,
            "MOMENTUM_BREAKOUT": 0.8
        },
        "volatility_multipliers": {
            "high_quality": 1.2,   # vol_qual > 7
            "low_quality": 0.9,    # vol_qual < 5
            "neutral": 1.0
        }
    },

    # ------------------------------------------------------------------
    # Volatility trade permission guards
    # ------------------------------------------------------------------
    "VOLATILITY_GUARDS": {
        "intraday": {
            "extreme_vol_buffer": 2.0,
            "min_quality_breakout": 2.5,
            "min_quality_normal": 4.0
        },
        "short_term": {
            "extreme_vol_buffer": 2.0,
            "min_quality_breakout": 3.0,
            "min_quality_normal": 4.0
        },
        "long_term": {
            "extreme_vol_buffer": 3.0,
            "min_quality_breakout": 4.0,
            "min_quality_normal": 5.0
        },
        "multibagger": {
            "extreme_vol_buffer": 4.0,
            "min_quality_breakout": 5.0,
            "min_quality_normal": 6.0
        }
    },

    # ------------------------------------------------------------------
    # Entry permission logic
    # ------------------------------------------------------------------
    "ENTRY_PERMISSION": {
        "confidence_discounts": {
            "trend": 15,
            "value_reversal": 25
        },
        "required_trend_strength": {
            "intraday": 2.0,
            "short_term": 3.5,
            "long_term": 5.0,
            "multibagger": 6.0
        }
    },

    # ------------------------------------------------------------------
    # Trend strength composite thresholds
    # ------------------------------------------------------------------
    "TREND_STRENGTH_THRESHOLDS": {
        "adx": {
            "strong": 25,
            "medium": 20,
            "weak": 15
        },
        "slope": {
            "intraday": {"strong": 15.0, "moderate": 5.0},
            "short_term": {"strong": 10.0, "moderate": 3.0},
            "long_term": {"strong": 5.0, "moderate": 2.0},
            "multibagger": {"strong": 30.0, "moderate": 10.0}  # Monthly slopes are huge!
        },
        "di_diff": {
            "strong": 15,
            "moderate": 10
        }
    },

    # ------------------------------------------------------------------
    # Momentum composite thresholds
    # ------------------------------------------------------------------
    "MOMENTUM_THRESHOLDS": {
        "rsi": {
            "strong": 70,
            "medium": 60,
            "neutral": 50,
            "weak": 40
        },
        "rsislope": {
            "intraday": {"positive": 0.10, "neutral": 0.0},
            "short_term": {"positive": 0.05, "neutral": 0.0},
            "long_term": {"positive": 0.03, "neutral": 0.0},
            "multibagger": {"positive": 0.02, "neutral": 0.0}
        },
        "macd_hist": {
            "positive": 0.5,
            "neutral": 0.0
        }
    },

    # ------------------------------------------------------------------
    # Volatility quality scoring thresholds
    # ------------------------------------------------------------------
    "VOLATILITY_QUALITY_THRESHOLDS": {
        "intraday": {
            "atrPct": {"very_low": 1.5, "low": 3.0, "high": 6.0},
            "bbWidth": {"very_tight": 0.01, "tight": 0.02, "wide": 0.04}
        },
        "short_term": {
            "atrPct": {"very_low": 1.0, "low": 2.5, "high": 5.0},
            "bbWidth": {"very_tight": 0.01, "tight": 0.02, "wide": 0.04}
        },
        "long_term": {
            "atrPct": {"very_low": 2.0, "low": 5.5, "high": 12.0},
            "bbWidth": {"very_tight": 0.02, "tight": 0.04, "wide": 0.08}
        },
        "multibagger": {
            "atrPct": {"very_low": 3.0, "low": 8.0, "high": 25.0},
            "bbWidth": {"very_tight": 0.03, "tight": 0.06, "wide": 0.12}
        }
    },

    # ------------------------------------------------------------------
    # Reversal / divergence
    # ------------------------------------------------------------------
    "REVERSAL_THRESHOLDS": {
        "rsi_oversold": 30
    },

    # ------------------------------------------------------------------
    # Spread adjustment
    # ------------------------------------------------------------------
    "SPREAD_ADJUSTMENT": {
        "market_cap_thresholds": {
            "large": 100000,
            "mid": 10000
        },
        "spread_values": {
            "large": 0.001,
            "mid": 0.002,
            "small": 0.005
        }
    },

    # ------------------------------------------------------------------
    # Pattern confluence bonuses
    # ------------------------------------------------------------------
    "PATTERN_BONUSES": {
        "additional_pattern": 5,
        "minervini_flag": 10,
        "squeeze_combo": 8,
        "golden_cross_cup": 7
    },

    # ------------------------------------------------------------------
    # Misc execution constants
    # ------------------------------------------------------------------
    "MISC": {
        "bear_market_dampener": 0.85,
        "min_trend_score": 0.35,
        "pullback_ma_pct": 0.05
    }
}

STRATEGY_ANALYZER = {

    "DEFAULT_INDICATOR_VALUES": {
        "rsi": 50,
        "rvol": 1.0,
        "atrPct": 1.0
    },
    # ------------------------------------------------------------------
    # Swing Trading
    # ------------------------------------------------------------------
    "swing": {
        "fit_thresh": 50,
        "bb_proximity_mult": 1.02,          # Price <= BB Low * 1.02
        "bb_proximity_score": 35,
        "rsi_dip_threshold": 45,
        "rsi_dip_score": 25,
        "pattern_score": 40,                # Double Bottom bonus
        "squeeze_bonus": 10
    },

    # ------------------------------------------------------------------
    # Day Trading
    # ------------------------------------------------------------------
    "day_trading": {
        "fit_thresh": 50,
        "rvol_threshold": 1.5,
        "rvol_score": 25,
        "atr_pct_threshold": 1.5,
        "atr_pct_score": 15,
        "pattern_score": 40                 # 3-Line Strike bonus
    },

    # ------------------------------------------------------------------
    # Trend Following
    # ------------------------------------------------------------------
    "trend_following": {
        "fit_thresh": 50,
        "ma_alignment_score": 30,
        "adx_threshold": 25,
        "adx_score": 20,
        "ichimoku_score": 25,
        "golden_cross_score": 25
    },

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    "momentum": {
        "fit_thresh": 50,
        "rsi_threshold": 60,
        "rsi_score": 20,
        "darvas_score": 30,
        "flag_score": 30,
        "squeeze_score": 20
    },

    # ------------------------------------------------------------------
    # Minervini VCP
    # ------------------------------------------------------------------
    "minervini": {
        "fit_thresh": 50,
        "vcp_pattern_score": 50,
        "stage2_fallback_score": 20,
        "stage2_fail_penalty": -50,
        "relative_strength_score": 20,
        "pos_52w_high_threshold": 85,
        "pos_52w_high_score": 20,
        "pos_52w_low_threshold": 50,
        "pos_52w_low_penalty": -20
    },

    # ------------------------------------------------------------------
    # CANSLIM
    # ------------------------------------------------------------------
    "canslim": {
        "fit_thresh": 50,
        "quarterly_growth_threshold": 20,    # C: Current Earnings
        "quarterly_growth_score": 20,
        "annual_growth_threshold": 15,       # A: Annual Earnings
        "annual_growth_score": 15,
        "cup_pattern_score": 30,             # N: New Pattern
        "new_high_threshold": 90,            # N: 52W Position
        "new_high_score": 15,
        "volume_threshold": 1.2,             # S: Supply/Demand
        "volume_score": 10,
        "relative_strength_threshold": 5,    # L: Leader
        "relative_strength_score": 15
    },

    # ------------------------------------------------------------------
    # Value Investing
    # ------------------------------------------------------------------
    "value": {
        "fit_thresh": 50,
        "pe_threshold": 15,
        "pe_score": 35,
        "pb_threshold": 1.5,
        "pb_score": 25
    },

    # ------------------------------------------------------------------
    # Income / Dividend
    # ------------------------------------------------------------------
    "income": {
        "fit_thresh": 50,
        "dividend_yield_threshold": 3.0,
        "dividend_yield_score": 40
    },

    # ------------------------------------------------------------------
    # Position Trading
    # ------------------------------------------------------------------
    "position_trading": {
        "fit_thresh": 50,
        "eps_growth_threshold": 10,
        "eps_growth_score": 30,
        "uptrend_score": 40,
        "golden_cross_bonus": 20
    },

    # ------------------------------------------------------------------
    # Fallback Configs (Legacy)
    # ------------------------------------------------------------------
    "breakout": {
        "high_vol_threshold": 1.5,
        "low_vol_threshold": 0.8,
        "min_confidence": 60
    },
    
    "accumulation": {
        "consolidation_days": 5,
        "volume_ratio": 0.8,
        "min_confidence": 50
    },
    
    "pullback": {
        "max_retracement": 0.5,
        "trend_confirmation": True,
        "min_confidence": 55
    },

    # 1️⃣ DEFAULT FIT THRESHOLD ✅
    "DEFAULT_FIT_THRESH": 50,

    # 2️⃣ STRATEGY EXECUTION ORDER ✅
    "STRATEGY_EXECUTION_ORDER": [
        "swing",
        "day_trading",
        "trend_following",
        "position_trading",
        "momentum",
        "value",
        "income",
        "minervini",
        "canslim"
    ],

    # 3️⃣ PATTERN DETECTION ✅ (CORRECTED)
    "PATTERN_MIN_SCORE": 0,  # Pattern score must be > 0 (exclusive)

    # 4️⃣ SQUEEZE DETECTION ✅ (ENHANCED)
    "SQUEEZE_DETECTION": {
        "indicator_key": "ttmSqueeze",
        "active_token": "on",
        "check_method": "contains"
    },

    # 5️⃣ RELATIVE STRENGTH THRESHOLDS ✅ (CORRECTED)
    "RELATIVE_STRENGTH_THRESHOLDS": {
        "minervini": 0,      # Must be > 0
        "canslim": 5,        # Must be > 5
        "operator": ">"
    },

    # 6️⃣ STAGE 2 TEMPLATE ✅
    "STAGE2_TEMPLATE": {
        "enabled": True,
        "ma_keys": ["maMid", "maSlow"],  # Price > MA50 > MA200
        "price_position": "above_all"
    },

    # 7️⃣ TREND STRUCTURE (OPTIONAL) 🟡
    "TREND_STRUCTURE": {
        "bullish_alignment": {
            "enabled": True,
            "order": ["price", "maFast", "maMid", "maSlow"],
            "operator": ">"
        }
    }
}

TRADE_ENHANCER = {

    # ---------------------------------------------------------
    # RR regime adjustments (fallbacks if config missing)
    # ---------------------------------------------------------
    "RR_REGIME_DEFAULTS": {
        "strong_trend": {
            "adx_min": 40,
            "t1_mult": 2.0,
            "t2_mult": 4.0
        },
        "normal_trend": {
            "adx_min": 20,
            "adx_max": 40,
            "t1_mult": 1.5,
            "t2_mult": 3.0
        },
        "weak_trend": {
            "adx_max": 20,
            "t1_mult": 1.2,
            "t2_mult": 2.5
        }
    },

    # ---------------------------------------------------------
    # Pattern expiration – applicable patterns (hardcoded list)
    # ---------------------------------------------------------
    "EXPIRING_PATTERNS": [
        "flagPennant",
        "threeLineStrike"
    ],

    # ---------------------------------------------------------
    # Pattern entry validation defaults
    # ---------------------------------------------------------
    "PATTERN_ENTRY_DEFAULTS": {

        "cupHandle": {
            "rim_clearance": 0.99,
            "rvol_min": 1.2,
            "rvol_bonus_threshold": 2.0,
            "volume_surge_bonus": 10
        },

        "darvasBox": {
            "box_clearance": 1.005
        },

        "minerviniStage2": {
            "contraction_max": 1.5,
            "contraction_warning_penalty": -5
        },

        "bollingerSqueeze": {
            "rsi_min": 50
        },

        "flagPennant": {
            "pole_length_min": 5,
            "pole_short_penalty": -5
        },

        "threeLineStrike": {
            "strike_candle_body_min": 0.6
        }
    },

    # ---------------------------------------------------------
    # Divergence detection (local severity logic) DUPLICATE of constants.py's RSI_SLOPE_THRESH
    # ---------------------------------------------------------
    "DIVERGENCE_THRESHOLDS": {
        "rsislope": {
            "severe": -0.08,
            "moderate": -0.03,
            "minor": 0.0
        }
    },

    # ---------------------------------------------------------
    # Pattern selection thresholds
    # ---------------------------------------------------------
    "PATTERN_SELECTION": {
        "min_score": 60
    },

    # ---------------------------------------------------------
    # Supported pattern scan order
    # ---------------------------------------------------------
    "PATTERN_KEYS": [
        "darvasBox",
        "cupHandle",
        "bollingerSqueeze",
        "flagPennant",
        "minerviniStage2",
        "threeLineStrike",
        "ichimokuSignals",
        "goldenCross",
        "bullishNecklinePattern",
        "bearishNecklinePattern"
    ],

    # ---------------------------------------------------------
    # Pattern priority (implicit via score sort, but thresholded)
    # ---------------------------------------------------------
    "PATTERN_SCORE_THRESHOLDS": {
        "high_quality": 80
    },

    # ---------------------------------------------------------
    # Pattern physics fallbacks
    # ---------------------------------------------------------
    "PATTERN_PHYSICS_DEFAULTS": {
        "target_ratio": 1.0,
        "t2_multiplier": 2
    },

    # ---------------------------------------------------------
    # Stop-loss defaults (when pattern SL missing)
    # ---------------------------------------------------------
    "STOP_LOSS_DEFAULTS": {
        "atr_fallback_mult": 2.0
    },

    # ---------------------------------------------------------
    # Directional sanity rules
    # ---------------------------------------------------------
    "STOP_LOSS_SANITY": {
        "long": {
            "must_be_below_entry": True
        },
        "short": {
            "must_be_above_entry": True
        }
    },

    # ---------------------------------------------------------
    # Execution quality scoring
    # ---------------------------------------------------------
    "EXECUTION_QUALITY_SCORES": {
        "has_geometry": 30,
        "has_stop_loss": 20,
        "has_target": 25,
        "high_pattern_score": 25
    },

    # ---------------------------------------------------------
    # Pattern role classification
    # ---------------------------------------------------------
    "PATTERN_ROLES": {
        "momentum_confirmation": [
            "bollingerSqueeze",
            "threeLineStrike"
        ],
        "trend_continuation": [
            "minerviniStage2"
        ],
        "regime_confirmation": [
            "ichimokuSignals",
            "goldenCross"
        ]
    },

    # ---------------------------------------------------------
    # Confidence adjustments
    # ---------------------------------------------------------
    "CONFIDENCE_ADJUSTMENTS": {
        "pattern_expired_penalty": -20
    },

    # ---------------------------------------------------------
    # ATR extraction fallback order (implicit priority)
    # ---------------------------------------------------------
    "ATR_FALLBACK_KEYS": [
        "atrDynamic",
        "atr14",
        "atr",
        "atr14"
    ],
    
    # ---------------------------------------------------------
    # Pattern reference levels for invalidation checks (NEW)
    # ---------------------------------------------------------
    "PATTERN_REFERENCE_LEVELS": {  # ✅ ADD THIS
        "darvasBox": "boxLow",
        "cupHandle": "handleLow",
        "flagPennant": "flagLow",
        "minerviniStage2": "pivotPoint",
        "bollingerSqueeze": "bbLow",
        "threeLineStrike": "entry",
        "ichimokuSignals": "cloud_bottom",
        "bullishNecklinePattern": "neckline",
        "bearishNecklinePattern": "neckline"
    },

    # ---------------------------------------------------------
    # MA key fallback paths (for legacy support) (NEW)
    # ---------------------------------------------------------
    "MA_KEY_FALLBACKS": {
        "fast": ["maFast", "ema20", "mafast", "ema20"],
        "mid": ["maMid", "ema50", "mamid", "ema50"],
        "slow": ["maSlow", "ema200", "maslow", "ema200"]
    },

    # ---------------------------------------------------------
    # Confidence score bounds (NEW)
    # ---------------------------------------------------------
    "CONFIDENCE_BOUNDS": {  # ✅ ADD THIS
        "min": 0,
        "max": 100,
        "default": 50
    },

    "DIVERGENCE_CONFIDENCE_PENALTIES": {
        "severe": 1.0,      # From MASTER_CONFIG (no entry allowed)
        "moderate": 0.85,   # 15% penalty
        "minor": 0.95       # 5% penalty
    },

    # ---------------------------------------------------------
    # Pattern-specific stop loss adjustments (NEW)
    # ---------------------------------------------------------
    "PATTERN_STOP_LOSS_MULTIPLIERS": {  # ✅ ADD THIS
        "darvasBox": 0.995  # 0.5% below boxLow
    },
}

PATTERNS = {
    
    # ==================================================================
    # BOLLINGER SQUEEZE
    # ==================================================================
    "bollingerSqueeze": {
        "squeeze_threshold": 0.10,              # BB Width < 10% = squeeze
        "breakout_confirmation": 0.02,          # 2% above band
        "squeeze_score": 75,
        "breakout_score": 95,
        "estimated_age_candles": 7,
        "squeeze_quality": 8.0,
        "breakout_quality": 10.0
    },

    # ==================================================================
    # CUP & HANDLE
    # ==================================================================
    "cupHandle": {
        "min_cup_len": 20,
        "max_cup_depth": 0.50,
        "min_cup_depth": 0.10,
        "require_volume": False,
        "handle_len": 5,
        "window_size": 60,
        "search_split_ratio": 0.5,
        "rim_alignment_tolerance": 0.15,
        "handle_upper_half_only": True,
        "forming_threshold": 0.90,
        "volume_dry_threshold": 0.9,
        "volume_bonus_quality": 2.0,
        "base_quality": 6.0,
        "breakout_bonus": 2.0,
        "min_history_buffer": 5
    },

    # ==================================================================
    # DOUBLE TOP / BOTTOM
    # ==================================================================
    "bullishNecklinePattern": {
        "peak_window": 5,
        "min_history": 60,
        "window_size": 60,
        "price_level_tolerance": 0.03,
        "pattern_score": 80,
        "pattern_quality": 8.5
    },
    "bearishNecklinePattern": {
        "peak_window": 5,
        "min_history": 60,
        "window_size": 60,
        "price_level_tolerance": 0.03,
        "pattern_score": 80,
        "pattern_quality": 8.5
    },

    # ==================================================================
    # GOLDEN / DEATH CROSS
    # ==================================================================
    "goldenCross": {
        "min_history": 200,
        "golden_cross_score": 90,
        "death_cross_score": 90,
        "pattern_quality": 9.0
    },

    # ==================================================================
    # MINERVINI VCP / STAGE 2
    # ==================================================================
    "minervini_vcp": {
        "min_history": 50,
        "max_atr_pct": 3.5,
        "recent_window": 5,
        "prev_window": 10,
        "contraction_threshold": 0.7,
        "tightness_threshold": 0.05,
        "vol_recent_window": 5,
        "vol_avg_window": 50,
        "formation_estimate_offset": 15,
        "base_quality": 7.0,
        "vol_dry_bonus": 2.0
    },

    # ==================================================================
    # DARVAS BOX
    # ==================================================================
    "darvasBox": {
        "lookback": 50,
        "box_length": 5,
        "box_lookback_multiplier": 2,
        "consolidation_tolerance_high": 1.01,
        "consolidation_tolerance_low": 0.99,
        "volume_threshold": 1.5,
        "base_quality": 5.0,
        "volume_bonus": 3.0,
        "trend_bonus": 2.0
    },

    # ==================================================================
    # THREE LINE STRIKE
    # ==================================================================
    "threeLineStrike": {
        "min_history": 5,
        "pattern_score": 90,
        "pattern_quality": 9.0
    },

    # ==================================================================
    # FLAG / PENNANT
    # ==================================================================
    "flagPennant": {
        "pole_days": 15,
        "flag_days": 5,
        "strong_pole_threshold": 0.05,
        "flag_drift_min": -0.03,
        "flag_drift_max": 0.01,
        "base_quality": 6.0,
        "volume_dry_bonus": 2.0,
        "breakout_threshold": 1.01,
        "breakout_bonus": 2.0
    },

    # ==================================================================
    # ICHIMOKU SIGNALS
    # ==================================================================
    "ichimokuSignals": {
        "tenkan_window": 9,
        "kijun_window": 26,
        "min_history_buffer": 2,
        
        "quality_scores": {
            "strong_bull_cross": 9.0,
            "weak_bull_cross": 5.0,
            "neutral_bull_cross": 7.0,
            "strong_bear_cross": 9.0,
            "weak_bear_cross": 5.0,
            "neutral_bear_cross": 7.0,
            "price_above_cloud": 5.0
        },
        
        "cross_bonus": 10,
        "fresh_cross_age": 1,
        "established_signal_age": 5
    },

    # ==================================================================
    # BASE PATTERN (Shared defaults)
    # ==================================================================
    "base": {
        "horizons_supported": ["intraday", "short_term", "swing", "long_term"],
        "debug": False,
        "coerce_numeric": True,
        "numeric_cols": ["Open", "High", "Low", "Close", "Volume"]
    },

    # ==================================================================
    # PATTERN DETECTION GLOBAL SETTINGS
    # ==================================================================
    "global": {
        "min_score_threshold": 60,
        "score_normalization": {
            "min": 0.0,
            "max": 100.0
        },
        "age_tracking_enabled": True,
        "require_formation_timestamp": True
    }
}

TIME_ESTIMATOR = {
    
    # ==================================================================
    # Velocity Factors (Trend Regime Adjustments)
    # ==================================================================
    "velocity_factors": {
        "strong_trend": {
            "min_strength": 7.0,
            "factor": 1.2
        },
        "normal_trend": {
            "min_strength": 5.0,
            "factor": 1.0
        },
        "weak_trend": {
            "max_strength": 5.0,
            "factor": 0.8
        }
    },
    
    # ==================================================================
    # Base Calculation Parameters
    # ==================================================================
    "base_friction": 0.8,                   # Global drag factor
    "default_trend_strength": 5.0,
    "default_velocity_factor": 1.0,
    "min_bars_per_atr": 2,
    "min_price_ratio": 0.001,               # 0.1% of price
    "min_atr_absolute": 0.01,
    "min_bars_result": 1,                   # Ensure at least 1 bar
    
    # ==================================================================
    # 🔴 HORIZON-SPECIFIC: Candles per Unit Time
    # ==================================================================
    "candles_per_unit": {
        "intraday": 4,                      # 4 x 15m candles = 1 hour
        "short_term": 1,                    # 1 daily candle = 1 day
        "long_term": 0.2,                   # 1 weekly candle = 5 days
        "multibagger": 0.05                 # 1 monthly candle = 20 days
    },
    
    # ==================================================================
    # Time Formatting Thresholds
    # ==================================================================
    "format_thresholds": {
        "hours_to_days": 1,                 # < 1 day = show hours
        "days_to_weeks": 30,                # < 30 days = show days
        "weeks_to_years": 365               # < 365 days = show weeks
    }
}


"""
Here is the definitive list of what you can DELETE and what you must KEEP to ensure a smooth transition.

✂️ DELETE THESE (Now Redundant)
These are fully handled by MASTER_CONFIG.

Horizon & Profile Logic:

❌ HORIZON_PROFILE_MAP (Replaced by MASTER_CONFIG["horizons"])

❌ HORIZON_FETCH_CONFIG (Replaced by MASTER_CONFIG["global"]["system"]["fetch"])

❌ ADX_HORIZON_CONFIG (Replaced by MASTER_CONFIG["horizons"][h]["indicators"]["adx_period"])

❌ STOCH_HORIZON_CONFIG (Replaced by MASTER_CONFIG["horizons"][h]["indicators"])

❌ ATR_HORIZON_CONFIG (Replaced by MASTER_CONFIG["horizons"][h]["indicators"]["atr_period"])

Volatility & Bands:

❌ VOL_BANDS (Replaced by MASTER_CONFIG["horizons"][h]["gates"]["volatility_bands_atr_pct"])

❌ VOL_BANDS_HORIZON_MULTIPLIERS (Implicit in MASTER_CONFIG's horizon-specific values)

❌ VOL_SCORING_THRESHOLDS (Replaced by MASTER_CONFIG["horizons"][h]["volatility"]["scoring_thresholds"])

❌ VOL_QUAL_MINS (Replaced by MASTER_CONFIG["global"]["boosts"]["volatility"])

Thresholds & Limits:

❌ RSI_SLOPE_THRESH (Replaced by MASTER_CONFIG["horizons"][h]["momentum_thresholds"])

❌ MACD_MOMENTUM_THRESH (Replaced by MASTER_CONFIG["horizons"][h]["momentum_thresholds"])

❌ TREND_THRESH (Replaced by MASTER_CONFIG["global"]["calculation_engine"]["composite_weights"])

❌ ATR_MULTIPLIERS (Replaced by MASTER_CONFIG["horizons"][h]["execution"]["stop_loss_atr_mult"])

❌ STOCH_FAST, STOCH_SLOW, STOCH_THRESHOLDS (Replaced by MASTER_CONFIG["global"]["calculation_engine"])

Weights (Merged into Global):

❌ FUNDAMENTAL_WEIGHTS (Replaced by MASTER_CONFIG["global"]["fundamental_weights"])

❌ QUALITY_WEIGHTS, GROWTH_WEIGHTS, VALUE_WEIGHTS, MOMENTUM_WEIGHTS (Replaced by MASTER_CONFIG["global"]["fundamental_weights"] and MASTER_CONFIG["global"]["calculation_engine"])

🛡️ KEEP THESE (Infrastructure & Reference)
These are NOT in MASTER_CONFIG and are required for the app to run.

Environment & App Settings:

✅ ENABLE_CACHE, ENABLE_CACHE_WARMER, ENABLE_JSON_ENRICHMENT

✅ ENABLE_VOLATILITY_QUALITY (Used as a feature flag)

Mappings & Labels (UI/Data Fetching):

✅ INDEX_TICKERS (Maps "nifty50" to "^NSEI")

✅ TECHNICAL_METRIC_MAP (Human-readable labels for UI)

✅ FUNDAMENTAL_ALIAS_MAP (Human-readable labels for UI)

✅ FUNDAMENTAL_FIELD_CANDIDATES (Critical for yfinance data parsing)

✅ SECTOR_PE_AVG (Reference data for valuation)

✅ flowchart_mapping (Used for the flowchart UI)

Legacy Compatibility (Optional but Recommended):

⚠️ TECHNICAL_WEIGHTS: Your indicators.py uses this in compute_technical_score. Keep this unless you refactor indicators.py to use MASTER_CONFIG scoring.

⚠️ CORE_TECHNICAL_SETUP_METRICS: Used in indicators.py to force specific calculations. Keep.
"""
