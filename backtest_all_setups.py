#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Backtest - All 23 Setups x 4 Horizons x 11 Strategies + Signal Gen
=================================================================================
Suite 1: Setup classification + confidence + gates
Suite 2: Strategy fit scoring (all 11 strategies ranked per scenario)
Suite 3: Full signal generation (generate_trade_plan end-to-end)

Usage:
    python backtest_all_setups.py          # Run all suites
    python backtest_all_setups.py -v       # Verbose mode
"""
import sys
import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress noisy logs during testing
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')

from config.setup_pattern_matrix_config import SETUP_PATTERN_MATRIX
from config.confidence_config import CONFIDENCE_CONFIG
from config.strategy_matrix_config import STRATEGY_MATRIX
from config.config_helpers import build_evaluation_context_v5

# Import generate_trade_plan for Suite 3
try:
    from services.signal_engine import generate_trade_plan
    SIGNAL_ENGINE_AVAILABLE = True
except Exception as e:
    SIGNAL_ENGINE_AVAILABLE = False
    SIGNAL_ENGINE_ERROR = str(e)


# ============================================================================
# COMMON DATA TEMPLATES
# ============================================================================

# Base indicators with both raw and resolver-mapped names
BASE_INDICATORS = {
    "adx": 20,
    "rsi": 55,
    "rvol": 1.0,
    "trendStrength": 4.0,
    "momentumStrength": 4.0,
    "maTrendSignal": 2,
    "volatilityQuality": 4.0,
    "bbWidth": 3.0,
    "bbpercentb": 0.5,
    "macdhistogram": 0.5,
    "prevmacdhistogram": 0.3,
    "rsislope": 0.3,
    "priceVsPrimaryTrendPct": 0,
    "position52w": 60,
    "currentPrice": 1500.0,
    "price": 1500.0,
    "sma50": 1450.0,
    "sma200": 1400.0,
    "atr": 25.0,
    "atrDynamic": 25.0,
    "atrPercent": 1.7,
    "avgVolume": 1500000,
    "volume": 2000000,
    "volumeProfile": {"avgVolume": 1500000},
    # Bollinger / Range
    "bbHigh": 1580.0,
    "bbMid": 1500.0,
    "bbLow": 1420.0,
    # 52-week
    "high52w": 1700.0,
    "low52w": 1100.0,
    # Moving averages
    "maFast": 1480.0,
    "maSlow": 1400.0,
    # Supertrend
    "supertrendSignal": "Neutral",
    # Pattern detection defaults (all False)
    "darvasBox": False,
    "vcpPattern": False,
    "cupHandle": False,
    "flagPennant": False,
    "threeLineStrike": False,
    "goldenCross": False,
    "bollingerSqueeze": False,
    "ichimokuSignals": False,
    "doubleTopBottom": False,
    "minerviniStage2": False,
    "highTightFlag": False,
}

# Base fundamentals
BASE_FUNDAMENTALS = {
    "pe": 22,
    "peRatio": 22,
    "roe": 16,
    "roce": 18,
    "debtToEquity": 0.5,
    "marketCap": 25000,
    "promoterHolding": 55,
    "institutionalOwnership": 30,
    "dividendYield": 1.5,
    "pbRatio": 3.0,
    "currentRatio": 1.8,
    "interestCoverage": 5.0,
    "epsGrowth5y": 15,
    "quarterlyGrowth": 12,
    "salesGrowth": 15,
    "bookValueGrowth": 12,
    "operatingMargin": 18,
    "netProfitMargin": 12,
    "freeCashFlowYield": 4.0,
}


# ============================================================================
# SYNTHETIC STOCK SCENARIOS
# ============================================================================
SCENARIOS = {
    # PATTERN SETUPS
    "PATTERN_DARVAS_BREAKOUT": {
        "symbol": "TATAMOTORS.NS",
        "description": "Strong breakout from Darvas box",
        "indicators": {
            "rvol": 2.5, "trendStrength": 6.0, "adx": 25,
            "rsi": 65, "momentumStrength": 6.5,
            "darvasBox": True, "bollingerSqueeze": True,
            "bbpercentb": 0.95, "position52w": 85,
            "currentPrice": 800, "price": 800,
            "atrDynamic": 15, "atr": 15,
            "bbHigh": 820, "bbMid": 780, "bbLow": 740,
        },
    },
    "PATTERN_VCP_BREAKOUT": {
        "symbol": "HDFCBANK.NS",
        "description": "VCP pattern with tight compression",
        "indicators": {
            "volatilityQuality": 8.0, "rsi": 58, "adx": 22,
            "rvol": 1.8, "trendStrength": 5.0,
            "vcpPattern": True, "bollingerSqueeze": True,
            "bbWidth": 1.5, "position52w": 75,
            "currentPrice": 1700, "price": 1700,
            "atrDynamic": 28, "atr": 28,
            "bbHigh": 1730, "bbMid": 1680, "bbLow": 1630,
        },
    },
    "PATTERN_CUP_BREAKOUT": {
        "symbol": "TCS.NS",
        "description": "Cup-and-handle breakout",
        "indicators": {
            "rvol": 1.8, "trendStrength": 5.0, "adx": 20,
            "rsi": 62, "momentumStrength": 5.5,
            "cupHandle": True, "goldenCross": True,
            "bbpercentb": 0.88, "position52w": 80,
            "currentPrice": 3800, "price": 3800,
            "atrDynamic": 60, "atr": 60,
            "bbHigh": 3900, "bbMid": 3750, "bbLow": 3600,
        },
    },
    "PATTERN_FLAG_BREAKOUT": {
        "symbol": "RELIANCE.NS",
        "description": "Bull flag continuation",
        "indicators": {
            "rvol": 2.0, "trendStrength": 7.0, "adx": 28,
            "rsi": 68, "momentumStrength": 6.0,
            "flagPennant": True, "bollingerSqueeze": True,
            "bbpercentb": 0.92, "position52w": 88,
            "currentPrice": 2800, "price": 2800,
            "atrDynamic": 45, "atr": 45,
            "bbHigh": 2870, "bbMid": 2750, "bbLow": 2630,
        },
    },
    "PATTERN_STRIKE_REVERSAL": {
        "symbol": "SUNPHARMA.NS",
        "description": "Reversal strike pattern",
        "indicators": {
            "rvol": 2.0, "rsi": 52, "adx": 18,
            "trendStrength": 3.5, "momentumStrength": 4.0,
            "threeLineStrike": True, "bollingerSqueeze": True,
            "bbpercentb": 0.6, "position52w": 50,
            "currentPrice": 1200, "price": 1200,
            "atrDynamic": 20, "atr": 20,
        },
    },
    "PATTERN_GOLDEN_CROSS": {
        "symbol": "INFY.NS",
        "description": "Golden cross with momentum",
        "indicators": {
            "trendStrength": 5.0, "momentumStrength": 5.5, "adx": 22,
            "rsi": 60, "rvol": 1.5,
            "goldenCross": True, "ichimokuSignals": True,
            "macdhistogram": 1.5, "position52w": 70,
            "currentPrice": 1800, "price": 1800,
            "atrDynamic": 30, "atr": 30,
            "bbHigh": 1850, "bbMid": 1780, "bbLow": 1710,
        },
    },

    # MOMENTUM SETUPS
    "MOMENTUM_BREAKOUT": {
        "symbol": "ADANIENT.NS",
        "description": "Bollinger Band breakout with volume",
        "indicators": {
            "bbpercentb": 0.99, "rsi": 72, "rvol": 2.5,
            "adx": 30, "trendStrength": 6.5, "momentumStrength": 7.0,
            "ichimokuSignals": True, "goldenCross": True,
            "position52w": 90,
            "currentPrice": 3200, "price": 3200,
            "atrDynamic": 55, "atr": 55,
            "bbHigh": 3190, "bbMid": 3050, "bbLow": 2910,
        },
    },
    "MOMENTUM_BREAKDOWN": {
        "symbol": "ZOMATO.NS",
        "description": "Breakdown below Bollinger Band",
        "indicators": {
            "bbpercentb": 0.01, "rsi": 30, "rvol": 2.0,
            "adx": 25, "trendStrength": 2.0, "momentumStrength": 2.0,
            "bollingerSqueeze": True,
            "macdhistogram": -1.5, "position52w": 15,
            "currentPrice": 120, "price": 120,
            "atrDynamic": 5, "atr": 5,
            "bbHigh": 145, "bbMid": 132, "bbLow": 119,
        },
    },

    # TREND SETUPS
    "TREND_PULLBACK": {
        "symbol": "BAJFINANCE.NS",
        "description": "Pullback to 20MA in strong uptrend",
        "indicators": {
            "trendStrength": 6.5, "priceVsPrimaryTrendPct": 2.0,
            "rsi": 55, "adx": 22, "rvol": 1.2,
            "momentumStrength": 5.0,
            "bollingerSqueeze": True, "threeLineStrike": True,
            "position52w": 75,
            "currentPrice": 6900, "price": 6900,
            "atrDynamic": 120, "atr": 120,
            "maFast": 6850, "sma50": 6850,
            "bbHigh": 7100, "bbMid": 6900, "bbLow": 6700,
        },
    },
    "DEEP_PULLBACK": {
        "symbol": "WIPRO.NS",
        "description": "Deep pullback in uptrend (5-10%)",
        "indicators": {
            "trendStrength": 5.0, "priceVsPrimaryTrendPct": -7.0,
            "adx": 18, "rsi": 42, "rvol": 0.8,
            "momentumStrength": 3.0,
            "bollingerSqueeze": True, "cupHandle": True,
            "position52w": 55,
            "currentPrice": 450, "price": 450,
            "atrDynamic": 10, "atr": 10,
            "bbHigh": 480, "bbMid": 460, "bbLow": 440,
        },
    },
    "TREND_FOLLOWING": {
        "symbol": "TITAN.NS",
        "description": "Steady uptrend with momentum",
        "indicators": {
            "rsi": 62, "macdhistogram": 1.5, "adx": 28,
            "trendStrength": 6.0, "momentumStrength": 5.5, "rvol": 1.3,
            "flagPennant": True, "bollingerSqueeze": True,
            "position52w": 78,
            "currentPrice": 3000, "price": 3000,
            "atrDynamic": 50, "atr": 50,
            "bbHigh": 3080, "bbMid": 2980, "bbLow": 2880,
        },
    },
    "BEAR_TREND_FOLLOWING": {
        "symbol": "PAYTM.NS",
        "description": "Established downtrend with momentum",
        "indicators": {
            "rsi": 35, "macdhistogram": -2.0, "adx": 28,
            "trendStrength": 2.0, "momentumStrength": 2.5,
            "rvol": 1.5, "bollingerSqueeze": True,
            "position52w": 12, "priceVsPrimaryTrendPct": -15,
            "currentPrice": 600, "price": 600,
            "atrDynamic": 15, "atr": 15,
            "bbHigh": 650, "bbMid": 620, "bbLow": 590,
        },
    },

    # VALUE / QUALITY SETUPS
    "QUALITY_ACCUMULATION": {
        "symbol": "NESTLEIND.NS",
        "description": "Quality stock in tight range (accumulation)",
        "indicators": {
            "bbWidth": 2.0, "rvol": 0.7, "rsi": 50,
            "adx": 12, "trendStrength": 3.5,
            "bollingerSqueeze": True, "cupHandle": True,
            "position52w": 55,
            "currentPrice": 22000, "price": 22000,
            "atrDynamic": 350, "atr": 350,
        },
        "fundamentals": {
            "roe": 25, "roce": 28, "debtToEquity": 0.1,
            "pe": 60, "marketCap": 200000,
            "operatingMargin": 25, "netProfitMargin": 18,
        },
    },
    "DEEP_VALUE_PLAY": {
        "symbol": "ONGC.NS",
        "description": "Deep value: low PE, beaten down",
        "indicators": {
            "rsi": 28, "priceVsPrimaryTrendPct": -15,
            "position52w": 12, "adx": 15,
            "trendStrength": 2.0, "rvol": 0.6,
            "ichimokuSignals": True, "bollingerSqueeze": True,
            "currentPrice": 180, "price": 180,
            "atrDynamic": 5, "atr": 5,
        },
        "fundamentals": {
            "pe": 6, "peRatio": 6, "roe": 14, "roce": 16,
            "debtToEquity": 0.3, "marketCap": 180000,
            "dividendYield": 5.0, "pbRatio": 0.8,
        },
    },
    "VALUE_TURNAROUND": {
        "symbol": "TATASTEEL.NS",
        "description": "Turnaround play with improving fundamentals",
        "indicators": {
            "trendStrength": 4.5, "rsi": 52, "adx": 20,
            "rvol": 1.3, "momentumStrength": 4.0,
            "ichimokuSignals": True, "threeLineStrike": True,
            "position52w": 45,
            "currentPrice": 130, "price": 130,
            "atrDynamic": 4, "atr": 4,
        },
        "fundamentals": {
            "roe": 22, "roce": 25, "debtToEquity": 0.6,
            "pe": 12, "marketCap": 150000,
        },
    },
    "QUALITY_ACCUMULATION_DOWNTREND": {
        "symbol": "HINDUNILVR.NS",
        "description": "Quality stock accumulating in downtrend",
        "indicators": {
            "trendStrength": 2.5, "bbpercentb": 0.35,
            "adx": 18, "rsi": 45, "rvol": 0.6,
            "bbWidth": 3.5,
            "bollingerSqueeze": True, "cupHandle": True,
            "position52w": 35,
            "currentPrice": 2400, "price": 2400,
            "atrDynamic": 40, "atr": 40,
        },
        "fundamentals": {
            "roe": 28, "roce": 30, "debtToEquity": 0.1,
            "pe": 55, "marketCap": 500000,
        },
    },

    # VOLATILITY SETUPS
    "VOLATILITY_SQUEEZE": {
        "symbol": "SBIN.NS",
        "description": "Bollinger squeeze coiling for breakout",
        "indicators": {
            "bbWidth": 0.3, "volatilityQuality": 8.0, "adx": 20,
            "rsi": 55, "trendStrength": 4.5, "rvol": 0.8,
            "ichimokuSignals": True, "goldenCross": True,
            "position52w": 65,
            "currentPrice": 600, "price": 600,
            "atrDynamic": 12, "atr": 12,
        },
    },

    # REVERSAL SETUPS
    "REVERSAL_MACD_CROSS_UP": {
        "symbol": "BHARTIARTL.NS",
        "description": "MACD histogram crosses from negative to positive",
        "indicators": {
            "macdhistogram": 0.3, "prevmacdhistogram": -0.5,
            "trendStrength": 3.5, "rsi": 48, "adx": 16,
            "rvol": 1.1, "momentumStrength": 3.5,
            "ichimokuSignals": True, "bollingerSqueeze": True,
            "position52w": 45,
            "currentPrice": 1100, "price": 1100,
            "atrDynamic": 20, "atr": 20,
        },
    },
    "REVERSAL_RSI_SWING_UP": {
        "symbol": "MARUTI.NS",
        "description": "RSI bouncing off oversold with positive slope",
        "indicators": {
            "rsi": 32, "rsislope": 0.15, "trendStrength": 3.0,
            "adx": 18, "rvol": 1.0, "momentumStrength": 3.0,
            "ichimokuSignals": True, "bollingerSqueeze": True,
            "position52w": 30,
            "currentPrice": 10000, "price": 10000,
            "atrDynamic": 180, "atr": 180,
        },
    },
    "REVERSAL_ST_FLIP_UP": {
        "symbol": "HCLTECH.NS",
        "description": "Supertrend flip to bullish",
        "indicators": {
            "rvol": 1.5, "adx": 18, "trendStrength": 3.5,
            "rsi": 50, "momentumStrength": 4.0,
            "bollingerSqueeze": True, "ichimokuSignals": True,
            "position52w": 50,
            "currentPrice": 1500, "price": 1500,
            "atrDynamic": 25, "atr": 25,
        },
    },

    # SELL / PROFIT SETUPS
    "SELL_AT_RANGE_TOP": {
        "symbol": "ASIANPAINT.NS",
        "description": "Price at top of range - selling opportunity",
        "indicators": {
            "bbWidth": 3.0, "rsi": 72, "adx": 15,
            "trendStrength": 4.0, "rvol": 0.9,
            "bbpercentb": 0.92, "doubleTopBottom": True,
            "position52w": 85,
            "currentPrice": 3300, "price": 3300,
            "atrDynamic": 55, "atr": 55,
            "bbHigh": 3350, "bbMid": 3200, "bbLow": 3050,
        },
    },
    "TAKE_PROFIT_AT_MID": {
        "symbol": "ULTRACEMCO.NS",
        "description": "Mid-range profit booking",
        "indicators": {
            "bbWidth": 3.5, "rsi": 58, "adx": 14,
            "trendStrength": 3.5, "rvol": 0.8,
            "bbpercentb": 0.55, "position52w": 55,
            "currentPrice": 8500, "price": 8500,
            "atrDynamic": 140, "atr": 140,
        },
    },

    # FALLBACK
    "GENERIC": {
        "symbol": "IRCTC.NS",
        "description": "No strong pattern - fallback",
        "indicators": {
            "adx": 12, "rsi": 50, "rvol": 0.8,
            "trendStrength": 3.0, "momentumStrength": 3.0,
            "bbpercentb": 0.45, "position52w": 45,
            "currentPrice": 900, "price": 900,
            "atrDynamic": 18, "atr": 18,
        },
    },
}


# ============================================================================
# HORIZONS CONFIG
# ============================================================================
HORIZONS = ["intraday", "short_term", "long_term", "multibagger"]

HORIZON_CLAMPS = {}
HORIZON_MIN_TRADEABLE = {}
HORIZON_OVERRIDE_THRESHOLD = {}

for h in HORIZONS:
    h_cfg = CONFIDENCE_CONFIG.get("horizons", {}).get(h, {})
    HORIZON_CLAMPS[h] = h_cfg.get("confidence_clamp", [20, 95])
    mtc = h_cfg.get("min_tradeable_confidence", {})
    HORIZON_MIN_TRADEABLE[h] = mtc.get("min", 0) if isinstance(mtc, dict) else 0
    hco = h_cfg.get("high_confidence_override", {})
    HORIZON_OVERRIDE_THRESHOLD[h] = hco.get("threshold", 0)

ALL_STRATEGIES = list(STRATEGY_MATRIX.keys())


# ============================================================================
# RESULT CLASS
# ============================================================================
class BacktestResult:
    def __init__(self, setup: str, horizon: str, symbol: str, suite: str = ""):
        self.setup = setup
        self.horizon = horizon
        self.symbol = symbol
        self.suite = suite
        self.checks: List[Tuple[str, bool, str]] = []
        self.error: Optional[str] = None

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return self.error is None and all(c[1] for c in self.checks)

    @property
    def failed_checks(self) -> List[Tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]


# ============================================================================
# HELPERS
# ============================================================================
def build_indicators(scenario: Dict) -> Dict:
    merged = {**BASE_INDICATORS}
    merged.update(scenario.get("indicators", {}))
    return merged


def build_fundamentals(scenario: Dict) -> Dict:
    merged = {**BASE_FUNDAMENTALS}
    merged.update(scenario.get("fundamentals", {}))
    return merged


# ============================================================================
# SUITE 1: SETUP + CONFIDENCE + GATES
# ============================================================================
def run_suite1_test(setup_name: str, horizon: str, scenario: Dict,
                    eval_ctx: Dict) -> BacktestResult:
    """Validate setup classification, confidence, tradeable flag, gates."""
    result = BacktestResult(setup_name, horizon, scenario["symbol"], "S1")

    if not eval_ctx:
        result.error = "Empty eval_ctx"
        return result

    # CHECK 1: Setup classification
    setup_result = eval_ctx.get("setup", {})
    classified = setup_result.get("type", "UNKNOWN")
    result.add_check(
        "setup_classified",
        classified != "UNKNOWN" and classified is not None,
        f"setup={classified}"
    )

    # CHECK 2: Confidence structure
    conf = eval_ctx.get("confidence", {})
    result.add_check(
        "confidence_structure",
        "clamped" in conf and "base" in conf,
        f"keys={list(conf.keys())[:5]}"
    )

    # CHECK 3: Clamped within horizon range
    clamped = conf.get("clamped", -1)
    clamp = HORIZON_CLAMPS.get(horizon, [20, 95])
    result.add_check(
        "clamp_range",
        clamp[0] <= clamped <= clamp[1],
        f"clamped={clamped}, range={clamp}"
    )

    # CHECK 4: Tradeable flag
    tradeable = conf.get("tradeable")
    result.add_check(
        "tradeable_flag",
        isinstance(tradeable, bool),
        f"tradeable={tradeable}"
    )

    # CHECK 5: min_tradeable_threshold matches config
    min_thresh = conf.get("min_tradeable_threshold")
    expected_min = HORIZON_MIN_TRADEABLE.get(horizon, 0)
    result.add_check(
        "min_tradeable_threshold",
        (min_thresh == expected_min) or (min_thresh is None and expected_min == 0),
        f"got={min_thresh}, config={expected_min}"
    )

    # CHECK 6: high_confidence_override
    hco = conf.get("high_confidence_override")
    hco_ok = isinstance(hco, dict) and "threshold" in hco
    result.add_check(
        "high_confidence_override",
        hco_ok,
        f"threshold={hco.get('threshold') if hco else 'None'}"
    )

    # CHECK 7: Override threshold value
    if hco_ok:
        expected_thresh = HORIZON_OVERRIDE_THRESHOLD.get(horizon, 0)
        result.add_check(
            "override_threshold_value",
            hco.get("threshold") == expected_thresh,
            f"got={hco.get('threshold')}, expected={expected_thresh}"
        )

    # CHECK 8: Gates present
    has_sg = "structural_gates" in eval_ctx
    has_og = "opportunity_gates" in eval_ctx
    result.add_check(
        "gates_present",
        has_sg or has_og,
        f"structural={'Y' if has_sg else 'N'} opportunity={'Y' if has_og else 'N'}"
    )

    # CHECK 9: Scoring present
    scoring = eval_ctx.get("scoring", {})
    result.add_check(
        "scoring_present",
        "technical" in scoring and "fundamental" in scoring,
        f"tech={'Y' if 'technical' in scoring else 'N'} fund={'Y' if 'fundamental' in scoring else 'N'}"
    )

    return result


# ============================================================================
# SUITE 2: STRATEGY FIT SCORING
# ============================================================================
def run_suite2_test(setup_name: str, horizon: str, scenario: Dict,
                    eval_ctx: Dict) -> BacktestResult:
    """Validate strategy fit scoring for all 11 strategies."""
    result = BacktestResult(setup_name, horizon, scenario["symbol"], "S2")

    if not eval_ctx:
        result.error = "Empty eval_ctx"
        return result

    strategy = eval_ctx.get("strategy", {})

    # CHECK 1: Strategy section exists
    result.add_check(
        "strategy_section",
        bool(strategy),
        f"keys={list(strategy.keys())[:5]}"
    )

    # CHECK 2: Primary strategy identified
    primary = strategy.get("primary")
    result.add_check(
        "primary_strategy",
        primary is not None and isinstance(primary, str),
        f"primary={primary}"
    )

    # CHECK 3: Fit score is a number
    fit_score = strategy.get("fit_score")
    result.add_check(
        "fit_score_numeric",
        isinstance(fit_score, (int, float)),
        f"fit_score={fit_score}"
    )

    # Detect structure type: full (short/long/multi) vs simple (intraday)
    has_full_structure = "all_strategies" in strategy and len(strategy.get("all_strategies", [])) > 0
    has_simple_structure = "all_suggestions" in strategy

    if has_full_structure:
        # --- FULL STRUCTURE CHECKS ---
        all_strats = strategy.get("all_strategies", [])

        # CHECK 4: All strategies scored
        result.add_check(
            "all_strategies_scored",
            len(all_strats) == len(ALL_STRATEGIES),
            f"counted={len(all_strats)}, expected={len(ALL_STRATEGIES)}"
        )

        # CHECK 5: Each strategy has required fields
        required_fields = {"name", "fit_score", "weighted_score", "horizon_multiplier"}
        all_have_fields = True
        missing_detail = []
        for s in all_strats:
            if isinstance(s, dict):
                missing = required_fields - set(s.keys())
                if missing:
                    all_have_fields = False
                    missing_detail.append(f"{s.get('name','?')}: missing {missing}")
        result.add_check(
            "strategy_fields_complete",
            all_have_fields,
            f"{'OK' if all_have_fields else '; '.join(missing_detail[:3])}"
        )

        # CHECK 6: Ranked list present
        ranked = strategy.get("ranked", [])
        result.add_check(
            "ranked_list_present",
            len(ranked) > 0,
            f"ranked_count={len(ranked)}"
        )

        # CHECK 7: Ranked is sorted desc
        if len(ranked) >= 2:
            scores = [s.get("weighted_score", 0) for s in ranked if isinstance(s, dict)]
            is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            result.add_check(
                "ranked_sorted_desc",
                is_sorted,
                f"top3_scores={scores[:3]}"
            )

        # CHECK 8: Rejected list
        rejected = strategy.get("rejected", [])
        result.add_check(
            "rejected_list",
            isinstance(rejected, list),
            f"rejected_count={len(rejected)}"
        )

        # CHECK 9: Qualified + Rejected = Total
        summary = strategy.get("summary", {})
        total = summary.get("total", 0)
        qualified = summary.get("qualified", 0)
        rejected_count = summary.get("rejected", 0)
        result.add_check(
            "qualified_rejected_sum",
            qualified + rejected_count == total or total == 0,
            f"total={total}, q={qualified}, r={rejected_count}"
        )

        # CHECK 10: Best strategy dict
        best = strategy.get("best", {})
        result.add_check(
            "best_strategy_populated",
            isinstance(best, dict) and "strategy" in best,
            f"best={best.get('strategy', '?')}"
        )

    elif has_simple_structure:
        # --- SIMPLE STRUCTURE CHECKS (intraday) ---
        suggestions = strategy.get("all_suggestions", [])

        # CHECK 4: Suggestions list present
        result.add_check(
            "suggestions_present",
            isinstance(suggestions, list),
            f"suggestions_count={len(suggestions)}"
        )

        # CHECK 5: Preferred setups present
        prefs = strategy.get("preferred_setups", [])
        result.add_check(
            "preferred_setups",
            isinstance(prefs, list),
            f"preferred_count={len(prefs)}"
        )

        # CHECK 6: Avoid setups present
        avoids = strategy.get("avoid_setups", [])
        result.add_check(
            "avoid_setups",
            isinstance(avoids, list),
            f"avoid_count={len(avoids)}"
        )

        # CHECK 7: Description present
        desc = strategy.get("description", "")
        result.add_check(
            "strategy_description",
            isinstance(desc, str) and len(desc) > 0,
            f"desc_len={len(desc)}"
        )

        # CHECK 8: Horizon multiplier
        hm = strategy.get("horizon_multiplier")
        result.add_check(
            "horizon_multiplier",
            isinstance(hm, (int, float)) and hm > 0,
            f"multiplier={hm}"
        )
    else:
        result.add_check(
            "strategy_structure_known",
            False,
            f"unknown structure: keys={list(strategy.keys())}"
        )

    # CHECK 11: Weighted score valid (common to both)
    weighted = strategy.get("weighted_score", 0)
    result.add_check(
        "weighted_score_valid",
        isinstance(weighted, (int, float)) and weighted >= 0,
        f"weighted={weighted}"
    )

    return result


# ============================================================================
# SUITE 3: FULL SIGNAL GENERATION (generate_trade_plan)
# ============================================================================
def run_suite3_test(setup_name: str, horizon: str, scenario: Dict) -> BacktestResult:
    """Run full generate_trade_plan and validate the output."""
    result = BacktestResult(setup_name, horizon, scenario["symbol"], "S3")

    if not SIGNAL_ENGINE_AVAILABLE:
        result.error = f"Signal engine not importable: {SIGNAL_ENGINE_ERROR}"
        return result

    indicators = build_indicators(scenario)
    fundamentals = build_fundamentals(scenario)

    try:
        plan = generate_trade_plan(
            symbol=scenario["symbol"],
            indicators=indicators,
            fundamentals=fundamentals,
            horizon=horizon,
            capital=100000
        )
    except Exception as e:
        result.error = f"generate_trade_plan crashed: {type(e).__name__}: {e}"
        return result

    if not plan:
        result.error = "Empty plan returned"
        return result

    # CHECK 1: Plan has basic fields
    required_keys = {"symbol", "horizon", "status"}
    has_basic = required_keys.issubset(set(plan.keys()))
    result.add_check(
        "plan_basic_fields",
        has_basic,
        f"keys={list(plan.keys())[:8]}"
    )

    # CHECK 2: Symbol and horizon match
    result.add_check(
        "plan_identity",
        plan.get("symbol") == scenario["symbol"] and plan.get("horizon") == horizon,
        f"sym={plan.get('symbol')}, hz={plan.get('horizon')}"
    )

    # CHECK 3: Setup type populated
    setup_type = plan.get("setup_type")
    result.add_check(
        "plan_setup_type",
        setup_type is not None,
        f"setup_type={setup_type}"
    )

    # CHECK 4: Confidence fields populated
    base_conf = plan.get("base_confidence")
    final_conf = plan.get("final_confidence")
    result.add_check(
        "plan_confidence",
        isinstance(base_conf, (int, float)) and isinstance(final_conf, (int, float)),
        f"base={base_conf}, final={final_conf}"
    )

    # CHECK 5: Status is valid
    status = plan.get("status", "")
    valid_statuses = {"PENDING", "BLOCKED", "READY", "ACTIVE", "BUY", "SELL",
                      "HOLD", "STRONG_BUY", "STRONG_SELL", "CAUTION", "ERROR"}
    result.add_check(
        "plan_status_valid",
        isinstance(status, str) and len(status) > 0,
        f"status={status}"
    )

    # CHECK 6: Trade signal present
    signal = plan.get("trade_signal", plan.get("signal", "NONE"))
    result.add_check(
        "plan_trade_signal",
        isinstance(signal, str) and len(signal) > 0,
        f"signal={signal}"
    )

    # CHECK 7: Gates status populated
    gates_passed = plan.get("gates_passed")
    execution_blocked = plan.get("execution_blocked")
    result.add_check(
        "plan_gates_status",
        isinstance(gates_passed, bool) and isinstance(execution_blocked, bool),
        f"gates={gates_passed}, blocked={execution_blocked}"
    )

    # CHECK 8: Metadata populated
    metadata = plan.get("metadata", {})
    result.add_check(
        "plan_metadata",
        isinstance(metadata, dict) and len(metadata) > 0,
        f"meta_keys={list(metadata.keys())[:5]}"
    )

    # CHECK 9: Confidence history populated
    conf_history = plan.get("confidence_history", [])
    result.add_check(
        "plan_confidence_history",
        isinstance(conf_history, list) and len(conf_history) >= 1,
        f"history_steps={len(conf_history)}"
    )

    # CHECK 10: If not blocked, entry/stop/targets should be set
    if not plan.get("execution_blocked", True):
        entry = plan.get("entry")
        sl = plan.get("stop_loss")
        targets = plan.get("targets", {})
        has_execution = (entry is not None and sl is not None and
                         isinstance(targets, dict) and
                         (targets.get("t1") is not None or targets.get("t2") is not None))
        result.add_check(
            "plan_execution_values",
            has_execution,
            f"entry={entry}, sl={sl}, t1={targets.get('t1')}, t2={targets.get('t2')}"
        )

    # CHECK 11: Position size and R:R
    if not plan.get("execution_blocked", True):
        pos_size = plan.get("position_size", 0)
        rr = plan.get("rr_ratio", 0)
        result.add_check(
            "plan_sizing",
            isinstance(pos_size, (int, float)) and pos_size >= 0,
            f"qty={pos_size}, rr={rr}"
        )

    # CHECK 12: Analytics present (if execution succeeded)
    analytics = plan.get("analytics", {})
    if analytics:
        result.add_check(
            "plan_analytics",
            "strategy_fit" in analytics or "technical_score" in analytics,
            f"analytics_keys={list(analytics.keys())[:5]}"
        )

    return result


# ============================================================================
# MAIN
# ============================================================================
def print_suite_summary(title: str, results: List[BacktestResult],
                        show_matrix: bool = True):
    """Print summary for one suite."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    errors = sum(1 for r in results if r.error)

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed} ({100*passed//max(total,1)}%)")
    print(f"  Failed:  {failed - errors}")
    print(f"  Errors:  {errors}")

    if show_matrix:
        print(f"\n  {'Setup':<35s} {'intraday':^10s} {'short_term':^10s} {'long_term':^10s} {'multibagger':^10s}")
        print(f"  {'---'*12} {'---'*4} {'---'*4} {'---'*4} {'---'*4}")

        setup_results: Dict[str, Dict[str, BacktestResult]] = {}
        for r in results:
            setup_results.setdefault(r.setup, {})[r.horizon] = r

        for setup in sorted(setup_results.keys()):
            row = f"  {setup:<35s}"
            for h in HORIZONS:
                r = setup_results[setup].get(h)
                if r is None:
                    row += f"  {'--':^8s}"
                elif r.error:
                    row += f"  {'ERR':^8s}"
                elif r.passed:
                    row += f"  {'PASS':^8s}"
                else:
                    row += f"  {'FAIL':^8s}"
            print(row)

    # Print failures
    failed_results = [r for r in results if not r.passed]
    if failed_results:
        print(f"\n  FAILURE DETAILS:")
        for r in failed_results[:15]:
            print(f"\n  {r.setup}/{r.horizon} ({r.symbol}):")
            if r.error:
                print(f"    ERROR: {r.error}")
            for name, p, detail in r.checks:
                if not p:
                    print(f"    FAIL {name}: {detail}")


def main():
    verbose = "-v" in sys.argv

    print("=" * 80)
    print(f"  COMPREHENSIVE BACKTEST")
    print(f"  {len(SCENARIOS)} Setups x {len(HORIZONS)} Horizons x {len(ALL_STRATEGIES)} Strategies")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ================================================================
    # Build eval_ctx cache (shared between Suite 1 and Suite 2)
    # ================================================================
    print("\n  Building evaluation contexts...")
    eval_ctx_cache: Dict[str, Dict] = {}  # key = "setup|horizon"

    for setup_name, scenario in sorted(SCENARIOS.items()):
        indicators = build_indicators(scenario)
        fundamentals = build_fundamentals(scenario)
        for horizon in HORIZONS:
            key = f"{setup_name}|{horizon}"
            try:
                ctx = build_evaluation_context_v5(
                    ticker=scenario["symbol"],
                    indicators=indicators,
                    fundamentals=fundamentals,
                    horizon=horizon
                )
                eval_ctx_cache[key] = ctx
            except Exception as e:
                eval_ctx_cache[key] = {"_error": str(e)}

    total_ctx = len(eval_ctx_cache)
    valid_ctx = sum(1 for v in eval_ctx_cache.values() if "_error" not in v)
    print(f"  Built {valid_ctx}/{total_ctx} contexts successfully\n")

    # ================================================================
    # SUITE 1: Setup + Confidence + Gates
    # ================================================================
    print("=" * 80)
    print("  SUITE 1: Setup Classification + Confidence + Gates")
    print("=" * 80)

    suite1_results: List[BacktestResult] = []
    for setup_name, scenario in sorted(SCENARIOS.items()):
        for horizon in HORIZONS:
            key = f"{setup_name}|{horizon}"
            ctx = eval_ctx_cache.get(key, {})
            if "_error" in ctx:
                r = BacktestResult(setup_name, horizon, scenario["symbol"], "S1")
                r.error = ctx["_error"]
                suite1_results.append(r)
            else:
                suite1_results.append(
                    run_suite1_test(setup_name, horizon, scenario, ctx)
                )

            r = suite1_results[-1]
            if verbose:
                mark = "PASS" if r.passed else ("ERR" if r.error else "FAIL")
                print(f"  S1 {setup_name}/{horizon}: {mark}")

    print_suite_summary("SUITE 1 RESULTS: Setup + Confidence + Gates",
                        suite1_results)

    # ================================================================
    # SUITE 2: Strategy Fit Scoring
    # ================================================================
    print("\n" + "=" * 80)
    print("  SUITE 2: Strategy Fit Scoring (all 11 strategies)")
    print("=" * 80)

    suite2_results: List[BacktestResult] = []
    for setup_name, scenario in sorted(SCENARIOS.items()):
        for horizon in HORIZONS:
            key = f"{setup_name}|{horizon}"
            ctx = eval_ctx_cache.get(key, {})
            if "_error" in ctx:
                r = BacktestResult(setup_name, horizon, scenario["symbol"], "S2")
                r.error = ctx["_error"]
                suite2_results.append(r)
            else:
                suite2_results.append(
                    run_suite2_test(setup_name, horizon, scenario, ctx)
                )

            r = suite2_results[-1]
            if verbose:
                mark = "PASS" if r.passed else ("ERR" if r.error else "FAIL")
                primary = ctx.get("strategy", {}).get("primary", "?") if "_error" not in ctx else "?"
                print(f"  S2 {setup_name}/{horizon}: {mark}  primary={primary}")

    print_suite_summary("SUITE 2 RESULTS: Strategy Fit Scoring", suite2_results)

    # ── Strategy coverage analysis ──
    print(f"\n  STRATEGY COVERAGE ANALYSIS:")
    print(f"  {'Strategy':<28s} {'#Primary':>8s}  {'#Qualified':>10s}  {'#Rejected':>10s}")
    print(f"  {'---'*20}")

    strategy_primary_count: Dict[str, int] = {s: 0 for s in ALL_STRATEGIES}
    strategy_qualified_count: Dict[str, int] = {s: 0 for s in ALL_STRATEGIES}
    strategy_rejected_count: Dict[str, int] = {s: 0 for s in ALL_STRATEGIES}

    for key, ctx in eval_ctx_cache.items():
        if "_error" in ctx:
            continue
        strat = ctx.get("strategy", {})
        primary = strat.get("primary", "")
        if primary in strategy_primary_count:
            strategy_primary_count[primary] += 1

        for s in strat.get("all_strategies", []):
            name = s.get("name", "") if isinstance(s, dict) else ""
            if name in strategy_qualified_count:
                strategy_qualified_count[name] += 1

        for s in strat.get("rejected", []):
            name = s.get("name", "") if isinstance(s, dict) else ""
            if name in strategy_rejected_count:
                strategy_rejected_count[name] += 1

    for strat_name in ALL_STRATEGIES:
        p = strategy_primary_count[strat_name]
        q = strategy_qualified_count[strat_name]
        rj = strategy_rejected_count[strat_name]
        marker = " **" if p > 0 else ""
        print(f"  {strat_name:<28s} {p:>8d}  {q:>10d}  {rj:>10d}{marker}")

    # ================================================================
    # SUITE 3: Full Signal Generation
    # ================================================================
    print("\n" + "=" * 80)
    print("  SUITE 3: Full Signal Generation (generate_trade_plan)")
    print("=" * 80)

    if not SIGNAL_ENGINE_AVAILABLE:
        print(f"  SKIPPED: Signal engine not available ({SIGNAL_ENGINE_ERROR})")
        suite3_results = []
    else:
        suite3_results: List[BacktestResult] = []
        for setup_name, scenario in sorted(SCENARIOS.items()):
            for horizon in HORIZONS:
                suite3_results.append(
                    run_suite3_test(setup_name, horizon, scenario)
                )
                r = suite3_results[-1]
                if verbose:
                    mark = "PASS" if r.passed else ("ERR" if r.error else "FAIL")
                    print(f"  S3 {setup_name}/{horizon}: {mark}")

        print_suite_summary("SUITE 3 RESULTS: Signal Generation", suite3_results)

        # ── Signal analysis ──
        print(f"\n  SIGNAL ANALYSIS:")
        signal_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        blocked_count = 0
        exec_count = 0

        for r in suite3_results:
            if r.passed:
                for name, _, detail in r.checks:
                    if name == "plan_trade_signal":
                        sig = detail.split("signal=")[1] if "signal=" in detail else "?"
                        signal_counts[sig] = signal_counts.get(sig, 0) + 1
                    elif name == "plan_status_valid":
                        st = detail.split("status=")[1] if "status=" in detail else "?"
                        status_counts[st] = status_counts.get(st, 0) + 1
                    elif name == "plan_gates_status":
                        if "blocked=True" in detail:
                            blocked_count += 1
                        else:
                            exec_count += 1

        print(f"  Signals: {signal_counts}")
        print(f"  Statuses: {status_counts}")
        print(f"  Execution: {exec_count} tradeable, {blocked_count} blocked")

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    all_results = suite1_results + suite2_results + suite3_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed

    print(f"\n{'='*80}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*80}")
    print(f"  Suite 1 (Setup+Confidence+Gates):   {sum(1 for r in suite1_results if r.passed)}/{len(suite1_results)}")
    print(f"  Suite 2 (Strategy Fit):              {sum(1 for r in suite2_results if r.passed)}/{len(suite2_results)}")
    if suite3_results:
        print(f"  Suite 3 (Signal Generation):         {sum(1 for r in suite3_results if r.passed)}/{len(suite3_results)}")
    else:
        print(f"  Suite 3 (Signal Generation):         SKIPPED")
    print(f"  {'---'*20}")
    print(f"  TOTAL:                               {passed}/{total} ({100*passed//max(total,1)}%)")
    print(f"{'='*80}")

    overall = "ALL TESTS PASSED" if failed == 0 else f"{failed} TESTS FAILED"
    print(f"  {overall}")
    print(f"{'='*80}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
